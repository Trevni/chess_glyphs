import numpy as np
import chess

def board_to_planes(board: chess.Board) -> np.ndarray:
    """
    Encode board as 17 planes: 12 piece planes, 1 side-to-move, 4 castling planes.
    Order of piece planes: [P,N,B,R,Q,K,p,n,b,r,q,k].
    """
    planes = np.zeros((17, 8, 8), dtype=np.float32)
    piece_order = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    idx = 0
    for color in [chess.WHITE, chess.BLACK]:
        for pt in piece_order:
            bb = board.pieces(pt, color)
            for sq in bb:
                r, c = divmod(sq, 8)
                planes[idx, 7 - r, c] = 1.0
            idx += 1

    # side to move
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    # castling rights
    planes[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    return planes

def squares_attacked_by(board: chess.Board, color: bool) -> np.ndarray:
    attacked = np.zeros((8,8), dtype=np.float32)
    # For every square attacked by color, set 1
    for sq in chess.SQUARES:
        if board.piece_at(sq) and board.piece_at(sq).color == color:
            for t in board.attacks(sq):
                r, c = divmod(t, 8)
                attacked[7 - r, c] = 1.0
    return attacked

def defended_piece_squares(board: chess.Board, color: bool) -> np.ndarray:
    defended = np.zeros((8,8), dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == color:
            # defenders of our piece (excluding the piece itself)
            defenders = board.attackers(color, sq)
            if len(defenders) > 0:
                r, c = divmod(sq, 8)
                defended[7 - r, c] = 1.0
    return defended

def hanging_piece_squares(board: chess.Board, color: bool) -> np.ndarray:
    """
    Squares of color's pieces that are attacked by opponent and defended by 0 own pieces.
    """
    hang = np.zeros((8,8), dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == color:
            attackers = board.attackers(not color, sq)
            defenders = board.attackers(color, sq)
            if len(attackers) > 0 and len(defenders) == 0:
                r, c = divmod(sq, 8)
                hang[7 - r, c] = 1.0
    return hang

def pinned_piece_squares(board: chess.Board, color: bool) -> np.ndarray:
    pins = np.zeros((8,8), dtype=np.float32)
    king_sq = board.king(color)
    if king_sq is None:
        return pins
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == color:
            if board.is_pinned(color, sq):
                r, c = divmod(sq, 8)
                pins[7 - r, c] = 1.0
    return pins

def king_danger_heat(board: chess.Board, color: bool) -> np.ndarray:
    """
    Simple king danger: for squares in king's 1- and 2-neighborhood, sum of opponent attacks.
    Normalize to [0,1].
    """
    heat = np.zeros((8,8), dtype=np.float32)
    king_sq = board.king(color)
    if king_sq is None:
        return heat
    king_r, king_c = divmod(king_sq, 8)
    # squares within Chebyshev distance <= 2
    for sq in chess.SQUARES:
        r, c = divmod(sq, 8)
        if max(abs(r-king_r), abs(c-king_c)) <= 2:
            attackers = board.attackers(not color, sq)
            val = float(len(attackers))
            heat[7 - r, c] = val
    # normalize
    if heat.max() > 0:
        heat /= heat.max()
    return heat

def label_glyphs(board: chess.Board) -> np.ndarray:
    """
    Returns [10,8,8] ground-truth glyph maps from cheap rule-based heuristics.
    """
    aw = squares_attacked_by(board, chess.WHITE)
    ab = squares_attacked_by(board, chess.BLACK)
    dw = defended_piece_squares(board, chess.WHITE)
    db = defended_piece_squares(board, chess.BLACK)
    hw = hanging_piece_squares(board, chess.WHITE)
    hb = hanging_piece_squares(board, chess.BLACK)
    pw = pinned_piece_squares(board, chess.WHITE)
    pb = pinned_piece_squares(board, chess.BLACK)
    kw = king_danger_heat(board, chess.WHITE)
    kb = king_danger_heat(board, chess.BLACK)
    return np.stack([aw, ab, dw, db, hw, hb, pw, pb, kw, kb], axis=0).astype(np.float32)

# ---- Global stats (non-spatial) ----
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

def global_stats(board: chess.Board) -> dict:
    """Compute per-position global stats for UI/Stage-2."""
    def piece_counts(color):
        counts = {p:0 for p in PIECE_VALUES.keys()}
        for sq in chess.SQUARES:
            pc = board.piece_at(sq)
            if pc and pc.color == color:
                counts[pc.piece_type] += 1
        return counts

    def developed_count(color):
        start_rank = 1 if color == chess.WHITE else 6  # 0-indexed ranks
        dev = 0
        for sq in chess.SQUARES:
            pc = board.piece_at(sq)
            if pc and pc.color == color and pc.piece_type in (chess.KNIGHT, chess.BISHOP):
                r = chess.square_rank(sq)
                if (color == chess.WHITE and r != start_rank) or (color == chess.BLACK and r != start_rank):
                    dev += 1
        return dev

    def control_count(color, piece_type):
        # number of distinct squares controlled by given piece type (union across those pieces)
        mask = set()
        for sq in chess.SQUARES:
            pc = board.piece_at(sq)
            if pc and pc.color == color and pc.piece_type == piece_type:
                for t in board.attacks(sq):
                    mask.add(t)
        return len(mask)

    w_counts = piece_counts(chess.WHITE)
    b_counts = piece_counts(chess.BLACK)
    mat_w = sum(PIECE_VALUES[k]*v for k,v in w_counts.items())
    mat_b = sum(PIECE_VALUES[k]*v for k,v in b_counts.items())

    stats = {
        "material_diff_white_minus_black": mat_w - mat_b,
        "dev_minors_white": developed_count(chess.WHITE),
        "dev_minors_black": developed_count(chess.BLACK),
        "knight_control_white": control_count(chess.WHITE, chess.KNIGHT),
        "knight_control_black": control_count(chess.BLACK, chess.KNIGHT),
        "bishop_control_white": control_count(chess.WHITE, chess.BISHOP),
        "bishop_control_black": control_count(chess.BLACK, chess.BISHOP),
        "rook_control_white": control_count(chess.WHITE, chess.ROOK),
        "rook_control_black": control_count(chess.BLACK, chess.ROOK),
        "rooks_connected_white": int(rooks_connected(board, chess.WHITE)),
        "rooks_connected_black": int(rooks_connected(board, chess.BLACK)),
        "pieces_white": {k: int(v) for k,v in w_counts.items()},
        "pieces_black": {k: int(v) for k,v in b_counts.items()},
    }
    return stats

def rooks_connected(board: chess.Board, color: bool) -> bool:
    rooks = [sq for sq in chess.SQUARES if (board.piece_at(sq) and board.piece_at(sq).color==color and board.piece_at(sq).piece_type==chess.ROOK)]
    if len(rooks) < 2:
        return False
    r1, r2 = sorted(rooks)
    # connected = same rank/file and no blockers between
    if chess.square_rank(r1) == chess.square_rank(r2):
        step = 1 if r2>r1 else -1
        for c in range(chess.square_file(r1)+1, chess.square_file(r2)):
            sq = chess.square(c, chess.square_rank(r1))
            if board.piece_at(sq): return False
        return True
    if chess.square_file(r1) == chess.square_file(r2):
        step = 1 if r2>r1 else -1
        for r in range(chess.square_rank(r1)+1, chess.square_rank(r2)):
            sq = chess.square(chess.square_file(r1), r)
            if board.piece_at(sq): return False
        return True
    return False

def contact_threat_squares(board: chess.Board, color: bool) -> np.ndarray:
    """Enemy piece squares that are attacked by 'color'."""
    mask = np.zeros((8,8), dtype=np.float32)
    for sq in chess.SQUARES:
        pc = board.piece_at(sq)
        if pc and pc.color != color:
            # is this square attacked by 'color'?
            if board.is_attacked_by(color, sq):
                r,c = divmod(sq,8)
                mask[7-r, c] = 1.0
    return mask

def underdefended_piece_squares(board: chess.Board, color: bool) -> np.ndarray:
    """Pieces of 'color' where opp attackers > own defenders (material-unaware)."""
    mask = np.zeros((8,8), dtype=np.float32)
    for sq in chess.SQUARES:
        pc = board.piece_at(sq)
        if pc and pc.color == color:
            att = len(board.attackers(not color, sq))
            deff = len(board.attackers(color, sq))
            if att > deff:
                r,c = divmod(sq,8)
                mask[7-r,c] = 1.0
    return mask

CENTRAL_SQS = [chess.D4, chess.E4, chess.D5, chess.E5, chess.C3, chess.D3, chess.E3, chess.F3, chess.C4, chess.F4, chess.C5, chess.F5, chess.C6, chess.D6, chess.E6, chess.F6]

def central_control_heat(board: chess.Board, color: bool) -> np.ndarray:
    """Normalized pressure on central squares by 'color'."""
    heat = np.zeros((8,8), dtype=np.float32)
    for sq in CENTRAL_SQS:
        r,c = divmod(sq,8)
        v = 0.0
        # count number of our attacks to this square
        for src in chess.SQUARES:
            pc = board.piece_at(src)
            if pc and pc.color == color and board.is_legal(chess.Move(src, sq)) or board.is_attacked_by(color, sq):
                pass
        # simpler: count attackers
        v = float(len(board.attackers(color, sq)))
        heat[7-r,c] = v
    # normalize by max>0
    m = heat.max()
    if m>0: heat /= m
    return heat

def piece_highlight(board: chess.Board, color: bool) -> np.ndarray:
    mask = np.zeros((8,8), dtype=np.float32)
    for sq in chess.SQUARES:
        pc = board.piece_at(sq)
        if pc and pc.color == color:
            r,c = divmod(sq,8)
            mask[7-r,c] = 1.0
    return mask

def empty_square_highlight(board: chess.Board) -> np.ndarray:
    # placeholder (all zero). Can be filled by future curator.
    return np.zeros((8,8), dtype=np.float32)

def pawn_chain_mask(board: chess.Board, color: bool) -> np.ndarray:
    """Mark pawns forming diagonal chains (double/triple), only for pawns that have moved."""
    mask = np.zeros((8,8), dtype=np.float32)
    pawns = [sq for sq in chess.SQUARES if (board.piece_at(sq) and board.piece_at(sq).color==color and board.piece_at(sq).piece_type==chess.PAWN)]
    # helper: has moved?
    def moved(sq):
        rank = chess.square_rank(sq)
        return (rank != 1) if color==chess.WHITE else (rank != 6)
    pawns = [sq for sq in pawns if moved(sq)]
    pawn_set = set(pawns)
    # For each pawn, walk diagonal forward links
    dir = 1 if color==chess.WHITE else -1
    for sq in pawns:
        chain = [sq]
        cur = sq
        while True:
            r = chess.square_rank(cur)
            f = chess.square_file(cur)
            nexts = []
            for df in (-1, +1):
                nf = f + df
                nr = r + dir
                if 0 <= nf <= 7 and 0 <= nr <= 7:
                    nsq = chess.square(nf, nr)
                    if nsq in pawn_set:
                        nexts.append(nsq)
            if not nexts: break
            # choose any next; record chain (we only need mask)
            cur = nexts[0]
            if cur in chain: break
            chain.append(cur)
            if len(chain)>=2:
                for csq in chain:
                    rr,cc = divmod(csq,8); mask[7-rr,cc] = 1.0
            if len(chain)>=3:
                # mark stronger intensity for triple
                for csq in chain:
                    rr,cc = divmod(csq,8); mask[7-rr,cc] = 1.0
    return mask

from glyphs.spec import GLYPH_CHANNELS

def label_glyphs(board: chess.Board) -> np.ndarray:
    """
    Returns [C,8,8] rule-based glyph maps, where C=len(GLYPH_CHANNELS).
    Preserves original channels; appends new ones.
    """
    # original signals
    aw = squares_attacked_by(board, chess.WHITE)
    ab = squares_attacked_by(board, chess.BLACK)
    dw = defended_piece_squares(board, chess.WHITE)
    db = defended_piece_squares(board, chess.BLACK)
    hw = hanging_piece_squares(board, chess.WHITE)
    hb = hanging_piece_squares(board, chess.BLACK)
    pw = pinned_piece_squares(board, chess.WHITE)
    pb = pinned_piece_squares(board, chess.BLACK)
    kw = king_danger_heat(board, chess.WHITE)
    kb = king_danger_heat(board, chess.BLACK)

    # new signals
    uw = underdefended_piece_squares(board, chess.WHITE)
    ub = underdefended_piece_squares(board, chess.BLACK)
    cw = central_control_heat(board, chess.WHITE)
    cb = central_control_heat(board, chess.BLACK)
    ctw = contact_threat_squares(board, chess.WHITE)
    ctb = contact_threat_squares(board, chess.BLACK)
    phw = piece_highlight(board, chess.WHITE)
    phb = piece_highlight(board, chess.BLACK)
    pcw = pawn_chain_mask(board, chess.WHITE)
    pcb = pawn_chain_mask(board, chess.BLACK)
    empt = empty_square_highlight(board)

    # Concatenate in spec order
    tensor = np.stack([
        aw,ab,dw,db,hw,hb,pw,pb,kw,kb,
        uw,ub,
        cw,cb,
        ctw,ctb,
        phw,phb,
        pcw,pcb,
        empt
    ], axis=0).astype(np.float32)

    # Sanity: if spec lists a different number, pad/trim to match
    Cspec = len(GLYPH_CHANNELS)
    Ccur = tensor.shape[0]
    if Ccur < Cspec:
        pad = np.zeros((Cspec-Ccur, 8, 8), dtype=np.float32)
        tensor = np.concatenate([tensor, pad], axis=0)
    elif Ccur > Cspec:
        tensor = tensor[:Cspec]

    return tensor
