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
