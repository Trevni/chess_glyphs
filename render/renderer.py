from PIL import Image, ImageDraw, ImageFont
import numpy as np
import chess

COLORS = {
    "light": (240, 217, 181),
    "dark": (181, 136, 99),
    "attack_w": (255, 0, 0, 90),
    "attack_b": (0, 0, 255, 90),
    "def_w": (0, 200, 0, 100),
    "def_b": (0, 200, 0, 100),
    "hang": (255, 165, 0, 140),
    "pin": (128, 0, 128, 140),
    "kd": (255, 0, 0, 60),
}

PIECE_UNICODE = {
    chess.PAWN:   {True: "♙", False: "♟"},
    chess.KNIGHT: {True: "♘", False: "♞"},
    chess.BISHOP: {True: "♗", False: "♝"},
    chess.ROOK:   {True: "♖", False: "♜"},
    chess.QUEEN:  {True: "♕", False: "♛"},
    chess.KING:   {True: "♔", False: "♚"},
}

def draw_board(board: chess.Board, size=512):
    S = size // 8
    img = Image.new("RGBA", (S*8, S*8), (0,0,0,0))
    dr = ImageDraw.Draw(img)
    for r in range(8):
        for c in range(8):
            col = COLORS["light"] if (r+c)%2==0 else COLORS["dark"]
            dr.rectangle([c*S, r*S, (c+1)*S, (r+1)*S], fill=col)
    # draw pieces
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", S-6)
    except:
        font = ImageFont.load_default()
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            r, c = divmod(sq, 8)
            rr = 7 - r
            txt = PIECE_UNICODE[piece.piece_type][piece.color]
            w, h = dr.textsize(txt, font=font)
            x = c*S + (S - w)//2
            y = rr*S + (S - h)//2
            dr.text((x,y), txt, fill=(0,0,0), font=font)
    return img

def overlay_heat(img: Image.Image, heat: np.ndarray, color=(255,0,0,80)):
    S = img.size[0] // 8
    dr = ImageDraw.Draw(img, "RGBA")
    m = heat.max() if heat.size>0 else 0
    for rr in range(8):
        for c in range(8):
            v = float(heat[rr, c])
            if v <= 0: continue
            alpha_scale = v if m==0 else (v / m)
            a = int(color[3] * alpha_scale)
            col = (color[0], color[1], color[2], a)
            dr.rectangle([c*S, rr*S, (c+1)*S, (rr+1)*S], fill=col)
    return img

def overlay_flags(img: Image.Image, mask: np.ndarray, color=(255,165,0,140), marker="•"):
    S = img.size[0] // 8
    dr = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", S//2)
    except:
        font = ImageFont.load_default()
    for rr in range(8):
        for c in range(8):
            if mask[rr, c] > 0.5:
                x = c*S + S//2 - 4
                y = rr*S + S//2 - 8
                dr.text((x,y), marker, fill=(color[0], color[1], color[2], color[3]), font=font)
    return img

def render_glyphs(board: chess.Board, glyphs: np.ndarray, size=512) -> Image.Image:
    """
    glyphs: [10,8,8] probabilities in [0,1]
    """
    img = draw_board(board, size=size)
    # Overlays
    img = overlay_heat(img, glyphs[0], COLORS["attack_w"])  # white attacks
    img = overlay_heat(img, glyphs[1], COLORS["attack_b"])  # black attacks
    img = overlay_flags(img, glyphs[4], COLORS["hang"], marker="H")  # hanging white
    img = overlay_flags(img, glyphs[5], COLORS["hang"], marker="H")  # hanging black
    img = overlay_flags(img, glyphs[6], COLORS["pin"], marker="P")   # pinned white
    img = overlay_flags(img, glyphs[7], COLORS["pin"], marker="P")   # pinned black
    # king danger heat
    img = overlay_heat(img, glyphs[8], COLORS["kd"])
    img = overlay_heat(img, glyphs[9], COLORS["kd"])
    return img
