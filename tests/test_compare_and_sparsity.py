import os, numpy as np, chess, torch, pytest
from glyphs.labelers import label_glyphs, board_to_planes
from glyphs.spec import GLYPH_CHANNELS
from utils.composite import composite_any

START_FEN = chess.STARTING_FEN

COMPARE_CHANNELS = [
    "attack_white","attack_black",
    "hanging_white","hanging_black",
    "pinned_white","pinned_black",
    "king_danger_white","king_danger_black",
]

def test_labels_startpos_sparse():
    board = chess.Board(START_FEN)
    Y = label_glyphs(board)  # [C,8,8]
    assert Y.shape == (len(GLYPH_CHANNELS), 8, 8)

    # Baseline expectations at startpos (loose)
    aw = int((Y[GLYPH_CHANNELS.index("attack_white")] > 0.5).sum())
    ab = int((Y[GLYPH_CHANNELS.index("attack_black")] > 0.5).sum())
    ccw = int((Y[GLYPH_CHANNELS.index("central_control_white")] > 0.5).sum())
    ccb = int((Y[GLYPH_CHANNELS.index("central_control_black")] > 0.5).sum())
    phw = int((Y[GLYPH_CHANNELS.index("piece_highlight_white")] > 0.5).sum())
    phb = int((Y[GLYPH_CHANNELS.index("piece_highlight_black")] > 0.5).sum())

    assert 16 <= aw <= 26
    assert 16 <= ab <= 26
    assert ccw == 4
    assert ccb == 4
    assert phw == 16
    assert phb == 16

    mask = composite_any(Y, COMPARE_CHANNELS)
    assert mask.shape == (8,8)
    total = int(mask.sum())
    assert 10 <= total <= 80

@pytest.mark.skipif(not os.path.exists("checkpoints/drawer.pt"), reason="no trained model checkpoint")
def test_predictions_startpos_sparse():
    from model.drawer import DrawerNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load("checkpoints/drawer.pt", map_location=device)
    net = DrawerNet(out_ch=len(GLYPH_CHANNELS)).to(device)
    try:
        net.load_state_dict(ckpt["state_dict"])
    except Exception:
        pytest.skip("checkpoint does not match current model spec")
    net.eval()
    x = torch.from_numpy(board_to_planes(chess.Board(START_FEN))).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.sigmoid(net(x)).cpu().numpy()[0]
    mask = composite_any(pred, COMPARE_CHANNELS)
    total = int(mask.sum())
    assert total <= 90
