import os
from flask import Flask, request, jsonify, send_from_directory
import chess
import torch
import numpy as np

from glyphs.labelers import board_to_planes, label_glyphs
from model.drawer import DrawerNet
from glyphs.spec import GLYPH_CHANNELS

app = Flask(__name__, static_folder="ui", static_url_path="/ui")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None

def maybe_load_model():
    global MODEL
    ckpt_path = os.path.join("checkpoints", "drawer.pt")
    if os.path.exists(ckpt_path):
        try:
            net = DrawerNet().to(DEVICE)
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            net.load_state_dict(ckpt["state_dict"])
            net.eval()
            MODEL = net
            print("[server] Loaded model from", ckpt_path)
        except Exception as e:
            print("[server] Failed to load model, falling back to rule-based:", e)
            MODEL = None
    else:
        print("[server] No checkpoint found; using rule-based glyphs.")
        MODEL = None

maybe_load_model()

@app.route("/health")
def health():
    return jsonify({"ok": True, "has_model": MODEL is not None})

@app.route("/")
def index():
    return send_from_directory("ui", "index.html")

@app.route("/predict", methods=["GET"])
def predict():
    fen = request.args.get("fen", None)
    if not fen:
        return jsonify({"error": "Missing fen param"}), 400
    try:
        board = chess.Board(fen)
    except Exception as e:
        return jsonify({"error": f"Invalid FEN: {e}"}), 400

    if MODEL is not None:
        x = torch.from_numpy(board_to_planes(board)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = torch.sigmoid(MODEL(x)).cpu().numpy()[0]  # [10,8,8]
        glyphs = pred
    else:
        glyphs = label_glyphs(board)

    out = {name: glyphs[idx].tolist() for idx, name in enumerate(GLYPH_CHANNELS)}
    return jsonify({"glyphs": out})

@app.route("/move", methods=["POST"])
def move():
    data = request.get_json(silent=True) or {}
    fen = data.get("fen")
    uci = data.get("uci")
    if not fen or not uci:
        return jsonify({"error": "Missing fen or uci"}), 400
    try:
        board = chess.Board(fen)
    except Exception as e:
        return jsonify({"error": f"Invalid FEN: {e}"}), 400
    try:
        mv = chess.Move.from_uci(uci)
    except Exception as e:
        return jsonify({"error": f"Invalid UCI: {e}"}), 400
    if mv not in board.legal_moves:
        return jsonify({"error": "Illegal move"}), 200

    board.push(mv)
    new_fen = board.fen()

    if MODEL is not None:
        x = torch.from_numpy(board_to_planes(board)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = torch.sigmoid(MODEL(x)).cpu().numpy()[0]
        glyphs = pred
    else:
        glyphs = label_glyphs(board)

    out = {name: glyphs[idx].tolist() for idx, name in enumerate(GLYPH_CHANNELS)}
    return jsonify({"fen": new_fen, "glyphs": out})

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print("[server] Starting at http://%s:%d" % (host, port))
    # Print routes so you can confirm /move is registered
    print("[server] Routes:", [str(r) for r in app.url_map.iter_rules()])
    app.run(host=host, port=port, debug=True)
