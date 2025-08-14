import argparse
import numpy as np
import torch
from PIL import Image
import chess

from glyphs.labelers import board_to_planes, label_glyphs
from model.drawer import DrawerNet
from render.renderer import render_glyphs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/drawer.pt", help="Path to trained model; use 'none' to skip and use rule-based labels.")
    ap.add_argument("--fen", type=str, default=chess.STARTING_FEN)
    ap.add_argument("--out", type=str, default="demo.png")
    ap.add_argument("--size", type=int, default=640)
    args = ap.parse_args()

    board = chess.Board(args.fen)

    if args.checkpoint.lower() != "none":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = DrawerNet().to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        x = torch.from_numpy(board_to_planes(board)).unsqueeze(0).to(device)  # [1,17,8,8]
        with torch.no_grad():
            pred = torch.sigmoid(net(x)).cpu().numpy()[0]  # [10,8,8]
        glyphs = pred
    else:
        glyphs = label_glyphs(board)  # use rule-based as a fast sanity check

    img = render_glyphs(board, glyphs, size=args.size)
    img.save(args.out)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
