"""
Precompute tensors (inputs + labels) to remove python-chess work from the training loop.

Usage (recommended, mmap-friendly):
  python -m data.precompute --positions data/positions.jsonl --out-prefix data/precomp_positions --dtype float16
  # writes: data/precomp_positions.x.npy and data/precomp_positions.y.npy

Back-compat (writes a single .npz; OK for --workers 0):
  python -m data.precompute --positions data/positions.jsonl --out data/precomp_positions.npz
"""
import argparse, json, numpy as np, os
import chess
from tqdm import tqdm
from glyphs.labelers import board_to_planes, label_glyphs
from glyphs.spec import GLYPH_CHANNELS

def iter_fens(positions_file):
    with open(positions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            yield obj["fen"] if isinstance(obj, dict) else obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", required=True)
    ap.add_argument("--out-prefix", type=str, default=None, help="prefix for .x.npy/.y.npy (mmap-friendly)")
    ap.add_argument("--out", type=str, default=None, help="legacy single .npz (not mmap-friendly)")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    args = ap.parse_args()

    if not args.out_prefix and not args.out:
        raise SystemExit("Specify --out-prefix (recommended) or --out (legacy .npz).")
    dtype = np.float16 if args.dtype == "float16" else np.float32

    xs, ys = [], []
    for fen in tqdm(iter_fens(args.positions), desc="Precomputing"):
        try:
            board = chess.Board(fen)
        except Exception:
            continue
        x = board_to_planes(board)            # (Cin,8,8) float32
        y = label_glyphs(board)               # (Cout,8,8)
        xs.append(x.astype(dtype, copy=False))
        ys.append(y.astype(dtype, copy=False))

    if not xs:
        X = np.zeros((0, 17, 8, 8), dtype=dtype)
        Y = np.zeros((0, len(GLYPH_CHANNELS), 8, 8), dtype=dtype)
    else:
        X = np.stack(xs, axis=0)
        Y = np.stack(ys, axis=0)

    if args.out_prefix:
        x_path = args.out_prefix + ".x.npy"
        y_path = args.out_prefix + ".y.npy"
        np.save(x_path, X)
        np.save(y_path, Y)
        print(f"Wrote {x_path} {X.shape}, {y_path} {Y.shape}")
    if args.out:
        np.savez_compressed(args.out, x=X, y=Y)
        print(f"Wrote {args.out} (compressed .npz)")

if __name__ == "__main__":
    main()