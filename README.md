# Chess Glyph Drawer — Prototype (Stage A)

A minimal, **trainable prototype** for a *drawer* model that takes a chess board (FEN) and outputs **glyph maps**
(attack maps, defended pieces, hanging pieces, pins, king-danger heat). It includes:
- Data generation from random legal positions (python-chess).
- Cheap supervision for glyphs (no engine required).
- A tiny CNN (PyTorch) that learns to predict glyph maps.
- Rendering utilities to visualize predicted glyphs over the board.
- A `demo.py` that loads a trained checkpoint and draws glyphs for any FEN.

> This is **Stage A only** (the "Drawer"). Stage B (policy) is intentionally omitted here.

## Quickstart

```bash
# 1) Create a venv (recommended)
python -m venv .venv && . .venv/Scripts/activate  # Windows (PowerShell)
# or: source .venv/bin/activate                    # macOS/Linux

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) Generate a small dataset of random positions
python -m data.generate_positions --out data/positions.jsonl --n 5000 --min-plies 4 --max-plies 24

# 4) Train the drawer (few minutes on CPU; faster on GPU)
python train.py --positions data/positions.jsonl --epochs 3 --batch-size 64 --out checkpoints/drawer.pt

# 5) Run a demo on any FEN (or startpos)
python demo.py --checkpoint checkpoints/drawer.pt --fen "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3" --out demo.png
```

> For a *very* quick visual sanity check without training, run `demo.py` with `--checkpoint none`.
It will use the **rule-based labelers** directly to render the glyphs.

## What this prototype does

- **Inputs**: 17 planes (12 piece-type planes, side-to-move, 4 castling planes), shape `[17,8,8]`.
- **Outputs**: 10 glyph channels (all 8×8), with `sigmoid`:
  0. `attack_white`      — squares attacked by White
  1. `attack_black`      — squares attacked by Black
  2. `defended_white`    — white piece squares defended by White
  3. `defended_black`    — black piece squares defended by Black
  4. `hanging_white`     — white piece squares attacked by Black and defended by 0 White pieces
  5. `hanging_black`     — black piece squares attacked by White and defended by 0 Black pieces
  6. `pinned_white`      — squares of pinned White pieces
  7. `pinned_black`      — squares of pinned Black pieces
  8. `king_danger_white` — heat around White king (opponent pressure near king)
  9. `king_danger_black` — heat around Black king

- **Loss**: BCE per channel + mild L1 sparsity on outputs (encourages parsimonious drawings).

## Extending toward your research plan

- Add **hard-concrete gates** or **top-K budget** on outputs for tighter parsimony.
- Introduce **free glyph channels** (arrows/zones) and validity oracles (engine or human stats).
- Freeze this Drawer and train a **Stage B** policy that sees only glyphs (fog-of-war).

## Dependencies

- Python ≥3.9
- `torch`, `numpy`, `pillow`, `tqdm`, `python-chess`

