"""
Lichess PGN ingestion (openings-focused).

Usage:
  python -m data.ingest_lichess_pgn --pgn lichess.pgn --out data/positions_openings.jsonl --max-games 20000 --ply-max 20 --bucket "1600-1999"
"""
import argparse, json, chess.pgn
from collections import defaultdict

def rating_bucket(white_elo, black_elo):
    try:
        w = int(white_elo); b = int(black_elo)
    except:
        return "all"
    mean = (w+b)//2
    if mean < 1200: return "U1200"
    if mean < 1600: return "1200-1599"
    if mean < 2000: return "1600-1999"
    if mean < 2300: return "2000-2299"
    return "2300+"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-games", type=int, default=20000)
    ap.add_argument("--ply-max", type=int, default=20, help="only positions up to this ply")
    ap.add_argument("--bucket", type=str, default=None, help="filter by rating bucket")
    args = ap.parse_args()

    seen = set()
    total = 0
    with open(args.out, "w", encoding="utf-8") as out:
        with open(args.pgn, "r", encoding="utf-8", errors="ignore") as f:
            for i in range(args.max_games):
                game = chess.pgn.read_game(f)
                if game is None: break
                welo = game.headers.get("WhiteElo", "0")
                belo = game.headers.get("BlackElo", "0")
                buck = rating_bucket(welo, belo)
                if args.bucket and buck != args.bucket:
                    continue
                board = game.board()
                ply = 0
                for mv in game.mainline_moves():
                    if ply > args.ply_max: break
                    fen = board.fen()
                    if fen not in seen:
                        out.write(json.dumps({"fen": fen, "bucket": buck, "ply": ply}) + "\n")
                        seen.add(fen)
                        total += 1
                    board.push(mv)
                    ply += 1
    print(f"Wrote {total} unique positions to {args.out}")

if __name__ == "__main__":
    main()
