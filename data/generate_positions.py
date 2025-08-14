import argparse, json, random
import chess, chess.pgn

def random_position(min_plies=4, max_plies=24) -> chess.Board:
    board = chess.Board()
    plies = random.randint(min_plies, max_plies)
    for _ in range(plies):
        if board.is_game_over():
            break
        moves = list(board.legal_moves)
        if not moves: break
        board.push(random.choice(moves))
    return board

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Output JSONL with FEN per line.")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--min-plies", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=24)
    args = ap.parse_args()

    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            b = random_position(args.min_plies, args.max_plies)
            f.write(json.dumps({"fen": b.fen()}) + "\n")
    print(f"Wrote {args.n} positions to {args.out}")

if __name__ == "__main__":
    main()
