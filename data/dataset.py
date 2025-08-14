import json, numpy as np, torch
from torch.utils.data import Dataset
import chess
from glyphs.labelers import board_to_planes, label_glyphs

class PositionDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))
        # simple split marker: last 10% test
        self.split = "train"

    def set_split(self, split: str):
        assert split in ("train", "val"), "split must be train or val"
        self.split = split

    def __len__(self):
        n = len(self.items)
        if self.split == "train":
            return int(n * 0.9)
        else:
            return n - int(n * 0.9)

    def __getitem__(self, idx):
        if self.split == "val":
            idx = int(len(self.items)*0.9) + idx
        fen = self.items[idx]["fen"]
        board = chess.Board(fen)
        x = board_to_planes(board)            # [17,8,8]
        y = label_glyphs(board)               # [10,8,8]
        return torch.from_numpy(x), torch.from_numpy(y)
