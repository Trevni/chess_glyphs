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


# ---- Fast path: precomputed arrays ----
import numpy as np
import torch

class PrecomputedDataset(torch.utils.data.Dataset):
    """
    Supports either:
      1) prefix.x.npy / prefix.y.npy  (fast, mmap-friendly)
      2) legacy .npz with {'x','y'}   (works best with workers=0 on Windows)
    """
    def __init__(self, path_or_prefix: str):
        self.path = path_or_prefix
        self._x = None
        self._y = None
        self._use_npz = self.path.endswith(".npz")
        # Probe shapes once (open/close quickly to avoid pickled file handles)
        if self._use_npz:
            with np.load(self.path) as arr:
                self._shape = (arr["x"].shape[0], arr["x"].shape[1:], arr["y"].shape[1:])
        else:
            x_path = self.path + ".x.npy"
            y_path = self.path + ".y.npy"
            X = np.load(x_path, mmap_mode='r')
            Y = np.load(y_path, mmap_mode='r')
            self._shape = (X.shape[0], X.shape[1:], Y.shape[1:])
            # Close views; reopen lazily in each worker
            del X, Y

    def _lazy_open(self):
        if self._x is not None:
            return
        if self._use_npz:
            # For .npz we avoid mmap; open once per worker process
            arr = np.load(self.path)
            self._x = arr["x"]
            self._y = arr["y"]
        else:
            self._x = np.load(self.path + ".x.npy", mmap_mode='r')
            self._y = np.load(self.path + ".y.npy", mmap_mode='r')

    def __len__(self):
        return self._shape[0]

    def __getitem__(self, idx):
        self._lazy_open()
        # Use torch.tensor to avoid "non-writable numpy array" warnings and ensure dtype conversion
        x = torch.tensor(self._x[idx], dtype=torch.float32)
        y = torch.tensor(self._y[idx], dtype=torch.float32)
        return x, y