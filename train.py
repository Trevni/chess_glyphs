import argparse, os, torch, numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataset import PositionDataset
from model.drawer import DrawerNet

def bce_loss(pred_logits, target):
    # pred_logits: [B,10,8,8], target: [B,10,8,8]
    return F.binary_cross_entropy_with_logits(pred_logits, target)

def l1_sparse_loss(pred_logits, weight=1e-3):
    # encourage sparse glyphs post-sigmoid
    p = torch.sigmoid(pred_logits)
    return weight * p.mean()

def train(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ds = PositionDataset(args.positions)
    n_total = len(ds.items)
    n_train = int(n_total * 0.9)
    n_val = n_total - n_train

    train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    ds.set_split("val")
    val_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DrawerNet(width=args.width, depth=args.depth).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    best_val = 1e9
    for epoch in range(1, args.epochs+1):
        ds.set_split("train")
        net.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        total = 0.0
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            logits = net(x)
            loss = bce_loss(logits, y) + l1_sparse_loss(logits, weight=args.l1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = total / n_train

        # validation
        ds.set_split("val")
        net.eval()
        total = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                logits = net(x)
                loss = bce_loss(logits, y) + l1_sparse_loss(logits, weight=args.l1)
                total += loss.item() * x.size(0)
        val_loss = total / n_val if n_val>0 else train_loss
        print(f"Epoch {epoch}: train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"state_dict": net.state_dict(), "args": vars(args)}, args.out)
            print(f"Saved best checkpoint to {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--l1", type=float, default=1e-3, help="L1 sparsity weight")
    ap.add_argument("--out", type=str, default="checkpoints/drawer.pt")
    args = ap.parse_args()
    train(args)
