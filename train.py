import argparse, os, torch, numpy as np, time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataset import PositionDataset
from model.drawer import DrawerNet
from utils.metrics import bce_with_logits, per_channel_iou_from_logits

def l1_sparse_loss(pred_logits, weight=5e-4):
    p = torch.sigmoid(pred_logits)
    return weight * p.mean()

def train(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device} | cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[train] GPU name: {torch.cuda.get_device_name(0)}")

    ds = PositionDataset(args.positions)
    n_total = len(ds.items)
    n_train = int(n_total * 0.9)
    n_val = n_total - n_train

    train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    ds.set_split("val")
    val_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    net = DrawerNet(width=args.width, depth=args.depth).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs+1):
        # Train
        ds.set_split("train")
        net.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        total = 0.0
        for x, y in pbar:
            x = x.to(device); y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits = net(x)
                loss = bce_with_logits(logits, y) + l1_sparse_loss(logits, weight=args.l1)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = total / max(1,n_train)

        # Val
        ds.set_split("val")
        net.eval()
        total = 0.0
        iou_sum = None
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                logits = net(x)
                loss = bce_with_logits(logits, y) + l1_sparse_loss(logits, weight=args.l1)
                total += loss.item() * x.size(0)
                iou = per_channel_iou_from_logits(logits, y)  # (C,)
                iou_sum = iou if iou_sum is None else (iou_sum + iou)
        val_loss = total / max(1,n_val) if n_val>0 else train_loss
        iou_mean = (iou_sum / max(1, len(val_loader))).mean().item() if iou_sum is not None else 0.0
        print(f"Epoch {epoch}: train {train_loss:.4f} | val {val_loss:.4f} | IoU(avg) {iou_mean:.3f}")

        improved = val_loss < best_val - 1e-5
        if improved:
            best_val = val_loss
            best_epoch = epoch
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"state_dict": net.state_dict(), "args": vars(args)}, args.out)
            print(f"Saved best checkpoint to {args.out}")
        elif epoch - best_epoch >= args.patience:
            print(f"[early stopping] no improvement in {args.patience} epochs (best @ {best_epoch})")
            break

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--l1", type=float, default=5e-4, help="L1 sparsity weight")
    ap.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs)")
    ap.add_argument("--out", type=str, default="checkpoints/drawer.pt")
    args = ap.parse_args()
    train(args)
