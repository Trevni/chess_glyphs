import argparse, os, torch, numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataset import PositionDataset
from data.dataset import PrecomputedDataset
from model.drawer import DrawerNet
from utils.metrics import bce_with_logits, per_channel_iou_from_logits
from torch.amp import autocast, GradScaler

def l1_sparse_loss(pred_logits, weight=5e-4):
    p = torch.sigmoid(pred_logits)
    return weight * p.mean()

def make_loader(ds, batch_size, workers):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers>0),
        prefetch_factor=4 if workers>0 else None,
    )

def train(args):
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device} | cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"[train] GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    # Dataset / Loader
    if args.precomputed:
        ds = PrecomputedDataset(args.precomputed)
        val_split = int(len(ds)*0.1)
        train_indices = np.arange(len(ds)-val_split)
        val_indices = np.arange(len(ds)-val_split, len(ds))
        train_ds = torch.utils.data.Subset(ds, train_indices)
        val_ds = torch.utils.data.Subset(ds, val_indices)
    else:
        ds = PositionDataset(args.positions)
        ds.set_split("train")
        train_ds = ds
        ds.set_split("val")
        val_ds = ds

    train_loader = make_loader(train_ds, args.batch_size, args.workers)
    val_loader   = make_loader(val_ds,   args.batch_size, args.workers)

    # Model
    net = DrawerNet(width=args.width, depth=args.depth).to(device)
    net = net.to(memory_format=torch.channels_last)
    if args.compile and hasattr(torch, "compile"):
        try:
            net = torch.compile(net, mode="reduce-overhead")
            print("[train] torch.compile enabled")
        except Exception as e:
            print(f"[train] torch.compile failed: {e}")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    scaler = GradScaler('cuda', enabled=(device.type=='cuda'))

    best_val = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs+1):
        # ---- Train ----
        net.train()
        total = 0.0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for step, (x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with autocast(device_type='cuda', enabled=(device.type=='cuda')):
                logits = net(x)
                loss = bce_with_logits(logits, y) + l1_sparse_loss(logits, weight=args.l1)
                loss = loss / max(1, args.grad_accum)
            scaler.scale(loss).backward()
            if step % args.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            total += loss.item() * x.size(0) * max(1, args.grad_accum)
            pbar.set_postfix(loss=f"{(loss.item()*max(1,args.grad_accum)):.4f}")
        train_loss = total / max(1, len(train_loader.dataset))

        # ---- Val ----
        net.eval()
        total = 0.0
        iou_sum = None
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)
                with autocast(device_type='cuda', enabled=(device.type=='cuda')):
                    logits = net(x)
                    loss = bce_with_logits(logits, y) + l1_sparse_loss(logits, weight=args.l1)
                total += loss.item() * x.size(0)
                iou = per_channel_iou_from_logits(logits, y)
                iou_sum = iou if iou_sum is None else (iou_sum + iou)
        val_loss = total / max(1, len(val_loader.dataset))
        iou_mean = (iou_sum / max(1, len(val_loader))).mean().item() if iou_sum is not None else 0.0
        print(f"Epoch {epoch}: train {train_loss:.4f} | val {val_loss:.4f} | IoU(avg) {iou_mean:.3f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss; best_epoch = epoch
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"state_dict": net.state_dict(), "args": vars(args)}, args.out)
            print(f"Saved best checkpoint to {args.out}")
        elif epoch - best_epoch >= args.patience:
            print(f"[early stopping] no improvement in {args.patience} epochs (best @ {best_epoch})")
            break

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=str, help="JSONL positions (ignored if --precomputed provided)")
    ap.add_argument("--precomputed", type=str, default=None, help="Either <prefix> for .x.npy/.y.npy or a .npz (legacy)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--l1", type=float, default=5e-4, help="L1 sparsity weight")
    ap.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs)")
    ap.add_argument("--compile", action="store_true", help="enable torch.compile if available")
    ap.add_argument("--out", type=str, default="checkpoints/drawer.pt")
    args = ap.parse_args()
    train(args)
