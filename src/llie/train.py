"""Train the residual U-Net on paired dark/bright images."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .data import PairedLowLightDataset
from .model import LowLightUNet
from .inference import choose_device


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return float((-10 * torch.log10(mse.clamp_min(1e-10))).item())


def run_training(args):
    seed_everything(args.seed); device = choose_device(args.device)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    train_ds = PairedLowLightDataset(args.data, "train", args.patch_size, True)
    val_ds = PairedLowLightDataset(args.data, "val", args.patch_size, False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    model = LowLightUNet(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.L1Loss(); start_epoch = 0; best_val = float("inf"); history = []
    last_path = output / "last.pth"
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1; best_val = ckpt.get("best_val", best_val)
        history = ckpt.get("history", [])
    for epoch in range(start_epoch, args.epochs):
        model.train(); train_losses = []
        for dark, bright in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} train"):
            dark, bright = dark.to(device), bright.to(device)
            optimizer.zero_grad(set_to_none=True); prediction = model(dark); loss = criterion(prediction, bright)
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite training loss; stopping to protect checkpoint.")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_losses.append(loss.item())
        model.eval(); val_losses = []; val_psnrs = []
        with torch.inference_mode():
            for dark, bright in val_dl:
                dark, bright = dark.to(device), bright.to(device); pred = model(dark)
                val_losses.append(criterion(pred, bright).item()); val_psnrs.append(psnr(pred, bright))
        scheduler.step(); train_loss = float(np.mean(train_losses)); val_loss = float(np.mean(val_losses)); val_psnr = float(np.mean(val_psnrs))
        record = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                  "val_psnr_db": val_psnr, "lr": optimizer.param_groups[0]["lr"]}
        history.append(record); print(json.dumps(record))
        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "best_val": min(best_val, val_loss), "args": vars(args), "history": history}
        torch.save(ckpt, last_path)
        if val_loss < best_val:
            best_val = val_loss; ckpt["best_val"] = best_val; torch.save(ckpt, output / "best.pth")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Training finished. Best validation loss: {best_val:.5f}. Checkpoints: {output.resolve()}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/paired", help="Dataset root with train/val dark and bright folders")
    p.add_argument("--output", default="checkpoints")
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=256); p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true")
    return p

if __name__ == "__main__": run_training(build_parser().parse_args())
