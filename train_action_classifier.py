"""Train the CSI action classifier on generated 3-second windows."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.csi_encoder import CSIActionClassifier


def device_from_config(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def center_windows(x: np.ndarray) -> np.ndarray:
    """Remove the static offset of every subcarrier within each window."""
    return x - x.mean(axis=1, keepdims=True)


def augment_batch(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    if not cfg.get("enabled", False):
        return x
    x = x.clone()
    scale = torch.empty((len(x), 1, 1), device=x.device).uniform_(
        float(cfg["amplitude_scale_min"]),
        float(cfg["amplitude_scale_max"]),
    )
    x *= scale
    x += torch.randn_like(x) * float(cfg["gaussian_noise_std"])
    mask = torch.rand_like(x) < float(cfg["subcarrier_dropout"])
    x.masked_fill_(mask, 0.0)
    max_shift = int(cfg.get("time_shift_frames", 0))
    if max_shift:
        shifts = torch.randint(-max_shift, max_shift + 1, (len(x),), device=x.device)
        x = torch.stack(
            [torch.roll(row, int(shift.item()), dims=0) for row, shift in zip(x, shifts)]
        )
    return x


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    truth, pred = [], []
    for x, y, sample_weight in loader:
        x, y, sample_weight = x.to(device), y.to(device), sample_weight.to(device)
        logits = model(x)
        batch_loss = criterion(logits, y)
        weighted_loss_sum += float((batch_loss * sample_weight).sum().item())
        weight_sum += float(sample_weight.sum().item())
        truth.extend(y.cpu().tolist())
        pred.extend(logits.argmax(1).cpu().tolist())
    if not truth:
        return {"loss": float("nan"), "macro_f1": float("nan"), "accuracy": float("nan")}
    return {
        "loss": weighted_loss_sum / max(weight_sum, 1e-9),
        "macro_f1": f1_score(truth, pred, average="macro", zero_division=0),
        "accuracy": float(np.mean(np.asarray(truth) == np.asarray(pred))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/action_fusion.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/csi_stream"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--all-seeds", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.all_seeds and args.seed is None:
        for run_seed in cfg.get("seeds", [cfg.get("seed", 42)]):
            run_dir = args.output_dir / f"seed_{run_seed}"
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--config",
                    str(args.config),
                    "--output-dir",
                    str(run_dir),
                    "--seed",
                    str(run_seed),
                ],
                check=True,
            )
        return

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_dir = Path(cfg["data"]["processed_dir"])
    x = np.load(data_dir / "X.npy").astype(np.float32)
    y = np.load(data_dir / "y.npy").astype(np.int64)
    sample_weights = np.load(data_dir / "sample_weights.npy").astype(np.float32)
    meta = pd.read_csv(data_dir / "metadata.csv")
    class_names = json.loads((data_dir / "class_names.json").read_text(encoding="utf-8"))
    if len(x) != len(y) or len(y) != len(meta):
        raise SystemExit("X, y and metadata lengths differ")

    indices = {
        name: np.flatnonzero(meta["split"].to_numpy() == name)
        for name in ("train", "val", "test")
    }
    if not len(indices["train"]) or not len(indices["val"]):
        raise SystemExit("Both train and val splits need at least one window")

    window_center = bool(
        cfg.get("preprocessing", {}).get("window_center", False)
    )

    if window_center:
        x = center_windows(x)

    mean = x[indices["train"]].mean(axis=(0, 1), keepdims=True)
    std = x[indices["train"]].std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std

    training = cfg["training"]
    loaders = {}
    for name, idx in indices.items():
        dataset = TensorDataset(
            torch.from_numpy(x[idx]),
            torch.from_numpy(y[idx]),
            torch.from_numpy(sample_weights[idx]),
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=int(training["batch_size"]),
            shuffle=name == "train",
            num_workers=int(training.get("num_workers", 0)),
        )

    model = CSIActionClassifier(num_classes=len(class_names), **cfg["model"])
    device = device_from_config(str(training.get("device", "auto")))
    model.to(device)

    counts = np.bincount(y[indices["train"]], minlength=len(class_names)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, device=device), reduction="none"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(training.get("scheduler_factor", 0.5)),
        patience=int(training.get("scheduler_patience", 4)),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1, stale = -1.0, 0
    history = []
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        for batch_x, batch_y, batch_weight in loaders["train"]:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_weight = batch_weight.to(device)
            batch_x = augment_batch(batch_x, cfg.get("augmentation", {}))
            optimizer.zero_grad(set_to_none=True)
            per_sample_loss = criterion(model(batch_x), batch_y)
            loss = (per_sample_loss * batch_weight).sum() / batch_weight.sum().clamp_min(1e-6)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), float(training.get("gradient_clip", 1.0))
            )
            optimizer.step()

        train_metrics = evaluate(model, loaders["train"], criterion, device)
        val_metrics = evaluate(model, loaders["val"], criterion, device)
        scheduler.step(val_metrics["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )
        print(
            f"Epoch {epoch:03d} train_f1={train_metrics['macro_f1']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        score = val_metrics["macro_f1"]
        if score > best_f1:
            best_f1, stale = score, 0
            try:
                git_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip()
            except Exception:
                git_commit = "unknown"
            manifest_path = Path("data/splits/split_manifest.csv")
            split_manifest = (
                pd.read_csv(manifest_path).to_dict(orient="records")
                if manifest_path.exists()
                else []
            )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": cfg["model"],
                    "class_names": class_names,
                    "class_to_id": {name: i for i, name in enumerate(class_names)},
                    "mean": mean.squeeze().astype(np.float32),
                    "std": std.squeeze().astype(np.float32),
                    "preprocessing": {
                        "window_center": window_center
                    },
                    "target_fps": cfg["data"]["target_fps"],                    "window_seconds": cfg["data"]["window_seconds"],
                    "config": cfg,
                    "seed": seed,
                    "postprocess": cfg.get("postprocess", {}),
                    "git_commit": git_commit,
                    "split_manifest": split_manifest,
                },
                args.output_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= int(training["patience"]):
                print("Early stopping")
                break

    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    print(f"Best validation macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()