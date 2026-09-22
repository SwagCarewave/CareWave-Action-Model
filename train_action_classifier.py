"""Train the binary CSI-only action classifier."""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.csi_encoder import CSIActionClassifier


def device_from_config(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def augment(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    if not cfg.get("enabled", False):
        return x
    x = x.clone()
    scale = torch.empty((len(x), 1, 1), device=x.device).uniform_(
        float(cfg["amplitude_scale_min"]), float(cfg["amplitude_scale_max"])
    )
    x *= scale
    x += torch.randn_like(x) * float(cfg["gaussian_noise_std"])
    x.masked_fill_(torch.rand_like(x) < float(cfg["feature_dropout"]), 0.0)
    max_shift = int(cfg.get("time_shift_frames", 0))
    if max_shift:
        shifts = torch.randint(-max_shift, max_shift + 1, (len(x),), device=x.device)
        x = torch.stack(
            [torch.roll(row, int(shift.item()), dims=0) for row, shift in zip(x, shifts)]
        )
    return x


def binary_metrics(truth: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
    }


@torch.no_grad()
def collect_predictions(model, loader, criterion, device):
    model.eval()
    truth, probabilities = [], []
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    for x, y, sample_weight in loader:
        x, y, sample_weight = x.to(device), y.to(device), sample_weight.to(device)
        logits = model(x)
        losses = criterion(logits, y)
        weighted_loss_sum += float((losses * sample_weight).sum().item())
        weight_sum += float(sample_weight.sum().item())
        truth.extend(y.int().cpu().tolist())
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    return (
        weighted_loss_sum / max(weight_sum, 1e-9),
        np.asarray(truth, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
    )


def choose_threshold(truth, probability, candidates, recall_floor):
    rows = []
    for threshold in candidates:
        rows.append({"threshold": float(threshold), **binary_metrics(truth, probability, threshold)})
    eligible = [row for row in rows if row["recall"] >= recall_floor]
    if eligible:
        selected = max(eligible, key=lambda row: (row["f1"], row["precision"], row["threshold"]))
    else:
        selected = max(rows, key=lambda row: (row["recall"], row["f1"], row["precision"]))
    return float(selected["threshold"]), selected, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/action_fusion.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/csi_stream"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--all-seeds", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    if args.all_seeds and args.seed is None:
        for run_seed in cfg.get("seeds", [cfg.get("seed", 42)]):
            subprocess.run(
                [sys.executable, __file__, "--config", str(args.config), "--output-dir",
                 str(args.output_dir / f"seed_{run_seed}"), "--seed", str(run_seed)],
                check=True,
            )
        return

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    set_seed(seed)
    data_cfg, training = cfg["data"], cfg["training"]
    data_dir = Path(data_cfg["processed_dir"])
    x = np.load(data_dir / "X.npy").astype(np.float32)
    y = np.load(data_dir / "y.npy").astype(np.float32)
    sample_weights = np.load(data_dir / "sample_weights.npy").astype(np.float32)
    metadata = pd.read_csv(data_dir / "metadata.csv")
    if not (len(x) == len(y) == len(sample_weights) == len(metadata)):
        raise SystemExit("X, y, sample_weights and metadata lengths differ")
    expected_shape = (
        round(float(data_cfg["window_seconds"]) * int(data_cfg["target_fps"])),
        int(data_cfg["input_dim"]),
    )
    if x.shape[1:] != expected_shape:
        raise SystemExit(f"Unexpected X shape: {x.shape}; expected (*, {expected_shape})")

    indices = {
        name: np.flatnonzero(metadata["split"].to_numpy() == name)
        for name in ("train", "val", "test")
    }
    if not len(indices["train"]) or not len(indices["val"]):
        raise SystemExit("Train and val splits must both be non-empty")
    loaders = {}
    for name, idx in indices.items():
        loaders[name] = DataLoader(
            TensorDataset(torch.from_numpy(x[idx]), torch.from_numpy(y[idx]), torch.from_numpy(sample_weights[idx])),
            batch_size=int(training["batch_size"]),
            shuffle=name == "train",
            num_workers=int(training.get("num_workers", 0)),
        )

    device = device_from_config(str(training.get("device", "auto")))
    model = CSIActionClassifier(**cfg["model"]).to(device)
    configured_pos_weight = float(training.get("pos_weight", 1.0))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(configured_pos_weight, device=device), reduction="none"
    )
    print(f"Device={device}; pos_weight={configured_pos_weight:.4f}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=float(training.get("scheduler_factor", 0.5)),
        patience=int(training.get("scheduler_patience", 4))
    )
    threshold_candidates = [
        float(value) for value in training.get(
            "threshold_candidates", [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        )
    ]
    recall_floor = float(training.get("recall_floor", 0.85))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1, stale_epochs, history = -1.0, 0, []

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        for batch_x, batch_y, batch_weight in loaders["train"]:
            batch_x = augment(batch_x.to(device), cfg.get("augmentation", {}))
            batch_y, batch_weight = batch_y.to(device), batch_weight.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = criterion(model(batch_x), batch_y)
            loss = (losses * batch_weight).sum() / batch_weight.sum().clamp_min(1e-6)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            optimizer.step()

        train_loss, train_truth, train_probability = collect_predictions(model, loaders["train"], criterion, device)
        val_loss, val_truth, val_probability = collect_predictions(model, loaders["val"], criterion, device)
        selected_threshold, val_metrics, threshold_table = choose_threshold(
            val_truth, val_probability, threshold_candidates, recall_floor
        )
        train_metrics = binary_metrics(train_truth, train_probability, selected_threshold)
        scheduler.step(val_metrics["f1"])
        history.append({
            "epoch": epoch, "selected_threshold": selected_threshold, "train_loss": train_loss,
            **{f"train_{key}": value for key, value in train_metrics.items()}, "val_loss": val_loss,
            **{f"val_{key}": value for key, value in val_metrics.items() if key != "threshold"},
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        print(
            f"Epoch {epoch:03d} threshold={selected_threshold:.2f} "
            f"train_f1={train_metrics['f1']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1, stale_epochs = val_metrics["f1"], 0
            try:
                commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            except Exception:
                commit = "unknown"
            torch.save({
                "model_state": model.state_dict(), "model_config": cfg["model"],
                "class_names": data_cfg["classes"], "threshold": selected_threshold,
                "threshold_selection": {"recall_floor": recall_floor, "candidates": threshold_table, "selected": val_metrics},
                "config": cfg, "seed": seed, "git_commit": commit,
            }, args.output_dir / "best.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= int(training["patience"]):
                print("Early stopping")
                break

    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    print(f"Best validation fall-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
