"""Train the Pose-only standing(0)/fall(1) classifier (E2 baseline)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.pose_encoder import PoseOnlyClassifier

CLIP = 10.0
THRESHOLD_GRID = np.round(np.arange(0.30, 0.801, 0.05), 2)


def load_split(window_dir: Path, split: str) -> dict:
    with np.load(window_dir / f"{split}_3s.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


class Standardizer:
    """Per-feature z-score fitted on the pose-stage TRAIN windows only."""

    def __init__(self, x: np.ndarray):
        flat = x.reshape(-1, x.shape[-1])
        self.mean = flat.mean(0).astype(np.float32)
        std = flat.std(0)
        self.std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / self.std, -CLIP, CLIP).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean, "std": self.std}


def joint_columns(feature_names: np.ndarray) -> dict[str, list[int]]:
    cols: dict[str, list[int]] = {}
    for i, name in enumerate(feature_names):
        parts = str(name).split("_")
        if parts[0] in {"norm", "vel", "acc"}:
            cols.setdefault(parts[1], []).append(i)
    return cols


def augment(x: torch.Tensor, joint_cols: dict[str, list[int]], noise: float, mask_prob: float) -> torch.Tensor:
    x = x + noise * torch.randn_like(x)
    shift = int(torch.randint(-1, 2, (1,)))
    if shift:
        x = torch.roll(x, shift, dims=1)
        edge = slice(0, 1) if shift > 0 else slice(-1, None)
        x[:, edge] = x[:, 1:2] if shift > 0 else x[:, -2:-1]
    if joint_cols and mask_prob > 0:
        keys = list(joint_cols)
        for b in np.where(np.random.rand(len(x)) < mask_prob)[0]:
            for k in np.random.choice(keys, size=np.random.randint(1, 3), replace=False):
                x[b, :, joint_cols[k]] = 0.0
    return x


def best_threshold(y: np.ndarray, p: np.ndarray, grid=THRESHOLD_GRID) -> float:
    scores = []
    for t in grid:
        pred = p >= t
        tp, fp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
        scores.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(grid[int(np.argmax(scores))])


def window_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    both = len(np.unique(y)) == 2
    return {
        "threshold": float(threshold), "n": int(len(y)), "n_fall": int(y.sum()),
        "precision": float(precision), "recall": float(recall),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-9)),
        "specificity": float(specificity), "balanced_accuracy": float(0.5 * (recall + specificity)),
        "auroc": float(roc_auc_score(y, p)) if both else float("nan"),
        "auprc": float(average_precision_score(y, p)) if both else float("nan"),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


@torch.no_grad()
def predict(model: nn.Module, pose: torch.Tensor, quality: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logit, aux = model(pose, quality)
    return torch.sigmoid(logit).numpy(), aux["gate"].numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/pose_branch.yaml"))
    ap.add_argument("--tag", required=True, help="folder under data/processed/pose_windows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-name", default=None, help="run folder name (default: <tag>)")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--no-quality-input", action="store_true")
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--mask-prob", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tcfg, mcfg = cfg["train"], cfg["model"]
    epochs, hidden = args.epochs or tcfg["max_epochs"], args.hidden or mcfg["pose_hidden"]
    dropout, lr = args.dropout if args.dropout is not None else mcfg["dropout"], args.lr or tcfg["lr"]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    window_dir = Path(cfg["paths"]["window_dir"]) / args.tag
    tr, va = load_split(window_dir, "train"), load_split(window_dir, "validation")
    fs, qs = Standardizer(tr["X_pose"]), Standardizer(tr["X_quality"])
    def tens(d):
        return (torch.from_numpy(fs(d["X_pose"])), torch.from_numpy(qs(d["X_quality"])),
                torch.from_numpy(d["y"].astype(np.float32)), torch.from_numpy(d["label_weight"]))
    xtr, qtr, ytr, wtr = tens(tr)
    xva, qva, yva, _ = tens(va)

    model = PoseOnlyClassifier(xtr.shape[-1], qtr.shape[-1], hidden=hidden, dropout=dropout,
                               use_gate=not args.no_gate, use_quality_input=not args.no_quality_input)
    n_pos, n_neg = float(ytr.sum()), float((1 - ytr).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / max(n_pos, 1.0)), reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tcfg["weight_decay"])
    joint_cols = joint_columns(tr["feature_names"])

    out_dir = Path("outputs") / "pose_only" / (args.out_name or args.tag) / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"train={len(ytr)} (fall {int(n_pos)}) val={len(yva)} (fall {int(yva.sum())}) F={xtr.shape[-1]} "
          f"params={sum(p.numel() for p in model.parameters()):,} pos_weight={n_neg / max(n_pos, 1):.2f}")

    history, best, best_state, bad = [], -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(ytr))
        losses = []
        for i in range(0, len(perm), tcfg["batch_size"]):
            idx = perm[i:i + tcfg["batch_size"]]
            if ytr[idx].sum() == 0 or ytr[idx].sum() == len(idx):
                continue
            xb = augment(xtr[idx].clone(), joint_cols, args.noise, args.mask_prob)
            logit, _ = model(xb, qtr[idx])
            loss = (loss_fn(logit, ytr[idx]) * wtr[idx]).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            losses.append(loss.item())
        pv, _ = predict(model, xva, qva)
        both = len(np.unique(yva.numpy())) == 2
        auprc = average_precision_score(yva.numpy(), pv) if both else float("nan")
        auroc = roc_auc_score(yva.numpy(), pv) if both else float("nan")
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_auprc": auprc, "val_auroc": auroc})
        if auprc > best:
            best, bad = auprc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                break
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    model.load_state_dict(best_state)
    pv, gate = predict(model, xva, qva)
    thr = best_threshold(yva.numpy().astype(int), pv)
    metrics = {"tag": args.tag, "seed": args.seed, "best_epoch": int(np.argmax([h["val_auprc"] for h in history]) + 1),
               "validation": window_metrics(yva.numpy().astype(int), pv, thr), "mean_gate_val": float(gate.mean()),
               "config": {"hidden": hidden, "dropout": dropout, "lr": lr, "gate": not args.no_gate,
                          "quality_input": not args.no_quality_input, "noise": args.noise, "mask_prob": args.mask_prob}}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model_state": best_state, "in_dim": xtr.shape[-1], "q_channels": qtr.shape[-1], "threshold": thr,
                "feature_scaler": fs.state(), "quality_scaler": qs.state(), "feature_names": tr["feature_names"],
                "quality_names": tr["quality_names"], "config": metrics["config"], "tag": args.tag,
                "source_hashes": str(tr["source_hashes"])}, out_dir / "best.pt")
    v = metrics["validation"]
    print(f"seed={args.seed} best_epoch={metrics['best_epoch']} val AUPRC={v['auprc']:.3f} AUROC={v['auroc']:.3f} "
          f"@thr {thr:.2f}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} spec={v['specificity']:.3f}")


if __name__ == "__main__":
    main()
