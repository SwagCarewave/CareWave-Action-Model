"""Out-of-fold retraining of the first-stage v2 model (read-only on the Round4 data)."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.first_stage_pose import CareWaveDirectPoseModel

AUX_WEIGHT = 0.2


def load_sequences(root: Path, manifest: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    data = {}
    for sid in manifest.sample_id:
        f = list((root / "sequences_315").glob(f"*/{sid}_sequences.npz"))
        assert len(f) == 1, sid
        with np.load(f[0], allow_pickle=True) as z:
            data[sid] = {"X": z["X"], "norm": z["y_pose"], "raw": z["y_raw_pose"], "vis": z["y_visibility"]}
    return data


def assign_folds(manifest: pd.DataFrame, k: int, seed: int) -> dict[str, int]:
    """Recording-level folds, stratified by action so every fold has every kind of action."""
    rng = np.random.default_rng(seed)
    fold, counter = {}, 0
    for _, grp in manifest.groupby(manifest.action.str.replace(r"_normal|_variation", "", regex=True)):
        ids = sorted(grp.sample_id)
        rng.shuffle(ids)
        for sid in ids:
            fold[sid] = counter % k
            counter += 1
    return fold


def stack(data: dict, ids: list[str]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.from_numpy(np.concatenate([data[s][key] for s in ids])).float() for key in ("X", "norm", "raw", "vis"))


def raw_loss(pred, target, vis):
    w = 0.25 + 0.75 * vis.clamp(0, 1).unsqueeze(-1).expand(-1, -1, 2).reshape(-1, 66)
    return (F.smooth_l1_loss(pred, target, beta=0.02, reduction="none") * w).sum() / w.sum().clamp_min(1e-8)


@torch.no_grad()
def evaluate(model, x, raw, batch=512) -> float:
    model.eval()
    dist = []
    for i in range(0, len(x), batch):
        out = model(x[i:i + batch])["raw_pose"].reshape(-1, 33, 2)
        dist.append(torch.sqrt(((out - raw[i:i + batch].reshape(-1, 33, 2)) ** 2).sum(-1) + 1e-8).mean(1))
    return float(torch.cat(dist).mean())


def train_fold(train_ids, val_ids, data, out_dir: Path, max_epochs: int, patience: int, seed: int, min_epochs: int = 0) -> dict:
    torch.manual_seed(seed)
    xtr, ntr, rtr, vtr = stack(data, train_ids)
    xva, _, rva, _ = stack(data, val_ids)
    model = CareWaveDirectPoseModel()
    with torch.no_grad():
        prior = rtr.mean(0).clamp(1e-4, 1 - 1e-4)
        model.raw_pose_head.weight.zero_(); model.raw_pose_head.bias.copy_(torch.log(prior / (1 - prior)))
        model.normalized_pose_head.weight.zero_(); model.normalized_pose_head.bias.copy_(ntr.mean(0))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4, min_lr=1e-6)
    gen = torch.Generator().manual_seed(seed)
    best, best_epoch, best_state, bad, history = float("inf"), 0, None, 0, []
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(xtr), generator=gen)
        losses = []
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            out = model(xtr[idx])
            loss = raw_loss(out["raw_pose"], rtr[idx], vtr[idx]) + AUX_WEIGHT * F.smooth_l1_loss(out["normalized_pose"], ntr[idx], beta=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        val = evaluate(model, xva, rva)
        sched.step(val)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mpjpe": val, "lr": opt.param_groups[0]["lr"]})
        print(f"    epoch {epoch:2d} train_loss={np.mean(losses):.4f} val_mpjpe={val:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if epoch < min_epochs:
            continue
        if val < best:
            best, best_epoch, bad, best_state = val, epoch, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "epoch": best_epoch, "val_mpjpe": best, "target": "raw_pose_66",
                "train_recordings": len(train_ids), "val_recordings": len(val_ids)}, out_dir / "best.pt")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    return {"best_epoch": best_epoch, "val_mpjpe": best}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/first_stage_oof_v2"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--only-fold", type=int, default=None)
    ap.add_argument("--min-epochs", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    manifest = pd.read_csv(args.data_root / "model_split_315" / "all_split_manifest.csv", encoding="utf-8-sig")
    manifest = manifest[manifest.model_split != "test"].reset_index(drop=True)
    fold_of = assign_folds(manifest, args.folds, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": list(fold_of), "fold": list(fold_of.values())}).to_csv(args.out / "folds.csv", index=False)
    data = load_sequences(args.data_root, manifest)
    print(f"{len(manifest)} recordings, {sum(len(d['X']) for d in data.values())} sequences, threads={torch.get_num_threads()}")

    rng = np.random.default_rng(args.seed + 1)
    for k in range(args.folds):
        if args.only_fold is not None and k != args.only_fold:
            continue
        fold_dir = args.out / f"fold{k}"
        if (fold_dir / "best.pt").exists():
            print(f"[fold {k}] already trained, skipping")
            continue
        train_pool = sorted(s for s in fold_of if fold_of[s] != k)
        val_ids = sorted(rng.choice(train_pool, size=max(1, len(train_pool) // 10), replace=False).tolist())
        train_ids = [s for s in train_pool if s not in set(val_ids)]
        print(f"[fold {k}] train={len(train_ids)} inner-val={len(val_ids)} held-out={sum(1 for s in fold_of if fold_of[s] == k)}", flush=True)
        print("   ", train_fold(train_ids, val_ids, data, fold_dir, args.max_epochs, args.patience, args.seed + k, args.min_epochs))


if __name__ == "__main__":
    main()
