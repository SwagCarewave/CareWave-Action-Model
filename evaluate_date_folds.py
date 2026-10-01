"""Leave-one-date-out evaluation: train on all other recording dates, score the held-out date.

Recordings in the test folder are never used. For each date fold the train-date recordings
keep their session-hash train/val assignment (val = early stopping and threshold only),
a robust scaler is refitted on the fold's train windows, and the held-out date is scored
with threshold-free window metrics (fall ROC AUC, average precision).

Variants:
  deep:<name>=<config>  CSIActionClassifier trained with train_action_classifier.py
  gbm:<name>=<config>   HistGradientBoosting on within-window motion statistics
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))  # build_action_windows imports build_action_labels as a sibling

from data.build_action_labels import assign_splits, discover_pairs  # noqa: E402
from data.build_action_windows import build_recording  # noqa: E402
from models.loading import build_model  # noqa: E402
from models.motion_gbm import make_gbm, motion_stats  # noqa: E402

EVAL_SOURCES = {"falling", "standing", "hard_negative"}


def recording_date(raw_path: Path) -> str:
    with raw_path.open(encoding="utf-8-sig") as handle:
        handle.readline()
        return handle.readline().split(",", 1)[0][:8]


def build_raw_windows(cfg: dict, cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Unscaled windows for every recording, cached per config."""
    if (cache_dir / "X_raw.npy").exists():
        return (np.load(cache_dir / "X_raw.npy"), np.load(cache_dir / "y.npy"),
                np.load(cache_dir / "sample_weights.npy"), pd.read_csv(cache_dir / "metadata.csv"))
    data = cfg["data"]
    build_cfg = dict(cfg, preprocessing=dict(cfg["preprocessing"], fit_scaler=True))  # -> build_recording skips scaling
    events = pd.read_csv(data["events_path"])
    pairs, _, _ = discover_pairs(Path(data["raw_csi_dir"]), Path(data["labels_dir"]), data["split"].get("test_dir_name", "test"))
    pairs = assign_splits(pairs, data["split"], int(cfg.get("seed", 42)))
    xs, ys, ws, meta = [], [], [], []
    for pair in pairs:
        x, y, w, m, _ = build_recording(pair, build_cfg, events)
        date = recording_date(Path(pair["raw_path"]))
        xs.extend(x); ys.extend(y); ws.extend(w)
        meta.extend(dict(row, date=date) for row in m)
        print(f"built {pair['sample_id']} [{pair['split']}, {date}]: {len(x)} windows", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    x, y, w, meta = np.stack(xs).astype(np.float32), np.asarray(ys, np.int64), np.asarray(ws, np.float32), pd.DataFrame(meta)
    np.save(cache_dir / "X_raw.npy", x); np.save(cache_dir / "y.npy", y); np.save(cache_dir / "sample_weights.npy", w)
    meta.to_csv(cache_dir / "metadata.csv", index=False, encoding="utf-8-sig")
    return x, y, w, meta


def fold_split(meta: pd.DataFrame, held_out: str) -> np.ndarray:
    """train/val from the session hash on train dates; 'test' = held-out date eval windows; '' = unused."""
    split = np.full(len(meta), "", dtype=object)
    is_test_folder = meta["split"].eq("test").to_numpy()
    on_date = meta["date"].astype(str).eq(held_out).to_numpy()
    train_dates = ~is_test_folder & ~on_date
    split[train_dates & meta["split"].eq("train").to_numpy()] = "train"
    split[train_dates & meta["split"].eq("val").to_numpy()] = "val"
    split[~is_test_folder & on_date & meta["binary_source"].isin(EVAL_SOURCES).to_numpy()] = "test"
    return split


def scale(x: np.ndarray, train_idx: np.ndarray, prep: dict) -> np.ndarray:
    frames = x[train_idx].reshape(-1, x.shape[-1]).astype(np.float64)
    median = np.median(frames, axis=0)
    q25, q75 = np.percentile(frames, [25, 75], axis=0)
    iqr = q75 - q25
    iqr[np.abs(iqr) < 1e-8] = 1.0
    return np.clip((x - median) / iqr, float(prep["clip_min"]), float(prep["clip_max"])).astype(np.float32)


def window_metrics(y: np.ndarray, p: np.ndarray, meta: pd.DataFrame) -> dict:
    hard = meta["binary_source"].eq("hard_negative").to_numpy()
    return {"windows": int(len(y)), "falls": int(y.sum()), "auc": float(roc_auc_score(y, p)),
            "ap": float(average_precision_score(y, p)), "fall_rate": float(y.mean()),
            "hard_negative_p_median": float(np.median(p[hard])) if hard.any() else None,
            "fall_p_median": float(np.median(p[y == 1]))}


def predict_deep(checkpoint_path: Path, x: np.ndarray) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint)
    out = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            out.append(torch.sigmoid(model(torch.from_numpy(x[start:start + 512]))).numpy().reshape(-1))
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", required=True, help="deep:name=config or gbm:name=config")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/date_folds"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    results, jobs = {}, []
    for spec in args.variant:
        kind, rest = spec.split(":", 1)
        name, config_path = rest.split("=", 1)
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        x_raw, y, w, meta = build_raw_windows(cfg, args.output_dir / "cache" / Path(config_path).stem)
        dates = sorted(meta.loc[~meta["split"].eq("test"), "date"].astype(str).unique())
        for held_out in dates:
            split = fold_split(meta, held_out)
            idx = {s: np.flatnonzero(split == s) for s in ("train", "val", "test")}
            fold_dir = args.output_dir / name / f"holdout_{held_out}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            eval_meta = meta.iloc[idx["test"]].reset_index(drop=True)
            if kind == "gbm":
                feats = motion_stats(x_raw, cfg["data"])
                model = make_gbm(args.seed)
                model.fit(feats[idx["train"]], y[idx["train"]], sample_weight=w[idx["train"]])
                p = model.predict_proba(feats[idx["test"]])[:, 1]
                results.setdefault(name, {})[held_out] = window_metrics(y[idx["test"]], p, eval_meta)
                pd.DataFrame({**eval_meta[["sample_id", "binary_source"]], "y": y[idx["test"]], "p": p}).to_csv(fold_dir / "predictions.csv", index=False)
                print(name, held_out, results[name][held_out], flush=True)
                continue
            keep = np.concatenate([idx["train"], idx["val"], idx["test"]])
            x = scale(x_raw, idx["train"], cfg["preprocessing"])
            fold_meta = meta.iloc[keep].assign(split=split[keep]).reset_index(drop=True)
            np.save(fold_dir / "X.npy", x[keep]); np.save(fold_dir / "y.npy", y[keep])
            np.save(fold_dir / "sample_weights.npy", w[keep])
            fold_meta.to_csv(fold_dir / "metadata.csv", index=False, encoding="utf-8-sig")
            fold_cfg = dict(cfg, data=dict(cfg["data"], processed_dir=str(fold_dir)))
            (fold_dir / "config.yaml").write_text(yaml.safe_dump(fold_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
            jobs.append((name, held_out, fold_dir, eval_meta))
            del x

    running, pending = [], list(jobs)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), PYTHONUNBUFFERED="1")
    while pending or running:
        while pending and len(running) < args.parallel:
            name, held_out, fold_dir, eval_meta = pending.pop(0)
            log = (fold_dir / "train.log").open("w", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, str(ROOT / "train_action_classifier.py"), "--config", str(fold_dir / "config.yaml"),
                                     "--output-dir", str(fold_dir), "--seed", str(args.seed)], stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append((proc, log, name, held_out, fold_dir, eval_meta))
            print(f"started {name} holdout {held_out}", flush=True)
        time.sleep(10)
        for item in list(running):
            proc, log, name, held_out, fold_dir, eval_meta = item
            if proc.poll() is None:
                continue
            running.remove(item); log.close()
            if proc.returncode != 0:
                print(f"FAILED {name} holdout {held_out}; see {fold_dir / 'train.log'}", flush=True)
                continue
            metadata = pd.read_csv(fold_dir / "metadata.csv")
            test_idx = np.flatnonzero(metadata["split"].eq("test").to_numpy())
            x = np.load(fold_dir / "X.npy", mmap_mode="r")[test_idx]
            y_test = np.load(fold_dir / "y.npy")[test_idx]
            p = predict_deep(fold_dir / "best.pt", np.ascontiguousarray(x))
            results.setdefault(name, {})[held_out] = window_metrics(y_test, p, eval_meta)
            pd.DataFrame({**eval_meta[["sample_id", "binary_source"]], "y": y_test, "p": p}).to_csv(fold_dir / "predictions.csv", index=False)
            (fold_dir / "X.npy").unlink()
            print(name, held_out, results[name][held_out], flush=True)

    summary = {name: {"mean_auc": float(np.mean([r["auc"] for r in folds.values()])),
                      "mean_ap": float(np.mean([r["ap"] for r in folds.values()])), "folds": folds}
               for name, folds in results.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "summary.json"
    previous = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    out.write_text(json.dumps({**previous, **summary}, indent=2), encoding="utf-8")
    for name, s in summary.items():
        print(f"{name}: mean AUC {s['mean_auc']:.3f}, mean AP {s['mean_ap']:.3f} | " +
              ", ".join(f"{d}: {r['auc']:.3f}/{r['ap']:.3f}" for d, r in s["folds"].items()))


if __name__ == "__main__":
    main()
