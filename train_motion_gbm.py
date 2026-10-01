"""Train the motion-statistics gradient-boosting fall detector (models/motion_gbm.py).

Model selection never touches the test folder:
  1. Leave-one-date-out over non-test recordings gives out-of-fold (OOF) window
     probabilities for every date the model has not seen.
  2. The fall threshold is the one with the best OOF alarm-level F1.
  3. OOF alarm-level results (post-processing from the config, unchanged) are
     reported per held-out date with that threshold.
  4. The final model is fitted on all non-test windows and saved as a checkpoint that
     models/loading.build_model understands, so evaluate_stream_events.py,
     evaluate_action_classifier.py and infer_action_stream.py work unchanged.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

from csi_dataset import load_csi_features, load_scaler  # noqa: E402
from data.build_action_labels import assign_splits, discover_pairs  # noqa: E402
from evaluate_date_folds import EVAL_SOURCES, build_raw_windows, recording_date  # noqa: E402
from evaluate_stream_events import score_recording  # noqa: E402
from models.loading import build_model  # noqa: E402
from models.motion_gbm import make_gbm, motion_stats  # noqa: E402

# Weighted training (hard negatives x4) keeps fall probabilities low, so include small values.
THRESHOLDS = np.round(np.r_[0.01, 0.02, 0.03, 0.04, np.arange(0.05, 0.96, 0.05)], 2)


def checkpoint_dict(gbm, cfg, median, iqr, threshold, seed, extra=None) -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    return {"kind": "motion_gbm", "gbm": gbm, "data_config": cfg["data"], "scaler_median": median,
            "scaler_iqr": iqr, "threshold": float(threshold), "class_names": cfg["data"]["classes"],
            "config": cfg, "seed": seed, "git_commit": commit, **(extra or {})}


def alarm_summary(event_rows: list[dict], alarm_rows: list[dict], minutes: float) -> dict:
    events, alarms = pd.DataFrame(event_rows), pd.DataFrame(alarm_rows)
    false = alarms[~alarms["true_alarm"]] if len(alarms) else alarms
    return {"events": int(len(events)), "detected": int(events["detected"].sum()) if len(events) else 0,
            "false_alarms": int(len(false)), "minutes": round(minutes, 1),
            "false_alarms_by_label": false["window_label"].value_counts().to_dict() if len(false) else {},
            "mean_latency_sec": float(events["latency_sec"].dropna().mean()) if len(events) and events["detected"].any() else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="stream_split config with a postprocess section")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/motion_gbm_v1"))
    parser.add_argument("--window-cache", type=Path, help="evaluate_date_folds.py cache dir (default: per config)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data, prep, post = cfg["data"], cfg["preprocessing"], dict(cfg.get("postprocess", {}))
    cache = args.window_cache or Path("outputs/date_folds/cache") / args.config.stem
    x_raw, y, w, meta = build_raw_windows(cfg, cache)
    median, iqr = load_scaler(Path(data["scaler_path"]), x_raw.shape[-1])
    # Train on exactly what the module sees at inference: clipped scaled windows mapped back.
    x = np.clip((x_raw - median) / iqr, float(prep["clip_min"]), float(prep["clip_max"])) * iqr + median
    features = motion_stats(x.astype(np.float32), data)
    del x

    usable = ~meta["split"].eq("test").to_numpy()
    dates = sorted(meta.loc[usable, "date"].astype(str).unique())
    oof = np.full(len(y), np.nan)
    fold_models = {}
    for date in dates:
        on_date = meta["date"].astype(str).eq(date).to_numpy()
        fit = usable & ~on_date
        gbm = make_gbm(args.seed).fit(features[fit], y[fit], sample_weight=w[fit])
        oof[usable & on_date] = gbm.predict_proba(features[usable & on_date])[:, 1]
        fold_models[date] = gbm
    scored = usable & meta["binary_source"].isin(EVAL_SOURCES).to_numpy()

    # Threshold = best OOF alarm-level F1 (alarm precision x fall-event recall) with the
    # config's post-processing; window F1 favoured near-zero thresholds that raise more alarms.
    events = pd.read_csv(data["events_path"])
    pairs, _, _ = discover_pairs(Path(data["raw_csi_dir"]), Path(data["labels_dir"]), data["split"].get("test_dir_name", "test"))
    pairs = [p for p in assign_splits(pairs, data["split"], int(cfg.get("seed", 42))) if p["split"] != "test"]
    recordings = {date: [] for date in dates}
    for pair in pairs:
        frame, raw_features, _ = load_csi_features(Path(pair["raw_path"]), data)
        recordings[recording_date(Path(pair["raw_path"]))].append((pair, frame, raw_features))
    device = torch.device("cpu")
    modules = {d: build_model(checkpoint_dict(fold_models[d], cfg, median, iqr, 0.5, args.seed)) for d in dates}

    def oof_alarms(threshold: float) -> dict:
        run_post = dict(post, fall_threshold=max(threshold, float(post.get("fall_threshold", 0.0))))
        out = {}
        for date in dates:
            event_rows, alarm_rows, minutes = [], [], 0.0
            for pair, frame, raw_features in recordings[date]:
                ev, al, _ = score_recording(modules[date], device, pair, frame, raw_features, data, prep, run_post,
                                            median, iqr, events)
                event_rows += ev; alarm_rows += al; minutes += float(frame["time_sec"].iloc[-1]) / 60
            out[date] = alarm_summary(event_rows, alarm_rows, minutes)
        return out

    table = {}
    for t in THRESHOLDS:
        per_date = oof_alarms(float(t))
        events_n = sum(r["events"] for r in per_date.values()); detected = sum(r["detected"] for r in per_date.values())
        false = sum(r["false_alarms"] for r in per_date.values())
        precision = detected / max(1, detected + false); recall = detected / max(1, events_n)
        table[float(t)] = {"detected": detected, "events": events_n, "false_alarms": false,
                           "alarm_f1": 2 * precision * recall / max(1e-9, precision + recall), "per_date": per_date}
        print(f"threshold {t:.2f}: detected {detected}/{events_n}, false alarms {false}, F1 {table[float(t)]['alarm_f1']:.3f}", flush=True)
    threshold = max(table, key=lambda t: table[t]["alarm_f1"])
    print(f"OOF alarm threshold {threshold:.2f} (F1 {table[threshold]['alarm_f1']:.3f})", flush=True)

    report = {"threshold": threshold, "selection": "oof_alarm_f1",
              "threshold_table": {t: {k: v for k, v in r.items() if k != "per_date"} for t, r in table.items()}, "dates": {}}
    for date in dates:
        mask = scored & meta["date"].astype(str).eq(date).to_numpy()
        yd, pd_ = y[mask], oof[mask]
        report["dates"][date] = {
            "windows": {"n": int(mask.sum()), "falls": int(yd.sum()), "auc": float(roc_auc_score(yd, pd_)),
                        "ap": float(average_precision_score(yd, pd_)),
                        "recall": float((pd_[yd == 1] >= threshold).mean()),
                        "false_positive_rate": float((pd_[yd == 0] >= threshold).mean())},
            "alarms": table[threshold]["per_date"][date]}
        print(date, json.dumps(report["dates"][date], ensure_ascii=False), flush=True)

    final = make_gbm(args.seed).fit(features[usable], y[usable], sample_weight=w[usable])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_dict(final, cfg, median, iqr, threshold, args.seed, {"oof_report": report}), args.output_dir / "best.pt")
    (args.output_dir / "oof_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
