"""Create 3-second CSI action windows from all matched recordings."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from build_action_labels import assign_splits, discover_pairs
from csi_dataset import FEATURE_COLUMNS, enrich_intervals, labels_for_times, load_config, load_labels, preprocess_raw, save_json


def build_recording(pair: dict, cfg: dict) -> tuple[list[np.ndarray], list[int], list[float], list[dict], dict]:
    data_cfg = cfg["data"]
    frame_df, quality = preprocess_raw(Path(pair["raw_path"]), int(data_cfg["target_fps"]))
    intervals = enrich_intervals(load_labels(Path(pair["label_path"])), pair["sample_id"], pair["subject"])
    frame_labels = labels_for_times(frame_df["time_sec"].to_numpy(), intervals)
    features = frame_df[FEATURE_COLUMNS].to_numpy(np.float32)
    class_names = list(data_cfg["classes"])
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    window = int(round(float(data_cfg["window_seconds"]) * int(data_cfg["target_fps"])))
    stride = int(round(float(data_cfg["stride_seconds"]) * int(data_cfg["target_fps"])))
    max_missing = float(data_cfg.get("max_missing_ratio", 1.0))
    fall_threshold = float(data_cfg.get("fall_overlap_threshold", 0.30))
    boundary_weight = float(data_cfg.get("boundary_weight", 0.5))
    split = pair["split"]

    xs: list[np.ndarray] = []
    ys: list[int] = []
    weights: list[float] = []
    meta: list[dict] = []
    dropped = Counter()
    for start in range(0, len(features) - window + 1, stride):
        end = start + window
        center = start + window // 2
        center_label = str(frame_labels[center])
        labels_in_window = frame_labels[start:end]
        fall_overlap = float(np.mean(labels_in_window == "falling"))
        target = "falling" if fall_overlap >= fall_threshold else center_label
        if target not in class_to_id:
            dropped[f"center_{center_label}"] += 1
            continue
        missing_ratio = float(frame_df["missing_ratio_before_fill"].iloc[start:end].mean())
        if missing_ratio > max_missing:
            dropped["quality"] += 1
            continue
        purity = float(np.mean(labels_in_window == target))
        sample_weight = boundary_weight if target == "falling" and center_label != "falling" else 1.0
        center_time = float(frame_df["time_sec"].iloc[center])
        center_rows = intervals[(intervals["start_sec"] <= center_time) & (center_time < intervals["end_sec"])]
        center_meta = center_rows.iloc[0] if len(center_rows) else None
        window_start = float(frame_df["time_sec"].iloc[start])
        window_end = float(frame_df["time_sec"].iloc[end - 1] + 1 / data_cfg["target_fps"])
        event_rows = intervals[
            (intervals["label"] == "falling")
            & (intervals["start_sec"] < window_end)
            & (intervals["end_sec"] > window_start)
        ]
        event_id = str(event_rows.iloc[0]["event_id"]) if len(event_rows) else (
            str(center_meta["event_id"]) if center_meta is not None else ""
        )
        xs.append(features[start:end])
        ys.append(class_to_id[target])
        weights.append(sample_weight)
        meta.append({
            "sample_id": pair["sample_id"],
            "subject": pair["subject"],
            "subject_id": pair["subject"],
            "session_id": pair["sample_id"],
            "room_id": "unknown",
            "is_explicit_test": pair["is_test"],
            "split": split,
            "start_sec": window_start,
            "end_sec": window_end,
            "center_sec": center_time,
            "label": target,
            "label_id": class_to_id[target],
            "label_purity": purity,
            "fall_overlap_ratio": fall_overlap,
            "sample_weight": sample_weight,
            "event_id": event_id,
            "phase": str(center_meta["phase"]) if center_meta is not None else "unlabeled",
            "label_confidence": float(center_meta["label_confidence"]) if center_meta is not None else 0.0,
            "missing_ratio_before_fill": missing_ratio,
        })
    quality["split"] = split
    quality["is_explicit_test"] = pair["is_test"]
    quality["windows_kept"] = len(xs)
    quality["windows_dropped"] = dict(dropped)
    return xs, ys, weights, meta, quality


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/action_fusion.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    raw_root = Path(data_cfg["raw_csi_dir"])
    label_root = Path(data_cfg["labels_dir"])
    out_dir = Path(data_cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    split_cfg = data_cfg["split"]
    pairs, raw_without_label, label_without_raw = discover_pairs(
        raw_root, label_root, str(split_cfg.get("test_dir_name", "test"))
    )
    pairs = assign_splits(pairs, split_cfg, int(cfg.get("seed", 42)))
    if not pairs:
        raise SystemExit(f"No matched files under {raw_root} and {label_root}")
    all_x: list[np.ndarray] = []
    all_y: list[int] = []
    all_weights: list[float] = []
    all_meta: list[dict] = []
    quality_reports = []
    for pair in pairs:
        try:
            x, y, weights, meta, quality = build_recording(pair, cfg)
        except Exception as exc:
            quality_reports.append({"sample_id": pair["sample_id"], "error": str(exc)})
            print(f"[ERROR] {pair['sample_id']}: {exc}")
            continue
        all_x.extend(x)
        all_y.extend(y)
        all_weights.extend(weights)
        all_meta.extend(meta)
        quality_reports.append(quality)
        print(f"[OK] {pair['sample_id']} [{pair['split']}]: {len(x)} windows")

    if not all_x:
        raise SystemExit("No action windows were created. Check labels, split config and quality threshold.")
    np.save(out_dir / "X.npy", np.stack(all_x).astype(np.float32))
    np.save(out_dir / "y.npy", np.asarray(all_y, dtype=np.int64))
    np.save(out_dir / "sample_weights.npy", np.asarray(all_weights, dtype=np.float32))
    pd.DataFrame(all_meta).to_csv(out_dir / "metadata.csv", index=False, encoding="utf-8-sig")
    save_json(out_dir / "class_names.json", list(data_cfg["classes"]))
    save_json(out_dir / "quality_report.json", {
        "recordings": quality_reports,
        "raw_without_label": raw_without_label,
        "label_without_raw": label_without_raw,
    })
    quality_rows = []
    for report in quality_reports:
        for rx in report.get("rx", []):
            quality_rows.append({"sample_id": report.get("sample_id"), **rx})
    pd.DataFrame(quality_rows).to_csv(out_dir / "csi_quality_report.csv", index=False, encoding="utf-8-sig")
    counts = pd.DataFrame(all_meta).groupby(["split", "subject_id", "label"]).size()
    print(f"Saved {len(all_x)} windows to {out_dir}")
    print(counts)


if __name__ == "__main__":
    main()
