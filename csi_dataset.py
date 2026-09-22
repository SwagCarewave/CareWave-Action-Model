"""Shared CSI loading and exact 315-D feature preprocessing."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["start_sec"] = pd.to_numeric(df["start_sec"], errors="raise")
    df["end_sec"] = pd.to_numeric(df["end_sec"], errors="raise")
    return df.sort_values(["start_sec", "end_sec"]).reset_index(drop=True)


def enrich_intervals(df: pd.DataFrame, sample_id: str, subject: str) -> pd.DataFrame:
    out = df.copy()
    out["sample_id"] = sample_id
    out["subject_id"] = subject
    out["event_id"] = ""
    out["phase"] = out["label"]
    out["label_confidence"] = 1.0
    fall_no = 0
    for index in out.index[out["label"].eq("falling")]:
        fall_no += 1
        out.at[index, "event_id"] = f"{sample_id}_fall_{fall_no:03d}"
    return out


def labels_for_times(times: np.ndarray, intervals: pd.DataFrame) -> np.ndarray:
    result = np.full(len(times), "unlabeled", dtype=object)
    for row in intervals.itertuples(index=False):
        result[(times >= float(row.start_sec)) & (times < float(row.end_sec))] = str(row.label)
    return result


def read_raw_csi(path: Path, sub_cols: list[str]) -> pd.DataFrame:
    """Read legacy variable-width exports without skipping rows or shifting metadata.

    Short amplitude rows are right-padded with NaN, as in the existing reader.
    This cannot recover subcarrier positions removed during acquisition.
    """
    try:
        raw = pd.read_csv(path)
        if set(sub_cols).issubset(raw.columns):
            return raw
    except pd.errors.ParserError:
        pass
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        prefix = ["experiment_id", "timestamp", "label", "rx"]
        if header[:4] != prefix or header[4:] != sub_cols[:len(header) - 4]:
            raise ValueError(f"{path.name}: unsupported raw header; manual review required")
        columns = prefix + sub_cols
        rows = []
        for line, row in enumerate(reader, 2):
            if not row:
                continue
            if not 5 <= len(row) <= len(columns):
                raise ValueError(f"{path.name}: unexpected width at line {line}: {len(row)}")
            rows.append(row + [None] * (len(columns) - len(row)))
    return pd.DataFrame(rows, columns=columns)


def load_csi_10hz(path: Path, data_cfg: dict) -> tuple[pd.DataFrame, np.ndarray, dict]:
    fps = int(data_cfg["target_fps"])
    rx_ids = list(data_cfg.get("rx_ids", ["RX1", "RX2", "RX3"]))
    sub_cols = [f"sub_{i}" for i in range(int(data_cfg.get("subcarriers_per_rx", 52)))]
    raw = read_raw_csi(path, sub_cols)
    missing = {"timestamp", "rx", *sub_cols} - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw["rx"] = raw["rx"].astype(str).str.strip().str.upper()
    raw[sub_cols] = raw[sub_cols].apply(pd.to_numeric, errors="coerce")
    raw = raw.dropna(subset=["timestamp"])
    raw = raw[raw["rx"].isin(rx_ids)].sort_values("timestamp")
    if raw.empty:
        raise ValueError(f"{path.name}: no usable CSI rows")
    origin = raw["timestamp"].min()
    raw["time_sec_raw"] = (raw["timestamp"] - origin).dt.total_seconds()
    raw["time_bin"] = np.rint(raw["time_sec_raw"] * fps).astype(int)
    grouped = raw.groupby(["time_bin", "rx"], as_index=False)[sub_cols].mean()
    first_bin, last_bin = int(grouped.time_bin.min()), int(grouped.time_bin.max())
    full_bins = np.arange(first_bin, last_bin + 1)
    arrays, missing_by_rx = [], []
    for rx in rx_ids:
        part = grouped[grouped.rx.eq(rx)].set_index("time_bin")[sub_cols].reindex(full_bins)
        missing_before = float(part.isna().all(axis=1).mean())
        part = part.interpolate(axis=0, limit_direction="both").interpolate(axis=1, limit_direction="both")
        if part.isna().any().any():
            raise ValueError(f"{path.name}: {rx} contains unfillable missing values")
        arrays.append(part.to_numpy(np.float32))
        missing_by_rx.append({"rx": rx, "missing_ratio_before_fill": missing_before})
    centered_parts, medians = [], []
    for array in arrays:
        receiver_median = np.median(array, axis=1, keepdims=True).astype(np.float32)
        centered_parts.append(array - receiver_median)
        medians.append(receiver_median)
    centered = np.concatenate(centered_parts, axis=1)
    delta = np.zeros_like(centered)
    delta[1:] = np.abs(centered[1:] - centered[:-1])
    features = np.concatenate([centered, delta, *medians], axis=1).astype(np.float32)
    if features.shape[1] != int(data_cfg.get("input_dim", 315)):
        raise ValueError(f"Expected 315 features, got {features.shape[1]}")
    frame = pd.DataFrame({"time_sec": (full_bins - first_bin).astype(np.float64) / fps})
    quality = {"sample_id": path.stem.removesuffix("_csi_raw"), "frame_count": len(frame), "rx": missing_by_rx,
               "max_time_gap_sec": float(raw.groupby("rx")["time_sec_raw"].diff().max())}
    return frame, features, quality


def load_scaler(path: Path, expected_dim: int = 315) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Bundled scaler not found: {path}")
    with np.load(path, allow_pickle=False) as scaler:
        median = np.asarray(scaler["median"], dtype=np.float32).reshape(-1)
        iqr = np.asarray(scaler["iqr"] if "iqr" in scaler else scaler["q75"] - scaler["q25"], dtype=np.float32).reshape(-1)
    if len(median) != expected_dim or len(iqr) != expected_dim:
        raise ValueError(f"Scaler dimension mismatch: median={len(median)}, iqr={len(iqr)}")
    iqr[np.abs(iqr) < 1e-8] = 1.0
    return median, iqr


def scale_features(features: np.ndarray, scaler_path: Path, clip_min: float, clip_max: float) -> np.ndarray:
    median, iqr = load_scaler(scaler_path, features.shape[1])
    return np.clip((features - median) / iqr, clip_min, clip_max).astype(np.float32)
