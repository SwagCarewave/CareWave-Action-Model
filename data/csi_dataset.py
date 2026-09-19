"""Shared CSI preprocessing and dataset helpers.

The historical collector removed zero-amplitude entries before writing rows.
For already collected CSVs the deleted original index cannot be reconstructed.
This module therefore keeps compatibility with the first-stage preprocessing:
52 named columns, 0.1 s bins, per-RX means and time-axis interpolation. Quality
statistics are retained so suspicious windows can be filtered or audited.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

SUBCARRIER_COLUMNS = [f"sub_{i}" for i in range(52)]
FEATURE_COLUMNS = [f"rx{rx}_sub_{i}" for rx in (1, 2, 3) for i in range(52)]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_raw_csi(path: Path) -> pd.DataFrame:
    fixed = ["experiment_id", "timestamp", "label", "rx", *SUBCARRIER_COLUMNS]
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        df = pd.read_csv(path, header=None, names=fixed, skiprows=1, engine="python", on_bad_lines="skip")
    for col in fixed:
        if col not in df.columns:
            df[col] = np.nan
    df = df[fixed].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp", "rx"]).copy()
    for col in SUBCARRIER_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["missing_subcarriers"] = df[SUBCARRIER_COLUMNS].isna().sum(axis=1)
    return df


def _rx_number(value: object) -> int | None:
    text = str(value).upper().replace("RX", "").strip()
    return int(text) if text in {"1", "2", "3"} else None


def preprocess_raw(path: Path, target_fps: int = 10) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = read_raw_csi(path)
    if df.empty:
        raise ValueError(f"No valid CSI rows: {path}")
    df["rx_num"] = df["rx"].map(_rx_number)
    df = df.dropna(subset=["rx_num"]).copy()
    df["rx_num"] = df["rx_num"].astype(int)
    start = df["timestamp"].min()
    df["time_sec"] = (df["timestamp"] - start).dt.total_seconds()
    df["time_bin"] = np.rint(df["time_sec"] * target_fps).astype(int)

    max_bin = int(df["time_bin"].max())
    out = pd.DataFrame({"time_bin": np.arange(max_bin + 1, dtype=int)})
    out["time_sec"] = out["time_bin"] / float(target_fps)
    quality_parts = []
    for rx in (1, 2, 3):
        part = df[df["rx_num"] == rx]
        timestamps = part["timestamp"].sort_values()
        gaps = timestamps.diff().dt.total_seconds().dropna().to_numpy(dtype=float)
        valid_gaps = gaps[gaps > 0]
        if len(valid_gaps):
            gap_p10, gap_median, gap_p90 = np.percentile(valid_gaps, [10, 50, 90])
            effective_hz = 1.0 / gap_median
        else:
            gap_p10 = gap_median = gap_p90 = effective_hz = float("nan")
        grouped = part.groupby("time_bin")[SUBCARRIER_COLUMNS].mean()
        grouped.columns = [f"rx{rx}_{c}" for c in grouped.columns]
        out = out.merge(grouped, left_on="time_bin", right_index=True, how="left")
        counts = part.groupby("time_bin").size().rename(f"rx{rx}_packet_count")
        out = out.merge(counts, left_on="time_bin", right_index=True, how="left")
        expected = max(1, int(round((float(part["time_sec"].max()) - float(part["time_sec"].min())) * effective_hz)) + 1) if len(part) and np.isfinite(effective_hz) else 0
        quality_parts.append({
            "rx": rx,
            "raw_rows": int(len(part)),
            "effective_hz": float(effective_hz),
            "timestamp_gap_p10_sec": float(gap_p10),
            "timestamp_gap_median_sec": float(gap_median),
            "timestamp_gap_p90_sec": float(gap_p90),
            "estimated_missing_ratio": float(max(0.0, 1.0 - len(part) / expected)) if expected else float("nan"),
            "start_offset_sec": float((timestamps.min() - start).total_seconds()) if len(timestamps) else float("nan"),
            "end_sec": float((timestamps.max() - start).total_seconds()) if len(timestamps) else float("nan"),
            "rows_with_missing_subcarriers": int((part["missing_subcarriers"] > 0).sum()),
        })

    for col in FEATURE_COLUMNS:
        if col not in out:
            out[col] = np.nan
    # Consolidate the frame after repeated RX merges to avoid fragmented-frame
    # warnings on large datasets.
    out = out.copy()
    missing_before = out[FEATURE_COLUMNS].isna().mean(axis=1)
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].interpolate(axis=0, limit_direction="both")
    # A feature missing for an entire recording cannot be interpolated.
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].fillna(0.0)
    for rx in (1, 2, 3):
        col = f"rx{rx}_packet_count"
        out[col] = out[col].fillna(0).astype(int)
    out = pd.concat([out, missing_before.astype(float).rename("missing_ratio_before_fill")], axis=1)
    report = {
        "sample_id": path.stem.replace("_csi_raw", ""),
        "raw_rows": int(len(df)),
        "duration_sec": float(out["time_sec"].iloc[-1]),
        "frames_10fps": int(len(out)),
        "raw_rows_with_missing_subcarriers": int((df["missing_subcarriers"] > 0).sum()),
        "mean_frame_missing_ratio_before_fill": float(missing_before.mean()),
        "fully_observed_frame_ratio": float((missing_before == 0).mean()),
        "video_csi_offset_sec": 0.0,
        "rx": quality_parts,
        "known_limitation": "Deleted zero-amplitude middle indices cannot be reconstructed from existing CSV rows.",
    }
    return out[["time_sec", *FEATURE_COLUMNS, "missing_ratio_before_fill", "rx1_packet_count", "rx2_packet_count", "rx3_packet_count"]], report


def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"start_sec", "end_sec", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing label columns in {path}: {sorted(required - set(df.columns))}")
    df = df.copy()
    df["start_sec"] = pd.to_numeric(df["start_sec"], errors="raise")
    df["end_sec"] = pd.to_numeric(df["end_sec"], errors="raise")
    df["label"] = df["label"].astype(str).str.strip()
    return df.sort_values(["start_sec", "end_sec"]).reset_index(drop=True)


def enrich_intervals(intervals: pd.DataFrame, sample_id: str, subject: str) -> pd.DataFrame:
    """Add deployable metadata without pretending onset/impact were annotated."""
    out = intervals.copy()
    event_counter = 0
    active_event = ""
    event_ids, phases = [], []
    for label in out["label"].astype(str):
        if label == "falling":
            event_counter += 1
            active_event = f"{sample_id}_fall_{event_counter:03d}"
            phase = "event"
        elif label == "lying" and active_event:
            phase = "post"
        elif label in {"transition", "getting_up"}:
            phase = "recovery"
        elif label == "standing":
            phase = "steady"
            active_event = ""
        else:
            phase = "other"
        event_ids.append(active_event)
        phases.append(phase)
    out["sample_id"] = sample_id
    out["subject_id"] = subject
    out["session_id"] = "unknown"
    out["room_id"] = "unknown"
    out["action_label"] = out["label"]
    out["event_id"] = event_ids
    out["phase"] = phases
    out["label_confidence"] = 1.0
    return out


def labels_for_times(times: np.ndarray, intervals: pd.DataFrame) -> np.ndarray:
    result = np.full(len(times), "unlabeled", dtype=object)
    for row in intervals.itertuples(index=False):
        mask = (times >= float(row.start_sec)) & (times < float(row.end_sec))
        result[mask] = str(row.label)
    return result.astype(str)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
