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


def assign_packet_streams(shapes: np.ndarray, init_packets: int = 60, ema_alpha: float = 0.05) -> np.ndarray:
    """Causally split one receiver's packets into its two interleaved spectral streams.

    Each RX receives two packet types whose amplitude spectra are strongly anti-correlated
    and alternate at random. Averaging them together (as load_csi_10hz does) turns the
    switching into large frame-to-frame noise that hides body motion. Centroids start from
    2-means on the first packets, then follow slow drift with an EMA; each packet only uses
    past data, so the same code works for streaming inference.
    """
    from sklearn.cluster import KMeans

    if len(shapes) < 4:
        raise ValueError("Too few packets to split streams")
    init = shapes[:max(4, min(init_packets, len(shapes)))]
    centroids = KMeans(2, n_init=5, random_state=0).fit(init).cluster_centers_
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    # Canonical order so stream 0/1 means the same thing across recordings.
    half = shapes.shape[1] // 2
    if (centroids[0, :half] ** 2).sum() < (centroids[1, :half] ** 2).sum():
        centroids = centroids[::-1].copy()
    labels = np.empty(len(shapes), dtype=np.int64)
    for i, shape in enumerate(shapes):
        k = int(np.argmax(centroids @ shape))
        labels[i] = k
        centroids[k] = (1.0 - ema_alpha) * centroids[k] + ema_alpha * shape
        centroids[k] /= np.linalg.norm(centroids[k])
    return labels


def stream_separation(shapes: np.ndarray, labels: np.ndarray) -> float | None:
    """Median of (cosine to own stream mean - cosine to the other stream mean).

    Higher is cleaner (median 0.20 over the 2026-06 recordings); values near 0 mean the
    two streams were barely separable and the recording's stream features should be checked.
    """
    if (labels == 0).sum() < 2 or (labels == 1).sum() < 2:
        return None
    means = np.stack([shapes[labels == k].mean(axis=0) for k in (0, 1)])
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    similarity = shapes @ means.T
    own = similarity[np.arange(len(labels)), labels]
    other = similarity[np.arange(len(labels)), 1 - labels]
    return round(float(np.median(own - other)), 4)


def subtract_causal_baseline(values: np.ndarray, tau_frames: float) -> np.ndarray:
    """Subtract a causal EMA of each column so only change relative to the recent past remains.

    Removes the slowly varying room/session spectrum while keeping within-window motion
    and posture changes that last shorter than about tau_frames.
    """
    alpha = 1.0 / max(1.0, tau_frames)
    baseline = np.empty_like(values)
    current = values[0].astype(np.float64)
    for i, row in enumerate(values):
        current += alpha * (row - current)
        baseline[i] = current
    return (values - baseline).astype(np.float32)


def load_csi_stream_split_10hz(path: Path, data_cfg: dict) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Per RX and packet stream: unit-norm spectrum (52) and packet-to-packet shape change (1).

    Feature layout: for rx in rx_ids, for stream in (0, 1): 52 shape values; then
    for rx, for stream: 1 motion value. 3 RX -> 312 + 6 = 318 dimensions.
    """
    fps = int(data_cfg["target_fps"])
    rx_ids = list(data_cfg.get("rx_ids", ["RX1", "RX2", "RX3"]))
    n_sub = int(data_cfg.get("subcarriers_per_rx", 52))
    sub_cols = [f"sub_{i}" for i in range(n_sub)]
    stream_cfg = data_cfg.get("stream_split", {})
    raw = read_raw_csi(path, sub_cols)
    missing = {"timestamp", "rx", *sub_cols} - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw["rx"] = raw["rx"].astype(str).str.strip().str.upper()
    raw[sub_cols] = raw[sub_cols].apply(pd.to_numeric, errors="coerce")
    raw = raw.dropna(subset=["timestamp"])
    raw = raw[raw["rx"].isin(rx_ids)].sort_values("timestamp", kind="stable")
    if raw.empty:
        raise ValueError(f"{path.name}: no usable CSI rows")
    origin = raw["timestamp"].min()
    raw["time_sec_raw"] = (raw["timestamp"] - origin).dt.total_seconds()
    last_bin = int(np.rint(raw["time_sec_raw"].max() * fps))
    full_bins = np.arange(0, last_bin + 1)
    shape_parts, motion_parts, quality_rx = [], [], []
    for rx in rx_ids:
        part = raw[raw["rx"].eq(rx)]
        # Short legacy rows lost trailing values; keep only complete spectra for shape tracking.
        part = part[part[sub_cols].notna().all(axis=1)]
        amplitude = part[sub_cols].to_numpy(np.float64)
        norm = np.linalg.norm(amplitude, axis=1)
        part, amplitude, norm = part[norm > 0], amplitude[norm > 0], norm[norm > 0]
        shapes = amplitude / norm[:, None]
        source_column = stream_cfg.get("source_column")
        if source_column and source_column in part.columns:
            # Recorded packet source (e.g. transmitter MAC): use it instead of shape clustering.
            sources = part[source_column].astype(str).str.strip()
            top = sources.value_counts().index[:2].sort_values()
            if len(top) != 2:
                raise ValueError(f"{path.name}: {rx} needs 2 packet sources in {source_column}, found {len(top)}")
            keep = sources.isin(top).to_numpy()
            part, shapes = part[keep], shapes[keep]
            labels = (sources[keep] == top[1]).to_numpy().astype(np.int64)
            assignment = f"column:{source_column}"
        else:
            labels = assign_packet_streams(shapes, int(stream_cfg.get("init_packets", 60)),
                                           float(stream_cfg.get("ema_alpha", 0.05)))
            assignment = "shape_clustering"
        bins = np.rint(part["time_sec_raw"].to_numpy() * fps).astype(int)
        rx_quality = {"rx": rx, "packets": int(len(shapes)), "stream_assignment": assignment,
                      "stream_separation": stream_separation(shapes, labels)}
        for stream in (0, 1):
            mask = labels == stream
            stream_shapes, stream_bins = shapes[mask], bins[mask]
            change = np.full(len(stream_shapes), np.nan)
            change[1:] = np.linalg.norm(np.diff(stream_shapes, axis=0), axis=1)
            frame = pd.DataFrame(stream_shapes, columns=sub_cols)
            frame["motion"] = change
            frame["time_bin"] = stream_bins
            binned = frame.groupby("time_bin").mean().reindex(full_bins)
            rx_quality[f"stream{stream}_packets"] = int(mask.sum())
            rx_quality[f"stream{stream}_missing_ratio_before_fill"] = float(binned[sub_cols].isna().all(axis=1).mean())
            binned = binned.interpolate(axis=0, limit_direction="both")
            if binned.isna().any().any():
                raise ValueError(f"{path.name}: {rx} stream {stream} contains unfillable missing values")
            shape = binned[sub_cols].to_numpy(np.float32)
            baseline_sec = float(data_cfg.get("shape_baseline_sec", 0.0))
            if baseline_sec > 0:
                shape = subtract_causal_baseline(shape, baseline_sec * fps)
            shape_parts.append(shape)
            motion_parts.append(binned[["motion"]].to_numpy(np.float32))
        quality_rx.append(rx_quality)
    features = np.concatenate(shape_parts + motion_parts, axis=1).astype(np.float32)
    expected = int(data_cfg.get("input_dim", features.shape[1]))
    if features.shape[1] != expected:
        raise ValueError(f"Expected {expected} features, got {features.shape[1]}")
    frame = pd.DataFrame({"time_sec": full_bins.astype(np.float64) / fps})
    quality = {"sample_id": path.stem.removesuffix("_csi_raw"), "frame_count": len(frame), "rx": quality_rx,
               "max_time_gap_sec": float(raw.groupby("rx")["time_sec_raw"].diff().max())}
    return frame, features, quality


def load_csi_features(path: Path, data_cfg: dict) -> tuple[pd.DataFrame, np.ndarray, dict]:
    if str(data_cfg.get("feature_mode", "legacy_315")) == "stream_split":
        return load_csi_stream_split_10hz(path, data_cfg)
    return load_csi_10hz(path, data_cfg)


def stream_shape_dims(data_cfg: dict) -> int:
    return len(data_cfg.get("rx_ids", ["RX1", "RX2", "RX3"])) * 2 * int(data_cfg.get("subcarriers_per_rx", 52))


def transform_window(window: np.ndarray, data_cfg: dict) -> np.ndarray:
    """Per-window transform shared by training data build and inference.

    With window_center_shape, each stream spectrum is expressed relative to its own
    mean inside the window, removing the static room/session fingerprint.
    """
    window = np.asarray(window, dtype=np.float32)
    if data_cfg.get("feature_mode") == "stream_split" and data_cfg.get("window_center_shape", False):
        window = window.copy()
        dims = stream_shape_dims(data_cfg)
        window[:, :dims] -= window[:, :dims].mean(axis=0, keepdims=True)
    return window


def fit_robust_scaler(windows: np.ndarray, path: Path) -> None:
    frames = windows.reshape(-1, windows.shape[-1]).astype(np.float64)
    median = np.median(frames, axis=0)
    q25, q75 = np.percentile(frames, [25, 75], axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, median=median.astype(np.float32), iqr=(q75 - q25).astype(np.float32))


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
