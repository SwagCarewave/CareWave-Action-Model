"""Flag fall events whose labelled timing does not match the CSI motion burst.

This does not replace video review and never sets review_status. It writes a CSV with,
per event, the time of the strongest CSI motion burst near the labelled onset, its offset
from the label, and how strong it is relative to the recording, so a person can check
the flagged events against the video first.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_action_labels import discover_pairs
from csi_dataset import load_config, load_csi_stream_split_10hz


def motion_curve(path: Path, data_cfg: dict, smooth_frames: int) -> tuple[np.ndarray, np.ndarray]:
    frame, features, _ = load_csi_stream_split_10hz(path, data_cfg)
    motion = features[:, -6:].mean(axis=1)
    motion = pd.Series(motion).rolling(smooth_frames, center=True, min_periods=1).mean().to_numpy()
    return frame["time_sec"].to_numpy(), motion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/action_stream_split_v2.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/metadata/fall_event_audit.csv"))
    parser.add_argument("--search-before-sec", type=float, default=3.0)
    parser.add_argument("--search-after-sec", type=float, default=2.0)
    parser.add_argument("--max-offset-sec", type=float, default=1.5)
    parser.add_argument("--min-strength", type=float, default=2.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = cfg["data"]
    events = pd.read_csv(data["events_path"])
    pairs, _, _ = discover_pairs(Path(data["raw_csi_dir"]), Path(data["labels_dir"]), data["split"].get("test_dir_name", "test"))
    raw_paths = {p["sample_id"]: Path(p["raw_path"]) for p in pairs}
    rows = []
    for sample_id, group in events.groupby("sample_id"):
        if sample_id not in raw_paths:
            continue
        times, motion = motion_curve(raw_paths[sample_id], data, int(data["target_fps"]))
        center = np.median(motion)
        spread = max(float(np.median(np.abs(motion - center))), 1e-9)
        for event in group.itertuples(index=False):
            onset, impact = float(event.onset_sec), float(event.impact_sec)
            mask = (times >= onset - args.search_before_sec) & (times <= impact + args.search_after_sec)
            if not mask.any():
                continue
            peak = int(np.flatnonzero(mask)[np.argmax(motion[mask])])
            strength = float((motion[peak] - center) / spread)
            offset = float(times[peak] - onset)
            reasons = []
            if strength < args.min_strength:
                reasons.append("weak_motion")
            if offset < -args.max_offset_sec or offset > (impact - onset) + args.max_offset_sec:
                reasons.append("peak_far_from_label")
            if onset - args.search_before_sec < 0:
                reasons.append("near_recording_start")
            rows.append({"sample_id": sample_id, "event_id": event.event_id, "onset_sec": onset, "impact_sec": impact,
                         "csi_peak_sec": float(times[peak]), "peak_offset_from_onset_sec": round(offset, 2),
                         "peak_strength_mad": round(strength, 2), "flag": "|".join(reasons),
                         "review_status": event.review_status})
    report = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False, encoding="utf-8-sig")
    flagged = report[report["flag"].ne("")]
    print(f"{len(report)} events audited, {len(flagged)} flagged -> {args.output}")
    print(flagged["flag"].str.split("|").explode().value_counts().to_string())
    print(f"peak offset from onset: median {report['peak_offset_from_onset_sec'].median():+.2f}s")


if __name__ == "__main__":
    main()
