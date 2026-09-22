"""Build 3-second, 315-D CSI windows with normal-motion hard negatives."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_action_labels import assign_splits, discover_pairs
from csi_dataset import (
    labels_for_times,
    load_config,
    load_csi_10hz,
    load_labels,
    save_json,
    scale_features,
)


def overlap_ratio(
    start: float,
    end: float,
    event_start: float,
    event_end: float,
) -> float:
    overlap = max(
        0.0,
        min(end, event_end) - max(start, event_start),
    )

    return overlap / max(end - start, 1e-9)


def build_recording(
    pair: dict,
    cfg: dict,
    events: pd.DataFrame,
):
    data = cfg["data"]
    preprocessing = cfg["preprocessing"]

    frame, raw_features, quality = load_csi_10hz(
        Path(pair["raw_path"]),
        data,
    )

    features = scale_features(
        raw_features,
        Path(data["scaler_path"]),
        float(preprocessing["clip_min"]),
        float(preprocessing["clip_max"]),
    )

    intervals = load_labels(Path(pair["label_path"]))

    times = frame["time_sec"].to_numpy()
    frame_labels = labels_for_times(times, intervals)

    sample_events = events[
        events["sample_id"].astype(str).eq(pair["sample_id"])
    ]

    fps = int(data["target_fps"])

    window = round(
        float(data["window_seconds"]) * fps
    )

    stride = round(
        float(data["stride_seconds"]) * fps
    )

    standing_threshold = float(
        data.get("standing_ratio_threshold", 0.80)
    )

    non_fall_threshold = float(
        data.get(
            "non_fall_ratio_threshold",
            standing_threshold,
        )
    )

    hard_negative_labels = {
        str(label).strip().lower()
        for label in data.get("hard_negative_labels", [])
    }

    non_fall_labels = {
        "standing",
        *hard_negative_labels,
    }

    xs: list[np.ndarray] = []
    ys: list[int] = []
    weights: list[float] = []
    metadata: list[dict] = []

    dropped = Counter()
    kept_sources = Counter()

    for first in range(
        0,
        len(features) - window + 1,
        stride,
    ):
        last = first + window

        start = float(times[first])
        end = start + window / fps
        center = (start + end) / 2.0

        positive = None
        any_fall_overlap = False

        for event in sample_events.itertuples(index=False):
            onset = float(event.onset_sec)
            impact = float(event.impact_sec)

            positive_start = (
                onset
                - float(data["fall_pre_onset_sec"])
            )

            positive_end = (
                impact
                + float(data["fall_post_impact_sec"])
            )

            ratio = overlap_ratio(
                start,
                end,
                positive_start,
                positive_end,
            )

            if ratio > 0.0:
                any_fall_overlap = True

            center_start = (
                onset
                - float(data["fall_center_pre_onset_sec"])
            )

            center_end = (
                impact
                + float(data["fall_center_post_impact_sec"])
            )

            center_ok = center_start <= center <= center_end

            if (
                ratio >= float(
                    data["fall_overlap_threshold"]
                )
                and center_ok
            ):
                positive = (event, ratio)
                break

        labels = np.asarray(
            frame_labels[first:last],
            dtype=object,
        )

        standing_ratio = float(
            np.mean(labels == "standing")
        )

        hard_negative_mask = np.isin(
            labels,
            list(hard_negative_labels),
        )

        hard_negative_ratio = float(
            np.mean(hard_negative_mask)
        )

        non_fall_mask = np.isin(
            labels,
            list(non_fall_labels),
        )

        non_fall_ratio = float(
            np.mean(non_fall_mask)
        )

        labels_in_window = Counter(
            str(label)
            for label in labels
        )

        if positive is not None:
            event, fall_ratio = positive

            target = 1
            event_id = str(event.event_id)
            binary_source = "falling"

            confidence = float(
                getattr(
                    event,
                    "label_confidence",
                    1.0,
                )
            )

            if fall_ratio < 0.50:
                sample_weight = float(
                    data["boundary_weight"]
                )
            else:
                sample_weight = max(
                    confidence,
                    float(data["boundary_weight"]),
                )

        elif (
            non_fall_ratio >= non_fall_threshold
            and not any_fall_overlap
        ):
            target = 0
            event_id = ""
            fall_ratio = 0.0

            if hard_negative_ratio > 0.0:
                binary_source = "hard_negative"
                sample_weight = float(
                    data.get("hard_negative_weight", 4.0)
                )
            else:
                binary_source = "standing"
                sample_weight = 1.0

        else:
            if any_fall_overlap:
                dropped["fall_boundary_ambiguous"] += 1
            elif non_fall_ratio < non_fall_threshold:
                dropped["excluded_or_mixed_label"] += 1
            else:
                dropped["ambiguous"] += 1

            continue

        xs.append(features[first:last])
        ys.append(target)
        weights.append(sample_weight)

        kept_sources[binary_source] += 1

        metadata.append(
            {
                "sample_id": pair["sample_id"],
                "subject_id": pair["subject"],
                "session_id": pair["sample_id"],
                "room_id": "unknown",
                "split": pair["split"],
                "start_sec": start,
                "end_sec": end,
                "center_sec": center,
                "label": (
                    "falling"
                    if target == 1
                    else "standing"
                ),
                "label_id": target,
                "binary_source": binary_source,
                "standing_ratio": standing_ratio,
                "hard_negative_ratio": (
                    hard_negative_ratio
                ),
                "non_fall_ratio": non_fall_ratio,
                "fall_overlap_ratio": fall_ratio,
                "sample_weight": sample_weight,
                "event_id": event_id,
                "source_labels": "|".join(
                    f"{label}:{count}"
                    for label, count
                    in sorted(labels_in_window.items())
                ),
            }
        )

    quality.update(
        {
            "split": pair["split"],
            "windows_kept": len(xs),
            "windows_kept_by_source": dict(
                kept_sources
            ),
            "windows_dropped": dict(dropped),
        }
    )

    return (
        xs,
        ys,
        weights,
        metadata,
        quality,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/action_fusion.yaml"
        ),
    )

    parser.add_argument(
        "--allow-unreviewed-events",
        action="store_true",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    data = cfg["data"]

    event_path = Path(data["events_path"])

    if not event_path.exists():
        raise SystemExit(
            "Run build_action_labels.py first: "
            f"{event_path} not found"
        )

    events = pd.read_csv(event_path)

    required_event_columns = {
        "sample_id",
        "event_id",
        "onset_sec",
        "impact_sec",
        "review_status",
    }

    missing_columns = (
        required_event_columns
        - set(events.columns)
    )

    if missing_columns:
        raise SystemExit(
            "Event metadata missing columns: "
            f"{sorted(missing_columns)}"
        )

    require_reviewed = bool(
        data.get(
            "require_reviewed_events",
            True,
        )
    )

    if (
        require_reviewed
        and not args.allow_unreviewed_events
    ):
        pending = (
            events["review_status"]
            .astype(str)
            .str.strip()
            .str.lower()
            .ne("reviewed")
        )

        if pending.any():
            raise SystemExit(
                f"{int(pending.sum())} events need "
                "review. Fix onset/impact and set "
                "review_status=reviewed, or use "
                "--allow-unreviewed-events for a "
                "temporary experiment."
            )

    raw_root = Path(data["raw_csi_dir"])
    label_root = Path(data["labels_dir"])

    split_config = data["split"]

    pairs, raw_only, label_only = discover_pairs(
        raw_root,
        label_root,
        split_config.get(
            "test_dir_name",
            "test",
        ),
    )

    pairs = assign_splits(
        pairs,
        split_config,
        int(cfg.get("seed", 42)),
    )

    all_x: list[np.ndarray] = []
    all_y: list[int] = []
    all_weights: list[float] = []
    all_metadata: list[dict] = []
    quality_reports: list[dict] = []

    failed_samples: list[str] = []

    for pair in pairs:
        try:
            (
                x,
                y,
                sample_weights,
                metadata,
                quality,
            ) = build_recording(
                pair,
                cfg,
                events,
            )

        except Exception as exc:
            failed_samples.append(
                pair["sample_id"]
            )

            quality_reports.append(
                {
                    "sample_id": pair["sample_id"],
                    "error": str(exc),
                }
            )

            print(
                f"[ERROR] "
                f"{pair['sample_id']}: {exc}"
            )

            continue

        all_x.extend(x)
        all_y.extend(y)
        all_weights.extend(sample_weights)
        all_metadata.extend(metadata)
        quality_reports.append(quality)

        source_counts = Counter(
            row["binary_source"]
            for row in metadata
        )

        print(
            f"[OK] {pair['sample_id']} "
            f"[{pair['split']}]: "
            f"{len(x)} windows "
            f"{dict(source_counts)}"
        )

    if failed_samples:
        raise SystemExit(
            "Window generation failed for: "
            + ", ".join(failed_samples)
        )

    if not all_x:
        raise SystemExit(
            "No windows were created."
        )

    stacked_x = np.stack(all_x).astype(
        np.float32
    )

    stacked_y = np.asarray(
        all_y,
        dtype=np.int64,
    )

    stacked_weights = np.asarray(
        all_weights,
        dtype=np.float32,
    )

    output_dir = Path(data["processed_dir"])
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_dir / "X.npy",
        stacked_x,
    )

    np.save(
        output_dir / "y.npy",
        stacked_y,
    )

    np.save(
        output_dir / "sample_weights.npy",
        stacked_weights,
    )

    metadata_df = pd.DataFrame(
        all_metadata
    )

    metadata_df.to_csv(
        output_dir / "metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_json(
        output_dir / "class_names.json",
        data["classes"],
    )

    save_json(
        output_dir / "quality_report.json",
        {
            "recordings": quality_reports,
            "raw_without_label": raw_only,
            "label_without_raw": label_only,
        },
    )

    class_counts = Counter(all_y)

    source_counts = Counter(
        row["binary_source"]
        for row in all_metadata
    )

    split_counts = (
        metadata_df
        .groupby(
            [
                "split",
                "label",
                "binary_source",
            ]
        )
        .size()
    )

    print(
        f"\nSaved X={stacked_x.shape}"
    )

    print(
        f"Class counts={class_counts}"
    )

    print(
        f"Source counts={source_counts}"
    )

    print("\nSplit/source counts:")
    print(split_counts)


if __name__ == "__main__":
    main()
