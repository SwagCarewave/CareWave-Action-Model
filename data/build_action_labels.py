"""Validate, match and split raw CSI recordings and interval labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from csi_dataset import enrich_intervals, load_config, load_labels

REQUIRED_LABEL_COLUMNS = {"video_name", "start_sec", "end_sec", "label"}


def sample_id_from_raw(path: Path) -> str:
    suffix = "_csi_raw"
    if not path.stem.endswith(suffix):
        raise ValueError(f"Unexpected raw CSI filename: {path.name}")
    return path.stem[: -len(suffix)]


def sample_id_from_label(path: Path) -> str:
    suffix = "_labels"
    if not path.stem.endswith(suffix):
        raise ValueError(f"Unexpected label filename: {path.name}")
    return path.stem[: -len(suffix)]


def _index_unique(paths: list[Path], id_fn) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in paths:
        sample_id = id_fn(path)
        if sample_id in indexed:
            raise ValueError(f"Duplicate sample_id {sample_id!r}: {indexed[sample_id]} and {path}")
        indexed[sample_id] = path
    return indexed


def _identity(path: Path, root: Path, sample_id: str, test_dir_name: str) -> tuple[str, bool]:
    relative_parts = path.relative_to(root).parts
    is_test = test_dir_name in relative_parts[:-1]
    if is_test:
        subject = sample_id.split("_", 1)[0]
    else:
        subject = relative_parts[0]
    return subject, is_test


def discover_pairs(
    raw_root: Path, label_root: Path, test_dir_name: str = "test"
) -> tuple[list[dict], list[str], list[str]]:
    raw = _index_unique(list(raw_root.rglob("*_csi_raw.csv")), sample_id_from_raw)
    labels = _index_unique(list(label_root.rglob("*_labels.csv")), sample_id_from_label)
    pairs = []
    for sample_id in sorted(raw.keys() & labels.keys()):
        raw_subject, raw_is_test = _identity(raw[sample_id], raw_root, sample_id, test_dir_name)
        label_subject, label_is_test = _identity(labels[sample_id], label_root, sample_id, test_dir_name)
        if (raw_subject, raw_is_test) != (label_subject, label_is_test):
            raise ValueError(f"Raw/label folder mismatch for {sample_id}")
        pairs.append({
            "sample_id": sample_id,
            "subject": raw_subject,
            "is_test": raw_is_test,
            "raw_path": str(raw[sample_id]),
            "label_path": str(labels[sample_id]),
        })
    return pairs, sorted(raw.keys() - labels.keys()), sorted(labels.keys() - raw.keys())


def _recording_family(sample_id: str, subject: str) -> str:
    name = sample_id.removeprefix(f"{subject}_")
    return re.sub(r"_\d+$", "", name)


def _stable_order(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def assign_splits(pairs: list[dict], split_cfg: dict, default_seed: int = 42) -> list[dict]:
    """Assign explicit test files to test and ordinary files to train/val.

    Train/val assignment is deterministic and happens at whole-recording level.
    Repeated recordings in the same subject/action family are approximately
    stratified so both splits retain useful action coverage.
    """
    mode = str(split_cfg.get("mode", "session"))
    if mode == "subject":
        result = []
        for pair in pairs:
            item = dict(pair)
            if item["is_test"]:
                item["split"] = "test"
            else:
                matches = [
                    name for name in ("train", "val", "test")
                    if item["subject"] in set(split_cfg.get(f"{name}_subjects", []))
                ]
                if len(matches) != 1:
                    raise ValueError(f"Subject {item['subject']!r} has invalid split assignment")
                item["split"] = matches[0]
            result.append(item)
        return result
    if mode not in {"session", "recording", "file"}:
        raise ValueError(f"Unsupported split mode: {mode!r}")

    ratio = float(split_cfg.get("val_ratio", 0.2))
    if not 0.0 < ratio < 1.0:
        raise ValueError("split.val_ratio must be between 0 and 1")
    seed = int(split_cfg.get("seed", default_seed))
    result = [dict(pair, split="test" if pair["is_test"] else "train") for pair in pairs]
    ordinary = [pair for pair in result if not pair["is_test"]]

    families: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for pair in ordinary:
        families[(pair["subject"], _recording_family(pair["sample_id"], pair["subject"]))].append(pair)
    for group in families.values():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda p: _stable_order(p["sample_id"], seed))
        val_count = min(len(group) - 1, max(1, round(len(group) * ratio)))
        for pair in ordered[:val_count]:
            pair["split"] = "val"

    # If a subject only has singleton action families, still reserve one whole
    # recording for validation while leaving at least one training recording.
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for pair in ordinary:
        by_subject[pair["subject"]].append(pair)
    for subject_pairs in by_subject.values():
        if len(subject_pairs) >= 2 and not any(p["split"] == "val" for p in subject_pairs):
            candidate = min(subject_pairs, key=lambda p: _stable_order(p["sample_id"], seed))
            candidate["split"] = "val"

    if not any(p["split"] == "train" for p in result) or not any(p["split"] == "val" for p in result):
        raise ValueError("The recording-level split did not produce both train and val data")
    return result


def validate_label_file(path: Path, allowed_labels: set[str]) -> list[str]:
    errors: list[str] = []
    df = pd.read_csv(path)
    missing = REQUIRED_LABEL_COLUMNS - set(df.columns)
    if missing:
        return [f"missing columns: {sorted(missing)}"]
    df["start_sec"] = pd.to_numeric(df["start_sec"], errors="coerce")
    df["end_sec"] = pd.to_numeric(df["end_sec"], errors="coerce")
    if df[["start_sec", "end_sec"]].isna().any().any():
        errors.append("non-numeric start_sec/end_sec")
    if (df["end_sec"] <= df["start_sec"]).any():
        errors.append("end_sec must be greater than start_sec")
    unknown = sorted(set(df["label"].astype(str).str.strip()) - allowed_labels)
    if unknown:
        errors.append(f"unknown labels: {unknown}")
    ordered = df.sort_values(["start_sec", "end_sec"])
    if len(ordered) > 1:
        overlap = ordered["start_sec"].iloc[1:].to_numpy() < ordered["end_sec"].iloc[:-1].to_numpy()
        if overlap.any():
            errors.append("overlapping intervals")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/action_fusion.yaml"))
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--labels-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/csi_stream/pair_report.json"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    split_cfg = data_cfg["split"]
    raw_root = args.raw_dir or Path(data_cfg["raw_csi_dir"])
    label_root = args.labels_dir or Path(data_cfg["labels_dir"])
    test_dir_name = str(split_cfg.get("test_dir_name", "test"))

    pairs, raw_without_label, label_without_raw = discover_pairs(raw_root, label_root, test_dir_name)
    pairs = assign_splits(pairs, split_cfg, int(cfg.get("seed", 42)))
    allowed = {
        "standing", "falling", "lying", "transition", "getting_up", "ignore",
        "slow_lying_down", "adjusting_position", "turning_body",
        "lowering_arms", "raising_arms", "turning_head", "walking",
    }
    invalid = {}
    enriched = []
    manifest = []
    for pair in pairs:
        errors = validate_label_file(Path(pair["label_path"]), allowed)
        if errors:
            invalid[pair["sample_id"]] = errors
            continue
        intervals = enrich_intervals(load_labels(Path(pair["label_path"])), pair["sample_id"], pair["subject"])
        enriched.append(intervals)
        manifest.append({
            "sample_id": pair["sample_id"], "subject_id": pair["subject"],
            "session_id": pair["sample_id"], "room_id": "unknown",
            "is_explicit_test": pair["is_test"], "split": pair["split"],
        })
    report = {
        "matched_count": len(pairs), "pairs": pairs,
        "raw_without_label": raw_without_label,
        "label_without_raw": label_without_raw,
        "invalid_labels": invalid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if enriched:
        pd.concat(enriched, ignore_index=True).to_csv(
            args.output.parent / "action_intervals.csv", index=False, encoding="utf-8-sig"
        )
    split_path = Path("data/splits/split_manifest.csv")
    split_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).to_csv(split_path, index=False, encoding="utf-8-sig")
    print(f"Matched: {len(pairs)}")
    print(f"Raw without label: {len(raw_without_label)}")
    print(f"Label without raw: {len(label_without_raw)}")
    print(f"Invalid label files: {len(invalid)}")
    print(pd.DataFrame(manifest).groupby(["split", "subject_id"]).size())
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
