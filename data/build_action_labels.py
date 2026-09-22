"""Validate pairs, assign leakage-safe splits, and prepare fall-event metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from csi_dataset import enrich_intervals, load_config, load_labels

REQUIRED = {"video_name", "start_sec", "end_sec", "label"}
ALLOWED = {"standing", "falling", "lying", "transition", "getting_up", "ignore", "slow_lying_down",
           "adjusting_position", "turning_body", "lowering_arms", "raising_arms", "turning_head", "walking"}


def sample_id_from_raw(path: Path) -> str:
    if not path.stem.endswith("_csi_raw"):
        raise ValueError(f"Unexpected raw filename: {path.name}")
    return path.stem[:-8]


def sample_id_from_label(path: Path) -> str:
    if not path.stem.endswith("_labels"):
        raise ValueError(f"Unexpected label filename: {path.name}")
    return path.stem[:-7]


def _index(paths: list[Path], fn) -> dict[str, Path]:
    result = {}
    for path in paths:
        key = fn(path)
        if key in result:
            raise ValueError(f"Duplicate sample_id: {key}")
        result[key] = path
    return result


def _identity(path: Path, root: Path, sample_id: str, test_name: str) -> tuple[str, bool]:
    parts = path.relative_to(root).parts
    is_test = test_name in parts[:-1]
    return (sample_id.split("_", 1)[0] if is_test else parts[0]), is_test


def discover_pairs(raw_root: Path, label_root: Path, test_name: str = "test") -> tuple[list[dict], list[str], list[str]]:
    raw = _index(list(raw_root.rglob("*_csi_raw.csv")), sample_id_from_raw)
    labels = _index(list(label_root.rglob("*_labels.csv")), sample_id_from_label)
    pairs = []
    for sample_id in sorted(raw.keys() & labels.keys()):
        subject, is_test = _identity(raw[sample_id], raw_root, sample_id, test_name)
        if (subject, is_test) != _identity(labels[sample_id], label_root, sample_id, test_name):
            raise ValueError(f"Raw/label folder mismatch: {sample_id}")
        pairs.append({"sample_id": sample_id, "subject": subject, "is_test": is_test,
                      "raw_path": str(raw[sample_id]), "label_path": str(labels[sample_id])})
    return pairs, sorted(raw.keys() - labels.keys()), sorted(labels.keys() - raw.keys())


def _stable(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _family(sample_id: str, subject: str) -> str:
    return re.sub(r"_\d+$", "", sample_id.removeprefix(f"{subject}_"))


def assign_splits(pairs: list[dict], cfg: dict, default_seed: int = 42) -> list[dict]:
    mode = str(cfg.get("mode", "session"))
    if mode == "subject":
        output = []
        for pair in pairs:
            item = dict(pair)
            if item["is_test"]:
                item["split"] = "test"
            else:
                matches = [s for s in ("train", "val", "test") if item["subject"] in cfg.get(f"{s}_subjects", [])]
                if len(matches) != 1:
                    raise ValueError(f"Invalid subject split for {item['subject']}")
                item["split"] = matches[0]
            output.append(item)
        return output
    if mode not in {"session", "recording", "file"}:
        raise ValueError(f"Unsupported split mode: {mode}")
    ratio, seed = float(cfg.get("val_ratio", .2)), int(cfg.get("seed", default_seed))
    result = [dict(p, split="test" if p["is_test"] else "train") for p in pairs]
    groups = defaultdict(list)
    for p in result:
        if not p["is_test"]:
            groups[(p["subject"], _family(p["sample_id"], p["subject"]))].append(p)
    for group in groups.values():
        if len(group) >= 2:
            n = min(len(group) - 1, max(1, round(len(group) * ratio)))
            for p in sorted(group, key=lambda x: _stable(x["sample_id"], seed))[:n]:
                p["split"] = "val"
    by_subject = defaultdict(list)
    for p in result:
        if not p["is_test"]:
            by_subject[p["subject"]].append(p)
    for group in by_subject.values():
        if len(group) >= 2 and not any(p["split"] == "val" for p in group):
            min(group, key=lambda x: _stable(x["sample_id"], seed))["split"] = "val"
    if not any(p["split"] == "train" for p in result) or not any(p["split"] == "val" for p in result):
        raise ValueError("Split must contain train and val recordings")
    return result


def validate_label_file(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if missing := REQUIRED - set(df.columns):
        return [f"missing columns: {sorted(missing)}"]
    errors = []
    starts, ends = pd.to_numeric(df.start_sec, errors="coerce"), pd.to_numeric(df.end_sec, errors="coerce")
    if starts.isna().any() or ends.isna().any(): errors.append("non-numeric time")
    if (ends <= starts).any(): errors.append("end_sec must exceed start_sec")
    if unknown := sorted(set(df.label.astype(str).str.strip().str.lower()) - ALLOWED): errors.append(f"unknown labels: {unknown}")
    ordered = df.assign(start_sec=starts, end_sec=ends).sort_values(["start_sec", "end_sec"])
    if len(ordered) > 1 and (ordered.start_sec.iloc[1:].to_numpy() < ordered.end_sec.iloc[:-1].to_numpy()).any():
        errors.append("overlapping intervals")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/action_fusion.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config); data = cfg["data"]; split_cfg = data["split"]
    pairs, raw_only, label_only = discover_pairs(Path(data["raw_csi_dir"]), Path(data["labels_dir"]), split_cfg.get("test_dir_name", "test"))
    pairs = assign_splits(pairs, split_cfg, int(cfg.get("seed", 42)))
    invalid, intervals_all, manifest, draft_events = {}, [], [], []
    for pair in pairs:
        errors = validate_label_file(Path(pair["label_path"]))
        if errors:
            invalid[pair["sample_id"]] = errors; continue
        intervals = enrich_intervals(load_labels(Path(pair["label_path"])), pair["sample_id"], pair["subject"])
        intervals_all.append(intervals)
        manifest.append({"sample_id": pair["sample_id"], "subject_id": pair["subject"], "session_id": pair["sample_id"],
                         "room_id": "unknown", "is_explicit_test": pair["is_test"], "split": pair["split"]})
        for row in intervals[intervals.label.eq("falling")].itertuples(index=False):
            draft_events.append({"sample_id": pair["sample_id"], "subject_id": pair["subject"],
                "session_id": pair["sample_id"], "room_id": "unknown", "event_id": row.event_id,
                "onset_sec": row.start_sec, "impact_sec": row.end_sec, "event_end_sec": row.end_sec + 1.0,
                "label_confidence": 0.5, "fall_subtype": "unknown", "review_status": "needs_review"})
    out = Path(data["processed_dir"]); out.mkdir(parents=True, exist_ok=True)
    (out / "pair_report.json").write_text(json.dumps({"matched_count": len(pairs), "pairs": pairs, "raw_without_label": raw_only,
        "label_without_raw": label_only, "invalid_labels": invalid}, ensure_ascii=False, indent=2), encoding="utf-8")
    if intervals_all: pd.concat(intervals_all).to_csv(out / "action_intervals.csv", index=False, encoding="utf-8-sig")
    split_path = Path("data/splits/split_manifest.csv"); split_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).to_csv(split_path, index=False, encoding="utf-8-sig")
    event_path = Path(data["events_path"]); event_path.parent.mkdir(parents=True, exist_ok=True)
    draft_path = event_path.with_name("action_events_draft.csv")
    pd.DataFrame(draft_events).to_csv(draft_path, index=False, encoding="utf-8-sig")
    if not event_path.exists():
        pd.DataFrame(draft_events).to_csv(event_path, index=False, encoding="utf-8-sig")
        print(f"Created {event_path}; review onset/impact and set review_status=reviewed")
    print(f"Matched: {len(pairs)} | invalid: {len(invalid)} | draft events: {len(draft_events)}")
    if manifest: print(pd.DataFrame(manifest).groupby(["split", "subject_id"]).size())


if __name__ == "__main__":
    main()
