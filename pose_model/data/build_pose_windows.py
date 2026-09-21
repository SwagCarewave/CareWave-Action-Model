"""Binary (standing 0 / fall 1) labels and 3 s pose windows from cached first-stage poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml



HARD_NEGATIVE_ACTIONS = {
    "walk_normal", "walk_large_motion", "sit_normal", "sit_move_normal", "sit_to_stand_normal", "chair_sit_variation",
    "pick_up_normal", "pick_up_variation", "bending_normal", "free_activity_normal", "free_activity", "arm_move_normal",
}
HARD_NEGATIVE_LABELS = {"hard_negative", "slow_lying_down"}
POST_FALL_LABELS = {"lying", "getting_up", "transition"}


def load_intervals(labels_dir: Path) -> dict[str, list[tuple[float, float, str]]]:
    out: dict[str, list[tuple[float, float, str]]] = {}
    for path in sorted(labels_dir.glob("*/*_labels.csv")):
        df = pd.read_csv(path, encoding="utf-8-sig")
        for name, grp in df.groupby("video_name"):
            sample_id = str(name).removesuffix(".mp4")
            out[sample_id] = [(float(r.start_sec), float(r.end_sec), str(r.label).strip()) for r in grp.itertuples()]
    return out


def split_recordings(index: pd.DataFrame, labeled: set[str], cfg: dict, seed: int) -> dict[str, str]:
    """Recording-level split. Whole recordings (hence whole fall events) stay in one split."""
    rng = np.random.default_rng(seed)
    df = index[index.sample_id.isin(labeled)].copy()
    df["group"] = df.action.str.replace(r"_normal|_variation", "", regex=True)
    assign: dict[str, str] = {}

    def draw(pool: pd.DataFrame, fraction: float, name: str, rest: str) -> None:
        for _, grp in pool.groupby("group"):
            ids = sorted(grp.sample_id)
            rng.shuffle(ids)
            k = max(1, round(len(ids) * fraction)) if len(ids) > 1 else 0
            for i, sid in enumerate(ids):
                assign[sid] = name if i < k else rest

    if cfg["mode"] == "lineage":
        for sid in df[df.first_stage_split == "validation"].sample_id:
            assign[sid] = "test"
        draw(df[df.first_stage_split == "train"], cfg["val_fraction"], "validation", "train")
    elif cfg["mode"] == "lineage_oos":
        for sid in df[df.first_stage_split == "train"].sample_id:
            assign[sid] = "train"
        draw(df[df.first_stage_split.isin(["validation", "val", "test6"])], 0.5, "validation", "test")
    elif cfg["mode"] == "stratified":
        draw(df, cfg["test_fraction"], "test", "rest")
        rest = df[df.sample_id.map(assign) == "rest"]
        draw(rest, cfg["val_fraction"] / (1 - cfg["test_fraction"]), "validation", "train")
    else:
        raise ValueError(cfg["mode"])
    return assign


def build_recording(sample_id: str, feat: dict, intervals: list, wcfg: dict, action: str = "") -> tuple[list[dict], list[dict], list[dict]]:
    fps, win = int(wcfg["fps"]), int(round(wcfg["window_seconds"] * wcfg["fps"]))
    stride = int(round(wcfg["stride_seconds"] * fps))
    t, valid = feat["time_sec"].astype(np.float64), feat["valid_mask"].astype(bool)
    n = len(t)
    events = []
    for start, end, label in intervals:
        if label == "falling":
            events.append((f"{sample_id}_fall_{len(events) + 1:02d}", start, end))
    zone = [(eid, on - wcfg["fall_margin_pre_sec"], im + wcfg["fall_margin_post_sec"]) for eid, on, im in events]
    standing = np.zeros(n, bool)
    for start, end, label in intervals:
        if label == "standing":
            standing |= (t >= start) & (t < end)
    in_zone = np.zeros(n, bool)
    for _, lo, hi in zone:
        in_zone |= (t >= lo) & (t < hi)
    standing &= ~in_zone
    hardneg = np.zeros(n, bool)
    if wcfg.get("hard_negatives", True):
        for start, end, label in intervals:
            if label in HARD_NEGATIVE_LABELS or (wcfg.get("post_fall_negatives") and label in POST_FALL_LABELS):
                hardneg |= (t >= start) & (t < end)
    hardneg &= ~in_zone
    t_end_labeled = max(end for _, end, _ in intervals)

    rows, meta, excl = [], [], []
    for s in range(0, n - win + 1, stride):
        sl = slice(s, s + win)
        tw = t[sl]
        start_sec, end_sec = float(tw[0]), float(tw[-1] + 1.0 / fps)
        base = {"sample_id": sample_id, "start_sec": start_sec, "end_sec": end_sec}
        if np.diff(tw).max() > 0.15 + 1e-6:
            excl.append({**base, "reason": "timestamp_gap"})
            continue
        if end_sec > t_end_labeled + 1e-6:
            excl.append({**base, "reason": "beyond_labels"})
            continue
        pose_valid = float(valid[sl].mean())
        if pose_valid < wcfg["min_pose_valid_ratio"]:
            excl.append({**base, "reason": "pose_invalid", "pose_valid_ratio": pose_valid})
            continue

        center = 0.5 * (tw[0] + tw[-1])
        best_eid, best_overlap, best_center_ok = "", 0.0, False
        for (eid, lo, hi), (_, on, im) in zip(zone, events):
            overlap = float(np.mean((tw >= lo) & (tw < hi)))
            if overlap > best_overlap:
                best_eid, best_overlap = eid, overlap
                best_center_ok = on - wcfg["positive_center_pre_sec"] <= center <= im + wcfg["positive_center_post_sec"]
        if best_overlap >= wcfg["positive_overlap"] and best_center_ok:
            y, weight, eid, subtype = 1, (wcfg["soft_boundary_weight"] if best_overlap < 0.5 else 1.0), best_eid, "fall"
        elif best_overlap == 0.0 and standing[sl].mean() >= wcfg["standing_ratio"]:
            y, weight, eid, subtype = 0, 1.0, "", "standing"
        elif best_overlap == 0.0 and hardneg[sl].mean() >= wcfg["standing_ratio"]:
            neg_labels = HARD_NEGATIVE_LABELS | (POST_FALL_LABELS if wcfg.get("post_fall_negatives") else set())
            labels_here = {lab for st, en, lab in intervals if lab in neg_labels and st < tw[-1] and en > tw[0]}
            y, weight, eid = 0, 1.0, ""
            subtype = action if "hard_negative" in labels_here else sorted(labels_here)[0]
        else:
            excl.append({**base, "reason": "ambiguous_label", "pose_valid_ratio": pose_valid})
            continue
        rows.append({"s": s, "y": y, "w": weight})
        meta.append({**base, "event_id": eid, "subtype": subtype, "label_weight": weight, "pose_valid_ratio": pose_valid, "y": y})
    return rows, meta, excl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/pose_branch.yaml"))
    ap.add_argument("--variant", default=None)
    ap.add_argument("--feature-mode", default=None)
    ap.add_argument("--split-mode", default=None)
    ap.add_argument("--no-hard-negatives", action="store_true", help="labeled standing/fall only (lab binary)")
    ap.add_argument("--post-fall-negatives", action="store_true", help="lying/getting_up/transition intervals become y=0")
    ap.add_argument("--feature-dir", type=Path, default=None, help="override data/processed/pose_features/<variant>_<mode>")
    ap.add_argument("--window-seconds", type=float, default=None)
    ap.add_argument("--tag", default=None, help="output subfolder name (default variant_featuremode_splitmode)")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    variant = args.variant or cfg["first_stage"]["variant"]
    fmode = args.feature_mode or cfg["features"]["mode"]
    cfg["split"]["mode"] = args.split_mode or cfg["split"]["mode"]
    if args.post_fall_negatives:
        cfg["windows"]["post_fall_negatives"] = True
    if args.no_hard_negatives:
        cfg["windows"]["hard_negatives"] = False
    tag = args.tag or f"{variant}_{fmode}_{cfg['split']['mode']}" + ("" if cfg["windows"].get("hard_negatives", True) else "_nohn") + ("_pfn" if cfg["windows"].get("post_fall_negatives") else "")

    if args.window_seconds:
        cfg["windows"]["window_seconds"] = args.window_seconds
    feat_dir = args.feature_dir or Path(cfg["paths"]["feature_dir"]) / f"{variant}_{fmode}"
    index = pd.read_csv(Path(cfg["paths"]["pose_cache_dir"]) / variant / "index.csv", encoding="utf-8-sig")
    intervals = load_intervals(Path(cfg["paths"]["labels_dir"]))
    no_cache = sorted(set(intervals) - set(index.sample_id))
    action_of = dict(zip(index.sample_id, index.action))
    labeled = set(intervals) & set(index.sample_id)
    if cfg["windows"].get("hard_negatives", True):
        labeled |= {sid for sid, a in action_of.items() if a in HARD_NEGATIVE_ACTIONS and sid not in intervals}
    assign = split_recordings(index, labeled, cfg["split"], int(cfg["seed"]))
    lineage = dict(zip(index.sample_id, index.first_stage_split))
    person = dict(zip(index.sample_id, index.person))
    hashes = {"source_checkpoint": str(index.source_checkpoint.iloc[0]), "scaler_id": str(index.scaler_id.iloc[0]),
              "variant": variant, "feature_mode": fmode, "split_mode": cfg["split"]["mode"]}

    buckets: dict[str, dict[str, list]] = {s: {"pose": [], "quality": [], "meta": []} for s in ("train", "validation", "test")}
    excl_all: list[dict] = []
    names: dict[str, np.ndarray] = {}
    for sid in sorted(labeled):
        with np.load(feat_dir / f"{sid}.npz", allow_pickle=True) as z:
            feat = {k: z[k] for k in z.files}
        names = {"feature_names": feat["feature_names"], "quality_names": feat["quality_names"]}
        recording_intervals = intervals.get(sid) or [(0.0, float(feat["time_sec"][-1]) + 0.1, "hard_negative")]
        rows, meta, excl = build_recording(sid, feat, recording_intervals, cfg["windows"], action_of[sid])
        win = int(round(cfg["windows"]["window_seconds"] * cfg["windows"]["fps"]))
        b = buckets[assign[sid]]
        for r, m in zip(rows, meta):
            b["pose"].append(feat["features"][r["s"]:r["s"] + win])
            b["quality"].append(feat["quality"][r["s"]:r["s"] + win])
            b["meta"].append({**m, "subject_id": person[sid], "session_id": sid, "room_id": "unknown",
                              "first_stage_split": lineage[sid], "split": assign[sid]})
        excl_all += [{**e, "split": assign[sid]} for e in excl]

    out_dir = Path(cfg["paths"]["window_dir"]) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"tag": tag, "hashes": hashes, "recordings_without_pose_cache": no_cache, "splits": {}}
    for split, b in buckets.items():
        if not b["meta"]:
            report["splits"][split] = {"windows": 0}
            continue
        m = pd.DataFrame(b["meta"])
        np.savez_compressed(
            out_dir / f"{split}_3s.npz",
            X_pose=np.stack(b["pose"]).astype(np.float32), X_quality=np.stack(b["quality"]).astype(np.float32),
            y=m.y.to_numpy(np.int64), subtype=m.subtype.to_numpy(str), sample_id=m.sample_id.to_numpy(str), event_id=m.event_id.to_numpy(str),
            start_sec=m.start_sec.to_numpy(np.float32), end_sec=m.end_sec.to_numpy(np.float32),
            subject_id=m.subject_id.to_numpy(str), session_id=m.session_id.to_numpy(str), room_id=m.room_id.to_numpy(str),
            label_weight=m.label_weight.to_numpy(np.float32), pose_valid_ratio=m.pose_valid_ratio.to_numpy(np.float32),
            first_stage_split=m.first_stage_split.to_numpy(str), source_hashes=json.dumps(hashes), **names,
        )
        m.to_csv(out_dir / f"{split}_meta.csv", index=False, encoding="utf-8-sig")
        report["splits"][split] = {
            "windows": len(m), "recordings": int(m.sample_id.nunique()), "standing": int((m.y == 0).sum()),
            "fall": int((m.y == 1).sum()), "fall_events": int(m[m.y == 1].event_id.nunique()), "by_subtype": m.groupby("subtype").size().to_dict(),
            "by_subject": m.groupby("subject_id").size().to_dict(),
            "by_lineage": m.groupby("first_stage_split").size().to_dict(),
        }

    excl_df = pd.DataFrame(excl_all)
    excl_df.to_csv(out_dir / "exclusion_log.csv", index=False, encoding="utf-8-sig")
    manifest = pd.DataFrame([{"sample_id": s, "split": sp, "subject_id": person[s], "first_stage_split": lineage[s]}
                             for s, sp in sorted(assign.items())])
    manifest.to_csv(out_dir / "split_manifest_binary.csv", index=False, encoding="utf-8-sig")
    report["exclusions"] = excl_df.groupby(["split", "reason"]).size().to_dict() if len(excl_df) else {}
    report["exclusions"] = {f"{k[0]}/{k[1]}": int(v) for k, v in report["exclusions"].items()}

    ids = {s: set(manifest[manifest.split == s].sample_id) for s in buckets}
    assert not (ids["train"] & ids["validation"]) and not (ids["train"] & ids["test"]) and not (ids["validation"] & ids["test"])
    for s, r in report["splits"].items():
        if r["windows"] and (r["standing"] == 0 or r["fall"] == 0):
            print(f"[WARN] split '{s}' lacks a class: {r}")
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
