"""Stage-2 context verification of fall candidates.

Follows docs/CSI_낙상패턴_분석보고서.docx (sections 4-7): a fall and walking/getting up
share the same motion peak; they differ in how quiet it gets 1.5-5 s after the peak
(post_drop), how sharply the peak rises over the preceding motion (sharpness), and
whether the stream spectra stay changed relative to their own pre-event shape.

Pipeline per recording (causal; each decision uses frames up to its decision time):
  1. Stage-1 window probabilities -> confirmed candidate episodes (same smoothing as
     postprocess_state_machine.confirmed_candidates).
  2. For each episode, t0 = motion peak; decide at t0 + post_end_sec.
  3. Context features around t0 -> verifier probability.
  4. Alarm if probability >= threshold, with cooldown and an optional floor state:
     after an alarm the next candidate is treated as getting up and never alarms.
"""
from __future__ import annotations

import numpy as np

from csi_dataset import stream_shape_dims
from postprocess_state_machine import FloorState, active_segments, confirmed_candidates

FEATURES = ["post_drop", "sharpness", "log_peak", "log_pre", "log_post", "pre_activity", "post_activity",
            "posture_shift_mean", "posture_shift_max", "post_shape_spread", "prob_max", "episode_sec"]


def relative_motion(raw_features: np.ndarray, data_cfg: dict, quiet_window_sec: float = 60.0) -> np.ndarray:
    """Per-frame packet motion (mean of the stream motion channels), 0.5 s trailing mean,
    divided by a trailing 10th-percentile quiet level so 1.0 = this recording's quiet."""
    fps = int(data_cfg["target_fps"])
    motion = np.nanmean(raw_features[:, stream_shape_dims(data_cfg):], axis=1)
    kernel = max(1, fps // 2)
    padded = np.r_[np.full(kernel - 1, motion[0]), motion]
    smooth = np.convolve(padded, np.ones(kernel) / kernel, mode="valid")
    span = int(quiet_window_sec * fps)
    quiet = np.array([np.percentile(smooth[max(0, i - span):i + 1], 10) for i in range(len(smooth))])
    return smooth / np.maximum(quiet, 1e-6)


def _mean(values: np.ndarray, times: np.ndarray, lo: float, hi: float) -> float:
    sel = (times >= lo) & (times < hi)
    return float(np.mean(values[sel])) if sel.any() else float("nan")


def candidates(decision_times, probabilities, frame_times, motion, raw_features, data_cfg, post) -> list[dict]:
    """Candidate events with context features; drops those whose decision time is past the recording end."""
    window_sec = float(data_cfg["window_seconds"])
    post_end = float(post.get("post_end_sec", 5.0))
    flags = confirmed_candidates(probabilities, float(post.get("ema_alpha", 0.6)), float(post["fall_threshold"]),
                                 int(post.get("confirm_count", 2)), int(post.get("confirm_window", 3)))
    episodes, start, last, peak_prob = [], None, None, 0.0
    for t, flag, p in zip(decision_times, flags, probabilities):
        if flag:
            if start is None or t - last > float(post.get("episode_gap_sec", 1.0)):
                if start is not None:
                    episodes.append((start, last, peak_prob))
                start, peak_prob = t, 0.0
            last, peak_prob = t, max(peak_prob, float(p))
    if start is not None:
        episodes.append((start, last, peak_prob))

    dims = stream_shape_dims(data_cfg)
    n_sub = int(data_cfg.get("subcarriers_per_rx", 52))
    shape = raw_features[:, :dims].reshape(len(raw_features), -1, n_sub)
    end_time = float(frame_times[-1])
    out, used_t0 = [], []
    for start, last, prob_max in episodes:
        sel = (frame_times >= start - window_sec) & (frame_times <= last)
        if not sel.any():
            continue
        idx = np.flatnonzero(sel)
        t0 = float(frame_times[idx[np.argmax(motion[idx])]])
        decision = t0 + post_end
        if decision > end_time or any(abs(t0 - u) < 1.0 for u in used_t0):
            continue
        used_t0.append(t0)
        peak = float(np.max(motion[(frame_times >= t0 - 0.5) & (frame_times <= t0 + 0.5)]))
        pre = _mean(motion, frame_times, t0 - 4.0, t0 - 1.5)
        after = _mean(motion, frame_times, t0 + 1.5, decision)
        before_shape = (frame_times >= t0 - 4.0) & (frame_times < t0 - 1.5)
        after_shape = (frame_times >= t0 + 1.5) & (frame_times < decision)
        if before_shape.any() and after_shape.any():
            shift = np.linalg.norm(shape[after_shape].mean(0) - shape[before_shape].mean(0), axis=-1)
            spread = float(shape[after_shape].std(axis=0).mean())
        else:
            shift, spread = np.full(shape.shape[1], np.nan), float("nan")
        pre_sel = (frame_times >= t0 - 4.0) & (frame_times < t0 - 1.5)
        post_sel = after_shape
        out.append({
            "t0": t0, "decision_sec": decision, "candidate_sec": float(start),
            "post_drop": after / peak, "sharpness": peak / pre if pre == pre else float("nan"),
            "log_peak": float(np.log(peak)), "log_pre": float(np.log(pre)) if pre == pre else float("nan"),
            "log_post": float(np.log(after)),
            "pre_activity": float(np.mean(motion[pre_sel] > 1.5)) if pre_sel.any() else float("nan"),
            "post_activity": float(np.mean(motion[post_sel] > 1.5)) if post_sel.any() else float("nan"),
            "posture_shift_mean": float(np.nanmean(shift)), "posture_shift_max": float(np.nanmax(shift)) if np.isfinite(shift).any() else float("nan"),
            "post_shape_spread": spread, "prob_max": prob_max, "episode_sec": float(last - start),
        })
    return out


def feature_matrix(rows: list[dict], medians: np.ndarray | None = None) -> np.ndarray:
    """Missing context (too close to the recording start) is filled with training medians."""
    x = np.array([[row[name] for name in FEATURES] for row in rows], dtype=np.float64).reshape(-1, len(FEATURES))
    if medians is not None:
        x = np.where(np.isfinite(x), x, medians)
    return x


def select_alarms(rows: list[dict], scores: np.ndarray, post: dict, floor_segments=None) -> list[dict]:
    """Threshold verifier scores in decision-time order, with cooldown and floor state
    (postprocess_state_machine.FloorState; the candidate starts at its motion peak t0)."""
    threshold = float(post["verifier_threshold"])
    cooldown = float(post.get("cooldown_sec", 10.0))
    floor = FloorState(floor_segments or [], post) if post.get("floor_state", False) else None
    alarms, last_alarm = [], -np.inf
    for row, score in sorted(zip(rows, scores), key=lambda item: item[0]["decision_sec"]):
        t = row["decision_sec"]
        if score < threshold or t - last_alarm < cooldown:
            continue
        if floor is not None and floor.suppresses(row["t0"], t):
            continue
        alarms.append({"time_sec": float(t), "candidate_sec": row["candidate_sec"], "verifier_score": float(score)})
        last_alarm = t
        if floor is not None:
            floor.enter(t)
    return alarms


def floor_segments(frame_times, motion, post) -> list[tuple[float, float]]:
    return active_segments(frame_times, motion, float(post.get("floor_motion_level", 1.5)),
                           float(post.get("floor_min_sec", 1.5)), float(post.get("floor_gap_sec", 1.0)))


def rule_scores(rows: list[dict], post: dict) -> np.ndarray:
    """Report section 7 rule: motion 1.5-5 s after the peak falls to at most half of the peak."""
    return np.array([1.0 if row["post_drop"] <= float(post.get("post_drop_max", 0.5)) else 0.0 for row in rows])


def verifier_scores(rows: list[dict], verifier: dict) -> np.ndarray:
    if not rows:
        return np.zeros(0)
    x = feature_matrix(rows, np.asarray(verifier["medians"]))
    return verifier["model"].predict_proba(x)[:, 1]


def detect_verified_alarms(decision_times, probabilities, frame_times, raw_features, data_cfg, post, verifier=None):
    motion = relative_motion(raw_features, data_cfg, float(post.get("quiet_window_sec", 60.0)))
    rows = candidates(decision_times, probabilities, frame_times, motion, raw_features, data_cfg, post)
    scores = rule_scores(rows, post) if verifier is None else verifier_scores(rows, verifier)
    return select_alarms(rows, scores, dict(post, verifier_threshold=0.5) if verifier is None else post,
                         floor_segments(frame_times, motion, post))
