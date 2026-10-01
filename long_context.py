"""Dense long-context fall decision (every 0.5 s, using the last 30 s).

Why: a fall and walking/getting up share the same motion peak; they differ in what comes
after (quiet 1.5-5 s later) and before (walking, or lying still after an earlier fall)
(docs/CSI_낙상패턴_분석보고서.docx). A 3 s window sees neither. Every input is relative
to the recording itself (quiet-level motion, self-referenced spectrum change), so the
room/date spectrum fingerprint is not available to the model.

A decision at time t answers: "did a fall happen about 2-6 s ago?"
"""
from __future__ import annotations

import numpy as np

from context_verifier import relative_motion
from csi_dataset import stream_shape_dims

BIN_SEC = 1.0
RECENT_BINS = 10          # last 10 s in 1 s bins
HISTORY_SEC = (30.0, 10.0)  # older context window [t-30, t-10]
LABEL_AFTER_IMPACT = (2.0, 6.0)


def recording_signals(raw_features: np.ndarray, data_cfg: dict, quiet_window_sec: float = 60.0) -> dict:
    dims = stream_shape_dims(data_cfg)
    n_sub = int(data_cfg.get("subcarriers_per_rx", 52))
    motion = relative_motion(raw_features, data_cfg, quiet_window_sec)
    return {"motion": motion, "shape": raw_features[:, :dims].reshape(len(raw_features), -1, n_sub)}


def _window(values, frame_times, lo, hi):
    sel = (frame_times > lo) & (frame_times <= hi)
    return values[sel]


def features_at(t: float, frame_times: np.ndarray, signals: dict, stage1_times: np.ndarray,
                stage1_probs: np.ndarray) -> list[float]:
    motion, shape = signals["motion"], signals["shape"]
    feats = []
    for b in range(RECENT_BINS):  # oldest -> newest
        seg = _window(motion, frame_times, t - (RECENT_BINS - b) * BIN_SEC, t - (RECENT_BINS - b - 1) * BIN_SEC)
        feats += [float(seg.mean()) if len(seg) else np.nan, float(seg.max()) if len(seg) else np.nan]
    recent = _window(motion, frame_times, t - 7.0, t - 2.0)
    after = _window(motion, frame_times, t - 2.0, t)
    before = _window(motion, frame_times, t - 10.0, t - 7.0)
    peak = float(recent.max()) if len(recent) else np.nan
    feats += [float(after.mean()) / peak if len(after) and peak == peak else np.nan,          # post_drop
              peak / float(before.mean()) if len(before) and peak == peak else np.nan]        # sharpness
    history = _window(motion, frame_times, t - HISTORY_SEC[0], t - HISTORY_SEC[1])
    if len(history):
        feats += [float(np.mean(history > 1.5)), float(history.max()), float(np.mean(history < 1.2)),
                  float(np.percentile(history, 90))]
    else:
        feats += [np.nan] * 4
    # Time since the last strong burst before the recent window (capped at 30 s).
    older = (frame_times <= t - 7.0) & (frame_times > t - 37.0) & (motion > 2.0)
    feats.append(float(t - 7.0 - frame_times[older].max()) if older.any() else 30.0)
    # Self-referenced spectrum change: now vs 8-10 s ago, and stability now.
    now = (frame_times > t - 2.0) & (frame_times <= t)
    then = (frame_times > t - 10.0) & (frame_times <= t - 8.0)
    if now.any() and then.any():
        shift = np.linalg.norm(shape[now].mean(0) - shape[then].mean(0), axis=-1)
        feats += [float(shift.mean()), float(shift.max()), float(shape[now].std(0).mean())]
    else:
        feats += [np.nan] * 3
    near = (stage1_times > t - 7.0) & (stage1_times <= t - 2.0)
    feats += [float(stage1_probs[near].max()) if near.any() else np.nan,
              float(stage1_probs[stage1_times <= t][-1]) if (stage1_times <= t).any() else np.nan]
    return feats


FEATURE_NAMES = ([f"bin{b}_{s}" for b in range(RECENT_BINS) for s in ("mean", "max")]
                 + ["post_drop", "sharpness", "hist_active", "hist_max", "hist_quiet", "hist_p90",
                    "since_burst", "shift_mean", "shift_max", "now_spread", "stage1_recent_max", "stage1_now"])


def recording_rows(frame_times, signals, stage1_times, stage1_probs, stride_sec: float = 0.5,
                   start_sec: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Decision times every stride_sec and their feature matrix."""
    times = np.arange(max(start_sec, frame_times[0] + stride_sec), frame_times[-1] + 1e-9, stride_sec)
    x = np.array([features_at(t, frame_times, signals, stage1_times, stage1_probs) for t in times], dtype=np.float32)
    return times, x.reshape(len(times), len(FEATURE_NAMES))


def labels_for(times: np.ndarray, events) -> np.ndarray:
    """1: 2-6 s after an impact; 0: outside every fall's alarm-match window; -1: ignore."""
    y = np.zeros(len(times), dtype=np.int64)
    for e in events.itertuples(index=False):
        y[(times >= e.onset_sec - 1.0) & (times <= e.impact_sec + 8.0) & (y == 0)] = -1
    for e in events.itertuples(index=False):
        y[(times >= e.impact_sec + LABEL_AFTER_IMPACT[0]) & (times <= e.impact_sec + LABEL_AFTER_IMPACT[1])] = 1
    return y


def alarms_from_scores(times: np.ndarray, scores: np.ndarray, threshold: float, confirm: int = 2,
                       window: int = 3, cooldown: float = 10.0) -> list[float]:
    """Alarm when `confirm` of the last `window` decisions reach the threshold; then cooldown."""
    alarms, last, recent = [], -np.inf, []
    for t, s in zip(times, scores):
        recent = (recent + [s >= threshold])[-window:]
        if sum(recent) >= confirm and t - last >= cooldown:
            alarms.append(float(t)); last = t
    return alarms


STOP_AFTER = {"walking", "getting_up", "transition"}


def stop_weights(times: np.ndarray, intervals, weight: float, span_sec: float = 8.0) -> np.ndarray:
    """Up-weight decisions 0-span_sec after walking/getting up ends in standing: the
    'move, then stand still' moments that look like a fall and are rare in training."""
    w = np.ones(len(times), dtype=np.float64)
    if weight == 1.0:
        return w
    rows = intervals.sort_values("start_sec")
    labels = rows["label"].astype(str).str.strip().str.lower().tolist()
    for (prev, cur), start in zip(zip(labels, labels[1:]), rows["start_sec"].tolist()[1:]):
        if prev in STOP_AFTER and cur == "standing":
            w[(times >= start) & (times <= start + span_sec)] = weight
    return w
