"""Probability smoothing and event-level fall state machine."""

from __future__ import annotations

from collections import deque

import numpy as np


class ActionStateMachine:
    def __init__(self, class_names: list[str], cfg: dict) -> None:
        self.class_names = class_names
        self.ema_alpha = float(cfg["ema_alpha"])
        self.unknown_threshold = float(cfg["unknown_threshold"])
        self.fall_threshold = float(cfg["fall_threshold"])
        self.confirm_count = int(cfg["confirm_count"])
        self.history = deque(maxlen=int(cfg["confirm_window"]))
        self.cooldown = float(cfg["duplicate_suppression_seconds"])
        self.smoothed: np.ndarray | None = None
        self.state = "UPRIGHT"
        self.last_event_time = -float("inf")
        self.event_index = 0

    def update(self, time_sec: float, probabilities: np.ndarray) -> dict:
        probabilities = np.asarray(probabilities, dtype=float)
        self.smoothed = probabilities if self.smoothed is None else (
            self.ema_alpha * probabilities + (1.0 - self.ema_alpha) * self.smoothed
        )
        raw = self.class_names[int(probabilities.argmax())]
        confidence = float(self.smoothed.max())
        smooth = "unknown" if confidence < self.unknown_threshold else self.class_names[int(self.smoothed.argmax())]
        fall_idx = self.class_names.index("falling")
        candidate = float(self.smoothed[fall_idx]) >= self.fall_threshold
        self.history.append(candidate)
        confirmed = sum(self.history) >= self.confirm_count
        event_id = ""
        if self.state == "FALL_CONFIRMED" and candidate:
            self.state = "FALL_CONFIRMED"
        elif confirmed and time_sec - self.last_event_time >= self.cooldown:
            self.state = "FALL_CONFIRMED"
            self.event_index += 1
            event_id = f"fall_{self.event_index:04d}"
            self.last_event_time = time_sec
        elif candidate:
            self.state = "FALL_CANDIDATE"
        elif smooth == "lying":
            self.state = "POST_FALL" if self.last_event_time > -float("inf") else "LYING"
        elif smooth == "standing":
            self.state = "UPRIGHT"
        return {
            "pred_raw": raw,
            "pred_smoothed": smooth,
            "confidence": confidence,
            "state": self.state,
            "event_id": event_id,
            **{f"p_{name}": float(self.smoothed[i]) for i, name in enumerate(self.class_names)},
        }


def confirmed_candidates(probabilities, alpha: float, threshold: float, count: int, window: int) -> np.ndarray:
    """EMA smoothing followed by count-of-window confirmation (causal)."""
    recent: deque[int] = deque(maxlen=window)
    ema, out = None, np.zeros(len(probabilities), dtype=bool)
    for i, probability in enumerate(probabilities):
        ema = float(probability) if ema is None else alpha * float(probability) + (1.0 - alpha) * ema
        recent.append(int(ema >= threshold))
        out[i] = len(recent) >= window and sum(recent) >= count
    return out


def detect_fall_alarms(times, probabilities, frame_times, frame_motion, cfg: dict) -> list[dict]:
    """Turn per-window fall probabilities into alarm events.

    mode "confirm": alarm on the rising edge of the confirmed candidate.
    mode "fall_then_still": a candidate episode raises an alarm only if CSI motion drops
    to a lying-still level while the episode is short. Walking produces long candidate
    episodes and keeps moving, so it is discarded; a real fall is brief and ends in
    stillness. Every decision uses data up to times[i] only.
    """
    times = np.asarray(times, dtype=float)
    candidate = confirmed_candidates(probabilities, float(cfg.get("ema_alpha", 0.6)), float(cfg["fall_threshold"]),
                                     int(cfg.get("confirm_count", 3)), int(cfg.get("confirm_window", 5)))
    mode = str(cfg.get("mode", "confirm"))
    cooldown = float(cfg.get("cooldown_sec", 10.0))
    alarms, last_alarm = [], -np.inf
    start = last = None
    for i, t in enumerate(times):
        if mode == "confirm":
            if candidate[i] and (i == 0 or not candidate[i - 1]) and t - last_alarm >= cooldown:
                alarms.append({"time_sec": float(t), "candidate_sec": float(t)}); last_alarm = t
            continue
        if candidate[i]:
            if start is None or t - last > float(cfg.get("episode_gap_sec", 1.0)):
                start = t
            last = t
        if start is None:
            continue
        if t - last > float(cfg.get("still_max_wait_sec", 4.0)):
            start = last = None
            continue
        if last - start > float(cfg.get("max_episode_sec", 6.0)) or t - last_alarm < cooldown:
            continue
        quiet = (frame_times > t - float(cfg.get("still_sec", 2.0))) & (frame_times <= t)
        if t > start and quiet.any() and float(np.mean(frame_motion[quiet])) <= float(cfg["still_motion_threshold"]):
            alarms.append({"time_sec": float(t), "candidate_sec": float(start)})
            last_alarm = t
    return alarms
