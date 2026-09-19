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
