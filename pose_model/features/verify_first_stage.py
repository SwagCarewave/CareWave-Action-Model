"""Reproduce first-stage results with the ported models before caching anything."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.first_stage_pose import FirstStagePose

CKPT = {"v1": "trained_model_315/carewave_315_best.pt", "v2": "trained_model_315_v2/carewave_315_v2_best.pt"}
STORED_VAL_MPJPE = {"v1": 0.11449848639768655, "v2": 0.12220024390496308}


def mpjpe(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(pred.reshape(-1, 33, 2) - target.reshape(-1, 33, 2), axis=-1).mean())


def load_split(root: Path, split: str) -> dict[str, np.ndarray]:
    files = sorted((root / "sequences_315" / split).glob("*_sequences.npz"))
    parts = [np.load(f, allow_pickle=True) for f in files]
    return {"X": np.concatenate([p["X"] for p in parts]), "raw": np.concatenate([p["y_raw_pose"] for p in parts]),
            "ids": np.concatenate([np.repeat(str(p["sample_id"][0]), len(p["X"])) for p in parts])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True, help="round4_v1_final directory")
    args = ap.parse_args()
    root = args.data_root

    val = load_split(root, "validation")
    for variant in ("v1", "v2"):
        model = FirstStagePose(variant, root / CKPT[variant])
        pred = model.predict(val["X"])["raw_xy"]
        got, want = mpjpe(pred, val["raw"]), STORED_VAL_MPJPE[variant]
        print(f"[{variant}] val sequences={len(val['X'])} epoch={model.epoch} MPJPE={got:.6f} stored={want:.6f} diff={abs(got - want):.2e}")

    test = load_split(root, "test")
    model = FirstStagePose("v1", root / CKPT["v1"])
    pred = model.predict(test["X"])["raw_xy"].reshape(len(test["X"]), 66)
    ref = np.load(root / "final_test_evaluation_v1" / "test_predictions_v1.npz", allow_pickle=True)
    print(f"[v1] test sequences={len(pred)} max|pred-ref|={np.abs(pred - ref['predictions']).max():.2e} "
          f"MPJPE={mpjpe(pred, test['raw']):.6f} (reported 0.157276)")


if __name__ == "__main__":
    main()
