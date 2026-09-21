"""Compare Pose-input variants on the carewaveAI (Round4 OOF-b) skeleton under one fixed protocol."""
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
PY = sys.executable
CACHE = "data/pose_predictions/v2_oofb"
OUT = Path("outputs/pose_variants"); OUT.mkdir(parents=True, exist_ok=True)
VARIANTS = [
    ("base_3s", 0.0, "full", 3.0),
    ("ema03_3s", 0.3, "full", 3.0),
    ("ema03_2s", 0.3, "full", 2.0),
    ("ema03_1.5s", 0.3, "full", 1.5),
    ("ema015_2s", 0.15, "full", 2.0),
    ("ema03_core12_2s", 0.3, "core12", 2.0),
]
only = sys.argv[1:]


def stats(X):
    return np.concatenate([X.mean(1), X.std(1), np.ptp(X, axis=1)], axis=1)


def cv(S, y, g, kind):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(S, y, g):
        if kind == "GBM":
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=80, class_weight="balanced", random_state=0)
        else:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=300))
        p[te] = m.fit(S[tr], y[tr]).predict_proba(S[te])[:, 1]
    return p


for name, ema, mode, win in VARIANTS:
    if only and name not in only:
        continue
    t0 = time.time()
    fdir = f"data/processed/pose_features/v2_oofb_{name}"
    args = [PY, "features/extract_pose_motion_features.py", "--cache", CACHE, "--mode", mode, "--output", fdir]
    if ema > 0:
        args += ["--ema", str(ema)]
    subprocess.run(args, check=True, capture_output=True)
    tag = f"pv_{name}"
    subprocess.run([PY, "data/build_pose_windows.py", "--variant", "v2_oofb", "--feature-mode", mode, "--feature-dir", fdir,
                    "--window-seconds", str(win), "--split-mode", "stratified", "--post-fall-negatives", "--tag", tag],
                   check=True, capture_output=True)
    d = Path("data/processed/pose_windows") / tag
    parts = [np.load(d / f"{s}_3s.npz", allow_pickle=True) for s in ("train", "validation")]
    X = np.concatenate([q["X_pose"] for q in parts]); y = np.concatenate([q["y"] for q in parts])
    g = np.concatenate([q["sample_id"] for q in parts]); sub = np.concatenate([q["subtype"] for q in parts])
    S = stats(X)
    for kind in ("LR", "GBM"):
        p = cv(S, y, g, kind)
        np.savez_compressed(OUT / f"pred_{name}_{kind}.npz", p=p, y=y, g=g, sub=sub)
        row = {"variant": name, "ema_alpha": ema, "features": mode, "window_sec": win, "classifier": kind,
               "windows": len(y), "fall_windows": int(y.sum()), "AUROC": roc_auc_score(y, p),
               "AUPRC": average_precision_score(y, p), "chance_AUPRC": float(y.mean()),
               "mean_p_fall": float(p[y == 1].mean()), "mean_p_nonfall": float(p[y == 0].mean()), "minutes": (time.time() - t0) / 60}
        pd.DataFrame([row]).to_csv(OUT / "results.csv", mode="a", header=not (OUT / "results.csv").exists(), index=False)
        print(f"{name:18s} {kind:4s} AUROC={row['AUROC']:.3f} AUPRC={row['AUPRC']:.3f} (chance {row['chance_AUPRC']:.3f}) [{row['minutes']:.1f} min]", flush=True)
