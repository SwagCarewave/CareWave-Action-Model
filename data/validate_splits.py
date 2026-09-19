"""Check recording/session leakage across dataset splits."""

import argparse
from pathlib import Path

import pandas as pd

from csi_dataset import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/action_fusion.yaml"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/csi_stream/metadata.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/split_manifest.csv"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    split_mode = str(cfg["data"]["split"].get("mode", "session"))
    df = pd.read_csv(args.metadata)
    required = {"sample_id", "subject_id", "session_id", "room_id", "split", "label", "is_explicit_test"}
    if not required.issubset(df.columns):
        raise SystemExit(f"Missing columns: {sorted(required - set(df.columns))}")

    sample_leak = df.groupby("sample_id")["split"].nunique()
    session_leak = df.groupby("session_id")["split"].nunique()
    known_room = df[df["room_id"].astype(str) != "unknown"]
    room_leak = known_room.groupby("room_id")["split"].nunique()
    leak_series = [sample_leak, session_leak, room_leak]
    if split_mode == "subject":
        leak_series.append(df.groupby("subject_id")["split"].nunique())
    if any((series > 1).any() for series in leak_series):
        raise SystemExit("Split leakage detected")

    explicit_test = df["is_explicit_test"].astype(str).str.lower().isin({"true", "1"})
    if (df.loc[explicit_test, "split"] != "test").any():
        raise SystemExit("An explicit test-folder recording was assigned outside the test split")
    if ((~explicit_test) & (df["split"] == "test")).any():
        raise SystemExit("A non-test-folder recording was assigned to the test split")

    if args.manifest.exists():
        manifest = pd.read_csv(args.manifest)
        actual = df[["sample_id", "split"]].drop_duplicates()
        checked = actual.merge(
            manifest[["sample_id", "split"]], on="sample_id",
            suffixes=("_actual", "_manifest"), how="left",
        )
        if checked["split_manifest"].isna().any() or (checked["split_actual"] != checked["split_manifest"]).any():
            raise SystemExit("Metadata and split_manifest.csv do not match")

    if split_mode == "subject":
        print("No sample, subject, session or room leakage detected.")
    else:
        print("No sample, session or room leakage detected. Subject overlap is allowed by session split.")
    print(df.groupby(["split", "subject_id", "label"]).size())


if __name__ == "__main__":
    main()
