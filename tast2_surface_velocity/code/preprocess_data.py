"""Create processed metadata, leakage-aware splits, and dataset figures.

Run from the repository root:
    python3 task2_surface_velocity/code/preprocess_data.py
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", str(__import__("pathlib").Path(__file__).resolve().parents[1] / "results" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from task2_utils import DAMAGE_DESCRIPTIONS, build_metadata, create_grouped_splits, ensure_dirs, project_paths


def save_sample_visualization(metadata: pd.DataFrame, paths) -> None:
    sample = metadata.iloc[0]
    frame = Image.open(paths.root / sample.frame_path).convert("RGB")
    mask = Image.open(paths.root / sample.mask_path).convert("L")
    x1, y1, x2, y2 = int(sample.bbox_x1), int(sample.bbox_y1), int(sample.bbox_x2), int(sample.bbox_y2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(frame)
    axes[0].set_title("Frame")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask")
    crop = frame.crop((x1, y1, x2 + 1, y2 + 1))
    axes[2].imshow(crop)
    axes[2].set_title(f"Building crop: level {sample.damage_level}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(paths.figures / "sample_frame_mask_visualization.png", dpi=160)
    plt.close(fig)


def save_label_distribution(metadata: pd.DataFrame, paths) -> None:
    counts = metadata["damage_level"].value_counts().sort_index()
    labels = [f"{idx}: {DAMAGE_DESCRIPTIONS[idx].split()[0]}" for idx in counts.index]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, counts.values, color="#4C78A8")
    ax.set_xlabel("Damage level")
    ax.set_ylabel("Building instances")
    ax.set_title("Target distribution")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(paths.figures / "target_distribution.png", dpi=160)
    plt.close(fig)


def main() -> None:
    paths = ensure_dirs(project_paths())
    metadata = build_metadata(paths)
    splits = create_grouped_splits(metadata)
    metadata_path = paths.processed / "metadata.csv"
    splits_path = paths.processed / "splits.csv"
    metadata.to_csv(metadata_path, index=False)
    splits.to_csv(splits_path, index=False)

    merged = metadata.merge(splits[["sample_id", "split"]], on="sample_id", how="left")
    save_sample_visualization(merged, paths)
    save_label_distribution(merged, paths)

    print(f"Wrote {metadata_path.relative_to(paths.root)} with {len(metadata)} rows")
    print(f"Wrote {splits_path.relative_to(paths.root)}")
    print("Split counts:")
    print(merged["split"].value_counts().to_string())
    print("Damage counts by split:")
    print(pd.crosstab(merged["split"], merged["damage_level"]).to_string())


if __name__ == "__main__":
    main()
