"""Shared utilities for the Task 2 DoriaNET computer-vision pipeline.

The assignment prompt mentions surface velocity, but the provided raw data and
papers are DoriaNET building-damage annotations. These helpers therefore build
a defensible frame/mask-assisted damage-level prediction dataset.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset


DAMAGE_DESCRIPTIONS = {
    0: "No or very minor damage",
    1: "Minor damage",
    2: "Moderate damage",
    3: "Severe damage",
    4: "Destruction",
    5: "Destroyed or under construction",
}


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw: Path
    frames: Path
    jsons: Path
    masks: Path
    processed: Path
    results: Path
    figures: Path
    models: Path


def project_paths() -> ProjectPaths:
    """Return paths relative to the task2_surface_velocity project folder."""
    root = Path(__file__).resolve().parents[1]
    return ProjectPaths(
        root=root,
        raw=root / "data" / "raw",
        frames=root / "data" / "raw" / "FRAME",
        jsons=root / "data" / "raw" / "JSON",
        masks=root / "data" / "raw" / "MASK",
        processed=root / "data" / "processed",
        results=root / "results",
        figures=root / "figures",
        models=root / "models",
    )


def ensure_dirs(paths: ProjectPaths | None = None) -> ProjectPaths:
    paths = paths or project_paths()
    for path in [paths.processed, paths.results, paths.figures, paths.models]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def read_json_lenient(path: Path) -> dict:
    """Read the Labelbox-style JSON files, which contain JavaScript NaN values."""
    return json.loads(path.read_text())


def mask_bbox(mask_path: Path) -> tuple[int, int, int, int, float, float, float]:
    """Return bbox, area fraction, and centroid from a binary mask image."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")
    ys, xs = np.where(mask > 0)
    h, w = mask.shape[:2]
    if len(xs) == 0:
        return 0, 0, w - 1, h - 1, 0.0, 0.5, 0.5
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    area_fraction = float(len(xs) / (h * w))
    centroid_x = float(xs.mean() / w)
    centroid_y = float(ys.mean() / h)
    return x1, y1, x2, y2, area_fraction, centroid_x, centroid_y


def parse_lat_lon(value: str) -> tuple[float, float]:
    lat_text, lon_text = value.split(",", 1)
    return float(lat_text.strip()), float(lon_text.strip())


def build_metadata(paths: ProjectPaths | None = None) -> pd.DataFrame:
    """Parse FRAME/JSON/MASK records into one row per building object."""
    paths = ensure_dirs(paths)
    rows: list[dict] = []
    for json_path in sorted(paths.jsons.glob("*.json")):
        data = read_json_lenient(json_path)
        frame_rel = data["Frame_Name"]
        frame_path = paths.raw / frame_rel
        frame_name = Path(frame_rel).name
        frame_stem = Path(frame_name).stem
        video_id, frame_index = frame_stem.split("_", 1)
        for item_index, building in enumerate(data.get("Buildings", [])):
            # DoriaNET building schema from inspection:
            # [building_id, "lat, lon", mask_file, damage_level, stories, annotation_effort, reserved]
            building_id = str(building[0])
            lat, lon = parse_lat_lon(str(building[1]))
            mask_name = str(building[2])
            damage_level = int(building[3])
            stories = int(building[4]) if not pd.isna(building[4]) else -1
            annotation_effort = int(building[5]) if not pd.isna(building[5]) else -1
            mask_path = paths.masks / mask_name
            x1, y1, x2, y2, area_fraction, centroid_x, centroid_y = mask_bbox(mask_path)
            rows.append(
                {
                    "sample_id": f"{frame_stem}_{item_index:03d}",
                    "video_id": int(video_id),
                    "frame_index": int(frame_index),
                    "frame_stem": frame_stem,
                    "building_id": building_id,
                    "latitude": lat,
                    "longitude": lon,
                    "frame_path": str(frame_path.relative_to(paths.root)),
                    "mask_path": str(mask_path.relative_to(paths.root)),
                    "json_path": str(json_path.relative_to(paths.root)),
                    "mask_file": mask_name,
                    "damage_level": damage_level,
                    "target": damage_level,
                    "damage_description": DAMAGE_DESCRIPTIONS.get(damage_level, "Unknown"),
                    "stories": stories,
                    "annotation_effort": annotation_effort,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "bbox_width": x2 - x1 + 1,
                    "bbox_height": y2 - y1 + 1,
                    "mask_area_fraction": area_fraction,
                    "mask_centroid_x": centroid_x,
                    "mask_centroid_y": centroid_y,
                    "capture_date": data.get("Capture date", ""),
                    "region": data.get("Region", ""),
                    "source_video_url": data.get("Original video link", ""),
                }
            )
    return pd.DataFrame(rows)


def create_grouped_splits(
    metadata: pd.DataFrame,
    train_fraction: float = 0.60,
    val_fraction: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """Split by building_id so the same physical building stays in one split."""
    rng = random.Random(seed)
    building_ids = sorted(metadata["building_id"].unique())
    rng.shuffle(building_ids)
    n = len(building_ids)
    train_end = int(round(n * train_fraction))
    val_end = train_end + int(round(n * val_fraction))
    split_by_building = {}
    for building_id in building_ids[:train_end]:
        split_by_building[building_id] = "train"
    for building_id in building_ids[train_end:val_end]:
        split_by_building[building_id] = "val"
    for building_id in building_ids[val_end:]:
        split_by_building[building_id] = "test"
    splits = metadata[["sample_id", "building_id", "damage_level"]].copy()
    splits["split"] = splits["building_id"].map(split_by_building)
    return splits


def load_metadata_and_splits(paths: ProjectPaths | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = ensure_dirs(paths)
    metadata_path = paths.processed / "metadata.csv"
    splits_path = paths.processed / "splits.csv"
    if not metadata_path.exists() or not splits_path.exists():
        metadata = build_metadata(paths)
        metadata.to_csv(metadata_path, index=False)
        create_grouped_splits(metadata).to_csv(splits_path, index=False)
    metadata = pd.read_csv(metadata_path)
    splits = pd.read_csv(splits_path)
    return metadata.merge(splits[["sample_id", "split"]], on="sample_id", how="left"), splits


def crop_with_padding(image: Image.Image, bbox: tuple[int, int, int, int], padding: float = 0.15) -> Image.Image:
    x1, y1, x2, y2 = bbox
    w, h = image.size
    box_w = x2 - x1 + 1
    box_h = y2 - y1 + 1
    pad = int(max(box_w, box_h) * padding)
    return image.crop((max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)))


def pil_to_tensor(image: Image.Image, size: int, grayscale: bool = False) -> torch.Tensor:
    mode = "L" if grayscale else "RGB"
    image = image.convert(mode).resize((size, size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if grayscale:
        arr = arr[None, :, :]
    else:
        arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)


class DoriaNetDataset(Dataset):
    """One dataset class serving the three model variants."""

    def __init__(self, rows: pd.DataFrame, root: Path, image_size: int = 128, mode: str = "flownet", augment: bool = False):
        self.rows = rows.reset_index(drop=True)
        self.root = root
        self.image_size = image_size
        self.mode = mode
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        frame = Image.open(self.root / row.frame_path).convert("RGB")
        mask = Image.open(self.root / row.mask_path).convert("L")
        bbox = (int(row.bbox_x1), int(row.bbox_y1), int(row.bbox_x2), int(row.bbox_y2))
        frame_crop = crop_with_padding(frame, bbox)
        mask_crop = crop_with_padding(mask, bbox)
        if self.augment and random.random() < 0.5:
            frame_crop = frame_crop.transpose(Image.FLIP_LEFT_RIGHT)
            mask_crop = mask_crop.transpose(Image.FLIP_LEFT_RIGHT)

        image_tensor = pil_to_tensor(frame_crop, self.image_size)
        mask_tensor = pil_to_tensor(mask_crop, self.image_size, grayscale=True)
        masked_tensor = image_tensor * (mask_tensor > 0.1).float()

        if self.mode == "raft":
            # RAFT-like models compare two feature streams. Here we compare the
            # masked building crop with its surrounding context because the data
            # does not provide optical-flow vectors or paired velocity labels.
            x = torch.cat([masked_tensor, image_tensor], dim=0)
            extra = torch.zeros(1, dtype=torch.float32)
        elif self.mode == "segmented":
            x = torch.cat([image_tensor, mask_tensor], dim=0)
            extra = torch.tensor(
                [
                    float(row.mask_area_fraction),
                    float(row.bbox_width) / 1280.0,
                    float(row.bbox_height) / 720.0,
                    float(row.mask_centroid_x),
                    float(row.mask_centroid_y),
                ],
                dtype=torch.float32,
            )
        else:
            x = torch.cat([image_tensor, mask_tensor], dim=0)
            extra = torch.zeros(1, dtype=torch.float32)

        y = torch.tensor(int(row.damage_level), dtype=torch.long)
        return x, extra, y


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict:
    y_true_arr = np.asarray(list(y_true), dtype=np.float32)
    y_pred_arr = np.asarray(list(y_pred), dtype=np.float32)
    rmse = math.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    return {
        "accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "within_1_accuracy": float(np.mean(np.abs(y_true_arr - y_pred_arr) <= 1)),
        "mae_class_error": mean_absolute_error(y_true_arr, y_pred_arr),
        "rmse_class_error": rmse,
        "r2_ordinal": r2_score(y_true_arr, y_pred_arr) if len(np.unique(y_true_arr)) > 1 else float("nan"),
        "macro_f1": f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0),
    }


def update_metrics_csv(paths: ProjectPaths, row: dict) -> None:
    metrics_path = paths.results / "model_metrics.csv"
    existing = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    existing = existing[existing["model"] != row["model"]] if "model" in existing else existing
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(metrics_path, index=False)


def save_history_csv(paths: ProjectPaths, model_name: str, history: list[dict]) -> Path:
    path = paths.results / f"{model_name}_history.csv"
    pd.DataFrame(history).to_csv(path, index=False)
    return path

