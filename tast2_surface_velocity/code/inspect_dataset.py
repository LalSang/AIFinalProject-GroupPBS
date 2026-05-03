"""Inspect the raw DoriaNET dataset and write a human-readable summary.

Run from the repository root:
    python3 task2_surface_velocity/code/inspect_dataset.py
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

from PIL import Image

from task2_utils import build_metadata, ensure_dirs, project_paths


def image_size_counts(files: list[Path]) -> tuple[collections.Counter, list[str]]:
    sizes: collections.Counter = collections.Counter()
    bad: list[str] = []
    for path in files:
        try:
            with Image.open(path) as image:
                sizes[image.size] += 1
        except Exception as exc:  # pragma: no cover - defensive data check
            bad.append(f"{path.name}: {exc}")
    return sizes, bad


def main() -> None:
    paths = ensure_dirs(project_paths())
    frame_files = sorted(paths.frames.glob("*.jpg"))
    json_files = sorted(paths.jsons.glob("*.json"))
    mask_files = sorted(paths.masks.glob("*.jpg"))

    frame_sizes, bad_frames = image_size_counts(frame_files)
    mask_sizes, bad_masks = image_size_counts(mask_files)

    frame_stems = {path.stem for path in frame_files}
    json_stems = {path.stem for path in json_files}
    mask_pattern = re.compile(r"^(?P<frame>\d+_\d+)_B(?P<masked_id>[^_]+)_(?P<object_index>\d+)_Level(?P<level>\d+)$")
    parsed_masks = []
    unparsed_masks = []
    for path in mask_files:
        match = mask_pattern.match(path.stem)
        if match:
            parsed_masks.append((path.name, match.groupdict()))
        else:
            unparsed_masks.append(path.name)
    mask_frame_stems = {item["frame"] for _, item in parsed_masks}
    masks_per_frame = collections.Counter(item["frame"] for _, item in parsed_masks)
    mask_level_counts = collections.Counter(int(item["level"]) for _, item in parsed_masks)

    top_keys: collections.Counter = collections.Counter()
    buildings_per_json: collections.Counter = collections.Counter()
    bad_json: list[str] = []
    for path in json_files:
        try:
            data = json.loads(path.read_text())
            top_keys.update(data.keys())
            buildings_per_json[len(data.get("Buildings", []))] += 1
        except Exception as exc:
            bad_json.append(f"{path.name}: {exc}")

    metadata = build_metadata(paths)
    json_mask_names = set(metadata["mask_file"])
    actual_mask_names = {path.name for path in mask_files}
    missing_mask_files = sorted(json_mask_names - actual_mask_names)
    extra_mask_files = sorted(actual_mask_names - json_mask_names)

    summary_lines = [
        "DoriaNET Dataset Inspection",
        "==========================",
        "",
        "Important interpretation note:",
        "The supplied PDFs and raw labels describe post-hurricane building damage assessment, not surface-water velocity estimation.",
        "The prediction target available in this dataset is the ordinal building damage level 0-5.",
        "",
        "Raw file counts:",
        f"- FRAME jpg files: {len(frame_files)}",
        f"- JSON files: {len(json_files)}",
        f"- MASK jpg files / building instances: {len(mask_files)}",
        "",
        "Image dimensions:",
        f"- Frame dimensions: {dict(frame_sizes)}",
        f"- Mask dimensions: {dict(mask_sizes)}",
        f"- Corrupted/unreadable frames: {bad_frames or 'none'}",
        f"- Corrupted/unreadable masks: {bad_masks or 'none'}",
        "",
        "JSON structure:",
        f"- Top-level keys: {dict(top_keys)}",
        "- Building row schema inferred from files and PDFs:",
        "  [building_id, 'latitude, longitude', mask_file, damage_level, stories, annotation_effort, reserved_nan]",
        f"- Buildings per JSON distribution: {dict(sorted(buildings_per_json.items()))}",
        f"- Bad JSON files: {bad_json or 'none'}",
        "",
        "Matching checks:",
        f"- Frames missing JSON: {sorted(frame_stems - json_stems) or 'none'}",
        f"- JSON missing frame: {sorted(json_stems - frame_stems) or 'none'}",
        f"- Frames missing any mask: {sorted(frame_stems - mask_frame_stems) or 'none'}",
        f"- Mask frame stems missing FRAME: {sorted(mask_frame_stems - frame_stems) or 'none'}",
        f"- JSON mask references missing MASK file: {missing_mask_files or 'none'}",
        f"- MASK files not referenced by JSON: {extra_mask_files or 'none'}",
        f"- Unparsed mask filenames: {unparsed_masks or 'none'}",
        "",
        "Label and metadata distributions:",
        f"- Damage level counts: {dict(sorted(metadata['damage_level'].value_counts().to_dict().items()))}",
        f"- Stories counts: {dict(sorted(metadata['stories'].value_counts().to_dict().items()))}",
        f"- Annotation effort counts: {dict(sorted(metadata['annotation_effort'].value_counts().to_dict().items()))}",
        f"- Video/frame counts: {dict(sorted(metadata.groupby('video_id')['frame_stem'].nunique().to_dict().items()))}",
        f"- Video/object counts: {dict(sorted(metadata['video_id'].value_counts().to_dict().items()))}",
        f"- Unique physical building IDs: {metadata['building_id'].nunique()}",
        "",
        "Mask filename interpretation:",
        "- Filename format: <video>_<frame>_B0XX_<object_index>_Level<damage_level>.jpg",
        f"- Parsed mask count: {len(parsed_masks)}",
        f"- Damage levels from mask filenames: {dict(sorted(mask_level_counts.items()))}",
        f"- Masks per frame: min={min(masks_per_frame.values())}, max={max(masks_per_frame.values())}, mean={sum(masks_per_frame.values()) / len(masks_per_frame):.2f}",
        "",
        "Task interpretation:",
        "- This is not optical flow and does not contain water velocity vectors, endpoint-error targets, or sequential flow ground truth.",
        "- It is a segmentation-assisted ordinal image classification setup: frames + building masks -> damage level.",
    ]

    output_path = paths.results / "dataset_summary.txt"
    output_path.write_text("\n".join(summary_lines) + "\n")
    print(f"Wrote {output_path.relative_to(paths.root)}")
    print("\n".join(summary_lines[:18]))


if __name__ == "__main__":
    main()

