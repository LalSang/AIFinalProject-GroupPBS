# Task 2 Surface Velocity / DoriaNET Computer Vision Pipeline

## Critical Dataset Note

The folder name and course task mention surface-velocity estimation, but the supplied PDFs and raw data are the DoriaNET post-hurricane UAV building damage dataset. There are no water-surface velocity vectors, optical-flow labels, endpoint-error targets, or stream images in the raw files. The available supervised prediction target is ordinal building damage level `0` through `5`.

This pipeline therefore implements a defensible deep computer-vision workflow for the actual dataset:

- input: UAV frame crop plus building mask
- target: DoriaNET/FEMA-style building damage level
- task type: segmentation-assisted ordinal image classification

## Dataset Structure

Raw data:

- `data/raw/FRAME/`: 271 full UAV frames, all `1280x720`
- `data/raw/JSON/`: 271 annotation files, one per frame
- `data/raw/MASK/`: 1,458 building mask images, all `1280x720`

Each JSON has:

- `Frame_Name`
- `Buildings`
- `Capture date`
- `Region`
- `Original video link`

Each `Buildings` row is inferred as:

```text
[building_id, "latitude, longitude", mask_file, damage_level, stories, annotation_effort, reserved_nan]
```

Mask filenames follow:

```text
<video>_<frame>_B0XX_<object_index>_Level<damage_level>.jpg
```

## Preprocessing

The preprocessing script parses JSON annotations, checks mask files, computes each mask bounding box and area statistics, and writes:

- `data/processed/metadata.csv`
- `data/processed/splits.csv`
- `figures/sample_frame_mask_visualization.png`
- `figures/target_distribution.png`

The split is grouped by `building_id` to reduce leakage from the same physical building appearing across multiple frames. The paper used 60/20/20 for Dataset 1, so this project follows 60% train, 20% validation, and 20% test.

## Models

Three simple PyTorch models are included:

- `train_flownet_style.py`: strided encoder over stacked RGB crop plus mask, inspired by FlowNet-style dense visual feature extraction.
- `train_raft_style.py`: RAFT-inspired two-stream feature-correlation model comparing masked building appearance with local context.
- `train_cnn_velocity.py`: segmentation-assisted CNN using RGB plus mask and mask geometry. The filename follows the assignment request, but the model predicts damage level because velocity labels are absent.

## How To Run

From the repository root:

```bash
python3 task2_surface_velocity/code/inspect_dataset.py
python3 task2_surface_velocity/code/preprocess_data.py

python3 task2_surface_velocity/code/train_flownet_style.py --epochs 8 --batch-size 32 --image-size 128
python3 task2_surface_velocity/code/train_raft_style.py --epochs 8 --batch-size 32 --image-size 128
python3 task2_surface_velocity/code/train_cnn_velocity.py --epochs 8 --batch-size 32 --image-size 128
```

For a fast smoke test, use `--epochs 3`.

## Outputs

Inspection and preprocessing:

- `results/dataset_summary.txt`
- `data/processed/metadata.csv`
- `data/processed/splits.csv`

Training/evaluation:

- `models/*.pt`
- `results/model_metrics.csv`
- `results/*_history.csv`
- `figures/*_loss_curve.png`
- `figures/*_predicted_vs_actual.png`
- `figures/model_comparison_bar_chart.png`

## Evaluation Metrics

Because the actual target is ordinal damage classification, the scripts report:

- exact accuracy
- within-one-class accuracy
- MAE of class error
- RMSE of class error
- ordinal R2
- macro F1

IoU is relevant to the provided masks, but these scripts do not train a segmentation model that predicts masks. The masks are used as supervised localization/context inputs.

