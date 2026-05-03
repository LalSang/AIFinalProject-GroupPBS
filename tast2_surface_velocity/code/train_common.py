"""Training helpers used by the three Task 2 model scripts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "results" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from task2_utils import (
    DoriaNetDataset,
    choose_device,
    classification_metrics,
    ensure_dirs,
    load_metadata_and_splits,
    project_paths,
    save_history_csv,
    update_metrics_csv,
)


def parser_with_defaults(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(args, mode: str):
    paths = ensure_dirs(project_paths())
    metadata, _ = load_metadata_and_splits(paths)
    train_rows = metadata[metadata["split"] == "train"]
    val_rows = metadata[metadata["split"] == "val"]
    test_rows = metadata[metadata["split"] == "test"]

    train_ds = DoriaNetDataset(train_rows, paths.root, image_size=args.image_size, mode=mode, augment=True)
    val_ds = DoriaNetDataset(val_rows, paths.root, image_size=args.image_size, mode=mode, augment=False)
    test_ds = DoriaNetDataset(test_rows, paths.root, image_size=args.image_size, mode=mode, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return paths, metadata, train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device: torch.device, train: bool) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    for x, extra, y in loader:
        x = x.to(device)
        extra = extra.to(device)
        y = y.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(x, extra)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach().cpu()) * y.numel()
        correct += int((logits.argmax(dim=1) == y).sum().detach().cpu())
        total += y.numel()
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def predict(model, loader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for x, extra, y in loader:
        logits = model(x.to(device), extra.to(device))
        y_true.extend(y.numpy().astype(int).tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().astype(int).tolist())
    return y_true, y_pred


def save_loss_plot(paths, model_name: str, history: list[dict]) -> None:
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["train_loss"], label="train")
    ax.plot(df["epoch"], df["val_loss"], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"{model_name} loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths.figures / f"{model_name}_loss_curve.png", dpi=160)
    plt.close(fig)


def save_predicted_vs_actual(paths, model_name: str, y_true: list[int], y_pred: list[int]) -> None:
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(5, 5))
    jitter_true = np.asarray(y_true) + rng.normal(0, 0.05, size=len(y_true))
    jitter_pred = np.asarray(y_pred) + rng.normal(0, 0.05, size=len(y_pred))
    ax.scatter(jitter_true, jitter_pred, alpha=0.55, s=18)
    ax.plot([0, 5], [0, 5], color="black", linewidth=1)
    ax.set_xlim(-0.4, 5.4)
    ax.set_ylim(-0.4, 5.4)
    ax.set_xlabel("Actual damage level")
    ax.set_ylabel("Predicted damage level")
    ax.set_title(f"{model_name}: predicted vs actual")
    fig.tight_layout()
    fig.savefig(paths.figures / f"{model_name}_predicted_vs_actual.png", dpi=160)
    plt.close(fig)


def save_comparison_bar_chart(paths) -> None:
    metrics_path = paths.results / "model_metrics.csv"
    if not metrics_path.exists():
        return
    df = pd.read_csv(metrics_path)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["model"], df["accuracy"], label="Accuracy", color="#4C78A8")
    ax.bar(df["model"], df["within_1_accuracy"], label="Within +/-1", alpha=0.55, color="#F58518")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on held-out test split")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths.figures / "model_comparison_bar_chart.png", dpi=160)
    plt.close(fig)


def train_and_evaluate(model_name: str, model: nn.Module, args, mode: str) -> dict:
    set_seed(args.seed)
    paths, _, train_loader, val_loader, test_loader = make_loaders(args, mode)
    device = choose_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    best_state = None
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model_path = paths.models / f"{model_name}.pt"
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, model_path)

    y_true, y_pred = predict(model, test_loader, device)
    metrics = classification_metrics(y_true, y_pred)
    row = {"model": model_name, "target": "damage_level_ordinal_0_5", "epochs": args.epochs, **metrics}
    update_metrics_csv(paths, row)
    save_history_csv(paths, model_name, history)
    save_loss_plot(paths, model_name, history)
    save_predicted_vs_actual(paths, model_name, y_true, y_pred)
    save_comparison_bar_chart(paths)

    print(f"Saved model to {model_path.relative_to(paths.root)}")
    print(pd.Series(row).to_string())
    return row
