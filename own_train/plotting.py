from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve


def save_learning_curves(history: list[dict], output_path: str | Path) -> None:
    if not history:
        return

    output = Path(output_path)
    epochs = [item["epoch"] for item in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    axes[0].plot(epochs, [item["train_loss"] for item in history], label="Train")
    axes[0].plot(epochs, [item["val_loss"] for item in history], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [item["train_accuracy"] for item in history], label="Train")
    axes[1].plot(epochs, [item["val_accuracy"] for item in history], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, [item["train_f1"] for item in history], label="Train")
    axes[2].plot(epochs, [item["val_f1"] for item in history], label="Val")
    axes[2].set_title("F1 Score")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    axes[3].plot(epochs, [item["train_auc"] for item in history], label="Train")
    axes[3].plot(epochs, [item["val_auc"] for item in history], label="Val")
    axes[3].set_title("ROC AUC")
    axes[3].set_xlabel("Epoch")
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(
    labels: Iterable[int],
    predictions: Iterable[int],
    output_path: str | Path,
    title: str,
) -> None:
    matrix = confusion_matrix(list(labels), list(predictions), labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["REAL", "FAKE"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap="Blues", ax=ax, colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(
    labels: Iterable[int],
    probabilities: Iterable[float],
    output_path: str | Path,
    title: str,
) -> None:
    labels_array = np.asarray(list(labels))
    probs_array = np.asarray(list(probabilities))
    if len(np.unique(labels_array)) < 2:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(labels_array, probs_array, ax=ax, name="Xception")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_error_gallery(
    records: list[dict],
    output_path: str | Path,
    max_items: int = 16,
) -> None:
    if not records:
        return

    selected = records[:max_items]
    columns = 4
    rows = math.ceil(len(selected) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for axis in axes:
        axis.axis("off")

    for axis, record in zip(axes, selected):
        image = Image.open(record["path"]).convert("RGB")
        axis.imshow(image)
        axis.set_title(
            f"GT: {'FAKE' if record['label'] else 'REAL'}\n"
            f"Pred: {'FAKE' if record['prediction'] else 'REAL'}\n"
            f"P(fake)={record['probability']:.3f}",
            fontsize=9,
        )
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
