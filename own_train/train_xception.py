from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from own_train.data import DeepfakeFrameDataset, SampleRecord, build_grouped_splits, serialize_samples
from own_train.plotting import save_confusion_matrix, save_error_gallery, save_learning_curves, save_roc_curve


matplotlib.use("Agg")


@dataclass
class TrainConfig:
    dataset_root: str
    output_root: str
    image_size: int = 299
    batch_size: int = 32
    epochs: int = 12
    frozen_epochs: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42
    num_workers: int = 4
    patience: int = 4
    use_weighted_sampler: bool = True
    decision_threshold: float = 0.5
    model_name: str = "xception"
    pretrained: bool = True
    amp: bool = True
    device: str | None = None


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Fine-tune an Xception model on the local deepfake dataset.")
    parser.add_argument("--dataset-root", default="Dataset", help="Path to the dataset root.")
    parser.add_argument("--output-root", default="own_train/runs", help="Directory where training runs are stored.")
    parser.add_argument("--image-size", type=int, default=299, help="Input image size for Xception.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=12, help="Maximum number of epochs.")
    parser.add_argument("--frozen-epochs", type=int, default=2, help="Number of warmup epochs with only the classifier unfrozen.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Base learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Identity-level train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Identity-level validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience.")
    parser.add_argument("--disable-weighted-sampler", action="store_true", help="Disable the weighted sampler.")
    parser.add_argument("--decision-threshold", type=float, default=0.5, help="Probability threshold for binary predictions.")
    parser.add_argument("--device", default=None, help="Force a device such as cpu or cuda.")

    args = parser.parse_args()
    return TrainConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        frozen_epochs=args.frozen_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        patience=args.patience,
        use_weighted_sampler=not args.disable_weighted_sampler,
        decision_threshold=args.decision_threshold,
        device=args.device,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    return train_transform, eval_transform


def create_weighted_sampler(samples: list[SampleRecord]) -> WeightedRandomSampler:
    label_counts = np.bincount([sample.label for sample in samples], minlength=2)
    class_weights = np.array([0.0, 0.0], dtype=np.float64)
    for label, count in enumerate(label_counts):
        if count > 0:
            class_weights[label] = 1.0 / count
    sample_weights = [class_weights[sample.label] for sample in samples]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def create_dataloaders(
    split_records: dict[str, list[SampleRecord]],
    config: TrainConfig,
) -> tuple[dict[str, DataLoader], dict[str, DeepfakeFrameDataset]]:
    train_transform, eval_transform = build_transforms(config.image_size)
    datasets = {
        "train": DeepfakeFrameDataset(split_records["train"], transform=train_transform),
        "val": DeepfakeFrameDataset(split_records["val"], transform=eval_transform),
        "test": DeepfakeFrameDataset(split_records["test"], transform=eval_transform),
    }

    train_loader_kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if config.use_weighted_sampler:
        train_loader_kwargs["sampler"] = create_weighted_sampler(split_records["train"])
    else:
        train_loader_kwargs["shuffle"] = True

    dataloaders = {
        "train": DataLoader(datasets["train"], **train_loader_kwargs),
        "val": DataLoader(
            datasets["val"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }
    return dataloaders, datasets


def build_model(config: TrainConfig, device: torch.device) -> torch.nn.Module:
    model = timm.create_model(config.model_name, pretrained=config.pretrained, num_classes=1)
    return model.to(device)


def set_backbone_trainability(model: torch.nn.Module, freeze_backbone: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = not freeze_backbone

    if freeze_backbone:
        classifier = getattr(model, "fc", None)
        if classifier is None:
            raise AttributeError("Expected Xception model to expose classifier as `fc`.")
        for parameter in classifier.parameters():
            parameter.requires_grad = True


def make_loss(samples: list[SampleRecord], device: torch.device) -> nn.Module:
    labels = np.array([sample.label for sample in samples], dtype=np.int64)
    real_count = max(int((labels == 0).sum()), 1)
    fake_count = max(int((labels == 1).sum()), 1)
    pos_weight = torch.tensor([real_count / fake_count], device=device, dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].float().unsqueeze(1).to(device, non_blocking=True)
    return images, labels


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    amp_enabled: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    probabilities: list[float] = []
    labels: list[int] = []
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            images, target = batch_to_device(batch, device)
            with autocast(enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, target)

            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            batch_labels = target.squeeze(1).detach().cpu().numpy().astype(int)

            total_loss += float(loss.item()) * len(batch_labels)
            total_items += len(batch_labels)
            probabilities.extend(probs.tolist())
            labels.extend(batch_labels.tolist())

            for path, label, probability in zip(batch["path"], batch_labels.tolist(), probs.tolist()):
                prediction = int(probability >= threshold)
                records.append(
                    {
                        "path": path,
                        "label": int(label),
                        "probability": float(probability),
                        "prediction": prediction,
                    }
                )

    predictions = [int(probability >= threshold) for probability in probabilities]
    metrics = compute_metrics(labels, probabilities, predictions)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics, records


def compute_metrics(labels: list[int], probabilities: list[float], predictions: list[int]) -> dict[str, float]:
    if not labels:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.0,
        }

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(labels, probabilities))
    except ValueError:
        metrics["auc"] = 0.0
    return metrics


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    threshold: float,
    amp_enabled: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_items = 0
    probabilities: list[float] = []
    labels: list[int] = []

    progress = tqdm(loader, leave=False)
    for batch in progress:
        images, target = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
        batch_labels = target.squeeze(1).detach().cpu().numpy().astype(int)

        total_loss += float(loss.item()) * len(batch_labels)
        total_items += len(batch_labels)
        probabilities.extend(probs.tolist())
        labels.extend(batch_labels.tolist())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    predictions = [int(probability >= threshold) for probability in probabilities]
    metrics = compute_metrics(labels, probabilities, predictions)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


def save_csv_records(records: list[dict[str, Any]], output_path: str | Path) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_json(data: Any, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def train_model(config: TrainConfig) -> Path:
    seed_everything(config.seed)
    device = resolve_device(config.device)

    run_dir = Path(config.output_root).resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_xception")
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    checkpoints_dir = run_dir / "checkpoints"
    reports_dir = run_dir / "reports"
    for path in (plots_dir, checkpoints_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    split_records, split_summary = build_grouped_splits(
        dataset_root=config.dataset_root,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    dataloaders, _ = create_dataloaders(split_records, config)

    model = build_model(config, device)
    criterion = make_loss(split_records["train"], device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    scaler = GradScaler(enabled=config.amp and device.type == "cuda")

    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_score = (-1.0, -1.0, float("inf"))
    patience_counter = 0

    save_json(asdict(config), run_dir / "config.json")
    save_json(split_summary, run_dir / "split_summary.json")
    save_json({split: serialize_samples(samples[:100]) for split, samples in split_records.items()}, run_dir / "sample_manifest_preview.json")

    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        freeze_backbone = epoch <= config.frozen_epochs
        set_backbone_trainability(model, freeze_backbone=freeze_backbone)

        train_metrics = train_one_epoch(
            model,
            dataloaders["train"],
            criterion,
            optimizer,
            scaler,
            device,
            config.decision_threshold,
            amp_enabled=config.amp and device.type == "cuda",
        )
        val_metrics, val_records = evaluate_model(
            model,
            dataloaders["val"],
            criterion,
            device,
            config.decision_threshold,
            amp_enabled=config.amp and device.type == "cuda",
        )
        scheduler.step(val_metrics["loss"])

        epoch_record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_auc": train_metrics["auc"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
            "backbone_frozen": freeze_backbone,
        }
        history.append(epoch_record)
        save_json(history, run_dir / "history.json")

        current_score = (val_metrics["auc"], val_metrics["f1"], -val_metrics["loss"])
        if current_score > best_score:
            best_score = current_score
            patience_counter = 0
            best_state = {
                "epoch": epoch,
                "config": asdict(config),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "val_metrics": val_metrics,
            }
            torch.save(best_state, checkpoints_dir / "best_model.pt")
            save_csv_records(val_records, reports_dir / "best_val_predictions.csv")
        else:
            patience_counter += 1

        torch.save(
            {
                "epoch": epoch,
                "config": asdict(config),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            },
            checkpoints_dir / "last_model.pt",
        )

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f}"
        )

        if patience_counter >= config.patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    if best_state is None:
        raise RuntimeError("Training completed without a valid best checkpoint.")

    model.load_state_dict(best_state["model_state_dict"])

    val_metrics, val_records = evaluate_model(
        model,
        dataloaders["val"],
        criterion,
        device,
        config.decision_threshold,
        amp_enabled=config.amp and device.type == "cuda",
    )
    test_metrics, test_records = evaluate_model(
        model,
        dataloaders["test"],
        criterion,
        device,
        config.decision_threshold,
        amp_enabled=config.amp and device.type == "cuda",
    )

    save_csv_records(val_records, reports_dir / "val_predictions.csv")
    save_csv_records(test_records, reports_dir / "test_predictions.csv")

    val_labels = [record["label"] for record in val_records]
    val_probs = [record["probability"] for record in val_records]
    val_preds = [record["prediction"] for record in val_records]
    test_labels = [record["label"] for record in test_records]
    test_probs = [record["probability"] for record in test_records]
    test_preds = [record["prediction"] for record in test_records]

    save_learning_curves(history, plots_dir / "learning_curves.png")
    save_confusion_matrix(val_labels, val_preds, plots_dir / "val_confusion_matrix.png", "Validation Confusion Matrix")
    save_confusion_matrix(test_labels, test_preds, plots_dir / "test_confusion_matrix.png", "Test Confusion Matrix")
    save_roc_curve(val_labels, val_probs, plots_dir / "val_roc_curve.png", "Validation ROC Curve")
    save_roc_curve(test_labels, test_probs, plots_dir / "test_roc_curve.png", "Test ROC Curve")

    val_errors = [record for record in val_records if record["label"] != record["prediction"]]
    val_errors.sort(key=lambda item: abs(item["probability"] - item["label"]), reverse=True)
    save_error_gallery(val_errors, plots_dir / "val_error_gallery.png")

    test_errors = [record for record in test_records if record["label"] != record["prediction"]]
    test_errors.sort(key=lambda item: abs(item["probability"] - item["label"]), reverse=True)
    save_error_gallery(test_errors, plots_dir / "test_error_gallery.png")

    final_report = {
        "best_epoch": best_state["epoch"],
        "training_time_minutes": round((time.time() - start_time) / 60.0, 2),
        "device": str(device),
        "split_summary": split_summary,
        "final_val_metrics": val_metrics,
        "final_test_metrics": test_metrics,
        "classification_report_val": classification_report(val_labels, val_preds, target_names=["REAL", "FAKE"], output_dict=True, zero_division=0),
        "classification_report_test": classification_report(test_labels, test_preds, target_names=["REAL", "FAKE"], output_dict=True, zero_division=0),
    }
    save_json(final_report, run_dir / "final_report.json")

    print("\nTraining complete.")
    print(f"Best checkpoint: {checkpoints_dir / 'best_model.pt'}")
    print(f"Final report: {run_dir / 'final_report.json'}")
    print(f"Plots directory: {plots_dir}")
    print(
        "Summary metrics | "
        f"val_acc={val_metrics['accuracy']:.4f}, val_auc={val_metrics['auc']:.4f}, "
        f"test_acc={test_metrics['accuracy']:.4f}, test_auc={test_metrics['auc']:.4f}"
    )
    return run_dir


def main() -> int:
    config = parse_args()
    train_model(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
