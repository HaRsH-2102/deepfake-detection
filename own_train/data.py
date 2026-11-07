from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class SampleRecord:
    path: str
    label: int
    split: str
    group_id: str
    primary_id: str
    secondary_id: str | None
    method: str


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _scan_real_records(dataset_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    real_root = dataset_root / "real"
    for identity_dir in sorted(path for path in real_root.iterdir() if path.is_dir()):
        identity_id = identity_dir.name
        for image_path in sorted(path for path in identity_dir.iterdir() if _is_image(path)):
            records.append(
                {
                    "path": str(image_path.resolve()),
                    "label": 0,
                    "primary_id": identity_id,
                    "secondary_id": None,
                    "method": "real",
                    "group_id": f"real/{identity_id}",
                }
            )
    return records


def _scan_fake_records(dataset_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    fake_root = dataset_root / "fake"
    for method_dir in sorted(path for path in fake_root.iterdir() if path.is_dir()):
        method = method_dir.name
        for pair_dir in sorted(path for path in method_dir.iterdir() if path.is_dir()):
            pair_name = pair_dir.name
            ids = pair_name.split("_", maxsplit=1)
            if len(ids) != 2:
                continue
            primary_id, secondary_id = ids
            for image_path in sorted(path for path in pair_dir.iterdir() if _is_image(path)):
                records.append(
                    {
                        "path": str(image_path.resolve()),
                        "label": 1,
                        "primary_id": primary_id,
                        "secondary_id": secondary_id,
                        "method": method,
                        "group_id": f"{method}/{pair_name}",
                    }
                )
    return records


def scan_dataset(dataset_root: str | Path) -> dict[str, list[dict[str, Any]]]:
    root = Path(dataset_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    return {
        "real": _scan_real_records(root),
        "fake": _scan_fake_records(root),
    }


def split_identities(
    identity_ids: Iterable[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    ids = list(identity_ids)
    if not ids:
        raise ValueError("No identity IDs were found for splitting.")

    rng = random.Random(seed)
    rng.shuffle(ids)

    train_count = int(len(ids) * train_ratio)
    val_count = int(len(ids) * val_ratio)
    test_count = len(ids) - train_count - val_count
    if test_count <= 0:
        raise ValueError("Split ratios leave no identities for the test split.")

    split_map: dict[str, str] = {}
    for index, identity_id in enumerate(ids):
        if index < train_count:
            split_map[identity_id] = "train"
        elif index < train_count + val_count:
            split_map[identity_id] = "val"
        else:
            split_map[identity_id] = "test"
    return split_map


def build_grouped_splits(
    dataset_root: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[dict[str, list[SampleRecord]], dict[str, Any]]:
    scanned = scan_dataset(dataset_root)
    real_records = scanned["real"]
    fake_records = scanned["fake"]

    identity_ids = sorted({record["primary_id"] for record in real_records})
    identity_split_map = split_identities(identity_ids, train_ratio, val_ratio, seed)

    split_records: dict[str, list[SampleRecord]] = {"train": [], "val": [], "test": []}
    split_summary: dict[str, Any] = {
        "split_mode": "identity_grouped",
        "identity_split_counts": Counter(identity_split_map.values()),
        "real_frame_counts": {},
        "fake_frame_counts": {},
        "fake_method_counts": {},
        "dropped_cross_split_fake_frames": 0,
        "dropped_cross_split_fake_groups": 0,
        "retained_fake_groups": 0,
    }

    for record in real_records:
        split = identity_split_map[record["primary_id"]]
        sample = SampleRecord(split=split, **record)
        split_records[split].append(sample)

    fake_group_status: dict[str, str] = {}
    for record in fake_records:
        left_split = identity_split_map.get(record["primary_id"])
        right_split = identity_split_map.get(record["secondary_id"] or "")
        if left_split is None or right_split is None or left_split != right_split:
            split_summary["dropped_cross_split_fake_frames"] += 1
            fake_group_status.setdefault(record["group_id"], "dropped")
            continue

        split = left_split
        sample = SampleRecord(split=split, **record)
        split_records[split].append(sample)
        fake_group_status[record["group_id"]] = "retained"

    split_summary["dropped_cross_split_fake_groups"] = sum(
        1 for status in fake_group_status.values() if status == "dropped"
    )
    split_summary["retained_fake_groups"] = sum(
        1 for status in fake_group_status.values() if status == "retained"
    )

    for split_name, samples in split_records.items():
        label_counts = Counter(sample.label for sample in samples)
        method_counts = Counter(sample.method for sample in samples if sample.label == 1)
        group_counts = Counter(sample.group_id for sample in samples)
        split_summary["real_frame_counts"][split_name] = int(label_counts.get(0, 0))
        split_summary["fake_frame_counts"][split_name] = int(label_counts.get(1, 0))
        split_summary["fake_method_counts"][split_name] = dict(sorted(method_counts.items()))
        split_summary[f"{split_name}_group_count"] = len(group_counts)
        split_summary[f"{split_name}_sample_count"] = len(samples)

    return split_records, split_summary


class DeepfakeFrameDataset(Dataset):
    def __init__(self, samples: list[SampleRecord], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": sample.label,
            "path": sample.path,
            "group_id": sample.group_id,
            "method": sample.method,
            "split": sample.split,
        }


def serialize_samples(samples: list[SampleRecord]) -> list[dict[str, Any]]:
    return [asdict(sample) for sample in samples]
