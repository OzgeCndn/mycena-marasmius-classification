"""
Veri yükleme ve ön işleme.

Makale (Ekinci et al., 2025, Sensors 25:1642) veri setini şöyle tanımlıyor:
  - 7 sınıf (Marasmius x2, Mycena x5), toplam 1582 görsel
  - %70 eğitim / %30 bağımsız test, sınıf oranı korunarak (stratified)
  - Eğitim setine her epoch başında: random horizontal flip, random rotation
    (+-15 derece), color jitter (brightness/contrast/saturation=0.1, hue=0.05)
  - Görseller 224x224'e yeniden boyutlandırılıyor
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Sample:
    path: Path
    label: int


def scan_dataset(data_dir: str | Path, class_names: list[str]) -> list[Sample]:
    """data_dir/<class_name>/*.{jpg,jpeg,png} yapısını tarar."""
    data_dir = Path(data_dir)
    samples: list[Sample] = []
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    for idx, cname in enumerate(class_names):
        class_dir = data_dir / cname
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Sınıf klasörü bulunamadı: {class_dir}. "
                f"README'deki 'Veri setini indirme' adımlarını kontrol edin ve "
                f"configs/hyperparams.yaml -> dataset.class_names listesinin "
                f"gerçek klasör isimleriyle eşleştiğinden emin olun."
            )
        files = sorted(p for p in class_dir.iterdir() if p.suffix in exts)
        for p in files:
            samples.append(Sample(path=p, label=idx))
    return samples


def stratified_split(samples: list[Sample], train_ratio: float, seed: int):
    labels = [s.label for s in samples]
    train_samples, test_samples = train_test_split(
        samples,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed,
    )
    return train_samples, test_samples


class MushroomDataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: int, augment: bool, aug_cfg: dict | None = None):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment
        self.aug_cfg = aug_cfg or {}
        self.transform = self._build_transform()

    def _build_transform(self):
        ops = [transforms.Resize((self.image_size, self.image_size))]
        if self.augment:
            ops.append(transforms.RandomHorizontalFlip(p=self.aug_cfg.get("horizontal_flip_p", 0.5)))
            ops.append(transforms.RandomRotation(self.aug_cfg.get("rotation_degrees", 15)))
            cj = self.aug_cfg.get("color_jitter", {})
            ops.append(
                transforms.ColorJitter(
                    brightness=cj.get("brightness", 0.1),
                    contrast=cj.get("contrast", 0.1),
                    saturation=cj.get("saturation", 0.1),
                    hue=cj.get("hue", 0.05),
                )
            )
        ops.append(transforms.ToTensor())
        ops.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        return transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        img = self.transform(img)
        return img, s.label


def build_datasets(cfg: dict, data_dir: str | Path):
    ds_cfg = cfg["dataset"]
    class_names = ds_cfg["class_names"]
    samples = scan_dataset(data_dir, class_names)

    # Sağlık kontrolü: beklenen sayımlarla karşılaştır (uyarı verir, durdurmaz)
    counts: dict[str, int] = {c: 0 for c in class_names}
    for s in samples:
        counts[class_names[s.label]] += 1
    expected = ds_cfg.get("expected_counts", {})
    mismatches = {c: (counts[c], expected.get(c)) for c in class_names if expected.get(c) not in (None, counts[c])}
    if mismatches:
        print(f"[UYARI] Bulunan görsel sayıları Table 4 ile tam eşleşmiyor: {mismatches}")

    train_samples, test_samples = stratified_split(samples, ds_cfg["train_split"], cfg["seed"])

    train_ds = MushroomDataset(
        train_samples, ds_cfg["image_size"], augment=True, aug_cfg=cfg.get("augmentation", {})
    )
    test_ds = MushroomDataset(test_samples, ds_cfg["image_size"], augment=False)
    return train_ds, test_ds, class_names
