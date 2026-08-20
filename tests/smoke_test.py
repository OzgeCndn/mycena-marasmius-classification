"""
Gerçek veri/eğitim olmadan tüm pipeline'ın uçtan uca ÇALIŞTIĞINI doğrulayan hızlı
sağlık kontrolü: sentetik (rastgele gürültü) görsellerle her model mimarisinin
kurulup forward-pass yapabildiğini, veri yükleyicinin doğru train/test bölmesi
ürettiğini, hibrit SOM modellerinin ve tablo üretim kodunun doğru çalıştığını
birkaç saniye/dakika içinde test eder. GERÇEK bir eğitim/sonuç DEĞİLDİR.

Kullanım: python3 -m tests.smoke_test
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.data import set_seed
from src.evaluate import build_table5, build_table6
from src.main import run_evaluate_stage, run_train_stage

CLASS_NAMES = [
    "Marasmius_oreades",
    "Marasmius_rotula",
    "Mycena_crocata",
    "Mycena_epipterygia",
    "Mycena_pura",
    "Mycena_rosea",
    "Mycena_seynii",
]

# Gerçek makale konfigürasyonunun ÇOK küçültülmüş / hızlandırılmış bir versiyonu.
# Amaç sonuç kalitesi değil, kodun uçtan uca hatasız çalıştığını doğrulamak.
SMOKE_CFG = {
    "seed": 42,
    "dataset": {
        "class_names": CLASS_NAMES,
        "expected_counts": {c: 10 for c in CLASS_NAMES},
        "image_size": 224,  # MaxViT'in pencere bölümlemesi (window=7) 224 boyutunu gerektirir
        "train_split": 0.7,
        "test_split": 0.3,
    },
    "augmentation": {
        "horizontal_flip_p": 0.5,
        "rotation_degrees": 15,
        "color_jitter": {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.05},
    },
    "training": {
        "optimizer": "adam",
        "lr_scheduler": "step_lr",
        "step_lr": {"step_size": 7, "gamma": 0.1},
        "hyperparam_search": {
            "method": "random",
            "n_trials": 1,
            "batch_size": [4],
            "learning_rate": [0.001],
            "weight_decay": [0.0001],
        },
        "epochs": {m: 1 for m in [
            "custom_cnn", "googlenet", "mobilenetv3_large", "vgg19", "resnetv2_50",
            "efficientnet_b0", "efficientnetv2_m", "maxvit_s", "cnn_kan", "ensemble_maxvit_resnet",
        ]},
        "early_stopping": {"patience": 5, "monitor": "val_loss"},
    },
    "som": {
        "cnn_som": {"grid_size": [4, 4], "sigma": 1.0, "learning_rate": 0.5, "iterations": 20},
        "maxvit_som": {"grid_size": [4, 4], "sigma": 1.0, "learning_rate": 0.5, "iterations": 20},
    },
    "kan": {"hidden_dim": 32, "n_basis": 4},
    "ensemble": {"feature_dims": {"maxvit_s": 768, "resnetv2_50": 2048}, "fc1": 64, "dropout": 0.5},
}


def make_synthetic_dataset(root: Path, n_per_class: int = 10):
    rng = np.random.default_rng(0)
    for cname in CLASS_NAMES:
        cdir = root / cname
        cdir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            arr = rng.integers(0, 255, size=(80, 80, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cdir / f"{cname}_{i}.jpg")


def main():
    set_seed(42)
    tmp_dir = Path(tempfile.mkdtemp(prefix="mantar_smoke_"))
    data_dir = tmp_dir / "Classes"
    output_dir = tmp_dir / "outputs"
    try:
        print(f"Geçici dizin: {tmp_dir}")
        make_synthetic_dataset(data_dir, n_per_class=10)

        # Hızlı olması için sadece bir alt küme model test edilir; her mimari
        # ailesinden (custom, torchvision, timm, hibrit-SOM, KAN, ensemble) en az
        # bir temsilci içerir.
        models_to_test = [
            "custom_cnn",
            "googlenet",
            "resnetv2_50",
            "maxvit_s",
            "cnn_kan",
            "ensemble_maxvit_resnet",
            "cnn_som",
            "maxvit_som",
        ]

        print("\n--- Eğitim aşaması (sentetik veri, pretrained=False) ---")
        run_train_stage(SMOKE_CFG, data_dir, output_dir, models_to_test, pretrained=False)

        print("\n--- Değerlendirme aşaması ---")
        run_evaluate_stage(SMOKE_CFG, output_dir)

        table5_path = output_dir / "table5_performance.csv"
        table6_path = output_dir / "table6_chisquare.csv"
        assert table5_path.exists(), "Table 5 üretilmedi!"
        assert table6_path.exists(), "Table 6 üretilmedi!"

        print("\n✅ SMOKE TEST BAŞARILI: pipeline uçtan uca hatasız çalıştı.")
        print(f"   (Bu sonuçlar sentetik rastgele veriyle üretildi, ANLAMLI DEĞİLDİR.)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
