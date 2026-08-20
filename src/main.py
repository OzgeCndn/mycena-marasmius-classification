"""
Uçtan uca orkestratör.

Kullanım:
    python3 src/main.py --data-dir data/Classes --output-dir outputs --stage all
    python3 src/main.py --data-dir data/Classes --output-dir outputs --stage train --models maxvit_s cnn_kan
    python3 src/main.py --data-dir data/Classes --output-dir outputs --stage evaluate
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from .data import build_datasets, load_config, set_seed
from .evaluate import build_table5, build_table6, load_predictions, save_predictions, write_outputs
from .hybrid import train_cnn_som, train_maxvit_som
from .models.pretrained import build_feature_extractor
from .models.registry import ALL_MODELS, STANDARD_MODELS, build_model
from .train import evaluate_loader, train_standard_model

# custom_cnn ve maxvit_s, SOM varyantlarının ön koşuludur -> önce onlar eğitilmeli
TRAIN_ORDER = [
    "custom_cnn",
    "googlenet",
    "mobilenetv3_large",
    "vgg19",
    "resnetv2_50",
    "efficientnet_b0",
    "efficientnetv2_m",
    "maxvit_s",
    "cnn_kan",
    "ensemble_maxvit_resnet",
    "cnn_som",     # custom_cnn checkpoint'ine bağımlı
    "maxvit_som",  # maxvit_s checkpoint'ine bağımlı
]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_or_train(name, cfg, train_ds, num_classes, device, ckpt_dir: Path, pretrained: bool, force_retrain: bool = False):
    """Checkpoint diskte varsa (ve force_retrain=False ise) yeniden eğitmek yerine
    ağırlıkları yükler -- böylece cnn_som/maxvit_som gibi bağımlı modeller için
    custom_cnn/maxvit_s'i tekrar tekrar eğitmek gerekmez."""
    ckpt_path = ckpt_dir / f"{name}.pt"
    if not force_retrain and ckpt_path.exists():
        print(f"[{name}] checkpoint bulundu ({ckpt_path}), eğitim ATLANIYOR, ağırlıklar yükleniyor...")
        model = build_model(name, num_classes, pretrained=pretrained).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        return model
    model, _ = train_standard_model(name, cfg, train_ds, num_classes, device, ckpt_dir, pretrained=pretrained)
    return model


def run_train_stage(cfg, data_dir, output_dir: Path, models_filter, pretrained: bool = True, force_retrain: bool = False):
    device = get_device()
    print(f"Kullanılan cihaz: {device}")
    train_ds, test_ds, class_names = build_datasets(cfg, data_dir)
    num_classes = len(class_names)
    print(f"Eğitim örnekleri: {len(train_ds)}, Test örnekleri: {len(test_ds)}, sınıflar: {class_names}")

    ckpt_dir = output_dir / "checkpoints"
    pred_dir = output_dir / "predictions"

    order = [m for m in TRAIN_ORDER if models_filter is None or m in models_filter]

    trained_models = {}  # model_name -> nn.Module (bellek içinde, SOM bağımlılıkları için)

    for name in order:
        print(f"\n===== {name} =====")
        if name == "cnn_som":
            if "custom_cnn" not in trained_models:
                print("[cnn_som] custom_cnn bellekte yok, checkpoint'ten yükleniyor/gerekirse eğitiliyor...")
                trained_models["custom_cnn"] = _load_or_train(
                    "custom_cnn", cfg, train_ds, num_classes, device, ckpt_dir, pretrained, force_retrain
                )
            preds, y_true, _ = train_cnn_som(trained_models["custom_cnn"], train_ds, test_ds, cfg, device, ckpt_dir)
            save_predictions(pred_dir, name, y_true, preds, None)
            continue

        if name == "maxvit_som":
            if "maxvit_s" not in trained_models:
                print("[maxvit_som] maxvit_s bellekte yok, checkpoint'ten yükleniyor/gerekirse eğitiliyor...")
                trained_models["maxvit_s"] = _load_or_train(
                    "maxvit_s", cfg, train_ds, num_classes, device, ckpt_dir, pretrained, force_retrain
                )
            feat_model, _ = build_feature_extractor("maxvit_s", pretrained=pretrained)
            # eğitilmiş maxvit_s gövde ağırlıklarını feature-extractor'a aktar
            trained_state = trained_models["maxvit_s"].state_dict()
            feat_model.load_state_dict({k: v for k, v in trained_state.items() if k in feat_model.state_dict()}, strict=False)
            feat_model = feat_model.to(device)
            preds, y_true, _ = train_maxvit_som(feat_model, train_ds, test_ds, cfg, device, ckpt_dir)
            save_predictions(pred_dir, name, y_true, preds, None)
            continue

        model = _load_or_train(name, cfg, train_ds, num_classes, device, ckpt_dir, pretrained, force_retrain)

        test_loader_acc, preds, proba, y_true = evaluate_loader(
            model, torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0), device, name
        )
        print(f"[{name}] test doğruluğu: {test_loader_acc:.4f}")
        save_predictions(pred_dir, name, y_true, preds, proba)

        # cnn_som/maxvit_som SADECE custom_cnn/maxvit_s'e ihtiyaç duyar; diğer
        # modelleri bellekte tutmanın anlamı yok -- RAM'i boşalt.
        if name in ("custom_cnn", "maxvit_s"):
            trained_models[name] = model
        else:
            del model
            gc.collect()


def run_evaluate_stage(cfg, output_dir: Path):
    pred_dir = output_dir / "predictions"
    predictions = load_predictions(pred_dir)
    if not predictions:
        raise RuntimeError(f"{pred_dir} içinde tahmin bulunamadı. Önce --stage train çalıştırın.")

    num_classes = len(cfg["dataset"]["class_names"])
    table5 = build_table5(predictions, num_classes)
    table6 = build_table6(predictions)

    write_outputs(table5, output_dir, "table5_performance")
    write_outputs(table6, output_dir, "table6_chisquare")

    print("\n=== Table 5: Performans Metrikleri ===")
    print(table5.to_string(index=False))
    print("\n=== Table 6: Ki-kare Anlamlılık Testi (p-değerleri) ===")
    print(table6.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--config", default="configs/hyperparams.yaml")
    parser.add_argument("--stage", choices=["train", "evaluate", "all"], default="all")
    parser.add_argument("--models", nargs="*", default=None, help=f"Alt küme seçin: {ALL_MODELS}")
    parser.add_argument("--no-pretrained", action="store_true", help="ImageNet ön-eğitimli ağırlıkları indirmeden rastgele başlat (ör. internetsiz ortamda smoke test için)")
    parser.add_argument("--force-retrain", action="store_true", help="Checkpoint diskte olsa bile modeli yeniden eğit")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    output_dir = Path(args.output_dir)

    if args.stage in ("train", "all"):
        run_train_stage(cfg, args.data_dir, output_dir, args.models, pretrained=not args.no_pretrained, force_retrain=args.force_retrain)
    if args.stage in ("evaluate", "all"):
        run_evaluate_stage(cfg, output_dir)


if __name__ == "__main__":
    main()
