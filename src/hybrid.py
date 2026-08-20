"""CNN-SOM ve MaxViT-SOM: önce ilgili gövde (custom_cnn / maxvit_s) eğitilir,
sonra ondan çıkarılan özelliklerle bir SOM sınıflandırıcı eğitilir (BMU
çoğunluk oyu ile), bkz. `src/models/som.py`."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .models.som import SOMClassifier


@torch.no_grad()
def _extract_features(feature_fn, dataset, device, batch_size: int = 32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        f = feature_fn(x)
        feats.append(f.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def train_cnn_som(custom_cnn_model, train_ds, test_ds, cfg, device, output_dir: Path):
    custom_cnn_model.eval()
    som_cfg = cfg["som"]["cnn_som"]

    train_feat, train_y = _extract_features(lambda x: custom_cnn_model.flat_features(x), train_ds, device)
    test_feat, test_y = _extract_features(lambda x: custom_cnn_model.flat_features(x), test_ds, device)

    clf = SOMClassifier(
        grid_size=tuple(som_cfg["grid_size"]),
        input_len=train_feat.shape[1],
        sigma=som_cfg["sigma"],
        learning_rate=som_cfg["learning_rate"],
        iterations=som_cfg["iterations"],
        seed=cfg["seed"],
    )
    clf.fit(train_feat, train_y)
    preds = clf.predict(test_feat)
    return preds, test_y, clf


def train_maxvit_som(maxvit_feature_model, train_ds, test_ds, cfg, device, output_dir: Path):
    maxvit_feature_model.eval()
    som_cfg = cfg["som"]["maxvit_som"]

    train_feat, train_y = _extract_features(lambda x: maxvit_feature_model(x), train_ds, device)
    test_feat, test_y = _extract_features(lambda x: maxvit_feature_model(x), test_ds, device)

    clf = SOMClassifier(
        grid_size=tuple(som_cfg["grid_size"]),
        input_len=train_feat.shape[1],
        sigma=som_cfg["sigma"],
        learning_rate=som_cfg["learning_rate"],
        iterations=som_cfg["iterations"],
        seed=cfg["seed"],
    )
    clf.fit(train_feat, train_y)
    preds = clf.predict(test_feat)
    return preds, test_y, clf
