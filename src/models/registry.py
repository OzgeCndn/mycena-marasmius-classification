"""Tüm modelleri isimle inşa etmek için tek merkezi kayıt (registry)."""
from __future__ import annotations

import torch.nn as nn

from .custom_cnn import CustomCNN
from .ensemble import MaxViTResNetEnsemble
from .kan import CNNKAN
from .pretrained import PRETRAINED_BUILDERS

# Standart (uçtan uca geri yayılımla eğitilen) modeller
STANDARD_MODELS = [
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
]

# SOM tabanlı hibrit modeller (özellik çıkarma + BMU sınıflandırma, ayrı eğitilir)
SOM_MODELS = ["cnn_som", "maxvit_som"]

ALL_MODELS = STANDARD_MODELS + SOM_MODELS


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name == "custom_cnn":
        return CustomCNN(num_classes=num_classes)
    if name == "cnn_kan":
        return CNNKAN(num_classes=num_classes)
    if name == "ensemble_maxvit_resnet":
        return MaxViTResNetEnsemble(num_classes=num_classes, pretrained=pretrained)
    if name in PRETRAINED_BUILDERS:
        return PRETRAINED_BUILDERS[name](num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Bilinmeyen model adı: {name}")
