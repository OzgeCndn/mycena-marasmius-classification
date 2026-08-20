"""
Makaledeki önceden eğitilmiş (transfer learning) mimariler.

torchvision'da bulunanlar: GoogleNet (Inception v1), MobileNetV3-Large, VGG19,
EfficientNet-B0.
torchvision'da BULUNMAYANLAR (ResNetV2-50, EfficientNetV2-M, MaxViT-Small) için
`timm` kütüphanesi kullanılıyor -- bunlar timm'de birebir isimleriyle mevcut:
`resnetv2_50`, `efficientnetv2_m`, `maxvit_small_tf_224`.

NOT: torchvision'ın `maxvit_t` (tiny) modeli MaxViT-Small ile AYNI DEĞİL; makale
"MaxViT-S" dediği için timm'in `maxvit_small_tf_224` varyantı tercih edildi.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
from torchvision import models


def build_googlenet(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.GoogLeNet_Weights.DEFAULT if pretrained else None
    m = models.googlenet(weights=weights, aux_logits=True)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    if m.aux1 is not None:
        m.aux1.fc2 = nn.Linear(m.aux1.fc2.in_features, num_classes)
    if m.aux2 is not None:
        m.aux2.fc2 = nn.Linear(m.aux2.fc2.in_features, num_classes)
    return m


def build_mobilenetv3_large(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    m = models.mobilenet_v3_large(weights=weights)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, num_classes)
    return m


def build_vgg19(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.VGG19_Weights.DEFAULT if pretrained else None
    m = models.vgg19(weights=weights)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, num_classes)
    return m


def build_efficientnet_b0(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    m = models.efficientnet_b0(weights=weights)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, num_classes)
    return m


def build_resnetv2_50(num_classes: int, pretrained: bool = True) -> nn.Module:
    return timm.create_model("resnetv2_50", pretrained=pretrained, num_classes=num_classes)


def build_efficientnetv2_m(num_classes: int, pretrained: bool = True) -> nn.Module:
    # timm'de "efficientnetv2_m"in ön-eğitimli ağırlığı yok; ImageNet üzerinde
    # eğitilmiş varyant "tf_efficientnetv2_m" ismiyle mevcut.
    model_name = "tf_efficientnetv2_m" if pretrained else "efficientnetv2_m"
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def build_maxvit_s(num_classes: int, pretrained: bool = True) -> nn.Module:
    return timm.create_model("maxvit_small_tf_224", pretrained=pretrained, num_classes=num_classes)


def build_feature_extractor(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Ensemble / *-SOM modelleri için sınıflandırma başlığı olmadan (num_classes=0)
    özellik çıkarıcı döndürür. (module, feature_dim) tuple'ı verir."""
    if name == "maxvit_s":
        m = timm.create_model("maxvit_small_tf_224", pretrained=pretrained, num_classes=0)
        feat_dim = m.num_features
    elif name == "resnetv2_50":
        m = timm.create_model("resnetv2_50", pretrained=pretrained, num_classes=0)
        feat_dim = m.num_features
    else:
        raise ValueError(f"Bilinmeyen özellik çıkarıcı: {name}")
    return m, feat_dim


PRETRAINED_BUILDERS = {
    "googlenet": build_googlenet,
    "mobilenetv3_large": build_mobilenetv3_large,
    "vgg19": build_vgg19,
    "efficientnet_b0": build_efficientnet_b0,
    "resnetv2_50": build_resnetv2_50,
    "efficientnetv2_m": build_efficientnetv2_m,
    "maxvit_s": build_maxvit_s,
}
