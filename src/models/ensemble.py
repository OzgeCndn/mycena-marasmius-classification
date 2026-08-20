"""
Ensemble (MaxViT-Small - ResNetV2-50), makale Bölüm 2.4.4:
İki modelden paralel özellik çıkarma -> concat -> FC(2816->512) -> ReLU ->
Dropout(0.5) -> FC(512->num_classes).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .pretrained import build_feature_extractor


class MaxViTResNetEnsemble(nn.Module):
    def __init__(self, num_classes: int = 7, pretrained: bool = True, fc1: int = 512, dropout: float = 0.5):
        super().__init__()
        self.maxvit, maxvit_dim = build_feature_extractor("maxvit_s", pretrained)
        self.resnet, resnet_dim = build_feature_extractor("resnetv2_50", pretrained)
        concat_dim = maxvit_dim + resnet_dim
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, fc1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc1, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.maxvit(x)
        f2 = self.resnet(x)
        f = torch.cat([f1, f2], dim=1)
        return self.classifier(f)
