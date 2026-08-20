"""
Makaledeki "Custom CNN" mimarisi (Bölüm 2.3.1, Tablo 1 benzeri):
4 konvolüsyon bloğu (32 -> 64 -> 128 -> 256 filtre), her blokta
Conv2d(3x3) -> BatchNorm -> ReLU -> MaxPool(2x2).
224x224 girişte 4 kez 2x2 pooling sonrası uzamsal boyut 14x14 olur,
bu da makaledeki "14x14x256 = 50176 özellik" (CNN-SOM için) ile eşleşir.

Sınıflandırma başlığı: GlobalAvgPool -> FC(512, dropout 0.5) -> FC(256) -> FC(num_classes)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class CustomCNN(nn.Module):
    """num_classes=0 verilirse sadece özellik çıkarıcı olarak davranır (FC katmanları yok)."""

    def __init__(self, num_classes: int = 7, in_channels: int = 3):
        super().__init__()
        self.block1 = ConvBlock(in_channels, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 256)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        if num_classes > 0:
            self.fc1 = nn.Linear(256, 512)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(512, 256)
            self.fc_out = nn.Linear(256, num_classes)

    def conv_features(self, x: torch.Tensor) -> torch.Tensor:
        """4. blok sonrası uzamsal harita, GAP'ten ÖNCE (B, 256, 14, 14)."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

    def flat_features(self, x: torch.Tensor) -> torch.Tensor:
        """CNN-SOM için 50176-boyutlu düzleştirilmiş özellik vektörü."""
        feat = self.conv_features(x)
        return torch.flatten(feat, 1)

    def pooled_features(self, x: torch.Tensor) -> torch.Tensor:
        """256-boyutlu GAP sonrası özellik (CNN-KAN gövdesi için)."""
        feat = self.conv_features(x)
        return torch.flatten(self.gap(feat), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.pooled_features(x)
        if self.num_classes == 0:
            return feat
        h = self.dropout(torch.relu(self.fc1(feat)))
        h = torch.relu(self.fc2(h))
        return self.fc_out(h)
