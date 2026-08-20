"""
CNN-KAN: Custom CNN gövdesi + Kolmogorov-Arnold Network (KAN) sınıflandırma katmanı.

Makale, KAN katmanını "sine-based transformation with learnable activation
functions" olarak tanımlıyor ama tam mimariyi (temel fonksiyon sayısı, frekans
parametrelendirmesi vb.) vermiyor ve orijinal kod yayımlanmamış. Aşağıdaki
`SineKANLayer`, Kolmogorov-Arnold ayrışım fikrini ("her çıktı, girdilerin
öğrenilebilir tek-değişkenli fonksiyonlarının toplamıdır") sinüs tabanlı,
öğrenilebilir genlik/frekans/faz parametreleriyle uygulayan makul bir
yorumdur -- makale yazarlarının birebir implementasyonu ile aynı olduğu
garanti edilemez (bkz. README "Sınırlamalar").
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .custom_cnn import CustomCNN


class SineKANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_basis: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_basis = n_basis
        # her (out, in) çifti için n_basis adet öğrenilebilir sinüs bileşeni
        self.freq = nn.Parameter(torch.linspace(0.5, 4.0, n_basis))
        self.amplitude = nn.Parameter(torch.randn(out_features, in_features, n_basis) * 0.05)
        self.phase = nn.Parameter(torch.zeros(out_features, in_features, n_basis))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_features) -> (B, 1, in_features, 1)
        x_e = x.unsqueeze(1).unsqueeze(-1)
        freq = self.freq.view(1, 1, 1, -1)
        basis = torch.sin(x_e * freq + self.phase.unsqueeze(0))  # (B, out, in, n_basis)
        weighted = basis * self.amplitude.unsqueeze(0)
        out = weighted.sum(dim=(2, 3)) + self.bias
        return out


class CNNKAN(nn.Module):
    def __init__(self, num_classes: int = 7, hidden_dim: int = 256, n_basis: int = 8):
        super().__init__()
        self.backbone = CustomCNN(num_classes=0)  # 256-d GAP özellik
        self.kan1 = SineKANLayer(256, hidden_dim, n_basis)
        self.kan2 = SineKANLayer(hidden_dim, num_classes, n_basis)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        h = self.act(self.kan1(feat))
        return self.kan2(h)
