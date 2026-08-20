"""
Kendinden Organize Eden Harita (Self-Organizing Map).

Makale "MiniSom" kütüphanesini kullandığını belirtiyor. Bu bulut ortamında
`minisom` paketinin derlenmesi (pip build) başarısız oldu (setuptools/py3.12
uyumsuzluğu), bu yüzden aynı algoritmayı (Gaussian komşuluk fonksiyonu,
online/random eğitim, BMU = en yakın ağırlık vektörü) uygulayan bağımsız,
NumPy tabanlı minimal bir SOM implementasyonu yazıldı.

Kendi makinenizde `pip install minisom` çalışıyorsa aşağıdaki `MiniSomCompatible`
sınıfı yerine gerçek `minisom.MiniSom`'u kullanabilirsiniz (API neredeyse aynı):
    from minisom import MiniSom
    som = MiniSom(x, y, input_len, sigma=sigma, learning_rate=lr)
    som.random_weights_init(data)
    som.train_random(data, num_iteration)
    som.winner(x)  # -> (i, j)
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


class SimpleSOM:
    def __init__(self, x: int, y: int, input_len: int, sigma: float = 1.0, learning_rate: float = 0.5, seed: int = 42):
        self.x, self.y, self.input_len = x, y, input_len
        self.sigma0 = sigma
        self.lr0 = learning_rate
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, 0.1, size=(x, y, input_len)).astype(np.float32)
        # grid koordinatları, komşuluk hesaplamak için
        gx, gy = np.meshgrid(np.arange(x), np.arange(y), indexing="ij")
        self._grid = np.stack([gx, gy], axis=-1).astype(np.float32)  # (x, y, 2)

    def random_weights_init(self, data: np.ndarray) -> None:
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(data), size=(self.x, self.y))
        self.weights = data[idx].astype(np.float32)

    def winner(self, sample: np.ndarray) -> tuple[int, int]:
        d = np.linalg.norm(self.weights - sample.astype(np.float32), axis=-1)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        return int(i), int(j)

    def train_random(self, data: np.ndarray, num_iteration: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        n = len(data)
        for t in range(num_iteration):
            sample = data[rng.integers(0, n)]
            bmu_i, bmu_j = self.winner(sample)

            # zamanla azalan öğrenme oranı / komşuluk yarıçapı
            frac = t / max(num_iteration, 1)
            lr = self.lr0 * np.exp(-frac)
            sigma = max(self.sigma0 * np.exp(-frac), 1e-3)

            dist2 = ((self._grid[..., 0] - bmu_i) ** 2 + (self._grid[..., 1] - bmu_j) ** 2)
            neighborhood = np.exp(-dist2 / (2 * sigma**2))  # (x, y)

            delta = sample.astype(np.float32) - self.weights
            self.weights += lr * neighborhood[..., None] * delta


class SOMClassifier:
    """SOM'u BMU-çoğunluk-oyu ile sınıflandırıcıya çevirir (makaledeki
    'Best Matching Unit (BMU) classification approach')."""

    def __init__(self, grid_size: tuple[int, int], input_len: int, sigma: float, learning_rate: float, iterations: int, seed: int = 42):
        self.grid_size = grid_size
        self.iterations = iterations
        self.som = SimpleSOM(grid_size[0], grid_size[1], input_len, sigma, learning_rate, seed)
        self.node_labels: dict[tuple[int, int], int] = {}
        self.default_label = 0

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.som.random_weights_init(features)
        self.som.train_random(features, self.iterations)

        node_votes: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for f, y in zip(features, labels):
            node_votes[self.som.winner(f)][int(y)] += 1
        for node, votes in node_votes.items():
            self.node_labels[node] = votes.most_common(1)[0][0]
        self.default_label = int(Counter(labels.tolist()).most_common(1)[0][0])

    def predict(self, features: np.ndarray) -> np.ndarray:
        preds = []
        for f in features:
            node = self.som.winner(f)
            preds.append(self.node_labels.get(node, self.default_label))
        return np.array(preds)
