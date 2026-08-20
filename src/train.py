"""
Eğitim döngüsü: hiperparametre random search + belirtilen epoch sayısıyla
final eğitim (Adam + StepLR(step=7, gamma=0.1), makale Bölüm 2.5).

Bu modül gerçek çoklu-saatlik eğitimi bu bulut ortamında ÇALIŞTIRMAZ (kullanıcı
tarafından kapsam GPU'lu kendi makinesinde çalıştırmak olarak belirlendi);
`tests/smoke_test.py` sadece birkaç mini-batch / 1 epoch ile kodun uçtan uca
çalıştığını doğrular.
"""
from __future__ import annotations

import copy
import gc
import itertools
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .models.registry import build_model


def _make_loader(dataset, indices, batch_size, shuffle):
    ds = Subset(dataset, indices) if indices is not None else dataset
    # num_workers=0: Windows'ta DataLoader worker'ları ayrı süreç olarak açılır
    # (spawn) ve her biri PyTorch/timm'i baştan yükleyip GB'larca fazladan RAM
    # tüketebilir. Veri setimiz küçük olduğu için tek süreç yeterince hızlı.
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


def _forward_loss(model, x, y, criterion, model_name: str):
    out = model(x)
    if model_name == "googlenet" and isinstance(out, tuple):
        # torchvision GoogLeNet eğitim modunda (main, aux1, aux2) döndürür
        main_out, aux1, aux2 = out
        loss = criterion(main_out, y) + 0.3 * criterion(aux1, y) + 0.3 * criterion(aux2, y)
        return main_out, loss
    if hasattr(out, "logits"):
        out = out.logits
    loss = criterion(out, y)
    return out, loss


@torch.no_grad()
def evaluate_loader(model, loader, device, model_name: str):
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        if hasattr(out, "logits"):
            out = out.logits
        all_logits.append(out.cpu())
        all_labels.append(y.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    proba = torch.softmax(logits, dim=1).numpy()
    preds = proba.argmax(axis=1)
    acc = float((preds == labels.numpy()).mean())
    return acc, preds, proba, labels.numpy()


def _sample_hyperparams(search_cfg: dict, seed: int, batch_size_cap: int | None = None):
    rng = random.Random(seed)
    batch_sizes = search_cfg["batch_size"]
    if batch_size_cap is not None:
        # bellek yoğun modeller için: cap'in üstündeki tüm batch size'ları cap'e indir,
        # sonra tekrarları kaldır (ör. [16,32] -> cap=8 -> [8])
        batch_sizes = sorted(set(min(bs, batch_size_cap) for bs in batch_sizes))
    grid = list(
        itertools.product(
            batch_sizes, search_cfg["learning_rate"], search_cfg["weight_decay"]
        )
    )
    rng.shuffle(grid)
    return grid[: search_cfg["n_trials"]]


def hyperparam_search(model_name, cfg, num_classes, train_ds, val_indices, train_indices, device, quick_epochs=3, pretrained=True):
    search_cfg = cfg["training"]["hyperparam_search"]
    batch_size_cap = cfg["training"].get("memory_safe_batch_size", {}).get(model_name)
    if batch_size_cap:
        print(f"[{model_name}] bellek-güvenli mod: batch_size <= {batch_size_cap} olarak sınırlandı", flush=True)
    candidates = _sample_hyperparams(search_cfg, cfg["seed"], batch_size_cap)
    best = None
    n_total = len(candidates)
    for trial_idx, (bs, lr, wd) in enumerate(candidates, start=1):
        t0 = time.time()
        print(f"[{model_name}]   deneme {trial_idx}/{n_total}: batch_size={bs}, lr={lr}, weight_decay={wd} ...", flush=True)
        model = build_model(model_name, num_classes, pretrained=pretrained).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss()
        train_loader = _make_loader(train_ds, train_indices, bs, shuffle=True)
        val_loader = _make_loader(train_ds, val_indices, bs, shuffle=False)

        model.train()
        for ep in range(quick_epochs):
            for step, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                _, loss = _forward_loss(model, x, y, criterion, model_name)
                loss.backward()
                opt.step()
                if step % 10 == 0:
                    print(f"[{model_name}]     deneme {trial_idx}/{n_total}, epoch {ep+1}/{quick_epochs}, batch {step}/{len(train_loader)}, loss={loss.item():.4f}", flush=True)

        val_acc, *_ = evaluate_loader(model, val_loader, device, model_name)
        dt = time.time() - t0
        print(f"[{model_name}]   deneme {trial_idx}/{n_total} bitti: val_acc={val_acc:.4f} ({dt:.1f}s)", flush=True)
        if best is None or val_acc > best["val_acc"]:
            best = {"batch_size": bs, "lr": lr, "weight_decay": wd, "val_acc": val_acc}

        # Bir sonraki denemeye geçmeden önce bu modelin belleğini serbest bırak
        # (art arda büyük modeller kurmak RAM'i şişirebilir, özellikle Windows'ta).
        del model, opt, train_loader, val_loader
        gc.collect()
    return best


def train_standard_model(model_name: str, cfg: dict, train_ds, num_classes: int, device, output_dir: Path, pretrained: bool = True):
    """Random search + final eğitim. train_ds tüm eğitim (train_split) verisidir;
    içeriden bir doğrulama (validation) alt kümesi ayrılır (README'de belirtildi)."""
    rng = np.random.default_rng(cfg["seed"])
    n = len(train_ds)
    idx = rng.permutation(n)
    val_size = max(1, int(0.15 * n))
    val_indices = idx[:val_size].tolist()
    train_indices = idx[val_size:].tolist()

    quick_epochs = cfg["training"]["hyperparam_search"].get("quick_epochs", 3)
    n_trials = cfg["training"]["hyperparam_search"].get("n_trials", 10)
    print(f"[{model_name}] hiperparametre random search çalışıyor ({n_trials} deneme x {quick_epochs} epoch)...")
    best_hp = hyperparam_search(model_name, cfg, num_classes, train_ds, val_indices, train_indices, device, quick_epochs=quick_epochs, pretrained=pretrained)
    print(f"[{model_name}] en iyi hiperparametreler: {best_hp}")

    epochs = cfg["training"]["epochs"].get(model_name, 20)
    step_cfg = cfg["training"]["step_lr"]
    patience = cfg["training"]["early_stopping"]["patience"]

    model = build_model(model_name, num_classes, pretrained=pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=best_hp["lr"], weight_decay=best_hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_cfg["step_size"], gamma=step_cfg["gamma"])
    criterion = nn.CrossEntropyLoss()

    train_loader = _make_loader(train_ds, train_indices, best_hp["batch_size"], shuffle=True)
    val_loader = _make_loader(train_ds, val_indices, best_hp["batch_size"], shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            _, loss = _forward_loss(model, x, y, criterion, model_name)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)
        scheduler.step()

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                if hasattr(out, "logits"):
                    out = out.logits
                val_loss += criterion(out, y).item() * x.size(0)
                n_val += x.size(0)
        val_loss /= max(n_val, 1)
        print(f"[{model_name}] epoch {epoch+1}/{epochs} train_loss={running_loss/len(train_indices):.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[{model_name}] early stopping (patience={patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"{model_name}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[{model_name}] checkpoint kaydedildi: {ckpt_path}")
    return model, best_hp
