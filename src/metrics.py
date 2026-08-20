"""
Değerlendirme metrikleri ve istatistiksel anlamlılık testi.

Makale Bölüm 2.6 / Formül 12-18: Accuracy, Precision, Recall/Sensitivity,
Specificity, F1, MCC (çok sınıflı), AUC (One-vs-Rest) -- hepsi 7 sınıf
üzerinde makro-ortalama.

Table 6 (ki-kare anlamlılık testi): Makale, "karışıklık matrislerinden
oluşturulan kontenjans tablosu" üzerinde ki-kare testi uyguladığını
belirtiyor ama tam kontenjans tablosu tanımını vermiyor. Burada, her model
çifti için "doğru sınıflandırılan" / "yanlış sınıflandırılan" örnek
sayılarından 2x2 kontenjans tablosu kurup `scipy.stats.chi2_contingency`
uygulayan, literatürde yaygın olan bir yorum kullanılmıştır (iki modelin
doğruluk oranlarının şans eseri farklı olup olmadığını test eder).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None, num_classes: int) -> dict:
    acc = float(np.mean(y_true == y_pred))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities = []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    specificity = float(np.mean(specificities))

    auc = None
    if y_proba is not None:
        try:
            auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=list(range(num_classes))))
        except ValueError:
            auc = None

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
        "auc_ovr": auc,
        "confusion_matrix": cm,
    }


def chi_square_significance(correct_a: int, total_a: int, correct_b: int, total_b: int) -> dict:
    """İki modelin doğruluk oranlarını 2x2 kontenjans tablosuyla karşılaştırır."""
    table = np.array(
        [
            [correct_a, total_a - correct_a],
            [correct_b, total_b - correct_b],
        ]
    )
    chi2, p, dof, _ = chi2_contingency(table, correction=True)
    acc_a = correct_a / total_a
    acc_b = correct_b / total_b
    favored = "a" if acc_a > acc_b else ("b" if acc_b > acc_a else "tie")
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof), "favored": favored, "significant": bool(p < 0.05)}
