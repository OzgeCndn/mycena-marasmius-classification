"""Table 5 (performans metrikleri) ve Table 6 (ki-kare anlamlılık testi) üretimi."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import chi_square_significance, compute_metrics

BASE_MODELS = [
    "googlenet",
    "mobilenetv3_large",
    "resnetv2_50",
    "efficientnet_b0",
    "efficientnetv2_m",
    "vgg19",
    "maxvit_s",
]
PROPOSED_MODELS = ["cnn_som", "cnn_kan", "maxvit_som", "ensemble_maxvit_resnet"]


def load_predictions(pred_dir: Path) -> dict:
    preds = {}
    for f in sorted(Path(pred_dir).glob("*.npz")):
        data = np.load(f, allow_pickle=True)
        proba = data["y_proba"] if "y_proba" in data and data["y_proba"].size > 0 else None
        preds[f.stem] = {
            "y_true": data["y_true"],
            "y_pred": data["y_pred"],
            "y_proba": proba,
        }
    return preds


def save_predictions(pred_dir: Path, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None):
    pred_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"y_true": y_true, "y_pred": y_pred}
    kwargs["y_proba"] = y_proba if y_proba is not None else np.array([])
    np.savez(pred_dir / f"{model_name}.npz", **kwargs)


def build_table5(predictions: dict, num_classes: int) -> pd.DataFrame:
    rows = []
    for name, d in predictions.items():
        m = compute_metrics(d["y_true"], d["y_pred"], d["y_proba"], num_classes)
        rows.append(
            {
                "Model": name,
                "Accuracy": round(m["accuracy"], 3),
                "Precision": round(m["precision"], 3),
                "Recall": round(m["recall"], 3),
                "F1-Score": round(m["f1"], 3),
                "Specificity": round(m["specificity"], 3),
                "MCC": round(m["mcc"], 3),
                "AUC (OvR)": round(m["auc_ovr"], 3) if m["auc_ovr"] is not None else "N/A",
            }
        )
    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return df


def build_table6(predictions: dict) -> pd.DataFrame:
    rows = []
    for proposed in PROPOSED_MODELS:
        if proposed not in predictions:
            continue
        pd_ = predictions[proposed]
        correct_p = int((pd_["y_true"] == pd_["y_pred"]).sum())
        total_p = len(pd_["y_true"])
        row = {"Model": proposed}
        for base in BASE_MODELS:
            if base not in predictions:
                row[base] = "N/A"
                continue
            bd = predictions[base]
            correct_b = int((bd["y_true"] == bd["y_pred"]).sum())
            total_b = len(bd["y_true"])
            res = chi_square_significance(correct_p, total_p, correct_b, total_b)
            marker = ""
            if res["significant"]:
                marker = " (proposed)" if res["favored"] == "a" else " (base)"
            row[base] = f"{res['p_value']:.3e}{marker}"
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(df: pd.DataFrame, out_dir: Path, basename: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{basename}.csv", index=False)
    with open(out_dir / f"{basename}.md", "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
