#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation-Guided Reporting + Lightweight Deep Learning for Pediatric CXR Pneumonia
-------------------------------------------------------------------------------------
PSEUDOCODE TEMPLATE (.py) — ready to adapt into a working project.


Usage:
    python seg_report_cxr_pipeline_pseudocode.py \
        --data_root "/path/to/chest_xray" \
        --out_dir "./outputs" \
        --epochs 20 \
        --batch_size 32 \
        --img_size 224

Directory structure expected (Kaggle CXR):
    chest_xray/
      train/
        NORMAL/
        PNEUMONIA/
      val/
        NORMAL/
        PNEUMONIA/
      test/
        NORMAL/
        PNEUMONIA/

Key design choices (align with manuscript):
- Backbone: MobileNet (ImageNet init), unfreeze last ~10 layers.
- Head: Dense(256)->Dropout(0.45)->Dense(128)->Dropout(0.30)->Dense(64)->sigmoid.
- Augmentation: rotations ±20°, shifts up to 0.25, zoom (0.8–1.2), horizontal/vertical flips (moderate).
- Priority: recall (sensitivity) for triage; interpretability via segmentation-guided reporting.
"""

import os
import sys
import json
import time
import math
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, List, Optional

import numpy as np

# Optional deps (install if you make this fully runnable)
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.applications import MobileNet
except Exception as e:
    tf = None
    ImageDataGenerator = None
    layers = None
    models = None
    EarlyStopping = None
    ModelCheckpoint = None
    MobileNet = None

try:
    from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
except Exception:
    confusion_matrix = None
    precision_recall_curve = None
    roc_curve = None
    auc = None

import matplotlib.pyplot as plt  # For figures. Do not rely on seaborn.

# ----------------------------
# Configuration & Reproducibility
# ----------------------------

@dataclass
class Config:
    data_root: str
    out_dir: str = "./outputs"
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-3
    unfreeze_last_k: int = 10
    seed: int = 42
    class_mode: str = "binary"  # for Keras directory iterator
    # Reporting/Segmentation
    enable_reporting: bool = True
    # Placeholder: external vision-language provider for segmentation masks/text
    seg_provider: str = "EXTERNAL_API_OR_LOCAL_MODEL"
    # Save flags
    save_model: bool = True
    save_plots: bool = True


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # local import to avoid global failure
        tf.random.set_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass


# ----------------------------
# Data Pipeline
# ----------------------------

def build_datagens(cfg: Config):
    """Create ImageDataGenerators for train/val/test. Aligns with §2.2."""
    if ImageDataGenerator is None:
        raise ImportError("TensorFlow/Keras not available. Install tensorflow to use data generators.")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.25,
        height_shift_range=0.25,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    test_val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(cfg.data_root, "train"),
        target_size=(cfg.img_size, cfg.img_size),
        batch_size=cfg.batch_size,
        class_mode=cfg.class_mode,
        subset=None,
        shuffle=True
    )

    val_gen = test_val_datagen.flow_from_directory(
        os.path.join(cfg.data_root, "val"),
        target_size=(cfg.img_size, cfg.img_size),
        batch_size=cfg.batch_size,
        class_mode=cfg.class_mode,
        shuffle=False
    )

    test_gen = test_val_datagen.flow_from_directory(
        os.path.join(cfg.data_root, "test"),
        target_size=(cfg.img_size, cfg.img_size),
        batch_size=cfg.batch_size,
        class_mode=cfg.class_mode,
        shuffle=False
    )
    return train_gen, val_gen, test_gen


# ----------------------------
# Model
# ----------------------------

def build_model(cfg: Config):
    """MobileNet backbone + compact dense head. Unfreeze last k layers. Aligns with §2.3."""
    if MobileNet is None or models is None or layers is None:
        raise ImportError("TensorFlow/Keras not available. Install tensorflow to build the model.")

    base = MobileNet(weights="imagenet", include_top=False, pooling="avg",
                     input_shape=(cfg.img_size, cfg.img_size, 3))

    # Unfreeze last k layers
    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers[-cfg.unfreeze_last_k:]:
        layer.trainable = True

    x = layers.Dense(256, activation="relu")(base.output)
    x = layers.Dropout(0.45)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=base.input, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=cfg.lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


# ----------------------------
# Training
# ----------------------------

def train_model(cfg: Config, model, train_gen, val_gen):
    """Train with early stopping and checkpointing. Aligns with §2.3."""
    callbacks = []
    if EarlyStopping is not None:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True))
    ckpt_path = os.path.join(cfg.out_dir, "best_model.keras")
    if ModelCheckpoint is not None:
        callbacks.append(ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_loss"))

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=cfg.epochs,
        callbacks=callbacks
    )
    return history, ckpt_path


# ----------------------------
# Evaluation & Figures
# ----------------------------

def compute_confusion_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, Any]:
    """Compute CM and derived metrics. Matches paper’s reported metrics section."""
    if confusion_matrix is None:
        raise ImportError("scikit-learn not available. Install scikit-learn to compute metrics.")

    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])  # [[TN, FP],[FN, TP]]
    TN, FP, FN, TP = cm.ravel()
    total = cm.sum()

    acc = (TP + TN) / total
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    spec = TN / (TN + FP) if (TN + FP) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    npv = TN / (TN + FN) if (TN + FN) else 0.0
    fpr = FP / (FP + TN) if (FP + TN) else 0.0
    fnr = FN / (FN + TP) if (FN + TP) else 0.0
    bal_acc = 0.5 * (rec + spec)

    return {
        "cm": cm,
        "TN": TN, "FP": FP, "FN": FN, "TP": TP,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "npv": npv,
        "fpr": fpr,
        "fnr": fnr,
        "balanced_accuracy": bal_acc
    }


def precision_recall_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    if precision_recall_curve is None or auc is None:
        raise ImportError("scikit-learn not available. Install scikit-learn to compute PR AUC.")
    p, r, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(r, p)
    return p, r, pr_auc


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    if roc_curve is None or auc is None:
        raise ImportError("scikit-learn not available. Install scikit-learn to compute ROC AUC.")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)
    return fpr, tpr, roc_auc_val


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, dict]:
    """Compute ECE and bin stats (for reliability diagram)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    stats = []
    for b in range(n_bins):
        idx = bin_ids == b
        if np.sum(idx) == 0:
            stats.append({"bin": b, "conf": np.nan, "acc": np.nan, "count": 0})
            continue
        conf = np.mean(y_prob[idx])
        acc = np.mean((y_prob[idx] >= 0.5) == (y_true[idx] == 1))
        w = np.sum(idx) / len(y_prob)
        ece += w * abs(acc - conf)
        stats.append({"bin": b, "conf": float(conf), "acc": float(acc), "count": int(np.sum(idx))})
    return float(ece), {"bins": bins.tolist(), "stats": stats}


def plot_confusion_matrix_percent(cm: np.ndarray, out_path: str, title: str = "Confusion Matrix (Percent of Total)"):
    """Save a confusion matrix image with percentages only, as per journal style."""
    total = cm.sum()
    pct = cm / total * 100.0
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    # Simple greyscale for print-friendliness
    im = ax.imshow(pct, cmap="Greys", vmin=0, vmax=100)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred NORMAL", "Pred PNEUMONIA"])
    ax.set_yticklabels(["True NORMAL", "True PNEUMONIA"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{pct[i,j]:.1f}%", ha="center", va="center",
                    color="black", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(p: np.ndarray, r: np.ndarray, auc_val: float, out_path: str, title: str = "Precision–Recall Curve"):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(r, p, lw=2, label=f"PR AUC = {auc_val:.2f}")
    ax.fill_between(r, p, alpha=0.15)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_val: float, out_path: str, title: str = "ROC Curve"):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC AUC = {auc_val:.2f}")
    ax.plot([0,1], [0,1], "--", color="grey", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_diagram(ece_info: dict, out_path: str, title: str = "Reliability Diagram"):
    bins = ece_info["bins"]
    stats = ece_info["stats"]
    centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    conf = [s["conf"] for s in stats]
    acc = [s["acc"] for s in stats]

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.plot([0,1], [0,1], "--", color="grey", label="Perfect calibration")
    ax.plot(centers, conf, "-o", label="Avg confidence")
    ax.plot(centers, acc, "-o", label="Empirical accuracy")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Segmentation-Guided Reporting (Placeholder)
# ----------------------------

def segmentation_guided_reporting(image_path: str, provider: str) -> Dict[str, Any]:
    """
    Return anatomy-aligned masks and a short textual narrative.
    This is a placeholder — implement using your chosen provider (e.g., local segmentation model,
    foundation VLM, or rules). Aligns with §2.4 in the paper.

    Expected return schema:
    {
        "items": [
            {"label": "Right Lung", "box_2d": [y0, x0, y1, x1], "mask": <binary_mask_np_or_png_b64>,
             "finding": "diffuse reticulogranular opacities"},
            ...
        ],
        "impression": "Short radiology-style prose summarizing salient features."
    }
    """
    # TODO: Replace with real implementation. For now, return a stub.
    return {
        "items": [
            {"label": "Right Lung", "box_2d": [100, 120, 900, 880], "mask": None,
             "finding": "patchy air-space opacities with visible air bronchograms"},
            {"label": "Left Lung", "box_2d": [110, 130, 900, 880], "mask": None,
             "finding": "diffuse reticulogranular pattern, perihilar accentuation"},
            {"label": "Heart", "box_2d": [400, 350, 780, 650], "mask": None,
             "finding": "cardiomediastinal silhouette within normal size"}
        ],
        "impression": "Bilateral air-space disease, right greater than left; features consistent with pneumonia."
    }


def generate_reports_for_directory(cfg: Config, split_dir: str, out_dir: str):
    """
    Iterate images in a directory and generate overlays + narrative JSON per image.
    Store outputs to disk for audit and qualitative results (§3.2).
    """
    os.makedirs(out_dir, exist_ok=True)
    image_paths = []
    for root, _, files in os.walk(split_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, f))
    for img_path in image_paths:
        report = segmentation_guided_reporting(img_path, cfg.seg_provider)
        # Save JSON
        base = Path(img_path).stem
        json_path = os.path.join(out_dir, f"{base}_report.json")
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)


# ----------------------------
# End-to-end Orchestration
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to chest_xray root folder")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--unfreeze_last_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_reporting", action="store_true", help="Disable segmentation-guided reporting")

    args = parser.parse_args()
    cfg = Config(
        data_root=args.data_root,
        out_dir=args.out_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        unfreeze_last_k=args.unfreeze_last_k,
        seed=args.seed,
        enable_reporting=(not args.no_reporting)
    )

    # Setup
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    # Data
    train_gen, val_gen, test_gen = build_datagens(cfg)

    # Model
    model = build_model(cfg)

    # Train
    history, ckpt_path = train_model(cfg, model, train_gen, val_gen)

    # Evaluate on test
    # Note: test_gen.class_indices -> label mapping; ensure 0 = NORMAL, 1 = PNEUMONIA in your loaders.
    y_true = np.array(test_gen.classes)  # 0/1
    y_prob = model.predict(test_gen).ravel()

    metrics = compute_confusion_metrics(y_true, y_prob, thr=0.5)
    print("Confusion matrix [[TN, FP],[FN, TP]]:\n", metrics["cm"])
    print({k: v for k, v in metrics.items() if k not in ("cm",)})

    # Plots
    if cfg.save_plots:
        plot_confusion_matrix_percent(metrics["cm"],
                                      os.path.join(cfg.out_dir, "confusion_matrix_percent.png"))
        # PR
        try:
            p, r, pr_auc = precision_recall_auc(y_true, y_prob)
            plot_pr_curve(p, r, pr_auc, os.path.join(cfg.out_dir, "pr_curve.png"))
        except Exception as e:
            print("PR curve skipped:", e)
        # ROC
        try:
            fpr, tpr, roc_auc_val = roc_auc(y_true, y_prob)
            plot_roc_curve(fpr, tpr, roc_auc_val, os.path.join(cfg.out_dir, "roc_curve.png"))
        except Exception as e:
            print("ROC curve skipped:", e)
        # Calibration
        try:
            ece, info = expected_calibration_error(y_true, y_prob, n_bins=10)
            print(f"ECE (10 bins): {ece:.4f}")
            plot_reliability_diagram(info, os.path.join(cfg.out_dir, "reliability_diagram.png"))
            with open(os.path.join(cfg.out_dir, "calibration_bins.json"), "w") as f:
                json.dump(info, f, indent=2)
        except Exception as e:
            print("Calibration skipped:", e)

    # Save model
    if cfg.save_model:
        model.save(os.path.join(cfg.out_dir, "final_model.keras"))

    # Reporting on test set (optional; can be slow if using external provider)
    if cfg.enable_reporting:
        test_dir = os.path.join(cfg.data_root, "test")
        rep_out_dir = os.path.join(cfg.out_dir, "reports_test")
        generate_reports_for_directory(cfg, test_dir, rep_out_dir)

    # Save run config & history
    run_meta = {
        "config": asdict(cfg),
        "class_indices": getattr(train_gen, "class_indices", None),
        "train_samples": getattr(train_gen, "samples", None),
        "val_samples": getattr(val_gen, "samples", None),
        "test_samples": getattr(test_gen, "samples", None),
    }
    with open(os.path.join(cfg.out_dir, "run_config.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    try:
        # History object contains metrics across epochs (if Keras fit ran)
        hist = getattr(history, "history", {})
        with open(os.path.join(cfg.out_dir, "train_history.json"), "w") as f:
            json.dump(hist, f, indent=2)
    except Exception:
        pass

    print("Pipeline complete. Outputs in:", cfg.out_dir)


if __name__ == "__main__":
    main()
