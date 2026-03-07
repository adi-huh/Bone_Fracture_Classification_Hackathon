"""
evaluate.py — Test-set evaluation & CSV export
================================================

Computes:
  • Overall accuracy
  • Per-class precision / recall / F1
  • Macro F1-Score
  • Confusion matrix
  • AUC-ROC (one-vs-rest)
  • Training time, inference time per image, model size
  • 5-fold CV mean ± std (if available)

Exports everything to **final_results.csv** and updates the generalization
section of **model_performance_analysis.csv**.
"""

import os
import csv
import time
import json
import numpy as np

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

from data_loader import load_config, set_seed, get_dataloaders
from model import build_model

# ------------------------------------------------------------------ #
#  Device helper (same as train.py)                                   #
# ------------------------------------------------------------------ #

def get_device(config: dict) -> torch.device:
    requested = config.get("device", "auto")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


# ------------------------------------------------------------------ #
#  Core evaluation                                                    #
# ------------------------------------------------------------------ #

@torch.no_grad()
def run_evaluation(model, loader, device, num_classes):
    """Return all_labels, all_preds, all_probs, total_inference_time."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    total_time = 0.0

    for images, labels in loader:
        images = images.to(device)
        t0 = time.time()
        logits = model(images)
        total_time += time.time() - t0

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    return (
        np.array(all_labels),
        np.array(all_preds),
        all_probs,
        total_time,
    )


# ------------------------------------------------------------------ #
#  Compute all metrics                                                #
# ------------------------------------------------------------------ #

def compute_metrics(labels, preds, probs, class_names, inference_time,
                    training_time, model_path, cv_results=None):
    """
    Returns a list of dicts ready to write as CSV rows in
    ``final_results.csv``.
    """
    num_classes = len(class_names)
    num_images = len(labels)

    # Overall accuracy
    overall_acc = accuracy_score(labels, preds) * 100.0

    # Per-class P / R / F1
    prec, rec, f1, sup = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    # AUC-ROC
    try:
        if num_classes == 2:
            auc = roc_auc_score(labels, probs[:, 1])
        else:
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    # Model size
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0.0

    # Inference time
    inf_per_img_ms = (inference_time / num_images) * 1000.0

    # ---- Build rows ---- #
    header = ["metric_name", "overall_value"] + [f"{cn}_value" for cn in class_names] + ["interpretation"]
    rows = []

    def _row(name, overall, per_class=None, interp=""):
        r = {"metric_name": name, "overall_value": f"{overall}"}
        for i, cn in enumerate(class_names):
            key = f"{cn}_value"
            r[key] = f"{per_class[i]:.4f}" if per_class is not None else ""
        r["interpretation"] = interp
        return r

    rows.append(_row("Accuracy (%)", f"{overall_acc:.2f}",
                      interp="Overall classification accuracy on test set"))

    rows.append(_row("Precision", f"{np.mean(prec):.4f}", prec,
                      "Positive-predictive value per class"))
    rows.append(_row("Recall", f"{np.mean(rec):.4f}", rec,
                      "Sensitivity / true-positive rate per class"))
    rows.append(_row("F1-Score", f"{macro_f1:.4f}", f1,
                      "Harmonic mean of precision and recall"))
    rows.append(_row("Macro F1", f"{macro_f1:.4f}",
                      interp="Unweighted mean of per-class F1"))
    rows.append(_row("AUC-ROC (OVR)", f"{auc:.4f}",
                      interp="One-vs-rest macro AUC"))

    # Confusion matrix (flattened as a single row for convenience)
    cm_flat = ";".join([",".join(map(str, row)) for row in cm])
    rows.append({"metric_name": "Confusion Matrix",
                 "overall_value": cm_flat,
                 **{f"{cn}_value": "" for cn in class_names},
                 "interpretation": "Rows=true, Cols=predicted; semicolon-separated rows"})

    rows.append(_row("Training Time (s)", f"{training_time:.1f}",
                      interp="Total training wall-clock time"))
    rows.append(_row("Inference Time (ms/img)", f"{inf_per_img_ms:.2f}",
                      interp="Average inference latency per image"))
    rows.append(_row("Model Size (MB)", f"{model_size_mb:.2f}",
                      interp="Checkpoint file size on disk"))

    # Cross-validation
    if cv_results is not None:
        mean_acc, std_acc = cv_results["mean_acc"], cv_results["std_acc"]
        mean_f1, std_f1   = cv_results["mean_f1"],  cv_results["std_f1"]
        rows.append(_row("CV Accuracy (mean±std)", f"{mean_acc:.2f}±{std_acc:.2f}",
                          interp="5-fold stratified cross-validation accuracy"))
        rows.append(_row("CV F1 (mean±std)", f"{mean_f1:.4f}±{std_f1:.4f}",
                          interp="5-fold stratified cross-validation macro F1"))
    else:
        rows.append(_row("CV Accuracy (mean±std)", "N/A",
                          interp="Run `python train.py --cv` first"))
        rows.append(_row("CV F1 (mean±std)", "N/A",
                          interp="Run `python train.py --cv` first"))

    return header, rows, overall_acc, cm


# ------------------------------------------------------------------ #
#  Write CSV                                                          #
# ------------------------------------------------------------------ #

def write_final_results_csv(header, rows, path="final_results.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[evaluate] Saved → {path}")


def update_performance_csv(test_acc, train_best_acc, cv_str,
                           path="model_performance_analysis.csv"):
    """Append test-set and CV info to the generalization section."""
    if not os.path.exists(path):
        return
    with open(path, "r") as f:
        content = f.read()
    content = content.replace("Test Accuracy,TBD (run evaluate.py)",
                              f"Test Accuracy,{test_acc:.2f}")
    delta = train_best_acc - test_acc if train_best_acc else 0.0
    content = content.replace("Train/Test Accuracy Delta,TBD",
                              f"Train/Test Accuracy Delta,{delta:.2f}")
    content = content.replace("CV Mean ± Std,TBD (run train.py --cv)",
                              f"CV Mean ± Std,{cv_str}")
    with open(path, "w") as f:
        f.write(content)
    print(f"[evaluate] Updated → {path}")


# ------------------------------------------------------------------ #
#  Pretty-print confusion matrix                                      #
# ------------------------------------------------------------------ #

def print_confusion_matrix(cm, class_names):
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/confusion_matrix.png", dpi=150)
        plt.close()
        print("[evaluate] Confusion matrix plot → results/confusion_matrix.png")
    except ImportError:
        pass


# ------------------------------------------------------------------ #
#  CLI entry point                                                    #
# ------------------------------------------------------------------ #

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate bone fracture classifier on test set")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model.pth (default: checkpoints/model.pth)")
    parser.add_argument("--cv-results", default=None,
                        help="Path to JSON with CV results (optional)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device(config)

    ckpt_path = args.checkpoint or os.path.join(
        config.get("checkpoint_dir", "checkpoints"), "model.pth"
    )

    # Load data (only test split is used)
    _, _, test_loader, class_names = get_dataloaders(config)
    num_classes = len(class_names)
    config["num_classes"] = num_classes

    # Load model
    model = build_model(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    training_time = ckpt.get("config", {}).get("training_time", 0.0)
    # Try to recover training time from the performance CSV
    if training_time == 0.0:
        perf_csv = "model_performance_analysis.csv"
        if os.path.exists(perf_csv):
            with open(perf_csv) as f:
                for line in f:
                    if line.startswith("Training Time"):
                        try:
                            training_time = float(line.split(",")[1])
                        except Exception:
                            pass

    print(f"[evaluate] Loaded checkpoint from {ckpt_path}")
    print(f"[evaluate] Classes: {class_names}")

    # Evaluate
    labels, preds, probs, inf_time = run_evaluation(
        model, test_loader, device, num_classes
    )

    # Optional CV results
    cv_results = None
    if args.cv_results and os.path.exists(args.cv_results):
        with open(args.cv_results) as f:
            cv_results = json.load(f)

    header, rows, test_acc, cm = compute_metrics(
        labels, preds, probs, class_names, inf_time,
        training_time, ckpt_path, cv_results,
    )

    # Save
    write_final_results_csv(header, rows)
    print_confusion_matrix(cm, class_names)

    # Try to get best training acc from checkpoint
    best_train_acc = ckpt.get("val_accuracy", 0.0)  # rough proxy
    cv_str = "N/A"
    if cv_results:
        cv_str = f"{cv_results['mean_acc']:.2f} ± {cv_results['std_acc']:.2f}"
    update_performance_csv(test_acc, best_train_acc, cv_str)

    # Print summary
    print("\n" + "=" * 60)
    print("  TEST SET RESULTS")
    print("=" * 60)
    from sklearn.metrics import classification_report as cr
    print(cr(labels, preds, target_names=class_names, digits=4, zero_division=0))
    print(f"Overall Accuracy : {test_acc:.2f}%")
    print(f"Macro F1         : {f1_score(labels, preds, average='macro', zero_division=0):.4f}")
    print(f"Inference        : {(inf_time / len(labels))*1000:.2f} ms/image")
    print("=" * 60)


if __name__ == "__main__":
    main()
