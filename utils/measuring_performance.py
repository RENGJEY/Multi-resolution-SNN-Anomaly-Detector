import itertools
from collections.abc import Iterable
import numpy as np
from bokeh.io import export_svgs
from bokeh.plotting import figure, show
from bokeh.models import (
    Band,
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    NumeralTickFormatter,
)
from bokeh.models.annotations import Label
import matplotlib.pyplot as plt
import seaborn as sns

def get_prediction(score, threshold=0.5):
    return np.where(score >= threshold, 1, 0)


def get_histogram(score, bins=30):
    hist, edges = np.histogram(score, bins=bins)
    percent = list(map(lambda x: x, hist / hist.sum()))
    alpha = hist / hist.sum() + 0.5 * (1.0 - np.max(hist) / hist.sum())

    histogram = dict(
        count=hist, percent=percent, left=edges[:-1], right=edges[1:], alpha=alpha
    )
    histogram["interval"] = [
        f"{left:.2f} to {right:.2f}"
        for left, right in zip(histogram["left"], histogram["right"])
    ]
    return histogram


def plot_confusion_matrix(conf_mat, model_name=None, file_name=None):
    if model_name is None:
        model_name = ""
    else:
        model_name += ": "

    plt.figure(figsize=(5, 4))
    ax = sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Greys",
        cbar=False,
        xticklabels=["False(normal)", "True(abnormal)"],
        yticklabels=["False(normal)", "True(abnormal)"],
        linewidths=1.5,
        linecolor="white"
    )

    ax.set_xlabel("Predicted", labelpad=10)
    ax.set_ylabel("Actual", labelpad=10) 
    ax.set_title(f"{model_name}Confusion Matrix", pad=15)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight', dpi=300)

    plt.show()


def plot_best_threshold_exploration(
    errors, model_name="Model", save_path=None
):

    thresholds = errors[:, 0]
    false_negative = errors[:, 1]
    false_positive = errors[:, 2]

    plt.figure(figsize=(10, 6))

    # False Negative
    plt.plot(thresholds, false_negative, color="crimson", label="False Negative", marker='o', markersize=5, markerfacecolor='white')
    # False Positive
    plt.plot(thresholds, false_positive, color="indigo", label="False Positive", marker='o', markersize=5, markerfacecolor='white')

    plt.title(f"{model_name}: Best Threshold Exploration", fontsize=14, fontweight='bold')
    plt.xlabel("Reconstruction Error")
    plt.ylabel("# Samples")
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()


def plot_threshold_range_exploration(
    stack, test_labels, threshold_min=2.0, threshold_max=7.0, model_name="Model", file_name=None
):

    normal_idx = stack[test_labels == 0][:, 0]
    normal_error = stack[test_labels == 0][:, 1]
    abnormal_idx = stack[test_labels == 1][:, 0]
    abnormal_error = stack[test_labels == 1][:, 1]

    plt.figure(figsize=(10, 6))
    plt.scatter(normal_idx, normal_error, color="crimson", alpha=0.6, label="Normal Signals")
    plt.scatter(abnormal_idx, abnormal_error, color="indigo", alpha=0.6, label="Abnormal Signals")

    # 畫出門檻範圍陰影區塊
    plt.axhspan(threshold_min, threshold_max, color="magenta", alpha=0.1, label="Threshold Range")

    plt.title(f"{model_name}: Threshold Range Exploration", fontsize=14, fontweight='bold')
    plt.xlabel("Samples")
    plt.ylabel("Reconstruction Error")
    plt.legend(fontsize=10, loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if file_name:
        plt.savefig(file_name, format="svg")

    plt.show()


def plot_histogram_by_class(score_false, score_true, bins=30, class_name=None, model_name=None, file_name=None,
                            show_means=True, show_dprime=True, return_dprime=True):
    if not isinstance(bins, Iterable):
        bins = [bins, bins]
    if class_name is None or not isinstance(class_name, Iterable):
        class_name = ["Normal Signal", "Abnormal Signal"]
    if model_name is None:
        model_name = ""
    else:
        model_name += ": "
    
    plt.figure(figsize=(8, 5))

    # compute histograms and convert to percentages
    counts_f, bins_f = np.histogram(score_false, bins=bins[0], density=False)
    counts_t, bins_t = np.histogram(score_true,  bins=bins[1], density=False)
    total_f = counts_f.sum() if counts_f.sum() > 0 else 1
    total_t = counts_t.sum() if counts_t.sum() > 0 else 1

    percent_f = counts_f / total_f * 100.0
    percent_t = counts_t / total_t * 100.0

    bin_centers_f = (bins_f[:-1] + bins_f[1:]) / 2.0
    bin_centers_t = (bins_t[:-1] + bins_t[1:]) / 2.0

    # line-style histograms for visual overlap
    plt.plot(bin_centers_f, percent_f, color='crimson', label=class_name[0], 
             linewidth=2, marker='o', markersize=4, alpha=0.8)
    plt.plot(bin_centers_t, percent_t, color='indigo',  label=class_name[1], 
             linewidth=2, marker='s', markersize=4, alpha=0.8)

    # optional: annotate means with vertical dashed lines
    if show_means:
        mu_f = np.mean(score_false) if len(score_false) else np.nan
        mu_t = np.mean(score_true)  if len(score_true)  else np.nan
        if np.isfinite(mu_f):
            plt.axvline(mu_f, color='crimson', linestyle='--', linewidth=1.2, alpha=0.85)
        if np.isfinite(mu_t):
            plt.axvline(mu_t, color='indigo',  linestyle='--', linewidth=1.2, alpha=0.85)

    # compute d'
    dp = None
    if show_dprime:
        dp = dprime_from_scores(score_false, score_true, signed=False)
        suffix = f" (d' = {dp:.3f})" if np.isfinite(dp) else " (d' = n/a)"
    else:
        suffix = ""

    plt.xlabel("Reconstruction Error")
    plt.ylabel("Percentage (%)")
    plt.title(f"{model_name}Reconstruction Error Histogram by class{suffix}")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, alpha=0.3)

    if file_name:
        plt.savefig(file_name, format="svg")
    plt.show()

    return dp if return_dprime else None


def plot_loss_per_epoch(history, model_name="", file_name=None):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss", linestyle="dotted", color="black")
    
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss", color="coral")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss per Epoch")
    plt.legend()
    plt.grid(True)
    

    if file_name:
        plt.savefig(file_name, format="svg")  # 可以是 .svg, .png 等格式
    plt.show()



def plot_pr_curve(pr_curve_data, auprc, model_name=None, file_name=None):
    if model_name is None:
        model_name = ""
    else:
        model_name += ": "

    precision, recall, thresholds = pr_curve_data

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, color='coral', lw=2, label=f'AUPRC: {auprc:.2%}')
    plt.fill_between(recall, 0, precision, color='coral', alpha=0.2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"{model_name}Precision - Recall Curve")
    plt.legend(loc="upper right")
    plt.grid(True)

    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight', dpi=300)

    plt.show()


def plot_roc_curve(roc_curve_data, auroc, model_name=None, file_name=None):
    if model_name is None:
        model_name = ""
    else:
        model_name += ": "

    fpr, tpr, thresholds = roc_curve_data

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='coral', lw=2, label=f'AUROC: {auroc:.2%}')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{model_name}ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)

    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight', dpi=300)

    plt.show()

def dprime_from_scores(score_false: np.ndarray, score_true: np.ndarray, signed: bool=False):
    """
    Compute d' (d-prime) from two sample arrays.
    score_false: scores for negative class (e.g., normal)
    score_true : scores for positive class (e.g., abnormal)

    d' = (mu_true - mu_false) / sqrt(0.5*(var_true + var_false))
    If signed=False, return absolute value to represent separation magnitude only.
    """
    x0 = np.asarray(score_false, dtype=float)
    x1 = np.asarray(score_true, dtype=float)

    # unbiased estimates
    mu0, mu1 = x0.mean(), x1.mean()
    v0, v1 = x0.var(ddof=1), x1.var(ddof=1)

    denom = np.sqrt(0.5 * (v0 + v1))
    if denom == 0 or np.isnan(denom):
        return np.nan

    d = (mu1 - mu0) / denom
    return d if signed else abs(d)