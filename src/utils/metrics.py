import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


def evaluate(model, X_test, y_test, model_name: str = "") -> dict:
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)

    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": auc,
    }


def print_report(results: list, title: str = ""):
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    df = pd.DataFrame(results)
    df = df.set_index("model")
    print(df.round(4).to_string())
    print()


def save_results(results: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(path, index=False)
    print(f"Results saved to {path}")


def plot_roc_curves(models_dict: dict, X_test, y_test, title: str, save_path: str):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            proba = model.decision_function(X_test)
        else:
            continue
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"ROC curves saved to {save_path}")


def plot_confusion_matrix(model, X_test, y_test, labels: list, title: str, save_path: str):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(importances: np.ndarray, feature_names: list, top_n: int, title: str, save_path: str):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    ax.barh([feature_names[i] for i in idx], importances[idx])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Feature importance plot saved to {save_path}")


def plot_correlation_heatmap(X: pd.DataFrame, title: str, save_path: str, top_n: int = 40):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    corr = X.iloc[:, :top_n].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, xticklabels=True, yticklabels=True, linewidths=0.3)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Correlation heatmap saved to {save_path}")


def plot_class_distribution(y: pd.Series, title: str, save_path: str):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    counts = y.value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax, color=["steelblue", "tomato"])
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    for i, v in enumerate(counts):
        ax.text(i, v + 5, str(v), ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Class distribution plot saved to {save_path}")


def plot_metric_comparison(results: list, metrics: list, title: str, save_path: str):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results).set_index("model")[metrics]
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(kind="bar", ax=ax, rot=30)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Metric comparison plot saved to {save_path}")
