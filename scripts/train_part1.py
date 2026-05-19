"""
Part I — Classifiers on D1 and D2 pre-extracted feature datasets.

Steps:
    1. EDA: class balance, correlation heatmap, feature distributions
    2. Preprocess: impute, scale, optional PCA
    3. GridSearchCV 5-fold for DT, KNN, SVM, NB, LR
    4. Evaluate: Accuracy, Precision, Recall, F1, AUC
    5. Compare: with and without PCA, D1 vs D2

Usage:
    python scripts/train_part1.py --config configs/config.yaml
    python scripts/train_part1.py --dataset d2 --no-pca
"""

import argparse
import sys
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from src.utils.preprocessing import load_config, load_dataset, preprocess
from src.utils.metrics import (
    evaluate, print_report, save_results,
    plot_roc_curves, plot_confusion_matrix,
    plot_correlation_heatmap, plot_class_distribution,
    plot_metric_comparison,
)


PARAM_GRIDS = {
    "DecisionTree": {
        "clf__max_depth": [3, 5, 10, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__criterion": ["gini", "entropy"],
    },
    "KNN": {
        "clf__n_neighbors": [3, 5, 7, 11, 15],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"],
    },
    "SVM": {
        "clf__C": [0.1, 1, 10, 100],
        "clf__kernel": ["rbf", "linear"],
        "clf__gamma": ["scale", "auto"],
    },
    "NaiveBayes": {
        "clf__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
    },
    "LogisticRegression": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__solver": ["lbfgs", "liblinear"],
        "clf__max_iter": [2000],
    },
}

BASE_ESTIMATORS = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42),
    "NaiveBayes": GaussianNB(),
    "LogisticRegression": LogisticRegression(random_state=42),
}


def build_pipeline(estimator, apply_pca: bool, pca_variance: float) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if apply_pca:
        steps.append(("pca", PCA(n_components=pca_variance, random_state=42)))
    steps.append(("clf", estimator))
    return Pipeline(steps)


def run_gridsearch(name: str, estimator, X_train, y_train, apply_pca: bool, pca_variance: float, cv: int, n_jobs: int):
    pipe = build_pipeline(estimator, apply_pca, pca_variance)
    param_grid = PARAM_GRIDS[name]
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, cv=skf, scoring="f1", n_jobs=n_jobs, refit=True, verbose=0)
    gs.fit(X_train, y_train)
    print(f"  {name}: best params = {gs.best_params_}  best CV F1 = {gs.best_score_:.4f}")
    return gs.best_estimator_


def eda(X: pd.DataFrame, y: pd.Series, dataset_name: str, figures_dir: str):
    print(f"\n--- EDA: {dataset_name} ---")
    print(f"  Shape: {X.shape}")
    print(f"  Missing values: {X.isna().sum().sum()}")
    print(f"  Class distribution:\n{y.value_counts().to_string()}")

    plot_class_distribution(
        y,
        title=f"Class Distribution — {dataset_name}",
        save_path=f"{figures_dir}/{dataset_name}_class_dist.png",
    )

    numeric_X = X.apply(pd.to_numeric, errors="coerce")
    plot_correlation_heatmap(
        numeric_X,
        title=f"Feature Correlation — {dataset_name} (top 40 features)",
        save_path=f"{figures_dir}/{dataset_name}_correlation.png",
        top_n=40,
    )


def run_experiment(
    X_raw: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
    apply_pca: bool,
    cfg: dict,
) -> list:
    test_size = cfg["preprocessing"]["test_size"]
    random_state = cfg["preprocessing"]["random_state"]
    pca_variance = cfg["preprocessing"]["pca_variance"]
    cv_folds = cfg["models"]["cv_folds"]
    n_jobs = cfg["models"]["n_jobs"]
    figures_dir = cfg["results"]["figures_dir"]
    metrics_dir = cfg["results"]["metrics_dir"]

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw.values, y_enc,
        test_size=test_size,
        random_state=random_state,
        stratify=y_enc,
    )

    suffix = "pca" if apply_pca else "raw"
    print(f"\n{'='*60}")
    print(f"  {dataset_name} | PCA={apply_pca}")
    print(f"{'='*60}")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    results = []
    fitted_models = {}

    for name, base_est in BASE_ESTIMATORS.items():
        print(f"\n[{name}] Grid search...")
        best_pipe = run_gridsearch(
            name, base_est, X_train, y_train, apply_pca, pca_variance, cv_folds, n_jobs
        )
        metrics = evaluate(best_pipe, X_test, y_test, model_name=name)
        results.append(metrics)
        fitted_models[name] = best_pipe

    print_report(results, title=f"{dataset_name} | PCA={apply_pca}")

    tag = f"{dataset_name}_{suffix}"
    save_results(results, f"{metrics_dir}/{tag}_results.csv")

    plot_roc_curves(
        fitted_models, X_test, y_test,
        title=f"ROC Curves — {dataset_name} | PCA={apply_pca}",
        save_path=f"{figures_dir}/{tag}_roc.png",
    )

    plot_metric_comparison(
        results,
        metrics=["accuracy", "precision", "recall", "f1", "auc"],
        title=f"Classifier Comparison — {dataset_name} | PCA={apply_pca}",
        save_path=f"{figures_dir}/{tag}_comparison.png",
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Part I — Webshell detection on D1/D2 datasets")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset", choices=["d1", "d2", "both"], default="both",
                        help="Which dataset(s) to use")
    parser.add_argument("--no-pca", action="store_true", help="Skip PCA experiments")
    args = parser.parse_args()

    cfg = load_config(args.config)
    metadata_cols = cfg["data"]["metadata_cols"]
    label_col = cfg["data"]["label_col"]
    figures_dir = cfg["results"]["figures_dir"]
    metrics_dir = cfg["results"]["metrics_dir"]

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    datasets = {}

    if args.dataset in ("d1", "both"):
        d1_path = cfg["data"]["d1_path"]
        if os.path.exists(d1_path):
            print(f"\nLoading D1 from {d1_path} ...")
            X1, y1, _ = load_dataset(d1_path, label_col=label_col, metadata_cols=metadata_cols)
            datasets["D1"] = (X1, y1)
            eda(X1, y1, "D1", figures_dir)
        else:
            print(f"WARNING: D1 file not found at {d1_path}")

    if args.dataset in ("d2", "both"):
        d2_path = cfg["data"]["d2_path"]
        if os.path.exists(d2_path):
            print(f"\nLoading D2 from {d2_path} ...")
            X2, y2, _ = load_dataset(d2_path, label_col=label_col, metadata_cols=metadata_cols)
            datasets["D2"] = (X2, y2)
            eda(X2, y2, "D2", figures_dir)
        else:
            print(f"WARNING: D2 file not found at {d2_path}")

    if not datasets:
        print("No datasets found. Check your data/ directory.")
        sys.exit(1)

    all_results = {}

    for ds_name, (X, y) in datasets.items():
        all_results[f"{ds_name}_raw"] = run_experiment(X, y, ds_name, apply_pca=False, cfg=cfg)
        if not args.no_pca:
            all_results[f"{ds_name}_pca"] = run_experiment(X, y, ds_name, apply_pca=True, cfg=cfg)

    print("\n" + "="*60)
    print("  FINAL COMPARISON ACROSS DATASETS & SETTINGS")
    print("="*60)
    for tag, res in all_results.items():
        print(f"\n  [{tag}]")
        print_report(res)

    flat = []
    for tag, res in all_results.items():
        for r in res:
            r["experiment"] = tag
            flat.append(r)
    save_results(flat, f"{metrics_dir}/part1_all_results.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
