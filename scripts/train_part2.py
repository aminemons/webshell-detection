"""
Part II — Feature extraction + 7 classifiers on WhiteBlackPHP dataset (d3.csv).

Steps:
    1. Load extracted features (run extract_features.py first)
    2. Feature selection: SelectKBest (chi2) + RF importance
    3. Train DT, KNN, SVM, NB, LR, RF, AdaCost with 5-fold CV GridSearch
    4. Evaluate: Accuracy, Precision, Recall, F1, AUC
    5. Compare: with and without feature selection

Usage:
    python scripts/train_part2.py --config configs/config.yaml
    python scripts/train_part2.py --no-feature-selection
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import MinMaxScaler

from src.models.adacost import AdaCost
from src.utils.preprocessing import load_config
from src.utils.metrics import (
    evaluate, print_report, save_results,
    plot_roc_curves, plot_confusion_matrix,
    plot_feature_importance, plot_class_distribution,
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
    "RandomForest": {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [5, 10, None],
        "clf__min_samples_split": [2, 5],
        "clf__max_features": ["sqrt", "log2"],
    },
    "AdaCost": {
        "clf__n_estimators": [50, 100, 200],
        "clf__cost_positive": [1.0, 2.0, 3.0, 5.0],
        "clf__cost_negative": [1.0],
    },
}

BASE_ESTIMATORS = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42),
    "NaiveBayes": GaussianNB(),
    "LogisticRegression": LogisticRegression(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "AdaCost": AdaCost(random_state=42),
}


def build_pipeline(estimator, use_feature_selection: bool, k_best: int) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if use_feature_selection:
        steps.append(("mms", MinMaxScaler()))
        steps.append(("selector", SelectKBest(f_classif, k=k_best)))
    steps.append(("clf", estimator))
    return Pipeline(steps)


def rf_feature_importance(X_train, y_train, feature_names: list, top_k: int, figures_dir: str) -> list:
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    idx = np.argsort(importances)[-top_k:]
    selected = [feature_names[i] for i in idx]

    plot_feature_importance(
        importances, feature_names,
        top_n=top_k,
        title=f"RF Feature Importance (top {top_k})",
        save_path=f"{figures_dir}/part2_rf_importance.png",
    )
    print(f"\n  Top {top_k} features by RF importance:")
    for fname in reversed(selected):
        print(f"    {fname}: {importances[feature_names.index(fname)]:.4f}")

    return selected


def run_gridsearch(name, estimator, X_train, y_train, use_fs, k_best, cv, n_jobs):
    pipe = build_pipeline(estimator, use_fs, k_best)
    param_grid = PARAM_GRIDS[name]
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, cv=skf, scoring="f1", n_jobs=n_jobs, refit=True, verbose=0)
    gs.fit(X_train, y_train)
    print(f"  {name}: best CV F1 = {gs.best_score_:.4f}")
    return gs.best_estimator_


def run_experiment(X_raw, y_enc, feature_names, tag, use_fs, k_best, cfg):
    cv = cfg["models"]["cv_folds"]
    n_jobs = cfg["models"]["n_jobs"]
    figures_dir = cfg["results"]["figures_dir"]
    metrics_dir = cfg["results"]["metrics_dir"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_enc,
        test_size=cfg["preprocessing"]["test_size"],
        random_state=cfg["preprocessing"]["random_state"],
        stratify=y_enc,
    )

    print(f"\n{'='*60}")
    print(f"  WhiteBlackPHP | FeatureSelection={use_fs}")
    print(f"{'='*60}")

    if use_fs:
        rf_feature_importance(X_train, y_train, feature_names, k_best, figures_dir)

    results = []
    fitted_models = {}

    for name, base_est in BASE_ESTIMATORS.items():
        print(f"\n[{name}] Grid search...")
        best_pipe = run_gridsearch(name, base_est, X_train, y_train, use_fs, k_best, cv, n_jobs)
        metrics = evaluate(best_pipe, X_test, y_test, model_name=name)
        results.append(metrics)
        fitted_models[name] = best_pipe

    print_report(results, title=f"WhiteBlackPHP | FeatureSelection={use_fs}")

    save_results(results, f"{metrics_dir}/part2_{tag}_results.csv")

    plot_roc_curves(
        fitted_models, X_test, y_test,
        title=f"ROC Curves — WhiteBlackPHP | FeatureSelection={use_fs}",
        save_path=f"{figures_dir}/part2_{tag}_roc.png",
    )

    plot_metric_comparison(
        results,
        metrics=["accuracy", "precision", "recall", "f1", "auc"],
        title=f"Classifier Comparison — WhiteBlackPHP | FeatureSelection={use_fs}",
        save_path=f"{figures_dir}/part2_{tag}_comparison.png",
    )

    return results, fitted_models


def main():
    parser = argparse.ArgumentParser(description="Part II — 7 classifiers on WhiteBlackPHP")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--features", default=None, help="Path to extracted features CSV")
    parser.add_argument("--no-feature-selection", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    features_path = args.features or cfg["data"]["d3_features_path"]
    figures_dir = cfg["results"]["figures_dir"]
    metrics_dir = cfg["results"]["metrics_dir"]
    k_best = cfg["feature_selection"]["k_best"]
    label_col = cfg["data"]["label_col"]
    metadata_cols = cfg["data"]["metadata_cols"]

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    if not os.path.exists(features_path):
        print(f"Extracted features not found at {features_path}")
        print("Run: python scripts/extract_features.py")
        sys.exit(1)

    print(f"\nLoading extracted features from {features_path} ...")
    df = pd.read_csv(features_path, low_memory=False)
    print(f"  Shape: {df.shape}")

    drop_cols = set(metadata_cols) | {label_col, "filename"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").values
    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    feature_names = feature_cols

    plot_class_distribution(
        df[label_col],
        title="Class Distribution — WhiteBlackPHP",
        save_path=f"{figures_dir}/part2_class_dist.png",
    )

    all_results = {}

    all_results["no_fs"], models_no_fs = run_experiment(
        X, y, feature_names, tag="no_fs", use_fs=False, k_best=k_best, cfg=cfg
    )

    if not args.no_feature_selection:
        all_results["with_fs"], models_with_fs = run_experiment(
            X, y, feature_names, tag="with_fs", use_fs=True, k_best=k_best, cfg=cfg
        )

    print("\n" + "="*60)
    print("  FINAL COMPARISON")
    print("="*60)
    for tag, res in all_results.items():
        print(f"\n  [{tag}]")
        print_report(res)

    flat = []
    for tag, res in all_results.items():
        for r in res:
            r["experiment"] = tag
            flat.append(r)
    save_results(flat, f"{metrics_dir}/part2_all_results.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
