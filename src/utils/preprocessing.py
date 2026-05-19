import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_dataset(path: str, label_col: str = "label", metadata_cols: list = None) -> tuple:
    p = Path(path)
    if p.suffix in (".xls", ".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    df.columns = [c.strip() for c in df.columns]

    drop_cols = set(metadata_cols or []) | {label_col}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df[label_col].copy()

    X = X.apply(pd.to_numeric, errors="coerce")

    return X, y, feature_cols


def preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_pca: bool = False,
    pca_variance: float = 0.95,
) -> tuple:
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    pca = None
    if apply_pca:
        pca = PCA(n_components=pca_variance, random_state=random_state)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    return X_train, X_test, y_train, y_test, le, scaler, pca
