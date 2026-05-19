"""
Extract features from raw PHP source code (d3.csv) for Part II.

Usage:
    python scripts/extract_features.py --config configs/config.yaml
    python scripts/extract_features.py --input data/d3.csv --output data/d3_features.csv
"""

import argparse
import sys
import os
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tqdm import tqdm

from src.features.extractor import FeatureExtractor
from src.utils.preprocessing import load_config


def main():
    parser = argparse.ArgumentParser(description="Extract features from raw PHP source code")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", default=None, help="Override input path")
    parser.add_argument("--output", default=None, help="Override output path")
    parser.add_argument("--chunksize", type=int, default=500, help="Rows per chunk")
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_path = args.input or cfg["data"]["d3_path"]
    output_path = args.output or cfg["data"]["d3_features_path"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading raw PHP data from {input_path} ...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Loaded {len(df)} rows")

    extractor = FeatureExtractor()
    records = []

    print("Extracting features...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Files"):
        code = str(row.get("sourcecode", "")) if pd.notna(row.get("sourcecode", "")) else ""
        feat = extractor.extract_one(code)
        feat["filename"] = row.get("filename", "")
        feat["label"] = row.get("label", "")
        records.append(feat)

    result = pd.DataFrame(records)
    result.to_csv(output_path, index=False)

    print(f"\nExtracted {len(result)} rows x {result.shape[1]} columns")
    print(f"Saved to {output_path}")

    print("\nLabel distribution:")
    print(result["label"].value_counts().to_string())

    print("\nFeature summary (first 10):")
    numeric_cols = [c for c in result.columns if c not in ("filename", "label")]
    print(result[numeric_cols[:10]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
