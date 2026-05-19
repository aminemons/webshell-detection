"""
Part III — Compare model predictions against VirusTotal detections.

Usage:
    python scripts/virustotal_compare.py --api-key YOUR_KEY --config configs/config.yaml
    python scripts/virustotal_compare.py --api-key YOUR_KEY --input data/d3.csv --limit 100
"""

import argparse
import sys
import os
import time
import json
import hashlib
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils.preprocessing import load_config


VT_API_URL = "https://www.virustotal.com/api/v3"


def vt_get_file_report(file_hash: str, api_key: str) -> dict:
    headers = {"x-apikey": api_key}
    url = f"{VT_API_URL}/files/{file_hash}"
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 200:
        return resp.json()
    return {}


def vt_is_malicious(report: dict, threshold: int = 3) -> int:
    try:
        stats = report["data"]["attributes"]["last_analysis_stats"]
        return 1 if stats.get("malicious", 0) >= threshold else 0
    except (KeyError, TypeError):
        return -1


def compute_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Part III — VirusTotal comparison")
    parser.add_argument("--api-key", required=True, help="VirusTotal API key")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--limit", type=int, default=200, help="Max files to query (free API: 4 req/min)")
    parser.add_argument("--threshold", type=int, default=3, help="Min detections to label malicious")
    parser.add_argument("--delay", type=float, default=15.0, help="Seconds between VT requests")
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_path = args.input or cfg["data"]["d3_path"]
    figures_dir = cfg["results"]["figures_dir"]
    metrics_dir = cfg["results"]["metrics_dir"]

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path, low_memory=False)

    sample = df.sample(n=min(args.limit, len(df)), random_state=42).reset_index(drop=True)
    print(f"Querying {len(sample)} files on VirusTotal ...")
    print(f"Estimated time: {len(sample) * args.delay / 60:.1f} minutes")

    rows = []
    for i, row in sample.iterrows():
        code = str(row.get("sourcecode", "")) if pd.notna(row.get("sourcecode", "")) else ""
        true_label = row.get("label", "unknown")
        filename = row.get("filename", f"file_{i}")
        file_hash = compute_md5(code)

        report = vt_get_file_report(file_hash, args.api_key)
        vt_pred = vt_is_malicious(report, threshold=args.threshold)

        rows.append({
            "filename": filename,
            "true_label": true_label,
            "vt_prediction": vt_pred,
            "file_hash": file_hash,
        })

        if (i + 1) % 10 == 0:
            print(f"  Queried {i+1}/{len(sample)}")

        time.sleep(args.delay)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(f"{metrics_dir}/virustotal_results.csv", index=False)

    known = results_df[results_df["vt_prediction"] != -1].copy()
    if len(known) == 0:
        print("No results found on VirusTotal for these files.")
        return

    known["true_binary"] = (known["true_label"] == "webshell").astype(int)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    acc = accuracy_score(known["true_binary"], known["vt_prediction"])
    prec = precision_score(known["true_binary"], known["vt_prediction"], zero_division=0)
    rec = recall_score(known["true_binary"], known["vt_prediction"], zero_division=0)
    f1 = f1_score(known["true_binary"], known["vt_prediction"], zero_division=0)

    print(f"\nVirusTotal Results ({len(known)} files with reports):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")

    summary = [{"model": "VirusTotal", "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": float("nan")}]
    pd.DataFrame(summary).to_csv(f"{metrics_dir}/virustotal_summary.csv", index=False)

    cm = confusion_matrix(known["true_binary"], known["vt_prediction"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Webshell"],
                yticklabels=["Normal", "Webshell"], ax=ax)
    ax.set_xlabel("VirusTotal Prediction")
    ax.set_ylabel("True Label")
    ax.set_title("VirusTotal Confusion Matrix")
    fig.tight_layout()
    fig.savefig(f"{figures_dir}/virustotal_cm.png", dpi=150)
    plt.close(fig)

    print(f"\nResults saved to {metrics_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
