# Webshell Detection

Detecting malicious PHP webshells using classical machine learning. This project implements a full pipeline covering exploratory data analysis, custom feature engineering, dimensionality reduction, hyperparameter-tuned classifiers, and VirusTotal validation.

## Project Structure

```
.
├── configs/
│   └── config.yaml          # All hyperparameter grids and paths
├── data/                    # Place data files here (not tracked by git)
├── results/
│   ├── metrics/             # CSV result tables
│   └── figures/             # Generated plots
├── scripts/
│   ├── train_part1.py       # Part I: 5 classifiers on D1/D2 pre-extracted features
│   ├── extract_features.py  # Part II: feature extraction from raw PHP source code
│   ├── train_part2.py       # Part II: 7 classifiers on WhiteBlackPHP
│   └── virustotal_compare.py # Part III: VirusTotal validation
└── src/
    ├── features/
    │   ├── statistical.py   # Entropy, IC, longest word, compression ratio
    │   ├── syntactical.py   # String counts, special chars, if/loop counts
    │   ├── lexical.py       # Dangerous functions, commands, variables, keywords
    │   └── extractor.py     # Unified feature extraction pipeline
    ├── models/
    │   └── adacost.py       # AdaCost: cost-sensitive boosting (Fan et al., 1999)
    └── utils/
        ├── preprocessing.py # Data loading, scaling, PCA, train/test split
        └── metrics.py       # Evaluation, plots, result export
```

## Datasets

| File | Dataset | Rows | Description |
|------|---------|------|-------------|
| `d1.xls` | D1 | 3,373 | Pre-extracted features, MD5 dedup |
| `d2.csv` | D2 | 2,238 | Pre-extracted features, MD5+VLD dedup (balanced) |
| `d3.csv` | WhiteBlackPHP | 6,134 | Raw PHP source code for feature extraction |

Data files are **not tracked by git**. Copy them to the `data/` directory manually.

## Setup

```bash
git clone https://github.com/aminemons/webshell-detection.git
cd webshell-detection
pip install -r requirements.txt

# Copy data files
scp -P 59172 /path/to/d1.xls /path/to/d2.csv /path/to/d3.csv data/
```

## Reproducing Results

### Part I — Classifiers on D1 / D2

```bash
# Run on both datasets, with and without PCA
python scripts/train_part1.py

# Run only on D2, skip PCA
python scripts/train_part1.py --dataset d2 --no-pca
```

Outputs:
- `results/metrics/D1_raw_results.csv`, `D2_raw_results.csv`, etc.
- `results/figures/D1_roc.png`, `D2_comparison.png`, etc.

### Part II — Feature Extraction + 7 Classifiers

```bash
# Step 1: extract features from raw PHP source code
python scripts/extract_features.py

# Step 2: train 7 classifiers with and without feature selection
python scripts/train_part2.py
```

Outputs:
- `data/d3_features.csv` — extracted features
- `results/metrics/part2_no_fs_results.csv`, `part2_with_fs_results.csv`
- `results/figures/part2_roc.png`, `part2_rf_importance.png`, etc.

### Part III — VirusTotal Validation

```bash
python scripts/virustotal_compare.py --api-key YOUR_KEY --limit 200
```

## Models

| Model | Type | Tuned Hyperparameters |
|-------|------|-----------------------|
| Decision Tree | Base | max_depth, min_samples_split, criterion |
| KNN | Base | n_neighbors, weights, metric |
| SVM | Base | C, kernel, gamma |
| Naive Bayes | Base | var_smoothing |
| Logistic Regression | Base | C, solver |
| Random Forest | Ensemble | n_estimators, max_depth, max_features |
| AdaCost | Ensemble (cost-sensitive) | n_estimators, cost_positive |

All models are tuned via **5-fold stratified cross-validation GridSearchCV**, optimizing F1-score.

## Features (Part II)

**Statistical** (neopi-inspired): Entropy, Index of Coincidence, Longest Word, Compression Ratio, Signature count

**Syntactical**: File size, string count, unique string count, special char count, max line length, if-statement count, loop count

**Lexical**: Binary count of 47 dangerous functions, 16 dangerous commands, 10 dangerous variables, 16 dangerous keywords

## Feature Selection

Two methods are compared:
1. **SelectKBest with F-score** (top 30 features) — filter method
2. **Random Forest feature importance** — embedded method (visualization only)

## Evaluation Metrics

Accuracy, Precision, Recall, F1-score, AUC — reported on held-out test set (80/20 stratified split).

## Remote Training (RTX A5000)

```bash
# Connect to GPU server
ssh amine@192.168.3.16 -p 59172

# First time setup
git clone https://github.com/aminemons/webshell-detection.git
cd webshell-detection
pip install -r requirements.txt
mkdir -p data

# Copy data files from local machine (run this locally)
scp -P 59172 "d:\machine learning project\d1.xls" amine@192.168.3.16:~/webshell-detection/data/
scp -P 59172 "d:\machine learning project\d2.csv" amine@192.168.3.16:~/webshell-detection/data/
scp -P 59172 "d:\machine learning project\d3.csv" amine@192.168.3.16:~/webshell-detection/data/

# Run experiments (inside the server)
cd ~/webshell-detection
python scripts/train_part1.py
python scripts/extract_features.py
python scripts/train_part2.py

# Push results
git add results/ data/d3_features.csv
git commit -m "results: add Part I and Part II outputs"
git push
```

Then locally:
```bash
cd "d:\machine learning project"
git pull
```

## References

- Fan, W., Stolfo, S. J., Zhang, J., & Chan, P. K. (1999). AdaCost: Misclassification cost-sensitive boosting. *ICML*.
- Neopi: [https://code.google.com/archive/p/neo-pi/](https://code.google.com/archive/p/neo-pi/)
- Dataset: [https://data.mendeley.com/datasets/wt8m6bcwbr/2](https://data.mendeley.com/datasets/wt8m6bcwbr/2)
