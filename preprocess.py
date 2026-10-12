"""Data Collection & Preprocessing pipeline for the UCI handwritten digits dataset.
Dataset: scikit-learn load_digits (1797 original samples; UCI Optical Recognition of Handwritten Digits subset).
Run: python scripts/preprocess.py
Dependencies: pandas, numpy, scikit-learn, matplotlib
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Collect public dataset and save the original source data if not already present.
source_file = RAW_DIR / "handwritten_digits_source.csv"
if not source_file.exists():
    digits = load_digits()
    columns = [f"pixel_{i}" for i in range(64)]
    source = pd.DataFrame(digits.data, columns=columns)
    source["digit"] = digits.target
    source.to_csv(source_file, index=False)

# 2. Load the working raw file (contains intentionally introduced quality issues for demonstration).
raw_file = RAW_DIR / "handwritten_digits_dirty.csv"
if not raw_file.exists():
    raise FileNotFoundError(f"Missing {raw_file}. Keep the supplied raw file in the project.")
raw = pd.read_csv(raw_file)
initial_rows = len(raw)
initial_duplicates = int(raw.duplicated().sum())
initial_missing = int(raw.isna().sum().sum())

# 3. Correct data types. Pixel features and label must be numeric; invalid strings become NaN.
feature_cols = [c for c in raw.columns if c.startswith("pixel_")]
raw[feature_cols] = raw[feature_cols].apply(pd.to_numeric, errors="coerce")
raw["digit"] = pd.to_numeric(raw["digit"], errors="coerce")

# 4. Remove exact duplicate rows.
clean = raw.drop_duplicates().copy()
rows_after_dedup = len(clean)

# 5. Handle missing values in pixel features using column median; remove rows with missing target.
for col in feature_cols:
    clean[col] = clean[col].fillna(clean[col].median())
clean = clean.dropna(subset=["digit"])

# 6. Handle outliers using domain knowledge: pixel intensities must be between 0 and 16.
# Clip any invalid intensity values to the valid range instead of deleting a complete sample.
for col in feature_cols:
    clean[col] = clean[col].clip(lower=0, upper=16)
clean["digit"] = clean["digit"].astype(int)

# 7. Feature engineering: total ink intensity and number of active pixels.
clean["total_ink"] = clean[feature_cols].sum(axis=1)
clean["active_pixels"] = (clean[feature_cols] > 0).sum(axis=1).astype(int)

# 8. Save clean data and a scaled feature file (pixel features standardized; engineered features retained).
clean_path = OUT_DIR / "handwritten_digits_clean.csv"
clean.to_csv(clean_path, index=False)
scaler = StandardScaler()
scaled = clean.copy()
scaled[feature_cols] = scaler.fit_transform(scaled[feature_cols])
scaled.to_csv(OUT_DIR / "handwritten_digits_scaled.csv", index=False)

# 9. Create concise data-quality report and visualization.
report = {
    "dataset": "UCI Optical Recognition of Handwritten Digits (scikit-learn load_digits subset)",
    "original_public_source_samples": 1797,
    "working_raw_rows_including_demo_duplicates": initial_rows,
    "raw_duplicate_rows_detected": initial_duplicates,
    "raw_missing_cells_detected_before_type_correction": initial_missing,
    "rows_after_duplicate_removal": rows_after_dedup,
    "final_clean_rows": len(clean),
    "final_columns_including_engineered_features": len(clean.columns),
    "missing_cells_after_cleaning": int(clean.isna().sum().sum()),
    "duplicate_rows_after_cleaning": int(clean.duplicated().sum()),
    "pixel_value_min_after_cleaning": float(clean[feature_cols].min().min()),
    "pixel_value_max_after_cleaning": float(clean[feature_cols].max().max()),
    "engineered_features": ["total_ink", "active_pixels"],
    "target_column": "digit"
}
(REPORT_DIR / "preprocessing_report.json").write_text(__import__('json').dumps(report, indent=2))
with (REPORT_DIR / "execution_output.txt").open("w") as f:
    f.write("DATA COLLECTION & PREPROCESSING - EXECUTION SUMMARY\n")
    f.write("="*52 + "\n")
    for k,v in report.items(): f.write(f"{k}: {v}\n")
    f.write("\nStatus: SUCCESS - cleaned dataset saved.\n")

counts = clean["digit"].value_counts().sort_index()
plt.figure(figsize=(8,4.5))
counts.plot(kind="bar")
plt.title("Number of Samples per Digit (Clean Dataset)")
plt.xlabel("Digit label")
plt.ylabel("Number of samples")
plt.tight_layout()
plt.savefig(REPORT_DIR / "digit_distribution.png", dpi=150)
plt.close()
print("Preprocessing completed successfully.")
print(f"Rows: {initial_rows} raw -> {len(clean)} clean")
print(f"Missing cells: {initial_missing} detected before type correction -> {int(clean.isna().sum().sum())} after cleaning")
print(f"Duplicates removed: {initial_duplicates}")
print(f"Clean dataset: {clean_path}")
