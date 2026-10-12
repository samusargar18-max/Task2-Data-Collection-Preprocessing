# Data Collection & Preprocessing — Handwritten Digits Dataset

## Project objective
Collect a public dataset with more than 1,000 samples, inspect data quality, clean it, engineer useful features, and document a repeatable preprocessing pipeline.

## Dataset source
- **Dataset:** Optical Recognition of Handwritten Digits (the 8×8 handwritten digit dataset distributed by scikit-learn from the UCI dataset family).
- **Public source page:** https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
- **Access method:** `sklearn.datasets.load_digits()`; the included source CSV contains 1,797 samples, 64 pixel features, and one target label (`digit`).
- **Pixel values:** integer-like grayscale intensity values from 0 to 16.
- **Purpose:** demonstrate data cleaning and feature preparation, not to make claims about a production model.

## Important note about the raw data
`handwritten_digits_source.csv` is the original clean dataset as loaded from scikit-learn. `handwritten_digits_dirty.csv` is a **teaching copy** in which a small number of missing values, invalid strings, duplicate rows, and out-of-range pixel values were intentionally added so the cleaning steps can be demonstrated and verified. This is explicitly documented rather than presented as naturally occurring errors in the original source.

## Project structure
- `data/raw/handwritten_digits_source.csv` — original source data
- `data/raw/handwritten_digits_dirty.csv` — working copy with controlled data-quality issues
- `data/processed/handwritten_digits_clean.csv` — cleaned dataset with engineered features
- `data/processed/handwritten_digits_scaled.csv` — scaled pixel features for ML workflows
- `scripts/preprocess.py` — repeatable preprocessing script
- `output/preprocessing_report.json` — machine-readable report
- `output/execution_output.txt` — execution summary
- `output/digit_distribution.png` — class distribution chart
- `notebooks/preprocessing_walkthrough.ipynb` — notebook version / walkthrough
- `DATA_DICTIONARY.md` — column descriptions

## Preprocessing pipeline
1. Load the public dataset / working CSV.
2. Convert pixel and target columns to numeric types; invalid strings become missing values.
3. Detect and remove exact duplicate rows.
4. Fill missing pixel values with each column's median; remove rows only if target is missing.
5. Clip pixel intensities to the valid domain `[0, 16]` to handle outliers.
6. Engineer `total_ink` (sum of pixel intensities) and `active_pixels` (count of pixels greater than zero).
7. Save cleaned and standardized versions.
8. Generate a report and a digit-distribution plot.

## How to run (Mac / Windows)
Install Python 3.10+ and run in Terminal from the project folder:

```bash
python -m pip install pandas numpy scikit-learn matplotlib jupyter
python scripts/preprocess.py
```

To open the notebook:

```bash
jupyter notebook notebooks/preprocessing_walkthrough.ipynb
```

## Expected outputs
The script prints the number of raw and clean rows, missing-cell and duplicate counts, and the output file path. The exact report is saved under `output/`. Re-running the script regenerates the processed CSVs and report.

## Limitations
The dataset is a small 8×8 grayscale representation of handwritten digits and is not a full-resolution image corpus. Missing values, duplicate rows, invalid strings, and outliers in the dirty working file are controlled examples introduced to demonstrate preprocessing. Standardization is saved separately because it changes pixel scales and is usually applied as part of an ML training pipeline.

## Citation
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine. Optical Recognition of Handwritten Digits. Dataset family/source page linked above. The scikit-learn `load_digits` documentation describes the 1,797-sample 8×8 dataset.
