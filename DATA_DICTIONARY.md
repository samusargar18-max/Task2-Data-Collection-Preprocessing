# Data Dictionary

The dataset has 64 pixel columns (`pixel_0` through `pixel_63`), a target column, and two engineered columns in the processed version.

| Column | Type | Meaning |
|---|---|---|
| `pixel_0` … `pixel_63` | Numeric | Intensity of one pixel in the 8×8 digit image; valid range 0–16. |
| `digit` | Integer | Correct handwritten digit label from 0 to 9; prediction target. |
| `total_ink` | Numeric | Sum of all 64 pixel intensities; engineered feature. |
| `active_pixels` | Integer | Count of pixels with intensity greater than zero; engineered feature. |

The raw source file has the 64 pixel features and `digit`. Engineered features are added only during preprocessing.
