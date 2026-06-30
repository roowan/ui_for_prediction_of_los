# ui_for_prediction_of_los — EHR Workup Time Predictor

A desktop app (Tkinter GUI) and training pipeline that predicts how long a
patient's clinical workup will take, using only information known at
intake — before the visit happens. Built on real EHR data from an eye
hospital (225,961 visits), using XGBoost.

> **Note:** the original training dataset (`ehr_age7_ar1.xlsx`, ~33MB) is
> not included in this repo due to size and patient privacy. See
> [Dataset Format](#dataset-format) below for exactly what columns your
> own data needs, so you can retrain on your own EHR export.

---

## What's in this repo

| File | Description |
|---|---|
| `prediction_ui.py` | Desktop GUI — single-patient and bulk (Excel) prediction |
| `ml_model.py` | Training script — reads `ehr_age7_ar1.xlsx`, outputs `workup_model_age7_ar1.pkl` |
| `workup_model_age7_ar1.pkl` | Pre-trained model (included, ready to use without retraining) |

---

## Setup

### 1. Requirements

Python 3.9+ recommended. Install dependencies:

```bash
pip install pandas numpy scikit-learn xgboost openpyxl
```

`openpyxl` is required for reading/writing `.xlsx` files (used by both
`pandas.read_excel` and the bulk-export feature) — it's not bundled with
pandas by default, so don't skip it.

`tkinter` is required for the GUI. On most systems it ships with Python
already. If you get `ModuleNotFoundError: No module named 'tkinter'`:

- **Windows / macOS (official python.org installer):** already included
- **Linux (Debian/Ubuntu):** `sudo apt-get install python3-tk`
- **Linux (Fedora):** `sudo dnf install python3-tkinter`

### 2. What order to run things in

You have two options depending on whether you want to retrain or just use
the existing model:

**Option A — just run the predictor (fastest, uses the included model):**

```bash
python prediction_ui.py
```

This loads `workup_model_age7_ar1.pkl` directly. No dataset needed.

**Option B — retrain from scratch on your own data:**

1. Place your EHR Excel file in this folder, named exactly
   `ehr_age7_ar1.xlsx` (or edit the filename at the top of `ml_model.py`
   to point at your file).
2. Run the training script:
   ```bash
   python ml_model.py
   ```
   This takes roughly **15–30 minutes** on a CPU laptop (4,000 boosting
   rounds with early stopping — it usually doesn't need all 4,000, but
   budget for the full time). It prints progress through 7 stages: load →
   clean → feature engineering → split → train → evaluate → save.
3. It overwrites `workup_model_age7_ar1.pkl` with your newly trained model.
4. Run the GUI as in Option A — it'll now use your retrained model.

---

## Dataset Format

This is the part that matters if you want to retrain on your own hospital's
data. `ml_model.py` expects an Excel file (`.xlsx`) with a single header
row and the following columns. **Column order does not matter** — the
script selects columns by name — but **column names must match exactly**
(case-sensitive, including spaces).

### Required columns

| Column name | Type | Example values | Notes |
|---|---|---|---|
| `Age category` | text | `0-12`, `13-19`, `20-30`, `31-45`, `46-60`, `61-75`, `76-90` | Pre-binned age group. If you only have raw numeric age, bin it into these 7 ranges before training (see mapping below). |
| `Gender` | text | `Male`, `Female`, `Unknown` | Script also auto-cleans `Fema` → `Female` and `TBU` → `Unknown`, but cleaner input is safer. |
| `Patient visit category` | text | `MRE`, `REG`, `SRE` | Visit type code. Use whatever 3-letter (or any consistent) codes your hospital system uses — the script doesn't validate against a fixed list. |
| `Patient Flow type` | text | `Dilated`, `Non-Dilated`, `Procedure` | Whether the patient's pupils were dilated, not dilated, or it was a procedure visit. |
| `Specialty` | text | `Cornea`, `General`, `Glaucoma`, `Pediatric and Low vision`, `Retina` | Department/specialty seen. |
| `Consultant Name` | text | e.g. `NELSON JESUDASAN` | Full name or unique ID of the treating consultant. Used for target encoding — the more visits per consultant in your data, the more reliable this signal is. |
| `Consultant Designation` | text | `Specialist`, `Post Graduate` | Seniority/role of the consultant. |
| `Day` | text | `Mon`, `Tue`, `Wed`, `Thu`, `Fri`, `Sat`, `Sun` | 3-letter day abbreviation. |
| `Session` | text | `Forenoon`, `Lunch`, `Afternoon` | Clinic session block. |
| `Arrival hour` | text | `08:00 - 09:00`, `09:00 - 10:00`, ... `16:00 - 17:00` | 1-hour arrival slot, formatted exactly as `HH:00 - HH:00` (space-hyphen-space). See full list below. |
| `VISITDATE` | date | `2024-03-15` | Used to derive Month/Quarter. Must be a proper Excel date column, not text — `pandas.read_excel` needs to parse it as a datetime automatically. |
| `TOTAL_WORKUP_TIME` | time | `01:15:00` (1h 15m) | **This is the target/label.** Must be an Excel time-formatted column (so it loads as a Python `datetime.time` object), not a plain number or string. The script converts it to total minutes internally. |

### Exact accepted values for `Arrival hour`

```
08:00 - 09:00
09:00 - 10:00
10:00 - 11:00
11:00 - 12:00
12:00 - 13:00
13:00 - 14:00
14:00 - 15:00
15:00 - 16:00
16:00 - 17:00
```

### Session / arrival-hour consistency

The GUI enforces that `Session` and `Arrival hour` agree with each other —
this isn't required by the training script, but if your data violates it,
predictions made through the GUI later may behave oddly for those rows.
Keep them consistent:

| Session | Valid arrival hours |
|---|---|
| `Forenoon` | 08:00–09:00 through 12:00–13:00 (start hour 8–12) |
| `Lunch` | 13:00–14:00 or 14:00–15:00 (start hour 13–14) |
| `Afternoon` | 15:00–16:00 or 16:00–17:00 (start hour 15–16) |

### Age binning, if you only have raw age

If your EHR export has numeric age instead of pre-binned categories, bin
it like this before training (this is the exact logic the GUI uses for
single/bulk predictions, so keep your training data consistent with it):

```
age <= 12   → "0-12"
age 13-19   → "13-19"
age 20-30   → "20-30"
age 31-45   → "31-45"
age 46-60   → "46-60"
age 61-75   → "61-75"
age > 75    → "76-90"
```

### Columns that exist in the original dataset but are NOT used

If your EHR export has timestamp columns like `OPTO_SIGN_IN`,
`OPTO_SIGN_OUT`, `CONS_SIGN_IN`, `CONS_SIGN_OUT`, or any
`*_WAITING_TIME` / `*_WORKUP_TIME` columns other than the target — leave
them in the sheet if you like, `ml_model.py` will just ignore them. **Do
not feed them in as features** if you modify the script — they're
recorded during or after the visit, so using them as predictors would be
data leakage (you wouldn't have this information at the time you're
actually trying to predict workup time, i.e. when the patient walks in).

### Outlier handling

Rows where `TOTAL_WORKUP_TIME` converts to less than 5 minutes or more
than 300 minutes are automatically dropped before training (almost
certainly data entry errors or non-standard visit types). If your
hospital's typical workup times fall outside this range, adjust the
bounds near the top of the `CLEAN` section in `ml_model.py`.

### Minimum data size

The original model was trained on ~226k rows after cleaning. You can
train on far less, but expect:
- High-cardinality features (especially `Consultant Name`, with 61
  distinct values in the original data) need enough rows per consultant
  for target encoding to be reliable — a consultant with only 2-3 visits
  in your data will fall back to the dataset-wide average.
- A few thousand rows is a reasonable minimum to get a model that
  isn't just memorizing noise; tens of thousands is where this approach
  starts to shine.

---

## Bulk prediction (Excel upload via the GUI)

The "Bulk Predict" tab in `prediction_ui.py` accepts a *separate* Excel
file at inference time — this is different from the training dataset
above, and is meant for hospital staff to upload a sheet of patients
who haven't visited yet, to get predictions for all of them at once.

Required columns (any order, **case-insensitive** column headers):

```
Gender
Patient visit category
Patient Flow type
Specialty
Consultant Designation
Day
Session
Arrival hour
```

Plus exactly one of these for age:
```
Age            (numeric, e.g. 14 — auto-binned into a category)
Age category   (pre-binned, e.g. "13-19" or "13-19 (Teenage)" — both formats accepted)
```

`Consultant Name` is optional in bulk uploads — if omitted, the model
falls back to the dataset-wide average consultant time rather than a
specific consultant's average.

Output: a results table in the GUI plus an exportable Excel file with
predicted minutes, a confidence range (±18%/-18% band around the point
estimate), and a human-readable time string per row. Rows with
inconsistent Session/Arrival-hour combinations are flagged with a
validation error rather than silently predicted.

---

## How retraining changes the model file

Running `ml_model.py` produces a pickle (`workup_model_age7_ar1.pkl`)
containing three things bundled together:

1. The trained XGBoost model itself
2. `feature_columns` — the exact list and order of the 138 encoded
   feature columns the model expects at prediction time
3. `encoding_maps` — the target-encoding lookup tables (consultant →
   avg time, specialty → avg time, flow type → avg time) computed from
   the training set only

`prediction_ui.py` loads all three from the pickle and uses them
together to build a matching feature row for any new patient at
prediction time — so retraining and prediction always stay in sync as
long as you don't edit the feature list in `ml_model.py` without also
updating `build_features()` in `prediction_ui.py` to match.
