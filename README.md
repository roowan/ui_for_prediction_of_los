## `ui_for_prediction_of_los/` — Desktop Prediction GUI

A standalone Tkinter desktop app for predicting workup time for individual patients or a batch via Excel upload.

### Files

| File | Description |
|------|-------------|
| `prediction_ui.py` | Main GUI application |
| `ml_model.py` | Training script that produces `workup_model_age7_ar1.pkl` |
| `ehr_age7_ar1.xlsx` | Dataset (copy) | |must place your ehr file|
| `workup_model_age7_ar1.pkl` | Model (copy) |

### Features
- **Single predict tab** — select gender, age category, specialty, arrival slot, session, patient flow type; get instant workup time prediction
- **Bulk predict tab** — browse for an `.xlsx` file, runs predictions for all rows in a background thread, shows scrollable results table, and exports to Excel
- Session / arrival-slot cross-validation with inline error banners (Forenoon: 08:00–13:00, Lunch: 13:00–15:00, Afternoon: 15:00–17:00)

### How to run
```bash
cd ui_for_prediction_of_los
python prediction_ui.py
```

### Train / retrain the model
```bash
python ml_model.py
# Outputs: workup_model_age7_ar1.pkl
```
