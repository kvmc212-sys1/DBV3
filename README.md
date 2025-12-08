# DriftBreaker: Credit Risk Model Drift Detection

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place parquet files in project root
# - train_low_risk.parquet
# - train_medium_risk.parquet  
# - train_high_risk.parquet
# - fred_macro.csv (optional)

# 3. Run pipeline (generates artifacts)
python run_pipeline.py

# 4. Launch dashboard
streamlit run dashboard.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      PORTFOLIO VIEW                             │
│  Roll-up: Portfolio totals                                      │
│  Drill-down: Segment → Sub-Segment                              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    MICRO MODEL         MACRO OVERLAY       DRIFT DETECTOR
    (Per Segment)        (SEPARATE!)        (Both Levels)
```

### Key Design Decision: Macro as Overlay

**Macro is a SEPARATE layer, NOT merged into the model formula.**

```python
# MICRO: Predict base PD from borrower features
base_pd = micro_model.predict(features)  # 18 features, no macro

# MACRO: Adjust based on economic scenario (separate step)
adjusted_pd = base_pd * (1 + (unemployment_rate - 0.04) * sensitivity)
```

Why?
- No `issue_d` needed in data
- Swap scenarios without retraining
- Clean attribution: micro vs macro contribution

## Project Structure

```
driftbreaker/
├── src/
│   ├── config.py           # Constants, 18 features, LGD values
│   ├── data_loader.py      # Load parquet files
│   ├── survival_model.py   # Person-period hazard model
│   ├── macro_overlay.py    # SEPARATE macro adjustment
│   └── portfolio_view.py   # Roll-up / drill-down hierarchy
├── api/
│   └── index.py            # Vercel serverless API
├── run_pipeline.py         # Main orchestration
├── dashboard.py            # Streamlit app
├── requirements.txt
└── vercel.json
```

## Features (18 - No Leakage)

| # | Feature | Type |
|---|---------|------|
| 1 | collections_12_mths_ex_med | #1 predictor |
| 2 | avg_cur_bal | Credit utilization |
| 3 | tot_cur_bal | Debt burden |
| 4 | tot_hi_cred_lim | Credit capacity |
| 5 | bc_open_to_buy | Available credit |
| 6 | dti | Affordability |
| 7 | annual_inc | Income |
| 8 | loan_amnt | Loan size |
| 9 | acc_open_past_24mths | Credit seeking |
| 10 | num_tl_30dpd | Current distress |
| 11 | delinq_2yrs | Payment history |
| 12 | pub_rec | Public records |
| 13 | mths_since_recent_bc | Credit activity |
| 14 | credit_utilization_ratio | Engineered |
| 15 | income_to_loan_ratio | Engineered |
| 16 | has_collections | Binary flag |
| 17 | has_delinq_history | Binary flag |
| 18 | has_public_record | Binary flag |

**Excluded (leakage)**: `int_rate`, `installment`, `grade`

## API Usage

```bash
# Health check
curl https://your-app.vercel.app/api

# Predict
curl -X POST https://your-app.vercel.app/api \
  -H "Content-Type: application/json" \
  -d '{
    "segment": "medium_risk",
    "dti": 20,
    "annual_inc": 60000,
    "unemployment_rate": 0.06
  }'
```

## Deployment (Vercel)

```bash
# 1. Run pipeline locally
python run_pipeline.py

# 2. Commit artifacts
git add model_artifacts.json dashboard_*.csv
git commit -m "Update model artifacts"
git push

# Vercel auto-deploys on push
```

## Key Rules

✅ **DO**:
- Use parquet files directly
- Keep macro as separate overlay
- Train segment-specific models
- Expect 8-18% R² (realistic)

❌ **DON'T**:
- Use `int_rate`/`grade` as features (leakage)
- Merge macro into model formula
- Expect high R²
