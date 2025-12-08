# =============================================================================
# DRIFTBREAKER: MAIN PIPELINE
# =============================================================================
"""
Run this script to:
1. Load parquet data
2. Train segment models (micro - NO macro in formula)
3. Generate dashboard artifacts
4. Export lightweight model JSON for Vercel API

Usage:
    python run_pipeline.py
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def convert_to_native(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native(item) for item in obj]
    elif pd.isna(obj):
        return None
    return obj

# Add src to path
import sys
sys.path.insert(0, 'src')

from config import (
    FEATURES, SEGMENTS, LGD, SCENARIOS,
    BASELINE_UNEMPLOYMENT, MACRO_SENSITIVITY, MAX_MONTHS
)
from data_loader import load_all_segments, get_combined_data, load_macro_data
from survival_model import train_segment_models, predict_hazard_curve
from macro_overlay import MacroOverlay
from portfolio_view import PortfolioView


print("="*70)
print("DRIFTBREAKER PIPELINE")
print("="*70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1/5] LOADING DATA")
print("-"*50)

# Try multiple locations for parquet files
data_locations = ['.', 'data', '../data']
segment_data = None

for loc in data_locations:
    try:
        segment_data = load_all_segments(loc)
        print(f"  Found data in: {loc}")
        break
    except FileNotFoundError:
        continue

if segment_data is None:
    print("ERROR: Could not find parquet files!")
    print("  Expected: train_low_risk.parquet, train_medium_risk.parquet, train_high_risk.parquet")
    sys.exit(1)

combined = get_combined_data(segment_data)

# Load macro (for reference - NOT merged into model)
macro_df = load_macro_data('fred_macro.csv')


# =============================================================================
# 2. TRAIN MICRO MODELS (No Macro in Formula!)
# =============================================================================
print("\n[2/5] TRAINING MICRO MODELS")
print("-"*50)
print("  Note: Macro is a SEPARATE overlay, not in model formula")

model_results = train_segment_models(segment_data, FEATURES, sample_size=30000)

# Extract segment PDs
segment_pds = {
    seg: res['default_rate'] 
    for seg, res in model_results.items()
}


# =============================================================================
# 3. PORTFOLIO VIEW (Roll-Up / Drill-Down)
# =============================================================================
print("\n[3/5] PORTFOLIO ANALYSIS")
print("-"*50)

portfolio = PortfolioView(segment_data, segment_pds)
summary = portfolio.portfolio_summary()

print(f"\n  PORTFOLIO TOTALS:")
print(f"    Total Loans: {summary['portfolio']['total_loans']:,}")
print(f"    Total Exposure: ${summary['portfolio']['total_exposure']:,.0f}")
print(f"    Weighted PD: {summary['portfolio']['weighted_pd']:.1%}")
print(f"    Expected Loss: ${summary['portfolio']['total_expected_loss']:,.0f}")

print(f"\n  SEGMENT BREAKDOWN:")
for seg, data in summary['segments'].items():
    print(f"    {seg}: {data['pct_of_portfolio']:.1%} of portfolio, "
          f"PD={data['avg_pd']:.1%}, EL=${data['expected_loss']:,.0f}")


# =============================================================================
# 4. MACRO STRESS TEST (Separate Overlay)
# =============================================================================
print("\n[4/5] MACRO STRESS TEST")
print("-"*50)
print("  Macro applied as POST-PREDICTION overlay")

macro_overlay = MacroOverlay(
    baseline_unemployment=BASELINE_UNEMPLOYMENT,
    sensitivity=MACRO_SENSITIVITY
)

stress_results = macro_overlay.stress_test_portfolio(segment_data, segment_pds, SCENARIOS)

print(f"\n  SCENARIO RESULTS (Portfolio Level):")
for scenario, data in stress_results['portfolio'].items():
    print(f"    {scenario}: UE={data['unemployment_rate']:.0%} -> "
          f"Loss=${data['total_expected_loss']:,.0f} ({data['portfolio_loss_rate']:.1%})")


# =============================================================================
# 5. EXPORT ARTIFACTS
# =============================================================================
print("\n[5/5] EXPORTING ARTIFACTS")
print("-"*50)

# Create artifacts directory
os.makedirs('artifacts', exist_ok=True)

# A. Risk Curves (per segment)
print("  Exporting risk curves...")
all_curves = []

for segment, res in model_results.items():
    curve = res['hazard_curve'].copy()
    curve['segment'] = segment
    all_curves.append(curve)
    
    # Individual segment file
    curve.to_csv(f'artifacts/{segment}_risk_curve.csv', index=False)

# Combined curves
combined_curves = pd.concat(all_curves, ignore_index=True)
combined_curves.to_csv('artifacts/dashboard_risk_curves.csv', index=False)

# B. Finance/Scenario Results
print("  Exporting scenario results...")
scenario_rows = []

for scenario, port_data in stress_results['portfolio'].items():
    row = {
        'Scenario': scenario,
        'Unemployment': port_data['unemployment_rate'],
        'Total_Exposure': port_data['total_exposure'],
        'Expected_Loss': port_data['total_expected_loss'],
        'Loss_Rate': port_data['portfolio_loss_rate']
    }
    
    # Add segment detail
    for seg, seg_data in stress_results['segments'][scenario].items():
        row[f'{seg}_pd'] = seg_data['adjusted_pd']
        row[f'{seg}_loss'] = seg_data['expected_loss']
    
    scenario_rows.append(row)

finance_df = pd.DataFrame(scenario_rows)
finance_df.to_csv('artifacts/dashboard_finance.csv', index=False)

# Also save to root for Vercel
finance_df.to_csv('dashboard_finance.csv', index=False)

# C. Model Artifacts (Lightweight JSON for API)
print("  Exporting model artifacts (JSON)...")

# Collect all coefficients
all_coefficients = {}
for segment, res in model_results.items():
    all_coefficients[segment] = res['coefficients']

model_artifact = {
    "meta": {
        "version": "v4.0",
        "generated_at": datetime.now().isoformat(),
        "algorithm": "Discrete-Time Hazard (Logistic)",
        "n_features": len(FEATURES),
        "features": FEATURES,
        "segments": SEGMENTS
    },
    "coefficients": all_coefficients,
    "macro_overlay": {
        "baseline_unemployment": BASELINE_UNEMPLOYMENT,
        "sensitivity": MACRO_SENSITIVITY,
        "note": "Macro is SEPARATE layer, not in model formula"
    },
    "lgd": LGD,
    "portfolio_summary": {
        "total_loans": summary['portfolio']['total_loans'],
        "total_exposure": summary['portfolio']['total_exposure'],
        "weighted_pd": summary['portfolio']['weighted_pd']
    },
    "segment_pds": segment_pds
}

# Convert numpy types to native Python types
model_artifact = convert_to_native(model_artifact)

# Save to artifacts and root
with open('artifacts/model_artifacts.json', 'w') as f:
    json.dump(model_artifact, f, indent=2)

with open('model_artifacts.json', 'w') as f:
    json.dump(model_artifact, f, indent=2)

# D. Vitals (for dashboard)
print("  Exporting vitals...")

# Find peak hazard month
peak_months = {}
for seg, res in model_results.items():
    curve = res['hazard_curve']
    peak_months[seg] = int(curve.loc[curve['hazard'].idxmax(), 'month'])

vitals = {
    "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_version": model_artifact['meta']['version'],
    "total_loans": summary['portfolio']['total_loans'],
    "portfolio_pd": summary['portfolio']['weighted_pd'],
    "peak_hazard_months": peak_months,
    "drift_status": "Stable",
    "macro_overlay": "Separate Layer (NOT merged)"
}

vitals = convert_to_native(vitals)

with open('artifacts/dashboard_vitals.json', 'w') as f:
    json.dump(vitals, f, indent=2)

with open('dashboard_vitals.json', 'w') as f:
    json.dump(vitals, f, indent=2)

# E. LLM Context
print("  Exporting LLM context...")

llm_context = portfolio.to_llm_context()
llm_context['stress_test'] = {
    'scenarios': list(stress_results['portfolio'].keys()),
    'baseline_loss': stress_results['portfolio']['baseline']['total_expected_loss'],
    'severe_loss': stress_results['portfolio']['severe_recession']['total_expected_loss']
}
llm_context['model_info'] = model_artifact['meta']
llm_context = convert_to_native(llm_context)

with open('artifacts/llm_context.json', 'w') as f:
    json.dump(llm_context, f, indent=2)


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("PIPELINE COMPLETE")
print("="*70)

print("\nARTIFACTS GENERATED:")
print("  - artifacts/dashboard_risk_curves.csv")
print("  - artifacts/dashboard_finance.csv (+ root copy)")
print("  - artifacts/model_artifacts.json (+ root copy)")
print("  - artifacts/dashboard_vitals.json (+ root copy)")
print("  - artifacts/llm_context.json")

print("\nKEY METRICS:")
print(f"  Portfolio PD: {summary['portfolio']['weighted_pd']:.1%}")
print(f"  Expected Loss: ${summary['portfolio']['total_expected_loss']:,.0f}")
print(f"  Severe Scenario Loss: ${stress_results['portfolio']['severe_recession']['total_expected_loss']:,.0f}")

print("\nNEXT STEPS:")
print("  1. Run: streamlit run dashboard.py")
print("  2. Deploy: git push (Vercel will pick up model_artifacts.json)")

print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
