# =============================================================================
# DRIFTBREAKER: STREAMLIT DASHBOARD
# =============================================================================
"""
Dashboard with hierarchy selector: Portfolio → Segment → Sub-Segment

Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# Page config
st.set_page_config(
    page_title="DriftBreaker",
    page_icon="📊",
    layout="wide"
)

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_artifacts():
    """Load pre-computed artifacts from pipeline."""
    artifacts = {}
    
    # Risk curves
    try:
        artifacts['risk_curves'] = pd.read_csv('artifacts/dashboard_risk_curves.csv')
    except:
        artifacts['risk_curves'] = pd.read_csv('dashboard_risk_curves.csv') if os.path.exists('dashboard_risk_curves.csv') else None
    
    # Finance/scenarios
    try:
        artifacts['finance'] = pd.read_csv('artifacts/dashboard_finance.csv')
    except:
        artifacts['finance'] = pd.read_csv('dashboard_finance.csv') if os.path.exists('dashboard_finance.csv') else None
    
    # Vitals
    try:
        with open('artifacts/dashboard_vitals.json') as f:
            artifacts['vitals'] = json.load(f)
    except:
        try:
            with open('dashboard_vitals.json') as f:
                artifacts['vitals'] = json.load(f)
        except:
            artifacts['vitals'] = None
    
    # Model artifacts
    try:
        with open('artifacts/model_artifacts.json') as f:
            artifacts['model'] = json.load(f)
    except:
        try:
            with open('model_artifacts.json') as f:
                artifacts['model'] = json.load(f)
        except:
            artifacts['model'] = None
    
    # LLM context
    try:
        with open('artifacts/llm_context.json') as f:
            artifacts['llm_context'] = json.load(f)
    except:
        artifacts['llm_context'] = None
    
    return artifacts


@st.cache_data
def load_segment_data():
    """Load raw segment data for drill-down analysis."""
    data = {}
    for segment in ['low_risk', 'medium_risk', 'high_risk']:
        for path in [f'train_{segment}.parquet', f'data/train_{segment}.parquet']:
            if os.path.exists(path):
                data[segment] = pd.read_parquet(path)
                break
    return data


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🎛️ DriftBreaker")
st.sidebar.markdown("---")

# Hierarchy selector
view_level = st.sidebar.radio(
    "View Level",
    ["Portfolio", "Segment", "Sub-Segment"],
    index=0
)

# Segment selector (for segment and sub-segment views)
if view_level in ["Segment", "Sub-Segment"]:
    selected_segment = st.sidebar.selectbox(
        "Select Segment",
        ["low_risk", "medium_risk", "high_risk"],
        format_func=lambda x: x.replace('_', ' ').title()
    )

# Sub-segment dimension (for sub-segment view)
if view_level == "Sub-Segment":
    breakdown_dim = st.sidebar.selectbox(
        "Break Down By",
        ["dti_bucket", "loan_size", "collections", "income"],
        format_func=lambda x: x.replace('_', ' ').title()
    )

st.sidebar.markdown("---")

# Macro scenario selector
st.sidebar.subheader("🌍 Macro Scenario")
unemployment_rate = st.sidebar.slider(
    "Unemployment Rate",
    min_value=0.03,
    max_value=0.12,
    value=0.04,
    step=0.01,
    format="%.0f%%"
)

scenario_name = "Baseline"
if unemployment_rate >= 0.08:
    scenario_name = "Severe Recession"
elif unemployment_rate >= 0.06:
    scenario_name = "Mild Recession"

st.sidebar.info(f"Scenario: **{scenario_name}**")


# =============================================================================
# LOAD DATA
# =============================================================================

artifacts = load_artifacts()
segment_data = load_segment_data()


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.title("📊 DriftBreaker: Credit Risk Monitor")

# Check if artifacts loaded
if artifacts['vitals'] is None:
    st.error("⚠️ Artifacts not found. Please run `python run_pipeline.py` first.")
    st.stop()


# -----------------------------------------------------------------------------
# PORTFOLIO VIEW
# -----------------------------------------------------------------------------

if view_level == "Portfolio":
    st.header("Portfolio Overview")
    
    # Top metrics
    model = artifacts.get('model', {})
    vitals = artifacts.get('vitals', {})
    portfolio_summary = model.get('portfolio_summary', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Loans",
            f"{portfolio_summary.get('total_loans', 0):,}"
        )
    
    with col2:
        st.metric(
            "Total Exposure",
            f"${portfolio_summary.get('total_exposure', 0):,.0f}"
        )
    
    with col3:
        base_pd = portfolio_summary.get('weighted_pd', 0)
        # Apply macro overlay
        multiplier = 1 + (unemployment_rate - 0.04) * 4
        stressed_pd = min(base_pd * multiplier, 0.99)
        delta = stressed_pd - base_pd
        
        st.metric(
            "Portfolio PD",
            f"{stressed_pd:.1%}",
            delta=f"{delta:+.1%}" if delta != 0 else None,
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            "Model Version",
            vitals.get('model_version', 'v4.0')
        )
    
    st.markdown("---")
    
    # Segment breakdown
    st.subheader("Segment Breakdown")
    
    segment_pds = model.get('segment_pds', {})
    lgd = model.get('lgd', {})
    
    seg_data = []
    for seg in ['low_risk', 'medium_risk', 'high_risk']:
        if seg in segment_data:
            df = segment_data[seg]
            base_pd = segment_pds.get(seg, df['default'].mean())
            stressed_pd = min(base_pd * multiplier, 0.99)
            exposure = df['loan_amnt'].sum()
            expected_loss = exposure * stressed_pd * lgd.get(seg, 0.70)
            
            seg_data.append({
                'Segment': seg.replace('_', ' ').title(),
                'Loans': len(df),
                'Exposure': exposure,
                'Base PD': base_pd,
                'Stressed PD': stressed_pd,
                'Expected Loss': expected_loss
            })
    
    seg_df = pd.DataFrame(seg_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Exposure pie chart
        fig = px.pie(
            seg_df, 
            values='Exposure', 
            names='Segment',
            title='Exposure by Segment',
            color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # PD bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Base PD',
            x=seg_df['Segment'],
            y=seg_df['Base PD'],
            marker_color='#3b82f6'
        ))
        fig.add_trace(go.Bar(
            name='Stressed PD',
            x=seg_df['Segment'],
            y=seg_df['Stressed PD'],
            marker_color='#ef4444'
        ))
        fig.update_layout(
            title='PD by Segment (Base vs Stressed)',
            barmode='group',
            yaxis_tickformat='.1%'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk curves
    st.subheader("Hazard Curves by Segment")
    
    if artifacts['risk_curves'] is not None:
        curves = artifacts['risk_curves']
        
        fig = px.line(
            curves,
            x='month',
            y='hazard',
            color='segment',
            title='Monthly Hazard Rate by Segment',
            labels={'month': 'Month', 'hazard': 'Hazard Rate', 'segment': 'Segment'}
        )
        fig.update_layout(yaxis_tickformat='.2%')
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# SEGMENT VIEW
# -----------------------------------------------------------------------------

elif view_level == "Segment":
    st.header(f"Segment: {selected_segment.replace('_', ' ').title()}")
    
    if selected_segment not in segment_data:
        st.error(f"No data for segment: {selected_segment}")
        st.stop()
    
    df = segment_data[selected_segment]
    model = artifacts.get('model', {})
    segment_pds = model.get('segment_pds', {})
    lgd = model.get('lgd', {})
    
    base_pd = segment_pds.get(selected_segment, df['default'].mean())
    multiplier = 1 + (unemployment_rate - 0.04) * 4
    stressed_pd = min(base_pd * multiplier, 0.99)
    exposure = df['loan_amnt'].sum()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Loans", f"{len(df):,}")
    
    with col2:
        st.metric("Exposure", f"${exposure:,.0f}")
    
    with col3:
        st.metric(
            "Segment PD",
            f"{stressed_pd:.1%}",
            delta=f"{stressed_pd - base_pd:+.1%}" if unemployment_rate != 0.04 else None,
            delta_color="inverse"
        )
    
    with col4:
        el = exposure * stressed_pd * lgd.get(selected_segment, 0.70)
        st.metric("Expected Loss", f"${el:,.0f}")
    
    st.markdown("---")
    
    # Segment-specific hazard curve
    if artifacts['risk_curves'] is not None:
        curves = artifacts['risk_curves']
        seg_curve = curves[curves['segment'] == selected_segment]
        
        if len(seg_curve) > 0:
            st.subheader("Hazard Curve")
            
            # Apply macro to curve
            seg_curve = seg_curve.copy()
            seg_curve['stressed_hazard'] = (seg_curve['hazard'] * multiplier).clip(upper=0.99)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=seg_curve['month'],
                y=seg_curve['hazard'],
                name='Base Hazard',
                line=dict(color='#3b82f6')
            ))
            fig.add_trace(go.Scatter(
                x=seg_curve['month'],
                y=seg_curve['stressed_hazard'],
                name='Stressed Hazard',
                line=dict(color='#ef4444', dash='dash')
            ))
            fig.update_layout(
                title=f'Hazard Curve: {selected_segment.replace("_", " ").title()}',
                xaxis_title='Month',
                yaxis_title='Hazard Rate',
                yaxis_tickformat='.2%'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='dti', nbins=30, title='DTI Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x='annual_inc', nbins=30, title='Income Distribution')
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# SUB-SEGMENT VIEW
# -----------------------------------------------------------------------------

elif view_level == "Sub-Segment":
    st.header(f"Sub-Segment Analysis: {selected_segment.replace('_', ' ').title()}")
    st.subheader(f"Breakdown by: {breakdown_dim.replace('_', ' ').title()}")
    
    if selected_segment not in segment_data:
        st.error(f"No data for segment: {selected_segment}")
        st.stop()
    
    df = segment_data[selected_segment].copy()
    
    # Create grouping
    if breakdown_dim == 'dti_bucket':
        df['group'] = pd.cut(df['dti'], bins=[0, 15, 25, 35, 100], 
                             labels=['Low (<15)', 'Mid (15-25)', 'High (25-35)', 'Very High (35+)'])
    elif breakdown_dim == 'loan_size':
        df['group'] = pd.cut(df['loan_amnt'], bins=[0, 10000, 20000, 35000, 100000], 
                             labels=['Small (<$10K)', 'Medium ($10-20K)', 'Large ($20-35K)', 'Jumbo (>$35K)'])
    elif breakdown_dim == 'collections':
        df['group'] = df['has_collections'].map({0: 'No Collections', 1: 'Has Collections'})
    elif breakdown_dim == 'income':
        df['group'] = pd.cut(df['annual_inc'], bins=[0, 50000, 75000, 100000, 1000000],
                             labels=['<$50K', '$50-75K', '$75-100K', '>$100K'])
    
    # Aggregate
    breakdown = df.groupby('group', observed=True).agg({
        'loan_amnt': ['count', 'sum'],
        'default': 'mean',
        'dti': 'mean'
    }).round(4)
    breakdown.columns = ['Loans', 'Exposure', 'Default Rate', 'Avg DTI']
    breakdown = breakdown.reset_index()
    
    # Display table
    st.dataframe(
        breakdown.style.format({
            'Exposure': '${:,.0f}',
            'Default Rate': '{:.1%}',
            'Avg DTI': '{:.1f}'
        }),
        use_container_width=True
    )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            breakdown,
            x='group',
            y='Exposure',
            title='Exposure by Group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            breakdown,
            x='group',
            y='Default Rate',
            title='Default Rate by Group',
            color='Default Rate',
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.caption(f"Last updated: {artifacts['vitals'].get('last_run', 'Unknown')} | "
           f"Macro Overlay: Separate Layer (UE sensitivity: 4.0)")
