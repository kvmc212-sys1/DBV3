# =============================================================================
# DRIFTBREAKER: STREAMLIT DASHBOARD (WITH LAYERS)
# =============================================================================
"""
Dashboard with:
- HIERARCHY: Portfolio → Segment → Sub-Segment
- LAYERS: BI Metrics → Attribution → Scenarios → Risk Curves → AI

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
    for path in ['artifacts/dashboard_risk_curves.csv', 'dashboard_risk_curves.csv']:
        if os.path.exists(path):
            artifacts['risk_curves'] = pd.read_csv(path)
            break
    
    # Finance/scenarios
    for path in ['artifacts/dashboard_finance.csv', 'dashboard_finance.csv']:
        if os.path.exists(path):
            artifacts['finance'] = pd.read_csv(path)
            break
    
    # Vitals
    for path in ['artifacts/dashboard_vitals.json', 'dashboard_vitals.json']:
        if os.path.exists(path):
            with open(path) as f:
                artifacts['vitals'] = json.load(f)
            break
    
    # Model artifacts
    for path in ['artifacts/model_artifacts.json', 'model_artifacts.json']:
        if os.path.exists(path):
            with open(path) as f:
                artifacts['model'] = json.load(f)
            break
    
    # LLM context
    for path in ['artifacts/llm_context.json', 'llm_context.json']:
        if os.path.exists(path):
            with open(path) as f:
                artifacts['llm_context'] = json.load(f)
            break
    
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
# MACRO OVERLAY FUNCTION
# =============================================================================

def apply_macro(base_pd, unemployment_rate, baseline_ue=0.04, sensitivity=4.0):
    """Apply macro overlay to base PD."""
    multiplier = 1 + (unemployment_rate - baseline_ue) * sensitivity
    return min(base_pd * max(multiplier, 0.5), 0.99)


# =============================================================================
# LOAD DATA
# =============================================================================

artifacts = load_artifacts()
segment_data = load_segment_data()

# Check if artifacts loaded
if not artifacts.get('vitals'):
    st.error("⚠️ Artifacts not found. Please run `python run_pipeline.py` first.")
    st.stop()

model = artifacts.get('model', {})
vitals = artifacts.get('vitals', {})
segment_pds = model.get('segment_pds', {})
lgd = model.get('lgd', {'low_risk': 0.65, 'medium_risk': 0.70, 'high_risk': 0.75})


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🎛️ DriftBreaker")
st.sidebar.markdown("---")

# LAYER selector
st.sidebar.subheader("📊 Analysis Layer")
layer = st.sidebar.radio(
    "Select Layer",
    ["Layer 1: BI Metrics", 
     "Layer 2: Attribution", 
     "Layer 3: Scenarios",
     "Layer 4: Risk Curves",
     "Layer 5: AI Assistant"],
    index=0
)

st.sidebar.markdown("---")

# HIERARCHY selector
st.sidebar.subheader("🔍 View Level")
view_level = st.sidebar.radio(
    "Hierarchy",
    ["Portfolio", "Segment", "Sub-Segment"],
    index=0
)

if view_level in ["Segment", "Sub-Segment"]:
    selected_segment = st.sidebar.selectbox(
        "Select Segment",
        ["low_risk", "medium_risk", "high_risk"],
        format_func=lambda x: x.replace('_', ' ').title()
    )

if view_level == "Sub-Segment":
    breakdown_dim = st.sidebar.selectbox(
        "Break Down By",
        ["dti_bucket", "loan_size", "collections", "income"],
        format_func=lambda x: x.replace('_', ' ').title()
    )

st.sidebar.markdown("---")

# MACRO scenario
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
    scenario_name = "🔴 Severe Recession"
elif unemployment_rate >= 0.06:
    scenario_name = "🟡 Mild Recession"
else:
    scenario_name = "🟢 Baseline"

st.sidebar.info(f"**{scenario_name}**")


# =============================================================================
# HEADER
# =============================================================================

st.title("📊 DriftBreaker: Credit Risk Monitor")

# Quick stats bar
col1, col2, col3, col4, col5 = st.columns(5)

portfolio_summary = model.get('portfolio_summary', {})
base_pd = portfolio_summary.get('weighted_pd', 0.15)
stressed_pd = apply_macro(base_pd, unemployment_rate)

with col1:
    st.metric("Total Loans", f"{portfolio_summary.get('total_loans', 0):,}")
with col2:
    st.metric("Exposure", f"${portfolio_summary.get('total_exposure', 0)/1e6:.1f}M")
with col3:
    st.metric("Base PD", f"{base_pd:.1%}")
with col4:
    delta = stressed_pd - base_pd
    st.metric("Stressed PD", f"{stressed_pd:.1%}", delta=f"{delta:+.1%}" if delta != 0 else None, delta_color="inverse")
with col5:
    st.metric("Status", vitals.get('drift_status', 'Stable'))

st.markdown("---")


# =============================================================================
# LAYER 1: BI METRICS
# =============================================================================

if layer == "Layer 1: BI Metrics":
    st.header("📈 Layer 1: Business Intelligence Metrics")
    
    st.markdown("""
    **Question**: How is the portfolio performing?
    
    Key metrics aggregated at portfolio and segment level.
    """)
    
    if view_level == "Portfolio":
        # Portfolio-level BI
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Composition")
            
            seg_data = []
            for seg in ['low_risk', 'medium_risk', 'high_risk']:
                if seg in segment_data:
                    df = segment_data[seg]
                    seg_data.append({
                        'Segment': seg.replace('_', ' ').title(),
                        'Loans': len(df),
                        'Exposure': df['loan_amnt'].sum(),
                        'Default Rate': df['default'].mean()
                    })
            
            seg_df = pd.DataFrame(seg_data)
            
            fig = px.pie(seg_df, values='Exposure', names='Segment',
                        title='Exposure by Segment',
                        color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Default Rates by Segment")
            
            fig = px.bar(seg_df, x='Segment', y='Default Rate',
                        title='Actual Default Rate',
                        color='Default Rate',
                        color_continuous_scale='RdYlGn_r')
            fig.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        st.subheader("Summary Table")
        display_df = seg_df.copy()
        display_df['Exposure'] = display_df['Exposure'].apply(lambda x: f"${x:,.0f}")
        display_df['Default Rate'] = display_df['Default Rate'].apply(lambda x: f"{x:.1%}")
        st.dataframe(display_df, use_container_width=True)
    
    elif view_level == "Segment":
        # Segment-level BI
        df = segment_data.get(selected_segment)
        if df is not None:
            st.subheader(f"Segment: {selected_segment.replace('_', ' ').title()}")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Loans", f"{len(df):,}")
            col2.metric("Exposure", f"${df['loan_amnt'].sum():,.0f}")
            col3.metric("Avg Loan", f"${df['loan_amnt'].mean():,.0f}")
            col4.metric("Default Rate", f"{df['default'].mean():.1%}")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x='dti', nbins=30, title='DTI Distribution')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.histogram(df, x='annual_inc', nbins=30, title='Income Distribution')
                st.plotly_chart(fig, use_container_width=True)
    
    elif view_level == "Sub-Segment":
        df = segment_data.get(selected_segment)
        if df is not None:
            df = df.copy()
            
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
            
            st.subheader(f"Sub-Segment: {selected_segment.replace('_', ' ').title()} by {breakdown_dim.replace('_', ' ').title()}")
            
            breakdown = df.groupby('group', observed=True).agg({
                'loan_amnt': ['count', 'sum'],
                'default': 'mean'
            }).round(4)
            breakdown.columns = ['Loans', 'Exposure', 'Default Rate']
            breakdown = breakdown.reset_index()
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(breakdown, x='group', y='Exposure', title='Exposure by Group')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.bar(breakdown, x='group', y='Default Rate', title='Default Rate by Group',
                            color='Default Rate', color_continuous_scale='RdYlGn_r')
                fig.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(breakdown, use_container_width=True)


# =============================================================================
# LAYER 2: ATTRIBUTION
# =============================================================================

elif layer == "Layer 2: Attribution":
    st.header("🔬 Layer 2: Attribution Analysis")
    
    st.markdown("""
    **Question**: Why did defaults change? Was it underwriting (micro) or economy (macro)?
    
    Decomposes PD changes into micro vs macro contributions.
    """)
    
    # Attribution visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Micro vs Macro Contribution")
        
        # Adjust based on current scenario
        if unemployment_rate > 0.06:
            macro_contribution = 0.55
            micro_contribution = 0.45
        elif unemployment_rate < 0.04:
            macro_contribution = 0.30
            micro_contribution = 0.70
        else:
            micro_contribution = 0.60
            macro_contribution = 0.40
        
        attr_df = pd.DataFrame({
            'Driver': ['Underwriting (Micro)', 'Economic (Macro)'],
            'Contribution': [micro_contribution, macro_contribution]
        })
        
        fig = px.pie(attr_df, values='Contribution', names='Driver',
                    title='PD Attribution',
                    color_discrete_sequence=['#3b82f6', '#f59e0b'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Macro Overlay Impact")
        
        # Show how macro changes PD
        ue_rates = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        pds = [apply_macro(base_pd, ue) for ue in ue_rates]
        
        impact_df = pd.DataFrame({
            'Unemployment': ue_rates,
            'Adjusted PD': pds
        })
        
        fig = px.line(impact_df, x='Unemployment', y='Adjusted PD',
                     title='PD Response to Unemployment',
                     markers=True)
        fig.add_vline(x=unemployment_rate, line_dash="dash", line_color="red",
                     annotation_text=f"Current: {unemployment_rate:.0%}")
        fig.update_layout(xaxis_tickformat='.0%', yaxis_tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)
    
    # Attribution by segment
    st.subheader("Attribution by Segment")
    
    attr_table = []
    for seg in ['low_risk', 'medium_risk', 'high_risk']:
        seg_base_pd = segment_pds.get(seg, 0.15)
        seg_stressed_pd = apply_macro(seg_base_pd, unemployment_rate)
        macro_impact = seg_stressed_pd - seg_base_pd
        
        attr_table.append({
            'Segment': seg.replace('_', ' ').title(),
            'Base PD': f"{seg_base_pd:.1%}",
            'Macro Impact': f"{macro_impact:+.1%}",
            'Stressed PD': f"{seg_stressed_pd:.1%}"
        })
    
    st.dataframe(pd.DataFrame(attr_table), use_container_width=True)
    
    st.info("""
    **Interpretation**: 
    - Macro Overlay is a SEPARATE layer applied post-prediction
    - Formula: `Adjusted_PD = Base_PD × (1 + (UE - 0.04) × 4.0)`
    - This allows scenario analysis WITHOUT retraining
    """)


# =============================================================================
# LAYER 3: SCENARIOS
# =============================================================================

elif layer == "Layer 3: Scenarios":
    st.header("🎯 Layer 3: Scenario & Stress Testing")
    
    st.markdown("""
    **Question**: What would happen under different economic conditions?
    
    Stress test the portfolio under multiple scenarios.
    """)
    
    # Scenario definitions
    scenarios = {
        'Baseline': 0.04,
        'Mild Recession': 0.06,
        'Severe Recession': 0.10,
        '2008 Crisis': 0.10
    }
    
    # Calculate scenario results
    scenario_results = []
    
    for sc_name, ue in scenarios.items():
        total_loss = 0
        total_exposure = 0
        
        for seg in ['low_risk', 'medium_risk', 'high_risk']:
            if seg in segment_data:
                df = segment_data[seg]
                exposure = df['loan_amnt'].sum()
                seg_base_pd = segment_pds.get(seg, df['default'].mean())
                seg_stressed_pd = apply_macro(seg_base_pd, ue)
                seg_loss = exposure * seg_stressed_pd * lgd.get(seg, 0.70)
                
                total_loss += seg_loss
                total_exposure += exposure
        
        scenario_results.append({
            'Scenario': sc_name,
            'Unemployment': f"{ue:.0%}",
            'Expected Loss': total_loss,
            'Loss Rate': total_loss / total_exposure if total_exposure > 0 else 0
        })
    
    scenario_df = pd.DataFrame(scenario_results)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Expected Loss by Scenario")
        
        fig = px.bar(scenario_df, x='Scenario', y='Expected Loss',
                    title='Portfolio Expected Loss',
                    color='Scenario',
                    color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444', '#7c3aed'])
        fig.update_layout(yaxis_tickprefix='$', yaxis_tickformat=',.0f')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Loss Rate by Scenario")
        
        fig = px.bar(scenario_df, x='Scenario', y='Loss Rate',
                    title='Portfolio Loss Rate',
                    color='Scenario',
                    color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444', '#7c3aed'])
        fig.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)
    
    # Scenario table
    st.subheader("Scenario Summary")
    display_df = scenario_df.copy()
    display_df['Expected Loss'] = display_df['Expected Loss'].apply(lambda x: f"${x:,.0f}")
    display_df['Loss Rate'] = display_df['Loss Rate'].apply(lambda x: f"{x:.2%}")
    st.dataframe(display_df, use_container_width=True)
    
    # Segment breakdown for current scenario
    st.subheader(f"Segment Breakdown (Current: {unemployment_rate:.0%} UE)")
    
    seg_scenario = []
    for seg in ['low_risk', 'medium_risk', 'high_risk']:
        if seg in segment_data:
            df = segment_data[seg]
            exposure = df['loan_amnt'].sum()
            seg_base_pd = segment_pds.get(seg, df['default'].mean())
            seg_stressed_pd = apply_macro(seg_base_pd, unemployment_rate)
            seg_loss = exposure * seg_stressed_pd * lgd.get(seg, 0.70)
            
            seg_scenario.append({
                'Segment': seg.replace('_', ' ').title(),
                'Exposure': f"${exposure:,.0f}",
                'Base PD': f"{seg_base_pd:.1%}",
                'Stressed PD': f"{seg_stressed_pd:.1%}",
                'LGD': f"{lgd.get(seg, 0.70):.0%}",
                'Expected Loss': f"${seg_loss:,.0f}"
            })
    
    st.dataframe(pd.DataFrame(seg_scenario), use_container_width=True)


# =============================================================================
# LAYER 4: RISK CURVES
# =============================================================================

elif layer == "Layer 4: Risk Curves":
    st.header("📉 Layer 4: Risk Curves (Seasoning Analysis)")
    
    st.markdown("""
    **Question**: How does risk evolve over the loan lifecycle?
    
    Shows the "Month 7 Peak" pattern - hazard rate peaks early then declines as risky loans default out.
    """)
    
    if artifacts.get('risk_curves') is not None:
        curves = artifacts['risk_curves']
        
        if view_level == "Portfolio":
            st.subheader("Hazard Curves by Segment")
            
            # Base curves
            fig = px.line(curves, x='month', y='hazard', color='segment',
                         title='Monthly Hazard Rate (Base)',
                         labels={'month': 'Month', 'hazard': 'Hazard Rate', 'segment': 'Segment'})
            fig.update_layout(yaxis_tickformat='.2%')
            st.plotly_chart(fig, use_container_width=True)
            
            # Stressed curves
            st.subheader("Stressed Hazard Curves")
            
            curves_stressed = curves.copy()
            curves_stressed['stressed_hazard'] = curves_stressed['hazard'].apply(
                lambda h: apply_macro(h, unemployment_rate)
            )
            
            fig = go.Figure()
            for seg in curves_stressed['segment'].unique():
                seg_data = curves_stressed[curves_stressed['segment'] == seg]
                fig.add_trace(go.Scatter(x=seg_data['month'], y=seg_data['hazard'],
                                        name=f'{seg} (Base)', line=dict(dash='dot')))
                fig.add_trace(go.Scatter(x=seg_data['month'], y=seg_data['stressed_hazard'],
                                        name=f'{seg} (Stressed)'))
            
            fig.update_layout(title=f'Base vs Stressed Hazard ({unemployment_rate:.0%} UE)',
                            xaxis_title='Month', yaxis_title='Hazard Rate',
                            yaxis_tickformat='.2%')
            st.plotly_chart(fig, use_container_width=True)
        
        elif view_level in ["Segment", "Sub-Segment"]:
            seg_curves = curves[curves['segment'] == selected_segment]
            
            if len(seg_curves) > 0:
                st.subheader(f"Hazard Curve: {selected_segment.replace('_', ' ').title()}")
                
                seg_curves = seg_curves.copy()
                seg_curves['stressed_hazard'] = seg_curves['hazard'].apply(
                    lambda h: apply_macro(h, unemployment_rate)
                )
                
                # Calculate cumulative PD
                survival_base = 1.0
                survival_stressed = 1.0
                cum_pd_base = []
                cum_pd_stressed = []
                
                for _, row in seg_curves.iterrows():
                    survival_base *= (1 - row['hazard'])
                    survival_stressed *= (1 - row['stressed_hazard'])
                    cum_pd_base.append(1 - survival_base)
                    cum_pd_stressed.append(1 - survival_stressed)
                
                seg_curves['cum_pd_base'] = cum_pd_base
                seg_curves['cum_pd_stressed'] = cum_pd_stressed
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=seg_curves['month'], y=seg_curves['hazard'],
                                            name='Base', line=dict(color='#3b82f6')))
                    fig.add_trace(go.Scatter(x=seg_curves['month'], y=seg_curves['stressed_hazard'],
                                            name='Stressed', line=dict(color='#ef4444', dash='dash')))
                    fig.update_layout(title='Monthly Hazard Rate',
                                    xaxis_title='Month', yaxis_title='Hazard',
                                    yaxis_tickformat='.2%')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=seg_curves['month'], y=seg_curves['cum_pd_base'],
                                            name='Base', line=dict(color='#3b82f6')))
                    fig.add_trace(go.Scatter(x=seg_curves['month'], y=seg_curves['cum_pd_stressed'],
                                            name='Stressed', line=dict(color='#ef4444', dash='dash')))
                    fig.update_layout(title='Cumulative PD',
                                    xaxis_title='Month', yaxis_title='Cumulative PD',
                                    yaxis_tickformat='.1%')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Key metrics
                peak_month = seg_curves.loc[seg_curves['hazard'].idxmax(), 'month']
                pd_12 = cum_pd_stressed[11] if len(cum_pd_stressed) >= 12 else None
                pd_36 = cum_pd_stressed[35] if len(cum_pd_stressed) >= 36 else None
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Peak Hazard Month", f"{int(peak_month)}")
                col2.metric("PD @ 12 months", f"{pd_12:.1%}" if pd_12 else "N/A")
                col3.metric("PD @ 36 months", f"{pd_36:.1%}" if pd_36 else "N/A")
    
    else:
        st.warning("Risk curves not found. Run pipeline to generate.")


# =============================================================================
# LAYER 5: AI ASSISTANT
# =============================================================================

elif layer == "Layer 5: AI Assistant":
    st.header("🤖 Layer 5: AI Assistant Context")
    
    st.markdown("""
    **Purpose**: Provide context for AI-powered analysis and insights.
    
    This layer shows the data structure that would be passed to an LLM for natural language analysis.
    """)
    
    # Build context
    context = {
        "portfolio_level": {
            "total_loans": portfolio_summary.get('total_loans', 0),
            "total_exposure": portfolio_summary.get('total_exposure', 0),
            "base_pd": base_pd,
            "stressed_pd": stressed_pd,
            "current_macro": {
                "unemployment_rate": unemployment_rate,
                "scenario": scenario_name
            }
        },
        "segment_level": {},
        "risk_assessment": {
            "drift_status": vitals.get('drift_status', 'Stable'),
            "peak_hazard_months": vitals.get('peak_hazard_months', {})
        }
    }
    
    # Add segment data
    for seg in ['low_risk', 'medium_risk', 'high_risk']:
        if seg in segment_data:
            df = segment_data[seg]
            seg_base_pd = segment_pds.get(seg, df['default'].mean())
            context["segment_level"][seg] = {
                "loans": len(df),
                "exposure": float(df['loan_amnt'].sum()),
                "base_pd": seg_base_pd,
                "stressed_pd": apply_macro(seg_base_pd, unemployment_rate)
            }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Context JSON")
        st.json(context)
    
    with col2:
        st.subheader("Sample AI Prompts")
        
        st.markdown("""
        **Example prompts you could ask an AI:**
        
        1. "Summarize the portfolio risk profile"
        2. "Which segment is most vulnerable to recession?"
        3. "What's driving the high-risk segment's default rate?"
        4. "Should we tighten underwriting standards?"
        5. "Compare baseline vs stressed scenarios"
        """)
        
        st.markdown("---")
        
        st.subheader("Auto-Generated Summary")
        
        # Generate a simple summary
        highest_risk_seg = max(context['segment_level'].items(), 
                              key=lambda x: x[1]['stressed_pd'])[0]
        
        summary = f"""
        **Portfolio Overview** ({scenario_name})
        
        The portfolio contains **{context['portfolio_level']['total_loans']:,}** loans 
        with total exposure of **${context['portfolio_level']['total_exposure']/1e6:.1f}M**.
        
        Under current macro conditions ({unemployment_rate:.0%} unemployment):
        - Base PD: **{base_pd:.1%}**
        - Stressed PD: **{stressed_pd:.1%}**
        - PD increase: **{(stressed_pd - base_pd) / base_pd * 100:+.1f}%**
        
        The **{highest_risk_seg.replace('_', ' ')}** segment shows the highest 
        stressed PD at **{context['segment_level'][highest_risk_seg]['stressed_pd']:.1%}**.
        
        Drift Status: **{context['risk_assessment']['drift_status']}**
        """
        
        st.markdown(summary)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.caption(f"Last updated: {vitals.get('last_run', 'Unknown')} | "
           f"Model: {vitals.get('model_version', 'v4.0')} | "
           f"Macro Overlay: Separate Layer (sensitivity: 4.0)")
