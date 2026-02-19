#!/usr/bin/env python3
"""
Fabric SKU Advisor - CLI
Analyse Microsoft Fabric capacity consumption from CSV and generate an HTML report.

Usage:
    python fabric_sku_advisor.py -i data.csv -o report.html --capacity-name "My Capacity"

CSV format: Date, Item Name, Item Type, CUs

Author: Prathy Kamasani | Data Nova (www.data-nova.io)
"""

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("[WARNING] plotly not installed. Install with: pip install plotly")
    print("          HTML report generation requires plotly.")


# ============================================================================
# SECTION 2: SKU DEFINITIONS (match notebook exactly)
# ============================================================================

SKUS = [
    {"name": "F2",    "cus_per_second": 2,    "budget_30s": 60,    "monthly_usd": 262,    "monthly_reserved_usd": 155},
    {"name": "F4",    "cus_per_second": 4,    "budget_30s": 120,   "monthly_usd": 525,    "monthly_reserved_usd": 310},
    {"name": "F8",    "cus_per_second": 8,    "budget_30s": 240,   "monthly_usd": 1049,   "monthly_reserved_usd": 619},
    {"name": "F16",   "cus_per_second": 16,   "budget_30s": 480,   "monthly_usd": 2099,   "monthly_reserved_usd": 1239},
    {"name": "F32",   "cus_per_second": 32,   "budget_30s": 960,   "monthly_usd": 4198,   "monthly_reserved_usd": 2477},
    {"name": "F64",   "cus_per_second": 64,   "budget_30s": 1920,  "monthly_usd": 8395,   "monthly_reserved_usd": 4954},
    {"name": "F128",  "cus_per_second": 128,  "budget_30s": 3840,  "monthly_usd": 16790,  "monthly_reserved_usd": 9907},
    {"name": "F256",  "cus_per_second": 256,  "budget_30s": 7680,  "monthly_usd": 33580,  "monthly_reserved_usd": 19813},
    {"name": "F512",  "cus_per_second": 512,  "budget_30s": 15360, "monthly_usd": 67161,  "monthly_reserved_usd": 39625},
    {"name": "F1024", "cus_per_second": 1024, "budget_30s": 30720, "monthly_usd": 134321, "monthly_reserved_usd": 79250},
    {"name": "F2048", "cus_per_second": 2048, "budget_30s": 61440, "monthly_usd": 268643, "monthly_reserved_usd": 158499},
]

TARGET_UTILISATION = 0.80
RESERVED_DISCOUNT_PCT = 41
RESERVED_BREAKEVEN_UTIL = 0.60

# Item Type -> Operation Type mapping (from Fabric documentation)
OPERATION_TYPE_MAP = {
    "Semantic Model": "Interactive",
    "Report": "Interactive",
    "Report Views": "Interactive",
    "Dashboard": "Interactive",
    "Paginated Report": "Interactive",
    "SQL Database": "Interactive",
    "Database": "Interactive",
    "GraphQL": "Interactive",
    "Dataflow Gen1": "Background",
    "Dataflow Gen2": "Background",
    "Data Pipeline": "Background",
    "Pipeline": "Background",
    "Notebook": "Background",
    "Spark Job Definition": "Background",
    "Lakehouse": "Background",
    "Warehouse": "Background",
    "SQL Endpoint": "Background",
    "Eventhouse": "Background",
    "KQL Database": "Background",
    "KQL Queryset": "Background",
    "Eventstream": "Background",
    "OneLake": "Background",
    "Copilot": "Background",
    "AI": "Background",
}

DISCLAIMER_TEXT = (
    "DISCLAIMER: This analysis is based on the provided consumption data. "
    "Recommendations are indicative, not prescriptive. This tool can make mistakes. "
    "Key limitations: (1) Spark Autoscale workloads billed separately are NOT included. "
    "(2) Pricing shown uses published PAYG/reserved list prices (USD) which vary by region, "
    "currency, and agreement. (3) CSV data may not capture throttling or carryforward metrics "
    "available in the Capacity Metrics semantic model. "
    "Always validate recommendations against your own workload knowledge and business context "
    "before making capacity changes."
)


# ============================================================================
# SECTION 3: CORE HELPER FUNCTIONS (match notebook cell 8 EXACTLY)
# ============================================================================

def calculate_utilisation(daily_cus, sku):
    """Calculate utilisation as a decimal (0.0-1.0+). Matches notebook."""
    cus_per_window = daily_cus / 2880
    return cus_per_window / sku["budget_30s"]


def calculate_required_budget(daily_cus, target_utilisation=0.80):
    """Calculate required 30s budget to achieve target utilisation. Matches notebook."""
    cus_per_window = daily_cus / 2880
    return cus_per_window / target_utilisation


def get_sku_status(avg_util, max_util, needs_free_viewers, sku):
    """Determine SKU status label based on utilisation and constraints. Matches notebook."""
    if needs_free_viewers and sku["cus_per_second"] < 64:
        return "NO FREE VIEWERS"
    if max_util > 1.0:
        return "THROTTLING RISK"
    if avg_util > 0.95:
        return "TOO SMALL"
    if avg_util > 0.85:
        return "TIGHT"
    if avg_util > 0.60:
        return "GOOD FIT"
    if avg_util > 0.40:
        return "COMFORTABLE"
    return "OVERSIZED"


def calculate_health_score(avg_util_pct, throttle_pct=0.0, carryover_pct=0.0):
    """Composite capacity health score (0-100). Matches notebook exactly."""
    # Utilisation component (40% weight)
    if avg_util_pct <= 70:
        util_score = 100
    elif avg_util_pct <= 85:
        util_score = 100 - ((avg_util_pct - 70) * 2)
    elif avg_util_pct <= 100:
        util_score = 70 - ((avg_util_pct - 85) * 3)
    else:
        util_score = max(0, 25 - ((avg_util_pct - 100) * 5))

    # Throttling component (40% weight)
    if throttle_pct == 0:
        throttle_score = 100
    elif throttle_pct < 5:
        throttle_score = 80
    elif throttle_pct < 15:
        throttle_score = 50
    elif throttle_pct < 30:
        throttle_score = 25
    else:
        throttle_score = 0

    # Carryover component (20% weight)
    if carryover_pct == 0:
        carry_score = 100
    elif carryover_pct < 10:
        carry_score = 80
    elif carryover_pct < 30:
        carry_score = 50
    elif carryover_pct < 50:
        carry_score = 25
    else:
        carry_score = 0

    score = round(util_score * 0.4 + throttle_score * 0.4 + carry_score * 0.2)

    if score >= 90:
        rating = "EXCELLENT"
    elif score >= 75:
        rating = "GOOD"
    elif score >= 50:
        rating = "FAIR"
    elif score >= 25:
        rating = "POOR"
    else:
        rating = "CRITICAL"

    return score, rating


def format_duration(minutes):
    """Format minutes into human-readable duration string."""
    if minutes < 60:
        return f"{minutes:.0f} min"
    elif minutes < 1440:
        return f"{minutes / 60:.1f} hours"
    else:
        return f"{minutes / 1440:.1f} days"


# ============================================================================
# SECTION 4: TREND ANALYSIS AND SPIKE DETECTION (match notebook cell 10)
# ============================================================================

def calculate_trend(daily_summary, column='ActualCUs_sum'):
    """Linear trend and growth rate. Matches notebook."""
    result = {'has_trend': False}
    vals = daily_summary[column].dropna().values
    if len(vals) < 3:
        return result

    x = np.arange(len(vals), dtype=float)
    y = vals.astype(float)

    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)

    denom = n * sum_x2 - sum_x ** 2
    if denom == 0:
        return result

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = round(1 - (ss_res / ss_tot), 3) if ss_tot > 0 else 0

    mean_y = np.mean(y)
    daily_growth = (slope / mean_y * 100) if mean_y > 0 else 0
    weekly_growth = round(daily_growth * 7, 2)

    if weekly_growth > 2:
        direction = "GROWING"
    elif weekly_growth < -2:
        direction = "DECLINING"
    else:
        direction = "STABLE"

    return {
        'has_trend': True,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'daily_growth_pct': round(daily_growth, 2),
        'weekly_growth_pct': weekly_growth,
        'direction': direction,
    }


def detect_pause_spikes(daily_summary, threshold_factor=2.0):
    """Detect days with anomalously high CUs (pause/resume spikes). Matches notebook."""
    col = 'ActualCUs_sum'  # CLI uses total CUs (notebook uses ActualCUs_max)
    if col not in daily_summary.columns:
        return pd.Series([False] * len(daily_summary), index=daily_summary.index)
    vals = pd.to_numeric(daily_summary[col], errors='coerce').fillna(0)
    median_val = vals.median()
    if median_val <= 0:
        return pd.Series([False] * len(daily_summary), index=daily_summary.index)
    return vals > (median_val * threshold_factor)


# ============================================================================
# SECTION 5: CSV LOADER AND DAILY SUMMARY BUILDER
# ============================================================================

def load_csv_and_build_summary(filepath):
    """
    Load item-level CSV and build daily_summary matching notebook schema.
    CSV format: Date, Item Name, Item Type, CUs
    Returns (daily_summary DataFrame, raw DataFrame).
    """
    df = pd.read_csv(filepath)

    # Validate columns
    required = {'Date', 'Item Name', 'Item Type', 'CUs'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Expected: Date, Item Name, Item Type, CUs")

    df['Date'] = pd.to_datetime(df['Date'])
    df['CUs'] = pd.to_numeric(df['CUs'], errors='coerce').fillna(0)

    # Map operation types
    df['OperationType'] = df['Item Type'].map(OPERATION_TYPE_MAP).fillna('Background')

    # Daily totals
    daily = df.groupby('Date').agg(ActualCUs_sum=('CUs', 'sum')).reset_index()

    # Interactive/Background split by date
    int_by_date = df[df['OperationType'] == 'Interactive'].groupby('Date')['CUs'].sum()
    bg_by_date = df[df['OperationType'] == 'Background'].groupby('Date')['CUs'].sum()

    daily = daily.set_index('Date')
    daily['InteractiveCUs'] = int_by_date.reindex(daily.index, fill_value=0)
    daily['BackgroundCUs'] = bg_by_date.reindex(daily.index, fill_value=0)

    # Calculate utilisation-like percentages (relative to daily total, not SKU)
    # These will be recalculated against recommended SKU later for charts
    total = daily['InteractiveCUs'] + daily['BackgroundCUs']
    daily['InteractivePct_mean'] = np.where(total > 0, daily['InteractiveCUs'] / total * 100, 50)
    daily['BackgroundPct_mean'] = np.where(total > 0, daily['BackgroundCUs'] / total * 100, 50)
    daily['TotalUtilPct_mean'] = 100.0  # Placeholder, recalculated against SKU later

    # Throttle/carryover columns (not available in CSV, set to 0)
    for col in ['InteractiveDelayPct_mean', 'InteractiveDelayPct_max',
                'InteractiveRejectPct_mean', 'InteractiveRejectPct_max',
                'BackgroundRejectPct_mean', 'BackgroundRejectPct_max',
                'CarryoverCumulativePct_mean', 'CarryoverCumulativePct_max',
                'ExpectedBurndownMin_mean', 'ExpectedBurndownMin_max']:
        daily[col] = 0.0

    daily = daily.reset_index()
    daily['DayOfWeek'] = daily['Date'].dt.dayofweek
    daily['IsWeekday'] = daily['DayOfWeek'] < 5
    daily['IsSuspectedSpike'] = detect_pause_spikes(daily)

    return daily, df


# ============================================================================
# SECTION 6: METRICS CALCULATION (adapted from notebook cell 23)
# ============================================================================

def calculate_capacity_metrics(daily_summary, current_sku=None):
    """Calculate all key metrics from daily summary. Adapted from notebook."""
    if len(daily_summary) == 0:
        return None

    m = {}
    m['days_analyzed'] = len(daily_summary)

    # Core metrics
    m['avg_daily_cus'] = daily_summary['ActualCUs_sum'].mean()
    m['max_daily_cus'] = daily_summary['ActualCUs_sum'].max()
    m['p80_daily_cus'] = daily_summary['ActualCUs_sum'].quantile(0.8)

    # For CSV data, utilisation % is computed against recommended SKU later
    # For now, store raw percentages
    m['avg_interactive'] = daily_summary['InteractivePct_mean'].mean()
    m['avg_background'] = daily_summary['BackgroundPct_mean'].mean()

    # Throttling (all zeros for CSV)
    m['avg_delay_pct'] = daily_summary['InteractiveDelayPct_mean'].mean()
    m['max_delay_pct'] = daily_summary['InteractiveDelayPct_max'].max()
    m['avg_int_reject_pct'] = daily_summary['InteractiveRejectPct_mean'].mean()
    m['max_int_reject_pct'] = daily_summary['InteractiveRejectPct_max'].max()
    m['avg_bg_reject_pct'] = daily_summary['BackgroundRejectPct_mean'].mean()
    m['max_bg_reject_pct'] = daily_summary['BackgroundRejectPct_max'].max()

    # Carryforward (all zeros for CSV)
    m['avg_carryover_pct'] = daily_summary['CarryoverCumulativePct_mean'].mean()
    m['max_carryover_pct'] = daily_summary['CarryoverCumulativePct_max'].max()
    m['avg_burndown_min'] = daily_summary['ExpectedBurndownMin_mean'].mean()
    m['max_burndown_min'] = daily_summary['ExpectedBurndownMin_max'].max()

    # Days with issues
    m['days_with_delay'] = int((daily_summary['InteractiveDelayPct_max'] > 0).sum())
    m['days_with_rejection'] = int((daily_summary['InteractiveRejectPct_max'] > 0).sum())
    m['days_with_carryover'] = int((daily_summary['CarryoverCumulativePct_max'] > 0).sum())

    # Weekday/Weekend split
    weekdays = daily_summary[daily_summary['IsWeekday']]
    weekends = daily_summary[~daily_summary['IsWeekday']]

    if len(weekdays) > 0:
        m['weekday_avg_cus'] = weekdays['ActualCUs_sum'].mean()
        m['weekday_p80_cus'] = weekdays['ActualCUs_sum'].quantile(0.8)
        m['weekday_count'] = len(weekdays)
    else:
        m['weekday_avg_cus'] = m['avg_daily_cus']
        m['weekday_p80_cus'] = m['p80_daily_cus']
        m['weekday_count'] = 0

    if len(weekends) > 0:
        m['weekend_avg_cus'] = weekends['ActualCUs_sum'].mean()
        m['weekend_p80_cus'] = weekends['ActualCUs_sum'].quantile(0.8)
        m['weekend_count'] = len(weekends)
    else:
        m['weekend_avg_cus'] = 0
        m['weekend_p80_cus'] = 0
        m['weekend_count'] = 0

    if m['weekend_avg_cus'] > 0:
        m['weekday_weekend_ratio'] = round(m['weekday_avg_cus'] / m['weekend_avg_cus'], 1)
    else:
        m['weekday_weekend_ratio'] = float('inf')

    # Spike-filtered P80
    non_spike = daily_summary[~daily_summary['IsSuspectedSpike']]
    m['spike_days_detected'] = int(daily_summary['IsSuspectedSpike'].sum())
    if len(non_spike) >= 3:
        m['p80_daily_cus_filtered'] = non_spike['ActualCUs_sum'].quantile(0.8)
        ns_wd = non_spike[non_spike['IsWeekday']]
        m['weekday_p80_cus_filtered'] = ns_wd['ActualCUs_sum'].quantile(0.8) if len(ns_wd) > 0 else m['weekday_p80_cus']
    else:
        m['p80_daily_cus_filtered'] = m['p80_daily_cus']
        m['weekday_p80_cus_filtered'] = m['weekday_p80_cus']

    # Trend
    m['trend'] = calculate_trend(daily_summary, 'ActualCUs_sum')

    return m


def update_utilisation_metrics(metrics, recommended_sku):
    """After SKU recommendation, update utilisation % against that SKU."""
    m = metrics
    avg_util = calculate_utilisation(m['avg_daily_cus'], recommended_sku) * 100
    max_util = calculate_utilisation(m['max_daily_cus'], recommended_sku) * 100
    m['avg_util'] = avg_util
    m['max_util'] = max_util

    if m['weekday_count'] > 0:
        m['weekday_avg_util'] = calculate_utilisation(m['weekday_avg_cus'], recommended_sku) * 100
    else:
        m['weekday_avg_util'] = avg_util
    if m['weekend_count'] > 0:
        m['weekend_avg_util'] = calculate_utilisation(m['weekend_avg_cus'], recommended_sku) * 100
    else:
        m['weekend_avg_util'] = 0

    # Health score (use utilisation only for CSV, throttle/carryover = 0)
    m['health_score'], m['health_rating'] = calculate_health_score(
        avg_util, m['avg_delay_pct'], m['avg_carryover_pct']
    )
    return m


# ============================================================================
# SECTION 7: SKU RECOMMENDATION (match notebook cell 23)
# ============================================================================

def recommend_sku_for_capacity(metrics, skus, needs_free_viewers, weekday_split=True, spike_filter=True):
    """Recommend SKU using 80/80 approach. Matches notebook."""
    if weekday_split and spike_filter:
        p80_cus = metrics.get('weekday_p80_cus_filtered', metrics['p80_daily_cus'])
    elif weekday_split:
        p80_cus = metrics.get('weekday_p80_cus', metrics['p80_daily_cus'])
    elif spike_filter:
        p80_cus = metrics.get('p80_daily_cus_filtered', metrics['p80_daily_cus'])
    else:
        p80_cus = metrics['p80_daily_cus']

    required_budget = calculate_required_budget(p80_cus, TARGET_UTILISATION)

    sku_analysis = []
    recommended_sku = None

    for sku in skus:
        avg_util = calculate_utilisation(metrics['avg_daily_cus'], sku)
        max_util = calculate_utilisation(metrics['max_daily_cus'], sku)
        p80_util = calculate_utilisation(p80_cus, sku)
        status = get_sku_status(avg_util, max_util, needs_free_viewers, sku)

        payg = sku.get("monthly_usd", 0)
        reserved = sku.get("monthly_reserved_usd", 0)

        sku_analysis.append({
            'SKU': sku['name'],
            'CUs/sec': sku['cus_per_second'],
            'Daily Budget': sku['budget_30s'] * 2880,
            'Avg Util %': round(avg_util * 100, 1),
            'Peak Util %': round(max_util * 100, 1),
            'P80 Util %': round(p80_util * 100, 1),
            'Status': status,
            'PAYG $/mo': payg,
            'Reserved $/mo': reserved,
            'Savings $/mo': payg - reserved,
        })

        if recommended_sku is None and sku['budget_30s'] >= required_budget:
            if needs_free_viewers and sku['cus_per_second'] < 64:
                continue
            recommended_sku = sku

    if recommended_sku is None:
        recommended_sku = skus[-1]  # F2048 fallback

    return sku_analysis, recommended_sku

# === PART 2: Charts, HTML, Console, CLI (replaces lines 476+) ===


# ============================================================================
# SECTION 8: CHART GENERATION (matches notebook cell 24)
# ============================================================================

def create_charts(daily_summary, metrics, sku_analysis, recommended_sku, current_sku, df_raw):
    """Create all Plotly charts for the report. Returns dict of figure objects."""
    if not HAS_PLOTLY:
        return {}

    charts = {}

    # 1. Health Gauge
    fig_health = go.Figure(go.Indicator(
        mode="gauge+number",
        value=metrics['health_score'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Capacity Health Score<br><span style='font-size:0.6em;color:gray'>{metrics['health_rating']}</span>"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': '#dc3545'},
                {'range': [25, 50], 'color': '#fd7e14'},
                {'range': [50, 75], 'color': '#ffc107'},
                {'range': [75, 90], 'color': '#28a745'},
                {'range': [90, 100], 'color': '#20c997'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': metrics['health_score']
            }
        }
    ))
    fig_health.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20), font={'size': 14})
    charts['fig_health'] = fig_health

    # 2. Utilisation Gauges (Average & Peak on recommended SKU)
    title_sku = recommended_sku["name"]
    avg_util_pct = metrics['avg_util']
    peak_util_pct = metrics['max_util']

    gauge_steps = [
        {"range": [0, 40], "color": "#EBF2F9"},
        {"range": [40, 80], "color": "#D4EDDA"},
        {"range": [80, 95], "color": "#FFF3CD"},
        {"range": [95, 100], "color": "#F8D7DA"},
    ]
    fig_util = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]],
        horizontal_spacing=0.12
    )
    fig_util.add_trace(go.Indicator(
        mode="gauge+number",
        value=round(avg_util_pct, 1),
        title={"text": "Average", "font": {"size": 16, "color": "#333"}},
        number={"suffix": "%", "valueformat": ".1f", "font": {"size": 28, "color": "#2D65BC"}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": "#6A8DDC"},
            "bgcolor": "white",
            "steps": gauge_steps,
            "threshold": {"line": {"color": "#28A745", "width": 3}, "thickness": 0.8, "value": 80}
        },
    ), row=1, col=1)
    fig_util.add_trace(go.Indicator(
        mode="gauge+number",
        value=round(peak_util_pct, 1),
        title={"text": "Peak", "font": {"size": 16, "color": "#333"}},
        number={"suffix": "%", "valueformat": ".1f", "font": {"size": 28, "color": "#2D65BC"}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": "#F59E0B"},
            "bgcolor": "white",
            "steps": gauge_steps,
            "threshold": {"line": {"color": "#DC3545", "width": 3}, "thickness": 0.8, "value": 95}
        },
    ), row=1, col=2)
    fig_util.update_layout(
        title=None, height=280,
        margin=dict(l=40, r=40, t=70, b=20),
        paper_bgcolor="white",
        annotations=[dict(
            text=f"Utilisation on {title_sku}",
            x=0.5, y=1.02, xref="paper", yref="paper",
            showarrow=False, font=dict(size=18), xanchor="center"
        )]
    )
    charts['fig_util_gauges'] = fig_util

    # 3. Daily Utilisation (stacked bar with weekday/weekend colouring and trend)
    if len(daily_summary) > 0:
        fig_daily = go.Figure()
        dates_str = pd.to_datetime(daily_summary['Date']).dt.strftime('%d %b')
        day_names = pd.to_datetime(daily_summary['Date']).dt.strftime('%a')

        bg_colors = [
            'rgba(40, 167, 69, 0.5)' if wd else 'rgba(40, 167, 69, 0.25)'
            for wd in daily_summary['IsWeekday']
        ]
        int_colors = [
            'rgba(102, 126, 234, 0.8)' if wd else 'rgba(102, 126, 234, 0.4)'
            for wd in daily_summary['IsWeekday']
        ]

        fig_daily.add_trace(go.Bar(
            x=dates_str, y=daily_summary['BackgroundPct_mean'],
            name='Background', marker_color=bg_colors,
            text=[f"{d}" for d in day_names], textposition='none',
            hovertemplate='%{x} (%{text})<br>Background: %{y:.1f}%<extra></extra>'
        ))
        fig_daily.add_trace(go.Bar(
            x=dates_str, y=daily_summary['InteractivePct_mean'],
            name='Interactive', marker_color=int_colors,
            hovertemplate='%{x}<br>Interactive: %{y:.1f}%<extra></extra>'
        ))

        # Spike markers
        spike_mask = daily_summary['IsSuspectedSpike']
        if spike_mask.any():
            spike_dates = dates_str[spike_mask]
            spike_vals = daily_summary.loc[spike_mask, 'TotalUtilPct_mean']
            fig_daily.add_trace(go.Scatter(
                x=spike_dates, y=spike_vals + 3,
                mode='markers+text', name='Suspected Spike',
                marker=dict(symbol='triangle-down', size=12, color='red'),
                text=['SPIKE'] * len(spike_dates), textposition='top center',
                textfont=dict(size=9, color='red'),
                hovertemplate='%{x}<br>Suspected pause/settlement spike<extra></extra>'
            ))

        # Trend line
        trend = metrics.get('trend', {})
        if trend.get('has_trend'):
            x_vals = list(range(len(daily_summary)))
            trend_y = [trend['intercept'] + trend['slope'] * xi for xi in x_vals]
            t_color = '#e74c3c' if trend['direction'] == 'GROWING' else '#3498db' if trend['direction'] == 'DECLINING' else '#95a5a6'
            fig_daily.add_trace(go.Scatter(
                x=dates_str, y=trend_y,
                mode='lines',
                name=f"Trend ({trend['direction']}, {trend['weekly_growth_pct']:+.1f}%/wk)",
                line=dict(color=t_color, width=2, dash='dash'),
            ))

        fig_daily.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="80% Target")
        fig_daily.add_hline(y=100, line_dash="solid", line_color="red", annotation_text="100% Capacity")
        fig_daily.update_layout(
            title="Daily Capacity Utilisation (lighter bars = weekends)",
            barmode='stack', xaxis_title="Date", yaxis_title="Utilisation %",
            height=400, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        charts['fig_daily'] = fig_daily

    # 4. SKU Comparison (horizontal bar, iterate sku_analysis as LIST)
    sku_names = [f"{sa['SKU']}   " for sa in sku_analysis]
    sku_utils = [sa['Avg Util %'] for sa in sku_analysis]
    sku_colors = [
        '#dc3545' if sa['Status'] in ['THROTTLING RISK', 'TOO SMALL']
        else '#fd7e14' if sa['Status'] == 'TIGHT'
        else '#28a745' if sa['Status'] == 'GOOD FIT'
        else '#667eea' if sa['Status'] == 'COMFORTABLE'
        else '#6c757d'
        for sa in sku_analysis
    ]

    fig_sku = go.Figure(go.Bar(
        y=sku_names,
        x=[min(u, 100) for u in sku_utils],
        orientation='h',
        marker_color=sku_colors,
        text=[f"{u:.0f}%" + (" (over)" if u > 100 else "") for u in sku_utils],
        textposition='outside'
    ))

    rec_idx = next((i for i, sa in enumerate(sku_analysis) if sa['SKU'] == recommended_sku['name']), None)
    if rec_idx is not None:
        rec_util = min(sku_utils[rec_idx], 100)
        fig_sku.add_annotation(
            x=rec_util, y=sku_names[rec_idx],
            ax=50, ay=0,
            text="<b>RECOMMENDED</b>",
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2,
            arrowcolor='#28a745',
            font=dict(color='#28a745', size=11),
            xanchor='left',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#28a745', borderwidth=1, borderpad=4
        )

    fig_sku.add_vline(x=80, line_dash="dash", line_color="green", annotation_text="80% Target")
    fig_sku.add_vline(x=100, line_dash="solid", line_color="red", annotation_text="100% Limit")
    fig_sku.update_layout(
        title="SKU Comparison (Average Utilisation)",
        xaxis_title="Utilisation %",
        xaxis=dict(range=[0, 120]),
        height=450, margin=dict(l=100),
        template="plotly_white"
    )
    charts['fig_sku'] = fig_sku

    # 5. Top 10 Workloads
    if df_raw is not None and len(df_raw) > 0:
        item_totals = df_raw.groupby('Item Name')['CUs'].sum().reset_index()
        item_totals = item_totals.sort_values('CUs', ascending=True).tail(10)
        fig_items = go.Figure(go.Bar(
            y=item_totals['Item Name'],
            x=item_totals['CUs'],
            orientation='h',
            marker_color='#667eea',
            text=[
                f"{v/1e6:,.1f}M CUs" if v >= 1e6
                else f"{v/1e3:,.0f}K CUs" if v >= 1e3
                else f"{v:,.0f} CUs"
                for v in item_totals['CUs']
            ],
            textposition='outside'
        ))
        fig_items.update_layout(
            title="Top 10 Workloads by CU Consumption",
            xaxis_title="Total CUs",
            height=400, margin=dict(l=250, r=150),
            template="plotly_white"
        )
        charts['fig_items'] = fig_items

    # 6. Weekday vs Weekend comparison
    if metrics.get('weekday_count', 0) > 0 and metrics.get('weekend_count', 0) > 0:
        categories = ['Avg CUs', 'P80 CUs', 'Avg Util %']
        wd_vals = [metrics['weekday_avg_cus'], metrics['weekday_p80_cus'], metrics.get('weekday_avg_util', 0)]
        we_vals = [metrics['weekend_avg_cus'], metrics['weekend_p80_cus'], metrics.get('weekend_avg_util', 0)]

        fig_wdwe = make_subplots(rows=1, cols=3, subplot_titles=categories)
        for i, cat in enumerate(categories):
            fig_wdwe.add_trace(go.Bar(
                x=['Weekday', 'Weekend'],
                y=[wd_vals[i], we_vals[i]],
                marker_color=['#667eea', '#b0c4de'],
                text=(
                    [f"{wd_vals[i]:,.0f}", f"{we_vals[i]:,.0f}"] if i < 2
                    else [f"{wd_vals[i]:.1f}%", f"{we_vals[i]:.1f}%"]
                ),
                textposition='outside',
                showlegend=False
            ), row=1, col=i + 1)
        fig_wdwe.update_layout(
            title="Weekday vs Weekend Comparison",
            height=350, template="plotly_white"
        )
        charts['fig_weekday_weekend'] = fig_wdwe

    return charts


# ============================================================================
# SECTION 9: HTML REPORT GENERATION (matches notebook cell 26)
# ============================================================================

def generate_html_report(metrics, sku_analysis, recommended_sku, current_sku,
                         charts, daily_summary, capacity_name, df_raw):
    """Generate complete HTML report. Returns HTML string."""

    # Derive values
    _cur_sku_name = current_sku["name"] if current_sku else "Unknown"
    _dates = pd.to_datetime(daily_summary["Date"])
    _date_range = f"{_dates.min().strftime('%d %b %Y')} to {_dates.max().strftime('%d %b %Y')}"
    daily_budget = recommended_sku['budget_30s'] * 2880
    rec_payg_monthly = recommended_sku.get("monthly_usd", 0)
    rec_reserved_monthly = recommended_sku.get("monthly_reserved_usd", 0)
    rec_saving_monthly = rec_payg_monthly - rec_reserved_monthly

    html_parts = []

    # --- HTML header + CSS ---
    html_parts.append("""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fabric Capacity Health Report | Data Nova</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',Arial,sans-serif;background:linear-gradient(135deg,#f5f7fa 0%,#e4e8ec 100%);min-height:100vh;color:#333;line-height:1.6;padding:24px}
.container{max-width:1200px;margin:0 auto}
.header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:40px;border-radius:16px;margin-bottom:24px;box-shadow:0 10px 40px rgba(102,126,234,0.3)}
.header h1{font-size:2.5em;font-weight:700;margin-bottom:8px}
.header .subtitle{opacity:0.9;font-size:1.1em}
.header .meta{margin-top:16px;font-size:0.9em;opacity:0.8}
.section{background:white;border-radius:16px;padding:24px;margin-bottom:24px;box-shadow:0 2px 12px rgba(0,0,0,0.08)}
.section h2{font-size:1.2em;color:#333;margin-bottom:16px;padding-bottom:12px;border-bottom:2px solid #667eea}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px}
.kpi{background:white;border-radius:12px;padding:24px;text-align:center;box-shadow:0 2px 12px rgba(0,0,0,0.08)}
.kpi .value{font-size:2.2em;font-weight:700;color:#667eea;margin-bottom:4px}
.kpi .label{color:#666;font-size:0.9em;font-weight:500}
.chart-wrap{margin:16px 0}
.rec-card{background:#f8f9fa;border-radius:12px;padding:20px;border-left:4px solid #667eea}
.rec-card p{margin-bottom:8px}
.rec-card ul{padding-left:24px;margin-bottom:12px}
.rec-card li{margin-bottom:4px}
.disclaimer{background:#fff3cd;border-radius:12px;padding:16px;margin-top:16px;border-left:4px solid #ffc107;font-size:0.85em;color:#856404}
.insight-box{background:#e8f4f8;border-radius:12px;padding:16px;margin:12px 0;border-left:4px solid #17a2b8}
.cost-table{width:100%;border-collapse:collapse;margin:12px 0}
.cost-table th,.cost-table td{padding:10px 16px;text-align:left;border-bottom:1px solid #eee}
.cost-table th{background:#f8f9fa;font-weight:600;color:#333}
.cost-table .highlight{background:#e8f5e9;font-weight:600}
.footer{text-align:center;padding:32px;color:#666;font-size:0.9em}
</style></head><body><div class="container">""")

    # --- Header ---
    html_parts.append(f"""<div class="header">
  <h1>Fabric Capacity Health Report</h1>
  <p class="subtitle">Your capacity. Your costs. Your next move.</p>
  <p class="meta" style="margin-bottom:8px;"><strong>Capacity:</strong> {capacity_name} ({_cur_sku_name}) &bull; <strong>Period:</strong> {_date_range}</p>
  <p class="meta">Prathy Kamasani | Data Nova &mdash; Generated {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}</p>
</div>""")

    # --- Disclaimer ---
    html_parts.append(f'<div class="disclaimer"><strong>Important:</strong> {DISCLAIMER_TEXT}</div>')

    # --- KPI cards ---
    kpi_lines = []
    kpi_lines.append(
        f'<div class="kpi"><div class="value">{metrics["health_score"]}%</div>'
        f'<span class="label">Health Score &middot; {metrics["health_rating"]}</span></div>'
    )

    if current_sku and recommended_sku:
        _c_idx = next((i for i, s in enumerate(SKUS) if s['name'] == current_sku['name']), -1)
        _r_idx = next((i for i, s in enumerate(SKUS) if s['name'] == recommended_sku['name']), -1)
        _arrow = '&uarr;' if _r_idx > _c_idx else ('&darr;' if _r_idx < _c_idx else '&check;')
        _word = 'Upgrade' if _r_idx > _c_idx else ('Downsize' if _r_idx < _c_idx else 'Stay')
        kpi_lines.append(
            f'<div class="kpi"><div class="value">{current_sku["name"]} {_arrow} {recommended_sku["name"]}</div>'
            f'<span class="label">{_word} &middot; SKU Recommendation</span></div>'
        )
    elif recommended_sku:
        kpi_lines.append(
            f'<div class="kpi"><div class="value">{recommended_sku["name"]}</div>'
            f'<span class="label">Recommended SKU</span></div>'
        )

    if current_sku and recommended_sku:
        _diff = current_sku.get("monthly_usd", 0) - recommended_sku.get("monthly_usd", 0)
        if _diff > 0:
            kpi_lines.append(
                f'<div class="kpi"><div class="value" style="color:#28a745;">~${_diff:,}/mo</div>'
                f'<span class="label">Est. Savings (list price)</span></div>'
            )
        elif _diff < 0:
            kpi_lines.append(
                f'<div class="kpi"><div class="value" style="color:#fd7e14;">~+${abs(_diff):,}/mo</div>'
                f'<span class="label">Est. Upgrade Cost (list price)</span></div>'
            )

    kpi_lines.append(f'<div class="kpi"><span class="label">Days Analyzed</span><div class="value">{metrics["days_analyzed"]}</div></div>')
    kpi_lines.append(f'<div class="kpi"><span class="label">Avg Daily CUs</span><div class="value">{metrics["avg_daily_cus"]:,.0f}</div></div>')
    kpi_lines.append(f'<div class="kpi"><span class="label">Avg Util %</span><div class="value">{metrics["avg_util"]:.1f}%</div></div>')

    if metrics['days_with_delay'] > 0:
        kpi_lines.append(
            f'<div class="kpi"><span class="label">Days with Throttling</span>'
            f'<div class="value">{metrics["days_with_delay"]}/{metrics["days_analyzed"]}</div></div>'
        )

    _trend = metrics.get('trend', {})
    if _trend.get('has_trend'):
        _t_color = '#dc3545' if _trend['direction'] == 'GROWING' else '#28a745' if _trend['direction'] == 'DECLINING' else '#6c757d'
        kpi_lines.append(
            f'<div class="kpi"><div class="value" style="color:{_t_color};">{_trend["weekly_growth_pct"]:+.1f}%/wk</div>'
            f'<span class="label">Consumption Trend &middot; {_trend["direction"]}</span></div>'
        )

    html_parts.append(
        '<div class="section"><h2>Key Metrics</h2><div class="kpi-grid">'
        + "".join(kpi_lines) + "</div></div>"
    )

    # --- Weekday vs Weekend ---
    if metrics.get('weekday_count', 0) > 0 and metrics.get('weekend_count', 0) > 0:
        _ratio = metrics.get('weekday_weekend_ratio', 1)
        _insight = (
            f"Weekday consumption is {_ratio}x higher than weekends. "
            if _ratio > 1.5
            else "Weekday and weekend consumption are similar. "
        )
        _insight += f"SKU recommendation is based on weekday P80 ({metrics['weekday_p80_cus']:,.0f} CUs) for accurate working-day sizing."
        html_parts.append(
            f'<div class="section"><h2>Weekday vs Weekend</h2>'
            f'<div class="insight-box"><strong>Insight:</strong> {_insight}</div>'
            f'<div class="kpi-grid">'
            f'<div class="kpi"><div class="value">{metrics["weekday_avg_cus"]:,.0f}</div><span class="label">Weekday Avg CUs</span></div>'
            f'<div class="kpi"><div class="value">{metrics["weekend_avg_cus"]:,.0f}</div><span class="label">Weekend Avg CUs</span></div>'
            f'<div class="kpi"><div class="value">{metrics.get("weekday_avg_util", 0):.1f}%</div><span class="label">Weekday Avg Util</span></div>'
            f'<div class="kpi"><div class="value">{metrics.get("weekend_avg_util", 0):.1f}%</div><span class="label">Weekend Avg Util</span></div>'
            f'</div></div>'
        )

    # --- Spike filtering note ---
    if metrics.get('spike_days_detected', 0) > 0:
        _n = metrics['spike_days_detected']
        html_parts.append(
            f'<div class="section"><h2>Spike Filtering</h2>'
            f'<div class="insight-box"><strong>Note:</strong> {_n} day(s) showed anomalously high peak CUs, '
            f'likely from capacity pause/resume or settlement catch-up processing. '
            f'These were excluded from the P80 calculation used for SKU sizing. '
            f'P80 (all days): {metrics["p80_daily_cus"]:,.0f} CUs vs P80 (filtered): {metrics["p80_daily_cus_filtered"]:,.0f} CUs.</div></div>'
        )

    # --- Charts ---
    chart_order = [
        ("fig_util_gauges", f"Utilisation on {recommended_sku['name']}"),
        ("fig_daily", "Daily Utilisation (lighter bars = weekends)"),
        ("fig_weekday_weekend", "Weekday vs Weekend Comparison"),
        ("fig_sku", "SKU Utilisation Comparison"),
        ("fig_items", "Top Workloads by CUs"),
    ]

    first_chart = True
    for fig_name, title in chart_order:
        fig = charts.get(fig_name)
        if fig is not None and hasattr(fig, "to_html"):
            plotly_js = "cdn" if first_chart else False
            html_parts.append(f'<div class="section"><h2>{title}</h2><div class="chart-wrap">')
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs=plotly_js))
            html_parts.append("</div></div>")
            first_chart = False

    # --- Reserved vs PAYG cost table ---
    avg_util_decimal = metrics['avg_util'] / 100.0
    if avg_util_decimal > RESERVED_BREAKEVEN_UTIL:
        _payg_advice = (
            f'<div class="insight-box" style="background:#e8f5e9;border-color:#28a745;">'
            f'<strong>Recommendation:</strong> Your average utilisation ({metrics["avg_util"]:.0f}%) is above the '
            f'~{RESERVED_BREAKEVEN_UTIL*100:.0f}% break-even point. A <strong>reserved instance</strong> would likely '
            f'save ~${rec_saving_monthly:,}/mo vs PAYG for {recommended_sku["name"]}.</div>'
        )
    else:
        _payg_advice = (
            f'<div class="insight-box" style="background:#fff3cd;border-color:#ffc107;">'
            f'<strong>Recommendation:</strong> Your average utilisation ({metrics["avg_util"]:.0f}%) is below the '
            f'~{RESERVED_BREAKEVEN_UTIL*100:.0f}% break-even point. <strong>PAYG with pause/resume scheduling</strong> '
            f'may be more cost-effective than a reserved instance.</div>'
        )

    _cost_rows = ""
    for sa in sku_analysis:
        _hl = ' class="highlight"' if sa['SKU'] == recommended_sku['name'] else ''
        _cost_rows += (
            f'<tr{_hl}><td>{sa["SKU"]}</td><td>{sa["Avg Util %"]}%</td>'
            f'<td>${sa["PAYG $/mo"]:,}</td><td>${sa["Reserved $/mo"]:,}</td>'
            f'<td>${sa["Savings $/mo"]:,}</td></tr>'
        )

    html_parts.append(
        f'<div class="section"><h2>Reserved vs PAYG Cost Comparison</h2>'
        f'{_payg_advice}'
        f'<table class="cost-table"><thead><tr><th>SKU</th><th>Avg Util %</th>'
        f'<th>PAYG $/mo</th><th>Reserved $/mo</th><th>Savings $/mo</th></tr></thead>'
        f'<tbody>{_cost_rows}</tbody></table>'
        f'<p style="font-size:0.85em;color:#666;margin-top:8px;">Published list prices (USD). '
        f'Reserved = 1-year commitment. Actual costs vary by region, currency, and agreement. '
        f'Break-even at ~{RESERVED_BREAKEVEN_UTIL*100:.0f}% utilisation.</p></div>'
    )

    # --- Recommendations ---
    rec_html = '<div class="section"><h2>Recommendations</h2><div class="rec-card">'
    _h = metrics['health_rating']
    _rn = recommended_sku['name']

    if _h == 'CRITICAL':
        rec_html += (
            f'<p style="color:#dc3545;font-weight:600;">URGENT: Critically overloaded.</p>'
            f'<ul><li>Upgrade to <strong>{_rn}</strong> immediately</li>'
            f'<li>Reschedule heavy background jobs</li>'
            f'<li>Check if job-level bursting is enabled (default: ON). Disable if single jobs are monopolising capacity.</li></ul>'
        )
    elif _h == 'POOR':
        rec_html += (
            f'<p style="color:#fd7e14;font-weight:600;">WARNING: Significant stress.</p>'
            f'<ul><li>Plan upgrade to <strong>{_rn}</strong></li>'
            f'<li>Spread workloads across off-peak hours</li>'
            f'<li>Review job-level bursting settings if concurrency is a concern</li></ul>'
        )
    elif _h == 'FAIR':
        rec_html += (
            f'<p style="color:#856404;font-weight:600;">ATTENTION: Running warm.</p>'
            f'<ul><li>Consider <strong>{_rn}</strong></li>'
            f'<li>Optimise heavy refresh schedules</li></ul>'
        )
    elif _h == 'GOOD':
        rec_html += (
            '<p style="color:#28a745;font-weight:600;">HEALTHY: Well-sized.</p>'
            '<ul><li>Continue monitoring for growth trends</li></ul>'
        )
    else:  # EXCELLENT
        rec_html += (
            '<p style="color:#28a745;font-weight:600;">EXCELLENT: Plenty of headroom.</p>'
            '<ul><li>Consider downsizing to save costs</li></ul>'
        )

    if _trend.get('has_trend') and _trend['direction'] == 'GROWING' and _trend['weekly_growth_pct'] > 3:
        rec_html += (
            f'<li style="color:#dc3545;">Consumption is growing at {_trend["weekly_growth_pct"]:+.1f}%/week. '
            f'Plan for the next SKU tier within the coming weeks.</li>'
        )

    if metrics['avg_util'] > 100 and metrics['days_with_delay'] == 0:
        rec_html += (
            '<li>No throttling detected despite high utilisation. Check if Capacity Overage or '
            'Surge Protection is enabled, as these incur additional Azure charges.</li>'
        )

    rec_html += '</ul></div>'
    rec_html += (
        '<div class="insight-box"><strong>Spark Autoscale Note:</strong> If Spark Autoscale Billing '
        'is enabled for this capacity, Spark workloads are billed separately on a pay-as-you-go basis '
        'and are NOT reflected in these metrics. Factor in those costs separately when evaluating total '
        'Fabric spend.</div>'
    )
    rec_html += '</div>'
    html_parts.append(rec_html)

    # --- Data source note ---
    html_parts.append(
        f'<div class="section"><h2>Data Source</h2>'
        f'<div class="insight-box">This analysis is based on <strong>{len(df_raw)}</strong> individual workload records '
        f'across <strong>{metrics["days_analyzed"]}</strong> days of Fabric capacity usage data (CSV). '
        f'Note: CSV data does not include throttling or carryforward metrics available in the Capacity Metrics '
        f'semantic model. For a more complete analysis, use the Fabric SKU Advisor notebook with a live '
        f'semantic model connection.</div></div>'
    )

    # --- Footer ---
    html_parts.append(f'<div class="section disclaimer"><strong>Disclaimer:</strong> {DISCLAIMER_TEXT}</div>')
    html_parts.append(
        '<div class="footer"><p>Fabric SKU Advisor | '
        '<a href="https://www.data-nova.io">Data Nova</a></p></div>'
        '</div></body></html>'
    )

    return "\n".join(html_parts)


# ============================================================================
# SECTION 10: CONSOLE SUMMARY OUTPUT
# ============================================================================

def print_console_summary(metrics, recommended_sku, current_sku, sku_analysis, capacity_name):
    """Print a clean text summary to stdout."""

    daily_budget = recommended_sku['budget_30s'] * 2880

    print("\n" + "=" * 70)
    print("FABRIC CAPACITY HEALTH SUMMARY".center(70))
    print("=" * 70)

    print(f"\nCapacity: {capacity_name}")
    print(f"Health Score: {metrics['health_score']}/100 ({metrics['health_rating']})")
    print(f"Days Analyzed: {metrics['days_analyzed']}")

    print(f"\nConsumption Metrics:")
    print(f"  Average Daily CUs: {metrics['avg_daily_cus']:,.0f}")
    print(f"  Peak Daily CUs:    {metrics['max_daily_cus']:,.0f}")
    print(f"  P80 Daily CUs:     {metrics['p80_daily_cus']:,.0f}")
    print(f"  Average Utilisation: {metrics['avg_util']:.1f}%")
    print(f"  Peak Utilisation:    {metrics['max_util']:.1f}%")

    _trend = metrics.get('trend', {})
    if _trend.get('has_trend'):
        print(f"  Trend: {_trend['direction']} ({_trend['weekly_growth_pct']:+.1f}%/week)")

    if metrics.get('weekday_count', 0) > 0 and metrics.get('weekend_count', 0) > 0:
        print(f"\nWeekday/Weekend:")
        print(f"  Weekday Avg CUs: {metrics['weekday_avg_cus']:,.0f}")
        print(f"  Weekend Avg CUs: {metrics['weekend_avg_cus']:,.0f}")
        print(f"  Ratio: {metrics.get('weekday_weekend_ratio', 'N/A')}x")

    if metrics.get('spike_days_detected', 0) > 0:
        print(f"\nSpike Detection: {metrics['spike_days_detected']} day(s) flagged")

    print(f"\nCurrent SKU: {current_sku['name'] if current_sku else 'Unknown'}")
    if current_sku:
        cur_util = calculate_utilisation(metrics['avg_daily_cus'], current_sku) * 100
        print(f"  Daily Budget: {current_sku['budget_30s'] * 2880:,.0f} CUs")
        print(f"  Estimated Utilisation: {cur_util:.1f}%")

    print(f"\nRECOMMENDED SKU: {recommended_sku['name']}")
    rec_sa = next((sa for sa in sku_analysis if sa['SKU'] == recommended_sku['name']), {})
    print(f"  Daily Budget: {daily_budget:,.0f} CUs")
    print(f"  Estimated Utilisation: {rec_sa.get('Avg Util %', 0):.1f}%")
    print(f"  Status: {rec_sa.get('Status', 'Unknown')}")
    print(f"  PAYG: ${recommended_sku['monthly_usd']:,}/mo (${recommended_sku['monthly_usd'] * 12:,}/yr)")
    print(f"  Reserved: ${recommended_sku['monthly_reserved_usd']:,}/mo (${recommended_sku['monthly_reserved_usd'] * 12:,}/yr)")

    savings_monthly = recommended_sku['monthly_usd'] - recommended_sku['monthly_reserved_usd']
    if savings_monthly > 0:
        print(f"  Potential Savings: ${savings_monthly:,}/mo (${savings_monthly * 12:,}/yr)")

    print("\n" + "=" * 70 + "\n")


# ============================================================================
# SECTION 11: ARGPARSE AND MAIN
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fabric SKU Advisor - Analyse capacity consumption and recommend SKU",
        epilog="Author: Prathy Kamasani | Data Nova (https://www.data-nova.io)"
    )
    parser.add_argument('-i', '--input', required=True,
                        help='CSV file path (columns: Date, Item Name, Item Type, CUs)')
    parser.add_argument('-o', '--output', default=None,
                        help='HTML output path (default: sku_report_YYYYMMDD_HHMM.html)')
    parser.add_argument('--capacity-name', default='Fabric Capacity',
                        help='Capacity display name (default: Fabric Capacity)')
    parser.add_argument('--current-sku', default=None,
                        help='Current SKU for comparison (e.g. "F64", "F128")')
    parser.add_argument('--needs-free-viewers', action='store_true',
                        help='Require F64+ for free viewer support')
    parser.add_argument('--no-weekday-split', action='store_true',
                        help='Disable weekday/weekend split analysis')
    parser.add_argument('--no-spike-filter', action='store_true',
                        help='Disable spike filtering')
    parser.add_argument('--console', action='store_true',
                        help='Print text summary to stdout')
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    try:
        # Load and process CSV
        print(f"Loading: {args.input}")
        daily_summary, df_raw = load_csv_and_build_summary(args.input)
        print(f"  {len(daily_summary)} days, {len(df_raw)} item records loaded")

        # Calculate metrics
        print("Calculating metrics...")
        metrics = calculate_capacity_metrics(daily_summary)

        # SKU recommendation
        print("Analysing SKU recommendations...")
        sku_analysis, recommended_sku = recommend_sku_for_capacity(
            metrics, SKUS,
            needs_free_viewers=args.needs_free_viewers,
            weekday_split=not args.no_weekday_split,
            spike_filter=not args.no_spike_filter,
        )

        # Current SKU lookup
        current_sku = None
        if args.current_sku:
            current_sku = next((s for s in SKUS if s['name'].upper() == args.current_sku.upper()), None)
            if not current_sku:
                print(f"WARNING: SKU '{args.current_sku}' not found. Ignoring.")

        # Update utilisation metrics against recommended SKU
        metrics = update_utilisation_metrics(metrics, recommended_sku)

        # Console summary
        if args.console:
            print_console_summary(metrics, recommended_sku, current_sku, sku_analysis, args.capacity_name)

        # HTML report
        if HAS_PLOTLY:
            print("Generating charts...")
            charts = create_charts(daily_summary, metrics, sku_analysis,
                                   recommended_sku, current_sku, df_raw)

            print("Generating HTML report...")
            html = generate_html_report(
                metrics, sku_analysis, recommended_sku, current_sku,
                charts, daily_summary, args.capacity_name, df_raw
            )

            output_path = args.output or f"sku_report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.html"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"HTML report saved: {output_path}")
        else:
            print("[WARNING] Skipping HTML report (plotly not installed). Install: pip install plotly")
            if not args.console:
                print_console_summary(metrics, recommended_sku, current_sku, sku_analysis, args.capacity_name)

        print(f"\nDone: Recommended SKU is {recommended_sku['name']} "
              f"(Health: {metrics['health_score']}/100 {metrics['health_rating']})")

    except FileNotFoundError:
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
