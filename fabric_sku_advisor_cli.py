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


def calculate_health_score(avg_util_pct, throttle_pct=0.0, carryover_pct=0.0,
                           days_with_throttling=0, total_days=0):
    """Composite capacity health score (0-100). Matches notebook exactly.

    v2.0 additions:
    - Pervasive throttling cap: if throttled every day, cap at POOR
    - Carryforward data quality check: if throttling severe but carryforward 0,
      assume carryforward data is broken and apply pessimistic estimate
    """
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
    # v2.0: if throttling is severe but carryforward is zero, data is suspect —
    # replace with pessimistic estimate (carryforward MUST exist when throttling)
    if throttle_pct > 10 and carryover_pct == 0:
        carry_score = 25  # Assume moderate carryforward (data broken)
    elif carryover_pct == 0:
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

    # v2.0: Pervasive throttling cap — override score if throttled constantly
    if total_days > 0 and days_with_throttling >= total_days:
        score = min(score, 24)  # Cap at POOR
    elif total_days > 0 and days_with_throttling / total_days > 0.75:
        score = min(score, 49)  # Cap at FAIR

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


def update_utilisation_metrics(metrics, recommended_sku, daily_summary=None):
    """After SKU recommendation, update utilisation % against that SKU."""
    m = metrics
    avg_util = calculate_utilisation(m['avg_daily_cus'], recommended_sku) * 100
    max_util = calculate_utilisation(m['max_daily_cus'], recommended_sku) * 100
    m['avg_util'] = avg_util
    m['max_util'] = max_util

    # v2.0: Burst severity metrics
    if daily_summary is not None and 'TotalUtilPct_max' in daily_summary.columns and len(daily_summary) > 2:
        m['p95_peak_util'] = daily_summary['TotalUtilPct_max'].quantile(0.95)
    else:
        m['p95_peak_util'] = max_util
    m['burst_ratio'] = round(max_util / avg_util, 1) if avg_util > 0 else 1.0

    if m['weekday_count'] > 0:
        m['weekday_avg_util'] = calculate_utilisation(m['weekday_avg_cus'], recommended_sku) * 100
    else:
        m['weekday_avg_util'] = avg_util
    if m['weekend_count'] > 0:
        m['weekend_avg_util'] = calculate_utilisation(m['weekend_avg_cus'], recommended_sku) * 100
    else:
        m['weekend_avg_util'] = 0

    # Health score — pass throttling-day counts for v2.0 pervasive throttling cap
    days_with_throttling = m.get('days_with_delay', 0)
    total_days = m.get('days_analyzed', 1)
    m['health_score'], m['health_rating'] = calculate_health_score(
        avg_util, m['avg_delay_pct'], m['avg_carryover_pct'],
        days_with_throttling=days_with_throttling,
        total_days=total_days,
    )
    return m


# ============================================================================
# SECTION 7: SKU RECOMMENDATION (match notebook cell 23)
# ============================================================================

def estimate_true_demand(p80_cus, avg_rejection_pct):
    """
    Estimate true CU demand when rejection is suppressing measured consumption.

    When Fabric rejects requests, those CUs are never consumed — so measured
    utilisation is artificially low. This adjusts P80 upward to estimate what
    demand would be without rejection.

    Formula: true_demand = measured / (1 - rejection_rate)
    Capped at 2× to avoid runaway estimates from extreme rejection rates.
    """
    if avg_rejection_pct <= 0:
        return p80_cus
    rejection_rate = min(avg_rejection_pct / 100.0, 0.80)  # Cap at 80%
    adjustment_factor = min(1.0 / (1.0 - rejection_rate), 2.0)  # Cap at 2×
    return p80_cus * adjustment_factor


def format_hour_ranges(hours):
    """Convert a list of hours [6,7,8,9,14,15,16] to '06:00–10:00, 14:00–17:00'."""
    if not hours:
        return ""
    hours = sorted(hours)
    ranges = []
    start = hours[0]
    prev = hours[0]
    for h in hours[1:]:
        if h == prev + 1:
            prev = h
        else:
            ranges.append(f"{start:02d}:00\u2013{prev+1:02d}:00")
            start = h
            prev = h
    ranges.append(f"{start:02d}:00\u2013{prev+1:02d}:00")
    return ", ".join(ranges)


def generate_scheduling_suggestions(df_raw, peak_hours=None, off_peak_hours=None, top_n=5, sku_budget_daily=None):
    """Analyse top consumers and provide actionable scheduling guidance.

    Returns list of dicts with: item_name, item_type, daily_cus, total_cus,
    budget_pct, total_ops, rejected_ops, schedulable, action, n_days.
    """
    suggestions = []
    if df_raw is None or len(df_raw) == 0:
        return suggestions

    # --- Find columns ---
    type_cols = [c for c in df_raw.columns if 'item' in c.lower() and ('type' in c.lower() or 'kind' in c.lower())]
    item_type_col = type_cols[0] if type_cols else ('Item Type' if 'Item Type' in df_raw.columns else None)
    cu_cols = [c for c in df_raw.columns if c.lower() in ('cus', 'cu', 'actualcus_sum', 'totalcus')]
    cu_col = cu_cols[0] if cu_cols else ('CUs' if 'CUs' in df_raw.columns else None)
    name_cols = [c for c in df_raw.columns if 'item' in c.lower() and 'name' in c.lower() and 'workspace' not in c.lower()]
    name_col = name_cols[0] if name_cols else ('Item Name' if 'Item Name' in df_raw.columns else None)

    if not all([item_type_col, cu_col, name_col]):
        return suggestions

    # Background/schedulable item types (covers old and new naming)
    bg_types = {
        'Dataflow', 'Dataflow Gen1', 'Dataflow Gen2', 'DataPipeline', 'Pipeline',
        'DataPipelines', 'Data Pipeline', 'Copy', 'Notebook', 'SparkJob',
        'Spark Job Definition', 'Lakehouse', 'Warehouse', 'SQLEndpoint',
        'Dataset', 'SemanticModel', 'Semantic Model',
    }

    # --- Aggregate: one row per (item, type) with totals ---
    agg_dict = {cu_col: 'sum'}
    ops_col = next((c for c in df_raw.columns if c in ('TotalOps', 'Operations', 'SuccessCount')), None)
    reject_col = next((c for c in df_raw.columns if c in ('RejectCount', 'Rejected')), None)
    if ops_col:
        agg_dict[ops_col] = 'sum'
    if reject_col:
        agg_dict[reject_col] = 'sum'

    group_cols = [name_col, item_type_col]
    item_agg = df_raw.groupby(group_cols).agg(agg_dict).reset_index()
    item_agg = item_agg.sort_values(cu_col, ascending=False)

    n_days = max(1, df_raw['Date'].dt.date.nunique() if 'Date' in df_raw.columns else 1)
    has_peak = bool(peak_hours and off_peak_hours)
    peak_str = format_hour_ranges(peak_hours) if has_peak else None
    offpeak_str = format_hour_ranges(off_peak_hours) if has_peak else None

    for _, row in item_agg.head(top_n).iterrows():
        item_name = str(row[name_col])
        item_type = str(row[item_type_col])
        total_cus = float(row[cu_col]) if pd.notna(row[cu_col]) else 0
        if total_cus <= 0:
            continue

        daily_avg = total_cus / n_days
        budget_pct = round((daily_avg / sku_budget_daily) * 100, 1) if sku_budget_daily and sku_budget_daily > 0 else None
        total_ops = int(row[ops_col]) if ops_col and pd.notna(row.get(ops_col, None)) else None
        rejected = int(row[reject_col]) if reject_col and pd.notna(row.get(reject_col, None)) else None
        is_bg = item_type in bg_types

        # Build action guidance
        if is_bg and has_peak:
            action = f'Check refresh schedule. If overlapping peak ({peak_str}), move to {offpeak_str}.'
        elif is_bg:
            action = 'Check refresh schedule in workspace settings. Stagger to reduce peak contention.'
        else:
            action = 'Review query patterns and optimise DAX/report design.'

        suggestions.append({
            'item_name': item_name, 'item_type': item_type,
            'daily_cus': daily_avg, 'total_cus': total_cus,
            'budget_pct': budget_pct, 'total_ops': total_ops,
            'daily_ops': round(total_ops / n_days, 1) if total_ops else None,
            'rejected_ops': rejected, 'schedulable': is_bg,
            'action': action, 'n_days': n_days,
        })

    return suggestions


def generate_tiered_recommendations(metrics, current_sku, recommended_sku,
                                     sku_analysis, skus, peak_info=None, df_raw=None):
    """
    Generate structured, tiered recommendation options.

    Returns a dict with:
        urgency, option_a, option_b, option_c,
        growth_warning, scheduling_advice, notes
    """
    recs = {
        'urgency': 'NONE',
        'option_a': None,
        'option_b': None,
        'option_c': None,
        'growth_warning': None,
        'scheduling_advice': [],
        'notes': [],
    }

    health_rating = metrics.get('health_rating', 'FAIR')
    days_throttled = metrics.get('days_with_delay', 0)
    total_days = max(metrics.get('days_analyzed', 1), 1)
    avg_reject = metrics.get('avg_int_reject_pct', 0)
    avg_delay = metrics.get('avg_delay_pct', 0)

    # --- Urgency ---
    if health_rating == 'CRITICAL' or avg_reject > 20:
        recs['urgency'] = 'CRITICAL'
    elif health_rating == 'POOR' or days_throttled == total_days:
        recs['urgency'] = 'HIGH'
    elif health_rating == 'FAIR':
        recs['urgency'] = 'MODERATE'
    elif health_rating == 'GOOD' and metrics.get('trend', {}).get('direction') == 'GROWING':
        recs['urgency'] = 'LOW'

    # --- Option A: Scale change ---
    if current_sku and recommended_sku:
        cur_cost = current_sku.get('monthly_usd', 0)
        rec_cost = recommended_sku.get('monthly_usd', 0)
        rec_util = calculate_utilisation(metrics['avg_daily_cus'], recommended_sku) * 100

        if recommended_sku['cus_per_second'] > current_sku['cus_per_second']:
            recs['option_a'] = {
                'action': 'SCALE UP',
                'from_sku': current_sku['name'],
                'to_sku': recommended_sku['name'],
                'monthly_cost': rec_cost,
                'monthly_cost_delta': rec_cost - cur_cost,
                'expected_util': round(rec_util, 1),
                'impact': 'Eliminates throttling immediately. No workload changes needed.',
            }
        elif recommended_sku['cus_per_second'] < current_sku['cus_per_second']:
            if not metrics.get('_downsize_vetoed'):
                recs['option_a'] = {
                    'action': 'DOWNSIZE',
                    'from_sku': current_sku['name'],
                    'to_sku': recommended_sku['name'],
                    'monthly_savings': cur_cost - rec_cost,
                    'expected_util': round(rec_util, 1),
                    'impact': 'Reduces cost. Monitor closely after change for throttling.',
                }
        else:
            # v2.0: If urgency is CRITICAL/HIGH but rec == current, force SCALE UP
            if recs['urgency'] in ('CRITICAL', 'HIGH'):
                cur_idx = next((i for i, s in enumerate(skus) if s['name'] == current_sku['name']), -1)
                if cur_idx >= 0 and cur_idx + 1 < len(skus):
                    next_sku = skus[cur_idx + 1]
                    next_daily = next_sku['budget_30s'] * 2880
                    adj_util = round((metrics.get('_adjusted_p80', metrics['avg_daily_cus']) / next_daily) * 100, 1) if next_daily > 0 else 0
                    recs['option_a'] = {
                        'action': 'SCALE UP',
                        'from_sku': current_sku['name'],
                        'to_sku': next_sku['name'],
                        'monthly_cost': next_sku.get('monthly_usd', 0),
                        'monthly_cost_delta': next_sku.get('monthly_usd', 0) - cur_cost,
                        'expected_util': adj_util,
                        'impact': f'Current capacity is severely throttled. Scale up to {next_sku["name"]} ({next_daily:,} CUs/day) to eliminate throttling.',
                    }
                else:
                    recs['option_a'] = {
                        'action': 'STAY',
                        'sku': current_sku['name'],
                        'monthly_cost': cur_cost,
                        'expected_util': round(rec_util, 1),
                        'impact': 'Already at maximum SKU. Optimise workloads to reduce throttling.',
                    }
            else:
                recs['option_a'] = {
                    'action': 'STAY',
                    'sku': current_sku['name'],
                    'monthly_cost': cur_cost,
                    'expected_util': round(rec_util, 1),
                    'impact': 'Current SKU is appropriately sized.',
                }

    # --- Scheduling-issue detection ---
    # When 80/80 says right-sized but throttling is pervasive, it's a scheduling
    # problem not a sizing problem. Low avg util + high throttle = burst peaks.
    avg_util = metrics.get('avg_util', 0)
    throttle_pct = (days_throttled / total_days * 100) if total_days > 0 else 0
    _is_sched_issue = (
        avg_util < 50
        and throttle_pct > 50
        and recs.get('option_a', {}).get('action') in ('SCALE UP', 'STAY')
        and recommended_sku and current_sku
        and recommended_sku['cus_per_second'] == current_sku['cus_per_second']
    )
    recs['_scheduling_issue'] = _is_sched_issue
    recs['_effective_action'] = 'RESCHEDULE' if _is_sched_issue else recs.get('option_a', {}).get('action', 'STAY')

    # --- Option B: Optimise scheduling ---
    _has_peak = bool(peak_info and peak_info.get('peak_hours') and peak_info.get('off_peak_hours'))
    peak_hrs_str = format_hour_ranges(peak_info['peak_hours']) if _has_peak else None
    offpeak_str = format_hour_ranges(peak_info['off_peak_hours']) if _has_peak else None

    recs['option_b'] = {
        'action': 'OPTIMISE SCHEDULING',
        'peak_hours': peak_hrs_str,
        'off_peak_hours': offpeak_str,
        'has_peak_data': _has_peak,
        'impact': 'Move heavy background workloads to off-peak hours to reduce contention.',
    }

    # Always generate scheduling advice (top consumers) when throttling present
    if df_raw is not None and len(df_raw) > 0 and (avg_delay > 2 or avg_reject > 1 or _has_peak):
        _budget_daily = current_sku['budget_30s'] * 2880 if current_sku else None
        recs['scheduling_advice'] = generate_scheduling_suggestions(
            df_raw,
            peak_hours=peak_info.get('peak_hours') if _has_peak else None,
            off_peak_hours=peak_info.get('off_peak_hours') if _has_peak else None,
            sku_budget_daily=_budget_daily,
        )

    # --- Option C: Combined ---
    if recs['option_a'] and recs['option_a'].get('action') == 'SCALE UP' and recs['option_b']:
        # Find an intermediate SKU (one step below recommended)
        cur_idx = next((i for i, s in enumerate(skus) if s['name'] == current_sku['name']), None)
        rec_idx = next((i for i, s in enumerate(skus) if s['name'] == recommended_sku['name']), None)
        if cur_idx is not None and rec_idx is not None and rec_idx - cur_idx >= 2:
            mid_sku = skus[rec_idx - 1]
            mid_cost = mid_sku.get('monthly_usd', 0) - current_sku.get('monthly_usd', 0)
            recs['option_c'] = {
                'action': 'COMBINED',
                'sku': mid_sku['name'],
                'monthly_cost_delta': mid_cost,
                'impact': (
                    f'Moderate scale-up to {mid_sku["name"]} (+${mid_cost:,}/mo) '
                    f'AND reschedule background jobs to off-peak hours.'
                ),
            }

    # --- Option D: Optimise Only (no scaling) ---
    if _is_sched_issue and current_sku:
        burst_ratio = metrics.get('burst_ratio', 1)
        est_post_opt_util = min(avg_util * burst_ratio * 0.6, 95)  # staggering reduces peaks ~40%
        recs['option_d'] = {
            'action': 'OPTIMISE ONLY',
            'sku': current_sku['name'],
            'monthly_cost': current_sku.get('monthly_usd', 0),
            'cost_delta': 0,
            'expected_util': round(est_post_opt_util, 1),
            'impact': (
                f'Stay on {current_sku["name"]} — no cost change. '
                f'Stagger top background refreshes into off-peak windows to eliminate burst peaks.'
            ),
        }

    # --- Growth warning ---
    trend = metrics.get('trend', {})
    if trend.get('has_trend') and trend.get('direction') == 'GROWING' and trend.get('weekly_growth_pct', 0) > 3:
        recs['growth_warning'] = {
            'weekly_growth_pct': trend['weekly_growth_pct'],
            'direction': trend['direction'],
            'forecast': (
                f'Consumption is growing at {trend["weekly_growth_pct"]:+.1f}%/week. '
                f'Plan for the next SKU tier within the coming weeks.'
            ),
        }

    # --- Notes ---
    if metrics.get('_downsize_vetoed'):
        recs['notes'].append(('veto', metrics.get('_veto_reason', 'Downsize blocked due to active throttling.')))
    if metrics.get('_demand_adjusted'):
        recs['notes'].append(('demand', (
            f'Measured P80 CUs ({metrics["_unadjusted_p80"]:,.0f}) adjusted upward to '
            f'{metrics["_adjusted_p80"]:,.0f} to account for demand suppressed by rejection throttling.'
        )))
    if metrics.get('_carryforward_data_suspect'):
        recs['notes'].append(('carryforward', (
            f'Carryforward data shows 0% despite {days_throttled} throttled days. '
            f'This is physically impossible — throttling is caused by accumulated carryforward debt. '
            f'The Capacity Metrics semantic model may not be exposing this data correctly.'
        )))

    return recs


_ICON_PATHS = {
    'burst':        '<polyline points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    'target':       '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    'cog':          '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
    'calendar':     '<rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>',
    'bar-chart':    '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    'alert-circle': '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>',
    'dollar':       '<line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    'trending-up':  '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    'info':         '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    'shield':       '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    'trending-down':'<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/>',
    'zap':          '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    'layers':       '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
}

def _report_icon(name, color='currentColor', size=18):
    """Return inline SVG icon for HTML reports."""
    path = _ICON_PATHS.get(name, _ICON_PATHS.get('info', ''))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="vertical-align:middle;display:inline-block;">{path}</svg>'
    )

_CONSOLE_MARKERS = {
    'burst': '[!]', 'target': '[*]', 'cog': '[~]', 'calendar': '[=]',
    'bar-chart': '[-]', 'alert-circle': '[!]', 'dollar': '[$]',
    'trending-up': '[^]', 'info': '[i]', 'shield': '[#]',
    'trending-down': '[v]', 'zap': '[!]', 'layers': '[=]',
}


def generate_executive_summary(metrics, tiered_recs, current_sku, recommended_sku):
    """Generate 2-3 sentence plain-English summary for the top of the report."""
    sku_name = current_sku['name'] if current_sku else 'Unknown'
    days_throttled = metrics.get('days_with_delay', 0)
    total_days = max(metrics.get('days_analyzed', 1), 1)

    if tiered_recs and tiered_recs.get('_scheduling_issue'):
        return (f'Your {sku_name} capacity is correctly sized by the 80/80 method, '
                f'but {days_throttled}/{total_days} days of throttling indicate workload scheduling issues. '
                f'Staggering your top background refreshes should resolve this without scaling.')
    elif tiered_recs and tiered_recs.get('urgency') in ('CRITICAL', 'HIGH'):
        opt_a = tiered_recs.get('option_a', {})
        target = opt_a.get('to_sku', 'the next tier')
        return (f'Your {sku_name} is under severe pressure with {days_throttled}/{total_days} days throttled. '
                f'Scaling to {target} is recommended to eliminate throttling.')
    elif tiered_recs and tiered_recs.get('option_a', {}).get('action') == 'DOWNSIZE':
        target = tiered_recs['option_a'].get('to_sku', 'a smaller SKU')
        return (f'Your {sku_name} is oversized with low utilisation and minimal throttling. '
                f'Consider downsizing to {target} to reduce costs.')
    else:
        return (f'Your {sku_name} capacity is healthy with {metrics.get("avg_util", 0):.0f}% average utilisation. '
                f'Continue monitoring for changes in consumption patterns.')


def generate_capacity_intelligence(metrics, current_sku, recommended_sku,
                                    scheduling_advice, tiered_recs,
                                    sku_analysis, daily_summary=None):
    """Generate narrative insights from capacity data. Returns list of insight dicts.

    Each insight is a dict with:
      id       - unique key for deduplication
      icon     - icon ID (used with _report_icon() for HTML, _CONSOLE_MARKERS for terminal)
      title    - heading text
      body     - list of explanation strings
      severity - warning | info | success | danger
      action   - optional specific action to take
    """
    insights = []

    avg_util = metrics.get('avg_util', 0)
    max_util = metrics.get('max_util', 0)
    burst_ratio = metrics.get('burst_ratio', 1.0)
    days_analyzed = metrics.get('days_analyzed', 1)
    days_with_delay = metrics.get('days_with_delay', 0)
    avg_bg_pct = metrics.get('avg_background_pct', 0)
    avg_int_reject = metrics.get('avg_int_reject_pct', 0)
    avg_bg_reject = metrics.get('avg_bg_reject_pct', 0)
    combined_reject = avg_int_reject + avg_bg_reject
    weekday_avg = metrics.get('weekday_avg_cus', 0)
    weekend_avg = metrics.get('weekend_avg_cus', 0)
    wk_ratio = metrics.get('weekday_weekend_ratio', 1.0)
    trend_growth = metrics.get('trend_weekly_growth_pct', 0)
    spike_days = metrics.get('spike_days_filtered', 0)
    throttle_pct = days_with_delay / max(days_analyzed, 1) * 100

    # Workload concentration from scheduling_advice
    total_cus = sum(s.get('total_cus', s.get('daily_cus', 0)) for s in scheduling_advice) if scheduling_advice else 0
    top1_pct = 0
    top3_pct = 0
    if scheduling_advice and total_cus > 0:
        top1_pct = scheduling_advice[0].get('total_cus', scheduling_advice[0].get('daily_cus', 0)) / total_cus * 100
        top3_pct = sum(s.get('total_cus', s.get('daily_cus', 0)) for s in scheduling_advice[:3]) / total_cus * 100

    # --- Pattern 1: Low utilisation + high throttling (the paradox) ---
    if avg_util < 50 and throttle_pct > 50:
        body = [
            f'Average utilisation is only {avg_util:.0f}% \u2014 but {days_with_delay}/{days_analyzed} days experience throttling.'
        ]
        if burst_ratio > 2:
            body.append(
                f'Peak 30-second windows reach {max_util:.0f}% ({burst_ratio:.1f}\u00d7 the average), '
                f'creating carryforward debt that cascades into sustained throttling.'
            )
        if combined_reject > 5:
            body.append(
                f'An estimated {combined_reject:.0f}% of demand is being rejected before it is measured, '
                f'further suppressing the observed average.'
            )
        body.append('The problem is burst scheduling concentration, not overall capacity size.')
        insights.append({
            'id': 'low_util_high_throttle',
            'icon': 'burst',
            'title': 'Why is utilisation low but throttling high?',
            'body': body,
            'severity': 'warning',
            'action': 'Stagger your heaviest refresh schedules into separate time windows to spread the load.',
        })

    # --- Pattern 2: Workload concentration ---
    if scheduling_advice and top1_pct > 30:
        top_item = scheduling_advice[0]
        top_name = top_item.get('item_name', top_item.get('name', 'Unknown'))
        body = [f'"{top_name}" consumes {top1_pct:.0f}% of all capacity units.']
        if len(scheduling_advice) >= 3:
            body.append(f'The top 3 workloads account for {top3_pct:.0f}% of total consumption.')
        all_same_type = len(set(s.get('item_type', s.get('type', '')) for s in scheduling_advice[:5])) == 1
        if all_same_type and scheduling_advice:
            item_type = scheduling_advice[0].get('item_type', scheduling_advice[0].get('type', 'Unknown'))
            body.append(f'All top consumers are the same type ({item_type}) \u2014 a single scheduling change could have outsized impact.')
        insights.append({
            'id': 'workload_concentration',
            'icon': 'target',
            'title': 'High workload concentration',
            'body': body,
            'severity': 'info',
            'action': f'Optimising the schedule of the top workload alone could reduce peak demand by up to {top1_pct:.0f}%.',
        })

    # --- Pattern 3: All background workload ---
    if avg_bg_pct > 90:
        insights.append({
            'id': 'all_background',
            'icon': 'cog',
            'title': 'Pure refresh-processing capacity',
            'body': [
                f'Background workloads account for {avg_bg_pct:.0f}% of consumption.',
                'Interactive queries are minimal \u2014 this capacity is primarily a data processing engine.',
                'Scheduling is your primary optimisation lever, since refresh timing is fully controllable.',
            ],
            'severity': 'info',
        })

    # --- Pattern 4: Scheduling opportunity ---
    schedulable_count = sum(1 for s in scheduling_advice if s.get('schedulable', False)) if scheduling_advice else 0
    if schedulable_count >= 2 and burst_ratio > 2:
        sched_items = [s for s in scheduling_advice if s.get('schedulable', False)][:3]
        item_names = [f'"{s.get("item_name", s.get("name", "?"))}"' for s in sched_items]
        body = [
            f'{schedulable_count} of the top consumers are schedulable workloads.',
            f'With a burst ratio of {burst_ratio:.1f}\u00d7, staggering these into separate windows would flatten peak demand.',
        ]
        insights.append({
            'id': 'scheduling_opportunity',
            'icon': 'calendar',
            'title': 'Scheduling optimisation opportunity',
            'body': body,
            'severity': 'info',
            'action': f'Stagger {", ".join(item_names)} into non-overlapping windows (e.g. 30-minute gaps between each).',
        })

    # --- Pattern 5: Weekend same as weekday ---
    if wk_ratio and 0.8 <= wk_ratio <= 1.2 and weekday_avg > 0 and weekend_avg > 0:
        insights.append({
            'id': 'weekend_same_as_weekday',
            'icon': 'bar-chart',
            'title': 'Consistent weekday/weekend consumption',
            'body': [
                f'Weekday average ({weekday_avg:,.0f} CUs) and weekend average ({weekend_avg:,.0f} CUs) are within 20% of each other.',
                'Refreshes likely run on the same schedule 7 days a week.',
            ],
            'severity': 'info',
            'action': 'Consider reducing weekend refresh frequency, or pausing the capacity on weekends if weekend data freshness is not critical.',
        })

    # --- Pattern 6: Genuinely undersized ---
    if avg_util > 70 and throttle_pct > 50:
        rec_name = recommended_sku.get('name', 'next SKU')
        insights.append({
            'id': 'genuinely_undersized',
            'icon': 'alert-circle',
            'title': 'Capacity is genuinely undersized',
            'body': [
                f'Average utilisation is {avg_util:.0f}% with {days_with_delay}/{days_analyzed} days throttled.',
                'This is not a scheduling problem \u2014 the workload genuinely exceeds capacity.',
                'Scaling up is the primary fix; scheduling optimisation provides secondary relief.',
            ],
            'severity': 'danger',
            'action': f'Scale up to {rec_name} to bring utilisation under control.',
        })

    # --- Pattern 7: Oversized, no throttle ---
    if avg_util < 30 and throttle_pct == 0 and current_sku.get('name') != recommended_sku.get('name'):
        savings = 0
        if isinstance(sku_analysis, list):
            for s in sku_analysis:
                if s.get('SKU') == recommended_sku.get('name') or s.get('name') == recommended_sku.get('name'):
                    savings = s.get('Savings $/mo', s.get('savings_monthly', 0))
                    break
        body = [
            f'Average utilisation is only {avg_util:.0f}% with zero throttling detected.',
            'The capacity is significantly oversized for the current workload.',
        ]
        if savings > 0:
            body.append(f'Downsizing to {recommended_sku.get("name")} could save ${savings:,.0f}/month.')
        insights.append({
            'id': 'oversized_no_throttle',
            'icon': 'dollar',
            'title': 'Capacity is oversized',
            'body': body,
            'severity': 'success',
            'action': f'Consider downsizing to {recommended_sku.get("name")} for cost savings.',
        })

    # --- Pattern 8: Growth alert ---
    if trend_growth and trend_growth > 10:
        current_budget = current_sku.get('budget_daily', current_sku.get('daily_budget', 0))
        weeks_to_exceed = None
        if current_budget > 0 and metrics.get('p80_daily_cus', 0) > 0:
            p80 = metrics['p80_daily_cus']
            headroom = current_budget - p80
            weekly_increase = p80 * (trend_growth / 100)
            if weekly_increase > 0:
                weeks_to_exceed = int(headroom / weekly_increase)
        body = [f'Consumption is growing at approximately {trend_growth:.0f}% per week.']
        if weeks_to_exceed and 0 < weeks_to_exceed < 52:
            body.append(f'At this rate, the current SKU budget will be exceeded in approximately {weeks_to_exceed} weeks.')
        body.append('Monitor weekly and plan scaling proactively.')
        insights.append({
            'id': 'growth_alert',
            'icon': 'trending-up',
            'title': 'Rapid consumption growth detected',
            'body': body,
            'severity': 'warning',
        })

    # --- Pattern 9: Spike days ---
    if spike_days and spike_days > 0:
        insights.append({
            'id': 'spike_days_impact',
            'icon': 'info',
            'title': 'Anomalous days excluded from sizing',
            'body': [
                f'{spike_days} day(s) with anomalous consumption were detected (likely pause/resume settlement spikes).',
                'These days were excluded from the P80 calculation to avoid oversizing.',
                'If these spikes are recurring (e.g. weekly pause/resume), consider alternatives to pausing.',
            ],
            'severity': 'info',
        })

    # --- Pattern 10: Throttle veto active ---
    if metrics.get('_downsize_vetoed'):
        p80_sku = metrics.get('_p80_recommended_sku', recommended_sku.get('name', '?'))
        insights.append({
            'id': 'throttle_veto_active',
            'icon': 'shield',
            'title': 'Downsize blocked by throttling',
            'body': [
                f'The 80/80 sizing method suggests {p80_sku} would be sufficient for the measured workload.',
                f'However, downsizing is blocked because throttling is active on {days_with_delay}/{days_analyzed} days.',
                'Fix the scheduling concentration first, then re-evaluate sizing with reduced peak demand.',
            ],
            'severity': 'warning',
            'action': 'Optimise refresh schedules to eliminate throttling, then re-run the advisor to check if downsizing becomes safe.',
        })

    # --- Pattern 11: Demand suppressed ---
    if metrics.get('_demand_adjusted'):
        raw_p80 = metrics.get('p80_daily_cus', 0)
        adj_p80 = metrics.get('_adjusted_p80', 0)
        insights.append({
            'id': 'demand_suppressed',
            'icon': 'trending-down',
            'title': 'Measured demand is suppressed by rejection',
            'body': [
                f'Measured P80 is {raw_p80:,.0f} CUs, but estimated true demand is {adj_p80:,.0f} CUs.',
                f'Approximately {combined_reject:.0f}% of operations are being rejected before they consume CUs.',
                'The real workload is larger than what the metrics show \u2014 sizing accounts for this gap.',
            ],
            'severity': 'warning',
        })

    return insights


# ============================================================================
# SECTION 7b: CONTEXTUAL SUBTITLES (v2.1.1)
# ============================================================================

def generate_chart_subtitle(fig_name, metrics, recommended_sku, current_sku,
                            tiered_recs=None, sku_analysis=None):
    """Return a contextual one-line subtitle for a chart/section, or None."""

    if fig_name == 'fig_util_gauges':
        avg = metrics.get('avg_util', 0)
        peak = metrics.get('max_util', 0)
        sku_name = recommended_sku['name'] if recommended_sku else 'current SKU'
        if avg < 40:
            return (f'Average utilisation is well below 40% on {sku_name} '
                    '\u2014 this capacity has significant spare headroom.')
        elif avg <= 65:
            return (f'Average utilisation sits in the green zone ({avg:.0f}%) '
                    'with comfortable headroom for peaks.')
        elif avg <= 80:
            if peak > 95:
                return (f'Average is healthy ({avg:.0f}%) but peak hits {peak:.0f}% '
                        '\u2014 spiky workloads may trigger throttling.')
            return (f'Utilisation is well-balanced at {avg:.0f}% average, '
                    f'{peak:.0f}% peak \u2014 close to the 80% target.')
        else:
            return (f'Peak utilisation regularly exceeds 80% ({peak:.0f}% peak) '
                    '\u2014 capacity is under sustained pressure.')

    if fig_name == 'fig_daily':
        trend = metrics.get('trend', {})
        if not trend.get('has_trend'):
            return None
        direction = trend['direction']
        weekly = trend['weekly_growth_pct']
        days = metrics.get('days_analyzed', 0)
        caution = ' (based on limited data)' if days < 14 else ''
        if direction == 'GROWING':
            return (f'Consumption is growing at ~{weekly:+.1f}%/week{caution} '
                    '\u2014 plan for the next tier if this continues.')
        elif direction == 'DECLINING':
            return (f'Consumption is declining at ~{weekly:+.1f}%/week{caution} '
                    '\u2014 monitor before committing to a larger SKU.')
        return f'Consumption is stable over the analysis period{caution}.'

    if fig_name == 'fig_weekday_weekend':
        ratio = metrics.get('weekday_weekend_ratio', 1)
        if ratio == float('inf') or ratio > 100:
            return 'No significant weekend consumption \u2014 sizing is based entirely on weekday demand.'
        if ratio > 2.0:
            return (f'Weekdays drive {ratio:.1f}\u00d7 more consumption than weekends '
                    '\u2014 sizing is based on weekday P80.')
        if ratio > 1.3:
            return (f'Weekdays are moderately busier ({ratio:.1f}\u00d7 weekends) '
                    '\u2014 weekday P80 drives the SKU recommendation.')
        return ('Weekday and weekend consumption are similar '
                '\u2014 check whether weekend refreshes are genuinely needed.')

    if fig_name == 'fig_sku':
        if sku_analysis and recommended_sku:
            rec_name = recommended_sku['name']
            rec_row = next((sa for sa in sku_analysis if sa['SKU'] == rec_name), None)
            if rec_row:
                rec_util = rec_row['Avg Util %']
                fit = 'well within' if rec_util <= 75 else 'near'
                return (f'The recommended {rec_name} would run at ~{rec_util}% '
                        f'average utilisation \u2014 {fit} the 80% target.')
        return None

    if fig_name == 'fig_items':
        sched = (tiered_recs or {}).get('scheduling_advice', [])
        if not sched:
            return None
        n = min(len(sched), 3)
        total_budget_pct = sum(s.get('budget_pct', 0) or 0 for s in sched[:n])
        if total_budget_pct > 0:
            return (f'The top {n} items account for ~{total_budget_pct:.0f}% '
                    'of daily CU consumption.')
        top_cus = sum(s.get('daily_cus', 0) for s in sched[:n])
        avg_daily = metrics.get('avg_daily_cus', 1)
        if avg_daily > 0:
            pct = (top_cus / avg_daily) * 100
            return (f'The top {n} items account for ~{pct:.0f}% '
                    'of average daily CU consumption.')
        return None

    if fig_name == 'fig_cost':
        avg_util = metrics.get('avg_util', 0)
        if avg_util > RESERVED_BREAKEVEN_UTIL * 100:
            return (f'At {avg_util:.0f}% utilisation, a reserved instance '
                    'is likely cheaper than pay-as-you-go.')
        return (f'At {avg_util:.0f}% utilisation, PAYG with pause/resume '
                'may be more cost-effective than committing to reserved.')

    return None


def recommend_sku_for_capacity(metrics, skus, needs_free_viewers, weekday_split=True, spike_filter=True):
    """Recommend SKU using 80/80 approach.

    v2.0 additions:
    - Throttling-adjusted demand: if rejection is suppressing measured CUs,
      estimate true demand before sizing
    - Throttling veto: never downsize if throttling is active
    """
    if weekday_split and spike_filter:
        p80_cus = metrics.get('weekday_p80_cus_filtered', metrics['p80_daily_cus'])
    elif weekday_split:
        p80_cus = metrics.get('weekday_p80_cus', metrics['p80_daily_cus'])
    elif spike_filter:
        p80_cus = metrics.get('p80_daily_cus_filtered', metrics['p80_daily_cus'])
    else:
        p80_cus = metrics['p80_daily_cus']

    # v2.0: Adjust P80 upward if demand is being rejected
    avg_reject = metrics.get('avg_int_reject_pct', 0) + metrics.get('avg_bg_reject_pct', 0)
    if avg_reject > 2:
        metrics['_unadjusted_p80'] = p80_cus
        p80_cus = estimate_true_demand(p80_cus, avg_reject)
        metrics['_demand_adjusted'] = True
        metrics['_adjusted_p80'] = p80_cus

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

    # v2.0: Throttling veto — never downsize into active throttling
    current_sku = metrics.get('current_sku')
    if current_sku is not None and recommended_sku is not None:
        avg_delay = metrics.get('avg_delay_pct', 0)
        avg_int_reject = metrics.get('avg_int_reject_pct', 0)
        days_throttled = metrics.get('days_with_delay', 0)
        total_days = max(metrics.get('days_analyzed', 1), 1)
        throttle_ratio = days_throttled / total_days

        is_throttled = (
            avg_delay > 10
            or avg_int_reject > 5
            or throttle_ratio > 0.5
        )

        if is_throttled and recommended_sku['cus_per_second'] < current_sku['cus_per_second']:
            metrics['_downsize_vetoed'] = True
            metrics['_veto_reason'] = (
                f"Downsize blocked: {days_throttled}/{metrics.get('days_analyzed', '?')} days throttled, "
                f"avg delay {avg_delay:.1f}%, avg rejection {avg_int_reject:.1f}%. "
                f"Low utilisation ({metrics.get('avg_util', 0):.0f}%) is misleading — "
                f"demand is being rejected, not absent."
            )
            recommended_sku = current_sku

    # v2.0: Flag suspect carryforward data
    if (metrics.get('avg_delay_pct', 0) > 10 and metrics.get('avg_carryover_pct', 0) == 0
            and metrics.get('days_with_delay', 0) > 0):
        metrics['_carryforward_data_suspect'] = True

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

    # 5. Top 10 Workloads — v2.0: exclude workspace, add Item Kind/Type label
    if df_raw is not None and len(df_raw) > 0:
        _name_cols = [c for c in df_raw.columns if 'item' in c.lower() and 'name' in c.lower() and 'workspace' not in c.lower()]
        _name_col = _name_cols[0] if _name_cols else ('Item Name' if 'Item Name' in df_raw.columns else None)
        _kind_cols = [c for c in df_raw.columns if ('kind' in c.lower() or 'type' in c.lower()) and 'item' in c.lower()]
        _kind_col = _kind_cols[0] if _kind_cols else ('Item Type' if 'Item Type' in df_raw.columns else None)
        _cu_col = 'CUs' if 'CUs' in df_raw.columns else None

        if _name_col and _cu_col:
            if _kind_col:
                item_totals = df_raw.groupby([_name_col, _kind_col])[_cu_col].sum().reset_index()
                item_totals['_label'] = item_totals[_name_col] + ' (' + item_totals[_kind_col].fillna('Unknown') + ')'
            else:
                item_totals = df_raw.groupby(_name_col)[_cu_col].sum().reset_index()
                item_totals['_label'] = item_totals[_name_col]
            item_totals = item_totals.sort_values(_cu_col, ascending=True).tail(10)
            fig_items = go.Figure(go.Bar(
                y=item_totals['_label'],
                x=item_totals[_cu_col],
                orientation='h',
                marker_color='#667eea',
                text=[
                    f"{v/1e6:,.1f}M CUs" if v >= 1e6
                    else f"{v/1e3:,.0f}K CUs" if v >= 1e3
                    else f"{v:,.0f} CUs"
                    for v in item_totals[_cu_col]
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
                         charts, daily_summary, capacity_name, df_raw,
                         tiered_recs=None, intelligence=None):
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

    # Health rating for header tag
    _health_rating = metrics.get('health_rating', 'POOR').lower()
    _health_score = metrics.get('health_score', 0)

    # --- HTML header + CSS (editorial design system) ---
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fabric SKU Advisor — {capacity_name}</title>
<style>
*,*::before,*::after{{margin:0;padding:0;box-sizing:border-box}}
:root{{
--ink:#1a1a2e;--ink-light:#2d2d44;--ink-muted:#4a4a5a;--ink-faint:#6b6b7b;
--paper:#ffffff;--paper-warm:#f8f8fa;--white:#ffffff;
--accent:#9a3412;--accent-light:#fff7ed;--accent-hover:#7c2d12;
--rule:#d4d4d8;--rule-strong:#a1a1aa;
--success:#15803d;--success-bg:#f0fdf4;
--warning:#9a3412;--warning-bg:#fff7ed;
--danger:#b91c1c;--danger-bg:#fef2f2;
--info:#1d4ed8;--info-bg:#eff6ff;
--shadow-sm:0 1px 2px rgba(15,23,42,0.04);
--shadow-md:0 4px 12px rgba(15,23,42,0.06);
--radius:4px;--radius-lg:8px;
--font-display:'Segoe UI',system-ui,sans-serif;
--font-body:'Segoe UI',system-ui,sans-serif;
--font-mono:'Consolas','Courier New',monospace;
}}
body{{font-family:var(--font-body);background:var(--paper);color:var(--ink);line-height:1.65;font-size:15px;-webkit-font-smoothing:antialiased}}
.container{{max-width:920px;margin:0 auto;padding:48px 32px 64px}}
/* Header bar */
.header-bar{{background:var(--ink);color:white;padding:28px 40px 24px}}
.header-bar .brand{{font-size:10px;font-weight:600;letter-spacing:2.5px;text-transform:uppercase;color:rgba(255,255,255,0.45);margin-bottom:14px}}
.header-bar .brand-nova{{color:rgba(255,255,255,0.85);font-weight:700}}
.header-bar .main-row{{display:flex;justify-content:space-between;align-items:flex-end}}
.header-bar .cap-name{{font-size:2.2em;font-weight:700;letter-spacing:-0.02em;line-height:1.1}}
.header-bar .tags{{display:flex;gap:10px;align-items:center}}
.header-bar .sku-tag{{font-family:var(--font-mono);font-size:20px;font-weight:700;background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.15);padding:5px 16px;border-radius:4px}}
.header-bar .health-tag{{font-size:11px;font-weight:700;letter-spacing:0.5px;padding:6px 14px;border-radius:4px}}
.header-bar .health-tag.poor,.header-bar .health-tag.critical{{background:#fca5a5;color:#7f1d1d}}
.header-bar .health-tag.fair{{background:#fde68a;color:#78350f}}
.header-bar .health-tag.good{{background:#86efac;color:#14532d}}
.header-bar .health-tag.excellent{{background:#bbf7d0;color:#14532d}}
/* Header meta row */
.header-meta{{display:flex;gap:0;border-bottom:1px solid var(--rule)}}
.header-meta .meta-item{{flex:1;padding:14px 20px;border-right:1px solid var(--rule)}}
.header-meta .meta-item:first-child{{padding-left:40px}}
.header-meta .meta-item:last-child{{border-right:none}}
.header-meta .meta-label{{font-size:10px;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;color:var(--ink-muted);margin-bottom:2px}}
.header-meta .meta-value{{font-family:var(--font-mono);font-size:14px;font-weight:600;color:var(--ink)}}
/* Disclaimer line */
.header-disclaimer{{padding:10px 40px;font-size:11px;color:var(--ink-faint);background:var(--paper-warm)}}
/* KPI grid */
.kpi-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:var(--radius);margin-bottom:48px;overflow:hidden}}
.kpi{{background:var(--white);padding:24px 20px;text-align:center}}
.kpi .value{{font-family:var(--font-display);font-size:2.4em;font-weight:700;color:var(--ink);line-height:1;margin-bottom:6px;letter-spacing:-0.02em}}
.kpi .value.accent{{color:var(--accent)}}.kpi .value.danger{{color:var(--danger)}}.kpi .value.success{{color:var(--success)}}.kpi .value.warning{{color:#e67700}}
.kpi .label{{font-size:11px;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;color:var(--ink-muted)}}
.kpi .sublabel{{color:var(--ink-muted);margin-top:4px;font-family:var(--font-mono);font-size:12px}}
.kpi-note{{font-size:0.72em;color:var(--ink-muted);margin-top:2px;font-style:italic}}
.exec-summary{{background:#eff6ff;border-left:4px solid #3b82f6;padding:14px 18px;border-radius:8px;margin:0 0 20px 0;font-size:0.95em;line-height:1.5}}
.dim-row{{opacity:0.4}}
.kpi:last-child:nth-child(4n+1){{grid-column:1/-1}}
.kpi:nth-last-child(2):nth-child(4n+1),.kpi:last-child:nth-child(4n+2){{grid-column:span 2}}
.kpi:last-child:nth-child(4n+3){{grid-column:span 2}}
/* Sections */
.section{{margin-bottom:48px}}
.section-head{{display:flex;align-items:baseline;justify-content:space-between;border-bottom:1px solid var(--rule);padding-bottom:12px;margin-bottom:20px}}
.section-head h2{{font-family:var(--font-display);font-size:1.1em;font-weight:600;color:var(--ink-light);letter-spacing:-0.01em}}
.help-btn{{background:none;border:1px solid var(--rule);color:var(--ink-faint);width:24px;height:24px;border-radius:50%;cursor:pointer;font-size:12px;font-weight:600;font-family:var(--font-body);transition:all 0.15s;flex-shrink:0}}
.help-btn:hover{{border-color:var(--accent);color:var(--accent)}}
/* Group divider */
.group-divider{{display:flex;align-items:center;gap:12px;margin:56px 0 24px;color:var(--accent)}}
.group-divider::after{{content:'';flex:1;height:1px;background:var(--rule)}}
.group-divider .group-label{{font-size:14px;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;white-space:nowrap}}
.group-divider svg{{flex-shrink:0}}
/* Intel cards */
.intel-grid{{display:flex;flex-direction:column;gap:16px}}
.intel-card{{padding:20px 24px;border-left:3px solid var(--rule);background:var(--white)}}
.intel-card.warning{{border-color:var(--danger);background:var(--danger-bg)}}
.intel-card.info{{border-color:var(--info);background:var(--info-bg)}}
.intel-card.success{{border-color:var(--success);background:var(--success-bg)}}
.intel-card.danger{{border-color:var(--danger);background:var(--danger-bg)}}
.intel-card .intel-head{{display:flex;align-items:flex-start;gap:10px;margin-bottom:10px}}
.intel-card .intel-head svg{{flex-shrink:0;margin-top:2px}}
.intel-card .intel-head strong{{font-family:var(--font-display);font-size:1.05em;font-weight:600;color:var(--ink);line-height:1.3}}
.intel-card ul{{margin:0 0 0 30px;padding:0;font-size:14px;color:var(--ink-light);line-height:1.6}}
.intel-card ul li{{margin-bottom:4px}}
.intel-card .intel-action{{margin-top:12px;padding:10px 14px;background:rgba(15,23,42,0.03);border-radius:var(--radius);font-size:13px;color:var(--ink-light)}}
.intel-card .intel-action strong{{color:var(--accent)}}
/* Collapsible sections */
details.collapsible summary.section-head{{cursor:pointer;list-style:none;user-select:none}}
details.collapsible summary.section-head::-webkit-details-marker{{display:none}}
details.collapsible summary.section-head::after{{content:"\u25b8";font-size:18px;color:var(--ink-faint);transition:transform 0.2s;margin-left:8px}}
details.collapsible[open] summary.section-head::after{{transform:rotate(90deg)}}
details.collapsible summary.section-head .count-badge{{font-size:11px;font-weight:500;color:var(--ink-faint);background:var(--paper-warm);padding:2px 10px;border-radius:10px;margin-left:8px}}
details.collapsible summary.section-head .section-label{{display:flex;align-items:baseline;gap:8px;flex:1}}
/* Charts */
.chart-wrap{{overflow:hidden;background:var(--white);border:1px solid var(--rule);border-radius:var(--radius);padding:16px;margin:16px 0;min-height:320px}}
/* Stat pair */
.stat-pair{{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:var(--radius);overflow:hidden;margin:16px 0}}
.stat-pair .stat-item{{background:var(--white);padding:20px 24px}}
.stat-pair .stat-item .stat-label{{font-size:11px;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;color:var(--ink-muted);margin-bottom:4px}}
.stat-pair .stat-item .stat-value{{font-family:var(--font-mono);font-size:1.6em;font-weight:500;color:var(--ink)}}
.stat-pair .stat-item .stat-detail{{font-size:12px;color:var(--ink-muted);margin-top:2px}}
/* Tables */
table{{width:100%;border-collapse:collapse;font-size:14px}}
table th{{text-align:left;padding:10px 16px;font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:var(--ink-muted);border-bottom:2px solid var(--ink-muted)}}
table td{{padding:10px 16px;border-bottom:1px solid var(--rule);color:var(--ink)}}
table tr:last-child td{{border-bottom:none}}
table .highlight{{background:var(--success-bg);font-weight:600}}
table .text-right{{text-align:right}}
table .mono{{font-family:var(--font-mono);font-size:13px;font-weight:600}}
/* Urgency banner */
.urgency-banner{{padding:14px 20px;border-left:3px solid;font-weight:600;font-size:14px;margin-bottom:20px;letter-spacing:0.3px}}
.urgency-banner.critical{{background:var(--danger-bg);border-color:var(--danger);color:var(--danger)}}
.urgency-banner.high{{background:var(--warning-bg);border-color:var(--accent);color:var(--warning)}}
.urgency-banner.moderate{{background:var(--warning-bg);border-color:var(--accent);color:var(--warning)}}
.urgency-banner.low{{background:var(--success-bg);border-color:var(--success);color:var(--success)}}
/* Option cards */
.option-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin:20px 0}}
.option-card{{border:1px solid var(--rule);border-radius:var(--radius);padding:20px;background:var(--white)}}
.option-card h3{{font-family:var(--font-display);font-size:1.1em;font-weight:600;margin-bottom:8px;color:var(--ink)}}
.option-card .tag{{display:inline-block;padding:3px 12px;border-radius:2px;font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:10px}}
.tag.scale-up{{background:var(--danger-bg);color:var(--danger)}}
.tag.optimise{{background:var(--info-bg);color:var(--info)}}
.tag.combined{{background:var(--warning-bg);color:var(--warning)}}
.tag.stay{{background:#f1f5f9;color:var(--ink-muted)}}
.tag.downsize{{background:var(--success-bg);color:var(--success)}}
.option-card p{{font-size:14px;color:var(--ink-light);margin-bottom:8px;line-height:1.5}}
.option-card .cost-note{{font-size:13px;font-weight:500;color:var(--ink-light);margin-top:10px;padding-top:8px;border-top:1px solid var(--rule)}}
/* Scheduling list */
.scheduling-list{{border-left:3px solid var(--accent);padding:16px 24px;background:var(--accent-light);margin:24px 0;border-radius:0 var(--radius) var(--radius) 0}}
.scheduling-list h4{{font-size:12px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:var(--accent);margin-bottom:10px}}
.scheduling-list ul{{margin:0;padding-left:20px}}
.scheduling-list li{{font-size:14px;color:var(--ink);margin-bottom:6px;line-height:1.6;padding-left:4px}}
/* Alert items */
.alert-stack{{display:flex;flex-direction:column;gap:8px;margin:16px 0}}
.alert-item{{padding:12px 16px;border-left:3px solid;font-size:14px;line-height:1.6;color:var(--ink);border-radius:0 var(--radius) var(--radius) 0}}
.alert-item.danger{{border-color:var(--danger);background:var(--danger-bg)}}
.alert-item.info{{border-color:var(--info);background:var(--info-bg)}}
.alert-item.warning{{border-color:var(--accent);background:var(--warning-bg)}}
/* Peak badges */
.peak-summary{{display:flex;gap:16px;margin:16px 0}}
.peak-badge{{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;border-radius:2px;font-size:12px;font-weight:600;font-family:var(--font-mono)}}
.peak-badge.peak{{background:var(--danger-bg);color:var(--danger)}}
.peak-badge.offpeak{{background:var(--success-bg);color:var(--success)}}
/* Note box */
.note-box{{padding:12px 16px;border-left:2px solid var(--accent);background:var(--warning-bg);font-size:13px;color:var(--ink-light);margin:12px 0;line-height:1.5}}
.note-box.info{{border-color:var(--info);background:var(--info-bg)}}
.note-box.danger{{border-color:var(--danger);background:var(--danger-bg)}}
/* Spark note */
.spark-note{{font-size:12px;color:var(--ink-muted);margin-top:12px;padding:10px 14px;background:var(--paper-warm);border-radius:var(--radius)}}
/* Footer bar */
.footer-bar{{background:var(--ink);color:white;padding:24px 40px;margin-top:56px}}
.footer-bar .footer-top{{display:flex;justify-content:space-between;align-items:center}}
.footer-bar .footer-brand{{font-size:10px;font-weight:600;letter-spacing:2.5px;text-transform:uppercase;color:rgba(255,255,255,0.45)}}
.footer-bar .footer-brand .brand-nova{{color:rgba(255,255,255,0.85);font-weight:700}}
.footer-bar .footer-top a{{color:rgba(255,255,255,0.6);text-decoration:none;font-size:12px;transition:color 0.15s}}
.footer-bar .footer-top a:hover{{color:white}}
.footer-bar .footer-disclaimer{{margin-top:12px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.1);font-size:11px;color:rgba(255,255,255,0.35);line-height:1.5}}
/* Modal */
.modal-overlay{{display:none;position:fixed;inset:0;background:rgba(15,23,42,0.4);backdrop-filter:blur(4px);z-index:1000;justify-content:center;align-items:center}}
.modal-overlay.active{{display:flex}}
.modal{{background:var(--white);border-radius:var(--radius-lg);padding:36px;max-width:480px;width:90%;max-height:80vh;overflow-y:auto;position:relative;box-shadow:0 12px 40px rgba(15,23,42,0.08)}}
.modal h4{{font-family:var(--font-display);font-size:1.2em;font-weight:600;color:var(--ink);margin-bottom:16px}}
.modal p,.modal li{{font-size:14px;color:var(--ink-light);line-height:1.6;margin-bottom:8px}}
.modal ul{{padding-left:20px}}
.modal-close{{position:absolute;top:16px;right:16px;background:var(--paper);border:none;width:28px;height:28px;border-radius:50%;cursor:pointer;font-size:16px;color:var(--ink-muted);transition:all 0.15s}}
.modal-close:hover{{background:var(--rule);color:var(--ink)}}
/* Subtitles (v2.1.1) */
.section-subtitle{{font-size:13px;color:var(--ink-faint);margin:-8px 0 12px 0;line-height:1.4}}
/* TOC (v2.1.1) */
.toc-fab{{position:fixed;bottom:24px;right:24px;width:44px;height:44px;border-radius:50%;background:var(--ink);color:white;border:none;font-size:20px;cursor:pointer;z-index:900;box-shadow:0 2px 12px rgba(0,0,0,0.15);transition:transform 0.2s,box-shadow 0.2s;display:flex;align-items:center;justify-content:center;line-height:1}}
.toc-fab:hover{{transform:translateY(-2px);box-shadow:0 4px 16px rgba(0,0,0,0.22)}}
.toc-fab.active{{background:var(--ink-light)}}
.toc-panel{{position:fixed;bottom:80px;right:24px;width:240px;max-height:60vh;background:var(--paper-warm);border:1px solid var(--rule);border-radius:var(--radius-lg);box-shadow:0 8px 32px rgba(0,0,0,0.10);z-index:900;overflow-y:auto;opacity:0;transform:translateY(12px);pointer-events:none;transition:opacity 0.2s,transform 0.2s}}
.toc-panel.open{{opacity:1;transform:translateY(0);pointer-events:auto}}
.toc-panel-head{{padding:12px 16px 8px;font-weight:600;font-size:13px;color:var(--ink-muted);text-transform:uppercase;letter-spacing:0.5px}}
.toc-panel ul{{list-style:none;padding:0 0 12px;margin:0}}
.toc-panel li{{margin:0}}
.toc-panel a{{display:block;padding:6px 16px;font-size:13px;color:var(--ink-muted);text-decoration:none;border-left:3px solid transparent;transition:all 0.15s}}
.toc-panel a:hover{{color:var(--ink);background:rgba(0,0,0,0.03)}}
.toc-panel a.toc-active{{color:#6366f1;border-left-color:#6366f1;font-weight:600}}
.toc-panel a.toc-indent{{padding-left:28px;font-size:12px}}
html{{scroll-behavior:smooth}}
/* Print */
@media print{{body{{background:white;font-size:12px}}.container{{padding:0;max-width:none}}.help-btn,.modal-overlay,.toc-fab,.toc-panel{{display:none !important}}.chart-wrap{{break-inside:avoid}}}}
/* Responsive */
@media (max-width:768px){{.kpi-row{{grid-template-columns:repeat(2,1fr)}}.option-grid{{grid-template-columns:1fr}}.header-bar .main-row{{flex-direction:column;gap:12px;align-items:flex-start}}.header-meta{{flex-wrap:wrap;gap:0}}.footer-bar{{flex-direction:column;gap:12px;text-align:center}}}}
</style></head><body>

<!-- Header bar -->
<div class="header-bar">
  <div class="brand"><span class="brand-nova">Data Nova</span> — Fabric SKU Advisor</div>
  <div class="main-row">
    <div class="cap-name">{capacity_name}</div>
    <div class="tags">
      <div class="sku-tag">{_cur_sku_name}</div>
      <div class="health-tag {_health_rating}">{metrics['health_rating']} · {_health_score}/100</div>
    </div>
  </div>
</div>

<!-- Meta row -->
<div class="header-meta">
  <div class="meta-item"><div class="meta-label">Period</div><div class="meta-value">{_date_range}</div></div>
  <div class="meta-item"><div class="meta-label">Days Analysed</div><div class="meta-value">{metrics['days_analyzed']}</div></div>
  <div class="meta-item"><div class="meta-label">Avg Utilisation</div><div class="meta-value">{metrics['avg_util']:.1f}%</div></div>
  <div class="meta-item"><div class="meta-label">Days Throttled</div><div class="meta-value">{metrics.get('days_with_delay', 0)} / {metrics['days_analyzed']}</div></div>
</div>

<!-- Disclaimer -->
<div class="header-disclaimer">80/80 sizing method · Validate recommendations against your workload patterns and budget constraints</div>

<div class="container">""")

    # (Header, meta row, disclaimer already in template above)

    # --- Executive Summary ---
    exec_summary = generate_executive_summary(metrics, tiered_recs, current_sku, recommended_sku)
    html_parts.append(f'<div class="exec-summary"><strong>Summary</strong><p>{exec_summary}</p></div>')

    # (Floating TOC is rendered after container close — see below)

    # --- KPI grid ---
    kpi_lines = []

    # Row 1: Health, SKU rec, savings/cost, days
    kpi_lines.append(
        f'<div class="kpi"><div class="value">{metrics["health_score"]}</div>'
        f'<div class="label">Health Score</div>'
        f'<div class="sublabel">{metrics["health_rating"]}</div></div>'
    )

    if current_sku and recommended_sku:
        _c_idx = next((i for i, s in enumerate(SKUS) if s['name'] == current_sku['name']), -1)
        _r_idx = next((i for i, s in enumerate(SKUS) if s['name'] == recommended_sku['name']), -1)
        _arrow = '&uarr;' if _r_idx > _c_idx else ('&darr;' if _r_idx < _c_idx else '&check;')
        _word = 'Upgrade' if _r_idx > _c_idx else ('Downsize' if _r_idx < _c_idx else 'Stay')
        if _word == 'Stay' and tiered_recs and tiered_recs.get('_scheduling_issue'):
            # 80/80 says right-sized but workloads need rescheduling
            kpi_lines.append(
                f'<div class="kpi"><div class="value warning">&#9889; {current_sku["name"]}</div>'
                f'<div class="label">Reschedule Workloads</div></div>'
            )
        elif _word == 'Stay':
            _v_cls = ' success'
            kpi_lines.append(
                f'<div class="kpi"><div class="value{_v_cls}">&check; {current_sku["name"]}</div>'
                f'<div class="label">Right-Sized</div></div>'
            )
        else:
            _v_cls = ' danger' if _word == 'Upgrade' else ' success'
            kpi_lines.append(
                f'<div class="kpi"><div class="value{_v_cls}">{current_sku["name"]} {_arrow} {recommended_sku["name"]}</div>'
                f'<div class="label">{_word}</div></div>'
            )
    elif recommended_sku:
        kpi_lines.append(
            f'<div class="kpi"><div class="value">{recommended_sku["name"]}</div>'
            f'<div class="label">Recommended SKU</div></div>'
        )

    if current_sku and recommended_sku:
        _diff = current_sku.get("monthly_usd", 0) - recommended_sku.get("monthly_usd", 0)
        if _diff > 0:
            kpi_lines.append(
                f'<div class="kpi"><div class="value success">~${_diff:,}/mo</div>'
                f'<div class="label">Est. Savings</div></div>'
            )
        elif _diff < 0:
            kpi_lines.append(
                f'<div class="kpi"><div class="value accent">~+${abs(_diff):,}/mo</div>'
                f'<div class="label">Est. Upgrade Cost</div></div>'
            )
        else:
            kpi_lines.append(
                f'<div class="kpi"><div class="value">${current_sku.get("monthly_usd", 0):,}/mo</div>'
                f'<div class="label">Current Cost</div></div>'
            )

    kpi_lines.append(f'<div class="kpi"><div class="value">{metrics["avg_daily_cus"]:,.0f}</div><div class="label">Avg Daily CUs</div></div>')

    # Row 2: Utilisation, peak, throttling, trend
    if metrics.get('_demand_adjusted'):
        _adj_util = round(metrics['_adjusted_p80'] / (current_sku['budget_30s'] * 2880) * 100, 1) if current_sku else metrics['avg_util']
        kpi_lines.append(f'<div class="kpi"><div class="value">{metrics["avg_util"]:.1f}%</div><div class="label">Measured Avg Util</div></div>')
        kpi_lines.append(f'<div class="kpi"><div class="value danger">{_adj_util}%</div><div class="label">Adjusted Util (est.)</div><div class="kpi-note">Based on rejection rate — true demand is estimated higher than measured</div></div>')
    else:
        kpi_lines.append(f'<div class="kpi"><div class="value">{metrics["avg_util"]:.1f}%</div><div class="label">Avg Utilisation</div></div>')

    if metrics.get('max_util', 0) > metrics.get('avg_util', 0) * 2:
        _burst = metrics.get('burst_ratio', round(metrics['max_util'] / max(metrics['avg_util'], 1), 1))
        kpi_lines.append(f'<div class="kpi"><div class="value danger">{_burst}×</div><div class="label">Burst Ratio</div><div class="kpi-note">Peak 30s windows vs. daily average</div></div>')

    if metrics['days_with_delay'] > 0:
        kpi_lines.append(
            f'<div class="kpi"><div class="value accent">{metrics["days_with_delay"]}/{metrics["days_analyzed"]}</div>'
            f'<div class="label">Days Throttled</div></div>'
        )

    _trend = metrics.get('trend', {})
    if _trend.get('has_trend'):
        _t_cls = ' danger' if _trend['direction'] == 'GROWING' else ' success' if _trend['direction'] == 'DECLINING' else ''
        _trend_note = f'<div class="kpi-note">&#9888; Based on {metrics["days_analyzed"]} days — interpret with caution</div>' if metrics.get('days_analyzed', 99) < 14 else ''
        kpi_lines.append(
            f'<div class="kpi"><div class="value{_t_cls}">{_trend["weekly_growth_pct"]:+.1f}%/wk</div>'
            f'<div class="label">Consumption Trend</div>'
            f'<div class="sublabel">{_trend["direction"]}</div>{_trend_note}</div>'
        )

    html_parts.append('<div class="kpi-row" id="sec-kpi">' + "".join(kpi_lines) + "</div>")

    # --- Capacity Intelligence section ---
    if intelligence:
        html_parts.append(
            f'<details class="section collapsible" open id="sec-intelligence"><summary class="section-head"><span class="section-label"><h2>Capacity Intelligence</h2>'
            f'<span class="count-badge">{len(intelligence)} insights</span></span>'
            f'<button class="help-btn" onclick="event.stopPropagation();showHelp(\'intel\')" title="Help">?</button></summary>'
        )
        html_parts.append('<div class="intel-grid">')
        for _ins in intelligence:
            _sev = _ins.get('severity', 'info')
            html_parts.append(f'<div class="intel-card {_sev}">')
            html_parts.append(
                f'<div class="intel-head">{_report_icon(_ins["icon"], "#1a1a2e")} '
                f'<strong>{_ins["title"]}</strong></div>'
            )
            html_parts.append(
                '<ul>'
                + ''.join(f'<li>{b}</li>' for b in _ins['body'])
                + '</ul>'
            )
            if _ins.get('action'):
                html_parts.append(
                    f'<div class="intel-action"><strong>\u2192</strong> {_ins["action"]}</div>'
                )
            html_parts.append('</div>')
        html_parts.append('</div></details>')

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
            f'<div class="section" id="sec-weekday"><div class="section-head"><h2>Weekday vs Weekend</h2>'
            f'<button class="help-btn" onclick="showHelp(\'weekday\')" title="Help">?</button></div>'
            f'<div class="note-box info"><strong>Insight:</strong> {_insight}</div>'
            f'<div class="stat-pair">'
            f'<div class="stat-item"><div class="stat-label">Weekday Avg</div>'
            f'<div class="stat-value">{metrics["weekday_avg_cus"]:,.0f}</div>'
            f'<div class="stat-detail">{metrics.get("weekday_avg_util", 0):.1f}% utilisation</div></div>'
            f'<div class="stat-item"><div class="stat-label">Weekend Avg</div>'
            f'<div class="stat-value">{metrics["weekend_avg_cus"]:,.0f}</div>'
            f'<div class="stat-detail">{metrics.get("weekend_avg_util", 0):.1f}% utilisation</div></div>'
            f'</div></div>'
        )

    # --- Spike filtering note ---
    if metrics.get('spike_days_detected', 0) > 0:
        _n = metrics['spike_days_detected']
        html_parts.append(
            f'<div class="section"><div class="section-head"><h2>Spike Filtering</h2>'
            f'<button class="help-btn" onclick="showHelp(\'spike\')" title="Help">?</button></div>'
            f'<div class="note-box"><strong>Note:</strong> {_n} day(s) showed anomalously high peak CUs, '
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

    # Group dividers inserted before specific charts
    _group_breaks = {
        "fig_util_gauges": ("bar-chart", "Consumption Analysis"),
        "fig_throttle": ("zap", "Throttling & Health"),
        "fig_sku": ("layers", "Workload Breakdown"),
    }

    # Map chart fig names to anchor IDs for TOC navigation
    _section_ids = {
        'fig_util_gauges': 'sec-util-gauges',
        'fig_daily': 'sec-daily',
        'fig_weekday_weekend': 'sec-weekday-chart',
        'fig_sku': 'sec-sku-comparison',
        'fig_items': 'sec-top-workloads',
    }
    # Map group dividers to anchor IDs
    _group_ids = {
        'fig_util_gauges': 'sec-consumption',
    }

    first_chart = True
    for fig_name, title in chart_order:
        # Insert group divider if this chart starts a new section
        if fig_name in _group_breaks and charts.get(fig_name) is not None:
            _gicon, _glabel = _group_breaks[fig_name]
            _gid = _group_ids.get(fig_name, '')
            _gid_attr = f' id="{_gid}"' if _gid else ''
            html_parts.append(
                f'<div class="group-divider"{_gid_attr}>{_report_icon(_gicon, "#9a3412")}'
                f'<span class="group-label">{_glabel}</span></div>'
            )
        fig = charts.get(fig_name)
        if fig is not None and hasattr(fig, "to_html"):
            plotly_js = "cdn" if first_chart else False
            _sec_id = _section_ids.get(fig_name, '')
            _id_attr = f' id="{_sec_id}"' if _sec_id else ''
            _subtitle = generate_chart_subtitle(
                fig_name, metrics, recommended_sku, current_sku,
                tiered_recs=tiered_recs, sku_analysis=sku_analysis
            )
            _sub_html = f'<p class="section-subtitle">{_subtitle}</p>' if _subtitle else ''
            html_parts.append(
                f'<div class="section"{_id_attr}><div class="section-head"><h2>{title}</h2></div>'
                f'{_sub_html}'
                f'<div class="chart-wrap">'
            )
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs=plotly_js))
            html_parts.append("</div></div>")
            first_chart = False

    # --- Cost & Recommendations group divider ---
    html_parts.append(
        f'<div class="group-divider" id="sec-cost-recs">{_report_icon("dollar", "#9a3412")}'
        f'<span class="group-label">Cost & Recommendations</span></div>'
    )

    # --- Reserved vs PAYG cost table ---
    avg_util_decimal = metrics['avg_util'] / 100.0
    if avg_util_decimal > RESERVED_BREAKEVEN_UTIL:
        _payg_advice = (
            f'<div class="note-box info">'
            f'<strong>Recommendation:</strong> Your average utilisation ({metrics["avg_util"]:.0f}%) is above the '
            f'~{RESERVED_BREAKEVEN_UTIL*100:.0f}% break-even point. A <strong>reserved instance</strong> would likely '
            f'save ~${rec_saving_monthly:,}/mo vs PAYG for {recommended_sku["name"]}.</div>'
        )
    else:
        _payg_advice = (
            f'<div class="note-box">'
            f'<strong>Recommendation:</strong> Your average utilisation ({metrics["avg_util"]:.0f}%) is below the '
            f'~{RESERVED_BREAKEVEN_UTIL*100:.0f}% break-even point. <strong>PAYG with pause/resume scheduling</strong> '
            f'may be more cost-effective than a reserved instance.</div>'
        )

    _cost_rows = ""
    # Find indices for current and recommended SKUs to dim distant rows
    _sku_names = [sa['SKU'] for sa in sku_analysis]
    _cur_idx = _sku_names.index(current_sku['name']) if current_sku and current_sku['name'] in _sku_names else -1
    _rec_idx = _sku_names.index(recommended_sku['name']) if recommended_sku and recommended_sku['name'] in _sku_names else -1
    for _si, sa in enumerate(sku_analysis):
        if sa['SKU'] == recommended_sku['name']:
            _row_cls = ' class="highlight"'
        elif sa['SKU'] == current_sku['name']:
            _row_cls = ' class="highlight"'
        elif (_cur_idx >= 0 and abs(_si - _cur_idx) > 2) and (_rec_idx >= 0 and abs(_si - _rec_idx) > 2):
            _row_cls = ' class="dim-row"'
        else:
            _row_cls = ''
        _cost_rows += (
            f'<tr{_row_cls}><td>{sa["SKU"]}</td><td class="mono text-right">{sa["Avg Util %"]}%</td>'
            f'<td class="mono text-right">${sa["PAYG $/mo"]:,}</td>'
            f'<td class="mono text-right">${sa["Reserved $/mo"]:,}</td>'
            f'<td class="mono text-right">${sa["Savings $/mo"]:,}</td></tr>'
        )

    _cost_subtitle = generate_chart_subtitle(
        'fig_cost', metrics, recommended_sku, current_sku,
        tiered_recs=tiered_recs, sku_analysis=sku_analysis
    )
    _cost_sub_html = f'<p class="section-subtitle">{_cost_subtitle}</p>' if _cost_subtitle else ''
    html_parts.append(
        f'<div class="section" id="sec-cost"><div class="section-head"><h2>Reserved vs PAYG Cost Comparison</h2>'
        f'<button class="help-btn" onclick="showHelp(\'reserved\')" title="Help">?</button></div>'
        f'{_cost_sub_html}'
        f'{_payg_advice}'
        f'<table><thead><tr><th>SKU</th><th class="text-right">Avg Util %</th>'
        f'<th class="text-right">PAYG $/mo</th><th class="text-right">Reserved $/mo</th>'
        f'<th class="text-right">Savings $/mo</th></tr></thead>'
        f'<tbody>{_cost_rows}</tbody></table>'
        f'<p style="font-size:13px;color:var(--ink-muted);margin-top:12px;">Published list prices (USD). '
        f'Reserved = 1-year commitment. Actual costs vary by region, currency, and agreement. '
        f'Break-even at ~{RESERVED_BREAKEVEN_UTIL*100:.0f}% utilisation.</p></div>'
    )

    # --- Recommendations ---
    rec_html = '<div class="section" id="sec-recommendations"><div class="section-head"><h2>Recommendations</h2>'
    rec_html += '<button class="help-btn" onclick="showHelp(\'rec\')" title="Help">?</button></div>'

    if tiered_recs and tiered_recs.get('urgency', 'NONE') != 'NONE':
        # Urgency banner
        urgency = tiered_recs['urgency'].lower()
        urgency_labels = {
            'critical': '\u26a0\ufe0f IMMEDIATE ACTION REQUIRED',
            'high': '\u26a0\ufe0f ACTION RECOMMENDED',
            'moderate': '\u26a0 ATTENTION \u2014 CAPACITY UNDER STRESS',
            'low': '\u2139\ufe0f MONITOR \u2014 GROWING TREND DETECTED',
        }
        rec_html += (
            f'<div class="urgency-banner {urgency}">'
            f'{urgency_labels.get(urgency, urgency.upper())}</div>'
        )

        # Option cards
        rec_html += '<div class="option-grid">'

        opt_a = tiered_recs.get('option_a')
        if opt_a:
            _act = opt_a.get('action', 'STAY')
            _tag_css = _act.lower().replace(' ', '-')
            rec_html += f'<div class="option-card"><div class="tag {_tag_css}">{_act}</div>'
            if _act == 'SCALE UP':
                rec_html += (
                    f'<h3>Scale up to {opt_a["to_sku"]}</h3>'
                    f'<p>{opt_a["impact"]}</p>'
                    f'<div class="cost-note">${opt_a["monthly_cost"]:,}/mo '
                    f'(+${opt_a["monthly_cost_delta"]:,}) · '
                    f'Est. util: {opt_a["expected_util"]}%</div>'
                )
            elif _act == 'DOWNSIZE':
                rec_html += (
                    f'<h3>Downsize to {opt_a["to_sku"]}</h3>'
                    f'<p>{opt_a["impact"]}</p>'
                    f'<div class="cost-note">Saves ${opt_a["monthly_savings"]:,}/mo · '
                    f'Est. util: {opt_a["expected_util"]}%</div>'
                )
            elif _act == 'STAY':
                rec_html += (
                    f'<h3>Stay on {opt_a["sku"]}</h3>'
                    f'<p>{opt_a["impact"]}</p>'
                    f'<div class="cost-note">${opt_a["monthly_cost"]:,}/mo · '
                    f'Util: {opt_a["expected_util"]}%</div>'
                )
            rec_html += '</div>'

        opt_b = tiered_recs.get('option_b')
        if opt_b:
            rec_html += (
                '<div class="option-card">'
                '<div class="tag optimise">OPTIMISE</div>'
                '<h3>Reschedule workloads</h3>'
                f'<p>{opt_b["impact"]}</p>'
            )
            if opt_b.get('has_peak_data'):
                rec_html += f'<div class="cost-note">Peak: {opt_b["peak_hours"]} · Off-peak: {opt_b["off_peak_hours"]}</div>'
            rec_html += '<div class="cost-note" style="font-style:italic">Estimated impact: Could reduce peak utilisation by ~30–50% through staggered scheduling. Actual results depend on workload characteristics.</div>'
            rec_html += '</div>'

        opt_c = tiered_recs.get('option_c')
        if opt_c:
            rec_html += (
                '<div class="option-card">'
                '<div class="tag combined">COMBINED</div>'
                '<h3>Scale + Optimise</h3>'
                f'<p>{opt_c["impact"]}</p>'
                '</div>'
            )

        opt_d = tiered_recs.get('option_d')
        if opt_d:
            _recommended_tag = ' style="border:2px solid var(--success)"' if tiered_recs.get('_scheduling_issue') else ''
            rec_html += (
                f'<div class="option-card"{_recommended_tag}>'
                '<div class="tag" style="background:var(--success);color:#fff">OPTIMISE ONLY</div>'
                f'<h3>Stay on {opt_d["sku"]} — reschedule workloads</h3>'
                f'<p>{opt_d["impact"]}</p>'
                f'<div class="cost-note">$0/mo change · Est. smoothed util: {opt_d["expected_util"]}%</div>'
            )
            if tiered_recs.get('_scheduling_issue'):
                rec_html += '<div style="margin-top:8px;font-size:12px;font-weight:600;color:var(--success)">&#10004; Recommended — scheduling is the primary issue</div>'
            rec_html += '</div>'

        rec_html += '</div>'  # end option-grid

        # Top consumers table
        sched = tiered_recs.get('scheduling_advice', [])
        if sched:
            rec_html += '<h3 style="font-size:1.05em;font-weight:600;color:var(--ink-light);margin:24px 0 12px;">Top Consumers</h3>'
            if opt_b and opt_b.get('has_peak_data'):
                rec_html += (
                    '<div class="peak-summary">'
                    f'<span class="peak-badge peak">PEAK {opt_b["peak_hours"]}</span>'
                    f'<span class="peak-badge offpeak">OFF-PEAK {opt_b["off_peak_hours"]}</span>'
                    '</div>'
                )
            _has_budget = sched[0].get('budget_pct') is not None
            _has_ops = sched[0].get('total_ops') is not None
            _has_rej = any(s.get('rejected_ops') is not None for s in sched)
            rec_html += '<table><thead><tr><th>Item</th><th>Type</th><th class="text-right">CUs/Day</th>'
            if _has_budget:
                rec_html += '<th class="text-right">% of Total</th>'
            if _has_ops:
                rec_html += '<th class="text-right">Ops</th>'
            if _has_rej:
                rec_html += '<th class="text-right">Rejected</th>'
            rec_html += '<th>Action</th></tr></thead><tbody>'
            for s in sched:
                rec_html += f'<tr><td><strong>{s["item_name"]}</strong></td>'
                rec_html += f'<td>{s["item_type"]}</td>'
                rec_html += f'<td class="mono text-right">{s["daily_cus"]:,.0f}</td>'
                if _has_budget:
                    _bp = f'{s["budget_pct"]}%' if s.get("budget_pct") is not None else '\u2014'
                    rec_html += f'<td class="mono text-right">{_bp}</td>'
                if _has_ops:
                    _ops = f'~{s["daily_ops"]}/day' if s.get('daily_ops') else (f'{s["total_ops"]} total' if s.get('total_ops') else '\u2014')
                    rec_html += f'<td class="mono text-right">{_ops}</td>'
                if _has_rej:
                    _r = s.get('rejected_ops')
                    _rs = f'{_r} \u26a0\ufe0f' if _r and _r > 0 else ('0' if _r is not None else '\u2014')
                    rec_html += f'<td class="mono text-right">{_rs}</td>'
                rec_html += f'<td style="font-size:13px;">{s["action"]}</td></tr>'
            rec_html += '</tbody></table>'
            rec_html += '<p style="font-size:12px;color:var(--ink-muted);margin-top:8px;font-style:italic;">* Item-level timing not available via Capacity Metrics. Check refresh schedules in workspace item settings.</p>'

        # Alerts: growth warning + notes
        _alerts = []
        gw = tiered_recs.get('growth_warning')
        if gw:
            _alerts.append(('danger', f'\u26a0\ufe0f {gw["forecast"]}'))
        for note_type, note_text in tiered_recs.get('notes', []):
            _css = 'danger' if note_type == 'veto' else 'info'
            _ico = '\u26d4' if note_type == 'veto' else '\u2139\ufe0f'
            _alerts.append((_css, f'{_ico} {note_text}'))
        if _alerts:
            rec_html += '<div class="alert-stack">'
            for _css, _txt in _alerts:
                rec_html += f'<div class="alert-item {_css}">{_txt}</div>'
            rec_html += '</div>'

    else:
        # Fallback: simple recommendation when no tiered data
        _h = metrics['health_rating']
        _rn = recommended_sku['name']
        if _h in ('CRITICAL', 'POOR'):
            rec_html += (
                f'<div class="alert-item danger"><strong>Upgrade to {_rn}</strong></div>'
            )
        elif _h == 'FAIR':
            rec_html += (
                f'<div class="alert-item warning"><strong>Consider {_rn}</strong></div>'
            )
        else:
            rec_html += f'<div class="alert-item info"><strong>Well-sized on {_rn}</strong></div>'

    rec_html += '<p class="spark-note"><strong>Spark Autoscale:</strong> If enabled, Spark workloads are billed separately (pay-as-you-go) and not reflected in these metrics.</p>'

    # Surge protection caveat (when throttling is active)
    if metrics.get('days_with_delay', 0) > 0:
        rec_html += (
            '<div class="note-box">'
            '<strong>Note:</strong> These recommendations assume current Fabric throttling behaviour. '
            "Microsoft's planned <em>Surge Protection</em> feature may change how overages are handled. "
            'Monitor Fabric announcements for updates.</div>'
        )

    rec_html += '</div>'
    html_parts.append(rec_html)

    # --- Data source note ---
    html_parts.append(
        f'<div class="section" id="sec-data-source"><div class="section-head"><h2>Data Source</h2></div>'
        f'<div class="note-box info">This analysis is based on <strong>{len(df_raw)}</strong> individual workload records '
        f'across <strong>{metrics["days_analyzed"]}</strong> days of Fabric capacity usage data (CSV). '
        f'Note: CSV data does not include throttling or carryforward metrics available in the Capacity Metrics '
        f'semantic model. For a more complete analysis, use the Fabric SKU Advisor notebook with a live '
        f'semantic model connection.</div></div>'
    )

    # Close container
    html_parts.append('</div><!-- end container -->')

    # --- Floating TOC (button + slide-out panel) ---
    html_parts.append('''<button class="toc-fab" id="toc-fab" aria-label="Table of Contents">&#9776;</button>
<nav class="toc-panel" id="toc-panel">
<div class="toc-panel-head">Navigate</div>
<ul>
<li><a href="#sec-kpi">Key Metrics</a></li>
<li><a href="#sec-intelligence">Capacity Intelligence</a></li>
<li><a href="#sec-consumption">Consumption Analysis</a></li>
<li><a href="#sec-util-gauges" class="toc-indent">Utilisation Gauges</a></li>
<li><a href="#sec-daily" class="toc-indent">Daily Utilisation</a></li>
<li><a href="#sec-weekday-chart" class="toc-indent">Weekday vs Weekend</a></li>
<li><a href="#sec-sku-comparison" class="toc-indent">SKU Comparison</a></li>
<li><a href="#sec-top-workloads" class="toc-indent">Top Workloads</a></li>
<li><a href="#sec-cost-recs">Cost &amp; Recommendations</a></li>
<li><a href="#sec-cost" class="toc-indent">Cost Comparison</a></li>
<li><a href="#sec-recommendations" class="toc-indent">Recommendations</a></li>
<li><a href="#sec-data-source">Data Source</a></li>
</ul>
</nav>''')

    # --- Footer bar (mirrors header) ---
    html_parts.append(
        '<div class="footer-bar">'
        '<div class="footer-top">'
        '<div class="footer-brand">'
        '<span class="brand-nova">Data Nova</span> — Fabric SKU Advisor | Prathy Kamasani'
        '</div>'
        '<a href="https://www.data-nova.io">data-nova.io</a>'
        '</div>'
        f'<div class="footer-disclaimer">{DISCLAIMER_TEXT}</div>'
        '</div>'
    )

    # --- Help modals ---
    html_parts.append("""
<div class="modal-overlay" id="modal-kpi" onclick="closeModal(event,'kpi')">
  <div class="modal" onclick="event.stopPropagation()">
    <button class="modal-close" onclick="closeModal(event,'kpi')">&times;</button>
    <h4>Key Metrics</h4>
    <p>These KPIs summarise your capacity health at a glance.</p>
    <ul>
      <li><strong>Avg Utilisation</strong> &mdash; average daily smoothed CU consumption as a percentage of SKU budget</li>
      <li><strong>P80 Daily CUs</strong> &mdash; 80th-percentile daily consumption (the &ldquo;typical busy day&rdquo;)</li>
      <li><strong>Peak Utilisation</strong> &mdash; highest single-day utilisation observed</li>
      <li><strong>Burst Ratio</strong> &mdash; peak-to-average ratio. Higher values indicate spiky workloads</li>
    </ul>
  </div>
</div>

<div class="modal-overlay" id="modal-intel" onclick="closeModal(event,'intel')">
  <div class="modal" onclick="event.stopPropagation()">
    <button class="modal-close" onclick="closeModal(event,'intel')">&times;</button>
    <h4>Capacity Intelligence</h4>
    <p>These insights are generated from your data using pattern recognition &mdash; no AI involved. Each card identifies a specific pattern in your capacity metrics and suggests an action.</p>
    <ul>
      <li><strong>Warning</strong> cards indicate issues that need attention</li>
      <li><strong>Info</strong> cards highlight useful observations</li>
      <li><strong>Success</strong> cards confirm healthy patterns</li>
    </ul>
  </div>
</div>

<div class="modal-overlay" id="modal-consumption" onclick="closeModal(event,'consumption')">
  <div class="modal" onclick="event.stopPropagation()">
    <button class="modal-close" onclick="closeModal(event,'consumption')">&times;</button>
    <h4>Consumption Analysis</h4>
    <p>Charts and statistics showing how your capacity is consumed over time.</p>
    <ul>
      <li><strong>Daily utilisation</strong> shows the trend over the analysis period</li>
      <li><strong>Weekday vs weekend</strong> helps identify scheduling opportunities</li>
      <li><strong>Spike filtering</strong> excludes anomalous days (e.g. pause/resume settlement) from sizing</li>
    </ul>
  </div>
</div>

<div class="modal-overlay" id="modal-workload" onclick="closeModal(event,'workload')">
  <div class="modal" onclick="event.stopPropagation()">
    <button class="modal-close" onclick="closeModal(event,'workload')">&times;</button>
    <h4>Workload Breakdown</h4>
    <p>Analysis of which items and types consume the most capacity.</p>
    <ul>
      <li><strong>Top consumers</strong> identifies the items driving CU usage</li>
      <li><strong>Item type distribution</strong> shows the workload mix</li>
      <li><strong>Hourly patterns</strong> reveal scheduling opportunities</li>
    </ul>
  </div>
</div>

<div class="modal-overlay" id="modal-weekday" onclick="closeModal(event,'weekday')">
  <div class="modal" onclick="event.stopPropagation()">
    <button class="modal-close" onclick="closeModal(event,'weekday')">&times;</button>
    <h4>Weekday vs Weekend</h4>
    <p>Compares consumption patterns between weekdays and weekends to identify scheduling opportunities.</p>
    <ul>
      <li><strong>Ratio &gt; 1.5x</strong> means weekday-heavy workloads &mdash; typical for business reporting</li>
      <li><strong>Ratio near 1.0</strong> means similar load 7 days/week &mdash; consider whether weekend refreshes are needed</li>
    </ul>
  </div>
</div>

<div class="modal-overlay" id="modal-spike" onclick="closeModal(event,'spike')">
  <div class="modal" onclick="event.stopPropagation()">
    <button class="modal-close" onclick="closeModal(event,'spike')">&times;</button>
    <h4>Spike Filtering</h4>
    <p>Days with abnormally high consumption (e.g. pause/resume settlement spikes, one-off migrations) are detected and excluded from the P80 sizing calculation.</p>
    <p>This prevents anomalous days from inflating your SKU recommendation.</p>
  </div>
</div>

<div class="modal-overlay" id="modal-reserved" onclick="closeModal(event,'reserved')">
  <div class="modal" onclick="event.stopPropagation()">
    <button class="modal-close" onclick="closeModal(event,'reserved')">&times;</button>
    <h4>Reserved vs PAYG Pricing</h4>
    <p>Fabric capacity can be purchased on a pay-as-you-go (hourly) basis or with a 1-year reservation for significant savings.</p>
    <ul>
      <li><strong>PAYG</strong> &mdash; flexible, no commitment, higher per-hour cost</li>
      <li><strong>Reserved</strong> &mdash; 1-year commitment, typically 40-50% cheaper</li>
    </ul>
    <p>Prices shown are approximate USD monthly costs based on published Azure pricing.</p>
  </div>
</div>

<div class="modal-overlay" id="modal-rec" onclick="closeModal(event,'rec')">
  <div class="modal" onclick="event.stopPropagation()">
    <button class="modal-close" onclick="closeModal(event,'rec')">&times;</button>
    <h4>Recommendations</h4>
    <p>Three options are presented to give you flexibility:</p>
    <ul>
      <li><strong>Option A (Scale)</strong> &mdash; change SKU size to match demand</li>
      <li><strong>Option B (Optimise)</strong> &mdash; reschedule workloads to reduce peak contention</li>
      <li><strong>Option C (Combined)</strong> &mdash; moderate scale change plus scheduling improvements</li>
    </ul>
    <p>The top consumers table shows which items to prioritise for scheduling changes. Urgency level is based on health score and throttling severity.</p>
  </div>
</div>

<div class="modal-overlay" id="modal-cost" onclick="closeModal(event,'cost')">
  <div class="modal" onclick="event.stopPropagation()">
    <button class="modal-close" onclick="closeModal(event,'cost')">&times;</button>
    <h4>Cost &amp; Recommendations</h4>
    <p>SKU sizing uses the <strong>80/80 method</strong>: P80 daily consumption at 80% target utilisation, leaving 20% headroom.</p>
    <ul>
      <li><strong>Option A</strong> &mdash; pure scaling (change SKU)</li>
      <li><strong>Option B</strong> &mdash; optimise scheduling (move workloads to off-peak)</li>
      <li><strong>Option C</strong> &mdash; combined approach</li>
    </ul>
    <p>Reserved pricing assumes a 1-year commitment. PAYG = pay-as-you-go (no commitment).</p>
  </div>
</div>
""")

    # --- JavaScript ---
    html_parts.append("""
<script>
function showHelp(id) {
  var el = document.getElementById('modal-' + id);
  if (el) el.classList.add('active');
}
function closeModal(e, id) {
  var el = document.getElementById('modal-' + id);
  if (el) el.classList.remove('active');
}
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') {
    document.querySelectorAll('.modal-overlay.active').forEach(function(m) {
      m.classList.remove('active');
    });
    // Close floating TOC on Escape
    var p=document.getElementById('toc-panel'),b=document.getElementById('toc-fab');
    if(p&&p.classList.contains('open')){p.classList.remove('open');b.classList.remove('active');b.innerHTML='&#9776;';}
  }
});
// --- Floating TOC ---
(function(){
  var fab=document.getElementById('toc-fab'),panel=document.getElementById('toc-panel');
  if(!fab||!panel)return;
  fab.addEventListener('click',function(e){
    e.stopPropagation();
    var open=panel.classList.toggle('open');
    fab.classList.toggle('active',open);
    fab.innerHTML=open?'&times;':'&#9776;';
  });
  panel.querySelectorAll('a').forEach(function(a){
    a.addEventListener('click',function(){
      panel.classList.remove('open');fab.classList.remove('active');fab.innerHTML='&#9776;';
    });
  });
  document.addEventListener('click',function(e){
    if(!panel.contains(e.target)&&e.target!==fab){
      panel.classList.remove('open');fab.classList.remove('active');fab.innerHTML='&#9776;';
    }
  });
  var links=panel.querySelectorAll('a[href^="#sec-"]');
  var linkMap={};
  links.forEach(function(a){linkMap[a.getAttribute('href').slice(1)]=a;});
  var observer=new IntersectionObserver(function(entries){
    entries.forEach(function(entry){
      if(entry.isIntersecting){
        links.forEach(function(a){a.classList.remove('toc-active');});
        var a=linkMap[entry.target.id];
        if(a)a.classList.add('toc-active');
      }
    });
  },{rootMargin:'-20% 0px -70% 0px',threshold:0});
  document.querySelectorAll('[id^="sec-"]').forEach(function(el){observer.observe(el);});
})();
</script>
</body></html>
""")

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
    print(f"Days Analysed: {metrics['days_analyzed']}")

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

        # Resolve current SKU FIRST (needed by recommend_sku_for_capacity for veto logic)
        current_sku = None
        if args.current_sku:
            current_sku = next((s for s in SKUS if s['name'].upper() == args.current_sku.upper()), None)
            if not current_sku:
                print(f"WARNING: SKU '{args.current_sku}' not found. Ignoring.")

        # Calculate metrics
        print("Calculating metrics...")
        metrics = calculate_capacity_metrics(daily_summary)

        # Inject current_sku into metrics so recommend_sku_for_capacity can use it
        if current_sku:
            metrics['current_sku'] = current_sku

        # SKU recommendation (v2.0: includes throttling veto + demand adjustment)
        print("Analysing SKU recommendations...")
        sku_analysis, recommended_sku = recommend_sku_for_capacity(
            metrics, SKUS,
            needs_free_viewers=args.needs_free_viewers,
            weekday_split=not args.no_weekday_split,
            spike_filter=not args.no_spike_filter,
        )

        # Update utilisation metrics against recommended SKU
        metrics = update_utilisation_metrics(metrics, recommended_sku, daily_summary=daily_summary)

        # Generate tiered recommendations (v2.0)
        tiered_recs = generate_tiered_recommendations(
            metrics, current_sku, recommended_sku,
            sku_analysis, SKUS, peak_info=None, df_raw=df_raw,
        )

        # Generate capacity intelligence (v2.0)
        intelligence = generate_capacity_intelligence(
            metrics, current_sku, recommended_sku,
            tiered_recs.get('scheduling_advice', []),
            tiered_recs, sku_analysis,
        )
        print(f"Generated {len(intelligence)} capacity intelligence insights")

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
                charts, daily_summary, args.capacity_name, df_raw,
                tiered_recs=tiered_recs, intelligence=intelligence,
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
