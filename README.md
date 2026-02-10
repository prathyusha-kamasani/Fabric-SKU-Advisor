# Fabric SKU Advisor

A Microsoft Fabric notebook that analyses your capacity consumption and recommends the right SKU.

It connects to the [Capacity Metrics](https://learn.microsoft.com/en-us/fabric/enterprise/metrics-app) semantic model via Semantic Link, pulls your actual CU data, and generates an interactive HTML report with sizing recommendations, health scores, and cost comparisons.

![Fabric SKU Advisor](https://www.data-nova.io/post/fabric-sku-advisor-sizing-tool)

## What it does

- Recommends a Fabric SKU using the **80/80 approach**: 80th percentile of weekday CU consumption at 80% target utilisation
- Filters out pause/resume settlement spikes that would skew the recommendation
- Splits weekday vs weekend consumption (weekends don't drive the sizing)
- Calculates a **health score** (0-100) based on utilisation, throttling, and carryforward
- Runs **trend analysis** with weekly growth rate and direction
- Compares **reserved vs PAYG pricing** with break-even analysis
- Shows **workspace-level** and **item-level** CU breakdowns
- Generates a self-contained **HTML report** you can share with anyone
- Supports **single capacity** (deep dive) or **multi capacity** (analyse your whole estate)

## Quick start

1. Import the notebook into a Fabric workspace
2. Set your `WORKSPACE_ID` and `DATASET_ID` in the configuration cell (these point to your Capacity Metrics semantic model)
3. Set `ANALYSIS_MODE` to `"single"` or `"multi"`
4. If single mode, provide your `CAPACITY_ID`
5. Run all cells

That's it. No pip installs, no service principals, no REST APIs. It uses `sempy.fabric` which is pre-installed in Fabric notebooks.

## Configuration

```python
WORKSPACE_ID   = "your-workspace-guid"
DATASET_ID     = "your-dataset-guid"
ANALYSIS_MODE  = "single"       # "single" or "multi"
CAPACITY_ID    = "your-cap-id"  # required for single mode

DAYS_TO_ANALYZE        = 14
NEEDS_FREE_VIEWERS     = False
WEEKDAY_WEEKEND_SPLIT  = True
TREND_ANALYSIS         = True
RESERVED_VS_PAYG       = True
SPIKE_FILTERING        = True
SAVE_TO_LAKEHOUSE      = True
```

## Requirements

- A Fabric or Power BI Premium capacity
- The [Capacity Metrics app](https://learn.microsoft.com/en-us/fabric/enterprise/metrics-app) installed in your tenant
- Read/Build permissions on the Capacity Metrics semantic model
- XMLA endpoint enabled (Capacity Settings > Power BI Workloads)

## How it works

The full algorithm breakdown, design decisions, and what I learned building this are covered in the blog post:

**[I Built a Tool to Size Fabric SKUs](https://www.data-nova.io/post/fabric-sku-advisor-sizing-tool)**

## Version compatibility

The notebook auto-detects which version of the Capacity Metrics semantic model you're running (v37, v40, or v47+) and adjusts its DAX queries accordingly. Version detection approach adapted from [FUAM](https://github.com/microsoft/fabric-toolbox/tree/main/monitoring/fabric-unified-admin-monitoring).

## Disclaimer

This tool provides advisory recommendations based on historical consumption data. Actual sizing decisions should consider factors the tool cannot see: future growth plans, licensing strategy, regulatory requirements, and workload seasonality beyond the analysis window. Pricing is based on published list prices and may vary by region and agreement.

## Author

**Prathy Kamasani** | Microsoft MVP | [Data Nova](https://www.data-nova.io)

Built as part of an upcoming Fabric capacity course. Feedback, issues, and PRs welcome.
