# Walmart Stockout Prediction & Inventory Risk Management

**Team:** Yash Nirwan · Soham Nevgi · Rahul Nair · Tanvish Tuplondhe

---

## Overview

Over half of retail transactions result in stockout events — not because demand is unpredictable, but because existing replenishment logic fails to detect risk signals in time. This project builds a full ML + optimization pipeline to identify high-risk store-product combinations **before** stockouts occur, then allocates additional inventory within a fixed budget using Gurobi linear programming.

**Scope:** 5,000 transactions · 5 stores · 8 products · 47 engineered features

---

## Problem Statement

The original `stockout_indicator` column was near-random (AUC ≈ 0.50) — a label quality failure, not a model failure. We rebuilt the target logically to reflect true operational failure:

| Condition | Logic |
|---|---|
| Demand Spike | Actual demand exceeds available inventory |
| Supply Delay | Inventory buffer < 0 AND supplier lead time > 5 days |
| Compounding Risk | Demand deviation > 0 AND inventory buffer < 0 |

After target reconstruction and **3 rounds of leakage audits** (dropping 12 features derived from the new target), the dataset was training-ready.

---

## My Contribution — EDA & Visualization Lead

- Identified **New York, NY** as the highest-risk city (55.4% stockout rate) and **Tablets** as the highest-risk product (54.4%)
- Surfaced the **counterintuitive weather effect**: rainy and cloudy conditions spike indoor demand beyond inventory buffers, outpacing sunny days by ~5pp
- Discovered the **Saturday peak problem**: 54.2% stockout rate on Saturdays vs. mid-week lows — replenishment schedules were misaligned with traffic patterns
- Showed that the **buffer illusion**: avg buffer values for stockout vs. non-stockout events were nearly identical (151 vs. 155), invalidating threshold-based rules

---

## Feature Engineering

Four core features engineered to capture operational failure modes:

| Feature | Description |
|---|---|
| `demand_deviation` | Actual vs. forecasted demand gap — captures surprise spikes |
| `inventory_buffer` | Stock minus reorder point — true margin signal |
| `lead_time_risk` | Supplier delays × low-buffer interaction — compounding failure mode |
| `demand_vs_reorder_qty` | When demand outpaces standard order sizes, stockouts become structural |

Correlation heatmap confirmed near-zero linear correlations, validating the choice of tree-based models.

---

## Modeling

**Evaluation strategy:** 80/20 stratified split, optimizing for Recall and F1-Score — missing a true stockout costs more than triggering early replenishment.

| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.7714 | 0.7943 | 69.62% | 0.7420 |
| Random Forest | 0.7686 | 0.7853 | 69.62% | 0.7381 |
| Gradient Boosting | 0.7607 | 0.7542 | 74.79% | 0.7510 |
| **XGBoost (Tuned)** | **0.7714** | **0.7625** | **76.63%** | **0.7644** |

**XGBoost selected** — highest Recall (76.6%) while tying on AUC. Identifies the majority of true stockout events at 76.3% precision.

### Top Feature Importances (XGBoost)

1. **Actual Demand** — dominant signal; stores with demand spikes above historic range are prime targets
2. **Demand vs. Reorder Qty** — when standard order sizes structurally fail to match pulling volume
3. **Supplier Lead Time** — high demand × 7+ day lead time amplifies risk dramatically
4. **Demand × Lead Time** — interaction feature; highest combined risk signal

---

## Gurobi Inventory Optimization

After XGBoost scoring, risk scores feed a budget-constrained linear optimization:

```
MINIMIZE: Σ [Risk × (1/BaseQty) × (−ExtraUnits)]
SUBJECT TO: Σ (ExtraUnits × HoldingCost) ≤ $32,836
```

**Results:** 1,685 units allocated across 11 prioritized store-product combinations

| Location | Product | Risk Score | Reorder Increase |
|---|---|---|---|
| Chicago, IL | Laptop | 0.64 | +76.1% |
| Dallas, TX | Fridge | 0.63 | +72.3% |
| Miami, FL | Camera | 0.63 | +76.6% |
| New York, NY | Tablet | 0.61 | +74.6% |
| Los Angeles, CA | Washing Machine | 0.60 | +68.9% |

---

## Strategic Recommendations

1. **NY Safety Buffers** — Increase safety stock by 15% for NY stores on high-risk items (Tablets/Headphones)
2. **Lead Time Tiers** — Apply 1.25× reorder multiplier for 7–8 day suppliers; 1.5× for 9–10 day suppliers
3. **Promo Pre-Builds** — Mandate 20% inventory build in the week before any discount event
4. **Thursday Deliveries** — Shift replenishment schedules to Thu/Fri to absorb Saturday's traffic peak
5. **Deploy Weekly** — Score combinations Monday → Optimize Tuesday → Trigger orders Wednesday

---

## Tech Stack

`Python` · `XGBoost` · `scikit-learn` · `Gurobi` · `pandas` · `matplotlib` · `seaborn`

---

## Files

| File | Description |
|---|---|
| `Project_3_Walmart_Inventory_Predictions.ipynb` | Full notebook: EDA, feature engineering, modeling, Gurobi optimization |
