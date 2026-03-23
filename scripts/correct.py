"""
CRICKET BATTING PRESSURE INDEX CALCULATION (CORRECTED)

Creates interpretable pressure columns and a composite score.

Key fixes vs your version:
1) Non applicable innings columns are NaN (not 0).
2) Scaling uses robust percentile scaling to 1..10.
3) Scaling is done within the relevant innings where needed.
4) Phase mapping matches your dataset labels: Powerplay, Middle, Death.
5) balls_remaining is filled using 300 - balls_bowled when missing.
6) required_run_rate is recomputed for innings 2 if missing (from runs_remaining and balls_remaining).

Output:
cricket_with_pressure_indexes_corrected.csv
"""

import pandas as pd
import numpy as np
import re

FILE_PATH = "all_three_combined_latest.csv"
OUTPUT_PATH = "cricket_with_pressure_indexes_corrected.csv"

df = pd.read_csv(FILE_PATH)
df.columns = [c.strip() for c in df.columns]

print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")
print(df["inning"].value_counts(dropna=False).sort_index())

# ----------------------------
# Ensure numeric types
# ----------------------------
num_cols = [
    "inning", "balls_bowled", "balls_remaining", "run_rate",
    "required_run_rate", "wickets_lost", "current_score",
    "target_runs", "runs_remaining", "total_runs"
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

TOTAL_BALLS = 300.0
eps = 1.0

m1 = df["inning"].eq(1)
m2 = df["inning"].eq(2)

# ----------------------------
# Helper: robust scaling to 1..10
# Uses percentile clipping to avoid outliers dominating the scale
# ----------------------------
def scale_to_1_10(s: pd.Series, p_low=0.10, p_high=0.90) -> pd.Series:
    x = s.astype(float)
    lo = np.nanquantile(x, p_low)
    hi = np.nanquantile(x, p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.nan, index=s.index)
    z = (x - lo) / (hi - lo)
    z = np.clip(z, 0, 1)
    return 1 + 9 * z

# ----------------------------
# Fill balls_remaining if missing
# ----------------------------
df["balls_remaining_filled"] = df["balls_remaining"]
df.loc[df["balls_remaining_filled"].isna() & df["balls_bowled"].notna(), "balls_remaining_filled"] = (
    TOTAL_BALLS - df["balls_bowled"]
)

# ----------------------------
# Fill target_runs for innings 2 if missing using first-innings totals
# target = first_innings_total + 1
# ----------------------------
if "match_id" in df.columns and "total_runs" in df.columns:
    inn1_totals = (
        df.loc[m1]
        .groupby("match_id")["total_runs"]
        .sum()
        .rename("first_innings_total")
        .reset_index()
    )
    df = df.merge(inn1_totals, on="match_id", how="left")
else:
    df["first_innings_total"] = np.nan

df["target_runs_filled"] = df.get("target_runs", pd.Series(np.nan, index=df.index))
df.loc[m2 & df["target_runs_filled"].isna() & df["first_innings_total"].notna(), "target_runs_filled"] = (
    df.loc[m2 & df["target_runs_filled"].isna() & df["first_innings_total"].notna(), "first_innings_total"] + 1
)

# ----------------------------
# Fill runs_remaining for innings 2 if missing
# runs_remaining = target_runs - current_score
# ----------------------------
df["runs_remaining_filled"] = df.get("runs_remaining", pd.Series(np.nan, index=df.index))
mask_rm = m2 & df["runs_remaining_filled"].isna() & df["target_runs_filled"].notna() & df["current_score"].notna()
df.loc[mask_rm, "runs_remaining_filled"] = df.loc[mask_rm, "target_runs_filled"] - df.loc[mask_rm, "current_score"]
df.loc[m2 & df["runs_remaining_filled"].notna(), "runs_remaining_filled"] = df.loc[
    m2 & df["runs_remaining_filled"].notna(), "runs_remaining_filled"
].clip(lower=0)

# ----------------------------
# Fill required_run_rate for innings 2 if missing
# required_rr = runs_remaining / (balls_remaining/6)
# ----------------------------
df["required_run_rate_filled"] = df.get("required_run_rate", pd.Series(np.nan, index=df.index))
den_overs = df["balls_remaining_filled"] / 6.0
mask_rr = m2 & df["required_run_rate_filled"].isna() & df["runs_remaining_filled"].notna() & den_overs.notna() & (den_overs > 0)
df.loc[mask_rr, "required_run_rate_filled"] = df.loc[mask_rr, "runs_remaining_filled"] / den_overs[mask_rr]

# ----------------------------
# 1) FIRST INNINGS stage-adjusted par curve gap
# ExpectedRR = base + k1*t + k2*t^2
# Pressure = max(0, ExpectedRR - run_rate)
# ----------------------------
BASE_RR, K1, K2 = 4.5, 1.0, 1.0
t = df["balls_bowled"] / TOTAL_BALLS
expected_rr = BASE_RR + K1 * t + K2 * (t ** 2)

raw_stage_rr = np.maximum(0, expected_rr - df["run_rate"])
df["pi_stage_rr_pressure_raw_inn1"] = np.where(m1, raw_stage_rr, np.nan)
df.loc[m1, "pi_stage_rr_pressure_1to10_inn1"] = scale_to_1_10(df.loc[m1, "pi_stage_rr_pressure_raw_inn1"])

# ----------------------------
# 2) FIRST INNINGS wicket-weighted pace
# Pressure = max(0, RR0 - run_rate + alpha*wickets_lost)
# ----------------------------
RR0, ALPHA = 5.5, 0.2
raw_ww = np.maximum(0, RR0 - df["run_rate"] + ALPHA * df["wickets_lost"])
df["pi_wicket_weighted_pace_raw_inn1"] = np.where(m1, raw_ww, np.nan)
df.loc[m1, "pi_wicket_weighted_pace_1to10_inn1"] = scale_to_1_10(df.loc[m1, "pi_wicket_weighted_pace_raw_inn1"])

# ----------------------------
# 3) FIRST INNINGS resource remaining pressure
# ResourcesLeft = (10-w)*(300-b)
# Pressure = 1/(ResourcesLeft + eps)
# ----------------------------
wickets_remaining = (10 - df["wickets_lost"]).clip(lower=0)
balls_left = (TOTAL_BALLS - df["balls_bowled"]).clip(lower=0)
resources_left = wickets_remaining * balls_left

raw_res = 1.0 / (resources_left + eps)
df["pi_resource_remaining_raw_inn1"] = np.where(m1, raw_res, np.nan)
df.loc[m1, "pi_resource_remaining_1to10_inn1"] = scale_to_1_10(df.loc[m1, "pi_resource_remaining_raw_inn1"])

# ----------------------------
# 4) SECOND INNINGS required run rate gap
# Pressure = max(0, required_run_rate - run_rate)
# ----------------------------
raw_rrr = np.maximum(0, df["required_run_rate_filled"] - df["run_rate"])
df["pi_rrr_pressure_raw_inn2"] = np.where(m2, raw_rrr, np.nan)
df.loc[m2, "pi_rrr_pressure_1to10_inn2"] = scale_to_1_10(df.loc[m2, "pi_rrr_pressure_raw_inn2"])

# ----------------------------
# 5) BOTH INNINGS wicket pressure
# wickets_lost / (balls_remaining + eps)
# ----------------------------
raw_wp = df["wickets_lost"] / (df["balls_remaining_filled"] + eps)
df["pi_wicket_pressure_raw"] = raw_wp
df["pi_wicket_pressure_1to10"] = scale_to_1_10(df["pi_wicket_pressure_raw"])

# ----------------------------
# 6) BOTH INNINGS match phase pressure
# Your phase values: Powerplay, Middle, Death
# ----------------------------
PHASE_WEIGHTS = {"Powerplay": 3.0, "Middle": 6.0, "Death": 9.0}
df["pi_match_phase_raw"] = df["phase"].map(PHASE_WEIGHTS).astype(float)
df["pi_match_phase_1to10"] = scale_to_1_10(df["pi_match_phase_raw"])

# ----------------------------
# 7) Composite pressure per ball (1..10)
# Innings 1 uses 5 components
# Innings 2 uses 3 components
# Use mean so it remains interpretable and robust to missing
# ----------------------------
inn1_components = [
    "pi_stage_rr_pressure_1to10_inn1",
    "pi_wicket_weighted_pace_1to10_inn1",
    "pi_resource_remaining_1to10_inn1",
    "pi_wicket_pressure_1to10",
    "pi_match_phase_1to10",
]
inn2_components = [
    "pi_rrr_pressure_1to10_inn2",
    "pi_wicket_pressure_1to10",
    "pi_match_phase_1to10",
]

df["pi_composite_1to10"] = np.nan
df.loc[m1, "pi_composite_1to10"] = df.loc[m1, inn1_components].mean(axis=1)
df.loc[m2, "pi_composite_1to10"] = df.loc[m2, inn2_components].mean(axis=1)

# ----------------------------
# Quick validation prints
# ----------------------------
print("\nNon null counts")
for c in ["pi_stage_rr_pressure_1to10_inn1", "pi_wicket_weighted_pace_1to10_inn1", "pi_resource_remaining_1to10_inn1",
          "pi_rrr_pressure_1to10_inn2", "pi_wicket_pressure_1to10", "pi_match_phase_1to10", "pi_composite_1to10"]:
    print(c, int(df[c].notna().sum()))

print("\nComposite mean by phase")
print(df.groupby("phase")["pi_composite_1to10"].mean().round(2))

print("\nComposite mean by innings")
print(df.groupby("inning")["pi_composite_1to10"].mean().round(2))

# ----------------------------
# Save
# ----------------------------
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")