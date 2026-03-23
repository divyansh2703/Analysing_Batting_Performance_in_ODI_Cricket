import pandas as pd
import numpy as np
import json

INPUT_PATH = "cricket_with_pressure_indexes_corrected.csv"
PARAMS_JSON = "inn1_par_curve_params.json"
OUTPUT_PATH = "cricket_with_pressure_indexes_corrected_learnedcurve.csv"

df = pd.read_csv(INPUT_PATH)
df.columns = [c.strip() for c in df.columns]

for c in ["inning", "balls_bowled", "run_rate", "wickets_lost", "balls_remaining_filled"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

with open(PARAMS_JSON, "r", encoding="utf-8") as f:
    params = json.load(f)

base = params["base"]
k1 = params["k1"]
k2 = params["k2"]
TOTAL_BALLS = params.get("total_balls", 300.0)

m1 = df["inning"].eq(1)

# recompute expected RR using learned curve
t = df["balls_bowled"] / TOTAL_BALLS
expected_rr_learned = base + k1 * t + k2 * (t ** 2)

# raw pace pressure in innings 1
raw_gap = np.maximum(0, expected_rr_learned - df["run_rate"])
df["pi_stage_rr_pressure_raw_inn1_learned"] = np.where(m1, raw_gap, np.nan)

# robust scaling to 1..10 within innings 1
def scale_to_1_10(s: pd.Series, p_low=0.10, p_high=0.90) -> pd.Series:
    x = s.astype(float)
    lo = np.nanquantile(x, p_low)
    hi = np.nanquantile(x, p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.nan, index=s.index)
    z = (x - lo) / (hi - lo)
    z = np.clip(z, 0, 1)
    return 1 + 9 * z

df["pi_stage_rr_pressure_1to10_inn1_learned"] = np.nan
df.loc[m1, "pi_stage_rr_pressure_1to10_inn1_learned"] = scale_to_1_10(
    df.loc[m1, "pi_stage_rr_pressure_raw_inn1_learned"]
)

# if you want to rebuild the composite using learned curve, do it here
# reuse your existing components, but swap in the learned stage RR
inn1_components_learned = [
    "pi_stage_rr_pressure_1to10_inn1_learned",
    "pi_wicket_weighted_pace_1to10_inn1",
    "pi_resource_remaining_1to10_inn1",
    "pi_wicket_pressure_1to10",
    "pi_match_phase_1to10",
]

m2 = df["inning"].eq(2)
inn2_components = [
    "pi_rrr_pressure_1to10_inn2",
    "pi_wicket_pressure_1to10",
    "pi_match_phase_1to10",
]

df["pi_composite_1to10_learnedcurve"] = np.nan
df.loc[m1, "pi_composite_1to10_learnedcurve"] = df.loc[m1, inn1_components_learned].mean(axis=1)
df.loc[m2, "pi_composite_1to10_learnedcurve"] = df.loc[m2, inn2_components].mean(axis=1)

df.to_csv(OUTPUT_PATH, index=False)
print("Saved:", OUTPUT_PATH)