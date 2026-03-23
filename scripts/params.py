import pandas as pd
import numpy as np
import statsmodels.api as sm
import json

INPUT_PATH = "cricket_with_pressure_indexes_corrected.csv"
OUT_PARAMS_JSON = "inn1_par_curve_params.json"

df = pd.read_csv(INPUT_PATH)
df.columns = [c.strip() for c in df.columns]

# numeric safety
for c in ["inning", "balls_bowled", "run_rate"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# first innings only
inn1 = df[df["inning"] == 1].copy()

# remove rows with missing essentials
inn1 = inn1.dropna(subset=["balls_bowled", "run_rate"])

# define time progress in innings
TOTAL_BALLS = 300.0
inn1["t"] = inn1["balls_bowled"] / TOTAL_BALLS
inn1["t2"] = inn1["t"] ** 2

# Fit: run_rate = base + k1*t + k2*t^2
X = sm.add_constant(inn1[["t", "t2"]])  # adds intercept which becomes base
y = inn1["run_rate"]

model = sm.OLS(y, X).fit()

base = float(model.params["const"])
k1 = float(model.params["t"])
k2 = float(model.params["t2"])

print("Estimated first innings par curve parameters")
print("base:", base)
print("k1:", k1)
print("k2:", k2)
print("\nModel summary")
print(model.summary())

params = {
    "base": base,
    "k1": k1,
    "k2": k2,
    "total_balls": TOTAL_BALLS,
    "n_rows": int(len(inn1)),
    "r2": float(model.rsquared)
}

with open(OUT_PARAMS_JSON, "w", encoding="utf-8") as f:
    json.dump(params, f, indent=2)

print("\nSaved parameters to:", OUT_PARAMS_JSON)