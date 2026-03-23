import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("cricket_with_pressure_indexes_corrected_learnedcurve.csv")
df.columns = [c.strip() for c in df.columns]

# 1) Distribution of composite pressure by innings
plt.figure()
for inn in sorted(df["inning"].dropna().unique()):
    vals = df.loc[df["inning"].eq(inn), "pi_composite_1to10_learnedcurve"].dropna()
    plt.hist(vals, bins=30, alpha=0.5, label=f"innings {int(inn)}")
plt.xlabel("Composite pressure (1-10, learned curve)")
plt.ylabel("Ball count")
plt.legend()
plt.show()

# 2) Mean composite pressure by phase
plt.figure()
phase_means = df.groupby("phase")["pi_composite_1to10_learnedcurve"].mean().sort_values()
plt.bar(phase_means.index.astype(str), phase_means.values)
plt.xlabel("Phase")
plt.ylabel("Mean composite pressure")
plt.show()

# 3) Mean composite pressure over innings progress (over buckets)
tmp = df[["inning","balls_bowled","pi_composite_1to10_learnedcurve"]].dropna()
tmp["over_bucket"] = (tmp["balls_bowled"] // 6).astype(int)

agg = tmp.groupby(["inning","over_bucket"])["pi_composite_1to10_learnedcurve"].mean().reset_index()

plt.figure()
for inn in sorted(agg["inning"].unique()):
    sub = agg[agg["inning"] == inn].sort_values("over_bucket")
    plt.plot(sub["over_bucket"], sub["pi_composite_1to10_learnedcurve"], label=f"innings {int(inn)}")
plt.xlabel("Over number")
plt.ylabel("Mean composite pressure")
plt.legend()
plt.show()

# 4) Runs scoring vs pressure bucket (deciles)
tmp = df[["runs_off_bat","pi_composite_1to10_learnedcurve"]].dropna()
tmp["pressure_decile"] = pd.qcut(tmp["pi_composite_1to10_learnedcurve"], 10, labels=False, duplicates="drop")
runs_by_decile = tmp.groupby("pressure_decile")["runs_off_bat"].mean().reset_index()

plt.figure()
plt.plot(runs_by_decile["pressure_decile"], runs_by_decile["runs_off_bat"], marker="o")
plt.xlabel("Pressure decile (low to high)")
plt.ylabel("Mean runs_off_bat per ball")
plt.show()

# 5) Wicket rate vs pressure bucket
tmp = df[["is_wicket","pi_composite_1to10_learnedcurve"]].dropna()
tmp["pressure_decile"] = pd.qcut(tmp["pi_composite_1to10_learnedcurve"], 10, labels=False, duplicates="drop")
wicket_by_decile = tmp.groupby("pressure_decile")["is_wicket"].mean().reset_index()

plt.figure()
plt.plot(wicket_by_decile["pressure_decile"], wicket_by_decile["is_wicket"], marker="o")
plt.xlabel("Pressure decile (low to high)")
plt.ylabel("Wicket probability per ball")
plt.show()