import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "analysis_model_table.csv"
OUTPUT_DIR = "eda_outputs"


def save_plot(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()

    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # -----------------------------
    # Small fix for remaining phase pressure missings
    # -----------------------------
    if "pi_match_phase_1to10" in df.columns:
        df["pi_match_phase_1to10"] = df["pi_match_phase_1to10"].fillna(df["pi_match_phase_1to10"].median())

    # -----------------------------
    # Summary tables
    # -----------------------------
    summary_rows = []

    summary_rows.append({
        "metric": "rows",
        "value": len(df)
    })
    summary_rows.append({
        "metric": "columns",
        "value": len(df.columns)
    })
    summary_rows.append({
        "metric": "mean_composite_pressure",
        "value": df["pi_composite_1to10_learnedcurve"].mean()
    })
    summary_rows.append({
        "metric": "mean_runs_off_bat",
        "value": df["runs_off_bat"].mean()
    })
    summary_rows.append({
        "metric": "wicket_rate",
        "value": df["is_wicket"].mean()
    })
    summary_rows.append({
        "metric": "boundary_rate",
        "value": df["boundary_flag"].mean()
    })
    summary_rows.append({
        "metric": "dot_ball_rate",
        "value": df["dot_ball_flag"].mean()
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_metrics.csv"), index=False)
    print("Saved summary metrics")

    # -----------------------------
    # Pressure buckets
    # -----------------------------
    df["pressure_bucket"] = pd.cut(
        df["pi_composite_1to10_learnedcurve"],
        bins=[0, 3, 6, 8, 10],
        labels=["Low", "Moderate", "High", "Extreme"],
        include_lowest=True
    )

    # -----------------------------
    # 1. Composite pressure distribution
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(df["pi_composite_1to10_learnedcurve"].dropna(), bins=30)
    plt.xlabel("Composite Pressure (1 to 10)")
    plt.ylabel("Ball Count")
    plt.title("Distribution of Composite Pressure")
    save_plot("01_composite_pressure_distribution.png")

    # -----------------------------
    # 2. Composite pressure by innings
    # -----------------------------
    plt.figure(figsize=(8, 5))
    for inn in sorted(df["inning"].dropna().unique()):
        vals = df.loc[df["inning"] == inn, "pi_composite_1to10_learnedcurve"].dropna()
        plt.hist(vals, bins=30, alpha=0.5, label=f"Innings {int(inn)}")
    plt.xlabel("Composite Pressure (1 to 10)")
    plt.ylabel("Ball Count")
    plt.legend()
    plt.title("Composite Pressure by Innings")
    save_plot("02_composite_pressure_by_innings.png")

    # -----------------------------
    # 3. Mean pressure by phase
    # -----------------------------
    phase_means = df.groupby("phase")["pi_composite_1to10_learnedcurve"].mean().sort_values()
    plt.figure(figsize=(8, 5))
    plt.bar(phase_means.index.astype(str), phase_means.values)
    plt.xlabel("Phase")
    plt.ylabel("Mean Composite Pressure")
    plt.title("Mean Composite Pressure by Phase")
    save_plot("03_mean_pressure_by_phase.png")

    phase_means.to_csv(os.path.join(OUTPUT_DIR, "phase_mean_pressure.csv"))
    print("Saved phase mean pressure table")

    # -----------------------------
    # 4. Pressure across innings progression
    # -----------------------------
    temp = df[["inning", "balls_bowled", "pi_composite_1to10_learnedcurve"]].dropna().copy()
    temp["over_bucket"] = (temp["balls_bowled"] // 6).astype(int)

    agg = temp.groupby(["inning", "over_bucket"])["pi_composite_1to10_learnedcurve"].mean().reset_index()

    plt.figure(figsize=(10, 5))
    for inn in sorted(agg["inning"].unique()):
        sub = agg[agg["inning"] == inn].sort_values("over_bucket")
        plt.plot(sub["over_bucket"], sub["pi_composite_1to10_learnedcurve"], label=f"Innings {int(inn)}")
    plt.xlabel("Over Number")
    plt.ylabel("Mean Composite Pressure")
    plt.legend()
    plt.title("Pressure Across Innings Progression")
    save_plot("04_pressure_over_progression.png")

    agg.to_csv(os.path.join(OUTPUT_DIR, "pressure_over_progression_table.csv"), index=False)
    print("Saved pressure progression table")

    # -----------------------------
    # 5. Mean runs_off_bat by pressure bucket
    # -----------------------------
    runs_bucket = df.groupby("pressure_bucket")["runs_off_bat"].mean().reset_index()

    plt.figure(figsize=(8, 5))
    plt.bar(runs_bucket["pressure_bucket"].astype(str), runs_bucket["runs_off_bat"])
    plt.xlabel("Pressure Bucket")
    plt.ylabel("Mean Runs Off Bat per Ball")
    plt.title("Scoring by Pressure Bucket")
    save_plot("05_runs_by_pressure_bucket.png")

    runs_bucket.to_csv(os.path.join(OUTPUT_DIR, "runs_by_pressure_bucket.csv"), index=False)

    # -----------------------------
    # 6. Wicket probability by pressure bucket
    # -----------------------------
    wicket_bucket = df.groupby("pressure_bucket")["is_wicket"].mean().reset_index()

    plt.figure(figsize=(8, 5))
    plt.bar(wicket_bucket["pressure_bucket"].astype(str), wicket_bucket["is_wicket"])
    plt.xlabel("Pressure Bucket")
    plt.ylabel("Wicket Probability")
    plt.title("Wicket Probability by Pressure Bucket")
    save_plot("06_wicket_rate_by_pressure_bucket.png")

    wicket_bucket.to_csv(os.path.join(OUTPUT_DIR, "wicket_rate_by_pressure_bucket.csv"), index=False)

    # -----------------------------
    # 7. Boundary rate by pressure bucket
    # -----------------------------
    boundary_bucket = df.groupby("pressure_bucket")["boundary_flag"].mean().reset_index()

    plt.figure(figsize=(8, 5))
    plt.bar(boundary_bucket["pressure_bucket"].astype(str), boundary_bucket["boundary_flag"])
    plt.xlabel("Pressure Bucket")
    plt.ylabel("Boundary Rate")
    plt.title("Boundary Rate by Pressure Bucket")
    save_plot("07_boundary_rate_by_pressure_bucket.png")

    boundary_bucket.to_csv(os.path.join(OUTPUT_DIR, "boundary_rate_by_pressure_bucket.csv"), index=False)

    # -----------------------------
    # 8. Dot ball rate by pressure bucket
    # -----------------------------
    dot_bucket = df.groupby("pressure_bucket")["dot_ball_flag"].mean().reset_index()

    plt.figure(figsize=(8, 5))
    plt.bar(dot_bucket["pressure_bucket"].astype(str), dot_bucket["dot_ball_flag"])
    plt.xlabel("Pressure Bucket")
    plt.ylabel("Dot Ball Rate")
    plt.title("Dot Ball Rate by Pressure Bucket")
    save_plot("08_dot_ball_rate_by_pressure_bucket.png")

    dot_bucket.to_csv(os.path.join(OUTPUT_DIR, "dot_ball_rate_by_pressure_bucket.csv"), index=False)

    # -----------------------------
    # 9. Fine-grained runs vs pressure decile
    # -----------------------------
    temp = df[["runs_off_bat", "pi_composite_1to10_learnedcurve"]].dropna().copy()
    temp["pressure_decile"] = pd.qcut(
        temp["pi_composite_1to10_learnedcurve"],
        10,
        labels=False,
        duplicates="drop"
    )

    decile_runs = temp.groupby("pressure_decile")["runs_off_bat"].mean().reset_index()

    plt.figure(figsize=(8, 5))
    plt.plot(decile_runs["pressure_decile"], decile_runs["runs_off_bat"], marker="o")
    plt.xlabel("Pressure Decile (Low to High)")
    plt.ylabel("Mean Runs Off Bat per Ball")
    plt.title("Runs Off Bat by Pressure Decile")
    save_plot("09_runs_by_pressure_decile.png")

    decile_runs.to_csv(os.path.join(OUTPUT_DIR, "runs_by_pressure_decile.csv"), index=False)

    # -----------------------------
    # 10. Fine-grained wicket probability vs pressure decile
    # -----------------------------
    temp = df[["is_wicket", "pi_composite_1to10_learnedcurve"]].dropna().copy()
    temp["pressure_decile"] = pd.qcut(
        temp["pi_composite_1to10_learnedcurve"],
        10,
        labels=False,
        duplicates="drop"
    )

    decile_wicket = temp.groupby("pressure_decile")["is_wicket"].mean().reset_index()

    plt.figure(figsize=(8, 5))
    plt.plot(decile_wicket["pressure_decile"], decile_wicket["is_wicket"], marker="o")
    plt.xlabel("Pressure Decile (Low to High)")
    plt.ylabel("Wicket Probability")
    plt.title("Wicket Probability by Pressure Decile")
    save_plot("10_wicket_rate_by_pressure_decile.png")

    decile_wicket.to_csv(os.path.join(OUTPUT_DIR, "wicket_rate_by_pressure_decile.csv"), index=False)

    print("\nEDA completed successfully.")
    print(f"All outputs saved in folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()