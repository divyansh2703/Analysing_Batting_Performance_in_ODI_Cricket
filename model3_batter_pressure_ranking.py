import pandas as pd
import statsmodels.formula.api as smf

INPUT_FILE = "analysis_model_table.csv"


def main():

    df = pd.read_csv(INPUT_FILE)

    df.columns = df.columns.str.strip()

    # -----------------------
    # Fit scoring model again
    # -----------------------

    df["phase"] = df["phase"].astype("category")
    df["inning"] = df["inning"].astype("category")
    df["match_stage"] = df["match_stage"].astype("category")

    formula = """
    runs_off_bat ~
    pi_composite_1to10_learnedcurve +
    balls_bowled +
    C(phase) +
    C(inning)
    """

    model = smf.ols(formula=formula, data=df).fit()

    print("Scoring model fitted")

    # -----------------------
    # Predict expected runs
    # -----------------------

    df["expected_runs"] = model.predict(df)

    # -----------------------
    # Calculate residual performance
    # -----------------------

    df["performance_residual"] = df["runs_off_bat"] - df["expected_runs"]

    # -----------------------
    # Focus on high pressure
    # -----------------------

    pressure_df = df[df["pi_composite_1to10_learnedcurve"] >= 7].copy()

    print("High pressure deliveries:", len(pressure_df))

    # -----------------------
    # Batter level aggregation
    # -----------------------

    batter_stats = (
        pressure_df
        .groupby("batter")
        .agg(
            balls_faced=("runs_off_bat", "count"),
            actual_runs_per_ball=("runs_off_bat", "mean"),
            expected_runs_per_ball=("expected_runs", "mean"),
            pressure_adjusted_performance=("performance_residual", "mean")
        )
        .reset_index()
    )

    # -----------------------
    # Filter batters with enough balls
    # -----------------------

    batter_stats = batter_stats[batter_stats["balls_faced"] >= 50]

    # -----------------------
    # Rank batters
    # -----------------------

    batter_stats = batter_stats.sort_values(
        "pressure_adjusted_performance",
        ascending=False
    )

    # -----------------------
    # Save results
    # -----------------------

    batter_stats.to_csv("batter_pressure_ranking.csv", index=False)

    top10 = batter_stats.head(10)

    print("\nTop 10 Batters Under Pressure\n")
    print(top10)

    top10.to_csv("top10_batters_under_pressure.csv", index=False)


if __name__ == "__main__":
    main()