import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

INPUT_FILE = "analysis_model_table.csv"


def main():

    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()

    # ----------------------------
    # Fit scoring model (baseline)
    # ----------------------------
    formula = """
    runs_off_bat ~
    pi_composite_1to10_learnedcurve +
    balls_bowled +
    C(phase) +
    C(inning)
    """

    model = smf.ols(formula=formula, data=df).fit()

    # ----------------------------
    # Expected runs
    # ----------------------------
    df["expected_runs"] = model.predict(df)
    df["residual"] = df["runs_off_bat"] - df["expected_runs"]

    # ----------------------------
    # High pressure filter
    # ----------------------------
    high_df = df[df["pi_composite_1to10_learnedcurve"] >= 7].copy()

    print(f"High pressure rows: {len(high_df)}")

    # ----------------------------
    # Batter aggregation
    # ----------------------------
    batter_stats = (
        high_df.groupby("batter")
        .agg(
            balls_faced=("runs_off_bat", "count"),
            avg_residual=("residual", "mean"),
            actual_rpb=("runs_off_bat", "mean"),
            expected_rpb=("expected_runs", "mean")
        )
        .reset_index()
    )

    # ----------------------------
    # Filter reliable players
    # ----------------------------
    batter_stats = batter_stats[batter_stats["balls_faced"] >= 110]

    # ----------------------------
    # Stability weighting
    # ----------------------------
    batter_stats["weight"] = np.log(batter_stats["balls_faced"])

    batter_stats["final_score"] = (
        batter_stats["avg_residual"] *
        batter_stats["weight"]
    )

    # ----------------------------
    # Rank
    # ----------------------------
    batter_stats = batter_stats.sort_values(
        "final_score",
        ascending=False
    )

    # ----------------------------
    # Save results
    # ----------------------------
    batter_stats.to_csv("final_batter_ranking.csv", index=False)

    top10 = batter_stats.head(10)

    print("\n=== FINAL TOP 10 BATTERS UNDER PRESSURE ===\n")
    print(top10)

    top10.to_csv("top10_final_batters.csv", index=False)


if __name__ == "__main__":
    main()