import pandas as pd
from lifelines import CoxPHFitter

INPUT_FILE = "analysis_model_table.csv"


def main():
    # ----------------------------
    # Load data
    # ----------------------------
    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()

    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # ----------------------------
    # Keep required columns
    # ----------------------------
    keep_cols = [
        "match_id",
        "inning",
        "batter",
        "runs_off_bat",
        "is_wicket",
        "pi_composite_1to10_learnedcurve",
        "wickets_lost",
        "balls_bowled",
    ]
    df = df[keep_cols].dropna().copy()

    # ----------------------------
    # Batter-innings aggregation
    # ----------------------------
    grouped = (
        df.groupby(["match_id", "inning", "batter"], as_index=False)
        .agg(
            balls_faced=("runs_off_bat", "count"),
            dismissed=("is_wicket", "max"),
            avg_pressure=("pi_composite_1to10_learnedcurve", "mean"),
            avg_wickets_lost=("wickets_lost", "mean"),
            avg_balls_bowled=("balls_bowled", "mean"),
        )
    )

    print(f"Batter-innings rows: {grouped.shape[0]}")
    print(f"Dismissal rate: {grouped['dismissed'].mean():.4f}")

    # ----------------------------
    # Survival dataset
    # ----------------------------
    survival_df = grouped[
        [
            "balls_faced",
            "dismissed",
            "avg_pressure",
            "avg_wickets_lost",
            "avg_balls_bowled",
        ]
    ].copy()

    # ----------------------------
    # Fit Cox model
    # ----------------------------
    cph = CoxPHFitter()

    cph.fit(
        survival_df,
        duration_col="balls_faced",
        event_col="dismissed"
    )

    print("\n=== Corrected Cox Model Summary ===")
    cph.print_summary()

    # ----------------------------
    # Save output
    # ----------------------------
    summary_df = cph.summary.reset_index()
    summary_df.to_csv("model6_survival_summary_corrected.csv", index=False)

    print("\nSaved: model6_survival_summary_corrected.csv")

    # ----------------------------
    # Pressure interpretation
    # ----------------------------
    if "avg_pressure" in cph.summary.index:
        hr = cph.summary.loc["avg_pressure", "exp(coef)"]
        coef = cph.summary.loc["avg_pressure", "coef"]

        print("\n=== Clean Pressure Interpretation ===")
        print(f"Coefficient: {coef:.4f}")
        print(f"Hazard ratio: {hr:.4f}")

        pct = (hr - 1) * 100
        print(f"Each 1-point increase in pressure increases dismissal hazard by ~{pct:.2f}%.")


if __name__ == "__main__":
    main()