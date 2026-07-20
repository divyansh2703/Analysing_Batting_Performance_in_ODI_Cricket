import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

INPUT_FILE = "analysis_model_table.csv"
OUTPUT_FILE = "vif_results.csv"

# Use the same fixed-effect predictors as the scoring models.
FEATURES = [
    "pi_composite_1to10_learnedcurve",
    "balls_bowled",
    "phase",
    "inning",
    "match_stage",
]


def main():
    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()

    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df_vif = df[FEATURES].dropna().copy()
    df_vif = pd.get_dummies(
        df_vif,
        columns=["phase", "inning", "match_stage"],
        drop_first=True,
        dtype=float,
    )

    X = sm.add_constant(df_vif, has_constant="add")

    vif_data = pd.DataFrame(
        {
            "Feature": X.columns,
            "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        }
    )

    print("\n=== VIF RESULTS ===\n")
    print(vif_data)

    vif_data.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
