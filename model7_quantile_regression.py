import pandas as pd
import statsmodels.formula.api as smf

INPUT_FILE = "analysis_model_table.csv"


def main():
    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()

    print(f"Loaded dataset: {df.shape[0]} rows")

    # ----------------------------
    # Prepare data
    # ----------------------------
    df = df[[
        "runs_off_bat",
        "pi_composite_1to10_learnedcurve",
        "balls_bowled",
        "phase",
        "inning",
        "match_stage"
    ]].dropna()

    # ----------------------------
    # Model formula
    # ----------------------------
    formula = """
    runs_off_bat ~
    pi_composite_1to10_learnedcurve +
    balls_bowled +
    C(phase) +
    C(inning) +
    C(match_stage)
    """

    # ----------------------------
    # Fit quantiles
    # ----------------------------
    quantiles = [0.25, 0.5, 0.75, 0.9]

    results = []

    print("\nFitting Quantile Regression Models...\n")

    for q in quantiles:
        model = smf.quantreg(formula, df)
        res = model.fit(q=q)

        coef = res.params["pi_composite_1to10_learnedcurve"]

        print(f"Quantile {q}: Pressure coefficient = {coef:.4f}")

        results.append({
            "quantile": q,
            "pressure_coefficient": coef
        })

    # ----------------------------
    # Save results
    # ----------------------------
    results_df = pd.DataFrame(results)
    results_df.to_csv("model7_quantile_results.csv", index=False)

    print("\nSaved: model7_quantile_results.csv")


if __name__ == "__main__":
    main()