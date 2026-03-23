import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

INPUT_FILE = "analysis_model_table.csv"


def main():
    # ----------------------------
    # Load dataset
    # ----------------------------
    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()

    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # ----------------------------
    # Keep only rows with target
    # ----------------------------
    df = df[df["is_wicket"].notna()].copy()

    # ----------------------------
    # Fill tiny remaining missing for phase pressure if needed
    # ----------------------------
    if "pi_match_phase_1to10" in df.columns:
        df["pi_match_phase_1to10"] = df["pi_match_phase_1to10"].fillna(df["pi_match_phase_1to10"].median())

    # ----------------------------
    # Convert categorical variables
    # ----------------------------
    df["phase"] = df["phase"].astype("category")
    df["match_stage"] = df["match_stage"].astype("category")
    df["inning"] = df["inning"].astype("category")

    # ----------------------------
    # Corrected logistic model
    # Remove wickets_lost because it overlaps heavily with pressure
    # ----------------------------
    formula = """
    is_wicket ~
    pi_composite_1to10_learnedcurve +
    balls_bowled +
    C(phase) +
    C(inning) +
    C(match_stage)
    """

    model = smf.logit(formula=formula, data=df).fit()

    print("\n=== Corrected Logistic Regression Summary ===")
    print(model.summary())

    # ----------------------------
    # Odds ratios with confidence intervals
    # ----------------------------
    params = model.params
    conf = model.conf_int()
    conf.columns = ["ci_lower", "ci_upper"]

    results = pd.DataFrame({
        "coefficient": params,
        "odds_ratio": np.exp(params),
        "ci_lower_odds_ratio": np.exp(conf["ci_lower"]),
        "ci_upper_odds_ratio": np.exp(conf["ci_upper"]),
        "p_value": model.pvalues
    })

    results.to_csv("model2_wicket_odds_ratios_corrected.csv")

    print("\nSaved odds ratios to model2_wicket_odds_ratios_corrected.csv")

    # ----------------------------
    # Quick interpretation for pressure coefficient
    # ----------------------------
    if "pi_composite_1to10_learnedcurve" in results.index:
        coef = results.loc["pi_composite_1to10_learnedcurve", "coefficient"]
        odds = results.loc["pi_composite_1to10_learnedcurve", "odds_ratio"]
        pval = results.loc["pi_composite_1to10_learnedcurve", "p_value"]

        print("\n=== Pressure Effect Interpretation ===")
        print(f"Pressure coefficient: {coef:.4f}")
        print(f"Pressure odds ratio: {odds:.4f}")
        print(f"Pressure p-value: {pval:.6f}")

        if odds > 1:
            pct = (odds - 1) * 100
            print(f"Each 1-point increase in pressure increases wicket odds by about {pct:.2f}%.")
        else:
            pct = (1 - odds) * 100
            print(f"Each 1-point increase in pressure decreases wicket odds by about {pct:.2f}%.")


if __name__ == "__main__":
    main()