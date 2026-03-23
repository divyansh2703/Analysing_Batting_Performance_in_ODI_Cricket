import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

INPUT_FILE = "analysis_model_table.csv"


def main():

    df = pd.read_csv(INPUT_FILE)

    df.columns = df.columns.str.strip()

    # ----------------------------
    # Remove missing rows
    # ----------------------------

    df = df[df["is_wicket"].notna()].copy()

    # ----------------------------
    # Convert categorical variables
    # ----------------------------

    df["phase"] = df["phase"].astype("category")
    df["match_stage"] = df["match_stage"].astype("category")
    df["inning"] = df["inning"].astype("category")

    # ----------------------------
    # Logistic regression model
    # ----------------------------

    formula = """
    is_wicket ~
    pi_composite_1to10_learnedcurve +
    wickets_lost +
    balls_bowled +
    C(phase) +
    C(inning) +
    C(match_stage)
    """

    model = smf.logit(formula=formula, data=df).fit()

    print(model.summary())

    # ----------------------------
    # Convert coefficients to odds ratios
    # ----------------------------

    params = model.params
    odds_ratios = np.exp(params)

    results = pd.DataFrame({
        "coefficient": params,
        "odds_ratio": odds_ratios
    })

    results.to_csv("model2_wicket_odds_ratios.csv")

    print("\nSaved odds ratios to model2_wicket_odds_ratios.csv")


if __name__ == "__main__":
    main()