import pandas as pd
import statsmodels.formula.api as smf

INPUT_FILE = "analysis_model_table.csv"


def main():

    df = pd.read_csv(INPUT_FILE)

    df.columns = df.columns.str.strip()

    # ----------------------------
    # Drop rows with missing target
    # ----------------------------

    df = df[df["runs_off_bat"].notna()].copy()

    # ----------------------------
    # Convert categorical variables
    # ----------------------------

    df["phase"] = df["phase"].astype("category")
    df["match_stage"] = df["match_stage"].astype("category")
    df["inning"] = df["inning"].astype("category")

    # batter and bowler effects
    df["batter"] = df["batter"].astype("category")
    df["bowler"] = df["bowler"].astype("category")

    # ----------------------------
    # Regression formula
    # ----------------------------

    formula = """
    runs_off_bat ~
    pi_composite_1to10_learnedcurve +
    wickets_lost +
    balls_bowled +
    C(phase) +
    C(inning) +
    C(match_stage)
    """

    model = smf.ols(formula=formula, data=df).fit()

    print(model.summary())

    # ----------------------------
    # Save coefficients
    # ----------------------------

    coef_table = model.summary2().tables[1]

    coef_table.to_csv("model1_scoring_coefficients.csv")

    print("Saved coefficients to model1_scoring_coefficients.csv")


if __name__ == "__main__":
    main()