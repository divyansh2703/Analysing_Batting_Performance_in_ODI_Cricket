import warnings

import pandas as pd
import statsmodels.formula.api as smf

INPUT_FILE = "analysis_model_table.csv"


def fit_mixed_model(formula, df):
    candidate_specs = [
        {
            "name": "batter + match_id variance component",
            "kwargs": {
                "groups": df["batter"],
                "vc_formula": {"match_id": "0 + C(match_id)"},
            },
        },
        {
            "name": "batter random intercept",
            "kwargs": {
                "groups": df["batter"],
            },
        },
        {
            "name": "match_id random intercept",
            "kwargs": {
                "groups": df["match_id"],
            },
        },
    ]

    methods = ["lbfgs", "powell"]
    failures = []

    for spec in candidate_specs:
        for method in methods:
            print(f"Trying mixed model: {spec['name']} with {method}")
            try:
                model = smf.mixedlm(formula=formula, data=df, **spec["kwargs"])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = model.fit(reml=False, method=method, maxiter=200, disp=False)
                return model, result, spec["name"], method, failures
            except Exception as exc:
                failures.append(f"{spec['name']} with {method}: {type(exc).__name__}: {exc}")

    failure_text = "\n".join(failures)
    raise RuntimeError(f"All mixed model fits failed.\n{failure_text}")


def main():
    # ----------------------------
    # Load data
    # ----------------------------
    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()

    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # ----------------------------
    # Keep required rows
    # ----------------------------
    keep_cols = [
        "runs_off_bat",
        "pi_composite_1to10_learnedcurve",
        "balls_bowled",
        "phase",
        "inning",
        "match_stage",
        "batter",
        "match_id"
    ]
    df = df[keep_cols].dropna().copy()

    # Standardize string labels to avoid duplicate category levels such as group/Group.
    for col in ["phase", "match_stage", "batter"]:
        df[col] = df[col].astype(str).str.strip()
    df["match_stage"] = df["match_stage"].str.lower()

    print(f"Rows after dropping missing values: {df.shape[0]}")

    # ----------------------------
    # Convert categorical variables
    # ----------------------------
    df["phase"] = df["phase"].astype("category")
    df["inning"] = df["inning"].astype("category")
    df["match_stage"] = df["match_stage"].astype("category")
    df["batter"] = df["batter"].astype("category")
    df["match_id"] = df["match_id"].astype("category")

    # ----------------------------
    # Mixed effects model
    # groups = batter random intercept
    # vc_formula adds match_id random intercept
    # ----------------------------
    formula = """
    runs_off_bat ~
    pi_composite_1to10_learnedcurve +
    balls_bowled +
    C(phase) +
    C(inning) +
    C(match_stage)
    """

    print("\nFitting mixed effects model. This may take some time...\n")

    model, result, fitted_spec, fitted_method, failures = fit_mixed_model(formula, df)

    print(f"Selected specification: {fitted_spec}")
    print(f"Optimizer: {fitted_method}")
    if failures:
        print("\nFailed attempts before convergence:")
        for failure in failures:
            print(f"- {failure}")

    print(result.summary())

    # ----------------------------
    # Save fixed effects
    # ----------------------------
    fixed_effects = pd.DataFrame({
        "term": result.fe_params.index,
        "coefficient": result.fe_params.values
    })
    fixed_effects.to_csv("model4_mixed_scoring_fixed_effects.csv", index=False)

    model_info = pd.DataFrame(
        {
            "selected_specification": [fitted_spec],
            "optimizer": [fitted_method],
            "log_likelihood": [result.llf],
            "converged": [getattr(result, "converged", None)],
        }
    )
    model_info.to_csv("model4_mixed_scoring_model_info.csv", index=False)

    # ----------------------------
    # Save batter random effects
    # ----------------------------
    random_effects = []

    for batter, effects in result.random_effects.items():
        # batter random intercept is usually the first entry
        # depending on structure, effects may include vc components
        if hasattr(effects, "index"):
            values = effects.to_dict()
        else:
            values = {"random_effect": effects}

        batter_intercept = list(values.values())[0] if len(values) > 0 else None

        random_effects.append({
            "batter": batter,
            "batter_random_intercept": batter_intercept
        })

    random_effects_df = pd.DataFrame(random_effects)
    random_effects_df.to_csv("model4_mixed_scoring_batter_effects.csv", index=False)

    print("\nSaved:")
    print("model4_mixed_scoring_fixed_effects.csv")
    print("model4_mixed_scoring_batter_effects.csv")
    print("model4_mixed_scoring_model_info.csv")


if __name__ == "__main__":
    main()
