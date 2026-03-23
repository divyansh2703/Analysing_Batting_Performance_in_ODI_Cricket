import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency

INPUT_FILE = "analysis_model_table.csv"


def main():

    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()

    print(f"Loaded dataset: {df.shape[0]} rows")

    # -----------------------------
    # Define pressure groups
    # -----------------------------
    low_pressure = df[df["pi_composite_1to10_learnedcurve"] <= 3]
    high_pressure = df[df["pi_composite_1to10_learnedcurve"] >= 7]

    print("\nSample sizes")
    print("Low pressure:", len(low_pressure))
    print("High pressure:", len(high_pressure))

    # =============================
    # 1. Mann-Whitney U Test (Scoring)
    # =============================
    print("\n=== Mann-Whitney U Test (Runs per ball) ===")

    u_stat, p_value = mannwhitneyu(
        low_pressure["runs_off_bat"],
        high_pressure["runs_off_bat"],
        alternative="two-sided"
    )

    print(f"U statistic: {u_stat}")
    print(f"P-value: {p_value}")

    print("\nMean runs per ball")
    print("Low pressure:", low_pressure["runs_off_bat"].mean())
    print("High pressure:", high_pressure["runs_off_bat"].mean())

    if p_value < 0.05:
        print("Result: Significant difference in scoring between pressure levels")
    else:
        print("Result: No significant difference")

    # =============================
    # 2. Chi-Square Test (Wickets)
    # =============================
    print("\n=== Chi-Square Test (Wicket Probability) ===")

    contingency = pd.crosstab(
        df["pi_composite_1to10_learnedcurve"] >= 7,
        df["is_wicket"]
    )

    chi2, p_val, _, _ = chi2_contingency(contingency)

    print("Contingency Table:")
    print(contingency)

    print(f"\nChi-square statistic: {chi2}")
    print(f"P-value: {p_val}")

    if p_val < 0.05:
        print("Result: Wicket probability depends on pressure")
    else:
        print("Result: No significant relationship")

    # =============================
    # 3. Permutation Test (Robustness)
    # =============================
    print("\n=== Permutation Test (Scoring Difference) ===")

    observed_diff = (
        high_pressure["runs_off_bat"].mean() -
        low_pressure["runs_off_bat"].mean()
    )

    combined = df["runs_off_bat"].values
    pressure_flag = (df["pi_composite_1to10_learnedcurve"] >= 7).astype(int)

    n_perm = 1000
    perm_diffs = []

    for _ in range(n_perm):
        shuffled = np.random.permutation(pressure_flag)
        group1 = combined[shuffled == 1]
        group0 = combined[shuffled == 0]

        if len(group1) > 0 and len(group0) > 0:
            perm_diffs.append(group1.mean() - group0.mean())

    perm_diffs = np.array(perm_diffs)

    p_perm = np.mean(np.abs(perm_diffs) >= abs(observed_diff))

    print(f"Observed difference: {observed_diff}")
    print(f"Permutation p-value: {p_perm}")

    if p_perm < 0.05:
        print("Result: Difference is statistically robust")
    else:
        print("Result: Difference may be random")

    print("\nHypothesis testing completed.")


if __name__ == "__main__":
    main()