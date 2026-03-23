import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s

INPUT_FILE = "analysis_model_table.csv"


def main():

    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()

    print(f"Loaded dataset: {df.shape[0]} rows")

    # -----------------------------
    # Select relevant columns
    # -----------------------------
    df = df[[
        "runs_off_bat",
        "pi_composite_1to10_learnedcurve",
        "balls_bowled"
    ]].dropna()

    # -----------------------------
    # Define features
    # -----------------------------
    X = df[[
        "pi_composite_1to10_learnedcurve",
        "balls_bowled"
    ]].values

    y = df["runs_off_bat"].values

    # -----------------------------
    # Fit GAM model
    # -----------------------------
    print("\nFitting GAM model...")

    gam = LinearGAM(
        s(0) + s(1)
    ).fit(X, y)

    print("GAM fitted successfully")

    # -----------------------------
    # Generate predictions for plotting
    # -----------------------------
    pressure_range = np.linspace(
        df["pi_composite_1to10_learnedcurve"].min(),
        df["pi_composite_1to10_learnedcurve"].max(),
        100
    )

    # keep balls_bowled fixed at median
    balls_median = df["balls_bowled"].median()

    X_plot = np.column_stack([
        pressure_range,
        np.full_like(pressure_range, balls_median)
    ])

    y_pred = gam.predict(X_plot)

    # -----------------------------
    # Plot result
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(pressure_range, y_pred)
    plt.xlabel("Pressure (1 to 10)")
    plt.ylabel("Expected Runs per Ball")
    plt.title("Nonlinear Effect of Pressure on Scoring (GAM)")
    plt.savefig("gam_pressure_vs_runs.png", dpi=300)
    plt.show()

    print("Saved plot: gam_pressure_vs_runs.png")

    # -----------------------------
    # Print summary
    # -----------------------------
    print("\nGAM Summary:")
    print(gam.summary())


if __name__ == "__main__":
    main()