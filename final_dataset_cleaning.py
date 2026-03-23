import pandas as pd


INPUT_FILE = "cricket_with_pressure_indexes_corrected_learnedcurve.csv"
OUTPUT_FILE = "analysis_model_table.csv"


def main():
    # -------------------------
    # Load dataset
    # -------------------------
    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()

    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # -------------------------
    # Drop useless columns
    # -------------------------
    drop_patterns = [
        "team1_player",
        "team2_player",
        "Unnamed",
        "_meta",
    ]

    for pattern in drop_patterns:
        cols_to_drop = [c for c in df.columns if pattern in c]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop, errors="ignore")
            print(f"Dropped {len(cols_to_drop)} columns matching pattern: {pattern}")

    drop_cols = [
        "pressure_index",
        "projected_score",
        "pi_stage_rr_pressure_raw_inn1",
        "pi_resource_remaining_raw_inn1",
        "pi_rrr_pressure_raw_inn2",
        "pi_stage_rr_pressure_raw_inn1_learned",
    ]

    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    if existing_drop_cols:
        df = df.drop(columns=existing_drop_cols, errors="ignore")
        print(f"Dropped specific columns: {existing_drop_cols}")

    # -------------------------
    # Handle missing values
    # -------------------------
    if "phase" in df.columns:
        missing_phase_before = df["phase"].isna().sum()
        df["phase"] = df["phase"].fillna("Unknown")
        print(f"Filled missing phase values: {missing_phase_before}")

    if "match_stage" in df.columns:
        missing_stage_before = df["match_stage"].isna().sum()
        df["match_stage"] = df["match_stage"].fillna("Group")
        print(f"Filled missing match_stage values: {missing_stage_before}")

    # Remove invalid innings 2 rows where required RR is still missing
    if "inning" in df.columns and "required_run_rate_filled" in df.columns:
        before_rows = len(df)
        mask_invalid_chase = (df["inning"] == 2) & (df["required_run_rate_filled"].isna())
        removed_rows = mask_invalid_chase.sum()
        df = df[~mask_invalid_chase].copy()
        print(f"Removed innings 2 rows with missing required_run_rate_filled: {removed_rows}")
        print(f"Rows after removal: {len(df)} (removed {before_rows - len(df)})")

    # -------------------------
    # Create analysis features
    # -------------------------
    if "runs_off_bat" in df.columns:
        df["boundary_flag"] = df["runs_off_bat"].isin([4, 6]).astype(int)

    if "total_runs" in df.columns:
        df["dot_ball_flag"] = (df["total_runs"] == 0).astype(int)

    if "pi_composite_1to10_learnedcurve" in df.columns:
        df["high_pressure"] = (df["pi_composite_1to10_learnedcurve"] >= 7).astype(int)

    # Optional useful categorical cleaning
    cat_cols = ["inning", "phase", "match_stage", "batting_team", "bowling_team", "batter", "bowler", "venue"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # -------------------------
    # Keep only useful modelling columns
    # -------------------------
    keep_cols = [
        # identifiers and context
        "match_id",
        "inning",
        "phase",
        "match_stage",
        "venue",
        "batting_team",
        "bowling_team",
        "batter",
        "bowler",

        # ball progression and state
        "balls_bowled",
        "balls_remaining_filled",
        "current_score",
        "partnership_runs",
        "wickets_lost",
        "run_rate",

        # chase context
        "target_runs_filled",
        "runs_remaining_filled",
        "required_run_rate_filled",

        # outcomes
        "runs_off_bat",
        "total_runs",
        "is_wicket",

        # engineered pressure features
        "pi_stage_rr_pressure_1to10_inn1_learned",
        "pi_wicket_weighted_pace_1to10_inn1",
        "pi_resource_remaining_1to10_inn1",
        "pi_rrr_pressure_1to10_inn2",
        "pi_wicket_pressure_1to10",
        "pi_match_phase_1to10",
        "pi_composite_1to10_learnedcurve",

        # analysis helper features
        "boundary_flag",
        "dot_ball_flag",
        "high_pressure",
    ]

    keep_cols_existing = [c for c in keep_cols if c in df.columns]
    df_final = df[keep_cols_existing].copy()

    # -------------------------
    # Final checks
    # -------------------------
    print("\nFinal dataset summary")
    print(f"Rows: {df_final.shape[0]}")
    print(f"Columns: {df_final.shape[1]}")

    print("\nMissing values in final dataset")
    missing_summary = df_final.isna().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
    if len(missing_summary) > 0:
        print(missing_summary)
    else:
        print("No missing values")

    print("\nPreview of final columns")
    print(df_final.columns.tolist())

    # -------------------------
    # Save output
    # -------------------------
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved cleaned dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()