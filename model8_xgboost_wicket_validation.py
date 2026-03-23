import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

INPUT_FILE = "analysis_model_table.csv"


def main():
    # ----------------------------
    # Load data
    # ----------------------------
    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()

    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # ----------------------------
    # Select features and target
    # ----------------------------
    feature_cols = [
        "pi_composite_1to10_learnedcurve",
        "pi_wicket_pressure_1to10",
        "pi_match_phase_1to10",
        "balls_bowled",
        "wickets_lost",
        "inning",
        "phase",
        "match_stage",
    ]

    target_col = "is_wicket"

    df_model = df[feature_cols + [target_col]].dropna().copy()

    print(f"Rows after dropping missing values: {df_model.shape[0]}")

    X = df_model[feature_cols]
    y = df_model[target_col].astype(int)

    # ----------------------------
    # Train test split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")

    # ----------------------------
    # Define categorical and numeric columns
    # ----------------------------
    categorical_cols = ["inning", "phase", "match_stage"]
    numeric_cols = [
        "pi_composite_1to10_learnedcurve",
        "pi_wicket_pressure_1to10",
        "pi_match_phase_1to10",
        "balls_bowled",
        "wickets_lost",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    # ----------------------------
    # XGBoost classifier
    # ----------------------------
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    print("\nTraining XGBoost model...\n")
    clf.fit(X_train, y_train)

    # ----------------------------
    # Predictions
    # ----------------------------
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)

    print("=== XGBoost Wicket Validation Results ===")
    print(f"ROC-AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # ----------------------------
    # Feature importance extraction
    # ----------------------------
    ohe = clf.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(categorical_cols)
    all_feature_names = list(cat_feature_names) + numeric_cols

    booster = clf.named_steps["model"]
    importances = booster.feature_importances_

    importance_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    importance_df.to_csv("model8_xgboost_feature_importance.csv", index=False)

    print("\nTop 15 Features:")
    print(importance_df.head(15))

    # ----------------------------
    # Plot top 15 feature importances
    # ----------------------------
    top15 = importance_df.head(15).iloc[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(top15["feature"], top15["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 15 XGBoost Feature Importances for Wicket Prediction")
    plt.tight_layout()
    plt.savefig("model8_xgboost_feature_importance.png", dpi=300)
    plt.show()

    print("\nSaved:")
    print("model8_xgboost_feature_importance.csv")
    print("model8_xgboost_feature_importance.png")


if __name__ == "__main__":
    main()