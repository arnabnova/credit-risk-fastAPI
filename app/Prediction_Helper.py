"""
prediction_helper.py
--------------------
All preprocessing and prediction logic for the Loan Default Risk Predictor.
Import this module in app.py; keep Streamlit imports out of here.
"""

import joblib
import numpy as np
import pandas as pd
import os
# ── Constants ──────────────────────────────────────────────────────────────────

# Build path relative to this file's location, not the working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model_data.joblib")

RISK_BANDS = [
    (0.00, 0.20, "🟢 Low Risk",       "success"),
    (0.20, 0.50, "🟡 Moderate Risk",  "warning"),
    (0.50, 0.75, "🟠 High Risk",      "error"),
    (0.75, 1.01, "🔴 Very High Risk", "error"),
]

CREDIT_RATING_BANDS = [
    (800, 851, "Excellent", "#2ecc71"),
    (740, 800, "Very Good", "#27ae60"),
    (670, 740, "Good",      "#f39c12"),
    (580, 670, "Fair",      "#e67e22"),
    (300, 580, "Poor",      "#e74c3c"),
]

# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(path: str = MODEL_PATH) -> dict:
    """Load serialised model artefacts from disk."""
    return joblib.load(path)

# ── Preprocessing ──────────────────────────────────────────────────────────────

def build_input_df(
    age: int,
    loan_tenure_months: int,
    number_of_open_accounts: int,
    credit_utilization_ratio: float,
    loan_income_ratio: float,
    delinquency_ratio: float,
    avg_dpd_per_delinquency: float,
    residence_type: str,
    loan_purpose: str,
    loan_type: str,
) -> pd.DataFrame:
    """
    Construct a single-row DataFrame from user inputs and one-hot-encode
    categorical columns so they match training feature names.
    """
    raw = {
        "age":                      age,
        "loan_tenure_months":       loan_tenure_months,
        "number_of_open_accounts":  number_of_open_accounts,
        "credit_utilization_ratio": credit_utilization_ratio,
        "loan_income_ratio":        loan_income_ratio,
        "delinquency_ratio":        delinquency_ratio,
        "avg_dpd_per_delinquency":  avg_dpd_per_delinquency,
        "residence_type":           residence_type,
        "loan_purpose":             loan_purpose,
        "loan_type":                loan_type,
    }
    df = pd.DataFrame([raw])
    df = pd.get_dummies(df, columns=["residence_type", "loan_purpose", "loan_type"])
    return df


def preprocess(df: pd.DataFrame, model_data: dict) -> np.ndarray:
    features      = model_data["features"]
    scaler        = model_data["scaler"]
    cols_to_scale = model_data["cols_to_scale"]

    print("── DEBUG: preprocess() ──")
    print("Input df columns before scaling:", list(df.columns))
    print("Input df values:\n", df.to_dict(orient="records"))
    print("Expected cols_to_scale:", cols_to_scale)
    print("Expected model features:", features)

    # Add ALL columns the scaler expects (not just model features)
    for col in cols_to_scale:
        if col not in df.columns:
            df[col] = 0

    # Scale using ALL cols_to_scale (scaler needs exact same columns as fit time)
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # NOW add any missing model feature columns
    for col in features:
        if col not in df.columns:
            df[col] = 0

    final = df[features]
    print("Final df sent to model:\n", final.to_dict(orient="records"))
    print("Final array:", final.values)
    print("─────────────────────────")

    # Return only the final model features in the correct order
    return final.values

# ── Prediction ─────────────────────────────────────────────────────────────────

def predict(input_array: np.ndarray, model_data: dict):
    """
    Run inference and return (default_probability, credit_score).
    default_probability : float in [0, 1]
    credit_score        : int   in [300, 850]
    """
    model = model_data["model"]
    proba = float(model.predict_proba(input_array)[0][1])
    credit_score = int(np.clip(850 - proba * 550, 300, 850))
    return proba, credit_score


# ── Label helpers ──────────────────────────────────────────────────────────────

def get_risk_level(probability: float):
    """Return (label, streamlit_card_type) for the given probability."""
    for low, high, label, card in RISK_BANDS:
        if low <= probability < high:
            return label, card
    return "🔴 Very High Risk", "error"


def get_credit_rating(score: int):
    """Return (rating_label, hex_color) for the given credit score."""
    for low, high, rating, color in CREDIT_RATING_BANDS:
        if low <= score < high:
            return rating, color
    return "Poor", "#e74c3c"


def get_decision(probability: float):
    """Return (decision_text, streamlit_card_type)."""
    if probability < 0.5:
        return "✅ Loan Recommended", "success"
    return "❌ High Default Risk – Loan Not Recommended", "error"