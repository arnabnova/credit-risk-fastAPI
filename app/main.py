from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

from Prediction_Helper import (
    load_model,
    build_input_df,
    preprocess,
    predict,
    get_risk_level,
    get_credit_rating,
    get_decision,
)

app = FastAPI(title="Loan Default Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_data = load_model()

# ── Schemas ──────────────────────────────────────────────────────────────

class LoanInput(BaseModel):
    age: int = Field(..., ge=18, le=100)
    loan_tenure_months: int = Field(..., gt=0)
    number_of_open_accounts: int = Field(..., ge=0)
    credit_utilization_ratio: float = Field(..., ge=0, le=1)
    loan_income_ratio: float = Field(..., ge=0)
    delinquency_ratio: float = Field(..., ge=0, le=1)
    avg_dpd_per_delinquency: float = Field(..., ge=0)
    residence_type: Literal["Owned", "Rented", "Mortgage"]
    loan_purpose: Literal["Home", "Auto", "Personal", "Education"]
    loan_type: Literal["Secured", "Unsecured"]

class LoanOutput(BaseModel):
    default_probability: float
    credit_score: int
    risk_label: str
    credit_rating: str
    credit_rating_color: str
    decision: str

# ── Routes ───────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=LoanOutput)
def get_prediction(input: LoanInput):
    try:
        print("── DEBUG: /predict received ──")
        print(input.dict())

        df = build_input_df(
            age=input.age,
            loan_tenure_months=input.loan_tenure_months,
            number_of_open_accounts=input.number_of_open_accounts,
            credit_utilization_ratio=input.credit_utilization_ratio,
            loan_income_ratio=input.loan_income_ratio,
            delinquency_ratio=input.delinquency_ratio,
            avg_dpd_per_delinquency=input.avg_dpd_per_delinquency,
            residence_type=input.residence_type,
            loan_purpose=input.loan_purpose,
            loan_type=input.loan_type,
        )
        print("build_input_df() output columns:", list(df.columns))

        input_array = preprocess(df, model_data)
        proba, credit_score = predict(input_array, model_data)
        print(f"DEBUG: raw proba={proba}, credit_score={credit_score}")

        risk_label, _ = get_risk_level(proba)
        credit_rating, rating_color = get_credit_rating(credit_score)
        decision, _ = get_decision(proba)

        return LoanOutput(
            default_probability=round(proba, 4),
            credit_score=credit_score,
            risk_label=risk_label,
            credit_rating=credit_rating,
            credit_rating_color=rating_color,
            decision=decision,
        )
    except Exception as e:
        print("DEBUG ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))