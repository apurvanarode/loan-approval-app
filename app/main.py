import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, func
from sqlalchemy.orm import Session, declarative_base, sessionmaker

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(ROOT))  # lets joblib find ml.features when loading the model

# ---------- Database ----------
db_url = os.getenv('DATABASE_URL', f'sqlite:///{ROOT}/loans.db')
if db_url.startswith('postgres://'):  # some hosts use the old prefix
    db_url = db_url.replace('postgres://', 'postgresql://', 1)
engine = create_engine(db_url, connect_args={'check_same_thread': False} if db_url.startswith('sqlite') else {})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class LoanApplication(Base):
    __tablename__ = 'applications'
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    label = Column(String(60))
    applicant_income = Column(Float)
    coapplicant_income = Column(Float)
    loan_amount = Column(Float)
    loan_term = Column(Integer)
    credit_history = Column(Integer)
    dependents = Column(String(3))
    education = Column(String(20))
    self_employed = Column(String(3))
    property_area = Column(String(15))
    probability = Column(Float)
    decision = Column(String(20))


Base.metadata.create_all(engine)  # creates the table the first time the app starts


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------- Model ----------
model = joblib.load(BASE / 'artifacts' / 'model.joblib')
metrics = json.loads((BASE / 'artifacts' / 'metrics.json').read_text())


class LoanIn(BaseModel):
    label: str = Field('Applicant', max_length=60)
    applicant_income: float = Field(ge=0, le=1_000_000)
    coapplicant_income: float = Field(ge=0, le=1_000_000)
    loan_amount: float = Field(gt=0, le=10_000)  # in thousands
    loan_term: int = Field(ge=12, le=480)  # in months
    credit_history: Literal[0, 1]
    dependents: Literal['0', '1', '2', '3+']
    education: Literal['Graduate', 'Not Graduate']
    self_employed: Literal['Yes', 'No']
    property_area: Literal['Urban', 'Semiurban', 'Rural']


# ---------- API ----------
app = FastAPI(title='Loan Approval Predictor')


@app.get('/api/health')
def health():
    return {'status': 'ok'}


@app.post('/api/predict')
def predict(loan: LoanIn, db: Session = Depends(get_db)):
    total = loan.applicant_income + loan.coapplicant_income
    if total <= 0:
        raise HTTPException(status_code=422, detail='Total income must be above 0')

    row = pd.DataFrame([{
        'applicant_income': loan.applicant_income, 'coapplicant_income': loan.coapplicant_income,
        'loan_amount': loan.loan_amount, 'loan_term': loan.loan_term,
        'credit_history': float(loan.credit_history), 'dependents': loan.dependents,
        'education': loan.education, 'self_employed': loan.self_employed,
        'property_area': loan.property_area,
    }])
    p = float(model.predict_proba(row)[0, 1])
    decision = 'Likely approved' if p >= 0.5 else 'Likely rejected'

    # Simple, readable things to review (rules of thumb, not the model's internals)
    reasons = []
    if loan.credit_history == 0:
        reasons.append('No clean credit history on record. Credit history strongly influences approvals in the training data.')
    emi_ratio = (loan.loan_amount * 1000 / loan.loan_term) / total
    if emi_ratio > metrics['emi_ratio_p75']:
        reasons.append('The monthly instalment is high compared with household income.')
    if not reasons:
        reasons.append('No obvious warning signs in the details entered.' if p >= 0.5
                       else 'The overall combination of details is weaker than in most approved applications.')

    db.add(LoanApplication(
        label=loan.label, applicant_income=loan.applicant_income,
        coapplicant_income=loan.coapplicant_income, loan_amount=loan.loan_amount,
        loan_term=loan.loan_term, credit_history=loan.credit_history, dependents=loan.dependents,
        education=loan.education, self_employed=loan.self_employed,
        property_area=loan.property_area, probability=p, decision=decision))
    db.commit()
    return {'probability': round(p, 4), 'decision': decision, 'reasons': reasons}


@app.get('/api/history')
def history(limit: int = 10, db: Session = Depends(get_db)):
    rows = db.query(LoanApplication).order_by(LoanApplication.id.desc()).limit(limit).all()
    return [{'id': r.id, 'time': r.created_at.isoformat(), 'label': r.label,
             'income': r.applicant_income + r.coapplicant_income, 'loan_amount': r.loan_amount,
             'credit_history': r.credit_history, 'probability': round(r.probability, 3),
             'decision': r.decision} for r in rows]


@app.get('/api/stats')
def stats(db: Session = Depends(get_db)):
    counts = dict(db.query(LoanApplication.decision, func.count()).group_by(LoanApplication.decision).all())
    approved, rejected = counts.get('Likely approved', 0), counts.get('Likely rejected', 0)
    return {'total': approved + rejected, 'approved': approved, 'rejected': rejected}


@app.get('/api/metrics')
def get_metrics():
    return metrics


app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')


@app.get('/')
def index():
    return FileResponse(ROOT / 'static' / 'index.html')