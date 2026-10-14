import numpy as np


def add_features(df):
    """Adds new columns computed from the raw application details."""
    df = df.copy()
    df['total_income'] = df['applicant_income'] + df['coapplicant_income']
    df['emi'] = df['loan_amount'] * 1000 / df['loan_term']  # rough monthly instalment
    df['emi_to_income'] = df['emi'] / df['total_income'].replace(0, np.nan)
    df['log_total_income'] = np.log1p(df['total_income'])
    df['log_loan_amount'] = np.log1p(df['loan_amount'])
    return df