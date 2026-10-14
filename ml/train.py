import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml.features import add_features

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'data' / 'raw' / 'loans.csv'
OUT = ROOT / 'app' / 'artifacts'
OUT.mkdir(parents=True, exist_ok=True)

RENAME = {
    'ApplicantIncome': 'applicant_income', 'CoapplicantIncome': 'coapplicant_income',
    'LoanAmount': 'loan_amount', 'Loan_Amount_Term': 'loan_term', 'Credit_History': 'credit_history',
    'Dependents': 'dependents', 'Education': 'education', 'Self_Employed': 'self_employed',
    'Property_Area': 'property_area', 'Loan_Status': 'status',
}
RAW_NUM = ['applicant_income', 'coapplicant_income', 'loan_amount', 'loan_term', 'credit_history']
RAW_CAT = ['dependents', 'education', 'self_employed', 'property_area']
NUM = RAW_NUM + ['total_income', 'emi', 'emi_to_income', 'log_total_income', 'log_loan_amount']
CAT = RAW_CAT  # Gender and Married are deliberately NOT used (fairness)


def load_and_clean():
    df = pd.read_csv(RAW).rename(columns=RENAME).drop_duplicates()
    df = df.dropna(subset=['status'])
    df['target'] = (df['status'] == 'Y').astype(int)  # 1 = approved, 0 = rejected
    return df.reset_index(drop=True)


def build(clf):
    pre = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                          ('sc', StandardScaler())]), NUM),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('oh', OneHotEncoder(handle_unknown='ignore'))]), CAT),
    ])
    return Pipeline([('feat', FunctionTransformer(add_features)), ('pre', pre), ('clf', clf)])


def rate_by(df, col):
    return {str(k): round(float(v), 3) for k, v in df.groupby(col)['target'].mean().items()}


def main():
    df = load_and_clean()
    print('Applications:', len(df), '| approval rate:', round(df['target'].mean(), 3))
    X, y = df[RAW_NUM + RAW_CAT], df['target']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    candidates = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                class_weight='balanced', random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    }
    grids = {
        'Logistic Regression': {'clf__C': [0.1, 1, 10]},
        'Decision Tree': {'clf__max_depth': [3, 4, 5, 6], 'clf__min_samples_leaf': [1, 5, 10]},
        'Random Forest': {'clf__n_estimators': [200, 400], 'clf__min_samples_leaf': [2, 5, 10]},
        'Gradient Boosting': {'clf__n_estimators': [50, 100, 200], 'clf__learning_rate': [0.05, 0.1],
                              'clf__max_depth': [2, 3]},
    }

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    comparison, best_name, best_auc = [], None, 0.0
    for name, clf in candidates.items():
        auc = cross_val_score(build(clf), X_tr, y_tr, cv=cv, scoring='roc_auc').mean()
        comparison.append({'model': name, 'cv_auc': round(float(auc), 4)})
        print(f'{name}: cross-validated ROC-AUC {auc:.3f}')
        if auc > best_auc:
            best_name, best_auc = name, auc

    search = GridSearchCV(build(candidates[best_name]), grids[best_name], cv=cv, scoring='roc_auc', n_jobs=-1)
    search.fit(X_tr, y_tr)
    best = search.best_estimator_

    # Final exam on the hidden 20%
    proba = best.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test = {
        'accuracy': accuracy_score(y_te, pred), 'precision': precision_score(y_te, pred),
        'recall': recall_score(y_te, pred), 'f1': f1_score(y_te, pred), 'roc_auc': roc_auc_score(y_te, proba),
    }
    test = {k: round(float(v), 4) for k, v in test.items()}
    baseline = round(float(max(y_te.mean(), 1 - y_te.mean())), 4)
    print('BEST:', best_name, test, '| always-approve baseline accuracy:', baseline)

    fpr, tpr, _ = roc_curve(y_te, proba)
    keep = np.linspace(0, len(fpr) - 1, min(40, len(fpr))).astype(int)
    imp = permutation_importance(best, X_te, y_te, scoring='roc_auc', n_repeats=10, random_state=42)
    fi = sorted(zip(X.columns, imp.importances_mean), key=lambda t: -t[1])

    feats = add_features(df[RAW_NUM + RAW_CAT])
    df['credit_label'] = df['credit_history'].map({1.0: 'Good history', 0.0: 'Poor history'})
    metrics = {
        'best_model': best_name, 'comparison': comparison, 'test': test, 'baseline_accuracy': baseline,
        'confusion_matrix': confusion_matrix(y_te, pred).tolist(),  # [[TN, FP], [FN, TP]]
        'roc': {'fpr': [round(float(v), 3) for v in fpr[keep]], 'tpr': [round(float(v), 3) for v in tpr[keep]]},
        'importance': [{'feature': f, 'importance': round(float(v), 4)} for f, v in fi],
        'insights': {
            'rows': int(len(df)), 'approval_rate': round(float(df['target'].mean()), 3),
            'by_credit': rate_by(df, 'credit_label'), 'by_area': rate_by(df, 'property_area'),
            'by_education': rate_by(df, 'education'),
        },
        'emi_ratio_p75': round(float(feats['emi_to_income'].quantile(0.75)), 4),
        'options': {c: sorted(df[c].dropna().unique().tolist()) for c in CAT},
    }
    joblib.dump(best, OUT / 'model.joblib')
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print('Saved model and metrics to', OUT)


if __name__ == '__main__':
    main()