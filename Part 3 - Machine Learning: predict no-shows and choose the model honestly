# Part 3 — Machine Learning: predict no-shows and choose the model honestly

# Task 3.1 — Features, split, and baseline

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load cleaned data without shuffling or re-sorting
df = pd.read_csv("appointments_clean.csv")

# Build feature matrix X
feature_cols = [
    "age", "distance_km", "fee_inr", "lead_days",
    "prior_no_shows", "reminder_sent", "booking_channel", "visit_type"
]
X = df[feature_cols].copy()
X["reminder_sent"] = X["reminder_sent"].map({"Yes": 1, "No": 0})
X = pd.get_dummies(X, columns=["booking_channel", "visit_type"], dtype=int)

y = df["no_show"]

# Train / test split (80/20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

baseline_acc = (y_test == 0).mean()

print("X shape:", X.shape)
print("X columns:", list(X.columns))
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
print(f"No-shows in test set: {y_test.sum()} / {len(y_test)} ({y_test.mean()*100:.1f}%)")
print(f"Baseline accuracy (always predict Attended): {baseline_acc:.4f} ({(y_test == 0).sum()} of {len(y_test)})")

# Task 3.2 — Logistic regression

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, roc_auc_score
)

# Scale using training set only
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Fit Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_s, y_train)

lr_pred = lr.predict(X_test_s)
lr_prob = lr.predict_proba(X_test_s)[:, 1]

lr_acc = accuracy_score(y_test, lr_pred)
lr_prec = precision_score(y_test, lr_pred)
lr_rec = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_prob)

print(f"Accuracy : {lr_acc:.4f}")
print(f"Precision: {lr_prec:.4f}")
print(f"Recall   : {lr_rec:.4f}")
print(f"F1-score : {lr_f1:.4f}")
print(f"ROC-AUC  : {lr_auc:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, lr_pred))

# Coefficients alongside feature names
coef_df = pd.DataFrame({
    "feature": X.columns,
    "coef": lr.coef_[0],
    "scale": scaler.scale_
})
print("\n--- Logistic Regression Coefficients ---")
print(coef_df.to_string(index=False))

# Odds ratio for reminder_sent (scaled vs. raw 0 -> 1 change)
rem_row = coef_df[coef_df["feature"] == "reminder_sent"].iloc[0]
or_scaled = np.exp(rem_row["coef"])
or_unscaled = np.exp(rem_row["coef"] / rem_row["scale"])

print(f"\nreminder_sent coef           : {rem_row['coef']:.4f} (SD = {rem_row['scale']:.4f})")
print(f"np.exp(coef) [per 1 SD]      : {or_scaled:.4f}")
print(f"np.exp(coef / scale) [No->Yes]: {or_unscaled:.4f} (~{(1 - or_unscaled)*100:.0f}% lower odds)")

print(
    "\nExplanation: np.exp(coef) reflects a one-standard-deviation shift (~0.49) in the scaled "
    "feature, whereas np.exp(coef / scale) reflects the full 1-unit jump from No (0) to Yes (1)."
)
print(
    "Association: In this fitted model, holding the other features constant, appointments with "
    f"a reminder are associated with roughly {(1 - or_unscaled)*100:.0f}% lower odds of a no-show "
    "(observational association, not proof of causation)."
)

# Task 3.3 — Decision tree and overfitting

from sklearn.tree import DecisionTreeClassifier

# Full-depth tree
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)
dt_full_train_acc = accuracy_score(y_train, dt_full.predict(X_train))
dt_full_test_acc = accuracy_score(y_test, dt_full.predict(X_test))

# Depth-3 tree
dt_3 = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_3.fit(X_train, y_train)
dt_3_train_acc = accuracy_score(y_train, dt_3.predict(X_train))
dt_3_test_acc = accuracy_score(y_test, dt_3.predict(X_test))

dt_3_pred = dt_3.predict(X_test)
dt_3_prob = dt_3.predict_proba(X_test)[:, 1]
dt_3_rec = recall_score(y_test, dt_3_pred)
dt_3_auc = roc_auc_score(y_test, dt_3_prob)

print(f"Full-depth Tree (depth={dt_full.get_depth()}): Train Acc = {dt_full_train_acc:.4f} | Test Acc = {dt_full_test_acc:.4f}")
print(f"Depth-3 Tree    (depth={dt_3.get_depth()}) : Train Acc = {dt_3_train_acc:.4f} | Test Acc = {dt_3_test_acc:.4f}")
print(
    "\nInterpretation: The full-depth tree memorised training noise (1.00 train vs. below-baseline "
    "test accuracy), whereas limiting max_depth=3 prevents overfitting and generalizes much better."
)

# Task 3.4 — Random forest and comparison table

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

rf_acc = accuracy_score(y_test, rf_pred)
rf_rec = recall_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_prob)

print(f"Random Forest: Train Acc = {rf_train_acc:.4f} | Test Acc = {rf_acc:.4f} | Recall = {rf_rec:.4f} | ROC-AUC = {rf_auc:.4f}")

# Top 3 feature importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 3 Random Forest Feature Importances:")
print(importances.head(3).round(4))

# Comparison table (for reporting; formal selection happens in Task 3.5)
comparison_df = pd.DataFrame([
    {"Model": "Logistic Regression", "Accuracy": round(lr_acc, 4), "Recall": round(lr_rec, 4), "ROC-AUC": round(lr_auc, 4)},
    {"Model": "Decision Tree (depth=3)", "Accuracy": round(dt_3_test_acc, 4), "Recall": round(dt_3_rec, 4), "ROC-AUC": round(dt_3_auc, 4)},
    {"Model": "Random Forest (100 trees)", "Accuracy": round(rf_acc, 4), "Recall": round(rf_rec, 4), "ROC-AUC": round(rf_auc, 4)},
])
print("\n--- Test-Set Comparison Table (Reporting Only) ---")
print(comparison_df.to_string(index=False))
print(
    "\nObservation: Logistic Regression looks best on the test set because it achieves the highest "
    "ROC-AUC and recall, meaning it ranks risky bookings more reliably than the tree-based models."
)

# Task 3.5 — Tune, choose the model and threshold, then test once

from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline

# (a) Tune Random Forest on training set using 5-fold CV ROC-AUC
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={"n_estimators": [100, 200], "max_depth": [3, 5, 8]},
    cv=5,
    scoring="roc_auc"
)
grid.fit(X_train, y_train)
print("(a) Tuned Random Forest best_params_:", grid.best_params_)
print(f"(a) Tuned Random Forest best_score_ (CV ROC-AUC): {grid.best_score_:.4f}")

# (b) Compare 5-fold CV ROC-AUC on training set ONLY (no test set touched)
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

cv_scores = {
    "Logistic Regression (Pipeline)": cross_val_score(lr_pipe, X_train, y_train, cv=5, scoring="roc_auc").mean(),
    "Decision Tree (max_depth=3)": cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=42), X_train, y_train, cv=5, scoring="roc_auc").mean(),
    "Random Forest (default)": cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X_train, y_train, cv=5, scoring="roc_auc").mean(),
    "Tuned Random Forest": grid.best_score_,
}

print("\n(b) 5-Fold CV ROC-AUC (Training Set Only):")
for name, score in cv_scores.items():
    print(f"  {name:32s}: {score:.4f}")

winner_name = max(cv_scores, key=cv_scores.get)
print(f"--> Selected Winner: {winner_name}")

# (c) Choose threshold using out-of-fold training probabilities
oof_probs = cross_val_predict(lr_pipe, X_train, y_train, cv=5, method="predict_proba")[:, 1]

print("\n(c) Out-of-Fold Threshold Comparison (Training Set):")
thresh_results = {}
for t in [0.5, 0.35]:
    preds = (oof_probs >= t).astype(int)
    prec = precision_score(y_train, preds)
    rec = recall_score(y_train, preds)
    f1 = f1_score(y_train, preds)
    flagged = int(preds.sum())
    thresh_results[t] = f1
    print(f"  Threshold {t:.2f} -> Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | Flagged: {flagged}")

chosen_threshold = max(thresh_results, key=thresh_results.get)
print(
    f"--> Chosen Threshold: {chosen_threshold} (higher F1 and substantially higher recall; "
    "a missed warning costs an empty doctor slot, while an extra reminder call costs a minute)."
)

# (d) Fit winner on full training set and evaluate ONCE on test set
lr_pipe.fit(X_train, y_train)
test_probs = lr_pipe.predict_proba(X_test)[:, 1]
test_flags = (test_probs >= chosen_threshold).astype(int)

test_prec = precision_score(y_test, test_flags)
test_rec = recall_score(y_test, test_flags)
test_auc = roc_auc_score(y_test, test_probs)
test_flagged_count = int(test_flags.sum())

print(f"\n(d) Final Test-Set Evaluation (Threshold = {chosen_threshold}):")
print(f"  Precision : {test_prec:.4f}")
print(f"  Recall    : {test_rec:.4f}")
print(f"  Flagged   : {test_flagged_count} of {len(y_test)}")
print(f"  ROC-AUC   : {test_auc:.4f}")

# (e) Save test_risk_scores.csv
risk_df = pd.DataFrame({
    "appointment_id": df.loc[X_test.index, "appointment_id"].values,
    "no_show": y_test.values,
    "risk_prob": np.round(test_probs, 4),
    "flag": test_flags
})
risk_df.to_csv("test_risk_scores.csv", index=False)
print(f"\n(e) Saved test_risk_scores.csv ({len(risk_df)} rows, {risk_df['flag'].sum()} flagged)")
print(risk_df.head())

print(
    "\nReal-world threshold choice: A clinic would set the threshold by balancing how many "
    "follow-up calls the front desk can realistically make in an evening against the revenue "
    "and patient-access cost of an empty doctor slot."
)
