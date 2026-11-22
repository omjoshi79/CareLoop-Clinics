# CareLoop-Clinics
CareLoop-Clinics-No-Show-Guard

# CareLoop Clinics — No-Show Guard

## 4-Line Summary
1. **Part 1 (Data Foundations):** Cleaned 1,020 raw bookings down to 990 valid appointments (218 no-shows, **22.0%** overall rate) after removing 20 duplicates, 10 unreadable/blank records, and capping 26 distance and 3 fee outliers.
2. **Part 2 (SQL Verification):** No-shows rise sharply with booking lead time—from **16.2%** (1–7 days) to **60.5%** (31+ days)—and cluster most heavily at Gomti Nagar Care Point (**33.3%**, 90 bookings) and Hadapsar Care Centre (**27.3%**, 143 bookings).
3. **Part 3 (Machine Learning):** Logistic Regression achieved the highest 5-fold cross-validated ROC-AUC (**0.77**) and test ROC-AUC (**0.79**); appointments with a reminder are associated with **~69% lower odds** of a no-show (odds ratio **0.31**).
4. **Part 4 (GenAI Follow-Up Review):** Structured extraction on 10 call-centre notes (9 non-blank, 1 skipped blank) showed **2 of 2** `forgot_or_no_reminder` cases occurred on appointments where `reminder_sent = No`.

## Business Recommendation
CareLoop Clinics should immediately prioritize evening-before reminder calls for high-risk bookings flagged by the Logistic Regression model at the **0.35** probability threshold, which captures **~57%** of actual no-shows while flagging roughly **49** of every 198 bookings (~25%). Because bookings made **31+ days** ahead fail **60.5%** of the time and appointments with reminders show **~69% lower no-show odds** (reinforced by **100%** of `forgot_or_no_reminder` notes lacking a reminder), automating reminders and targeting manual calls at high-risk, long-lead bookings—starting with high-rate clinics like **Hadapsar Care Centre (27.3%)** and **Gomti Nagar Care Point (33.3%)**—is the highest-leverage first intervention.

## Model Choice
- **Selected Model:** Logistic Regression (with `StandardScaler` inside a 5-fold CV `Pipeline`).
- **Why:** It outperformed the depth-3 Decision Tree (CV AUC ~0.68), default Random Forest (CV AUC ~0.75), and tuned Random Forest (CV AUC ~0.77) on training-only 5-fold CV ROC-AUC, and achieved a test-set ROC-AUC of **~0.79**.
- **Chosen Threshold:** **0.35** (selected on out-of-fold training probabilities for higher F1 **~0.49** vs **~0.35** at 0.50, prioritizing recall because a missed no-show wastes a doctor slot while an extra call costs a minute).

## Limitations
1. **Small GenAI sample:** The 10 analysed call-centre notes are an illustration and cannot represent clinic-wide no-show reasons at scale.
2. **Pre-split cleaning & random split:** Medians, quartiles, and outlier fences were computed on all 990 rows before the train/test split, and the random split allows the same patient to appear in both train and test rather than evaluating strictly past-to-future.
3. **Observational data:** The strong negative association between `reminder_sent` and no-shows is observational and does not prove causation; additionally, the operational threshold should ultimately be calibrated to each clinic's evening call capacity.
