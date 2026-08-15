# Probability-first model selection and promotion

- [x] Tune every base model by chronological-validation log loss and log validation ROC-AUC alongside it.
- [x] Retain historical test ROC-AUC, accuracy, and Brier tracking; add test log loss and remove PR-AUC, precision, recall, F1, and MCC from promotion metrics.
- [x] Replace the weighted composite promotion rule: promote only when candidate test log loss improves and candidate ROC-AUC is no more than 0.01 below the incumbent. Preserve first-promotion, idempotency, and force behavior.
- [x] Update focused promotion tests and run relevant validation.
