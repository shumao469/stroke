## Methods text (paste-ready)

Clinical and demographic data were preprocessed using Python (pandas) to handle missing values
(MICE-style multiple imputation using IterativeImputer, 5 iterations) and normalize features
(z-score transformation for numeric predictors including age and stroke onset months). Key predictors
included age, stroke duration (months), pre-FMA, pre-MBI, and stroke type (ischemic/hemorrhagic).
The primary outcome was FMA improvement (ΔFMA = post-FMA − pre-FMA).

A random forest regression model was constructed to predict ΔFMA using scikit-learn.
Hyperparameters were optimized via 5-fold cross-validation (GridSearchCV) over n_estimators (50–200),
max_depth (3–10), and min_samples_split (2–10). The model was trained on 80% of the data (stratified by stroke type)
and validated on the remaining 20%, with performance metrics including R², mean absolute error (MAE),
and root mean squared error (RMSE).
