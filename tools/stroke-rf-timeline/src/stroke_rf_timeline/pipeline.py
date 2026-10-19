from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer

import joblib

@dataclass
class RFConfig:
    random_state: int = 42
    test_size: float = 0.2
    mice_max_iter: int = 5  # IterativeImputer (MICE-like)
    n_estimators: Tuple[int, ...] = (50, 100, 150, 200)
    max_depth: Tuple[int, ...] = (3, 4, 5, 6, 7, 8, 9, 10)
    min_samples_split: Tuple[int, ...] = (2, 4, 6, 8, 10)
    cv: int = 5
    n_jobs: int = -1

def build_pipeline(
    numeric_features=("age", "stroke_onset_months", "pre_fma", "pre_mbi"),
    categorical_features=("stroke_type",),
    config: Optional[RFConfig] = None,
) -> Pipeline:
    """Preprocess (MICE-like imputation + z-score + one-hot) then RandomForestRegressor."""
    if config is None:
        config = RFConfig()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", IterativeImputer(max_iter=config.mice_max_iter, random_state=config.random_state)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_features)),
            ("cat", categorical_transformer, list(categorical_features)),
        ],
        remainder="drop",
    )

    rf = RandomForestRegressor(random_state=config.random_state)
    return Pipeline(steps=[("preprocess", preprocessor), ("rf", rf)])

def train_validate(
    df: pd.DataFrame,
    outcome_col: str = "delta_fma",
    config: Optional[RFConfig] = None,
    stratify_col: str = "stroke_type",
    numeric_features=("age", "stroke_onset_months", "pre_fma", "pre_mbi"),
    categorical_features=("stroke_type",),
) -> Tuple[GridSearchCV, Dict[str, float], pd.DataFrame]:
    """Train RF with 5-fold GridSearchCV and evaluate on 20% held-out set (stratified)."""
    if config is None:
        config = RFConfig()

    if outcome_col not in df.columns:
        if ("post_fma" in df.columns) and ("pre_fma" in df.columns):
            df = df.copy()
            df[outcome_col] = df["post_fma"] - df["pre_fma"]
        else:
            raise ValueError(
                f"Outcome column '{outcome_col}' not found and cannot be derived. "
                "Need (post_fma, pre_fma) or provide delta_fma."
            )

    X = df[list(numeric_features) + list(categorical_features)].copy()
    y = df[outcome_col].astype(float).copy()
    stratify = df[stratify_col] if (stratify_col in df.columns) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=stratify
    )

    pipe = build_pipeline(numeric_features=numeric_features, categorical_features=categorical_features, config=config)

    param_grid = {
        "rf__n_estimators": list(config.n_estimators),
        "rf__max_depth": list(config.max_depth),
        "rf__min_samples_split": list(config.min_samples_split),
    }

    gs = GridSearchCV(
        pipe, param_grid=param_grid, cv=config.cv, n_jobs=config.n_jobs, scoring="r2", refit=True
    )
    gs.fit(X_train, y_train)

    y_pred = gs.predict(X_test)
    metrics = evaluate_model(y_test.to_numpy(), y_pred)
    preds_df = pd.DataFrame({"y_true": y_test.to_numpy(), "y_pred": y_pred})
    return gs, metrics, preds_df

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"r2": r2, "mae": mae, "rmse": rmse}

def save_model(model: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path: str | Path) -> Any:
    return joblib.load(str(path))
