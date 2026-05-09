from __future__ import annotations

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None


DEFAULT_FEATURES = [
    "balls_remaining",
    "overs_remaining",
    "current_run_rate",
    "required_run_rate",
    "wickets_remaining",
    "recent_over_momentum",
    "partnership_strength",
    "pressure_index",
    "venue_advantage",
    "batter_strike_rate",
    "bowler_economy",
]


def _build_preprocessor(frame: pd.DataFrame, feature_columns: list[str]) -> ColumnTransformer:
    numeric_features = [column for column in feature_columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical_features = [column for column in feature_columns if column not in numeric_features]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def _candidate_models(random_state: int = 42) -> dict[str, object]:
    models: dict[str, object] = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
        "random_forest": RandomForestClassifier(n_estimators=160, max_depth=8, min_samples_leaf=3, n_jobs=-1, random_state=random_state, class_weight="balanced_subsample"),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=2,
        )
    return models


def split_training_data(frame: pd.DataFrame, target_column: str = "target_win", group_column: str = "match_id", test_size: float = 0.2, random_state: int = 42):
    feature_frame = frame.copy().dropna(subset=[target_column])
    y = feature_frame[target_column].astype(int)
    X = feature_frame.drop(columns=[target_column])

    if group_column in X.columns and X[group_column].nunique() > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=X[group_column]))
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None)


def train_and_evaluate_models(frame: pd.DataFrame, feature_columns: list[str] | None = None, target_column: str = "target_win", group_column: str = "match_id", random_state: int = 42):
    feature_columns = feature_columns or [column for column in DEFAULT_FEATURES if column in frame.columns]
    if not feature_columns:
        raise ValueError("No feature columns were found for model training.")

    model_frame = frame.dropna(subset=[target_column]).copy()
    required_columns = feature_columns + [target_column]
    if group_column in model_frame.columns:
        required_columns.append(group_column)
    model_frame = model_frame[required_columns].copy()
    X_train, X_test, y_train, y_test = split_training_data(model_frame, target_column=target_column, group_column=group_column, random_state=random_state)

    preprocessor = _build_preprocessor(X_train, feature_columns)
    results = []
    fitted_models: dict[str, Pipeline] = {}

    for model_name, estimator in _candidate_models(random_state=random_state).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train[feature_columns], y_train)
        probabilities = pipeline.predict_proba(X_test[feature_columns])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = {
            "model": model_name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities) if y_test.nunique() > 1 else 0.0,
        }
        results.append(metrics)
        fitted_models[model_name] = pipeline

    scores = pd.DataFrame(results).sort_values(["roc_auc", "accuracy"], ascending=False).reset_index(drop=True)
    best_model_name = scores.iloc[0]["model"]
    return scores, fitted_models, str(best_model_name)


def refit_best_model(frame: pd.DataFrame, feature_columns: list[str] | None = None, target_column: str = "target_win", random_state: int = 42):
    feature_columns = feature_columns or [column for column in DEFAULT_FEATURES if column in frame.columns]
    if not feature_columns:
        raise ValueError("No feature columns were found for model fitting.")

    training_frame = frame.dropna(subset=[target_column]).copy()
    X = training_frame[feature_columns]
    y = training_frame[target_column].astype(int)
    preprocessor = _build_preprocessor(training_frame, feature_columns)
    best_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", _candidate_models(random_state=random_state)["logistic_regression"]),
        ]
    )
    best_pipeline.fit(X, y)
    return best_pipeline, feature_columns


def save_model_bundle(model: Pipeline, feature_columns: list[str], output_path: str) -> None:
    joblib.dump({"model": model, "feature_columns": feature_columns}, output_path)


def load_model_bundle(model_path: str):
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["feature_columns"]


def rebuild_pipeline_from_frame(estimator: object, frame: pd.DataFrame, feature_columns: list[str]) -> Pipeline:
    """Rebuild a sklearn Pipeline by creating a fresh preprocessor from `frame` and attaching `estimator`.

    Use this when a persisted pipeline is incompatible with the current sklearn version
    (for example due to internal attribute changes on transformers like SimpleImputer).
    """
    feature_columns = [c for c in feature_columns if c in frame.columns]
    preprocessor = _build_preprocessor(frame, feature_columns)
    # Fit the preprocessor on available data so it can be used with a pre-fitted estimator
    preprocessor.fit(frame[feature_columns])
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
    return pipeline
