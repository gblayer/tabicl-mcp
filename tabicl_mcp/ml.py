"""Modeling layer: task detection, fitting, evaluation, and explanations.

TabICL 2.x exposes sklearn-compatible estimators that accept DataFrames
(including categorical columns) directly; we keep an ordinal-encoding fallback
in case a given version/input trips on non-numeric data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DataError

RANDOM_STATE = 42
# Keep permutation importance affordable on free CPU hardware.
IMPORTANCE_MAX_ROWS = 500
IMPORTANCE_MAX_PASSES = 60


def detect_task(y: pd.Series) -> str:
    if y.dtype == object or y.dtype == bool or str(y.dtype) in ("category", "string", "str"):
        return "classification"
    if y.nunique() <= 20:
        return "classification"
    return "regression"


def resolve_task(y: pd.Series, task_type: str) -> str:
    if task_type in ("classification", "regression"):
        return task_type
    return detect_task(y)


def split_xy(df: pd.DataFrame, target_column: str):
    if target_column not in df.columns:
        raise DataError(
            f"Target column '{target_column}' not found. Available: {list(df.columns)}"
        )
    y = df[target_column]
    X = df.drop(columns=[target_column])
    if y.isna().any():
        keep = y.notna()
        X, y = X[keep], y[keep]
    if len(X) < 10:
        raise DataError(f"Only {len(X)} labeled rows — need at least 10 to fit a model.")
    return X, y


def make_model(task: str):
    from tabicl import TabICLClassifier, TabICLRegressor

    if task == "classification":
        return TabICLClassifier(random_state=RANDOM_STATE)
    return TabICLRegressor(random_state=RANDOM_STATE)


def fit(model, X: pd.DataFrame, y: pd.Series):
    """Fit, falling back to ordinal encoding if the estimator rejects raw dtypes."""
    try:
        model.fit(X, y)
        return model, None
    except (ValueError, TypeError):
        encoder = _OrdinalFallback().fit(X)
        model.fit(encoder.transform(X), y)
        return model, encoder


class _OrdinalFallback:
    def fit(self, X: pd.DataFrame):
        from sklearn.preprocessing import OrdinalEncoder

        self.cat_cols_ = X.select_dtypes(include=["object", "bool", "category", "string"]).columns.tolist()
        self.encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if self.cat_cols_:
            self.encoder_.fit(X[self.cat_cols_].astype(str))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        if self.cat_cols_:
            X[self.cat_cols_] = self.encoder_.transform(X[self.cat_cols_].astype(str))
        return X.apply(pd.to_numeric, errors="coerce").astype("float32").values


def _predict(model, encoder, X: pd.DataFrame):
    return model.predict(encoder.transform(X) if encoder else X)


def _predict_proba(model, encoder, X: pd.DataFrame):
    return model.predict_proba(encoder.transform(X) if encoder else X)


def classification_metrics(y_true, y_pred, proba, classes) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )

    classes = [str(c) for c in classes]
    y_true = pd.Series(y_true).astype(str)
    y_pred = pd.Series(y_pred).astype(str)
    metrics: dict = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }
    # ROC-AUC needs every class present in y_true and proba columns aligned to `classes`.
    present = set(y_true.unique())
    if proba is not None and present == set(classes):
        try:
            if len(classes) == 2:
                auc = roc_auc_score((y_true == classes[1]).astype(int), proba[:, 1])
            else:
                auc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes)
            metrics["roc_auc"] = round(float(auc), 4)
        except ValueError:
            pass
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    metrics["confusion_matrix"] = {"labels": classes, "matrix": cm.tolist()}
    return metrics


def regression_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "target_mean": round(float(y_true.mean()), 4),
        "target_std": round(float(y_true.std()), 4),
    }


def evaluate(df: pd.DataFrame, target_column: str, task_type: str = "auto") -> dict:
    """Honest quality estimate from a single labeled dataset via a held-out split."""
    from sklearn.model_selection import train_test_split

    X, y = split_xy(df, target_column)
    task = resolve_task(y, task_type)

    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    test_size = 0.5 if len(X) < 60 else 0.2
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify
    )
    if task == "classification":
        # Classes absent from the train split can never be predicted; note it.
        missing = set(y.astype(str).unique()) - set(y_tr.astype(str).unique())
    else:
        missing = set()

    model, encoder = fit(make_model(task), X_tr, y_tr)
    y_pred = _predict(model, encoder, X_te)

    result: dict = {
        "task_type": task,
        "target_column": target_column,
        "n_rows_used": int(len(X)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "features": [str(c) for c in X.columns],
        "note": (
            "Metrics come from a held-out split the model never saw during fitting "
            f"({int((1 - test_size) * 100)}/{int(test_size * 100)} train/test)."
        ),
    }
    if task == "classification":
        proba = _predict_proba(model, encoder, X_te)
        result["metrics"] = classification_metrics(y_te, y_pred, proba, model.classes_)
        result["class_distribution"] = {
            str(k): int(v) for k, v in y.astype(str).value_counts().items()
        }
        if missing:
            result["warning"] = (
                f"Classes {sorted(missing)} were too rare to appear in the training split."
            )
    else:
        result["metrics"] = regression_metrics(y_te, y_pred)
    return result


def predict(
    train_df: pd.DataFrame,
    new_df: pd.DataFrame,
    target_column: str,
    task_type: str = "auto",
    include_probabilities: bool = False,
) -> dict:
    X_tr, y_tr = split_xy(train_df, target_column)
    task = resolve_task(y_tr, task_type)

    missing_cols = [c for c in X_tr.columns if c not in new_df.columns]
    if missing_cols:
        raise DataError(
            f"The new data is missing feature columns {missing_cols} that exist in the "
            "training data."
        )
    X_new = new_df[X_tr.columns]

    model, encoder = fit(make_model(task), X_tr, y_tr)
    preds = _predict(model, encoder, X_new)

    result: dict = {
        "task_type": task,
        "target_column": target_column,
        "n_train": int(len(X_tr)),
        "n_predicted": int(len(X_new)),
    }
    if task == "classification":
        result["predictions"] = [str(p) for p in preds]
        proba = _predict_proba(model, encoder, X_new)
        classes = [str(c) for c in model.classes_]
        result["classes"] = classes
        # Confidence = probability of the predicted class; cheap and always useful.
        result["confidence"] = np.round(proba.max(axis=1), 4).tolist()
        if include_probabilities:
            result["probabilities"] = np.round(proba, 4).tolist()
    else:
        result["predictions"] = [round(float(p), 6) for p in preds]

    if target_column in new_df.columns and new_df[target_column].notna().all():
        y_true = new_df[target_column]
        if task == "classification":
            result["metrics"] = classification_metrics(
                y_true, preds, proba, model.classes_
            )
        else:
            result["metrics"] = regression_metrics(y_true, preds)
    return result


def feature_importance(
    df: pd.DataFrame, target_column: str, task_type: str = "auto", max_features: int = 20
) -> dict:
    """Permutation importance on a held-out split (model-agnostic, no extra deps)."""
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    X, y = split_xy(df, target_column)
    task = resolve_task(y, task_type)

    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=stratify
    )
    if len(X_te) > IMPORTANCE_MAX_ROWS:
        X_te = X_te.iloc[:IMPORTANCE_MAX_ROWS]
        y_te = y_te.iloc[:IMPORTANCE_MAX_ROWS]

    model, encoder = fit(make_model(task), X_tr, y_tr)
    n_repeats = max(1, min(5, IMPORTANCE_MAX_PASSES // max(1, len(X.columns))))
    scoring = "balanced_accuracy" if task == "classification" else "r2"

    if encoder:
        # Permute in the encoded numeric space; column order is preserved.
        X_te_enc = pd.DataFrame(encoder.transform(X_te), columns=X_te.columns)
        target = (model, X_te_enc)
    else:
        target = (model, X_te)
    r = permutation_importance(
        target[0], target[1], y_te,
        n_repeats=n_repeats, random_state=RANDOM_STATE, scoring=scoring,
    )

    order = np.argsort(r.importances_mean)[::-1]
    ranked = [
        {
            "feature": str(X.columns[i]),
            "importance": round(float(r.importances_mean[i]), 4),
            "std": round(float(r.importances_std[i]), 4),
        }
        for i in order[:max_features]
    ]
    return {
        "task_type": task,
        "target_column": target_column,
        "method": (
            f"Permutation importance ({scoring} drop when a column is shuffled, "
            f"{n_repeats} repeat(s) on {len(X_te)} held-out rows). Higher = the model "
            "relies on it more; values near 0 = the model barely uses it."
        ),
        "n_features_total": int(len(X.columns)),
        "importances": ranked,
    }
