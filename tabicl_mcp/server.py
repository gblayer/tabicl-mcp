"""
TabICL MCP Server
=================
A Model Context Protocol server that exposes TabICL (soda-inria/tabicl)
for in-context tabular ML — classification and regression — directly inside
LLM chat interfaces and agentic IDEs.

Supports both stdio (local) and HTTP/SSE (remote) transports.
"""

import io
import json
import sys
from typing import Literal

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
# Server declaration
# --------------------------------------------------------------------------- #
mcp = FastMCP(
    name="TabICL",
    instructions=(
        "You have access to TabICL, a state-of-the-art in-context learning model "
        "for tabular data. It can perform classification and regression on CSV data "
        "without any hyperparameter tuning — just pass train and test data.\n\n"
        "Workflow for the user:\n"
        "1. Ask the user to paste or share their CSV data (train split and test split).\n"
        "2. Ask which column is the target.\n"
        "3. Detect whether it's classification or regression (or ask).\n"
        "4. Call the appropriate tool and return predictions + metric.\n\n"
        "TabICL works best on datasets with < 10 000 rows and < 500 features."
    ),
)

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _parse_csv(raw: str) -> pd.DataFrame:
    """Parse a CSV string into a DataFrame, stripping BOM if present."""
    raw = raw.lstrip("\ufeff").strip()
    return pd.read_csv(io.StringIO(raw))


def _encode_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Ordinal-encode object/bool/category columns in place.
    Unknown categories in test are mapped to the closest known value.
    Returns (X_train_np, X_test_np).
    """
    from sklearn.preprocessing import OrdinalEncoder
    cat_cols = X_train.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    X_tr = X_train.copy()
    X_te = X_test.copy()
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols].astype(str))
        X_te[cat_cols] = enc.transform(X_te[cat_cols].astype(str))
    # Cast everything to float32 for TabICL
    return X_tr.astype("float32").values, X_te.astype("float32").values


def _detect_task(y: pd.Series) -> Literal["classification", "regression"]:
    """Heuristic: if target is string/bool or has ≤ 20 unique values → classification."""
    if y.dtype == object or y.dtype == bool or str(y.dtype) == "category":
        return "classification"
    if y.nunique() <= 20:
        return "classification"
    return "regression"


def _format_result(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Tool 1 — auto-detect task type
# --------------------------------------------------------------------------- #

@mcp.tool()
def predict(
    train_csv: str,
    test_csv: str,
    target_column: str,
    task_type: str = "auto",
) -> str:
    """
    Fit TabICL on the training data and predict on the test data.

    Args:
        train_csv:     Full CSV content of the training set (with header row).
                       Must contain the target column.
        test_csv:      Full CSV content of the test set (with header row).
                       If it contains the target column, an evaluation metric is computed.
                       If not, predictions are returned without a metric.
        target_column: Name of the column to predict.
        task_type:     "classification", "regression", or "auto" (default).
                       "auto" detects the task from the target column's dtype and cardinality.

    Returns:
        JSON string with:
          - task_type       : "classification" or "regression"
          - predictions     : list of predicted values (labels for classification, floats for regression)
          - predict_proba   : (classification only) list of class probability arrays
          - classes         : (classification only) list of class labels in order
          - metric          : {"ROC_AUC": float} for classification, {"R2": float} for regression
                              (only present when the test CSV includes the target column)
          - n_train / n_test: row counts
          - features        : list of feature column names used
    """
    try:
        from sklearn.metrics import r2_score, roc_auc_score
        from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
        from tabicl import TabICLClassifier, TabICLRegressor
    except ImportError:
        return _format_result({
            "error": "TabICL is not installed. Run: pip install tabicl"
        })

    try:
        train_df = _parse_csv(train_csv)
        test_df  = _parse_csv(test_csv)

        if target_column not in train_df.columns:
            return _format_result({
                "error": f"Target column '{target_column}' not found in train CSV. "
                         f"Available columns: {train_df.columns.tolist()}"
            })

        # Split features / target
        feature_cols = [c for c in train_df.columns if c != target_column]
        X_train_df   = train_df[feature_cols]
        y_train      = train_df[target_column]

        has_labels   = target_column in test_df.columns
        X_test_df    = test_df[feature_cols] if has_labels else test_df[feature_cols]
        y_test       = test_df[target_column] if has_labels else None

        X_train, X_test = _encode_features(X_train_df, X_test_df)

        # Auto-detect
        resolved_task = task_type if task_type in ("classification", "regression") \
                        else _detect_task(y_train)

        result: dict = {
            "task_type": resolved_task,
            "n_train":   len(X_train),
            "n_test":    len(X_test),
            "features":  feature_cols,
        }

        # ------------------------------------------------------------------ #
        # Classification
        # ------------------------------------------------------------------ #
        if resolved_task == "classification":
            le = LabelEncoder()
            y_enc = le.fit_transform(y_train.astype(str))
            classes = le.classes_.tolist()

            model = TabICLClassifier()
            model.fit(X_train, y_enc)

            preds_enc = model.predict(X_test)
            proba     = model.predict_proba(X_test)

            predictions = le.inverse_transform(preds_enc).tolist()

            result["predictions"]   = predictions
            result["predict_proba"] = proba.tolist()
            result["classes"]       = classes

            if y_test is not None:
                y_test_enc = le.transform(y_test.astype(str))
                if len(classes) == 2:
                    auc = roc_auc_score(y_test_enc, proba[:, 1])
                else:
                    auc = roc_auc_score(
                        y_test_enc, proba, multi_class="ovr", average="macro"
                    )
                result["metric"] = {"ROC_AUC": round(float(auc), 4)}

        # ------------------------------------------------------------------ #
        # Regression
        # ------------------------------------------------------------------ #
        else:
            model = TabICLRegressor()
            model.fit(X_train, y_train.astype("float32").values)

            preds = model.predict(X_test)
            result["predictions"] = [round(float(p), 6) for p in preds]

            if y_test is not None:
                r2 = r2_score(y_test.astype(float), preds)
                result["metric"] = {"R2": round(float(r2), 4)}

        return _format_result(result)

    except Exception as exc:
        return _format_result({"error": str(exc)})


# --------------------------------------------------------------------------- #
# Tool 2 — inspect a CSV before predicting
# --------------------------------------------------------------------------- #

@mcp.tool()
def inspect_csv(csv_content: str, target_column: str = "") -> str:
    """
    Summarise a CSV: shape, column types, missing values, and target distribution.
    Use this before calling predict to understand the data and confirm the task type.

    Args:
        csv_content:   Full CSV string.
        target_column: Optional. If provided, also prints the target's distribution
                       and suggests a task type.

    Returns:
        JSON summary of the dataset.
    """
    try:
        df = _parse_csv(csv_content)
        summary: dict = {
            "n_rows":    len(df),
            "n_cols":    len(df.columns),
            "columns":   df.columns.tolist(),
            "dtypes":    df.dtypes.astype(str).to_dict(),
            "missing":   df.isnull().sum().to_dict(),
        }
        if target_column and target_column in df.columns:
            y = df[target_column]
            summary["target"] = {
                "column":           target_column,
                "dtype":            str(y.dtype),
                "n_unique":         int(y.nunique()),
                "sample_values":    y.dropna().unique()[:10].tolist(),
                "suggested_task":   _detect_task(y),
            }
        elif target_column:
            summary["target_warning"] = (
                f"Column '{target_column}' not found. "
                f"Available: {df.columns.tolist()}"
            )
        return _format_result(summary)
    except Exception as exc:
        return _format_result({"error": str(exc)})


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #

def serve_stdio():
    """Run as a local stdio MCP server (for Claude Desktop, Cursor, VS Code)."""
    mcp.run(transport="stdio")


def serve_http(host: str = "0.0.0.0", port: int = None):
    """Run as an HTTP/SSE MCP server (for Claude.ai remote integrations, HF Spaces)."""
    import os
    import uvicorn
    if port is None:
        port = int(os.environ.get("PORT", 7860))
    uvicorn.run(mcp.sse_app(), host=host, port=port)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if mode == "http":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        serve_http(port=port)
    else:
        serve_stdio()
