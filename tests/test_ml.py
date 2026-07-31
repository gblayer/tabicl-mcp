import numpy as np
import pandas as pd
import pytest

from tabicl_mcp import ml
from tabicl_mcp.data import DataError

rng = np.random.default_rng(7)
N = 120


@pytest.fixture(scope="module")
def clf_df():
    x1 = rng.normal(0, 1, N)
    x2 = rng.normal(0, 1, N)
    color = rng.choice(["red", "blue"], N)
    label = np.where(x1 + (color == "red") * 2 + rng.normal(0, 0.3, N) > 1, "yes", "no")
    return pd.DataFrame({"x1": x1, "x2": x2, "color": color, "label": label})


@pytest.fixture(scope="module")
def reg_df():
    x1 = rng.normal(0, 1, N)
    x2 = rng.normal(0, 1, N)
    y = 3 * x1 + x2 + rng.normal(0, 0.1, N)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def test_detect_task():
    assert ml.detect_task(pd.Series(["a", "b", "a"])) == "classification"
    assert ml.detect_task(pd.Series([True, False])) == "classification"
    assert ml.detect_task(pd.Series([1, 2, 3, 1, 2])) == "classification"
    assert ml.detect_task(pd.Series(np.linspace(0, 1, 100))) == "regression"


def test_split_xy_validations(clf_df):
    with pytest.raises(DataError, match="not found"):
        ml.split_xy(clf_df, "nope")
    with pytest.raises(DataError, match="at least 10"):
        ml.split_xy(clf_df.head(5), "label")


def test_split_xy_drops_unlabeled_rows(clf_df):
    df = clf_df.copy()
    df.loc[df.index[:10], "label"] = None
    X, y = ml.split_xy(df, "label")
    assert len(X) == N - 10 and y.notna().all()


@pytest.mark.slow
def test_evaluate_classification(clf_df):
    result = ml.evaluate(clf_df, "label")
    assert result["task_type"] == "classification"
    m = result["metrics"]
    assert m["accuracy"] > 0.8
    assert "roc_auc" in m
    assert m["confusion_matrix"]["labels"] == ["no", "yes"]
    assert result["n_train"] + result["n_test"] == N


@pytest.mark.slow
def test_evaluate_regression(reg_df):
    result = ml.evaluate(reg_df, "y")
    assert result["task_type"] == "regression"
    assert result["metrics"]["r2"] > 0.9


@pytest.mark.slow
def test_predict_with_and_without_labels(clf_df):
    train, new = clf_df.iloc[:100], clf_df.iloc[100:]
    result = ml.predict(train, new, "label")
    assert len(result["predictions"]) == len(new)
    assert len(result["confidence"]) == len(new)
    assert "metrics" in result  # new data included labels

    result2 = ml.predict(train, new.drop(columns=["label"]), "label")
    assert "metrics" not in result2


@pytest.mark.slow
def test_predict_missing_feature_column(clf_df):
    with pytest.raises(DataError, match="missing feature columns"):
        ml.predict(clf_df, clf_df.drop(columns=["x1", "label"]), "label")


@pytest.mark.slow
def test_feature_importance_ranks_signal_first(clf_df):
    result = ml.feature_importance(clf_df, "label")
    features = [i["feature"] for i in result["importances"]]
    assert set(features) == {"x1", "x2", "color"}
    # x2 is pure noise — it must not outrank the true signal x1
    assert features.index("x1") < features.index("x2")
