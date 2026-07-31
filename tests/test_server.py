import numpy as np
import pandas as pd
import pytest

from tabicl_mcp import server
from tabicl_mcp.report import build_report

CSV = "x1,x2,label\n" + "\n".join(
    f"{i % 10},{(i * 7) % 5},{'yes' if i % 10 > 4 else 'no'}" for i in range(60)
)


def test_load_data_inline():
    result = server.load_data(csv_content=CSV, target_column="label")
    assert result["dataset_id"].startswith("ds_")
    assert result["n_rows"] == 60
    assert result["target"]["suggested_task"] == "classification"


def test_load_data_requires_one_source():
    assert "error" in server.load_data()
    assert "error" in server.load_data(csv_content=CSV, url="https://x.com/a.csv")


def test_load_data_bad_target_warns():
    result = server.load_data(csv_content=CSV, target_column="nope")
    assert "target_warning" in result


def test_evaluate_unknown_dataset_id_is_friendly():
    result = server.evaluate(target_column="label", dataset_id="ds_missing")
    assert "not found" in result["error"]


def test_export_predictions_pages():
    df = pd.DataFrame({"a": range(10), "p": range(10)})
    result_id = server.D.CACHE.put(df, prefix="pred")
    page = server.export_predictions(result_id, offset=8, limit=5)
    assert page["total_rows"] == 10
    assert page["returned_rows"] == 2
    assert "csv" in page


@pytest.mark.slow
def test_full_flow_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "REPORTS_DIR", str(tmp_path))
    loaded = server.load_data(csv_content=CSV)
    ds = loaded["dataset_id"]

    evaluation = server.evaluate(target_column="label", dataset_id=ds)
    assert "metrics" in evaluation, evaluation

    report = server.create_report(target_column="label", dataset_id=ds)
    assert "report_file" in report, report
    html = open(report["report_file"], encoding="utf-8").read()
    assert "<svg" in html and "Accuracy" in html


def test_upload_page_roundtrip():
    from starlette.testclient import TestClient

    app = server.mcp.streamable_http_app()
    with TestClient(app) as client:
        assert "Upload a CSV" in client.get("/").text
        resp = client.post("/upload", files={"file": ("t.csv", b"a,b\n1,2\n3,4\n", "text/csv")})
        assert "ds_" in resp.text
        assert "✔ Uploaded" in resp.text and "2 rows × 2 columns" in resp.text
        ds_id = resp.text.split("<code>")[1].split("</code>")[0]
        assert server.D.CACHE.get(ds_id).df.shape == (2, 2)

        bad = client.post("/upload", files={"file": ("t.csv", b"", "text/csv")})
        assert "✘" in bad.text


def test_build_report_smoke_without_model():
    evaluation = {
        "task_type": "classification",
        "target_column": "label",
        "note": "test",
        "metrics": {
            "accuracy": 0.9,
            "balanced_accuracy": 0.88,
            "f1_macro": 0.89,
            "roc_auc": 0.95,
            "confusion_matrix": {"labels": ["no", "yes"], "matrix": [[40, 5], [4, 51]]},
        },
        "class_distribution": {"no": 45, "yes": 55},
    }
    importance = {
        "importances": [
            {"feature": "income", "importance": 0.31, "std": 0.02},
            {"feature": "age", "importance": 0.12, "std": 0.01},
        ]
    }
    html = build_report(
        title="Test report",
        dataset_summary={"n_rows": 100, "n_columns": 3},
        evaluation=evaluation,
        importance=importance,
    )
    assert "Test report" in html
    assert "income" in html
    assert "pred: yes" in html
    assert "<script" not in html  # self-contained, no JS
