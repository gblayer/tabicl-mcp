import pandas as pd
import pytest

from tabicl_mcp import data as D


def test_parse_csv_basic():
    df = D.parse_csv("a,b\n1,x\n2,y\n")
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_parse_csv_bom_and_whitespace():
    df = D.parse_csv("﻿a,b\n1,2\n")
    assert list(df.columns) == ["a", "b"]


def test_parse_csv_empty():
    with pytest.raises(D.DataError):
        D.parse_csv("   ")


def test_parse_csv_single_column_rejected():
    with pytest.raises(D.DataError, match="single column"):
        D.parse_csv("a\n1\n2\n")


def test_parse_csv_row_limit():
    rows = "\n".join(f"{i},{i}" for i in range(30))
    with pytest.raises(D.DataError, match="row limit"):
        D.parse_csv("a,b\n" + rows, max_rows=10)


def test_normalize_google_sheets_url():
    url = "https://docs.google.com/spreadsheets/d/abc123XYZ/edit?usp=sharing#gid=42"
    assert (
        D.normalize_url(url)
        == "https://docs.google.com/spreadsheets/d/abc123XYZ/export?format=csv&gid=42"
    )


def test_normalize_github_blob_url():
    url = "https://github.com/user/repo/blob/main/data/x.csv"
    assert D.normalize_url(url) == "https://raw.githubusercontent.com/user/repo/main/data/x.csv"


def test_normalize_plain_url_untouched():
    assert D.normalize_url("https://example.com/x.csv") == "https://example.com/x.csv"


def test_load_from_path_chat_upload_guidance():
    # A path that only exists in an assistant's sandbox must produce an error
    # that teaches the model to retry with csv_content or url.
    with pytest.raises(D.DataError, match="csv_content"):
        D.load_from_path("/mnt/data/customer_churn.csv")


def test_cache_roundtrip_and_miss():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    ds_id = D.CACHE.put(df, name="t")
    assert D.CACHE.get(ds_id).df.equals(df)
    with pytest.raises(D.DataError, match="not found"):
        D.CACHE.get("ds_nope")


def test_resolve_requires_exactly_one_source():
    with pytest.raises(D.DataError):
        D.resolve("", "")
    with pytest.raises(D.DataError):
        D.resolve("some_id", "a,b\n1,2")


def test_summarize_with_target():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": ["a", "b", "a"]})
    s = D.summarize(df, "y")
    assert s["n_rows"] == 3
    assert s["target"]["suggested_task"] == "classification"
    assert len(s["preview"]) == 3
