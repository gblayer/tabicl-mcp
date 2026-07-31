"""Data ingestion and caching for the TabICL MCP server.

Datasets enter through one of three routes (pasted CSV text, a URL, or a local
file path) and are cached in memory under a short ``dataset_id`` so follow-up
tool calls (evaluate / predict / explain / report) never need the data to be
re-transferred through the model's context.
"""

from __future__ import annotations

import hashlib
import io
import re
import threading
import time
from dataclasses import dataclass, field

import pandas as pd

# Inline CSV rides through the LLM context, so it is the tightest budget.
MAX_INLINE_ROWS = 10_000
# URL / file ingestion bypasses the context, so the cap is the model's comfort zone.
MAX_FETCH_BYTES = 50 * 1024 * 1024
MAX_ROWS = 50_000

CACHE_TTL_SECONDS = 4 * 60 * 60
CACHE_MAX_ENTRIES = 32

_GSHEET_RE = re.compile(r"https://docs\.google\.com/spreadsheets/d/([\w-]+)")


class DataError(ValueError):
    """User-facing data problem — the message is meant to be shown as-is."""


@dataclass
class CachedDataset:
    df: pd.DataFrame
    name: str
    created_at: float = field(default_factory=time.time)


class DatasetCache:
    def __init__(self, max_entries: int = CACHE_MAX_ENTRIES, ttl: float = CACHE_TTL_SECONDS):
        self._lock = threading.Lock()
        self._entries: dict[str, CachedDataset] = {}
        self._max_entries = max_entries
        self._ttl = ttl

    def put(self, df: pd.DataFrame, name: str = "", prefix: str = "ds") -> str:
        digest = hashlib.sha1(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        ).hexdigest()[:8]
        dataset_id = f"{prefix}_{digest}"
        with self._lock:
            self._evict_locked()
            self._entries[dataset_id] = CachedDataset(df=df, name=name or dataset_id)
        return dataset_id

    def get(self, dataset_id: str) -> CachedDataset:
        with self._lock:
            self._evict_locked()
            entry = self._entries.get(dataset_id)
            if entry is None:
                raise DataError(
                    f"Dataset '{dataset_id}' not found (the cache may have expired — "
                    f"it keeps data for {CACHE_TTL_SECONDS // 3600}h). "
                    "Load the data again with load_data."
                )
            return entry

    def _evict_locked(self) -> None:
        now = time.time()
        expired = [k for k, v in self._entries.items() if now - v.created_at > self._ttl]
        for k in expired:
            del self._entries[k]
        while len(self._entries) >= self._max_entries:
            oldest = min(self._entries, key=lambda k: self._entries[k].created_at)
            del self._entries[oldest]


CACHE = DatasetCache()


def parse_csv(raw: str, max_rows: int = MAX_INLINE_ROWS) -> pd.DataFrame:
    raw = raw.lstrip("﻿").strip()
    if not raw:
        raise DataError("The CSV content is empty.")
    try:
        df = pd.read_csv(io.StringIO(raw))
    except Exception as exc:
        raise DataError(f"Could not parse the CSV: {exc}") from exc
    _check_shape(df, max_rows)
    return df


def normalize_url(url: str) -> str:
    """Turn share links into direct-download CSV links where we know how."""
    m = _GSHEET_RE.match(url)
    if m:
        gid_match = re.search(r"[#?&]gid=(\d+)", url)
        gid = f"&gid={gid_match.group(1)}" if gid_match else ""
        return (
            f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv{gid}"
        )
    # GitHub blob pages -> raw content
    gh = re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/(.+)", url)
    if gh:
        return f"https://raw.githubusercontent.com/{gh.group(1)}/{gh.group(2)}/{gh.group(3)}"
    return url


def load_from_url(url: str) -> pd.DataFrame:
    import httpx

    if not url.lower().startswith(("http://", "https://")):
        raise DataError("Only http(s) URLs are supported.")
    url = normalize_url(url)
    try:
        with httpx.Client(follow_redirects=True, timeout=60) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                declared = resp.headers.get("content-length")
                if declared and int(declared) > MAX_FETCH_BYTES:
                    raise DataError(
                        f"File is larger than the {MAX_FETCH_BYTES // 1024 // 1024} MB limit."
                    )
                chunks: list[bytes] = []
                size = 0
                for chunk in resp.iter_bytes():
                    size += len(chunk)
                    if size > MAX_FETCH_BYTES:
                        raise DataError(
                            f"File is larger than the {MAX_FETCH_BYTES // 1024 // 1024} MB limit."
                        )
                    chunks.append(chunk)
    except DataError:
        raise
    except Exception as exc:
        raise DataError(
            f"Could not download the file from {url}: {exc}. "
            "If this is a Google Sheet, make sure link sharing is set to "
            "'Anyone with the link can view'."
        ) from exc
    text = b"".join(chunks).decode("utf-8-sig", errors="replace")
    return parse_csv(text, max_rows=MAX_ROWS)


def load_from_path(path: str) -> pd.DataFrame:
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            text = fh.read(MAX_FETCH_BYTES + 1)
    except OSError as exc:
        # The most common mistake: passing a chat-upload path (e.g. ChatGPT's
        # /mnt/data/...) that only exists in the assistant's sandbox, not here.
        raise DataError(
            f"Could not read file '{path}' — this MCP server runs on its own machine "
            "and cannot see files uploaded to the chat. Instead: read the uploaded "
            "file in your environment and pass its text via csv_content, or provide "
            "a public URL via url. file_path only works when the user runs this "
            "server locally on their own computer."
        ) from exc
    if len(text) > MAX_FETCH_BYTES:
        raise DataError(f"File is larger than the {MAX_FETCH_BYTES // 1024 // 1024} MB limit.")
    return parse_csv(text, max_rows=MAX_ROWS)


def resolve(dataset_id: str = "", csv_content: str = "") -> pd.DataFrame:
    """Resolve a tool's data argument: cached id first, inline CSV second."""
    if dataset_id and csv_content:
        raise DataError("Pass either dataset_id or csv_content, not both.")
    if dataset_id:
        return CACHE.get(dataset_id).df
    if csv_content:
        return parse_csv(csv_content)
    raise DataError("No data given — pass a dataset_id (from load_data) or csv_content.")


def summarize(df: pd.DataFrame, target_column: str = "") -> dict:
    summary: dict = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": [
            {
                "name": str(c),
                "dtype": str(df[c].dtype),
                "n_missing": int(df[c].isna().sum()),
                "n_unique": int(df[c].nunique()),
            }
            for c in df.columns
        ],
        "preview": df.head(5).to_dict(orient="records"),
    }
    if target_column:
        if target_column not in df.columns:
            summary["target_warning"] = (
                f"Column '{target_column}' not found. Available: {list(df.columns)}"
            )
        else:
            from .ml import detect_task

            y = df[target_column]
            summary["target"] = {
                "column": target_column,
                "dtype": str(y.dtype),
                "n_unique": int(y.nunique()),
                "sample_values": [str(v) for v in y.dropna().unique()[:10]],
                "suggested_task": detect_task(y),
            }
    return summary


def _check_shape(df: pd.DataFrame, max_rows: int) -> None:
    if len(df) == 0:
        raise DataError("The CSV parsed to zero rows.")
    if len(df) > max_rows:
        raise DataError(
            f"This dataset has {len(df):,} rows, above the {max_rows:,}-row limit for "
            "this route. Options: share it as a URL (load_data with url=...), sample it, "
            "or run the server locally where file paths are supported."
        )
    if len(df.columns) < 2:
        raise DataError(
            "The CSV parsed to a single column — check the delimiter (commas expected)."
        )
