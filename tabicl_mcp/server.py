"""TabICL MCP Server.

Exposes TabICL (soda-inria/tabicl) — a tabular foundation model for
classification and regression — as MCP tools designed for non-technical users:
load data (paste / URL / file), evaluate, predict, explain, and generate a
shareable HTML report.

Transports: stdio (Claude Desktop, Cursor, VS Code) and Streamable HTTP
(claude.ai, ChatGPT, Gemini — one URL, e.g. hosted on a free HuggingFace Space).
"""

from __future__ import annotations

import os
import time
import uuid

from mcp.server.mcpserver import MCPServer

from . import data as D
from .data import DataError

INSTRUCTIONS = """\
You have TabICL, a state-of-the-art tabular foundation model (no hyperparameter
tuning needed) for classification and regression on CSV data.

Typical flows — prefer dataset_ids over re-pasting CSV text:
1. User shares data (pasted, a URL like a Google Sheet, or a file path when the
   server runs locally) -> call load_data once, reuse the returned dataset_id.
2. "Can you predict X from this?" / "how accurate would it be?" -> evaluate
   (single labeled dataset; it does an honest train/test split internally).
3. "Predict for these new rows" -> predict (labeled data + new rows).
4. "Why? What matters most?" -> explain (permutation feature importance).
5. "Make me a report/dashboard" -> create_report (returns a self-contained HTML
   report: metrics, confusion matrix, feature importance, distributions).

Guidance:
- If the user has one CSV and no separate test set, use evaluate — never report
  training-set accuracy as model quality.
- Ask which column is the target if it isn't obvious; task type is auto-detected.
- Data limits: ~10k rows pasted inline, ~50k via URL/file. Larger: ask the user
  to sample or share a URL.
- First model call downloads/loads checkpoints and can take a minute; later
  calls are much faster. On free hosting, a cold server adds startup time too.
"""

mcp = MCPServer(
    name="TabICL",
    instructions=INSTRUCTIONS,
    website_url="https://github.com/gblayer/tabicl-mcp",
)

# Reports are kept on disk so the HTTP deployment can serve them as shareable
# links; over stdio the file path itself is the deliverable.
REPORTS_DIR = os.environ.get(
    "TABICL_MCP_REPORTS_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "tabicl-mcp", "reports"),
)
_PUBLIC_URL = os.environ.get("TABICL_MCP_PUBLIC_URL") or (
    f"https://{os.environ['SPACE_HOST']}" if os.environ.get("SPACE_HOST") else ""
)
_TRANSPORT = "stdio"  # set by the entry points; controls how reports are returned

MAX_INLINE_PREDICTIONS = 2000


def _err(exc: Exception) -> dict:
    if isinstance(exc, DataError):
        return {"error": str(exc)}
    return {"error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


@mcp.tool()
def load_data(
    csv_content: str = "",
    url: str = "",
    file_path: str = "",
    name: str = "",
    target_column: str = "",
) -> dict:
    """Load a dataset once and get a dataset_id to reuse in every other tool.

    Provide exactly ONE source:
      csv_content — raw CSV text pasted by the user (best under ~10k rows)
      url         — link to a CSV: Google Sheets share link (must be viewable by
                    anyone with the link), raw GitHub file, or any direct CSV URL
      file_path   — path on the server's machine (only when running locally)

    Optionally pass target_column to get a suggested task type and target stats.

    Returns dataset_id, shape, per-column types/missing counts, and a 5-row preview.
    """
    try:
        sources = [s for s in (csv_content, url, file_path) if s]
        if len(sources) != 1:
            raise DataError("Provide exactly one of csv_content, url, or file_path.")
        if url:
            df = D.load_from_url(url)
        elif file_path:
            df = D.load_from_path(file_path)
        else:
            df = D.parse_csv(csv_content)
        dataset_id = D.CACHE.put(df, name=name)
        return {"dataset_id": dataset_id, **D.summarize(df, target_column)}
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def evaluate(
    target_column: str,
    dataset_id: str = "",
    csv_content: str = "",
    task_type: str = "auto",
) -> dict:
    """Estimate how well TabICL can predict `target_column` from ONE labeled dataset.

    Use this when the user has a single CSV and wants to know "can you predict X
    and how well?". It holds out part of the data the model never sees during
    fitting, so the metrics are honest. Task type (classification/regression) is
    auto-detected unless specified.

    Pass the data as dataset_id (from load_data, preferred) or inline csv_content.

    Returns metrics (classification: accuracy, balanced accuracy, F1, ROC-AUC,
    confusion matrix; regression: R², RMSE, MAE) plus context to interpret them.
    """
    try:
        from . import ml

        df = D.resolve(dataset_id, csv_content)
        return ml.evaluate(df, target_column, task_type)
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def predict(
    target_column: str,
    train_dataset_id: str = "",
    train_csv: str = "",
    new_data_dataset_id: str = "",
    new_data_csv: str = "",
    task_type: str = "auto",
    include_probabilities: bool = False,
) -> dict:
    """Fit TabICL on labeled data and predict `target_column` for new rows.

    Training data must include the target column; the new data needs the same
    feature columns (target optional — if present, evaluation metrics are
    computed too). Pass each dataset as a dataset_id (preferred) or inline CSV.

    Classification returns predicted labels plus a per-row confidence (probability
    of the predicted class); set include_probabilities=true for full class
    probabilities. Regression returns predicted numbers.

    If there are more than 2000 new rows, the full results are cached and a
    preview is returned — fetch the rest with export_predictions.
    """
    try:
        from . import ml

        train_df = D.resolve(train_dataset_id, train_csv)
        new_df = D.resolve(new_data_dataset_id, new_data_csv)
        result = ml.predict(train_df, new_df, target_column, task_type, include_probabilities)

        if result["n_predicted"] > MAX_INLINE_PREDICTIONS:
            out = new_df.copy()
            out[f"predicted_{target_column}"] = result["predictions"]
            if "confidence" in result:
                out["confidence"] = result["confidence"]
            result_id = D.CACHE.put(out, name="predictions", prefix="pred")
            for key in ("predictions", "confidence", "probabilities"):
                if key in result:
                    result[f"{key}_preview_first_50"] = result.pop(key)[:50]
            result["results_dataset_id"] = result_id
            result["note"] = (
                f"Only the first 50 of {result['n_predicted']:,} predictions are shown. "
                f"Use export_predictions with results_dataset_id='{result_id}' to get "
                "the full results as CSV."
            )
        return result
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def export_predictions(results_dataset_id: str, offset: int = 0, limit: int = 2000) -> dict:
    """Fetch cached prediction results (from a large `predict` call) as CSV text.

    Page through with offset/limit; each page returns at most 2000 rows.
    """
    try:
        entry = D.CACHE.get(results_dataset_id)
        limit = max(1, min(limit, MAX_INLINE_PREDICTIONS))
        page = entry.df.iloc[offset : offset + limit]
        return {
            "total_rows": int(len(entry.df)),
            "offset": offset,
            "returned_rows": int(len(page)),
            "csv": page.to_csv(index=False),
        }
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def explain(
    target_column: str,
    dataset_id: str = "",
    csv_content: str = "",
    task_type: str = "auto",
    max_features: int = 20,
) -> dict:
    """Rank which columns matter most for predicting `target_column`.

    Uses permutation importance on a held-out split: quality drop when a column's
    values are shuffled. Model-agnostic and honest, but takes roughly one model
    call per feature — on wide datasets or slow hardware expect ~a minute.

    Returns features ranked by importance with a plain-language method note.
    """
    try:
        from . import ml

        df = D.resolve(dataset_id, csv_content)
        return ml.feature_importance(df, target_column, task_type, max_features)
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def create_report(
    target_column: str,
    dataset_id: str = "",
    csv_content: str = "",
    task_type: str = "auto",
    title: str = "",
    include_importance: bool = True,
) -> dict:
    """Generate a shareable, self-contained HTML report for a labeled dataset.

    Runs evaluation (honest train/test split) and — unless include_importance is
    false — feature importance, then renders a styled report: metric cards with
    plain-language explanations, confusion matrix, target distribution, and a
    feature-importance chart. No external assets; opens in any browser.

    Returns a URL (remote server) or file path (local server) plus the headline
    numbers. Building it fits the model 2×, so it can take a minute or two.
    """
    try:
        from . import ml
        from . import report as R

        df = D.resolve(dataset_id, csv_content)
        evaluation = ml.evaluate(df, target_column, task_type)
        if "error" in evaluation:
            return evaluation
        importance = (
            ml.feature_importance(df, target_column, task_type) if include_importance else None
        )
        html = R.build_report(
            title=title or f"Predicting {target_column}",
            dataset_summary=D.summarize(df),
            evaluation=evaluation,
            importance=importance,
        )

        os.makedirs(REPORTS_DIR, exist_ok=True)
        report_id = uuid.uuid4().hex[:12]
        path = os.path.abspath(os.path.join(REPORTS_DIR, f"{report_id}.html"))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)

        result: dict = {
            "metrics": evaluation.get("metrics"),
            "top_features": (importance or {}).get("importances", [])[:5],
        }
        if _TRANSPORT == "stdio":
            result["report_file"] = path
            result["how_to_open"] = "Open this file in any browser."
        else:
            base = _PUBLIC_URL.rstrip("/") if _PUBLIC_URL else ""
            result["report_url"] = f"{base}/reports/{report_id}"
            result["note"] = "Link serves a standalone HTML page; reports are kept ~24h."
        return result
    except Exception as exc:
        return _err(exc)


# --------------------------------------------------------------------------- #
# HTTP extras: serve generated reports + healthcheck
# --------------------------------------------------------------------------- #


@mcp.custom_route("/reports/{report_id}", methods=["GET"])
async def serve_report(request):
    from starlette.responses import FileResponse, PlainTextResponse

    report_id = request.path_params["report_id"]
    if not report_id.isalnum():
        return PlainTextResponse("Invalid report id", status_code=400)
    path = os.path.join(REPORTS_DIR, f"{report_id}.html")
    if not os.path.exists(path):
        return PlainTextResponse("Report not found (reports expire)", status_code=404)
    return FileResponse(path, media_type="text/html")


@mcp.custom_route("/", methods=["GET"])
@mcp.custom_route("/health", methods=["GET"])
async def health(request):
    from starlette.responses import JSONResponse

    return JSONResponse({"status": "ok", "server": "tabicl-mcp", "mcp_endpoint": "/mcp"})


def _cleanup_reports(max_age_hours: float = 24) -> None:
    if not os.path.isdir(REPORTS_DIR):
        return
    cutoff = time.time() - max_age_hours * 3600
    for fname in os.listdir(REPORTS_DIR):
        fpath = os.path.join(REPORTS_DIR, fname)
        if fname.endswith(".html") and os.path.getmtime(fpath) < cutoff:
            try:
                os.remove(fpath)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def serve_stdio() -> None:
    """Local server for Claude Desktop, Cursor, VS Code, Claude Code."""
    global _TRANSPORT
    _TRANSPORT = "stdio"
    mcp.run(transport="stdio")


def serve_http() -> None:
    """Remote server (Streamable HTTP) for claude.ai, ChatGPT, Gemini, HF Spaces."""
    global _TRANSPORT
    _TRANSPORT = "http"
    _cleanup_reports()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 7860))

    from mcp.server.transport_security import TransportSecuritySettings

    # Behind HF Spaces / any reverse proxy the Host header is the public domain,
    # so the SDK's DNS-rebinding protection must not pin to localhost.
    security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    mcp.run(
        transport="streamable-http",
        host=host,
        port=port,
        stateless_http=True,
        transport_security=security,
    )


if __name__ == "__main__":
    import sys

    serve_http() if (len(sys.argv) > 1 and sys.argv[1] == "http") else serve_stdio()
