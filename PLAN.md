# TabICL MCP — Development Plan

**Goal:** let non-technical users run state-of-the-art tabular ML (TabICL) from inside
Claude / ChatGPT / Gemini through natural language — upload or link a CSV, ask a question,
get predictions, explanations, and a report.

**Hard constraints:**

1. **Completely free to operate.** Local install (stdio) is free by definition; the remote
   endpoint runs on HuggingFace Spaces free CPU tier. No paid APIs, no paid hosting, no
   auth service. (Trade-off we accept: free Spaces sleep after inactivity, so the first
   request after idle takes a couple of minutes while the Space wakes.)
2. **Intuitive end-to-end.** A user should be able to say "here's my customer data, who
   will churn and why?" and the assistant should be able to answer using only these tools:
   evaluate quality honestly, predict on new rows, rank feature importance, and produce a
   shareable report.

---

## Current state (2026-07-31)

- Local repo and GitHub are synced (local fixes rebased onto `origin/main` and pushed).
- Working: `predict` + `inspect_csv` over stdio (verified live: classification and
  regression both return correct results).
- A HuggingFace Space exists (`gblayer/tabicl-mcp`, `hf` git remote) but runs the old
  SSE server.

### What's stale / wrong (to be fixed by this plan)

| Issue | Why it matters |
|---|---|
| `tabicl>=0.1.0` pin | 0.x had no `TabICLRegressor` — import crashes. TabICL is now 2.1.1 (TabICLv2, ICML 2026): official regressor, ~48K-row pretraining range, ~10× faster than TabPFN-2.5 |
| SSE transport | Deprecated in MCP; clients are moving to Streamable HTTP. ChatGPT/Claude.ai/Gemini all take a single streamable-HTTP URL now |
| ChatGPT via OpenAPI "custom actions" | Obsolete — ChatGPT supports MCP connectors natively (Developer Mode); FastMCP never served `/openapi.json` anyway |
| CSV-as-string only | The real bottleneck: pasting >1–2K rows through the model is slow/expensive; results (full probability arrays) blow up the context |
| Manual `OrdinalEncoder` + float32 cast | TabICL 2.x's sklearn interface preprocesses DataFrames itself (verify; keep fallback if not) |
| Multiclass ROC-AUC | Crashes when the test split is missing a class |
| README limits ("<10K rows, <500 features") | Stale; v2 native range is ~2–100 features, up to ~48K train rows |
| No tests, no CI, not on PyPI | Can't recommend `uvx tabicl-mcp` to users until published |

---

## Tool design (the product)

Five tools. Every tool accepts **either** a `dataset_id` (returned by `load_data`) **or**
inline CSV, so data is transferred once per session, not once per call.

| Tool | What it does | Non-technical phrasing it serves |
|---|---|---|
| `load_data` | Ingest CSV from pasted text, a **URL** (Google Sheets share link, raw GitHub, any CSV URL), or a local file path (stdio only). Returns `dataset_id` + summary (shape, columns, types, missing values, suggested target/task) | "Here's my spreadsheet" |
| `evaluate` | One CSV in → honest quality estimate out. Auto train/test split (stratified holdout, or CV when small). Metrics in plain language: accuracy, balanced accuracy, F1, ROC-AUC, confusion matrix / R², RMSE, MAE | "How well can you predict churn from this?" |
| `predict` | Fit on labeled data, predict on new rows (labels optional → metric if present). Returns predictions inline (capped preview) + full results as CSV text | "Which of these 50 customers will churn?" |
| `explain` | Permutation feature importance (free, no extra deps, model-agnostic) → ranked features with plain-language summary | "Why? Which factors matter most?" |
| `create_report` | Self-contained HTML report (inline CSS/SVG, zero external assets): dataset overview, metric cards, confusion matrix, feature-importance chart, prediction distribution | "Give me something I can share with my boss" |

Server `instructions` steer the assistant: one CSV → `evaluate`; new unlabeled rows →
`predict`; "why" questions → `explain`; deliverable → `create_report`; large data → share
a link instead of pasting.

**Notes**
- Feature importance = permutation importance (sklearn), not SHAP: SHAP on a
  transformer-ICL model on free CPU is too slow, and permutation importance is honest,
  model-agnostic, and dependency-free. SHAP stays a stretch goal.
- The HTML report is generated server-side with hand-rolled inline SVG (no matplotlib) to
  stay small (<100KB) and render anywhere — the user saves it as `.html` or the client
  displays it directly.
- Probabilities rounded to 4 decimals, `predict_proba` opt-in, previews capped — context
  budget is a first-class constraint.

## Architecture

```
tabicl_mcp/
  server.py    # FastMCP app + tool definitions (thin layer)
  data.py      # CSV parsing, URL fetch (size-capped, http/https only), dataset cache
  ml.py        # task detection, fit/predict/evaluate, permutation importance
  report.py    # self-contained HTML report generation
app.py         # HF Spaces entry (streamable HTTP on $PORT)
Dockerfile     # CPU torch, checkpoints pre-downloaded at build time
tests/         # pytest: fast unit tests + slow model tests (marked)
```

- **Transports:** stdio (`tabicl-mcp`) for Claude Desktop / Cursor / VS Code;
  **Streamable HTTP** (`tabicl-mcp-http`, uvicorn) for claude.ai / ChatGPT / Gemini.
  SSE is dropped.
- **Dataset cache:** in-memory, TTL + LRU cap. Fine for stdio (one process per user) and
  for the single-instance free Space. Predictions are recomputable — losing the cache on
  Space restart only costs a re-upload.
- **Limits:** ≤10K train rows inline, ≤50K via URL; clear, actionable error messages
  beyond ("sample your data or run locally").
- **Auth:** none by default (free + frictionless); optional `TABICL_MCP_TOKEN` env var
  enables bearer auth for anyone who wants a private deployment.

---

## Phases

### Phase 1 — Core rewrite ← *in progress*
- [ ] `pyproject.toml`: `tabicl>=2.1`, `mcp>=1.10`, add `httpx`; restore real author info;
      `dev` extra with pytest
- [ ] `data.py`: parsing, URL ingestion (Google Sheets link normalization), cache
- [ ] `ml.py`: DataFrame-native fit (verify v2 categorical handling; ordinal-encode
      fallback), evaluate with stratified holdout/CV, robust metrics (multiclass-AUC
      edge cases), permutation importance
- [ ] `report.py`: HTML report with inline SVG charts
- [ ] `server.py`: 5 tools + rewritten instructions, structured (dict) outputs
- [ ] `tests/`: fast tests (no model) + slow tests (real TabICL, tiny data)
- [ ] Verified locally over stdio from Claude Code / Claude Desktop

### Phase 2 — Free remote endpoint
- [ ] Streamable HTTP entry point (uvicorn, `/mcp`)
- [ ] Dockerfile: CPU-only torch, **pre-download v2 checkpoints at build** (kills the
      cold-start checkpoint download), healthcheck
- [ ] Push to the existing HF Space; verify from claude.ai custom connector
      (Settings → Connectors → Add custom connector) and ChatGPT Developer Mode
- [ ] Document the wake-from-sleep behavior honestly in the README

### Phase 3 — Publish & distribute
- [ ] Publish to PyPI → `uvx tabicl-mcp` one-liner for local installs
- [ ] README rewrite: quick-start per client (Claude Desktop, claude.ai, ChatGPT, Gemini,
      Cursor, VS Code), honest TabPFN comparison (open weights / free / private vs their
      hosted GPU + time-series/causal coverage), demo prompts + sample datasets
- [ ] GitHub Actions CI (lint + fast tests)
- [ ] Submit to directories: Anthropic connectors directory, Smithery, PulseMCP, mcp.so

### Phase 4 — Stretch (still free)
- [ ] SHAP explanations behind an optional extra, with row caps
- [ ] Forecasting via `tabicl[forecast]` if v2's forecast API is stable
- [ ] Chunked upload tool for big datasets through chat (if URL ingestion proves
      insufficient)
- [ ] Keep-alive pinger for the HF Space (GitHub Actions cron hitting the healthcheck —
      free way to reduce cold starts)

---

## Decisions made

- **Self-host-first, shared free demo endpoint second.** The HF Space is the "two-click"
  path; `uvx tabicl-mcp` is the private/fast path. Nothing costs money.
- **Permutation importance over SHAP** for v1 explanations (speed on free CPU).
- **Server-generated HTML report** rather than relying on client-side artifacts, so the
  deliverable works identically in Claude, ChatGPT, and Gemini.
- **Drop SSE entirely** rather than maintaining both transports.
