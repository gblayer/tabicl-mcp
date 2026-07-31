---
title: TabICL MCP
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# TabICL MCP Server

> **State-of-the-art tabular ML inside Claude, ChatGPT, or Gemini — upload a CSV, ask a question, get predictions, explanations, and a report. Free and self-hostable.**

[TabICL](https://github.com/soda-inria/tabicl) (Inria, ICML 2025/2026) is a tabular foundation model: it classifies and regresses on tabular data **without any hyperparameter tuning**, via in-context learning. This MCP server makes it usable from natural language in any MCP-compatible assistant or agent.

```
You:    Here's my customer spreadsheet — who is likely to churn, and why?
Claude: [load_data]  → 2,340 rows, 12 columns, target "churned" looks binary
        [evaluate]   → 89% accuracy, 0.93 ROC-AUC on held-out data
        [explain]    → top drivers: contract_type, tenure, monthly_charges
        [create_report] → here's a shareable HTML report with the details…
```

## Tools

| Tool | What it does |
|---|---|
| `load_data` | Ingest a CSV — pasted text, a **URL** (Google Sheets share link, raw GitHub, any CSV URL), or a local file path. Returns a `dataset_id` so data is transferred once, not per call |
| `evaluate` | "How well can you predict X?" from a single labeled CSV — honest held-out metrics (accuracy, balanced accuracy, F1, ROC-AUC, confusion matrix / R², RMSE, MAE) |
| `predict` | Fit on labeled data, predict new rows — labels + per-row confidence (classification) or numbers (regression) |
| `explain` | Which columns drive the predictions (permutation feature importance) |
| `create_report` | Self-contained HTML report: metric cards in plain language, confusion matrix, feature importance chart, distributions. Served as a link (remote) or file (local) |
| `export_predictions` | Page through large prediction results as CSV |

**Data limits:** ~10k rows pasted inline, ~50k via URL or file. TabICLv2 handles 2–100 features natively.

## Try it in 60 seconds

Once connected (see below), paste this into your assistant:

> Load this CSV and tell me: can you predict the `churned` column? How accurate is it, what drives churn the most, and can you make me a report?
> https://raw.githubusercontent.com/gblayer/tabicl-mcp/main/examples/customer_churn.csv

## Use it

### Option A — Local (free, private, fastest)

Requires Python ≥ 3.10. With [uv](https://docs.astral.sh/uv/): nothing to install, clients run `uvx tabicl-mcp`.

**Claude Desktop** — `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "tabicl": { "command": "uvx", "args": ["tabicl-mcp"] }
  }
}
```

**Claude Code:** `claude mcp add tabicl -- uvx tabicl-mcp`

**Cursor / VS Code** (`.vscode/mcp.json` or Cursor Settings → MCP):

```json
{
  "servers": {
    "tabicl": { "type": "stdio", "command": "uvx", "args": ["tabicl-mcp"] }
  }
}
```

(Or `pip install tabicl-mcp` and use `"command": "tabicl-mcp"`.)

With a local server, `load_data` accepts file paths — no pasting needed, and no data ever leaves your machine.

### Option B — Remote URL (free, for claude.ai / ChatGPT / Gemini)

Deploy your own free endpoint on HuggingFace Spaces:

1. Create a Space → **Docker** SDK → CPU basic (free).
2. Push this repo to it (the `Dockerfile` pre-downloads model checkpoints at build).
3. Your MCP endpoint: `https://YOUR-SPACE.hf.space/mcp`

Then connect:

- **claude.ai** — Settings → Connectors → *Add custom connector* → paste the URL.
- **ChatGPT** — enable Developer Mode (Settings, paid plans) → Apps & Connectors → add the URL.
- **Gemini CLI / API** — add the URL as a remote MCP server in your config.

> Free Spaces sleep after inactivity — the first request after idle takes a minute or two while the Space wakes. Everything after that is fast.

### Option C — Docker anywhere

```bash
docker build -t tabicl-mcp .
docker run -p 7860:7860 tabicl-mcp
# MCP endpoint: http://localhost:7860/mcp   ·   healthcheck: /health
```

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -m "not slow"   # fast tests
.venv/bin/pytest                 # includes real-model tests (downloads checkpoints)
```

Layout: `tabicl_mcp/server.py` (MCP tools) · `data.py` (ingestion + cache) · `ml.py` (evaluate/predict/importance) · `report.py` (HTML reports).

MIT licensed. TabICL itself is by [soda-inria](https://github.com/soda-inria/tabicl) (BSD-3). PRs welcome.
