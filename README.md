# TabICL MCP Server

> **State-of-the-art tabular ML inside Claude — upload a CSV, ask a question, get predictions, explanations, and a report. Free and self-hostable.**

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

## Sharing your data: size guide

There are three ways to get a CSV to the server. Pick by file size:

| Your data | Best way to share it |
|---|---|
| Up to ~2,000 rows (≈1 MB) | **Upload the file straight into the chat** and ask your question — simplest, works out of the box |
| ~2,000–50,000 rows | **Homepage upload** or a **Google Sheets link** (see below) — the data goes directly to the server instead of through the AI model, which is faster and cheaper |
| More than 50,000 rows / 50 MB | Sample it down first, or [run the server locally](#option-a--local-free-private-fastest) where there's no transfer at all |

Hard server limits: 10,000 rows for data passed through the chat, 50,000 rows / 50 MB
via link or homepage upload. TabICL works best with 2–100 feature columns.

**Homepage upload** (for bigger files): open your server's homepage —
[gblayer-tabicl-mcp.hf.space](https://gblayer-tabicl-mcp.hf.space/) for the public
server — choose your CSV, click Upload, and copy the dataset id it returns (looks like
`ds_1a2b3c4d`). Then just mention it in the chat: *"Analyze my dataset ds_1a2b3c4d —
predict churn."* The id is valid for ~4 hours and the data stays in the server's memory
only.

**Google Sheets:** in your sheet, click **Share → General access → "Anyone with the
link" (Viewer)**, copy the link, and paste it into the chat: *"Load this sheet and
predict revenue: https://docs.google.com/spreadsheets/d/…"*. The server converts the
share link to a CSV export automatically. (Only do this for sheets that aren't
confidential — link-sharing makes them readable by anyone who has the URL.)

## Try it in 60 seconds

**Connect the public server to claude.ai** (Pro/Max/Team plans):

1. Go to **Settings → Connectors → Add custom connector**
2. Name: `tabicl` · URL: **`https://gblayer-tabicl-mcp.hf.space/mcp`**
3. Leave the OAuth fields empty and click **Add** — that's it.

<img src="https://raw.githubusercontent.com/gblayer/tabicl-mcp/main/docs/images/add-connector.png" alt="Add custom connector dialog in claude.ai" width="480">

Then paste this into a new chat:

> Load this CSV and tell me: can you predict the `churned` column? How accurate is it, what drives churn the most, and can you make me a report?
> https://raw.githubusercontent.com/gblayer/tabicl-mcp/main/examples/customer_churn.csv

You'll get honest held-out metrics, the churn drivers ranked, and a shareable report — [see a sample of what the report looks like](https://html-preview.github.io/?url=https://raw.githubusercontent.com/gblayer/tabicl-mcp/main/examples/customer_churn_report.html) ([source](examples/customer_churn_report.html)).

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

### Option B — Deploy your own private server (free)

**Who this is for:** when you use the public URL above, your CSV data is processed on a
server operated by this project. That's fine for demos and non-sensitive data — but if
you're working with confidential data (customer lists, medical records, company
financials), deploy your own copy instead: identical functionality, but **your data only
ever touches infrastructure you control**. A second reason: the public Space is a single
shared free instance — if it's busy or asleep you wait, while your own deployment serves
only you.

Deploying your own free endpoint on HuggingFace Spaces:

1. Create a Space → **Docker** SDK → CPU basic (free).
2. HF Spaces reads its deployment settings from a YAML header at the very top of
   the Space's `README.md` — add this before pushing (only needed for the copy
   that lives on HuggingFace, not for using the server):

   ```yaml
   ---
   title: TabICL MCP
   emoji: 🤖
   colorFrom: blue
   colorTo: green
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```

3. Push this repo to the Space (the `Dockerfile` pre-downloads model checkpoints at build).
4. Your MCP endpoint: `https://YOUR-SPACE.hf.space/mcp`

Then connect:

- **claude.ai** — Settings → Connectors → *Add custom connector* → paste the URL.
- Any other MCP-compatible client that accepts a Streamable HTTP URL.

> **Large files:** instead of uploading a big CSV into the chat, open the server's
> homepage (`https://YOUR-SPACE.hf.space/`), upload it there, and paste the returned
> dataset id into the chat — the data then never has to pass through the model.

> **Why not ChatGPT?** ChatGPT's custom-connector support currently can't reliably pass
> uploaded files (or sometimes even tool calls) to MCP servers, so we don't document it.
> The server is standard MCP — if ChatGPT's support matures, it will just work.

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
