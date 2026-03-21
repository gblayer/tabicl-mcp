# TabICL MCP Server

> **State-of-the-art in-context tabular ML, available as a tool inside any MCP-compatible LLM.**

TabICL ([soda-inria/tabicl](https://github.com/soda-inria/tabicl)) is an academic-grade model that performs classification and regression on tabular data without hyperparameter tuning, using in-context learning — the same paradigm as TabPFN. This MCP server exposes it as two tools (`predict`, `inspect_csv`) that any LLM agent can call.

---

## Quick comparison with TabPFN MCP

| | TabICL MCP (this) | TabPFN MCP (PriorLabs) |
|---|---|---|
| Model source | INRIA / open academia | PriorLabs (commercial) |
| License | MIT | Proprietary API |
| Free tier | ✅ Unlimited (run yourself) | ❌ API credits |
| Max rows | ~10 000 | ~10 000 |
| Self-hostable | ✅ Yes | ❌ No |
| HF Spaces deploy | ✅ Free CPU | N/A |

---

## Installation (user-facing)

```bash
# Requires Python 3.10+
pip install tabicl-mcp
```

This installs the `tabicl-mcp` (stdio) and `tabicl-mcp-http` (HTTP) CLI commands.

---

## Deployment options

### Option A — Local (stdio) · FREE · Claude Desktop, VS Code, Cursor

Best for: individual users who want zero latency and full privacy.

```bash
pip install tabicl-mcp
```

Then configure your client (see [Client configs](#client-configs)).

### Option B — HuggingFace Spaces (HTTP/SSE) · FREE · Claude.ai web

Best for: sharing a public endpoint that Claude.ai can call remotely.

1. Fork this repo.
2. Create a new HF Space: Docker SDK, CPU Basic (free).
3. Push the repo — the `Dockerfile` handles everything.
4. Your MCP URL will be: `https://YOUR-USERNAME-tabicl-mcp.hf.space/sse`

### Option C — Railway / Render · ~$5/month · HTTP/SSE

Best for: reliable uptime with a custom domain.

```bash
# Railway
railway login
railway init
railway up
```

Set env var `PORT=8000`.

### Option D — Docker (self-hosted VPS)

```bash
docker build -t tabicl-mcp .
docker run -p 7860:7860 tabicl-mcp
```

---

## Client configs

### Claude Desktop (local stdio)

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "tabicl": {
      "command": "tabicl-mcp",
      "args": []
    }
  }
}
```

Restart Claude Desktop. You'll see "TabICL" appear in the tools panel.

### Claude.ai (remote HTTP/SSE)

1. Go to **Profile → Integrations → Add custom integration**
2. Server URL: `https://YOUR-USERNAME-tabicl-mcp.hf.space/sse`
3. Click Connect.

### VS Code (GitHub Copilot / MCP extension)

Create `.vscode/mcp.json` in your workspace:

```json
{
  "servers": {
    "tabicl": {
      "type": "stdio",
      "command": "tabicl-mcp"
    }
  }
}
```

### Cursor

**Settings → MCP → Add server**:

```json
{
  "tabicl": { "command": "tabicl-mcp", "args": [] }
}
```

### ChatGPT (custom actions)

The HTTP server exposes `/openapi.json` automatically via FastMCP.
In GPT Builder → Actions → Schema URL: `https://YOUR-SERVER/openapi.json`

---

## How to use (Claude chat example)

```
User: I have a CSV of customer data. Can you predict who will churn?

[User pastes train.csv and test.csv]

Claude: I'll use TabICL to train a classifier on your data.
        [calls inspect_csv → detects "churn" column is binary → classification]
        [calls predict → returns predictions + ROC AUC]

        Here are the churn predictions for your 50 test customers:
        - Predictions: ["No", "Yes", "No", ...]
        - ROC AUC: 0.87

        An AUC of 0.87 means the model is doing well — random is 0.5,
        perfect is 1.0. Customers 3, 7, and 12 have the highest churn
        probability (check predict_proba for exact values).
```

---

## Tools reference

### `predict`

| Parameter | Type | Description |
|---|---|---|
| `train_csv` | string | Full CSV content of training set (must include target column) |
| `test_csv` | string | Full CSV content of test set (with or without target column) |
| `target_column` | string | Name of the column to predict |
| `task_type` | string | `"auto"` (default), `"classification"`, or `"regression"` |

**Returns** JSON with:
- `predictions` — list of predicted values
- `predict_proba` — (classification) list of probability arrays
- `classes` — (classification) class labels in order
- `metric` — `{"ROC_AUC": 0.87}` or `{"R2": 0.72}` (if test set has labels)

### `inspect_csv`

| Parameter | Type | Description |
|---|---|---|
| `csv_content` | string | Full CSV string |
| `target_column` | string | Optional — shows target distribution and suggests task type |

**Returns** JSON summary: shape, dtypes, missing values, target info.

---

## Agent usage (VS Code / Cursor)

In an agentic IDE the model can call TabICL as part of a multi-step workflow:

```
Agent plan:
1. Read train.csv from disk             [filesystem tool]
2. inspect_csv → confirm columns        [tabicl tool]
3. Read test.csv from disk              [filesystem tool]
4. predict → get predictions + metric   [tabicl tool]
5. Write predictions.csv                [filesystem tool]
6. Plot ROC curve                       [code execution]
```

Example Cursor agent prompt:
> "Load `data/train.csv` and `data/test.csv`, use TabICL to predict `income`,
>  save the predictions to `data/predictions.csv`, and report the R²."

---

## Limits and tips

- **Row limit**: TabICL works best with < 10 000 training rows. For larger datasets, sample.
- **Feature limit**: Handles high-dimensional data well, but < 500 features is ideal.
- **CSV format**: Headers required. Missing values are handled automatically.
- **Privacy**: With local stdio deployment, no data ever leaves your machine.
- **Speed**: First call is slower (model loads). Subsequent calls are faster.

---

## Contributing

TabICL is maintained by INRIA (academia). This MCP wrapper is MIT licensed.
PRs welcome — especially for streaming large predictions and caching model weights.
