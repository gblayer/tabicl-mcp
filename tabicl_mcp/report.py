"""Self-contained HTML report generation (inline CSS + SVG, no external assets)."""

from __future__ import annotations

import html
import time


def _esc(v) -> str:
    return html.escape(str(v))


def _metric_cards(metrics: dict, task: str) -> str:
    friendly = {
        "accuracy": ("Accuracy", "share of correct predictions"),
        "balanced_accuracy": ("Balanced accuracy", "accuracy averaged over classes"),
        "f1_macro": ("F1 (macro)", "balance of precision and recall"),
        "roc_auc": ("ROC AUC", "0.5 = random, 1.0 = perfect"),
        "r2": ("R²", "1.0 = perfect, 0 = no better than the mean"),
        "rmse": ("RMSE", "typical prediction error, in target units"),
        "mae": ("MAE", "average absolute error, in target units"),
    }
    cards = []
    for key, (label, hint) in friendly.items():
        if key in metrics:
            cards.append(
                f'<div class="card"><div class="metric">{_esc(metrics[key])}</div>'
                f'<div class="label">{label}</div><div class="hint">{hint}</div></div>'
            )
    return f'<div class="cards">{"".join(cards)}</div>'


def _confusion_table(cm: dict) -> str:
    labels, matrix = cm["labels"], cm["matrix"]
    peak = max(max(row) for row in matrix) or 1
    head = "".join(f"<th>pred: {_esc(l)}</th>" for l in labels)
    rows = []
    for label, row in zip(labels, matrix):
        cells = []
        for j, v in enumerate(row):
            alpha = 0.12 + 0.75 * (v / peak)
            color = f"rgba(37,99,235,{alpha:.2f})" if labels[j] == label else f"rgba(220,38,38,{alpha:.2f})"
            if v == 0:
                color = "transparent"
            cells.append(f'<td style="background:{color}">{v}</td>')
        rows.append(f"<tr><th>true: {_esc(label)}</th>{''.join(cells)}</tr>")
    return (
        "<h2>Where the model is right and wrong</h2>"
        '<p class="note">Rows are the true classes, columns the predictions. '
        "Blue diagonal = correct; red off-diagonal = mistakes.</p>"
        f'<table class="cm"><tr><th></th>{head}</tr>{"".join(rows)}</table>'
    )


def _importance_chart(importances: list[dict]) -> str:
    if not importances:
        return ""
    top = importances[:12]
    peak = max(abs(i["importance"]) for i in top) or 1
    bar_h, gap, label_w, chart_w = 26, 8, 220, 420
    rows = []
    for idx, item in enumerate(top):
        y = idx * (bar_h + gap)
        w = max(2, int(abs(item["importance"]) / peak * chart_w))
        rows.append(
            f'<text x="{label_w - 8}" y="{y + bar_h / 2 + 5}" text-anchor="end" class="feat">{_esc(item["feature"])}</text>'
            f'<rect x="{label_w}" y="{y}" width="{w}" height="{bar_h}" rx="4" fill="#2563eb" opacity="0.85"/>'
            f'<text x="{label_w + w + 6}" y="{y + bar_h / 2 + 5}" class="val">{item["importance"]}</text>'
        )
    height = len(top) * (bar_h + gap)
    return (
        "<h2>What drives the predictions</h2>"
        '<p class="note">Permutation importance: how much model quality drops when a '
        "column's values are shuffled. Near zero = barely used.</p>"
        f'<svg viewBox="0 0 {label_w + chart_w + 80} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="max-width:100%;height:auto">{"".join(rows)}</svg>'
    )


def _distribution(evaluation: dict) -> str:
    dist = evaluation.get("class_distribution")
    if not dist:
        return ""
    total = sum(dist.values()) or 1
    peak = max(dist.values()) or 1
    rows = []
    for label, count in sorted(dist.items(), key=lambda kv: -kv[1]):
        pct = 100 * count / total
        w = 100 * count / peak
        rows.append(
            f'<tr><th>{_esc(label)}</th>'
            f'<td class="bar"><div style="width:{w:.0f}%"></div></td>'
            f"<td>{count} ({pct:.0f}%)</td></tr>"
        )
    return (
        "<h2>Target distribution</h2>"
        f'<table class="dist">{"".join(rows)}</table>'
    )


def build_report(
    title: str,
    dataset_summary: dict,
    evaluation: dict,
    importance: dict | None = None,
    predictions: dict | None = None,
) -> str:
    task = evaluation["task_type"]
    metrics = evaluation.get("metrics", {})
    generated = time.strftime("%Y-%m-%d %H:%M")

    sections = [
        f"<h1>{_esc(title)}</h1>",
        f'<p class="sub">TabICL {task} · target: <b>{_esc(evaluation["target_column"])}</b> · '
        f'{dataset_summary["n_rows"]:,} rows × {dataset_summary["n_columns"]} columns · '
        f"generated {generated}</p>",
        "<h2>Model quality</h2>",
        f'<p class="note">{_esc(evaluation.get("note", ""))}</p>',
        _metric_cards(metrics, task),
    ]
    if "confusion_matrix" in metrics:
        sections.append(_confusion_table(metrics["confusion_matrix"]))
    sections.append(_distribution(evaluation))
    if importance:
        sections.append(_importance_chart(importance.get("importances", [])))
    if predictions:
        sections.append(
            "<h2>Predictions on new data</h2>"
            f'<p class="note">{predictions["n_predicted"]:,} rows scored.</p>'
        )
    if evaluation.get("warning"):
        sections.append(f'<p class="warn">⚠ {_esc(evaluation["warning"])}</p>')
    sections.append(
        '<p class="foot">Generated by <a href="https://github.com/gblayer/tabicl-mcp">TabICL MCP</a> — '
        'in-context tabular ML (<a href="https://github.com/soda-inria/tabicl">TabICL</a>, Inria).</p>'
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 16px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0 auto;
         max-width: 860px; padding: 32px 20px 60px; color: #1f2937; background: #fff; }}
  @media (prefers-color-scheme: dark) {{
    body {{ color: #e5e7eb; background: #111827; }}
    .card {{ background: #1f2937 !important; }}
    table.cm td, table.cm th, table.dist th, table.dist td {{ border-color: #374151 !important; }}
  }}
  h1 {{ margin: 0 0 4px; font-size: 1.7em; }}
  h2 {{ margin: 36px 0 6px; font-size: 1.15em; }}
  .sub {{ color: #6b7280; margin-top: 0; }}
  .note {{ color: #6b7280; font-size: 0.9em; margin: 2px 0 12px; }}
  .warn {{ color: #b45309; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 12px; }}
  .card {{ background: #f3f4f6; border-radius: 10px; padding: 14px 18px; min-width: 130px; }}
  .metric {{ font-size: 1.6em; font-weight: 700; }}
  .label {{ font-weight: 600; margin-top: 2px; }}
  .hint {{ color: #6b7280; font-size: 0.78em; }}
  table.cm {{ border-collapse: collapse; margin-top: 8px; }}
  table.cm th, table.cm td {{ border: 1px solid #e5e7eb; padding: 6px 12px; text-align: center; }}
  table.cm th {{ font-weight: 600; font-size: 0.85em; }}
  table.dist {{ width: 100%; border-collapse: collapse; }}
  table.dist th {{ text-align: left; padding: 4px 12px 4px 0; font-weight: 600; white-space: nowrap; }}
  table.dist td {{ padding: 4px 0 4px 8px; white-space: nowrap; }}
  table.dist td.bar {{ width: 60%; }}
  table.dist td.bar div {{ background: #2563eb; opacity: 0.85; height: 16px; border-radius: 4px; }}
  svg text.feat {{ font: 13px -apple-system, sans-serif; fill: currentColor; }}
  svg text.val {{ font: 12px -apple-system, sans-serif; fill: #6b7280; }}
  .foot {{ margin-top: 48px; color: #9ca3af; font-size: 0.85em; }}
  a {{ color: #2563eb; }}
</style></head>
<body>{"".join(s for s in sections if s)}</body></html>"""
