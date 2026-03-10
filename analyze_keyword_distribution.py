#!/usr/bin/env python3
"""
Analyze keyword distributions for a keyword table written by
extract_arxiv_physics_keywords.py.

The script connects to Postgres, explodes the JSONB `keywords` array, and writes
distribution tables to CSV for downstream analysis.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import psycopg2
import psycopg2.extras as extras
from psycopg2 import sql


DEFAULT_TABLE = (
    "classifications_and_keywords."
    "arxiv_from_kaggle_keywords_title_abstract_boost_on_q15_57588cc1"
)


def parse_schema_table(qualified_name: str) -> Tuple[str, str]:
    parts = qualified_name.split(".")
    if len(parts) == 1:
        return "public", parts[0]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"Invalid table name '{qualified_name}'. Use 'schema.table'.")


def load_dotenv_simple(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not path.exists():
        return env

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        env[key] = value
    return env


def get_env_or_default(env: Dict[str, str], key: str, default: str | None = None) -> str | None:
    if key in os.environ:
        return os.environ[key]
    if key in env:
        return env[key]
    return default


def slugify(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw).strip("._-") or "keyword_distribution"


def table_sql(qualified_name: str) -> sql.Composed:
    schema_name, table_name = parse_schema_table(qualified_name)
    return sql.SQL("{}.{}").format(sql.Identifier(schema_name), sql.Identifier(table_name))


def keyword_rows_cte(table_name: str) -> sql.Composed:
    return sql.SQL(
        """
        WITH table_rows AS (
            SELECT
                paper_id,
                keyword_count,
                keywords
            FROM {}
        ),
        keyword_rows AS (
            SELECT
                t.paper_id,
                t.keyword_count,
                kw.ordinality::integer AS keyword_rank,
                kw.item ->> 'keyword' AS keyword,
                NULLIF(kw.item ->> 'ngram', '')::integer AS ngram,
                NULLIF(kw.item ->> 'score', '')::double precision AS score,
                NULLIF(kw.item ->> 'base_score', '')::double precision AS base_score,
                NULLIF(kw.item ->> 'physics_boost', '')::double precision AS physics_boost
            FROM table_rows t
            CROSS JOIN LATERAL jsonb_array_elements(t.keywords) WITH ORDINALITY AS kw(item, ordinality)
        )
        """
    ).format(table_sql(table_name))


def fetch_all(
    conn: psycopg2.extensions.connection,
    query: sql.Composable,
    params: Iterable[Any] = (),
) -> List[Dict[str, Any]]:
    with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
        cur.execute(query, tuple(params))
        return [dict(row) for row in cur.fetchall()]


def fetch_one(
    conn: psycopg2.extensions.connection,
    query: sql.Composable,
    params: Iterable[Any] = (),
) -> Dict[str, Any]:
    rows = fetch_all(conn, query, params)
    if not rows:
        return {}
    return rows[0]


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def as_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(value)


def format_compact_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:.2f}"


def truncate_label(text: str, max_len: int = 32) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def svg_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke: str,
    stroke_width: float = 1.0,
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}"{dash_attr} />'
    )


def svg_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    rx: float = 0.0,
    stroke: str | None = None,
    stroke_width: float = 1.0,
) -> str:
    stroke_attr = ""
    if stroke:
        stroke_attr = f' stroke="{stroke}" stroke-width="{stroke_width}"'
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'fill="{fill}" rx="{rx:.2f}"{stroke_attr} />'
    )


def svg_text(
    x: float,
    y: float,
    content: str,
    font_size: int = 13,
    fill: str = "#111827",
    anchor: str = "start",
    font_weight: str = "400",
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" fill="{fill}" font-size="{font_size}" '
        f'font-family="Helvetica, Arial, sans-serif" text-anchor="{anchor}" '
        f'font-weight="{font_weight}">{html.escape(content)}</text>'
    )


def write_horizontal_bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    rows: List[Dict[str, Any]],
    label_key: str,
    value_key: str,
    value_formatter,
    label_formatter=None,
    color: str = "#2563eb",
) -> None:
    width = 1200
    title_height = 84
    row_height = 30
    bottom_padding = 44
    left_margin = 290
    right_margin = 150
    plot_width = width - left_margin - right_margin
    height = title_height + max(len(rows), 1) * row_height + bottom_padding
    plot_top = title_height
    plot_bottom = height - bottom_padding

    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="180" viewBox="0 0 {width} 180">',
            svg_rect(0, 0, width, 180, "#ffffff"),
            svg_text(24, 34, title, font_size=24, font_weight="700"),
            svg_text(24, 58, subtitle, font_size=13, fill="#475569"),
            svg_text(24, 110, "No rows available.", font_size=15, fill="#64748b"),
            "</svg>",
        ]
        path.write_text("\n".join(svg) + "\n", encoding="utf-8")
        return

    values = [max(as_float(row.get(value_key)), 0.0) for row in rows]
    max_value = max(values) if values else 1.0
    if max_value <= 0:
        max_value = 1.0

    grid_values = [max_value * step / 4 for step in range(5)]
    svg: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        svg_rect(0, 0, width, height, "#ffffff"),
        svg_text(24, 34, title, font_size=24, font_weight="700"),
        svg_text(24, 58, subtitle, font_size=13, fill="#475569"),
    ]

    for grid_value in grid_values:
        x = left_margin + (grid_value / max_value) * plot_width
        svg.append(svg_line(x, plot_top - 6, x, plot_bottom + 6, "#e2e8f0", dash="4 4"))
        tick = format_compact_number(grid_value)
        svg.append(svg_text(x, plot_bottom + 24, tick, font_size=12, fill="#64748b", anchor="middle"))

    for idx, row in enumerate(rows):
        y = plot_top + idx * row_height
        value = max(as_float(row.get(value_key)), 0.0)
        bar_width = (value / max_value) * plot_width
        label = str(row.get(label_key, ""))
        if label_formatter:
            label = label_formatter(row)
        value_text = value_formatter(row)

        svg.append(svg_line(left_margin, y + row_height, width - right_margin + 10, y + row_height, "#f1f5f9"))
        svg.append(svg_text(left_margin - 12, y + 20, truncate_label(label, 36), font_size=13, anchor="end"))
        svg.append(svg_rect(left_margin, y + 6, bar_width, 16, color, rx=4))
        svg.append(svg_text(left_margin + bar_width + 8, y + 19, value_text, font_size=12, fill="#334155"))

    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def write_dashboard_html(
    output_dir: Path,
    summary: Dict[str, Any],
    top_keywords: List[Dict[str, Any]],
    top_boosted_keywords: List[Dict[str, Any]],
) -> None:
    paper_count = f"{as_int(summary.get('paper_count')):,}"
    unique_keywords = f"{as_int(summary.get('unique_keywords')):,}"
    keyword_rows = f"{as_int(summary.get('exploded_keyword_rows')):,}"
    avg_keywords = f"{as_float(summary.get('avg_keyword_count')):.2f}"
    avg_score = f"{as_float(summary.get('avg_score')):.4f}"
    avg_boost = f"{as_float(summary.get('avg_physics_boost')):.4f}"

    cards = [
        ("Papers", paper_count),
        ("Unique keywords", unique_keywords),
        ("Keyword rows", keyword_rows),
        ("Avg keywords / paper", avg_keywords),
        ("Avg score", avg_score),
        ("Avg physics boost", avg_boost),
    ]

    top_keyword_rows = []
    for row in top_keywords[:12]:
        top_keyword_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['keyword']))}</td>"
            f"<td>{as_int(row['paper_frequency']):,}</td>"
            f"<td>{as_float(row['paper_prevalence_pct']):.3f}%</td>"
            f"<td>{as_float(row['avg_score']):.4f}</td>"
            "</tr>"
        )

    boosted_rows = []
    for row in top_boosted_keywords[:12]:
        boosted_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['keyword']))}</td>"
            f"<td>{as_int(row['paper_frequency']):,}</td>"
            f"<td>{as_float(row['avg_physics_boost']):.4f}</td>"
            f"<td>{as_float(row['avg_score']):.4f}</td>"
            "</tr>"
        )

    chart_specs = [
        ("Top keywords by paper frequency", "visualizations/top_keywords_by_paper_frequency.svg"),
        ("Top boosted keywords", "visualizations/top_boosted_keywords.svg"),
        ("N-gram distribution", "visualizations/ngram_distribution.svg"),
        ("Keyword rank distribution", "visualizations/keyword_rank_distribution.svg"),
        ("Score histogram", "visualizations/score_histogram.svg"),
        ("Physics boost histogram", "visualizations/physics_boost_histogram.svg"),
        ("Keyword count distribution", "visualizations/keyword_count_distribution.svg"),
    ]

    card_html = "".join(
        f'<div class="card"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></div>'
        for label, value in cards
    )
    chart_html = "".join(
        f'<section class="panel"><h2>{html.escape(title)}</h2><img src="{html.escape(src)}" alt="{html.escape(title)}" /></section>'
        for title, src in chart_specs
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Keyword Distribution Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8fafc;
      --panel: #ffffff;
      --border: #dbe4ee;
      --text: #0f172a;
      --muted: #475569;
      --accent: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 32px;
      background: radial-gradient(circle at top left, #eff6ff 0, #f8fafc 36%, #f8fafc 100%);
      color: var(--text);
      font-family: Helvetica, Arial, sans-serif;
    }}
    .wrap {{
      max-width: 1400px;
      margin: 0 auto;
    }}
    .hero {{
      margin-bottom: 24px;
      padding: 28px;
      background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
      border: 1px solid var(--border);
      border-radius: 20px;
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      line-height: 1.1;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      font-size: 15px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-bottom: 24px;
    }}
    .card, .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
    }}
    .card {{
      padding: 18px;
    }}
    .card .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 10px;
    }}
    .card .value {{
      font-size: 28px;
      font-weight: 700;
    }}
    .panels {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
      margin-bottom: 24px;
    }}
    .panel {{
      padding: 20px;
    }}
    .panel h2 {{
      margin: 0 0 14px;
      font-size: 20px;
    }}
    .panel img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 14px;
      border: 1px solid #eef2f7;
      background: #fff;
    }}
    .tables {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 10px 0;
      border-bottom: 1px solid #e2e8f0;
      text-align: left;
      font-size: 14px;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .files {{
      margin-top: 24px;
      color: var(--muted);
      font-size: 14px;
    }}
    .files a {{
      color: var(--accent);
      text-decoration: none;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Keyword Distribution Dashboard</h1>
      <p>Table: <code>{html.escape(str(summary.get("table_name", "")))}</code></p>
    </section>
    <section class="grid">{card_html}</section>
    <section class="panels">{chart_html}</section>
    <section class="tables">
      <div class="panel">
        <h2>Top keywords</h2>
        <table>
          <thead>
            <tr><th>Keyword</th><th>Papers</th><th>Prevalence</th><th>Avg score</th></tr>
          </thead>
          <tbody>
            {''.join(top_keyword_rows)}
          </tbody>
        </table>
      </div>
      <div class="panel">
        <h2>Top boosted keywords</h2>
        <table>
          <thead>
            <tr><th>Keyword</th><th>Papers</th><th>Avg boost</th><th>Avg score</th></tr>
          </thead>
          <tbody>
            {''.join(boosted_rows)}
          </tbody>
        </table>
      </div>
    </section>
    <div class="files">
      Data files:
      <a href="summary.json">summary.json</a>,
      <a href="top_keywords.csv">top_keywords.csv</a>,
      <a href="top_boosted_keywords.csv">top_boosted_keywords.csv</a>,
      <a href="ngram_distribution.csv">ngram_distribution.csv</a>,
      <a href="rank_distribution.csv">rank_distribution.csv</a>,
      <a href="keyword_count_distribution.csv">keyword_count_distribution.csv</a>,
      <a href="score_distribution.csv">score_distribution.csv</a>,
      <a href="boost_distribution.csv">boost_distribution.csv</a>
    </div>
  </div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_doc, encoding="utf-8")


def write_visualizations(
    output_dir: Path,
    summary: Dict[str, Any],
    top_keywords: List[Dict[str, Any]],
    top_boosted_keywords: List[Dict[str, Any]],
    ngram_distribution: List[Dict[str, Any]],
    rank_distribution: List[Dict[str, Any]],
    keyword_count_distribution: List[Dict[str, Any]],
    score_distribution: List[Dict[str, Any]],
    boost_distribution: List[Dict[str, Any]],
    chart_top_n: int,
) -> None:
    visualizations_dir = output_dir / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    top_keyword_chart_rows = top_keywords[:chart_top_n]
    top_boosted_chart_rows = top_boosted_keywords[:chart_top_n]

    write_horizontal_bar_chart(
        visualizations_dir / "top_keywords_by_paper_frequency.svg",
        title="Top keywords by paper frequency",
        subtitle=f"Top {len(top_keyword_chart_rows)} keywords ranked by number of distinct papers.",
        rows=top_keyword_chart_rows,
        label_key="keyword",
        value_key="paper_frequency",
        value_formatter=lambda row: (
            f"{as_int(row['paper_frequency']):,} papers | "
            f"{as_float(row['paper_prevalence_pct']):.3f}%"
        ),
        color="#2563eb",
    )
    write_horizontal_bar_chart(
        visualizations_dir / "top_boosted_keywords.svg",
        title="Top boosted keywords",
        subtitle=f"Top {len(top_boosted_chart_rows)} keywords ranked by average physics boost.",
        rows=top_boosted_chart_rows,
        label_key="keyword",
        value_key="avg_physics_boost",
        value_formatter=lambda row: (
            f"avg boost {as_float(row['avg_physics_boost']):.4f} | "
            f"papers {as_int(row['paper_frequency']):,}"
        ),
        color="#0f766e",
    )
    write_horizontal_bar_chart(
        visualizations_dir / "ngram_distribution.svg",
        title="N-gram distribution",
        subtitle="Share of exploded keyword rows by n-gram length.",
        rows=ngram_distribution,
        label_key="ngram",
        value_key="keyword_occurrences",
        label_formatter=lambda row: f"{as_int(row['ngram'])}-gram",
        value_formatter=lambda row: (
            f"{as_int(row['keyword_occurrences']):,} keywords | "
            f"{as_float(row['occurrence_share_pct']):.2f}%"
        ),
        color="#7c3aed",
    )
    write_horizontal_bar_chart(
        visualizations_dir / "keyword_rank_distribution.svg",
        title="Keyword rank distribution",
        subtitle="How often each keyword rank position occurs across papers.",
        rows=rank_distribution,
        label_key="keyword_rank",
        value_key="keyword_occurrences",
        label_formatter=lambda row: f"Rank {as_int(row['keyword_rank'])}",
        value_formatter=lambda row: (
            f"{as_int(row['keyword_occurrences']):,} keywords | "
            f"avg score {as_float(row['avg_score']):.4f}"
        ),
        color="#ea580c",
    )
    write_horizontal_bar_chart(
        visualizations_dir / "score_histogram.svg",
        title="Keyword score histogram",
        subtitle=f"Histogram of final keyword scores using bucket width {as_float(summary.get('score_bucket_width')):.3f}.",
        rows=score_distribution,
        label_key="bucket_id",
        value_key="keyword_occurrences",
        label_formatter=lambda row: (
            f"{as_float(row['bucket_start']):.2f} to {as_float(row['bucket_end']):.2f}"
        ),
        value_formatter=lambda row: f"{as_int(row['keyword_occurrences']):,} keywords",
        color="#2563eb",
    )
    write_horizontal_bar_chart(
        visualizations_dir / "physics_boost_histogram.svg",
        title="Physics boost histogram",
        subtitle=f"Histogram of applied physics boosts using bucket width {as_float(summary.get('boost_bucket_width')):.3f}.",
        rows=boost_distribution,
        label_key="bucket_id",
        value_key="keyword_occurrences",
        label_formatter=lambda row: (
            f"{as_float(row['bucket_start']):.2f} to {as_float(row['bucket_end']):.2f}"
        ),
        value_formatter=lambda row: f"{as_int(row['keyword_occurrences']):,} keywords",
        color="#0f766e",
    )
    write_horizontal_bar_chart(
        visualizations_dir / "keyword_count_distribution.svg",
        title="Keyword count distribution",
        subtitle="How many keywords are stored per paper.",
        rows=keyword_count_distribution,
        label_key="keyword_count",
        value_key="paper_count",
        label_formatter=lambda row: f"{as_int(row['keyword_count'])} keywords",
        value_formatter=lambda row: (
            f"{as_int(row['paper_count']):,} papers | "
            f"{as_float(row['paper_share_pct']):.2f}%"
        ),
        color="#7c3aed",
    )
    write_dashboard_html(output_dir, summary, top_keywords, top_boosted_keywords)


def summary_query(table_name: str) -> sql.Composed:
    return keyword_rows_cte(table_name) + sql.SQL(
        """
        SELECT
            (SELECT COUNT(*)::bigint FROM table_rows) AS paper_count,
            (SELECT COALESCE(SUM(keyword_count), 0)::bigint FROM table_rows) AS recorded_keyword_slots,
            (SELECT ROUND(AVG(keyword_count)::numeric, 4) FROM table_rows) AS avg_keyword_count,
            (
                SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY keyword_count)
                FROM table_rows
            ) AS median_keyword_count,
            (SELECT MIN(keyword_count) FROM table_rows) AS min_keyword_count,
            (SELECT MAX(keyword_count) FROM table_rows) AS max_keyword_count,
            (
                SELECT COUNT(*)::bigint
                FROM table_rows
                WHERE keyword_count = 0
            ) AS zero_keyword_papers,
            (SELECT COUNT(*)::bigint FROM keyword_rows) AS exploded_keyword_rows,
            (
                SELECT COUNT(DISTINCT paper_id)::bigint
                FROM keyword_rows
            ) AS papers_with_keywords,
            (
                SELECT COUNT(DISTINCT keyword)::bigint
                FROM keyword_rows
            ) AS unique_keywords,
            (SELECT ROUND(AVG(score)::numeric, 4) FROM keyword_rows) AS avg_score,
            (SELECT ROUND(AVG(base_score)::numeric, 4) FROM keyword_rows) AS avg_base_score,
            (SELECT ROUND(AVG(physics_boost)::numeric, 4) FROM keyword_rows) AS avg_physics_boost
        """
    )


def top_keywords_query(table_name: str) -> sql.Composed:
    return keyword_rows_cte(table_name) + sql.SQL(
        """
        SELECT
            keyword,
            COUNT(*)::bigint AS total_occurrences,
            COUNT(DISTINCT paper_id)::bigint AS paper_frequency,
            ROUND(
                100.0 * COUNT(DISTINCT paper_id)
                / NULLIF((SELECT COUNT(*) FROM table_rows), 0),
                4
            ) AS paper_prevalence_pct,
            ROUND(AVG(score)::numeric, 4) AS avg_score,
            ROUND(AVG(base_score)::numeric, 4) AS avg_base_score,
            ROUND(AVG(physics_boost)::numeric, 4) AS avg_physics_boost,
            ROUND(AVG(keyword_rank)::numeric, 2) AS avg_rank,
            MIN(keyword_rank) AS best_rank,
            MAX(keyword_rank) AS worst_rank,
            MIN(ngram) AS min_ngram,
            MAX(ngram) AS max_ngram
        FROM keyword_rows
        GROUP BY keyword
        ORDER BY paper_frequency DESC, total_occurrences DESC, avg_score DESC, keyword ASC
        LIMIT %s
        """
    )


def top_boosted_keywords_query(table_name: str) -> sql.Composed:
    return keyword_rows_cte(table_name) + sql.SQL(
        """
        SELECT
            keyword,
            COUNT(*)::bigint AS total_occurrences,
            COUNT(DISTINCT paper_id)::bigint AS paper_frequency,
            ROUND(AVG(physics_boost)::numeric, 4) AS avg_physics_boost,
            ROUND(MAX(physics_boost)::numeric, 4) AS max_physics_boost,
            ROUND(AVG(score)::numeric, 4) AS avg_score,
            ROUND(AVG(base_score)::numeric, 4) AS avg_base_score
        FROM keyword_rows
        GROUP BY keyword
        HAVING MAX(physics_boost) > 0
        ORDER BY avg_physics_boost DESC, paper_frequency DESC, avg_score DESC, keyword ASC
        LIMIT %s
        """
    )


def ngram_distribution_query(table_name: str) -> sql.Composed:
    return keyword_rows_cte(table_name) + sql.SQL(
        """
        SELECT
            ngram,
            COUNT(*)::bigint AS keyword_occurrences,
            COUNT(DISTINCT paper_id)::bigint AS papers_with_ngram,
            ROUND(
                100.0 * COUNT(*)
                / NULLIF(SUM(COUNT(*)) OVER (), 0),
                4
            ) AS occurrence_share_pct,
            ROUND(AVG(score)::numeric, 4) AS avg_score,
            ROUND(AVG(physics_boost)::numeric, 4) AS avg_physics_boost,
            ROUND(AVG(keyword_rank)::numeric, 2) AS avg_rank
        FROM keyword_rows
        GROUP BY ngram
        ORDER BY ngram
        """
    )


def rank_distribution_query(table_name: str) -> sql.Composed:
    return keyword_rows_cte(table_name) + sql.SQL(
        """
        SELECT
            keyword_rank,
            COUNT(*)::bigint AS keyword_occurrences,
            COUNT(DISTINCT paper_id)::bigint AS papers_with_rank,
            ROUND(AVG(score)::numeric, 4) AS avg_score,
            ROUND(AVG(base_score)::numeric, 4) AS avg_base_score,
            ROUND(AVG(physics_boost)::numeric, 4) AS avg_physics_boost
        FROM keyword_rows
        GROUP BY keyword_rank
        ORDER BY keyword_rank
        """
    )


def keyword_count_distribution_query(table_name: str) -> sql.Composed:
    return keyword_rows_cte(table_name) + sql.SQL(
        """
        SELECT
            keyword_count,
            COUNT(*)::bigint AS paper_count,
            ROUND(
                100.0 * COUNT(*)
                / NULLIF(SUM(COUNT(*)) OVER (), 0),
                4
            ) AS paper_share_pct
        FROM table_rows
        GROUP BY keyword_count
        ORDER BY keyword_count
        """
    )


def score_distribution_query(table_name: str) -> sql.Composed:
    return keyword_rows_cte(table_name) + sql.SQL(
        """
        ,
        bucketed AS (
            SELECT
                FLOOR(score / %s)::integer AS bucket_id,
                paper_id
            FROM keyword_rows
        )
        SELECT
            bucket_id,
            ROUND((bucket_id * %s)::numeric, 4) AS bucket_start,
            ROUND(((bucket_id + 1) * %s)::numeric, 4) AS bucket_end,
            COUNT(*)::bigint AS keyword_occurrences,
            COUNT(DISTINCT paper_id)::bigint AS paper_count
        FROM bucketed
        GROUP BY bucket_id
        ORDER BY bucket_id
        """
    )


def boost_distribution_query(table_name: str) -> sql.Composed:
    return keyword_rows_cte(table_name) + sql.SQL(
        """
        ,
        bucketed AS (
            SELECT
                FLOOR(physics_boost / %s)::integer AS bucket_id,
                paper_id
            FROM keyword_rows
        )
        SELECT
            bucket_id,
            ROUND((bucket_id * %s)::numeric, 4) AS bucket_start,
            ROUND(((bucket_id + 1) * %s)::numeric, 4) AS bucket_end,
            COUNT(*)::bigint AS keyword_occurrences,
            COUNT(DISTINCT paper_id)::bigint AS paper_count
        FROM bucketed
        GROUP BY bucket_id
        ORDER BY bucket_id
        """
    )


def render_preview(summary: Dict[str, Any], top_keywords: List[Dict[str, Any]]) -> None:
    print(f"table: {summary.get('table_name')}")
    print(
        "papers="
        f"{summary.get('paper_count', 0):,} | "
        f"unique_keywords={summary.get('unique_keywords', 0):,} | "
        f"keyword_rows={summary.get('exploded_keyword_rows', 0):,}"
    )
    print(
        "keywords/paper="
        f"avg {summary.get('avg_keyword_count')} | "
        f"median {summary.get('median_keyword_count')} | "
        f"range [{summary.get('min_keyword_count')}, {summary.get('max_keyword_count')}]"
    )
    print(
        "scores="
        f"avg {summary.get('avg_score')} | "
        f"base {summary.get('avg_base_score')} | "
        f"boost {summary.get('avg_physics_boost')}"
    )

    if not top_keywords:
        return

    print("")
    print("top keywords:")
    for row in top_keywords[:10]:
        keyword = row["keyword"]
        paper_frequency = row["paper_frequency"]
        prevalence = row["paper_prevalence_pct"]
        avg_score = row["avg_score"]
        avg_rank = row["avg_rank"]
        print(
            f"  {keyword:<30} papers={paper_frequency:>8,} "
            f"prevalence={prevalence:>7}% avg_score={avg_score:>6} avg_rank={avg_rank:>5}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze keyword distributions for a Postgres keyword table."
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help="Qualified Postgres table name containing a JSONB `keywords` column.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to dotenv-style file containing DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for CSV/JSON outputs. Defaults to analysis_output/<table_slug>/.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of rows to keep in top-keyword outputs.",
    )
    parser.add_argument(
        "--chart-top-n",
        type=int,
        default=20,
        help="Number of rows to render in the SVG bar charts.",
    )
    parser.add_argument(
        "--score-bucket-width",
        type=float,
        default=0.05,
        help="Bucket width for the keyword score histogram.",
    )
    parser.add_argument(
        "--boost-bucket-width",
        type=float,
        default=0.02,
        help="Bucket width for the physics-boost histogram.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    if args.top_n < 1:
        raise ValueError("--top-n must be >= 1.")
    if args.chart_top_n < 1:
        raise ValueError("--chart-top-n must be >= 1.")
    if args.score_bucket_width <= 0:
        raise ValueError("--score-bucket-width must be > 0.")
    if args.boost_bucket_width <= 0:
        raise ValueError("--boost-bucket-width must be > 0.")

    env = load_dotenv_simple(Path(args.env_file))
    db_host = get_env_or_default(env, "DB_HOST")
    db_port = int(get_env_or_default(env, "DB_PORT", "5432"))
    db_name = get_env_or_default(env, "DB_NAME")
    db_user = get_env_or_default(env, "DB_USER")
    db_password = get_env_or_default(env, "DB_PASSWORD")
    if not all([db_host, db_name, db_user, db_password]):
        raise RuntimeError(
            "Missing DB credentials. Set DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD "
            "in env vars or the env file."
        )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("analysis_output") / slugify(args.table)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
    )
    try:
        summary = fetch_one(conn, summary_query(args.table))
        summary["table_name"] = args.table
        summary["top_n"] = args.top_n
        summary["chart_top_n"] = args.chart_top_n
        summary["score_bucket_width"] = args.score_bucket_width
        summary["boost_bucket_width"] = args.boost_bucket_width

        top_keywords = fetch_all(conn, top_keywords_query(args.table), [args.top_n])
        top_boosted_keywords = fetch_all(
            conn, top_boosted_keywords_query(args.table), [args.top_n]
        )
        ngram_distribution = fetch_all(conn, ngram_distribution_query(args.table))
        rank_distribution = fetch_all(conn, rank_distribution_query(args.table))
        keyword_count_distribution = fetch_all(
            conn, keyword_count_distribution_query(args.table)
        )
        score_distribution = fetch_all(
            conn,
            score_distribution_query(args.table),
            [args.score_bucket_width, args.score_bucket_width, args.score_bucket_width],
        )
        boost_distribution = fetch_all(
            conn,
            boost_distribution_query(args.table),
            [args.boost_bucket_width, args.boost_bucket_width, args.boost_bucket_width],
        )
    finally:
        conn.close()

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    write_csv(
        output_dir / "top_keywords.csv",
        top_keywords,
        [
            "keyword",
            "total_occurrences",
            "paper_frequency",
            "paper_prevalence_pct",
            "avg_score",
            "avg_base_score",
            "avg_physics_boost",
            "avg_rank",
            "best_rank",
            "worst_rank",
            "min_ngram",
            "max_ngram",
        ],
    )
    write_csv(
        output_dir / "top_boosted_keywords.csv",
        top_boosted_keywords,
        [
            "keyword",
            "total_occurrences",
            "paper_frequency",
            "avg_physics_boost",
            "max_physics_boost",
            "avg_score",
            "avg_base_score",
        ],
    )
    write_csv(
        output_dir / "ngram_distribution.csv",
        ngram_distribution,
        [
            "ngram",
            "keyword_occurrences",
            "papers_with_ngram",
            "occurrence_share_pct",
            "avg_score",
            "avg_physics_boost",
            "avg_rank",
        ],
    )
    write_csv(
        output_dir / "rank_distribution.csv",
        rank_distribution,
        [
            "keyword_rank",
            "keyword_occurrences",
            "papers_with_rank",
            "avg_score",
            "avg_base_score",
            "avg_physics_boost",
        ],
    )
    write_csv(
        output_dir / "keyword_count_distribution.csv",
        keyword_count_distribution,
        ["keyword_count", "paper_count", "paper_share_pct"],
    )
    write_csv(
        output_dir / "score_distribution.csv",
        score_distribution,
        ["bucket_id", "bucket_start", "bucket_end", "keyword_occurrences", "paper_count"],
    )
    write_csv(
        output_dir / "boost_distribution.csv",
        boost_distribution,
        ["bucket_id", "bucket_start", "bucket_end", "keyword_occurrences", "paper_count"],
    )
    write_visualizations(
        output_dir=output_dir,
        summary=summary,
        top_keywords=top_keywords,
        top_boosted_keywords=top_boosted_keywords,
        ngram_distribution=ngram_distribution,
        rank_distribution=rank_distribution,
        keyword_count_distribution=keyword_count_distribution,
        score_distribution=score_distribution,
        boost_distribution=boost_distribution,
        chart_top_n=args.chart_top_n,
    )

    render_preview(summary, top_keywords)
    print("")
    print(f"wrote analysis files to {output_dir}")


if __name__ == "__main__":
    run(build_parser().parse_args())
