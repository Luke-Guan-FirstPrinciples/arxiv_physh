#!/usr/bin/env python3
"""
Focused keyword-topic analysis for extracted keyword tables.

This script is designed for cases where a user wants counts and visualizations
for a predefined list of topics that may map to several extracted keyword
variants. Each topic can define:

- exact_terms: exact keyword strings to count
- regex_patterns: regex patterns used for grouped matching across variants
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import psycopg2
import psycopg2.extras as extras
import yaml
from psycopg2 import sql

from analyze_keyword_distribution import (
    DEFAULT_TABLE,
    as_float,
    as_int,
    get_env_or_default,
    keyword_rows_cte,
    load_dotenv_simple,
    slugify,
    write_horizontal_bar_chart,
)


DEFAULT_CONFIG = "keyword_focus_quantum_information.yaml"


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def fetch_all(
    conn: psycopg2.extensions.connection,
    query: sql.Composable,
    params: Sequence[Any] = (),
) -> List[Dict[str, Any]]:
    with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
        cur.execute(query, tuple(params))
        return [dict(row) for row in cur.fetchall()]


def fetch_one(
    conn: psycopg2.extensions.connection,
    query: sql.Composable,
    params: Sequence[Any] = (),
) -> Dict[str, Any]:
    rows = fetch_all(conn, query, params)
    return rows[0] if rows else {}


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def build_predicate(
    exact_terms: Sequence[str],
    regex_patterns: Sequence[str],
) -> Tuple[sql.Composable, List[Any]]:
    clauses: List[sql.Composable] = []
    params: List[Any] = []

    clean_exact_terms = [term.strip().lower() for term in exact_terms if str(term).strip()]
    clean_patterns = [pattern.strip() for pattern in regex_patterns if str(pattern).strip()]

    if clean_exact_terms:
        clauses.append(sql.SQL("keyword = ANY(%s)"))
        params.append(clean_exact_terms)

    if clean_patterns:
        regex_clauses = [sql.SQL("keyword ~* %s") for _ in clean_patterns]
        clauses.append(
            sql.Composed([sql.SQL("("), sql.SQL(" OR ").join(regex_clauses), sql.SQL(")")])
        )
        params.extend(clean_patterns)

    if not clauses:
        raise ValueError("Each keyword focus group must define exact_terms or regex_patterns.")

    return sql.SQL(" OR ").join(clauses), params


def total_papers_query(table_name: str) -> sql.Composable:
    return sql.Composed(
        [
            keyword_rows_cte(table_name),
            sql.SQL("SELECT COUNT(*)::bigint AS paper_count FROM table_rows"),
        ]
    )


def group_stats_query(table_name: str, predicate: sql.Composable) -> sql.Composable:
    return sql.Composed(
        [
            keyword_rows_cte(table_name),
            sql.SQL(
                """
                SELECT
                    COUNT(*)::bigint AS keyword_occurrences,
                    COUNT(DISTINCT paper_id)::bigint AS paper_frequency,
                    COUNT(DISTINCT keyword)::bigint AS matched_variant_count,
                    ROUND(AVG(score)::numeric, 4) AS avg_score,
                    ROUND(AVG(base_score)::numeric, 4) AS avg_base_score,
                    ROUND(AVG(physics_boost)::numeric, 4) AS avg_physics_boost,
                    ROUND(AVG(keyword_rank)::numeric, 2) AS avg_rank
                FROM keyword_rows
                WHERE
                """
            ),
            predicate,
        ]
    )


def variant_breakdown_query(
    table_name: str,
    predicate: sql.Composable,
    top_variants: int,
) -> sql.Composable:
    return sql.Composed(
        [
            keyword_rows_cte(table_name),
            sql.SQL(
                """
                SELECT
                    keyword,
                    COUNT(*)::bigint AS keyword_occurrences,
                    COUNT(DISTINCT paper_id)::bigint AS paper_frequency,
                    ROUND(AVG(score)::numeric, 4) AS avg_score,
                    ROUND(AVG(base_score)::numeric, 4) AS avg_base_score,
                    ROUND(AVG(physics_boost)::numeric, 4) AS avg_physics_boost,
                    ROUND(AVG(keyword_rank)::numeric, 2) AS avg_rank
                FROM keyword_rows
                WHERE
                """
            ),
            predicate,
            sql.SQL(
                """
                GROUP BY keyword
                ORDER BY paper_frequency DESC, keyword_occurrences DESC, avg_score DESC, keyword ASC
                LIMIT %s
                """
            ),
        ]
    )


def rank_distribution_query(table_name: str, predicate: sql.Composable) -> sql.Composable:
    return sql.Composed(
        [
            keyword_rows_cte(table_name),
            sql.SQL(
                """
                SELECT
                    keyword_rank,
                    COUNT(*)::bigint AS keyword_occurrences,
                    COUNT(DISTINCT paper_id)::bigint AS paper_frequency,
                    ROUND(AVG(score)::numeric, 4) AS avg_score
                FROM keyword_rows
                WHERE
                """
            ),
            predicate,
            sql.SQL(
                """
                GROUP BY keyword_rank
                ORDER BY keyword_rank
                """
            ),
        ]
    )


def focus_dashboard_html(
    output_dir: Path,
    table_name: str,
    focus_label: str,
    summary_rows: List[Dict[str, Any]],
    top_variants: List[Dict[str, Any]],
) -> None:
    total_grouped_papers = sum(as_int(row["grouped_paper_frequency"]) for row in summary_rows)
    total_exact_papers = sum(as_int(row["exact_paper_frequency"]) for row in summary_rows)
    cards = [
        ("Focus set", focus_label),
        ("Topics", str(len(summary_rows))),
        ("Grouped paper hits", f"{total_grouped_papers:,}"),
        ("Exact paper hits", f"{total_exact_papers:,}"),
    ]

    card_html = "".join(
        f'<div class="card"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></div>'
        for label, value in cards
    )

    summary_table_rows = []
    for row in summary_rows:
        summary_table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['group_label']))}</td>"
            f"<td>{as_int(row['grouped_paper_frequency']):,}</td>"
            f"<td>{as_float(row['grouped_paper_prevalence_pct']):.3f}%</td>"
            f"<td>{as_int(row['exact_paper_frequency']):,}</td>"
            f"<td>{as_float(row['grouped_avg_score']):.4f}</td>"
            "</tr>"
        )

    variant_rows = []
    for row in top_variants[:20]:
        variant_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['group_label']))}</td>"
            f"<td>{html.escape(str(row['keyword']))}</td>"
            f"<td>{as_int(row['paper_frequency']):,}</td>"
            f"<td>{as_int(row['keyword_occurrences']):,}</td>"
            "</tr>"
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Focused Keyword Dashboard</title>
  <style>
    body {{
      margin: 0;
      padding: 32px;
      background: radial-gradient(circle at top left, #eefbf4 0, #f8fafc 36%, #f8fafc 100%);
      color: #0f172a;
      font-family: Helvetica, Arial, sans-serif;
    }}
    .wrap {{
      max-width: 1400px;
      margin: 0 auto;
    }}
    .hero, .card, .panel {{
      background: #ffffff;
      border: 1px solid #dbe4ee;
      border-radius: 18px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
    }}
    .hero {{
      padding: 28px;
      margin-bottom: 20px;
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 32px;
    }}
    .hero p {{
      margin: 0;
      color: #475569;
      font-size: 15px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }}
    .card {{
      padding: 18px;
    }}
    .label {{
      color: #475569;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 10px;
    }}
    .value {{
      font-size: 28px;
      font-weight: 700;
    }}
    .panel {{
      padding: 20px;
      margin-bottom: 18px;
    }}
    .panel h2 {{
      margin: 0 0 14px;
      font-size: 20px;
    }}
    img {{
      width: 100%;
      height: auto;
      border: 1px solid #eef2f7;
      border-radius: 14px;
      background: #fff;
    }}
    .tables {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
      gap: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      text-align: left;
      font-size: 14px;
      padding: 10px 0;
      border-bottom: 1px solid #e2e8f0;
    }}
    th {{
      color: #475569;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Focused Keyword Dashboard</h1>
      <p>Table: <code>{html.escape(table_name)}</code></p>
      <p>Focus set: <code>{html.escape(focus_label)}</code></p>
    </section>
    <section class="grid">{card_html}</section>
    <section class="panel">
      <h2>Grouped paper frequency</h2>
      <img src="visualizations/grouped_paper_frequency.svg" alt="Grouped paper frequency" />
    </section>
    <section class="panel">
      <h2>Exact paper frequency</h2>
      <img src="visualizations/exact_paper_frequency.svg" alt="Exact paper frequency" />
    </section>
    <section class="panel">
      <h2>Matched variant richness</h2>
      <img src="visualizations/matched_variant_count.svg" alt="Matched variant richness" />
    </section>
    <section class="tables">
      <div class="panel">
        <h2>Topic summary</h2>
        <table>
          <thead>
            <tr><th>Topic</th><th>Grouped papers</th><th>Grouped prevalence</th><th>Exact papers</th><th>Avg score</th></tr>
          </thead>
          <tbody>
            {''.join(summary_table_rows)}
          </tbody>
        </table>
      </div>
      <div class="panel">
        <h2>Top matched variants</h2>
        <table>
          <thead>
            <tr><th>Topic</th><th>Variant</th><th>Papers</th><th>Occurrences</th></tr>
          </thead>
          <tbody>
            {''.join(variant_rows)}
          </tbody>
        </table>
      </div>
    </section>
  </div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_doc, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze focused keyword-topic groups for a keyword table."
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help="Qualified Postgres keyword table.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="YAML config containing keyword focus groups.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to dotenv-style file containing DB credentials.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for focused outputs. Defaults under analysis_output/<table_slug>/<config_slug>/.",
    )
    parser.add_argument(
        "--top-variants",
        type=int,
        default=15,
        help="Max matched variants to keep per focus group.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    if args.top_variants < 1:
        raise ValueError("--top-variants must be >= 1.")

    config_path = Path(args.config)
    config = load_yaml_config(config_path)
    focus_label = str(config.get("label") or config_path.stem)
    groups = config.get("groups") or []
    if not isinstance(groups, list) or not groups:
        raise ValueError("Config must define a non-empty `groups` list.")

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
        else Path("analysis_output") / slugify(args.table) / slugify(config_path.stem)
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
        total_papers = as_int(fetch_one(conn, total_papers_query(args.table)).get("paper_count"))

        summary_rows: List[Dict[str, Any]] = []
        variant_rows: List[Dict[str, Any]] = []
        rank_rows: List[Dict[str, Any]] = []

        for group in groups:
            group_label = str(group.get("label") or "").strip()
            if not group_label:
                raise ValueError("Each group must define a non-empty `label`.")

            exact_terms = group.get("exact_terms") or []
            regex_patterns = group.get("regex_patterns") or []
            if not isinstance(exact_terms, list) or not isinstance(regex_patterns, list):
                raise ValueError(f"`exact_terms` and `regex_patterns` must be lists for {group_label}.")

            exact_predicate, exact_params = build_predicate(exact_terms, [])
            grouped_predicate, grouped_params = build_predicate(exact_terms, regex_patterns)

            exact_stats = fetch_one(conn, group_stats_query(args.table, exact_predicate), exact_params)
            grouped_stats = fetch_one(
                conn, group_stats_query(args.table, grouped_predicate), grouped_params
            )

            grouped_paper_frequency = as_int(grouped_stats.get("paper_frequency"))
            exact_paper_frequency = as_int(exact_stats.get("paper_frequency"))

            summary_row = {
                "group_label": group_label,
                "exact_terms": json.dumps(exact_terms),
                "regex_patterns": json.dumps(regex_patterns),
                "exact_keyword_occurrences": as_int(exact_stats.get("keyword_occurrences")),
                "exact_paper_frequency": exact_paper_frequency,
                "exact_paper_prevalence_pct": round(
                    100.0 * exact_paper_frequency / total_papers, 4
                )
                if total_papers
                else 0.0,
                "exact_avg_score": exact_stats.get("avg_score"),
                "grouped_keyword_occurrences": as_int(grouped_stats.get("keyword_occurrences")),
                "grouped_paper_frequency": grouped_paper_frequency,
                "grouped_paper_prevalence_pct": round(
                    100.0 * grouped_paper_frequency / total_papers, 4
                )
                if total_papers
                else 0.0,
                "grouped_matched_variant_count": as_int(grouped_stats.get("matched_variant_count")),
                "grouped_avg_score": grouped_stats.get("avg_score"),
                "grouped_avg_base_score": grouped_stats.get("avg_base_score"),
                "grouped_avg_physics_boost": grouped_stats.get("avg_physics_boost"),
                "grouped_avg_rank": grouped_stats.get("avg_rank"),
            }
            summary_rows.append(summary_row)

            group_variants = fetch_all(
                conn,
                variant_breakdown_query(args.table, grouped_predicate, args.top_variants),
                [*grouped_params, args.top_variants],
            )
            for row in group_variants:
                row["group_label"] = group_label
                variant_rows.append(row)

            group_ranks = fetch_all(
                conn,
                rank_distribution_query(args.table, grouped_predicate),
                grouped_params,
            )
            for row in group_ranks:
                row["group_label"] = group_label
                rank_rows.append(row)
    finally:
        conn.close()

    summary_rows.sort(key=lambda row: (-as_int(row["grouped_paper_frequency"]), row["group_label"]))
    variant_rows.sort(
        key=lambda row: (
            row["group_label"],
            -as_int(row["paper_frequency"]),
            -as_int(row["keyword_occurrences"]),
            str(row["keyword"]),
        )
    )
    rank_rows.sort(key=lambda row: (row["group_label"], as_int(row["keyword_rank"])))

    manifest = {
        "table_name": args.table,
        "focus_label": focus_label,
        "config_path": str(config_path),
        "top_variants": args.top_variants,
        "total_papers": total_papers,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(
        output_dir / "focus_group_summary.csv",
        summary_rows,
        [
            "group_label",
            "exact_terms",
            "regex_patterns",
            "exact_keyword_occurrences",
            "exact_paper_frequency",
            "exact_paper_prevalence_pct",
            "exact_avg_score",
            "grouped_keyword_occurrences",
            "grouped_paper_frequency",
            "grouped_paper_prevalence_pct",
            "grouped_matched_variant_count",
            "grouped_avg_score",
            "grouped_avg_base_score",
            "grouped_avg_physics_boost",
            "grouped_avg_rank",
        ],
    )
    write_csv(
        output_dir / "focus_group_variants.csv",
        variant_rows,
        [
            "group_label",
            "keyword",
            "keyword_occurrences",
            "paper_frequency",
            "avg_score",
            "avg_base_score",
            "avg_physics_boost",
            "avg_rank",
        ],
    )
    write_csv(
        output_dir / "focus_group_rank_distribution.csv",
        rank_rows,
        ["group_label", "keyword_rank", "keyword_occurrences", "paper_frequency", "avg_score"],
    )

    visualizations_dir = output_dir / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    write_horizontal_bar_chart(
        visualizations_dir / "grouped_paper_frequency.svg",
        title="Grouped topic frequency",
        subtitle="Distinct papers matching each topic after applying grouped regex variants.",
        rows=summary_rows,
        label_key="group_label",
        value_key="grouped_paper_frequency",
        value_formatter=lambda row: (
            f"{as_int(row['grouped_paper_frequency']):,} papers | "
            f"{as_float(row['grouped_paper_prevalence_pct']):.3f}%"
        ),
        color="#2563eb",
    )
    write_horizontal_bar_chart(
        visualizations_dir / "exact_paper_frequency.svg",
        title="Exact topic frequency",
        subtitle="Distinct papers matching the exact keyword phrases only.",
        rows=summary_rows,
        label_key="group_label",
        value_key="exact_paper_frequency",
        value_formatter=lambda row: (
            f"{as_int(row['exact_paper_frequency']):,} papers | "
            f"{as_float(row['exact_paper_prevalence_pct']):.3f}%"
        ),
        color="#0f766e",
    )
    write_horizontal_bar_chart(
        visualizations_dir / "matched_variant_count.svg",
        title="Matched variant richness",
        subtitle="Number of distinct extracted keyword variants matched per topic.",
        rows=summary_rows,
        label_key="group_label",
        value_key="grouped_matched_variant_count",
        value_formatter=lambda row: f"{as_int(row['grouped_matched_variant_count']):,} variants",
        color="#7c3aed",
    )

    focus_dashboard_html(output_dir, args.table, focus_label, summary_rows, variant_rows)

    print(f"focus set: {focus_label}")
    print(f"table: {args.table}")
    print(f"total papers: {total_papers:,}")
    print("")
    print("grouped topic frequency:")
    for row in summary_rows:
        print(
            f"  {row['group_label']:<30} "
            f"papers={as_int(row['grouped_paper_frequency']):>6,} "
            f"exact={as_int(row['exact_paper_frequency']):>6,} "
            f"variants={as_int(row['grouped_matched_variant_count']):>4,}"
        )
    print("")
    print(f"wrote focused analysis files to {output_dir}")


if __name__ == "__main__":
    run(build_parser().parse_args())
