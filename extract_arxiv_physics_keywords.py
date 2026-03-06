#!/usr/bin/env python3
"""
Batch physics-focused keyword extraction from Postgres with YAML configuration.

Reads arXiv rows from a source table and writes extracted keyword JSON to an
output table. Output table name can be auto-generated from configuration
combination (text mode + physics boost toggle).
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import psycopg2
import psycopg2.extras as extras
import torch
import yaml
from keybert import KeyBERT
from psycopg2 import sql
from sentence_transformers import SentenceTransformer


PHYSICS_TOKEN_HINTS = {
    "adiabatic",
    "astrophysics",
    "atom",
    "atomic",
    "boson",
    "bosonic",
    "cosmic",
    "cosmology",
    "dark",
    "diffusion",
    "electrodynamics",
    "electromagnetic",
    "electron",
    "entropy",
    "fermion",
    "fermionic",
    "fluid",
    "gauge",
    "gravity",
    "gravitational",
    "hadron",
    "hamiltonian",
    "heisenberg",
    "higgs",
    "holographic",
    "inflation",
    "ising",
    "lattice",
    "lagrangian",
    "magnetization",
    "magnetic",
    "many-body",
    "meson",
    "molecular",
    "neutrino",
    "neutron",
    "nuclear",
    "order-parameter",
    "oscillation",
    "particle",
    "perturbation",
    "phase",
    "phonon",
    "photon",
    "plasma",
    "polymer",
    "quantization",
    "quantum",
    "quark",
    "relativity",
    "renormalization",
    "scattering",
    "spin",
    "spectroscopy",
    "spacetime",
    "superconducting",
    "superconductivity",
    "symmetry",
    "thermodynamic",
    "thermodynamics",
    "topological",
    "vortex",
    "wavefunction",
}

PHYSICS_PHRASE_HINTS = {
    "critical exponents",
    "dark matter",
    "density functional theory",
    "effective field theory",
    "equation of state",
    "feynman diagram",
    "gauge field",
    "ground state",
    "hilbert space",
    "landau level",
    "matrix element",
    "mean field",
    "phase transition",
    "quantum hall",
    "quantum monte carlo",
    "renormalization group",
    "schrodinger equation",
    "spin orbit coupling",
    "standard model",
    "vacuum expectation value",
}

GENERIC_PENALTY_TOKENS = {
    "analysis",
    "approach",
    "data",
    "dataset",
    "framework",
    "method",
    "methods",
    "model",
    "result",
    "results",
    "study",
    "system",
}

TEXT_MODE_ALIASES = {
    "title": "title",
    "title_only": "title",
    "abstract": "abstract",
    "abstract_only": "abstract",
    "title+abstract": "title+abstract",
    "title_abstract": "title+abstract",
    "title_and_abstract": "title+abstract",
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "env_file": ".env",
    "database": {
        "host": None,
        "port": 5432,
        "name": None,
        "user": None,
        "password": None,
    },
    "input": {
        "source_table": "arxiv_base.arxiv_from_kaggle",
        "id_column": "id",
        "title_column": "title",
        "abstract_column": "abstract",
        "categories_column": "categories",
        "text_mode": "title+abstract",
        "physics_only": True,
        "physics_categories_csv": "arxiv_physics_categories.csv",
        "limit": None,
        "fetch_batch_size": 32,
        "force_recompute": False,
    },
    "keywords": {
        "model": "allenai/scibert_scivocab_uncased",
        "device": "gpu",
        "min_ngram": 1,
        "max_ngram": 3,
        "candidate_top_n": 30,
        "top_n": 10,
        "stop_words": "english",
        "use_mmr": True,
        "diversity": 0.35,
        "physics_boost_enabled": True,
    },
    "output": {
        "schema": "classifications_and_keywords",
        "table": "auto",
        "include_source_text": False,
        "write_batch_size": 500,
    },
    "runtime": {
        "dry_run": False,
    },
}


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


def get_env_or_default(
    env: Dict[str, str], key: str, default: str | None = None
) -> str | None:
    if key in os.environ:
        return os.environ[key]
    if key in env:
        return env[key]
    return default


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("YAML root must be a mapping/object.")

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    deep_merge_dicts(cfg, raw)
    return cfg


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def tokenize_lower(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower())


def normalize_text_mode(raw_text_mode: str) -> str:
    normalized = clean_text(raw_text_mode).lower().replace(" ", "_")
    if normalized not in TEXT_MODE_ALIASES:
        allowed = ", ".join(sorted(TEXT_MODE_ALIASES))
        raise ValueError(f"Unsupported text_mode='{raw_text_mode}'. Allowed: {allowed}")
    return TEXT_MODE_ALIASES[normalized]


def normalize_stop_words(raw_stop_words: Any) -> str | None:
    if raw_stop_words is None:
        return None
    if isinstance(raw_stop_words, str):
        value = raw_stop_words.strip()
        if value.lower() in {"none", "null", ""}:
            return None
        return value
    return str(raw_stop_words)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def clamp_float(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def resolve_device(preferred: str = "gpu") -> str:
    if preferred == "gpu":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_physics_ids(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Category CSV not found: {csv_path}")

    physics_ids: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "id" not in (reader.fieldnames or []):
            raise ValueError(f"CSV must include an 'id' column: {csv_path}")
        for row in reader:
            category_id = clean_text((row or {}).get("id"))
            if category_id:
                physics_ids.add(category_id)

    return sorted(physics_ids)


def select_input_text(title: str | None, abstract: str | None, text_mode: str) -> str:
    title_clean = clean_text(title)
    abstract_clean = clean_text(abstract)
    if text_mode == "title":
        return title_clean
    if text_mode == "abstract":
        return abstract_clean
    if title_clean and abstract_clean:
        return f"{title_clean} [SEP] {abstract_clean}"
    if title_clean:
        return title_clean
    return abstract_clean


def physics_boost(keyword: str, title_tokens: set[str]) -> float:
    tokens = tokenize_lower(keyword)
    if not tokens:
        return 0.0

    boost = 0.0
    matched_tokens = sum(1 for token in tokens if token in PHYSICS_TOKEN_HINTS)
    if matched_tokens:
        boost += min(0.22, 0.06 * matched_tokens)

    if any(phrase in keyword for phrase in PHYSICS_PHRASE_HINTS):
        boost += 0.12

    title_overlap = len(set(tokens) & title_tokens)
    if title_overlap:
        boost += min(0.12, 0.04 * title_overlap)

    if len(tokens) == 1 and tokens[0] in GENERIC_PENALTY_TOKENS:
        boost -= 0.08

    return clamp_float(boost, -0.1, 0.4)


def extract_keywords_from_text(
    kw_model: KeyBERT,
    text: str,
    title: str | None,
    min_ngram: int,
    max_ngram: int,
    candidate_top_n: int,
    top_n: int,
    stop_words: str | None,
    use_mmr: bool,
    diversity: float,
    physics_boost_enabled: bool,
) -> List[Dict[str, float | str]]:
    cleaned = clean_text(text)
    if not cleaned:
        return []

    raw_keywords = kw_model.extract_keywords(
        cleaned,
        keyphrase_ngram_range=(min_ngram, max_ngram),
        stop_words=stop_words,
        top_n=candidate_top_n,
        use_mmr=use_mmr,
        diversity=clamp_float(diversity, 0.0, 1.0),
    )

    title_tokens = set(tokenize_lower(title or ""))
    rescored: List[Dict[str, float | str]] = []
    seen: set[str] = set()
    for keyword, base_score in raw_keywords:
        normalized_keyword = clean_text(keyword).lower()
        if not normalized_keyword or normalized_keyword in seen:
            continue
        seen.add(normalized_keyword)

        boost = physics_boost(normalized_keyword, title_tokens) if physics_boost_enabled else 0.0
        final_score = float(base_score) + float(boost)
        rescored.append(
            {
                "keyword": normalized_keyword,
                "score": round(final_score, 6),
                "base_score": round(float(base_score), 6),
                "physics_boost": round(float(boost), 6),
            }
        )

    rescored.sort(key=lambda row: (row["score"], row["base_score"]), reverse=True)
    return rescored[:top_n]


def make_pg_identifier(raw: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]", "_", raw).lower()
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "keywords"


def truncate_pg_identifier(identifier: str, max_len: int = 63) -> str:
    if len(identifier) <= max_len:
        return identifier
    digest = hashlib.md5(identifier.encode("utf-8")).hexdigest()[:8]
    return f"{identifier[: max_len - 9]}_{digest}"


def auto_output_table_name(
    source_table: str,
    text_mode: str,
    physics_boost_enabled: bool,
    output_schema: str,
) -> str:
    _, source_table_name = parse_schema_table(source_table)
    mode_slug = make_pg_identifier(text_mode.replace("+", "_"))
    boost_slug = "boost_on" if physics_boost_enabled else "boost_off"
    raw_table_name = f"{source_table_name}_keywords_{mode_slug}_{boost_slug}"
    table_name = truncate_pg_identifier(make_pg_identifier(raw_table_name))
    return f"{output_schema}.{table_name}"


def ensure_output_table(conn, out_schema: str, out_table: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(out_schema))
        )
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.{} (
                    paper_id TEXT PRIMARY KEY,
                    categories TEXT,
                    source_title TEXT,
                    source_abstract TEXT,
                    text_mode TEXT NOT NULL,
                    physics_boost_enabled BOOLEAN NOT NULL,
                    keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
                    keyword_count INTEGER NOT NULL DEFAULT 0,
                    embedding_model TEXT NOT NULL,
                    min_ngram INTEGER NOT NULL,
                    max_ngram INTEGER NOT NULL,
                    candidate_top_n INTEGER NOT NULL,
                    top_n INTEGER NOT NULL,
                    diversity DOUBLE PRECISION NOT NULL,
                    stop_words TEXT,
                    use_mmr BOOLEAN NOT NULL,
                    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            ).format(sql.Identifier(out_schema), sql.Identifier(out_table))
        )
    conn.commit()


def build_source_filters(
    source_alias: str,
    physics_only: bool,
    categories_column: str | None,
    physics_ids: Sequence[str],
) -> Tuple[List[sql.Composed], List[Any]]:
    clauses: List[sql.Composed] = [sql.SQL("TRUE")]
    params: List[Any] = []

    if physics_only:
        if not categories_column:
            raise ValueError("input.categories_column is required when input.physics_only=true.")
        clauses.append(
            sql.SQL("{}.{} IS NOT NULL").format(
                sql.Identifier(source_alias), sql.Identifier(categories_column)
            )
        )
        clauses.append(
            sql.SQL("string_to_array({}.{}, ' ') && %s::text[]").format(
                sql.Identifier(source_alias), sql.Identifier(categories_column)
            )
        )
        params.append(list(physics_ids))

    return clauses, params


def get_counts(
    conn,
    source_schema: str,
    source_table: str,
    output_schema: str,
    output_table: str,
    source_id_column: str,
    physics_only: bool,
    categories_column: str | None,
    physics_ids: Sequence[str],
) -> Tuple[int, int]:
    clauses, params = build_source_filters(
        source_alias="s",
        physics_only=physics_only,
        categories_column=categories_column,
        physics_ids=physics_ids,
    )
    where_sql = sql.SQL(" AND ").join(clauses)

    with conn.cursor() as cur:
        total_query = sql.SQL(
            """
            SELECT COUNT(*)
            FROM {}.{} s
            WHERE {}
            """
        ).format(sql.Identifier(source_schema), sql.Identifier(source_table), where_sql)
        cur.execute(total_query, params)
        total_rows = int(cur.fetchone()[0])

        done_query = sql.SQL(
            """
            SELECT COUNT(*)
            FROM {}.{} s
            JOIN {}.{} o
              ON o.paper_id = s.{}
            WHERE {}
            """
        ).format(
            sql.Identifier(source_schema),
            sql.Identifier(source_table),
            sql.Identifier(output_schema),
            sql.Identifier(output_table),
            sql.Identifier(source_id_column),
            where_sql,
        )
        cur.execute(done_query, params)
        already_done = int(cur.fetchone()[0])

    return total_rows, already_done


def build_select_query(
    source_schema: str,
    source_table: str,
    output_schema: str,
    output_table: str,
    source_id_column: str,
    title_column: str,
    abstract_column: str,
    categories_column: str | None,
    force_recompute: bool,
    use_limit: bool,
    physics_only: bool,
    physics_ids: Sequence[str],
) -> Tuple[sql.SQL, Tuple[Any, ...]]:
    clauses, params = build_source_filters(
        source_alias="s",
        physics_only=physics_only,
        categories_column=categories_column,
        physics_ids=physics_ids,
    )

    join_clause: sql.SQL
    if force_recompute:
        join_clause = sql.SQL("")
    else:
        join_clause = sql.SQL(
            """
            LEFT JOIN {}.{} o
              ON o.paper_id = s.{}
            """
        ).format(
            sql.Identifier(output_schema),
            sql.Identifier(output_table),
            sql.Identifier(source_id_column),
        )
        clauses.append(sql.SQL("o.paper_id IS NULL"))

    categories_expr = (
        sql.SQL("s.{}").format(sql.Identifier(categories_column))
        if categories_column
        else sql.SQL("NULL::text")
    )

    where_sql = sql.SQL(" AND ").join(clauses)
    limit_sql = sql.SQL("LIMIT %s") if use_limit else sql.SQL("")

    query = sql.SQL(
        """
        SELECT
            s.{},
            s.{},
            s.{},
            {}
        FROM {}.{} s
        {}
        WHERE {}
        ORDER BY s.{}
        {}
        """
    ).format(
        sql.Identifier(source_id_column),
        sql.Identifier(title_column),
        sql.Identifier(abstract_column),
        categories_expr,
        sql.Identifier(source_schema),
        sql.Identifier(source_table),
        join_clause,
        where_sql,
        sql.Identifier(source_id_column),
        limit_sql,
    )

    return query, tuple(params)


def write_keywords(
    conn,
    output_schema: str,
    output_table: str,
    rows: Sequence[Tuple[Any, ...]],
    page_size: int,
) -> None:
    if not rows:
        return

    insert_query = sql.SQL(
        """
        INSERT INTO {}.{} (
            paper_id,
            categories,
            source_title,
            source_abstract,
            text_mode,
            physics_boost_enabled,
            keywords,
            keyword_count,
            embedding_model,
            min_ngram,
            max_ngram,
            candidate_top_n,
            top_n,
            diversity,
            stop_words,
            use_mmr,
            extracted_at
        )
        VALUES %s
        ON CONFLICT (paper_id) DO UPDATE SET
            categories = EXCLUDED.categories,
            source_title = EXCLUDED.source_title,
            source_abstract = EXCLUDED.source_abstract,
            text_mode = EXCLUDED.text_mode,
            physics_boost_enabled = EXCLUDED.physics_boost_enabled,
            keywords = EXCLUDED.keywords,
            keyword_count = EXCLUDED.keyword_count,
            embedding_model = EXCLUDED.embedding_model,
            min_ngram = EXCLUDED.min_ngram,
            max_ngram = EXCLUDED.max_ngram,
            candidate_top_n = EXCLUDED.candidate_top_n,
            top_n = EXCLUDED.top_n,
            diversity = EXCLUDED.diversity,
            stop_words = EXCLUDED.stop_words,
            use_mmr = EXCLUDED.use_mmr,
            extracted_at = EXCLUDED.extracted_at
        """
    ).format(sql.Identifier(output_schema), sql.Identifier(output_table))

    with conn.cursor() as cur:
        extras.execute_values(cur, insert_query.as_string(conn), rows, page_size=page_size)
    conn.commit()


def build_rows_for_batch(
    batch: Sequence[Tuple[Any, str | None, str | None, str | None]],
    kw_model: KeyBERT,
    text_mode: str,
    include_source_text: bool,
    model_name: str,
    min_ngram: int,
    max_ngram: int,
    candidate_top_n: int,
    top_n: int,
    stop_words: str | None,
    use_mmr: bool,
    diversity: float,
    physics_boost_enabled: bool,
) -> List[Tuple[Any, ...]]:
    now = datetime.now(timezone.utc)
    out_rows: List[Tuple[Any, ...]] = []

    for paper_id, title, abstract, categories in batch:
        input_text = select_input_text(title, abstract, text_mode)
        keywords = extract_keywords_from_text(
            kw_model=kw_model,
            text=input_text,
            title=title,
            min_ngram=min_ngram,
            max_ngram=max_ngram,
            candidate_top_n=candidate_top_n,
            top_n=top_n,
            stop_words=stop_words,
            use_mmr=use_mmr,
            diversity=diversity,
            physics_boost_enabled=physics_boost_enabled,
        )

        out_rows.append(
            (
                str(paper_id),
                categories,
                title if include_source_text else None,
                abstract if include_source_text else None,
                text_mode,
                physics_boost_enabled,
                extras.Json(keywords),
                len(keywords),
                model_name,
                min_ngram,
                max_ngram,
                candidate_top_n,
                top_n,
                diversity,
                stop_words,
                use_mmr,
                now,
            )
        )

    return out_rows


def validate_config(config: Dict[str, Any]) -> None:
    if "database" not in config or "input" not in config or "keywords" not in config:
        raise ValueError("Config must include: database, input, keywords sections.")

    input_cfg = config["input"]
    keyword_cfg = config["keywords"]
    output_cfg = config["output"]
    runtime_cfg = config["runtime"]

    input_cfg["text_mode"] = normalize_text_mode(str(input_cfg["text_mode"]))
    input_cfg["physics_only"] = as_bool(input_cfg["physics_only"])
    input_cfg["force_recompute"] = as_bool(input_cfg["force_recompute"])
    input_cfg["fetch_batch_size"] = int(input_cfg["fetch_batch_size"])
    input_cfg["limit"] = (
        None if input_cfg["limit"] in (None, "", "null") else int(input_cfg["limit"])
    )
    if input_cfg["fetch_batch_size"] < 1:
        raise ValueError("input.fetch_batch_size must be >= 1.")
    if input_cfg["limit"] is not None and input_cfg["limit"] < 1:
        raise ValueError("input.limit must be >= 1 when provided.")

    keyword_cfg["min_ngram"] = int(keyword_cfg["min_ngram"])
    keyword_cfg["max_ngram"] = int(keyword_cfg["max_ngram"])
    keyword_cfg["candidate_top_n"] = int(keyword_cfg["candidate_top_n"])
    keyword_cfg["top_n"] = int(keyword_cfg["top_n"])
    keyword_cfg["diversity"] = float(keyword_cfg["diversity"])
    keyword_cfg["use_mmr"] = as_bool(keyword_cfg["use_mmr"])
    keyword_cfg["physics_boost_enabled"] = as_bool(keyword_cfg["physics_boost_enabled"])
    keyword_cfg["stop_words"] = normalize_stop_words(keyword_cfg["stop_words"])
    if keyword_cfg["min_ngram"] < 1 or keyword_cfg["max_ngram"] < 1:
        raise ValueError("keywords.min_ngram and keywords.max_ngram must be >= 1.")
    if keyword_cfg["min_ngram"] > keyword_cfg["max_ngram"]:
        raise ValueError("keywords.min_ngram cannot exceed keywords.max_ngram.")
    if keyword_cfg["top_n"] < 1 or keyword_cfg["candidate_top_n"] < 1:
        raise ValueError("keywords.top_n and keywords.candidate_top_n must be >= 1.")
    if keyword_cfg["candidate_top_n"] < keyword_cfg["top_n"]:
        raise ValueError("keywords.candidate_top_n must be >= keywords.top_n.")
    if not (0.0 <= keyword_cfg["diversity"] <= 1.0):
        raise ValueError("keywords.diversity must be in [0, 1].")

    output_cfg["include_source_text"] = as_bool(output_cfg["include_source_text"])
    output_cfg["write_batch_size"] = int(output_cfg["write_batch_size"])
    output_cfg["schema"] = clean_text(str(output_cfg.get("schema") or ""))
    if output_cfg["write_batch_size"] < 1:
        raise ValueError("output.write_batch_size must be >= 1.")

    runtime_cfg["dry_run"] = as_bool(runtime_cfg["dry_run"])


def run(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    config = load_yaml_config(config_path)

    if args.limit is not None:
        config["input"]["limit"] = args.limit
    if args.force_recompute:
        config["input"]["force_recompute"] = True
    if args.dry_run:
        config["runtime"]["dry_run"] = True

    validate_config(config)

    env_file = Path(config["env_file"])
    dotenv = load_dotenv_simple(env_file)

    db_cfg = config["database"]
    input_cfg = config["input"]
    keyword_cfg = config["keywords"]
    output_cfg = config["output"]
    runtime_cfg = config["runtime"]

    db_host = db_cfg.get("host") or get_env_or_default(dotenv, "DB_HOST")
    db_port = int(db_cfg.get("port") or get_env_or_default(dotenv, "DB_PORT", "5432"))
    db_name = db_cfg.get("name") or get_env_or_default(dotenv, "DB_NAME")
    db_user = db_cfg.get("user") or get_env_or_default(dotenv, "DB_USER")
    db_password = db_cfg.get("password") or get_env_or_default(dotenv, "DB_PASSWORD")
    if not all([db_host, db_name, db_user, db_password]):
        raise RuntimeError(
            "Missing DB credentials. Set in YAML database section or .env / env vars."
        )

    source_schema, source_table = parse_schema_table(str(input_cfg["source_table"]))
    output_schema_cfg = clean_text(str(output_cfg.get("schema") or ""))
    if not output_schema_cfg:
        output_schema_cfg = "classifications_and_keywords"

    output_table_cfg = clean_text(str(output_cfg.get("table")))
    output_table_cfg_lower = output_table_cfg.lower()
    if output_table_cfg_lower in {"", "auto", "none", "null"}:
        output_table_name = auto_output_table_name(
            source_table=str(input_cfg["source_table"]),
            text_mode=str(input_cfg["text_mode"]),
            physics_boost_enabled=bool(keyword_cfg["physics_boost_enabled"]),
            output_schema=output_schema_cfg,
        )
    else:
        if "." in output_table_cfg:
            output_table_name = output_table_cfg
        else:
            output_table_name = f"{output_schema_cfg}.{output_table_cfg}"
    output_schema, output_table = parse_schema_table(output_table_name)

    categories_column = input_cfg.get("categories_column")
    categories_column_clean = clean_text(str(categories_column or "")).lower()
    if categories_column_clean in {"", "none", "null"}:
        categories_column = None

    physics_ids: List[str] = []
    if bool(input_cfg["physics_only"]):
        physics_ids = load_physics_ids(Path(str(input_cfg["physics_categories_csv"])))
        print(
            f"[setup] loaded {len(physics_ids)} physics category IDs "
            f"from {input_cfg['physics_categories_csv']}"
        )

    device = resolve_device(str(keyword_cfg["device"]))
    print(f"[setup] config={config_path}")
    print(f"[setup] source={source_schema}.{source_table}")
    print(f"[setup] output={output_schema}.{output_table}")
    print(
        "[setup] mode="
        f"{input_cfg['text_mode']} | boost={keyword_cfg['physics_boost_enabled']} | "
        f"physics_only={input_cfg['physics_only']} | dry_run={runtime_cfg['dry_run']}"
    )
    print(f"[model] loading encoder={keyword_cfg['model']} on device={device}")

    encoder = SentenceTransformer(str(keyword_cfg["model"]), device=device)
    kw_model = KeyBERT(model=encoder)

    write_conn = psycopg2.connect(
        host=db_host, port=db_port, dbname=db_name, user=db_user, password=db_password
    )
    write_conn.autocommit = False

    read_conn = psycopg2.connect(
        host=db_host, port=db_port, dbname=db_name, user=db_user, password=db_password
    )
    read_conn.autocommit = False

    try:
        ensure_output_table(write_conn, output_schema, output_table)

        total_rows, already_done = get_counts(
            conn=write_conn,
            source_schema=source_schema,
            source_table=source_table,
            output_schema=output_schema,
            output_table=output_table,
            source_id_column=str(input_cfg["id_column"]),
            physics_only=bool(input_cfg["physics_only"]),
            categories_column=categories_column,
            physics_ids=physics_ids,
        )
        pending = (
            total_rows
            if bool(input_cfg["force_recompute"])
            else max(total_rows - already_done, 0)
        )
        print(
            f"[setup] source rows={total_rows:,}, already in output={already_done:,}, pending={pending:,}"
        )

        select_query, params = build_select_query(
            source_schema=source_schema,
            source_table=source_table,
            output_schema=output_schema,
            output_table=output_table,
            source_id_column=str(input_cfg["id_column"]),
            title_column=str(input_cfg["title_column"]),
            abstract_column=str(input_cfg["abstract_column"]),
            categories_column=categories_column,
            force_recompute=bool(input_cfg["force_recompute"]),
            use_limit=bool(input_cfg["limit"]),
            physics_only=bool(input_cfg["physics_only"]),
            physics_ids=physics_ids,
        )
        if input_cfg["limit"] is not None:
            params = (*params, int(input_cfg["limit"]))

        select_cur = read_conn.cursor(name="keyword_extract_cursor")
        select_cur.itersize = int(input_cfg["fetch_batch_size"])
        select_cur.execute(select_query, params)

        processed = 0
        start = time.time()

        while True:
            batch = select_cur.fetchmany(int(input_cfg["fetch_batch_size"]))
            if not batch:
                break

            out_rows = build_rows_for_batch(
                batch=batch,
                kw_model=kw_model,
                text_mode=str(input_cfg["text_mode"]),
                include_source_text=bool(output_cfg["include_source_text"]),
                model_name=str(keyword_cfg["model"]),
                min_ngram=int(keyword_cfg["min_ngram"]),
                max_ngram=int(keyword_cfg["max_ngram"]),
                candidate_top_n=int(keyword_cfg["candidate_top_n"]),
                top_n=int(keyword_cfg["top_n"]),
                stop_words=keyword_cfg["stop_words"],
                use_mmr=bool(keyword_cfg["use_mmr"]),
                diversity=float(keyword_cfg["diversity"]),
                physics_boost_enabled=bool(keyword_cfg["physics_boost_enabled"]),
            )

            if not bool(runtime_cfg["dry_run"]):
                write_keywords(
                    conn=write_conn,
                    output_schema=output_schema,
                    output_table=output_table,
                    rows=out_rows,
                    page_size=int(output_cfg["write_batch_size"]),
                )

            processed += len(batch)
            elapsed = time.time() - start
            rate = processed / elapsed if elapsed > 0 else 0.0
            eta_text = "n/a"
            if pending > 0 and rate > 0:
                remaining = max(pending - processed, 0)
                eta_seconds = int(remaining / rate)
                eta_text = f"{eta_seconds // 3600:02d}h {(eta_seconds % 3600) // 60:02d}m"
            print(
                f"[progress] processed={processed:,} rows | rate={rate:,.2f} rows/s | eta={eta_text}"
            )

        select_cur.close()

        total_elapsed = time.time() - start
        final_rate = processed / total_elapsed if total_elapsed > 0 else 0.0
        print(
            f"[done] processed={processed:,} rows in {total_elapsed/60:.1f} min "
            f"({final_rate:,.2f} rows/s)"
        )
    finally:
        read_conn.close()
        write_conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract physics-emphasized keywords from Postgres rows using YAML config."
    )
    parser.add_argument(
        "--config",
        default="extract_keywords_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional override for input.limit in YAML",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Override YAML and recompute keywords for all selected source rows",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run extraction without writing output rows",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
