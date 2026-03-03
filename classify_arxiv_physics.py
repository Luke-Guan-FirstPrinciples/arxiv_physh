#!/usr/bin/env python3
"""
Classify physics-related arXiv papers from Postgres with PhySH models.

Input source table:
  arxiv_base.arxiv_from_kaggle

Physics filter:
  Any row whose `categories` contains at least one arXiv category ID from
  arxiv_physics_categories.csv (the `id` column).

Output table (default):
  arxiv_base.arxiv_from_kaggle_physh_predictions

The script is resumable by default:
- it upserts by `paper_id`
- it skips already-classified rows unless --force-reclassify is used
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import psycopg2
import psycopg2.extras as extras
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from psycopg2 import sql
from sentence_transformers import SentenceTransformer


# -----------------------------------------------------------------------------
# Model definitions (must match training architecture)
# -----------------------------------------------------------------------------


class MultiLabelMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: Tuple[int, ...] = (1024, 512),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DisciplineConditionedMLP(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        discipline_dim: int,
        output_dim: int,
        hidden_layers: Tuple[int, ...] = (1024, 512),
        dropout: float = 0.3,
        discipline_dropout: float = 0.0,
        use_logits: bool = False,
    ):
        super().__init__()
        self.use_logits = use_logits
        self.discipline_dropout = nn.Dropout(discipline_dropout)

        layers: List[nn.Module] = []
        prev_dim = embedding_dim + discipline_dim
        for hidden_dim in hidden_layers:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(
        self, embedding: torch.Tensor, discipline_probs: torch.Tensor
    ) -> torch.Tensor:
        if self.use_logits:
            disc_features = torch.clamp(discipline_probs, 1e-7, 1 - 1e-7)
            disc_features = torch.log(disc_features / (1 - disc_features))
        else:
            disc_features = discipline_probs
        disc_features = self.discipline_dropout(disc_features)
        return self.network(torch.cat([embedding, disc_features], dim=1))


@dataclass
class LoadedModels:
    embedding_model: SentenceTransformer
    discipline_model: MultiLabelMLP
    concept_model: DisciplineConditionedMLP
    discipline_labels: List[Dict]
    concept_labels: List[Dict]
    device: str


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


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def combine_title_abstract(title: str | None, abstract: str | None) -> str:
    title_clean = clean_text(title)
    abstract_clean = clean_text(abstract)
    if title_clean and abstract_clean:
        return f"{title_clean} [SEP] {abstract_clean}"
    if title_clean:
        return title_clean
    return abstract_clean


def load_physics_ids(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Category CSV not found: {csv_path}")

    physics_ids: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "id" not in (reader.fieldnames or []):
            raise ValueError(f"CSV must contain 'id' column: {csv_path}")
        for row in reader:
            cat_id = (row.get("id") or "").strip()
            if cat_id:
                physics_ids.add(cat_id)

    return sorted(physics_ids)


def resolve_device(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models(
    model_repo: str,
    discipline_model_file: str,
    concept_model_file: str,
    embedding_model_name: str,
    device: str,
    hf_token: str | None,
) -> LoadedModels:
    print(f"[model] device={device}")
    print(f"[model] loading embedding model: {embedding_model_name}")
    embedding_model = SentenceTransformer(
        embedding_model_name,
        device=device,
        token=hf_token,
    )

    print(f"[model] downloading checkpoints from {model_repo}")
    discipline_path = hf_hub_download(repo_id=model_repo, filename=discipline_model_file)
    concept_path = hf_hub_download(repo_id=model_repo, filename=concept_model_file)

    disc_ckpt = torch.load(discipline_path, map_location=device, weights_only=False)
    disc_cfg = disc_ckpt["model_config"]
    discipline_model = MultiLabelMLP(
        disc_cfg["input_dim"],
        disc_cfg["output_dim"],
        tuple(disc_cfg["hidden_layers"]),
        disc_cfg["dropout"],
    )
    discipline_model.load_state_dict(disc_ckpt["model_state_dict"])
    discipline_model.to(device).eval()

    conc_ckpt = torch.load(concept_path, map_location=device, weights_only=False)
    conc_cfg = conc_ckpt["model_config"]
    concept_model = DisciplineConditionedMLP(
        conc_cfg["embedding_dim"],
        conc_cfg["discipline_dim"],
        conc_cfg["output_dim"],
        tuple(conc_cfg["hidden_layers"]),
        conc_cfg["dropout"],
        conc_cfg.get("discipline_dropout", 0.0),
        conc_cfg.get("use_logits", False),
    )
    concept_model.load_state_dict(conc_ckpt["model_state_dict"])
    concept_model.to(device).eval()

    print(
        "[model] loaded labels: "
        f"disciplines={len(disc_ckpt['class_labels'])}, "
        f"concepts={len(conc_ckpt['class_labels'])}"
    )

    return LoadedModels(
        embedding_model=embedding_model,
        discipline_model=discipline_model,
        concept_model=concept_model,
        discipline_labels=disc_ckpt["class_labels"],
        concept_labels=conc_ckpt["class_labels"],
        device=device,
    )


def ensure_output_table(conn, out_schema: str, out_table: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.{} (
                    paper_id TEXT PRIMARY KEY,
                    categories TEXT,
                    discipline_id UUID,
                    discipline_label TEXT NOT NULL,
                    discipline_score DOUBLE PRECISION NOT NULL,
                    concept_id UUID,
                    concept_label TEXT NOT NULL,
                    concept_score DOUBLE PRECISION NOT NULL,
                    model_repo TEXT NOT NULL,
                    discipline_model_file TEXT NOT NULL,
                    concept_model_file TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    classified_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            ).format(sql.Identifier(out_schema), sql.Identifier(out_table))
        )
    conn.commit()


def get_counts(
    conn,
    source_schema: str,
    source_table: str,
    out_schema: str,
    out_table: str,
    physics_ids: Sequence[str],
) -> Tuple[int, int]:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT COUNT(*)
                FROM {}.{}
                WHERE categories IS NOT NULL
                  AND string_to_array(categories, ' ') && %s::text[]
                """
            ).format(sql.Identifier(source_schema), sql.Identifier(source_table)),
            (list(physics_ids),),
        )
        total_physics = int(cur.fetchone()[0])

        cur.execute(
            sql.SQL(
                """
                SELECT COUNT(*)
                FROM {}.{} s
                JOIN {}.{} o
                  ON o.paper_id = s.id
                WHERE s.categories IS NOT NULL
                  AND string_to_array(s.categories, ' ') && %s::text[]
                """
            ).format(
                sql.Identifier(source_schema),
                sql.Identifier(source_table),
                sql.Identifier(out_schema),
                sql.Identifier(out_table),
            ),
            (list(physics_ids),),
        )
        already_done = int(cur.fetchone()[0])

    return total_physics, already_done


def build_select_query(
    source_schema: str,
    source_table: str,
    out_schema: str,
    out_table: str,
    force_reclassify: bool,
    use_limit: bool,
) -> sql.SQL:
    if force_reclassify:
        if use_limit:
            return sql.SQL(
                """
                SELECT s.id, s.title, s.abstract, s.categories
                FROM {}.{} s
                WHERE s.categories IS NOT NULL
                  AND string_to_array(s.categories, ' ') && %s::text[]
                ORDER BY s.id
                LIMIT %s
                """
            ).format(sql.Identifier(source_schema), sql.Identifier(source_table))

        return sql.SQL(
            """
            SELECT s.id, s.title, s.abstract, s.categories
            FROM {}.{} s
            WHERE s.categories IS NOT NULL
              AND string_to_array(s.categories, ' ') && %s::text[]
            ORDER BY s.id
            """
        ).format(sql.Identifier(source_schema), sql.Identifier(source_table))

    if use_limit:
        return sql.SQL(
            """
            SELECT s.id, s.title, s.abstract, s.categories
            FROM {}.{} s
            LEFT JOIN {}.{} o
              ON o.paper_id = s.id
            WHERE s.categories IS NOT NULL
              AND string_to_array(s.categories, ' ') && %s::text[]
              AND o.paper_id IS NULL
            ORDER BY s.id
            LIMIT %s
            """
        ).format(
            sql.Identifier(source_schema),
            sql.Identifier(source_table),
            sql.Identifier(out_schema),
            sql.Identifier(out_table),
        )

    return sql.SQL(
        """
        SELECT s.id, s.title, s.abstract, s.categories
        FROM {}.{} s
        LEFT JOIN {}.{} o
          ON o.paper_id = s.id
        WHERE s.categories IS NOT NULL
          AND string_to_array(s.categories, ' ') && %s::text[]
          AND o.paper_id IS NULL
        ORDER BY s.id
        """
    ).format(
        sql.Identifier(source_schema),
        sql.Identifier(source_table),
        sql.Identifier(out_schema),
        sql.Identifier(out_table),
    )


def write_predictions(
    conn,
    out_schema: str,
    out_table: str,
    rows: Sequence[Tuple],
) -> None:
    if not rows:
        return

    insert_query = sql.SQL(
        """
        INSERT INTO {}.{} (
            paper_id,
            categories,
            discipline_id,
            discipline_label,
            discipline_score,
            concept_id,
            concept_label,
            concept_score,
            model_repo,
            discipline_model_file,
            concept_model_file,
            embedding_model,
            classified_at
        )
        VALUES %s
        ON CONFLICT (paper_id) DO UPDATE SET
            categories = EXCLUDED.categories,
            discipline_id = EXCLUDED.discipline_id,
            discipline_label = EXCLUDED.discipline_label,
            discipline_score = EXCLUDED.discipline_score,
            concept_id = EXCLUDED.concept_id,
            concept_label = EXCLUDED.concept_label,
            concept_score = EXCLUDED.concept_score,
            model_repo = EXCLUDED.model_repo,
            discipline_model_file = EXCLUDED.discipline_model_file,
            concept_model_file = EXCLUDED.concept_model_file,
            embedding_model = EXCLUDED.embedding_model,
            classified_at = EXCLUDED.classified_at
        """
    ).format(sql.Identifier(out_schema), sql.Identifier(out_table))

    with conn.cursor() as cur:
        extras.execute_values(cur, insert_query.as_string(conn), rows, page_size=1000)
    conn.commit()


def classify_batch(
    models: LoadedModels,
    paper_rows: Sequence[Tuple[str, str, str, str]],
    embedding_batch_size: int,
    model_repo: str,
    discipline_model_file: str,
    concept_model_file: str,
    embedding_model_name: str,
) -> List[Tuple]:
    texts = [combine_title_abstract(title, abstract) for _, title, abstract, _ in paper_rows]

    embeddings = models.embedding_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=False,
        batch_size=embedding_batch_size,
    )
    embedding_tensor = embeddings.to(models.device)

    with torch.no_grad():
        discipline_logits = models.discipline_model(embedding_tensor)
        discipline_probs_tensor = torch.sigmoid(discipline_logits)

        concept_logits = models.concept_model(embedding_tensor, discipline_probs_tensor)
        concept_probs_tensor = torch.sigmoid(concept_logits)

    discipline_probs = discipline_probs_tensor.cpu().numpy()
    concept_probs = concept_probs_tensor.cpu().numpy()

    discipline_top_idx = np.argmax(discipline_probs, axis=1)
    concept_top_idx = np.argmax(concept_probs, axis=1)

    now_ts = datetime.now(timezone.utc)
    out_rows: List[Tuple] = []
    for i, (paper_id, _, _, categories) in enumerate(paper_rows):
        d_idx = int(discipline_top_idx[i])
        c_idx = int(concept_top_idx[i])

        d_meta = models.discipline_labels[d_idx]
        c_meta = models.concept_labels[c_idx]

        out_rows.append(
            (
                paper_id,
                categories,
                d_meta.get("discipline_id"),
                d_meta.get("label", f"Discipline_{d_idx}"),
                float(discipline_probs[i, d_idx]),
                c_meta.get("concept_id"),
                c_meta.get("label", f"Concept_{c_idx}"),
                float(concept_probs[i, c_idx]),
                model_repo,
                discipline_model_file,
                concept_model_file,
                embedding_model_name,
                now_ts,
            )
        )

    return out_rows


def run(args: argparse.Namespace) -> None:
    dotenv_path = Path(args.env_file)
    dotenv = load_dotenv_simple(dotenv_path)

    db_host = args.db_host or get_env_or_default(dotenv, "DB_HOST")
    db_port = int(args.db_port or get_env_or_default(dotenv, "DB_PORT", "5432"))
    db_name = args.db_name or get_env_or_default(dotenv, "DB_NAME")
    db_user = args.db_user or get_env_or_default(dotenv, "DB_USER")
    db_password = args.db_password or get_env_or_default(dotenv, "DB_PASSWORD")

    if not all([db_host, db_name, db_user, db_password]):
        raise RuntimeError(
            "Missing DB credentials. Set in .env or pass --db-host/--db-name/--db-user/--db-password."
        )

    source_schema, source_table = parse_schema_table(args.source_table)
    out_schema, out_table = parse_schema_table(args.output_table)

    physics_ids = load_physics_ids(Path(args.physics_categories_csv))
    print(f"[setup] loaded {len(physics_ids)} physics arXiv IDs from {args.physics_categories_csv}")

    device = resolve_device(args.device)
    hf_token = args.hf_token or get_env_or_default(dotenv, "HF_TOKEN")
    models = load_models(
        model_repo=args.model_repo,
        discipline_model_file=args.discipline_model_file,
        concept_model_file=args.concept_model_file,
        embedding_model_name=args.embedding_model,
        device=device,
        hf_token=hf_token,
    )

    write_conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
    )
    write_conn.autocommit = False

    # Separate read connection keeps server-side cursor valid while write commits happen.
    read_conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
    )
    read_conn.autocommit = False

    try:
        ensure_output_table(write_conn, out_schema, out_table)

        total_physics, already_done = get_counts(
            write_conn,
            source_schema,
            source_table,
            out_schema,
            out_table,
            physics_ids,
        )
        pending = total_physics if args.force_reclassify else max(total_physics - already_done, 0)
        print(
            f"[setup] source physics papers={total_physics:,}, already in output={already_done:,}, pending={pending:,}"
        )

        use_limit = args.limit is not None
        select_query = build_select_query(
            source_schema,
            source_table,
            out_schema,
            out_table,
            args.force_reclassify,
            use_limit,
        )

        params: Tuple = (physics_ids, args.limit) if use_limit else (physics_ids,)

        select_cur = read_conn.cursor(name="physics_paper_cursor")
        select_cur.itersize = args.fetch_batch_size
        select_cur.execute(select_query, params)

        processed = 0
        start = time.time()

        while True:
            batch = select_cur.fetchmany(args.fetch_batch_size)
            if not batch:
                break

            predicted_rows = classify_batch(
                models=models,
                paper_rows=batch,
                embedding_batch_size=args.embedding_batch_size,
                model_repo=args.model_repo,
                discipline_model_file=args.discipline_model_file,
                concept_model_file=args.concept_model_file,
                embedding_model_name=args.embedding_model,
            )

            if not args.dry_run:
                write_predictions(write_conn, out_schema, out_table, predicted_rows)

            processed += len(batch)
            elapsed = time.time() - start
            rate = processed / elapsed if elapsed > 0 else 0.0
            eta_text = "n/a"
            if pending > 0 and rate > 0:
                remaining = max(pending - processed, 0)
                eta_seconds = int(remaining / rate)
                eta_text = f"{eta_seconds // 3600:02d}h {(eta_seconds % 3600) // 60:02d}m"

            print(
                f"[progress] processed={processed:,} rows | "
                f"rate={rate:,.1f} rows/s | eta={eta_text}"
            )

        select_cur.close()

        total_elapsed = time.time() - start
        final_rate = processed / total_elapsed if total_elapsed > 0 else 0.0
        print(
            f"[done] processed={processed:,} rows in {total_elapsed/60:.1f} min "
            f"({final_rate:,.1f} rows/s)"
        )

    finally:
        read_conn.close()
        write_conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify physics-related arXiv papers with PhySH discipline/concept models."
    )

    parser.add_argument("--env-file", default=".env", help="Path to .env file")

    parser.add_argument("--db-host", default=None)
    parser.add_argument("--db-port", default=None)
    parser.add_argument("--db-name", default=None)
    parser.add_argument("--db-user", default=None)
    parser.add_argument("--db-password", default=None)

    parser.add_argument(
        "--source-table",
        default="arxiv_base.arxiv_from_kaggle",
        help="Source table with arXiv records",
    )
    parser.add_argument(
        "--output-table",
        default="arxiv_base.arxiv_from_kaggle_physh_predictions",
        help="Output table for predictions",
    )
    parser.add_argument(
        "--physics-categories-csv",
        default="arxiv_physics_categories.csv",
        help="CSV containing physics arXiv IDs in `id` column",
    )

    parser.add_argument(
        "--model-repo",
        default="LukeFP/physh_topic_supervised_classifier",
        help="Hugging Face repo ID with the classifier checkpoints",
    )
    parser.add_argument(
        "--discipline-model-file",
        default="discipline_classifier_gemma_20260130_140842.pt",
    )
    parser.add_argument(
        "--concept-model-file",
        default="concept_conditioned_gemma_20260130_140842.pt",
    )
    parser.add_argument(
        "--embedding-model",
        default="google/embeddinggemma-300m",
        help="SentenceTransformer model name",
    )
    parser.add_argument("--hf-token", default=None, help="Optional HF token")

    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--fetch-batch-size", type=int, default=32)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows to process")

    parser.add_argument(
        "--force-reclassify",
        action="store_true",
        help="Reprocess rows even if already in output table",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run inference without DB writes")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    run(parser.parse_args())
