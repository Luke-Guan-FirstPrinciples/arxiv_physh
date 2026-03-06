# arxiv_physh

# quick smoke test
python3 classify_arxiv_physics.py --limit 100

# set threshold (default is 0.85)
python3 classify_arxiv_physics.py --threshold 0.9

# force reclassify everything (ignore existing rows)
python3 classify_arxiv_physics.py --force-reclassify

# force specific device
python3 classify_arxiv_physics.py --device cuda
python3 classify_arxiv_physics.py --device mps
python3 classify_arxiv_physics.py --device cpu


Batch classification of physics-related arXiv papers in Postgres using:
- `google/embeddinggemma-300m` embeddings
- `LukeFP/physh_topic_supervised_classifier` heads for:
  - PhySH Discipline
  - PhySH Concept

Source table:
- `arxiv_base.arxiv_from_kaggle`

Physics filter:
- uses `arxiv_physics_categories.csv` (`id` column)

Output table:
- `classifications_and_keywords.arxiv_from_kaggle_physh_predictions`

## Prediction behavior

This script uses **thresholded multi-label** selection for both discipline and concept.

- Default threshold: `0.85`
- Configurable via `--threshold`
- If no class passes threshold, it falls back to **top-1**

The output row stores:
- top-1 discipline/concept in scalar columns (`discipline_*`, `concept_*`)
- full selected labels in JSONB columns:
  - `discipline_predictions`
  - `concept_predictions`
- threshold used for that run: `prediction_threshold`

JSON entries include:
- `id` (`discipline_id` or `concept_id`)
- `label`
- `score`
- `rank`
- `selected_by` (`threshold` or `top1_fallback`)

## Run

From this folder:

```bash
python3 classify_arxiv_physics.py
```

Useful options:

```bash
# Use a different threshold
python3 classify_arxiv_physics.py --threshold 0.90

# Small smoke test
python3 classify_arxiv_physics.py --limit 100 --fetch-batch-size 16 --embedding-batch-size 16

# Reclassify even if already present
python3 classify_arxiv_physics.py --force-reclassify
```

## Physics keyword extraction (PoC)

Config file:

```bash
extract_keywords_config.yaml
```

Run extraction:

```bash
python3 extract_arxiv_physics_keywords.py --config extract_keywords_config.yaml
```

Useful overrides:

```bash
# Test first N rows without writing output table rows
python3 extract_arxiv_physics_keywords.py --config extract_keywords_config.yaml --limit 100 --dry-run

# Recompute even if output table already contains paper_id rows
python3 extract_arxiv_physics_keywords.py --config extract_keywords_config.yaml --force-recompute
```

Main config toggles in YAML:
- `input.text_mode`: `title` | `abstract` | `title+abstract`
- `keywords.physics_boost_enabled`: `true` | `false`
- `output.schema`: default target schema (`classifications_and_keywords`)
- `output.table`: `auto` (auto-generated table name from mode + boost flag in `output.schema`) or explicit `schema.table`
- `input.physics_only`: `true` to filter by `arxiv_physics_categories.csv`, `false` for all categories

## Resumability

- Upsert key is `paper_id`
- Non-force mode processes papers that are missing predictions for the current threshold
- Re-running with a new threshold will reprocess rows for that threshold
