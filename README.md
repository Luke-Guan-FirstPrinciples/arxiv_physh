# Keyword extraction on buzz
HF_HOME=~/.cache/huggingface python  extract_arxiv_physics_keywords.py --config extract_keywords_config_abstract_boost_on.yaml

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
- `keywords.quota_enabled`: `true` enables per-ngram quota mode (`(1,1)`, `(2,2)`, `(3,3)`)
- `keywords.quota_min_1gram` / `quota_min_2gram` / `quota_min_3gram`: minimum picks per n-gram bucket
- `keywords.quota_min_score`: global score floor used before quota backfill
- `keywords.top_n`: final output size (must be `>= quota_min_1gram + quota_min_2gram + quota_min_3gram`)
- `output.schema`: default target schema (`classifications_and_keywords`)
- `output.table`: `auto` (auto-generated table name from mode + boost + quota policy in `output.schema`) or explicit `schema.table`
- `input.physics_only`: `true` to filter by `arxiv_physics_categories.csv`, `false` for all categories

With quota mode enabled, auto table names include a policy suffix:
- `..._q1{min1}_q2{min2}_q3{min3}_top{top_n}_ms{min_score_percent}`

## Resumability

- Upsert key is `paper_id`
- Non-force mode processes papers that are missing predictions for the current threshold
- Re-running with a new threshold will reprocess rows for that threshold

## Keyword distribution analysis

For a keyword table such as
`classifications_and_keywords.arxiv_from_kaggle_keywords_title_abstract_boost_on_q15_57588cc1`,
you can export distribution summaries plus a visualization dashboard with:

```bash
python3 analyze_keyword_distribution.py \
  --table classifications_and_keywords.arxiv_from_kaggle_keywords_title_abstract_boost_on_q15_57588cc1
```

Default outputs go under:

```bash
analysis_output/classifications_and_keywords.arxiv_from_kaggle_keywords_title_abstract_boost_on_q15_57588cc1/
```

Files written:
- `index.html`
- `summary.json`
- `top_keywords.csv`
- `top_boosted_keywords.csv`
- `ngram_distribution.csv`
- `rank_distribution.csv`
- `keyword_count_distribution.csv`
- `score_distribution.csv`
- `boost_distribution.csv`
- `visualizations/top_keywords_by_paper_frequency.svg`
- `visualizations/top_boosted_keywords.svg`
- `visualizations/ngram_distribution.svg`
- `visualizations/keyword_rank_distribution.svg`
- `visualizations/score_histogram.svg`
- `visualizations/physics_boost_histogram.svg`
- `visualizations/keyword_count_distribution.svg`

If you want direct SQL instead of CSV exports, use:

```bash
psql "$DATABASE_URL" \
  -v target_table=classifications_and_keywords.arxiv_from_kaggle_keywords_title_abstract_boost_on_q15_57588cc1 \
  -f sql/keyword_distribution_queries.sql
```

## Focused keyword-topic analysis

To analyze a curated set of topics such as quantum-information keywords, run:

```bash
python3 analyze_keyword_focus.py \
  --table classifications_and_keywords.arxiv_from_kaggle_keywords_title_abstract_boost_on_q15_57588cc1 \
  --config keyword_focus_quantum_information.yaml
```

This writes a focused dashboard plus CSV outputs under:

```bash
analysis_output/classifications_and_keywords.arxiv_from_kaggle_keywords_title_abstract_boost_on_q15_57588cc1/keyword_focus_quantum_information/
```

Files written:
- `index.html`
- `manifest.json`
- `focus_group_summary.csv`
- `focus_group_variants.csv`
- `focus_group_rank_distribution.csv`
- `visualizations/grouped_paper_frequency.svg`
- `visualizations/exact_paper_frequency.svg`
- `visualizations/matched_variant_count.svg`
