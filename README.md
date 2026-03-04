# arxiv_physh

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
- `arxiv_base.arxiv_from_kaggle_physh_predictions`

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

## Resumability

- Upsert key is `paper_id`
- Non-force mode processes papers that are missing predictions for the current threshold
- Re-running with a new threshold will reprocess rows for that threshold
