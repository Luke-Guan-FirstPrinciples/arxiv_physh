-- Keyword distribution queries for a table produced by
-- extract_arxiv_physics_keywords.py.
--
-- Usage:
--   psql "$DATABASE_URL" \
--     -v target_table=classifications_and_keywords.arxiv_from_kaggle_keywords_title_abstract_boost_on_q15_57588cc1 \
--     -v top_n=50 \
--     -v score_bucket_width=0.05 \
--     -v boost_bucket_width=0.02 \
--     -f sql/keyword_distribution_queries.sql
--
-- Notes:
-- - Pass `target_table` without quotes because psql performs text substitution.
-- - The queries assume the table has columns:
--     paper_id TEXT, keyword_count INTEGER, keywords JSONB

\set target_table classifications_and_keywords.arxiv_from_kaggle_keywords_title_abstract_boost_on_q15_57588cc1
\set top_n 50
\set score_bucket_width 0.05
\set boost_bucket_width 0.02

\echo ''
\echo '== Summary =='
WITH table_rows AS (
    SELECT
        paper_id,
        keyword_count,
        keywords
    FROM :target_table
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
    (SELECT COUNT(DISTINCT paper_id)::bigint FROM keyword_rows) AS papers_with_keywords,
    (SELECT COUNT(DISTINCT keyword)::bigint FROM keyword_rows) AS unique_keywords,
    (SELECT ROUND(AVG(score)::numeric, 4) FROM keyword_rows) AS avg_score,
    (SELECT ROUND(AVG(base_score)::numeric, 4) FROM keyword_rows) AS avg_base_score,
    (SELECT ROUND(AVG(physics_boost)::numeric, 4) FROM keyword_rows) AS avg_physics_boost;

\echo ''
\echo '== Top Keywords =='
WITH table_rows AS (
    SELECT
        paper_id,
        keyword_count,
        keywords
    FROM :target_table
),
keyword_rows AS (
    SELECT
        t.paper_id,
        kw.ordinality::integer AS keyword_rank,
        kw.item ->> 'keyword' AS keyword,
        NULLIF(kw.item ->> 'ngram', '')::integer AS ngram,
        NULLIF(kw.item ->> 'score', '')::double precision AS score,
        NULLIF(kw.item ->> 'base_score', '')::double precision AS base_score,
        NULLIF(kw.item ->> 'physics_boost', '')::double precision AS physics_boost
    FROM table_rows t
    CROSS JOIN LATERAL jsonb_array_elements(t.keywords) WITH ORDINALITY AS kw(item, ordinality)
)
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
LIMIT :top_n;

\echo ''
\echo '== Top Boosted Keywords =='
WITH table_rows AS (
    SELECT
        paper_id,
        keywords
    FROM :target_table
),
keyword_rows AS (
    SELECT
        t.paper_id,
        kw.item ->> 'keyword' AS keyword,
        NULLIF(kw.item ->> 'score', '')::double precision AS score,
        NULLIF(kw.item ->> 'base_score', '')::double precision AS base_score,
        NULLIF(kw.item ->> 'physics_boost', '')::double precision AS physics_boost
    FROM table_rows t
    CROSS JOIN LATERAL jsonb_array_elements(t.keywords) AS kw(item)
)
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
LIMIT :top_n;

\echo ''
\echo '== Ngram Distribution =='
WITH table_rows AS (
    SELECT
        paper_id,
        keywords
    FROM :target_table
),
keyword_rows AS (
    SELECT
        t.paper_id,
        kw.ordinality::integer AS keyword_rank,
        NULLIF(kw.item ->> 'ngram', '')::integer AS ngram,
        NULLIF(kw.item ->> 'score', '')::double precision AS score,
        NULLIF(kw.item ->> 'physics_boost', '')::double precision AS physics_boost
    FROM table_rows t
    CROSS JOIN LATERAL jsonb_array_elements(t.keywords) WITH ORDINALITY AS kw(item, ordinality)
)
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
ORDER BY ngram;

\echo ''
\echo '== Rank Distribution =='
WITH table_rows AS (
    SELECT
        paper_id,
        keywords
    FROM :target_table
),
keyword_rows AS (
    SELECT
        t.paper_id,
        kw.ordinality::integer AS keyword_rank,
        NULLIF(kw.item ->> 'score', '')::double precision AS score,
        NULLIF(kw.item ->> 'base_score', '')::double precision AS base_score,
        NULLIF(kw.item ->> 'physics_boost', '')::double precision AS physics_boost
    FROM table_rows t
    CROSS JOIN LATERAL jsonb_array_elements(t.keywords) WITH ORDINALITY AS kw(item, ordinality)
)
SELECT
    keyword_rank,
    COUNT(*)::bigint AS keyword_occurrences,
    COUNT(DISTINCT paper_id)::bigint AS papers_with_rank,
    ROUND(AVG(score)::numeric, 4) AS avg_score,
    ROUND(AVG(base_score)::numeric, 4) AS avg_base_score,
    ROUND(AVG(physics_boost)::numeric, 4) AS avg_physics_boost
FROM keyword_rows
GROUP BY keyword_rank
ORDER BY keyword_rank;

\echo ''
\echo '== Keyword Count Distribution =='
SELECT
    keyword_count,
    COUNT(*)::bigint AS paper_count,
    ROUND(
        100.0 * COUNT(*)
        / NULLIF(SUM(COUNT(*)) OVER (), 0),
        4
    ) AS paper_share_pct
FROM :target_table
GROUP BY keyword_count
ORDER BY keyword_count;

\echo ''
\echo '== Score Distribution =='
WITH table_rows AS (
    SELECT
        paper_id,
        keywords
    FROM :target_table
),
keyword_rows AS (
    SELECT
        t.paper_id,
        NULLIF(kw.item ->> 'score', '')::double precision AS score
    FROM table_rows t
    CROSS JOIN LATERAL jsonb_array_elements(t.keywords) AS kw(item)
),
bucketed AS (
    SELECT
        FLOOR(score / :score_bucket_width)::integer AS bucket_id,
        paper_id
    FROM keyword_rows
)
SELECT
    bucket_id,
    ROUND((bucket_id * :score_bucket_width)::numeric, 4) AS bucket_start,
    ROUND(((bucket_id + 1) * :score_bucket_width)::numeric, 4) AS bucket_end,
    COUNT(*)::bigint AS keyword_occurrences,
    COUNT(DISTINCT paper_id)::bigint AS paper_count
FROM bucketed
GROUP BY bucket_id
ORDER BY bucket_id;

\echo ''
\echo '== Physics Boost Distribution =='
WITH table_rows AS (
    SELECT
        paper_id,
        keywords
    FROM :target_table
),
keyword_rows AS (
    SELECT
        t.paper_id,
        NULLIF(kw.item ->> 'physics_boost', '')::double precision AS physics_boost
    FROM table_rows t
    CROSS JOIN LATERAL jsonb_array_elements(t.keywords) AS kw(item)
),
bucketed AS (
    SELECT
        FLOOR(physics_boost / :boost_bucket_width)::integer AS bucket_id,
        paper_id
    FROM keyword_rows
)
SELECT
    bucket_id,
    ROUND((bucket_id * :boost_bucket_width)::numeric, 4) AS bucket_start,
    ROUND(((bucket_id + 1) * :boost_bucket_width)::numeric, 4) AS bucket_end,
    COUNT(*)::bigint AS keyword_occurrences,
    COUNT(DISTINCT paper_id)::bigint AS paper_count
FROM bucketed
GROUP BY bucket_id
ORDER BY bucket_id;
