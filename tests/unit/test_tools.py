"""
Unit tests for tools/calculate_metric.py, tools/rerank_documents.py,
tools/search_documents.py, and tools/run_finbert.py.

Pure logic (alias resolution, MMR math, sentence windowing, aggregation) is
tested directly with no mocking. Anything that crosses an Azure/model
boundary (SQL fetch, cross-encoder inference, the FinBERT pipeline) is
tested with that one boundary mocked, so these stay fast and offline.
"""
from unittest.mock import patch, MagicMock

from tools.calculate_metric import calculate_metric, _normalize_metric
from tools.rerank_documents import rerank_documents
from tools.search_documents import _cosine_similarity, _build_odata_filter, mmr_rerank
from tools.run_finbert import _split_sentences, _build_windows, _aggregate, run_finbert


# ── calculate_metric: _normalize_metric ─────────────────────────────────────

def test_normalize_metric_handles_parens_and_case():
    assert _normalize_metric("Revenue Growth (CC)") == "revenue_growth_cc"


def test_normalize_metric_handles_hyphens():
    assert _normalize_metric("Revenue-Growth-CC") == "revenue_growth_cc"


def test_normalize_metric_handles_percent_sign():
    assert _normalize_metric("Gross Margin % Change") == "gross_margin_pct_change"


# ── calculate_metric: unsupported metrics (no SQL call) ─────────────────────

def test_calculate_metric_segment_kpi_exact_match_no_sql_call():
    with patch("tools.calculate_metric._fetch_value") as mock_fetch:
        result = calculate_metric("MSFT", "FY2026-Q3", "azure_revenue_growth_cc")
    mock_fetch.assert_not_called()
    assert result["value"] is None
    assert "MD&A table extraction" in result["error"]


def test_calculate_metric_product_kpi_pattern_match():
    with patch("tools.calculate_metric._fetch_value") as mock_fetch:
        result = calculate_metric("AAPL", "FY2026-Q3", "iphone_revenue")
    mock_fetch.assert_not_called()
    assert result["error"] is not None


def test_calculate_metric_operational_kpi_pattern_match():
    with patch("tools.calculate_metric._fetch_value") as mock_fetch:
        result = calculate_metric("META", "FY2026-Q3", "family_daily_active_people")
    mock_fetch.assert_not_called()
    assert result["error"] is not None


def test_calculate_metric_unknown_metric_no_sql_call():
    with patch("tools.calculate_metric._fetch_value") as mock_fetch:
        result = calculate_metric("MSFT", "FY2026-Q3", "totally_made_up_metric")
    mock_fetch.assert_not_called()
    assert result["value"] is None
    assert "no alias mapping found" in result["error"]


# ── calculate_metric: direct value lookup ───────────────────────────────────

def test_calculate_metric_direct_lookup():
    with patch("tools.calculate_metric._fetch_value", return_value=(95359000000.0, "USD")):
        result = calculate_metric("AAPL", "FY2026-Q3", "revenue")
    assert result["value"] == 95359000000.0
    assert result["unit"] == "USD"
    assert result["concept"] == "Revenues"
    assert result["error"] is None


def test_calculate_metric_direct_lookup_no_fact_found():
    with patch("tools.calculate_metric._fetch_value", return_value=(None, None)):
        result = calculate_metric("AAPL", "FY2026-Q3", "revenue")
    assert result["value"] is None
    assert "No fact found" in result["error"]


# ── calculate_metric: margin metrics ─────────────────────────────────────────

def test_calculate_metric_margin_computed_from_two_facts():
    def fake_fetch(company, fiscal_label, concept):
        return (44_000_000.0, "USD") if concept == "GrossProfit" else (95_359_000_000.0, "USD")

    with patch("tools.calculate_metric._fetch_value", side_effect=fake_fetch):
        result = calculate_metric("AAPL", "FY2026-Q3", "gross_margin")
    assert result["unit"] == "%"
    assert result["value"] == round(44_000_000.0 / 95_359_000_000.0 * 100, 4)


def test_calculate_metric_margin_zero_revenue_errors_cleanly():
    def fake_fetch(company, fiscal_label, concept):
        return (44_000_000.0, "USD") if concept == "GrossProfit" else (0.0, "USD")

    with patch("tools.calculate_metric._fetch_value", side_effect=fake_fetch):
        result = calculate_metric("AAPL", "FY2026-Q3", "gross_margin")
    assert result["value"] is None
    assert "revenue" in result["error"].lower()


# ── calculate_metric: YoY growth ─────────────────────────────────────────────

def test_calculate_metric_yoy_growth_requires_prior_period():
    result = calculate_metric("MSFT", "FY2026-Q3", "revenue_growth_yoy")
    assert result["value"] is None
    assert "prior_fiscal_label required" in result["error"]


def test_calculate_metric_yoy_growth_computed():
    def fake_fetch(company, fiscal_label, concept):
        return (110.0, "USD") if fiscal_label == "FY2026-Q3" else (100.0, "USD")

    with patch("tools.calculate_metric._fetch_value", side_effect=fake_fetch):
        result = calculate_metric(
            "MSFT", "FY2026-Q3", "revenue_growth_yoy", prior_fiscal_label="FY2025-Q3"
        )
    assert result["value"] == 10.0


# ── rerank_documents ──────────────────────────────────────────────────────

def test_rerank_documents_empty_chunks():
    assert rerank_documents("query", []) == []


def test_rerank_documents_sorts_by_score_descending():
    chunks = [
        {"content": "irrelevant passage"},
        {"content": "highly relevant passage"},
    ]
    fake_model = MagicMock()
    fake_model.predict.return_value = MagicMock(tolist=lambda: [0.1, 0.9])

    with patch("tools.rerank_documents._get_cross_encoder", return_value=fake_model):
        result = rerank_documents("query", chunks, top_k=5)

    assert result[0]["content"] == "highly relevant passage"
    assert result[0]["rerank_score"] == 0.9
    assert result[1]["rerank_score"] == 0.1


def test_rerank_documents_respects_top_k():
    chunks = [{"content": f"passage {i}"} for i in range(5)]
    fake_model = MagicMock()
    fake_model.predict.return_value = MagicMock(tolist=lambda: [0.1, 0.5, 0.9, 0.3, 0.7])

    with patch("tools.rerank_documents._get_cross_encoder", return_value=fake_model):
        result = rerank_documents("query", chunks, top_k=2)

    assert len(result) == 2
    assert result[0]["rerank_score"] == 0.9


# ── search_documents: _cosine_similarity / _build_odata_filter ─────────────

def test_cosine_similarity_identical_vectors():
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_zero_vector_returns_zero():
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_build_odata_filter_escapes_single_quotes():
    result = _build_odata_filter(None, "O'Reilly", None)
    assert "O''Reilly" in result


def test_build_odata_filter_combines_clauses_with_and():
    result = _build_odata_filter("10-Q", "AAPL", None)
    assert " and " in result


def test_build_odata_filter_none_when_no_filters():
    assert _build_odata_filter(None, None, None) is None


# ── search_documents: mmr_rerank ────────────────────────────────────────────

def test_mmr_rerank_empty_chunks():
    assert mmr_rerank([], [1.0, 0.0], top_k=5) == []


def test_mmr_rerank_prefers_relevant_and_diverse_chunks():
    # Three chunks: one near-identical to the query, two identical to each
    # other but less relevant. With embeddings already on each chunk (the
    # retrievable-index path), mmr_rerank should never hit openai_client.
    query_embedding = [1.0, 0.0]
    chunks = [
        {"content": "a", "embedding": [1.0, 0.0]},     # most relevant
        {"content": "b", "embedding": [0.0, 1.0]},     # relevant to query=0, diverse
        {"content": "c", "embedding": [0.0, 1.0]},     # duplicate of b — should be deprioritized
    ]
    with patch("tools.search_documents.openai_client") as mock_client:
        result = mmr_rerank(chunks, query_embedding, top_k=2, lambda_param=0.5)
    mock_client.embed_batch.assert_not_called()
    assert result[0]["content"] == "a"
    assert len(result) == 2


def test_mmr_rerank_respects_top_k_larger_than_pool():
    chunks = [{"content": "a", "embedding": [1.0, 0.0]}]
    result = mmr_rerank(chunks, [1.0, 0.0], top_k=5)
    assert len(result) == 1


# ── run_finbert: pure helpers ────────────────────────────────────────────

def test_split_sentences_basic():
    sentences = _split_sentences("Revenue grew. Margins improved. Guidance was raised.")
    assert len(sentences) == 3


def test_split_sentences_strips_whitespace():
    sentences = _split_sentences("  One sentence.   ")
    assert sentences == ["One sentence."]


def test_build_windows_packs_under_budget():
    sentences = ["word " * 10] * 5  # 5 sentences, 10 words each
    windows = _build_windows(sentences, max_tokens=28)  # word_budget = 20
    assert len(windows) > 1
    for w in windows:
        assert sum(len(s.split()) for s in w) <= 20 or len(w) == 1


def test_build_windows_single_sentence_never_split():
    sentences = ["one giant sentence " * 50]
    windows = _build_windows(sentences, max_tokens=28)
    assert len(windows) == 1
    assert windows[0] == sentences


def test_aggregate_weighted_by_sentence_count():
    window_results = [
        {"sentence_count": 1, "scores": {"positive": 1.0, "negative": 0.0, "neutral": 0.0}},
        {"sentence_count": 3, "scores": {"positive": 0.0, "negative": 1.0, "neutral": 0.0}},
    ]
    result = _aggregate(window_results)
    assert result["label"] == "negative"


def test_aggregate_zero_weight_defaults_neutral():
    result = _aggregate([])
    assert result["label"] == "neutral"


def test_run_finbert_empty_text_returns_neutral_without_pipeline():
    with patch("tools.run_finbert._get_pipeline") as mock_pipeline:
        result = run_finbert("")
    mock_pipeline.assert_not_called()
    assert result["aggregate"]["label"] == "neutral"
    assert result["window_count"] == 0


def test_run_finbert_scores_via_mocked_pipeline():
    fake_pipe = MagicMock(return_value=[[
        {"label": "positive", "score": 0.9},
        {"label": "negative", "score": 0.05},
        {"label": "neutral", "score": 0.05},
    ]])
    with patch("tools.run_finbert._get_pipeline", return_value=fake_pipe):
        result = run_finbert("Revenue grew strongly this quarter.")
    assert result["aggregate"]["label"] == "positive"
    assert result["window_count"] == 1
