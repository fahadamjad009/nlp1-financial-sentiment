"""
Unit tests for the financial sentiment inference pipeline.
Run: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from predict import load_model, predict, predict_batch, LABEL_MAP, EMOJI_MAP


@pytest.fixture(scope="module")
def model_and_tokenizer():
    tokenizer, model = load_model()
    return tokenizer, model


def test_model_loads(model_and_tokenizer):
    tokenizer, model = model_and_tokenizer
    assert tokenizer is not None
    assert model is not None


def test_predict_returns_required_keys(model_and_tokenizer):
    tokenizer, model = model_and_tokenizer
    result = predict("Apple stock surges on strong earnings", tokenizer, model)
    assert "label" in result
    assert "emoji" in result
    assert "confidence" in result
    assert "scores" in result


def test_predict_label_is_valid(model_and_tokenizer):
    tokenizer, model = model_and_tokenizer
    result = predict("Markets crash on recession fears", tokenizer, model)
    assert result["label"] in ["bearish", "neutral", "bullish"]


def test_predict_confidence_range(model_and_tokenizer):
    tokenizer, model = model_and_tokenizer
    result = predict("Fed holds rates steady", tokenizer, model)
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_scores_sum_to_one(model_and_tokenizer):
    tokenizer, model = model_and_tokenizer
    result = predict("Goldman upgrades Tesla to Buy", tokenizer, model)
    total = sum(result["scores"].values())
    assert abs(total - 1.0) < 0.01


def test_predict_batch(model_and_tokenizer):
    tokenizer, model = model_and_tokenizer
    texts = [
        "Apple beats earnings expectations",
        "Company files for bankruptcy",
        "Markets remain flat ahead of Fed decision",
    ]
    results = predict_batch(texts, tokenizer, model)
    assert len(results) == 3
    for r in results:
        assert r["label"] in ["bearish", "neutral", "bullish"]


def test_bearish_headline(model_and_tokenizer):
    tokenizer, model = model_and_tokenizer
    result = predict("Tesla misses delivery targets for third consecutive quarter", tokenizer, model)
    assert result["label"] == "bearish"


def test_bullish_headline(model_and_tokenizer):
    tokenizer, model = model_and_tokenizer
    result = predict("Goldman Sachs upgrades Microsoft to Buy with $450 price target", tokenizer, model)
    assert result["label"] in ["bullish", "neutral"]