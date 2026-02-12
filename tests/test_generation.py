"""Tests for the generation engine."""

import pytest
import numpy as np

from src.casa.generation.engine import (
    STYLE_PROMPTS,
    AfroHouseGenerator,
    GenerationResult,
)
from src.casa.analysis.clave import ClaveType
from src.casa.utils.config import AudioConfig, GenerationConfig


class TestStylePrompts:
    def test_all_prompts_contain_bpm_placeholder(self):
        for name, template in STYLE_PROMPTS.items():
            assert "{bpm}" in template, f"{name} missing {{bpm}}"

    def test_all_prompts_contain_key_placeholder(self):
        for name, template in STYLE_PROMPTS.items():
            assert "{key}" in template, f"{name} missing {{key}}"


class TestAfroHouseGenerator:
    def test_build_prompt_default(self):
        gen = AfroHouseGenerator()
        prompt = gen.build_prompt()
        assert "124" in prompt or "bpm" in prompt.lower()
        assert "Am" in prompt

    def test_build_prompt_custom(self):
        gen = AfroHouseGenerator()
        prompt = gen.build_prompt(style="deep_afro", bpm=126, key="Dm")
        assert "126" in prompt
        assert "Dm" in prompt

    def test_build_prompt_extra(self):
        gen = AfroHouseGenerator()
        prompt = gen.build_prompt(extra="with marimba")
        assert "marimba" in prompt

    def test_generate_returns_result(self):
        """Generate should return a result even without the model."""
        gen = AfroHouseGenerator()
        result = gen.generate(duration_sec=2.0)
        assert isinstance(result, GenerationResult)
        assert result.sample_rate > 0
        assert result.duration_sec > 0

    def test_generate_with_clave(self):
        gen = AfroHouseGenerator()
        result = gen.generate(duration_sec=2.0, clave_type=ClaveType.SON_3_2)
        assert result.clave_pattern is not None
        assert len(result.clave_pattern) == 16 * 4  # 4 bars


class TestVocals:
    def test_random_phrase_spanish(self):
        from src.casa.generation.vocals import random_phrase

        phrase = random_phrase(language="es", seed=42)
        assert phrase.language == "es"
        assert len(phrase.text) > 0

    def test_random_phrase_portuguese(self):
        from src.casa.generation.vocals import random_phrase

        phrase = random_phrase(language="pt", seed=42)
        assert phrase.language == "pt"

    def test_call_response_pairs(self):
        from src.casa.generation.vocals import build_call_response

        phrases = build_call_response(language="es", pairs=3, seed=42)
        assert len(phrases) == 6  # 3 pairs × 2
        categories = [p.category for p in phrases]
        assert categories == ["call", "response"] * 3

    def test_deterministic_with_seed(self):
        from src.casa.generation.vocals import random_phrase

        a = random_phrase(seed=123)
        b = random_phrase(seed=123)
        assert a.text == b.text
