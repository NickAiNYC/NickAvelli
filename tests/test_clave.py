"""Tests for clave pattern recognition."""

import pytest
import numpy as np

from src.casa.analysis.clave import (
    CLAVE_PATTERNS,
    ClaveMatch,
    ClaveType,
    detect_clave,
    generate_clave_pattern,
    _normalised_cross_correlation,
)


class TestClavePatterns:
    """Verify built-in clave pattern definitions."""

    def test_all_patterns_are_16_steps(self):
        for name, pattern in CLAVE_PATTERNS.items():
            assert len(pattern) == 16, f"{name} has {len(pattern)} steps"

    def test_patterns_are_binary(self):
        for name, pattern in CLAVE_PATTERNS.items():
            assert all(v in (0, 1) for v in pattern), f"{name} has non-binary values"

    def test_son_3_2_has_5_hits(self):
        assert sum(CLAVE_PATTERNS["son_3_2"]) == 5

    def test_rumba_3_2_has_5_hits(self):
        assert sum(CLAVE_PATTERNS["rumba_3_2"]) == 5

    def test_bossa_has_5_hits(self):
        assert sum(CLAVE_PATTERNS["bossa"]) == 5


class TestDetectClave:
    """Tests for the detect_clave function."""

    def test_perfect_son_3_2_match(self):
        onset = CLAVE_PATTERNS["son_3_2"][:]
        result = detect_clave(onset)
        assert result.pattern_type == ClaveType.SON_3_2
        assert result.confidence == pytest.approx(1.0)

    def test_perfect_rumba_match(self):
        onset = CLAVE_PATTERNS["rumba_3_2"][:]
        result = detect_clave(onset)
        assert result.pattern_type == ClaveType.RUMBA_3_2
        assert result.confidence == pytest.approx(1.0)

    def test_perfect_bossa_match(self):
        onset = CLAVE_PATTERNS["bossa"][:]
        result = detect_clave(onset)
        assert result.pattern_type == ClaveType.BOSSA
        assert result.confidence == pytest.approx(1.0)

    def test_empty_grid_returns_unknown(self):
        onset = [0] * 16
        result = detect_clave(onset, threshold=0.7)
        assert result.pattern_type == ClaveType.UNKNOWN

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="16 elements"):
            detect_clave([1, 0, 0])

    def test_threshold_filters_weak_matches(self):
        # Slightly modified pattern
        onset = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        result = detect_clave(onset, threshold=0.9)
        assert result.pattern_type == ClaveType.UNKNOWN

    def test_onset_positions_are_correct(self):
        onset = CLAVE_PATTERNS["son_3_2"][:]
        result = detect_clave(onset)
        expected = [i for i, v in enumerate(onset) if v == 1]
        assert result.onset_positions == expected


class TestGenerateClavePattern:
    """Tests for clave pattern generation."""

    def test_generate_son_3_2(self):
        pattern = generate_clave_pattern(ClaveType.SON_3_2, bars=1)
        assert pattern == CLAVE_PATTERNS["son_3_2"]

    def test_generate_multiple_bars(self):
        pattern = generate_clave_pattern("son_3_2", bars=3)
        assert len(pattern) == 16 * 3

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown clave type"):
            generate_clave_pattern("nonexistent")

    def test_string_and_enum_produce_same_result(self):
        from_str = generate_clave_pattern("rumba_3_2")
        from_enum = generate_clave_pattern(ClaveType.RUMBA_3_2)
        assert from_str == from_enum


class TestNormalisedCrossCorrelation:
    """Tests for the correlation helper."""

    def test_identical_signals(self):
        sig = np.array([1, 0, 1, 0, 1], dtype=float)
        assert _normalised_cross_correlation(sig, sig) == pytest.approx(1.0)

    def test_orthogonal_signals(self):
        a = np.array([1, 0, 0, 0], dtype=float)
        b = np.array([0, 1, 0, 0], dtype=float)
        assert _normalised_cross_correlation(a, b) == pytest.approx(0.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            _normalised_cross_correlation(np.array([1, 0]), np.array([1]))
