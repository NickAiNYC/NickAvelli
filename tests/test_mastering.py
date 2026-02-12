"""Tests for the mastering processor."""

import numpy as np
import pytest

from src.casa.mastering.processor import (
    MasteringProcessor,
    MasteringResult,
    calculate_peak_dbfs,
    calculate_rms,
    estimate_lufs,
    limit_true_peak,
    normalise_loudness,
    widen_stereo,
)
from src.casa.utils.config import MasteringConfig


class TestCalculateRms:
    def test_silence_is_zero(self):
        assert calculate_rms(np.zeros(1000)) == 0.0

    def test_sine_wave(self):
        t = np.linspace(0, 1, 44100, dtype=np.float32)
        sine = np.sin(2 * np.pi * 440 * t)
        rms = calculate_rms(sine)
        assert 0.6 < rms < 0.8  # RMS of sine ≈ 0.707


class TestCalculatePeakDbfs:
    def test_full_scale(self):
        audio = np.array([1.0, -1.0, 0.5])
        assert calculate_peak_dbfs(audio) == pytest.approx(0.0, abs=0.01)

    def test_silence(self):
        assert calculate_peak_dbfs(np.zeros(100)) == -120.0

    def test_half_amplitude(self):
        audio = np.array([0.5, -0.5])
        assert calculate_peak_dbfs(audio) == pytest.approx(-6.02, abs=0.1)


class TestLimitTruePeak:
    def test_no_limiting_needed(self):
        audio = np.array([0.1, -0.1, 0.05], dtype=np.float32)
        result = limit_true_peak(audio, ceiling_dbfs=-1.0)
        np.testing.assert_array_equal(result, audio)

    def test_peak_is_reduced(self):
        audio = np.array([1.0, -1.0, 0.5], dtype=np.float32)
        result = limit_true_peak(audio, ceiling_dbfs=-3.0)
        peak = float(np.max(np.abs(result)))
        ceiling = 10 ** (-3.0 / 20.0)
        assert peak <= ceiling + 1e-6

    def test_silence_unchanged(self):
        audio = np.zeros(100, dtype=np.float32)
        result = limit_true_peak(audio, ceiling_dbfs=-1.0)
        np.testing.assert_array_equal(result, audio)


class TestWidenStereo:
    def test_width_1_unchanged(self):
        audio = np.random.randn(2, 1000).astype(np.float32)
        result = widen_stereo(audio, width=1.0)
        np.testing.assert_allclose(result, audio, atol=1e-6)

    def test_mono_input_skipped(self):
        audio = np.random.randn(1000).astype(np.float32)
        result = widen_stereo(audio, width=1.5)
        np.testing.assert_array_equal(result, audio)

    def test_wider_increases_side(self):
        left = np.array([1.0, 0.5], dtype=np.float32)
        right = np.array([0.5, 1.0], dtype=np.float32)
        audio = np.stack([left, right])
        result = widen_stereo(audio, width=2.0)
        # Side signal should be amplified → more L-R difference
        orig_diff = np.sum(np.abs(audio[0] - audio[1]))
        new_diff = np.sum(np.abs(result[0] - result[1]))
        assert new_diff > orig_diff


class TestMasteringProcessor:
    def test_process_returns_result(self):
        audio = np.random.randn(44100).astype(np.float32) * 0.1
        processor = MasteringProcessor()
        result = processor.process(audio, sr=44100)
        assert isinstance(result, MasteringResult)
        assert result.sample_rate == 44100
        assert len(result.processing_steps) > 0

    def test_peak_within_ceiling(self):
        audio = np.random.randn(44100).astype(np.float32) * 0.5
        config = MasteringConfig(true_peak_dbfs=-1.0)
        processor = MasteringProcessor(config)
        result = processor.process(audio, sr=44100)
        assert result.true_peak_dbfs <= -1.0 + 0.1  # small tolerance

    def test_stereo_processing(self):
        audio = np.random.randn(2, 44100).astype(np.float32) * 0.1
        processor = MasteringProcessor()
        result = processor.process(audio, sr=44100)
        assert "stereo_width" in result.processing_steps[0]
