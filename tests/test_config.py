"""Tests for the CASA configuration module."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.casa.utils.config import (
    AudioConfig,
    CASAConfig,
    GenerationConfig,
    MasteringConfig,
    PercussionConfig,
)


class TestAudioConfig:
    def test_defaults(self):
        cfg = AudioConfig()
        assert cfg.sample_rate == 44100
        assert cfg.bpm_range == (120, 128)
        assert cfg.target_bpm == 124
        assert cfg.target_lufs == -14.0
        assert cfg.true_peak_dbfs == -1.0

    def test_custom_bpm(self):
        cfg = AudioConfig(bpm_range=(118, 126), target_bpm=122)
        assert cfg.bpm_range == (118, 126)
        assert cfg.target_bpm == 122


class TestGenerationConfig:
    def test_defaults(self):
        cfg = GenerationConfig()
        assert cfg.model_name == "facebook/musicgen-medium"
        assert cfg.device == "cpu"


class TestCASAConfig:
    def test_defaults(self):
        cfg = CASAConfig()
        assert isinstance(cfg.audio, AudioConfig)
        assert isinstance(cfg.generation, GenerationConfig)
        assert isinstance(cfg.percussion, PercussionConfig)
        assert isinstance(cfg.mastering, MasteringConfig)

    def test_ensure_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = CASAConfig(
                output_dir=Path(tmpdir) / "out",
                cache_dir=Path(tmpdir) / "cache",
                models_dir=Path(tmpdir) / "models",
                data_dir=Path(tmpdir) / "data",
            )
            cfg.ensure_dirs()
            assert (Path(tmpdir) / "out").is_dir()
            assert (Path(tmpdir) / "cache").is_dir()

    def test_yaml_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = CASAConfig()
            yaml_path = Path(tmpdir) / "test.yaml"
            cfg.to_yaml(yaml_path)

            loaded = CASAConfig.from_yaml(yaml_path)
            assert loaded.audio.sample_rate == cfg.audio.sample_rate
            assert loaded.audio.target_bpm == cfg.audio.target_bpm
            assert loaded.mastering.target_lufs == cfg.mastering.target_lufs

    def test_from_yaml_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("")
            f.flush()
            cfg = CASAConfig.from_yaml(f.name)
            assert cfg.audio.sample_rate == 44100
