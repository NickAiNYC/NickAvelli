"""Tests for the data pipeline components."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.casa.data.dataset import AfroHouseDataset, AudioSample
from src.casa.data.augmentation import (
    AugmentationResult,
    eq_simulation,
    random_augment,
)
from src.casa.analysis.structure import (
    Section,
    SectionType,
    StructureAnalysis,
    analyse_structure,
)


class TestAudioSample:
    def test_to_dict_roundtrip(self):
        sample = AudioSample(
            file_path="/tmp/test.wav",
            bpm=124.0,
            key="Am",
            duration_sec=30.0,
            tags=["percussive"],
        )
        d = sample.to_dict()
        restored = AudioSample.from_dict(d)
        assert restored.file_path == sample.file_path
        assert restored.bpm == sample.bpm
        assert restored.tags == sample.tags


class TestAfroHouseDataset:
    def test_empty_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = AfroHouseDataset(tmpdir)
            assert len(ds) == 0

    def test_scan_finds_wav_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy files
            (Path(tmpdir) / "a.wav").touch()
            (Path(tmpdir) / "b.mp3").touch()
            (Path(tmpdir) / "c.txt").touch()

            ds = AfroHouseDataset(tmpdir)
            count = ds.scan()
            assert count == 2  # .wav and .mp3

    def test_save_and_load_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = AfroHouseDataset(tmpdir)
            ds.samples = [
                AudioSample(file_path="a.wav", bpm=124),
                AudioSample(file_path="b.wav", bpm=126),
            ]
            manifest = ds.save_manifest()
            assert manifest.exists()

            ds2 = AfroHouseDataset(tmpdir, manifest_path=manifest)
            assert len(ds2) == 2
            assert ds2[0].bpm == 124

    def test_filter_by_bpm(self):
        ds = AfroHouseDataset("/tmp")
        ds.samples = [
            AudioSample(file_path="a.wav", bpm=110),
            AudioSample(file_path="b.wav", bpm=124),
            AudioSample(file_path="c.wav", bpm=140),
        ]
        result = ds.filter_by_bpm(120, 128)
        assert len(result) == 1
        assert result[0].bpm == 124

    def test_filter_by_key(self):
        ds = AfroHouseDataset("/tmp")
        ds.samples = [
            AudioSample(file_path="a.wav", key="Am"),
            AudioSample(file_path="b.wav", key="C"),
            AudioSample(file_path="c.wav", key="Dm"),
        ]
        result = ds.filter_by_key(["Am", "Dm"])
        assert len(result) == 2


class TestEqSimulation:
    def test_no_change_with_zero_gains(self):
        audio = np.random.randn(4410).astype(np.float32)
        result = eq_simulation(audio, sr=44100, low_gain_db=0, mid_gain_db=0, high_gain_db=0)
        np.testing.assert_allclose(result, audio, atol=1e-5)

    def test_output_same_length(self):
        audio = np.random.randn(4410).astype(np.float32)
        result = eq_simulation(audio, sr=44100, low_gain_db=3, mid_gain_db=-2, high_gain_db=1)
        assert len(result) == len(audio)


class TestRandomAugment:
    def test_deterministic_with_seed(self):
        audio = np.random.randn(4410).astype(np.float32)
        r1 = random_augment(audio, sr=44100, seed=42)
        r2 = random_augment(audio, sr=44100, seed=42)
        assert r1.augmentations_applied == r2.augmentations_applied

    def test_returns_augmentation_result(self):
        audio = np.random.randn(4410).astype(np.float32)
        result = random_augment(audio, sr=44100, seed=0)
        assert isinstance(result, AugmentationResult)
        assert result.sample_rate == 44100


class TestStructureAnalysis:
    def test_basic_structure(self):
        # Simulate energy envelope: low → rising → high → low → high → low
        env = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.3, 0.2, 0.8, 0.7, 0.2, 0.1])
        result = analyse_structure(env, bpm=124, total_duration_sec=300, num_sections=6)
        assert len(result.sections) == 6
        assert result.sections[0].section_type == SectionType.INTRO
        assert result.sections[-1].section_type == SectionType.OUTRO

    def test_too_few_samples(self):
        env = np.array([0.5])
        result = analyse_structure(env, bpm=124, total_duration_sec=10, num_sections=6)
        assert len(result.sections) == 0

    def test_dj_friendly_intro(self):
        result = StructureAnalysis(
            sections=[
                Section(SectionType.INTRO, 0.0, 35.0, 0.3),
                Section(SectionType.DROP, 35.0, 120.0, 0.9),
                Section(SectionType.OUTRO, 120.0, 155.0, 0.2),
            ],
            total_duration_sec=155.0,
            bpm=124,
        )
        assert result.has_dj_friendly_intro  # 35s > 16 bars at 124 BPM
