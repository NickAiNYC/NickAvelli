"""CASA configuration management.

Centralised settings for all CASA components including audio parameters,
genre constraints, and model paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml


@dataclass
class AudioConfig:
    """Audio processing parameters for Latina Afro House."""

    sample_rate: int = 44100
    bit_depth: int = 24
    channels: int = 2  # stereo
    bpm_range: Tuple[int, int] = (120, 128)
    target_bpm: int = 124
    key_preferences: List[str] = field(
        default_factory=lambda: ["Am", "Dm", "Em", "Gm", "Cm"]
    )
    target_lufs: float = -14.0
    true_peak_dbfs: float = -1.0
    max_duration_sec: float = 420.0  # 7 minutes
    min_duration_sec: float = 180.0  # 3 minutes


@dataclass
class GenerationConfig:
    """Configuration for the generation pipeline."""

    model_name: str = "facebook/musicgen-medium"
    max_new_tokens: int = 1500
    guidance_scale: float = 3.0
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.device == "auto":
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"


@dataclass
class PercussionConfig:
    """Afro-Cuban percussion pattern parameters."""

    clave_patterns: List[str] = field(
        default_factory=lambda: ["son", "rumba", "bossa"]
    )
    default_clave: str = "son"
    swing_amount: float = 0.15
    ghost_note_velocity: float = 0.3
    conga_tuning_hz: Tuple[float, float, float] = (200.0, 250.0, 320.0)


@dataclass
class MasteringConfig:
    """Mastering chain parameters."""

    target_lufs: float = -14.0
    true_peak_dbfs: float = -1.0
    eq_bands: List[dict] = field(
        default_factory=lambda: [
            {"freq": 40, "gain": -2.0, "q": 0.7, "type": "highpass"},
            {"freq": 80, "gain": 1.5, "q": 1.0, "type": "peak"},
            {"freq": 250, "gain": -1.0, "q": 0.8, "type": "peak"},
            {"freq": 3000, "gain": 1.0, "q": 1.2, "type": "peak"},
            {"freq": 10000, "gain": 0.5, "q": 0.7, "type": "shelf"},
        ]
    )
    stereo_width: float = 1.2
    limiter_release_ms: float = 50.0


@dataclass
class CASAConfig:
    """Top-level CASA configuration."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    percussion: PercussionConfig = field(default_factory=PercussionConfig)
    mastering: MasteringConfig = field(default_factory=MasteringConfig)

    output_dir: Path = Path("output")
    cache_dir: Path = Path("cache")
    models_dir: Path = Path("models")
    data_dir: Path = Path("data")

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        for d in [self.output_dir, self.cache_dir, self.models_dir, self.data_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CASAConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        audio_data = data.get("audio", {})
        # Convert lists back to tuples for typed tuple fields
        if "bpm_range" in audio_data and isinstance(audio_data["bpm_range"], list):
            audio_data["bpm_range"] = tuple(audio_data["bpm_range"])

        perc_data = data.get("percussion", {})
        if "conga_tuning_hz" in perc_data and isinstance(perc_data["conga_tuning_hz"], list):
            perc_data["conga_tuning_hz"] = tuple(perc_data["conga_tuning_hz"])

        audio = AudioConfig(**audio_data)
        generation = GenerationConfig(**data.get("generation", {}))
        percussion = PercussionConfig(**perc_data)
        mastering = MasteringConfig(**data.get("mastering", {}))

        return cls(
            audio=audio,
            generation=generation,
            percussion=percussion,
            mastering=mastering,
            output_dir=Path(data.get("output_dir", "output")),
            cache_dir=Path(data.get("cache_dir", "cache")),
            models_dir=Path(data.get("models_dir", "models")),
            data_dir=Path(data.get("data_dir", "data")),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        import dataclasses

        def _convert(obj: object) -> object:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            if isinstance(obj, Path):
                return str(obj)
            return obj

        with open(path, "w") as f:
            yaml.dump(_convert(self), f, default_flow_style=False)
