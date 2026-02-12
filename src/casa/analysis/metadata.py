"""Audio metadata extraction for Afro House tracks.

Extracts BPM, musical key, energy, and instrumentation characteristics
from audio files using librosa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.casa.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrackMetadata:
    """Extracted metadata for an audio track."""

    file_path: str
    bpm: float
    key: str
    energy: float
    duration_sec: float
    sample_rate: int
    rms_db: float
    spectral_centroid_mean: float
    onset_density: float  # onsets per second
    is_afro_house_tempo: bool = False
    instrumentation_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to a plain dictionary."""
        return {
            "file_path": self.file_path,
            "bpm": self.bpm,
            "key": self.key,
            "energy": self.energy,
            "duration_sec": self.duration_sec,
            "sample_rate": self.sample_rate,
            "rms_db": self.rms_db,
            "spectral_centroid_mean": self.spectral_centroid_mean,
            "onset_density": self.onset_density,
            "is_afro_house_tempo": self.is_afro_house_tempo,
            "instrumentation_tags": self.instrumentation_tags,
        }


# Krumhansl-Kessler key profiles for major and minor keys
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

_KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _estimate_key(chroma: np.ndarray) -> str:
    """Estimate musical key from a chroma feature vector.

    Uses correlation with Krumhansl-Kessler key profiles.

    Args:
        chroma: 12-bin chroma feature array (mean over time).

    Returns:
        Key string such as ``"Am"`` or ``"C"``.
    """
    best_corr = -2.0
    best_key = "C"
    for shift in range(12):
        rotated = np.roll(chroma, -shift)
        corr_major = float(np.corrcoef(rotated, _MAJOR_PROFILE)[0, 1])
        corr_minor = float(np.corrcoef(rotated, _MINOR_PROFILE)[0, 1])
        if corr_major > best_corr:
            best_corr = corr_major
            best_key = _KEY_NAMES[shift]
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key = _KEY_NAMES[shift] + "m"
    return best_key


def extract_metadata(
    file_path: str | Path,
    bpm_range: Tuple[int, int] = (120, 128),
) -> TrackMetadata:
    """Extract comprehensive metadata from an audio file.

    Args:
        file_path: Path to the audio file (WAV, MP3, FLAC, etc.).
        bpm_range: The BPM range considered valid for Afro House.

    Returns:
        A populated :class:`TrackMetadata` instance.

    Raises:
        ImportError: If ``librosa`` is not installed.
    """
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "librosa is required for metadata extraction. "
            "Install with: pip install librosa"
        ) from exc

    path = str(file_path)
    logger.info("Extracting metadata from %s", path)

    y, sr = librosa.load(path, sr=None, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    # BPM
    tempo_arr = librosa.beat.tempo(y=y, sr=sr)
    bpm = float(tempo_arr[0]) if len(tempo_arr) > 0 else 0.0

    # Key
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    key = _estimate_key(chroma_mean)

    # RMS energy
    rms = librosa.feature.rms(y=y)
    rms_mean = float(np.mean(rms))
    rms_db = float(20 * np.log10(rms_mean + 1e-10))

    # Spectral centroid
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    sc_mean = float(np.mean(sc))

    # Onset density
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    onset_density = len(onsets) / max(duration, 0.01)

    # Energy metric (normalised RMS)
    energy = float(np.clip(rms_mean / 0.3, 0.0, 1.0))

    # Simple instrumentation tagging based on spectral features
    tags: List[str] = []
    if sc_mean < 1500:
        tags.append("bass_heavy")
    if sc_mean > 4000:
        tags.append("bright")
    if onset_density > 6:
        tags.append("percussive")
    if onset_density < 3:
        tags.append("sparse")

    is_afro = bpm_range[0] <= bpm <= bpm_range[1]

    return TrackMetadata(
        file_path=path,
        bpm=round(bpm, 1),
        key=key,
        energy=round(energy, 3),
        duration_sec=round(duration, 2),
        sample_rate=sr,
        rms_db=round(rms_db, 1),
        spectral_centroid_mean=round(sc_mean, 1),
        onset_density=round(onset_density, 2),
        is_afro_house_tempo=is_afro,
        instrumentation_tags=tags,
    )
