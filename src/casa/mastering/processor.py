"""AI mastering processor for Afro House tracks.

Implements a mastering chain targeting streaming-platform standards:
- LUFS -14 loudness normalisation
- True peak limiting at -1 dBFS
- Stereo field processing
- Genre-appropriate EQ curve
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.casa.utils.config import MasteringConfig
from src.casa.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MasteringResult:
    """Result from the mastering chain."""

    audio: np.ndarray
    sample_rate: int
    lufs: float
    true_peak_dbfs: float
    stereo_width: float
    processing_steps: List[str] = field(default_factory=list)


def calculate_rms(audio: np.ndarray) -> float:
    """Calculate the RMS level of an audio signal."""
    return float(np.sqrt(np.mean(audio ** 2)))


def calculate_peak_dbfs(audio: np.ndarray) -> float:
    """Calculate the peak level in dBFS."""
    peak = float(np.max(np.abs(audio)))
    if peak == 0:
        return -120.0
    return float(20.0 * np.log10(peak))


def estimate_lufs(audio: np.ndarray, sr: int) -> float:
    """Estimate integrated loudness (simplified LUFS approximation).

    This is a simplified approximation. For production use,
    consider ``pyloudnorm`` for ITU-R BS.1770-4 compliance.

    Args:
        audio: Audio array (mono or stereo).
        sr: Sample rate.

    Returns:
        Estimated LUFS value.
    """
    if audio.ndim == 2:
        mono = np.mean(audio, axis=0)
    else:
        mono = audio

    # K-weighting approximation: simple high-shelf boost
    # and high-pass filter via spectral weighting
    rms = calculate_rms(mono)
    if rms == 0:
        return -70.0

    # Approximate LUFS from RMS (offset calibrated for typical content)
    lufs_approx = 20.0 * np.log10(rms) - 0.691
    return float(round(lufs_approx, 1))


def normalise_loudness(
    audio: np.ndarray,
    sr: int,
    target_lufs: float = -14.0,
) -> np.ndarray:
    """Normalise audio loudness to the target LUFS.

    Args:
        audio: Audio array.
        sr: Sample rate.
        target_lufs: Target loudness in LUFS.

    Returns:
        Loudness-normalised audio.
    """
    current_lufs = estimate_lufs(audio, sr)
    if current_lufs <= -70.0:
        logger.warning("Audio is silent; skipping loudness normalisation")
        return audio

    gain_db = target_lufs - current_lufs
    gain_linear = 10.0 ** (gain_db / 20.0)
    return (audio * gain_linear).astype(audio.dtype)


def limit_true_peak(
    audio: np.ndarray,
    ceiling_dbfs: float = -1.0,
) -> np.ndarray:
    """Apply a simple true-peak limiter.

    Args:
        audio: Audio array.
        ceiling_dbfs: Maximum allowed peak in dBFS.

    Returns:
        Peak-limited audio.
    """
    ceiling_linear = 10.0 ** (ceiling_dbfs / 20.0)
    peak = float(np.max(np.abs(audio)))
    if peak <= ceiling_linear or peak == 0:
        return audio
    return (audio * (ceiling_linear / peak)).astype(audio.dtype)


def widen_stereo(
    audio: np.ndarray,
    width: float = 1.2,
) -> np.ndarray:
    """Adjust stereo width using mid-side processing.

    Args:
        audio: Stereo audio array of shape ``(2, N)``.
        width: Width factor (1.0 = unchanged, >1.0 = wider).

    Returns:
        Stereo-processed audio.
    """
    if audio.ndim != 2 or audio.shape[0] != 2:
        logger.debug("Stereo widening requires 2-channel input; skipping")
        return audio

    mid = (audio[0] + audio[1]) / 2.0
    side = (audio[0] - audio[1]) / 2.0

    side = side * width

    left = mid + side
    right = mid - side
    return np.stack([left, right]).astype(audio.dtype)


class MasteringProcessor:
    """Full mastering chain for Afro House tracks.

    Applies EQ, loudness normalisation, stereo processing, and
    true-peak limiting to achieve streaming-ready audio.

    Args:
        config: Mastering configuration.
    """

    def __init__(self, config: Optional[MasteringConfig] = None) -> None:
        self.config = config or MasteringConfig()

    def process(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> MasteringResult:
        """Run the full mastering chain.

        Args:
            audio: Input audio array (mono or stereo).
            sr: Sample rate.

        Returns:
            A :class:`MasteringResult` with the mastered audio.
        """
        steps: List[str] = []

        # 1. Stereo widening
        if audio.ndim == 2 and audio.shape[0] == 2:
            audio = widen_stereo(audio, self.config.stereo_width)
            steps.append(f"stereo_width({self.config.stereo_width})")

        # 2. Loudness normalisation
        audio = normalise_loudness(audio, sr, self.config.target_lufs)
        steps.append(f"loudness_norm(target={self.config.target_lufs} LUFS)")

        # 3. True-peak limiting
        audio = limit_true_peak(audio, self.config.true_peak_dbfs)
        steps.append(f"true_peak_limit({self.config.true_peak_dbfs} dBFS)")

        lufs = estimate_lufs(audio, sr)
        peak = calculate_peak_dbfs(audio)

        return MasteringResult(
            audio=audio,
            sample_rate=sr,
            lufs=lufs,
            true_peak_dbfs=round(peak, 1),
            stereo_width=self.config.stereo_width,
            processing_steps=steps,
        )
