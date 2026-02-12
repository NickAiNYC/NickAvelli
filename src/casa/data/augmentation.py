"""Audio augmentation for dataset expansion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AugmentationResult:
    """Result of audio augmentation."""

    audio: np.ndarray
    sample_rate: int
    augmentations_applied: list[str]


def eq_simulation(
    audio: np.ndarray,
    sr: int,
    low_gain_db: float = 0.0,
    mid_gain_db: float = 0.0,
    high_gain_db: float = 0.0,
) -> np.ndarray:
    """Apply simplified 3-band EQ simulation.

    Uses cascaded second-order Butterworth filters to approximate a 3-band EQ.
    Crossover frequencies: 250 Hz (low/mid), 4000 Hz (mid/high).

    Args:
        audio: Input audio signal
        sr: Sample rate in Hz
        low_gain_db: Gain for low band in dB (< 250 Hz)
        mid_gain_db: Gain for mid band in dB (250-4000 Hz)
        high_gain_db: Gain for high band in dB (> 4000 Hz)

    Returns:
        Processed audio with same length as input

    Security: Validates sample rate to prevent division by zero.
    """
    if sr <= 0:
        raise ValueError("Sample rate must be positive")

    # If no gain changes, return original audio
    if low_gain_db == 0 and mid_gain_db == 0 and high_gain_db == 0:
        return audio

    # Import scipy here to make it an optional dependency
    try:
        from scipy.signal import butter, sosfilt
    except ImportError:
        # Fallback: apply simple gain without filtering
        avg_gain_db = (low_gain_db + mid_gain_db + high_gain_db) / 3
        gain_linear = 10 ** (avg_gain_db / 20)
        return audio * gain_linear

    # Design filters - use second-order sections for numerical stability
    nyquist = sr / 2
    low_cutoff = min(250 / nyquist, 0.99)
    high_cutoff = min(4000 / nyquist, 0.99)

    # Low band: lowpass at 250 Hz
    sos_low = butter(2, low_cutoff, btype="low", output="sos")
    # Mid band: bandpass 250-4000 Hz
    sos_mid = butter(2, [low_cutoff, high_cutoff], btype="band", output="sos")
    # High band: highpass at 4000 Hz
    sos_high = butter(2, high_cutoff, btype="high", output="sos")

    # Filter and apply gains
    low_band = sosfilt(sos_low, audio) * (10 ** (low_gain_db / 20))
    mid_band = sosfilt(sos_mid, audio) * (10 ** (mid_gain_db / 20))
    high_band = sosfilt(sos_high, audio) * (10 ** (high_gain_db / 20))

    # Sum bands
    result = low_band + mid_band + high_band

    # Normalize to prevent clipping while preserving relative dynamics
    peak = np.abs(result).max()
    if peak > 1.0:
        result = result / peak

    return result.astype(audio.dtype)


def random_augment(
    audio: np.ndarray,
    sr: int,
    seed: int | None = None,
) -> AugmentationResult:
    """Apply random augmentation chain with seeded randomization.

    Args:
        audio: Input audio signal
        sr: Sample rate in Hz
        seed: Random seed for reproducibility

    Returns:
        AugmentationResult with processed audio and metadata

    Note: Currently implements only EQ augmentation. Future enhancements
          will add pitch shifting and time stretching.
    """
    # Use local random generator for thread safety
    rng = np.random.default_rng(seed)

    augmentations = []

    # Random EQ with ±6 dB range
    low_gain = rng.uniform(-6, 6)
    mid_gain = rng.uniform(-6, 6)
    high_gain = rng.uniform(-6, 6)

    augmented = eq_simulation(audio, sr, low_gain, mid_gain, high_gain)
    augmentations.append(
        f"eq(low={low_gain:.1f}dB, mid={mid_gain:.1f}dB, high={high_gain:.1f}dB)"
    )

    return AugmentationResult(
        audio=augmented,
        sample_rate=sr,
        augmentations_applied=augmentations,
    )
