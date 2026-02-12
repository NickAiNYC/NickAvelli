"""Clave pattern recognition for Afro-Cuban percussion.

Implements detection and generation of traditional clave patterns
(son clave, rumba clave, bossa nova clave) used in Latina Afro House.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from src.casa.utils.logging import get_logger

logger = get_logger(__name__)

# Standard clave patterns expressed as beat positions within a 16-step grid
# Each value is 1 (hit) or 0 (rest), representing 16th-note subdivisions
# in two measures of 4/4 time.
CLAVE_PATTERNS: dict[str, list[int]] = {
    # Son clave 3-2: hits on 1, &-of-2, 4, 2-&, 4 (classic)
    "son_3_2": [
        1, 0, 0, 1, 0, 0, 1, 0,   # bar 1: 3 side
        0, 0, 1, 0, 1, 0, 0, 0,   # bar 2: 2 side
    ],
    # Son clave 2-3: reversed
    "son_2_3": [
        0, 0, 1, 0, 1, 0, 0, 0,   # bar 1: 2 side
        1, 0, 0, 1, 0, 0, 1, 0,   # bar 2: 3 side
    ],
    # Rumba clave 3-2: similar but third hit is shifted
    "rumba_3_2": [
        1, 0, 0, 1, 0, 0, 0, 1,   # bar 1: 3 side
        0, 0, 1, 0, 1, 0, 0, 0,   # bar 2: 2 side
    ],
    # Rumba clave 2-3
    "rumba_2_3": [
        0, 0, 1, 0, 1, 0, 0, 0,   # bar 1: 2 side
        1, 0, 0, 1, 0, 0, 0, 1,   # bar 2: 3 side
    ],
    # Bossa nova clave
    "bossa": [
        1, 0, 0, 1, 0, 0, 1, 0,   # bar 1
        0, 0, 1, 0, 0, 1, 0, 0,   # bar 2
    ],
}


class ClaveType(str, Enum):
    """Known clave pattern types."""

    SON_3_2 = "son_3_2"
    SON_2_3 = "son_2_3"
    RUMBA_3_2 = "rumba_3_2"
    RUMBA_2_3 = "rumba_2_3"
    BOSSA = "bossa"
    UNKNOWN = "unknown"


@dataclass
class ClaveMatch:
    """Result of clave pattern matching."""

    pattern_type: ClaveType
    confidence: float
    correlation: float
    onset_positions: List[int]


def _normalised_cross_correlation(
    signal: np.ndarray, template: np.ndarray
) -> float:
    """Compute normalised cross-correlation between signal and template."""
    if len(signal) != len(template):
        raise ValueError("Signal and template must have the same length")
    sig_norm = np.linalg.norm(signal)
    tpl_norm = np.linalg.norm(template)
    if sig_norm == 0 or tpl_norm == 0:
        return 0.0
    return float(np.dot(signal, template) / (sig_norm * tpl_norm))


def detect_clave(
    onset_grid: List[int],
    threshold: float = 0.7,
) -> ClaveMatch:
    """Detect which clave pattern best matches the given onset grid.

    Args:
        onset_grid: A 16-element binary list where 1 indicates a percussive
            onset and 0 indicates silence.
        threshold: Minimum correlation for a positive match.

    Returns:
        A :class:`ClaveMatch` with the best-matching pattern.
    """
    if len(onset_grid) != 16:
        raise ValueError(f"onset_grid must have 16 elements, got {len(onset_grid)}")

    signal = np.array(onset_grid, dtype=float)
    best_type = ClaveType.UNKNOWN
    best_corr = -1.0

    for name, pattern in CLAVE_PATTERNS.items():
        template = np.array(pattern, dtype=float)
        corr = _normalised_cross_correlation(signal, template)
        if corr > best_corr:
            best_corr = corr
            best_type = ClaveType(name)

    confidence = max(0.0, min(1.0, best_corr))
    if confidence < threshold:
        best_type = ClaveType.UNKNOWN

    return ClaveMatch(
        pattern_type=best_type,
        confidence=confidence,
        correlation=best_corr,
        onset_positions=[i for i, v in enumerate(onset_grid) if v],
    )


def generate_clave_pattern(
    clave_type: ClaveType | str,
    bars: int = 2,
    swing: float = 0.0,
) -> List[int]:
    """Generate a clave pattern for the specified number of bars.

    Args:
        clave_type: Which clave pattern to generate.
        bars: Number of 2-bar repetitions (total bars = bars * 2).
        swing: Swing amount (0.0 to 1.0) — currently for metadata only.

    Returns:
        Binary onset grid (16 steps per 2 bars).
    """
    key = clave_type if isinstance(clave_type, str) else clave_type.value
    if key not in CLAVE_PATTERNS:
        raise ValueError(f"Unknown clave type: {key}")

    base = CLAVE_PATTERNS[key]
    return base * bars
