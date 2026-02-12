"""Afro House structure analysis.

Detects structural sections (intro, build, drop, breakdown, outro) in
audio tracks and provides DJ-friendly section markers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

import numpy as np

from src.casa.utils.logging import get_logger

logger = get_logger(__name__)


class SectionType(str, Enum):
    """Track section categories."""

    INTRO = "intro"
    BUILD = "build"
    DROP = "drop"
    BREAKDOWN = "breakdown"
    OUTRO = "outro"


@dataclass
class Section:
    """A single structural section of a track."""

    section_type: SectionType
    start_sec: float
    end_sec: float
    energy: float  # 0.0 – 1.0

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class StructureAnalysis:
    """Full structural analysis of a track."""

    sections: List[Section] = field(default_factory=list)
    total_duration_sec: float = 0.0
    bpm: float = 0.0

    @property
    def has_dj_friendly_intro(self) -> bool:
        """Check if the intro is at least 16 bars long (DJ-friendly)."""
        if not self.sections:
            return False
        intro = self.sections[0]
        if intro.section_type != SectionType.INTRO:
            return False
        # 16 bars at ~124 BPM ≈ 31 seconds
        min_bars = 16
        bar_duration = 60.0 / max(self.bpm, 1) * 4
        return intro.duration_sec >= min_bars * bar_duration

    @property
    def has_dj_friendly_outro(self) -> bool:
        """Check if the outro is at least 16 bars long."""
        if not self.sections:
            return False
        outro = self.sections[-1]
        if outro.section_type != SectionType.OUTRO:
            return False
        min_bars = 16
        bar_duration = 60.0 / max(self.bpm, 1) * 4
        return outro.duration_sec >= min_bars * bar_duration


def analyse_structure(
    rms_envelope: np.ndarray,
    bpm: float,
    total_duration_sec: float,
    num_sections: int = 6,
) -> StructureAnalysis:
    """Analyse track structure from an RMS energy envelope.

    Uses energy-based segmentation to identify structural sections.

    Args:
        rms_envelope: 1-D array of RMS energy values over time.
        bpm: Detected BPM of the track.
        total_duration_sec: Total duration in seconds.
        num_sections: Target number of sections to detect.

    Returns:
        A :class:`StructureAnalysis` instance.
    """
    if len(rms_envelope) < num_sections:
        return StructureAnalysis(
            total_duration_sec=total_duration_sec, bpm=bpm
        )

    # Normalise
    env = rms_envelope.astype(float)
    env_max = env.max()
    if env_max > 0:
        env = env / env_max

    # Split into equal-length segments
    seg_len = len(env) // num_sections
    sections: List[Section] = []

    for i in range(num_sections):
        start_idx = i * seg_len
        end_idx = (i + 1) * seg_len if i < num_sections - 1 else len(env)
        seg_energy = float(np.mean(env[start_idx:end_idx]))

        start_sec = (start_idx / len(env)) * total_duration_sec
        end_sec = (end_idx / len(env)) * total_duration_sec

        # Classify section based on position and energy
        if i == 0:
            stype = SectionType.INTRO
        elif i == num_sections - 1:
            stype = SectionType.OUTRO
        elif seg_energy > 0.7:
            stype = SectionType.DROP
        elif seg_energy < 0.4:
            stype = SectionType.BREAKDOWN
        else:
            stype = SectionType.BUILD

        sections.append(
            Section(
                section_type=stype,
                start_sec=round(start_sec, 2),
                end_sec=round(end_sec, 2),
                energy=round(seg_energy, 3),
            )
        )

    return StructureAnalysis(
        sections=sections,
        total_duration_sec=total_duration_sec,
        bpm=bpm,
    )
