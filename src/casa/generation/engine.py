"""Audio generation engine for Latina Afro House.

Provides the core generation pipeline using MusicGen (or compatible
models) with style prompts tailored to Afro House production.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.casa.analysis.clave import ClaveType, generate_clave_pattern
from src.casa.utils.config import AudioConfig, GenerationConfig
from src.casa.utils.logging import get_logger

logger = get_logger(__name__)

# Pre-defined prompt templates for various Afro House sub-styles
STYLE_PROMPTS: Dict[str, str] = {
    "latin_afro_house": (
        "{bpm} bpm afro house track with latin percussion, "
        "conga rhythms, shaker grooves, deep bass, "
        "warm pads, tribal energy, {key} key"
    ),
    "deep_afro": (
        "{bpm} bpm deep afro house, hypnotic percussion, "
        "minimal melodic elements, sub bass, organic textures, {key} key"
    ),
    "vocal_afro": (
        "{bpm} bpm afro house with spanish vocal chants, "
        "call and response, percussive polyrhythms, {key} key"
    ),
    "tribal_tech": (
        "{bpm} bpm tribal tech house, driving percussion, "
        "clave patterns, syncopated hi-hats, rolling bassline, {key} key"
    ),
}


@dataclass
class GenerationResult:
    """Result from the audio generation engine."""

    audio: np.ndarray
    sample_rate: int
    prompt: str
    duration_sec: float
    generation_time_sec: float
    model_name: str
    clave_pattern: Optional[List[int]] = None
    metadata: Dict = field(default_factory=dict)


class AfroHouseGenerator:
    """AI-powered Afro House track generator.

    Wraps MusicGen (or compatible models) with Afro House specific
    prompt engineering and clave pattern integration.

    Args:
        config: Audio configuration.
        gen_config: Generation model configuration.
    """

    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        gen_config: Optional[GenerationConfig] = None,
    ) -> None:
        self.config = config or AudioConfig()
        self.gen_config = gen_config or GenerationConfig()
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        """Lazily load the MusicGen model."""
        if self._model is not None:
            return
        try:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration

            logger.info("Loading model: %s", self.gen_config.model_name)
            self._processor = AutoProcessor.from_pretrained(
                self.gen_config.model_name
            )
            self._model = MusicgenForConditionalGeneration.from_pretrained(
                self.gen_config.model_name
            )
            if self.gen_config.device != "cpu":
                self._model = self._model.to(self.gen_config.device)
            logger.info("Model loaded on %s", self.gen_config.device)
        except ImportError:
            logger.warning(
                "transformers not installed. Generation will produce "
                "silence placeholder."
            )
        except Exception:
            logger.exception("Failed to load model %s", self.gen_config.model_name)

    def build_prompt(
        self,
        style: str = "latin_afro_house",
        bpm: Optional[int] = None,
        key: Optional[str] = None,
        extra: str = "",
    ) -> str:
        """Build a generation prompt from a style template.

        Args:
            style: Style key from :data:`STYLE_PROMPTS`.
            bpm: Target BPM (uses config default if not given).
            key: Musical key (uses first preference if not given).
            extra: Additional text to append.

        Returns:
            Formatted prompt string.
        """
        template = STYLE_PROMPTS.get(style, STYLE_PROMPTS["latin_afro_house"])
        bpm = bpm or self.config.target_bpm
        key = key or self.config.key_preferences[0]
        prompt = template.format(bpm=bpm, key=key)
        if extra:
            prompt = f"{prompt}, {extra}"
        return prompt

    def generate(
        self,
        prompt: Optional[str] = None,
        style: str = "latin_afro_house",
        duration_sec: float = 30.0,
        bpm: Optional[int] = None,
        key: Optional[str] = None,
        clave_type: Optional[ClaveType] = None,
    ) -> GenerationResult:
        """Generate an Afro House audio clip.

        Args:
            prompt: Custom text prompt. If *None*, builds from ``style``.
            style: Style template key (ignored when ``prompt`` is given).
            duration_sec: Desired duration in seconds.
            bpm: Target BPM.
            key: Musical key.
            clave_type: Optional clave pattern to embed in metadata.

        Returns:
            A :class:`GenerationResult` with the generated audio.
        """
        self._load_model()

        if prompt is None:
            prompt = self.build_prompt(style=style, bpm=bpm, key=key)

        clave = None
        if clave_type is not None:
            clave = generate_clave_pattern(clave_type, bars=4)

        t0 = time.time()

        if self._model is not None and self._processor is not None:
            audio, sr = self._generate_with_model(prompt, duration_sec)
        else:
            # Placeholder: generate silence at target sample rate
            sr = self.config.sample_rate
            audio = np.zeros(int(sr * duration_sec), dtype=np.float32)
            logger.warning("Model not available; generated silence placeholder")

        gen_time = time.time() - t0

        return GenerationResult(
            audio=audio,
            sample_rate=sr,
            prompt=prompt,
            duration_sec=len(audio) / sr,
            generation_time_sec=round(gen_time, 2),
            model_name=self.gen_config.model_name,
            clave_pattern=clave,
            metadata={
                "bpm": bpm or self.config.target_bpm,
                "key": key or self.config.key_preferences[0],
                "style": style,
            },
        )

    def _generate_with_model(
        self, prompt: str, duration_sec: float
    ) -> tuple[np.ndarray, int]:
        """Run inference through the loaded model."""
        import torch

        inputs = self._processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        )
        if self.gen_config.device != "cpu":
            inputs = {k: v.to(self.gen_config.device) for k, v in inputs.items()}

        # Approximate tokens for desired duration
        # MusicGen generates at ~50 tokens/sec at 32 kHz
        tokens_per_sec = 50
        max_tokens = int(duration_sec * tokens_per_sec)
        max_tokens = min(max_tokens, self.gen_config.max_new_tokens)

        with torch.no_grad():
            audio_values = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                guidance_scale=self.gen_config.guidance_scale,
                temperature=self.gen_config.temperature,
                do_sample=True,
                top_k=self.gen_config.top_k,
            )

        audio_np = audio_values[0, 0].cpu().numpy()
        sr = self._model.config.audio_encoder.sampling_rate
        return audio_np, sr
