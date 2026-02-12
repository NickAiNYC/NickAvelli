"""Latin vocal phrase generation utilities.

Provides randomised Spanish/Portuguese vocal phrases for embedding
in Afro House tracks, following common Latin music vocal patterns.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from src.casa.utils.logging import get_logger

logger = get_logger(__name__)

# Curated phrase banks for call-and-response patterns
SPANISH_PHRASES: List[str] = [
    "dale fuego",
    "siente el ritmo",
    "vamos a bailar",
    "la noche es nuestra",
    "mueve el cuerpo",
    "la clave suena",
    "desde la tierra",
    "en la oscuridad",
    "fuego en el alma",
    "el tambor llama",
]

PORTUGUESE_PHRASES: List[str] = [
    "sente o batuque",
    "vem dançar",
    "a noite é nossa",
    "fogo na alma",
    "o tambor chama",
    "ritmo do corpo",
    "na escuridão",
    "desde a terra",
    "o som da rua",
    "dança comigo",
]


@dataclass
class VocalPhrase:
    """A vocal phrase for use in track generation."""

    text: str
    language: str  # "es" or "pt"
    category: str  # "call", "response", "chant"
    energy: float  # 0.0 – 1.0


def random_phrase(
    language: str = "es",
    category: str = "chant",
    seed: Optional[int] = None,
) -> VocalPhrase:
    """Select a random vocal phrase.

    Args:
        language: ``"es"`` for Spanish, ``"pt"`` for Portuguese.
        category: Vocal category label.
        seed: Optional random seed.

    Returns:
        A :class:`VocalPhrase` instance.
    """
    rng = random.Random(seed)
    bank = SPANISH_PHRASES if language == "es" else PORTUGUESE_PHRASES
    text = rng.choice(bank)
    energy = rng.uniform(0.5, 1.0)
    return VocalPhrase(text=text, language=language, category=category, energy=energy)


def build_call_response(
    language: str = "es",
    pairs: int = 2,
    seed: Optional[int] = None,
) -> List[VocalPhrase]:
    """Build a call-and-response vocal sequence.

    Args:
        language: Language code.
        pairs: Number of call/response pairs.
        seed: Optional random seed.

    Returns:
        List of :class:`VocalPhrase` alternating call and response.
    """
    rng = random.Random(seed)
    bank = SPANISH_PHRASES if language == "es" else PORTUGUESE_PHRASES
    result: List[VocalPhrase] = []
    for _ in range(pairs):
        call_text = rng.choice(bank)
        response_text = rng.choice(bank)
        result.append(
            VocalPhrase(
                text=call_text,
                language=language,
                category="call",
                energy=rng.uniform(0.7, 1.0),
            )
        )
        result.append(
            VocalPhrase(
                text=response_text,
                language=language,
                category="response",
                energy=rng.uniform(0.5, 0.8),
            )
        )
    return result
