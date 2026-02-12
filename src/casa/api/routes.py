"""FastAPI backend for CASA.

Provides RESTful API endpoints for Afro House track generation,
metadata extraction, and mastering with async job processing.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.casa.analysis.clave import ClaveType, detect_clave, generate_clave_pattern
from src.casa.generation.vocals import build_call_response, random_phrase
from src.casa.mastering.processor import MasteringProcessor, estimate_lufs
from src.casa.utils.config import CASAConfig
from src.casa.utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="CASA API",
    description=(
        "Creative Afro House Synthesis Automation — "
        "AI-powered Latina Afro House music generation"
    ),
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    """Request body for track generation."""

    prompt: Optional[str] = None
    style: str = Field(
        default="latin_afro_house",
        description="Style preset: latin_afro_house, deep_afro, vocal_afro, tribal_tech",
    )
    bpm: int = Field(default=124, ge=100, le=140)
    key: str = Field(default="Am")
    duration_sec: float = Field(default=30.0, ge=5.0, le=300.0)
    clave_type: Optional[str] = None


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResponse(BaseModel):
    """Async job tracking."""

    job_id: str
    status: JobStatus
    message: str = ""
    result: Optional[Dict[str, Any]] = None


class ClaveDetectRequest(BaseModel):
    """Request body for clave detection."""

    onset_grid: List[int] = Field(
        ..., min_length=16, max_length=16,
        description="16-element binary onset grid",
    )
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class VocalRequest(BaseModel):
    """Request for vocal phrase generation."""

    language: str = Field(default="es", pattern="^(es|pt)$")
    pairs: int = Field(default=2, ge=1, le=8)


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_sec: float


# ---------------------------------------------------------------------------
# In-memory job store (production would use Redis)
# ---------------------------------------------------------------------------

_jobs: Dict[str, JobResponse] = {}
_start_time = time.time()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        uptime_sec=round(time.time() - _start_time, 1),
    )


@app.post("/generate", response_model=JobResponse, tags=["generation"])
async def generate_track(req: GenerateRequest) -> JobResponse:
    """Submit a track generation job.

    Returns a job ID that can be polled via ``/jobs/{job_id}``.
    """
    job_id = str(uuid.uuid4())
    job = JobResponse(job_id=job_id, status=JobStatus.PENDING, message="Queued")
    _jobs[job_id] = job

    # Launch async generation task
    asyncio.create_task(_run_generation(job_id, req))
    return job


@app.get("/jobs/{job_id}", response_model=JobResponse, tags=["jobs"])
async def get_job(job_id: str) -> JobResponse:
    """Check the status of a generation job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/analyse/clave", tags=["analysis"])
async def analyse_clave(req: ClaveDetectRequest) -> Dict[str, Any]:
    """Detect the clave pattern from a 16-step onset grid."""
    match = detect_clave(req.onset_grid, threshold=req.threshold)
    return {
        "pattern_type": match.pattern_type.value,
        "confidence": match.confidence,
        "correlation": match.correlation,
        "onset_positions": match.onset_positions,
    }


@app.post("/vocals/phrase", tags=["vocals"])
async def vocal_phrase(req: VocalRequest) -> Dict[str, Any]:
    """Generate random vocal call-and-response phrases."""
    phrases = build_call_response(language=req.language, pairs=req.pairs)
    return {
        "phrases": [
            {
                "text": p.text,
                "language": p.language,
                "category": p.category,
                "energy": round(p.energy, 2),
            }
            for p in phrases
        ]
    }


@app.get("/config", tags=["system"])
async def get_config() -> Dict[str, Any]:
    """Return the current CASA configuration."""
    cfg = CASAConfig()
    return {
        "audio": {
            "sample_rate": cfg.audio.sample_rate,
            "bpm_range": list(cfg.audio.bpm_range),
            "target_bpm": cfg.audio.target_bpm,
            "key_preferences": cfg.audio.key_preferences,
        },
        "mastering": {
            "target_lufs": cfg.mastering.target_lufs,
            "true_peak_dbfs": cfg.mastering.true_peak_dbfs,
            "stereo_width": cfg.mastering.stereo_width,
        },
    }


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------

async def _run_generation(job_id: str, req: GenerateRequest) -> None:
    """Execute generation in the background."""
    job = _jobs[job_id]
    job.status = JobStatus.PROCESSING
    job.message = "Generating audio..."
    try:
        # Run the CPU/GPU-bound generation in a thread pool
        from src.casa.generation.engine import AfroHouseGenerator

        generator = AfroHouseGenerator()
        clave = ClaveType(req.clave_type) if req.clave_type else None

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: generator.generate(
                prompt=req.prompt,
                style=req.style,
                duration_sec=req.duration_sec,
                bpm=req.bpm,
                key=req.key,
                clave_type=clave,
            ),
        )

        job.status = JobStatus.COMPLETED
        job.message = "Generation complete"
        job.result = {
            "prompt": result.prompt,
            "duration_sec": result.duration_sec,
            "generation_time_sec": result.generation_time_sec,
            "model": result.model_name,
            "metadata": result.metadata,
        }
    except Exception as exc:
        logger.exception("Generation failed for job %s", job_id)
        job.status = JobStatus.FAILED
        job.message = str(exc)
