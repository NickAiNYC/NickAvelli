"""Tests for the FastAPI routes."""

import pytest
from httpx import ASGITransport, AsyncClient

from src.casa.api.routes import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.anyio
async def test_config_endpoint(client):
    resp = await client.get("/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "audio" in data
    assert data["audio"]["target_bpm"] == 124


@pytest.mark.anyio
async def test_clave_analysis(client):
    # Son clave 3-2 pattern
    onset = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0]
    resp = await client.post("/analyse/clave", json={"onset_grid": onset})
    assert resp.status_code == 200
    data = resp.json()
    assert data["pattern_type"] == "son_3_2"
    assert data["confidence"] == pytest.approx(1.0)


@pytest.mark.anyio
async def test_clave_invalid_grid(client):
    resp = await client.post("/analyse/clave", json={"onset_grid": [1, 0]})
    assert resp.status_code == 422  # validation error


@pytest.mark.anyio
async def test_vocal_phrase(client):
    resp = await client.post("/vocals/phrase", json={"language": "es", "pairs": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["phrases"]) == 4


@pytest.mark.anyio
async def test_generate_returns_job(client):
    resp = await client.post(
        "/generate",
        json={"style": "latin_afro_house", "bpm": 124, "duration_sec": 5.0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] in ("pending", "processing", "completed")


@pytest.mark.anyio
async def test_job_not_found(client):
    resp = await client.get("/jobs/nonexistent-id")
    assert resp.status_code == 404
