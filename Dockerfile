FROM python:3.11-slim

WORKDIR /app

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
COPY config/ config/
COPY README.md .

RUN pip install --no-cache-dir ".[audio]"

EXPOSE 8000

CMD ["uvicorn", "src.casa.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
