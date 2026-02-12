# NickAvelli

## CASA — Creative Afro House Synthesis Automation

AI-powered music production system for generating professional Latina Afro House tracks.

### Quick Start

```bash
# Install core dependencies
pip install -e .

# Install with audio processing support
pip install -e ".[audio]"

# Install with ML model support
pip install -e ".[ml]"

# Install everything (including dev tools)
pip install -e ".[all]"

# Run tests
pytest tests/ -v

# Start the API server
uvicorn src.casa.api.routes:app --host 0.0.0.0 --port 8000
```

### Project Structure

```
src/casa/
├── analysis/        # Clave recognition, metadata extraction, structure analysis
│   ├── clave.py     # Afro-Cuban clave pattern detection & generation
│   ├── metadata.py  # BPM, key, energy, instrumentation extraction
│   └── structure.py # Track section analysis (intro/drop/outro)
├── data/            # Data pipeline
│   ├── augmentation.py  # Pitch shift, time stretch, EQ simulation
│   ├── dataset.py       # AfroHouseDataset class with manifest management
│   └── stems.py         # Demucs-based stem separation
├── generation/      # Audio generation
│   ├── engine.py    # MusicGen-based Afro House generator
│   └── vocals.py    # Latin vocal phrase randomiser (Spanish/Portuguese)
├── mastering/       # Mastering chain
│   └── processor.py # LUFS normalisation, true-peak limiting, stereo widening
├── api/             # REST API
│   └── routes.py    # FastAPI endpoints for generation, analysis, vocals
└── utils/           # Shared utilities
    ├── config.py    # YAML-based configuration management
    └── logging.py   # Logging setup
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/config` | Current configuration |
| POST | `/generate` | Submit a track generation job |
| GET | `/jobs/{job_id}` | Check generation job status |
| POST | `/analyse/clave` | Detect clave pattern from onset grid |
| POST | `/vocals/phrase` | Generate Latin vocal phrases |

### Docker

```bash
docker build -t casa .
docker run -p 8000:8000 casa
```

### Configuration

Edit `config/casa.yaml` to customise audio parameters, BPM range, mastering targets, and model settings. See `src/casa/utils/config.py` for all available options.

### Key Features

- **Clave Pattern Recognition** — Detects son, rumba, and bossa nova clave patterns
- **120–128 BPM Optimised** — Tailored for Latina Afro House tempo range
- **AI Mastering** — LUFS -14 loudness, true peak -1 dBFS, stereo widening
- **Latin Vocals** — Spanish/Portuguese call-and-response phrase generation
- **DJ-Friendly Structure** — Automated intro/outro length validation
- **Stem Separation** — Demucs integration for vocals/drums/bass/other
- **Audio Augmentation** — Pitch shift, time stretch, EQ for dataset expansion