"""Dataset management for Afro House audio samples."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AudioSample:
    """Metadata for a single audio file in the dataset."""

    file_path: str
    bpm: float | None = None
    key: str | None = None
    duration_sec: float | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON persistence."""
        return {
            "file_path": self.file_path,
            "bpm": self.bpm,
            "key": self.key,
            "duration_sec": self.duration_sec,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioSample:
        """Deserialize from dictionary."""
        return cls(
            file_path=data["file_path"],
            bpm=data.get("bpm"),
            key=data.get("key"),
            duration_sec=data.get("duration_sec"),
            tags=data.get("tags", []),
        )


class AfroHouseDataset:
    """In-memory dataset of Afro House audio samples with manifest persistence."""

    ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}

    def __init__(
        self,
        root_dir: str | Path,
        manifest_path: str | Path | None = None,
    ):
        """Initialize dataset.

        Args:
            root_dir: Root directory containing audio files
            manifest_path: Optional path to manifest JSON file to load
        """
        self.root_dir = Path(root_dir).resolve()
        self.samples: list[AudioSample] = []

        if manifest_path:
            self._load_manifest(Path(manifest_path))

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> AudioSample:
        """Get sample by index."""
        return self.samples[idx]

    def scan(self) -> int:
        """Scan root directory for audio files.

        Returns:
            Number of audio files found

        Security: Only scans for allowed extensions to prevent path traversal attacks.
        """
        self.samples = []
        count = 0

        for file_path in self.root_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.ALLOWED_EXTENSIONS:
                # Path.resolve() prevents path traversal
                resolved_path = file_path.resolve()
                # Ensure file is within root_dir
                try:
                    resolved_path.relative_to(self.root_dir)
                except ValueError:
                    # File is outside root_dir, skip
                    continue

                self.samples.append(
                    AudioSample(file_path=str(file_path.relative_to(self.root_dir)))
                )
                count += 1

        return count

    def save_manifest(self, path: str | Path | None = None) -> Path:
        """Save dataset manifest to JSON file.

        Args:
            path: Optional path to save manifest. Defaults to root_dir/manifest.json

        Returns:
            Path to saved manifest file
        """
        if path is None:
            path = self.root_dir / "manifest.json"
        else:
            path = Path(path)

        manifest_data = {
            "root_dir": str(self.root_dir),
            "samples": [sample.to_dict() for sample in self.samples],
        }

        with open(path, "w") as f:
            json.dump(manifest_data, f, indent=2)

        return path

    def _load_manifest(self, path: Path) -> None:
        """Load dataset manifest from JSON file."""
        try:
            with open(path) as f:
                manifest_data = json.load(f)

            self.samples = [
                AudioSample.from_dict(sample_dict)
                for sample_dict in manifest_data.get("samples", [])
            ]
        except (FileNotFoundError, json.JSONDecodeError):
            # Gracefully handle missing or corrupt manifest
            self.samples = []

    def filter_by_bpm(self, min_bpm: float, max_bpm: float) -> list[AudioSample]:
        """Filter samples by BPM range.

        Args:
            min_bpm: Minimum BPM (inclusive)
            max_bpm: Maximum BPM (inclusive)

        Returns:
            List of samples within BPM range
        """
        return [
            sample
            for sample in self.samples
            if sample.bpm is not None and min_bpm <= sample.bpm <= max_bpm
        ]

    def filter_by_key(self, keys: list[str]) -> list[AudioSample]:
        """Filter samples by musical key.

        Args:
            keys: List of keys to filter by (e.g., ["Am", "C"])

        Returns:
            List of samples matching any of the specified keys
        """
        # Convert to set for O(1) lookup performance
        keys_set = set(keys)
        return [sample for sample in self.samples if sample.key in keys_set]
