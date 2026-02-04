# src/curation/gatekeeper.py
import os
import shutil
import json
import hashlib
import yaml
from datetime import datetime, timedelta
from pathlib import Path
import librosa  # For audio validation

class RefineryGatekeeper:
    def __init__(self, config_path="config/constraints.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.latency_days = self.config.get("latency_days", 14)
        self.staging_file = "data/staging_queue.json"
        self.white_box = Path(self.config["white_box"])
        self.white_box.mkdir(parents=True, exist_ok=True)
        
        self._ensure_staging()

    def process(self, file_path, metadata):
        """Full pipeline: validate → stage → (later) release"""
        if not self._validate(file_path, metadata):
            return False
        
        self._stage(file_path, metadata)
        return True

    def _validate(self, file_path, metadata):
        """All checks from your 15% rule + genre constraints"""
        checks = {
            "ai_percentage": self._check_ai_percentage(file_path),
            "field_layer": self._check_field_layer(file_path),
            "payphone_present": "payphone_id" in metadata,
            "destruction_verified": metadata.get("destruction_chain") in self.config["valid_chains"],
            "tempo_range": self._check_tempo(file_path, 118, 126),
            "duration": self._check_duration(file_path, 3.0, 300.0),  # 3sec to 5min
        }
        
        failed = [k for k, v in checks.items() if not v]
        if failed:
            print(f"❌ REJECTED: {failed}")
            return False
        return True

    def _stage(self, file_path, metadata):
        """Add to latency queue"""
        release_date = (datetime.now() + timedelta(days=self.latency_days)).isoformat()
        entry = {
            "file": str(file_path),
            "metadata": metadata,
            "queued_at": datetime.now().isoformat(),
            "release_date": release_date,
            "hash": self._file_hash(file_path)
        }
        
        queue = self._load_staging()
        queue.append(entry)
        self._save_staging(queue)
        print(f"🕒 Staged for {self.latency_days} days: {Path(file_path).name}")

    def release_ready(self):
        """Check and auto-release if ready"""
        queue = self._load_staging()
        now = datetime.now()
        
        ready = []
        pending = []
        
        for item in queue:
            release_dt = datetime.fromisoformat(item["release_date"])
            if release_dt <= now:
                if self._verify_file_integrity(item):
                    self._promote_to_white_box(item)
                    ready.append(item)
                else:
                    print(f"⚠️ File modified since staging: {item['file']}")
            else:
                pending.append(item)
        
        self._save_staging(pending)
        return ready

    def _verify_file_integrity(self, item):
        """Ensure file wasn't modified during latency period"""
        current_hash = self._file_hash(item["file"])
        return current_hash == item["hash"]

    def _promote_to_white_box(self, item):
        """Clean promotion with provenance"""
        src = Path(item["file"])
        new_name = self._generate_name(item["metadata"])
        dest = self.white_box / new_name
        
        shutil.copy2(src, dest)
        self._create_provenance(dest, item)
        print(f"✅ RELEASED: {new_name}")

    def _generate_name(self, metadata):
        """Clean anonymous naming"""
        date = datetime.now().strftime("%y%m%d")
        # Include payphone ID in filename for your constraint
        payphone = metadata.get("payphone_id", "NO_PHONE")
        hash_id = hashlib.md5(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest()[:6]
        return f"REF_{date}_{payphone}_{hash_id}.wav"

    def _create_provenance(self, dest_path, item):
        """Certificate of process"""
        prov_path = dest_path.with_suffix(".json")
        provenance = {
            "audio_file": str(dest_path.name),
            "release_date": datetime.now().isoformat(),
            "latency_days": self.latency_days,
            "original_file": item["file"],
            "metadata": item["metadata"],
            "verification_hash": item["hash"]
        }
        with open(prov_path, "w") as f:
            json.dump(provenance, f, indent=2)

    def _file_hash(self, file_path):
        """MD5 of file content"""
        import hashlib
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    # Audio validation stubs (implement with librosa)
    def _check_ai_percentage(self, file_path):
        # Call your audio_analysis module
        from audio_analysis import calculate_ai_percentage
        return calculate_ai_percentage(file_path) <= 15

    def _check_field_layer(self, file_path):
        from audio_analysis import verify_field_layer
        return verify_field_layer(file_path)

    def _check_tempo(self, file_path, min_bpm, max_bpm):
        y, sr = librosa.load(file_path, sr=None)
        tempo = librosa.beat.tempo(y=y, sr=sr)[0]
        return min_bpm <= tempo <= max_bpm

    def _check_duration(self, file_path, min_sec, max_sec):
        duration = librosa.get_duration(path=file_path)
        return min_sec <= duration <= max_sec

    def _ensure_staging(self):
        Path(self.staging_file).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.staging_file):
            self._save_staging([])

    def _load_staging(self):
        with open(self.staging_file) as f:
            return json.load(f)

    def _save_staging(self, queue):
        with open(self.staging_file, "w") as f:
            json.dump(queue, f, indent=2)
