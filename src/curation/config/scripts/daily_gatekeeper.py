# scripts/daily_gatekeeper.py
#!/usr/bin/env python3
from src.curation.gatekeeper import RefineryGatekeeper

gk = RefineryGatekeeper()

# Evening: Process survivors from destruction
for survivor in Path("BLACK_BOX_RND/survivors").glob("*.wav"):
    metadata = json.load(survivor.with_suffix(".json"))
    gk.process(survivor, metadata)

# Check if anything ready for release
ready = gk.release_ready()
print(f"\n📦 {len(ready)} files released to White Box")
