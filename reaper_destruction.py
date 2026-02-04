# reaper_destruction.py - Runs inside REAPER
import reaper_python as RPR
import nn

# Load your trained RAVE model
model = nn.load("models/refinery_rave_2026w05.rt")

def destructive_pass(audio_path):
    # Neural time-stretch (not fixed 400%, varies 200-800% based on transient density)
    stretched = model.stretch(audio_path, ratio="adaptive")
    
    # Style transfer from field recording
    field_ir = random.choice(glob("field_recordings/payphones/*.wav"))
    destroyed = nn.style_transfer(stretched, field_ir, amount=0.7)
    
    # Granular with parameter modulation
    vital.granular(destroyed, 
                   density=analyze_transient_density(destroyed),
                   position=random.gauss(0.5, 0.2))
    
    return destroyed
