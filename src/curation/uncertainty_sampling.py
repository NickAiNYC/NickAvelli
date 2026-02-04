# src/curation/uncertainty_sampling.py
import torch
from transformers import Wav2Vec2ForSequenceClassification

class UncertaintySampler:
    def __init__(self):
        self.source_identifier = Wav2Vec2ForSequenceClassification.from_pretrained(
            "models/source_identifier"  # Trained to recognize your AI prompts
        )
    
    def calculate_uncertainty(self, audio_path):
        """Returns 0.0 (definitely source) to 1.0 (unrecognizable)"""
        logits = self.source_identifier(audio_path)
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs))
        return entropy.item()  # High entropy = uncertain = good destruction
    
    def filter(self, audio_files, threshold=0.7):
        """Keep only files where source is uncertain"""
        survivors = []
        for f in audio_files:
            if self.calculate_uncertainty(f) > threshold:
                survivors.append(f)
        return survivors
