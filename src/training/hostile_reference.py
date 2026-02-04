# src/training/hostile_reference.py
import random
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
import yaml

class HostileReferenceInjector:
    def __init__(self, config_path: str = "config/hostile_refs.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.library_path = Path(self.config.get("library_path", "data/hostile_library.json"))
        self.analysis_dir = Path(self.config.get("analysis_dir", "data/hostile_analysis"))
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        self.reference_library = self._load_library()
        self.disliked_genres = self.config.get("disliked_genres", [])
        
    def _load_library(self) -> Dict:
        """Load or initialize reference library"""
        if self.library_path.exists():
            with open(self.library_path) as f:
                return json.load(f)
        
        # Initialize with config defaults
        library = {genre: [] for genre in self.disliked_genres}
        self._save_library(library)
        return library
    
    def _save_library(self, library: Dict):
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.library_path, 'w') as f:
            json.dump(library, f, indent=2)
    
    def add_reference(self, genre: str, track: str, audio_path: Optional[str] = None):
        """Add track to hostile library with optional audio file"""
        if genre not in self.reference_library:
            self.reference_library[genre] = []
        
        entry = {
            "track": track,
            "added": datetime.now().isoformat(),
            "audio_path": audio_path
        }
        self.reference_library[genre].append(entry)
        self._save_library(self.reference_library)
        print(f"Added to {genre}: {track}")
    
    def session(self, interactive: bool = True, genre: Optional[str] = None) -> Dict:
        """
        Hostile reference session
        interactive=False for automated/logging mode
        """
        # Select genre
        if genre is None:
            genre = random.choice(self.disliked_genres)
        
        refs = self.reference_library.get(genre, [])
        if not refs:
            print(f"No references for {genre}")
            return {}
        
        ref = random.choice(refs)
        track_name = ref["track"] if isinstance(ref, dict) else ref
        
        print(f"\n🎯 HOSTILE REFERENCE: {genre.upper()}")
        print(f"Target: {track_name}")
        
        # Play audio if available and interactive
        audio_path = ref.get("audio_path") if isinstance(ref, dict) else None
        
        if interactive and audio_path and Path(audio_path).exists():
            self._play_audio(audio_path)
        elif interactive:
            print("⚠️ No audio file, manual listening required")
        
        # Capture analysis
        if interactive:
            analysis = self._interactive_analysis(genre, track_name)
        else:
            analysis = self._batch_analysis(genre, track_name)
        
        self._save_analysis(analysis)
        self._update_generation_prompts(analysis)  # Feed to AI prompts
        return analysis
    
    def _play_audio(self, audio_path: str):
        """Cross-platform audio playback"""
        try:
            if sys.platform == "darwin":  # macOS
                subprocess.run(["afplay", audio_path], check=True)
            elif sys.platform == "linux":
                subprocess.run(["mpg123", "-q", audio_path], check=True)
            else:  # Windows
                os.startfile(audio_path)
        except Exception as e:
            print(f"⚠️ Could not play audio: {e}")
            print(f"Manual play: {audio_path}")
    
    def _interactive_analysis(self, genre: str, track: str) -> Dict:
        """Full interactive session"""
        print("\nListen once. Identify:")
        
        effective = []
        for i in range(1, 4):
            elem = input(f"  {i}. Effective element: ")
            effective.append(elem)
        
        appeal = input("\nWhy do people like this (non-judgmental)? ")
        stealable = input("\nWhat to STEAL (not copy)? ")
        
        return {
            "genre": genre,
            "track": track,
            "effective_elements": effective,
            "audience_appeal": appeal,
            "stealable_element": stealable,
            "session_type": "interactive",
            "timestamp": datetime.now().isoformat()
        }
    
    def _batch_analysis(self, genre: str, track: str) -> Dict:
        """Quick logging for batch mode"""
        return {
            "genre": genre,
            "track": track,
            "effective_elements": [],  # Fill later if needed
            "audience_appeal": "",
            "stealable_element": "pending_review",
            "session_type": "batch",
            "timestamp": datetime.now().isoformat(),
            "needs_review": True
        }
    
    def _save_analysis(self, analysis: Dict):
        """Save with date-based filename"""
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.analysis_dir / f"hostile_{date_str}.json"
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"💾 Saved: {filename.name}")
    
    def _update_generation_prompts(self, analysis: Dict):
        """Feed stealable elements into AI prompt library"""
        stealable = analysis.get("stealable_element", "")
        if not stealable or stealable == "pending_review":
            return
        
        prompt_lib_path = Path("config/prompt_library.yaml")
        if not prompt_lib_path.exists():
            return
        
        with open(prompt_lib_path) as f:
            prompts = yaml.safe_load(f)
        
        # Add to "hostile_derived" techniques
        if "hostile_derived" not in prompts:
            prompts["hostile_derived"] = []
        
        prompts["hostile_derived"].append({
            "technique": stealable,
            "source": analysis["track"],
            "date": analysis["timestamp"]
        })
        
        with open(prompt_lib_lib_path, 'w') as f:
            yaml.dump(prompts, f)
        
        print(f"➕ Added to prompt library: {stealable[:50]}...")
    
    def review_pending(self):
        """Review batch sessions that need interactive analysis"""
        pending = list(self.analysis_dir.glob("hostile_*.json"))
        
        for file in pending:
            with open(file) as f:
                analysis = json.load(f)
            
            if analysis.get("needs_review"):
                print(f"\n🔍 Review: {analysis['track']}")
                print(f"Previous: {analysis.get('stealable_element', 'none')}")
                
                new_analysis = self._interactive_analysis(
                    analysis["genre"], 
                    analysis["track"]
                )
                
                # Update file
                with open(file, 'w') as f:
                    json.dump(new_analysis, f, indent=2)
                print("✅ Updated")
    
    def quarterly_report(self) -> Dict:
        """Analyze patterns in hostile sessions"""
        analyses = []
        for file in self.analysis_dir.glob("hostile_*.json"):
            with open(file) as f:
                analyses.append(json.load(f))
        
        if not analyses:
            return {}
        
        # Pattern analysis
        stealable_freq = {}
        genre_freq = {}
        effective_freq = {}
        
        for a in analyses:
            # Stealable elements
            elem = a.get("stealable_element", "unknown")
            stealable_freq[elem] = stealable_freq.get(elem, 0) + 1
            
            # Genres
            g = a.get("genre", "unknown")
            genre_freq[g] = genre_freq.get(g, 0) + 1
            
            # Effective elements
            for e in a.get("effective_elements", []):
                effective_freq[e] = effective_freq.get(e, 0) + 1
        
        report = {
            "total_sessions": len(analyses),
            "top_stealable": sorted(stealable_freq.items(), key=lambda x: x[1], reverse=True)[:5],
            "genre_breakdown": genre_freq,
            "common_effective": sorted(effective_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        
        # Save report
        report_path = self.analysis_dir / "quarterly_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Quarterly Report ({len(analyses)} sessions)")
        print(f"Top stealable: {report['top_stealable'][0][0] if report['top_stealable'] else 'N/A'}")
        
        return report
