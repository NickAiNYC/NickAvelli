CHAINS = {
    "digital_santeria": [
        ("isolate_clave", {"transient_threshold": 0.1}),
        ("timestretch", {"ratio": 3.0, "preserve_transients": True}),
        ("convolve", {"impulse": "field_recordings/subway_rumble.wav"}),
        ("repitch", {"semitones": -12})
    ],
    "bodega_radio": [
        ("bandpass", {"low": 200, "high": 800}),
        ("fm_degradation", {"noise_level": 0.1}),
        ("resample", {"target_sr": 22050})
    ]
}

def apply_chain(input_path: Path, chain_name: str, output_path: Path):
    """Apply destruction chain, log every step for provenance"""
    chain = CHAINS[chain_name]
    for step, params in chain:
        # Apply step
        # Log to provenance doc
        pass
