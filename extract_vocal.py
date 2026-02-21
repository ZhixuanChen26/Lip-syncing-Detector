#extract_vocal.py

import os
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
import soundfile as sf
import numpy as np

def extract_vocals(wave_array, sr=44100, save_output=False, output_path="extracted_vocals.wav"):
    """Extract vocals from mixed audio using Spleeter"""
    # Write to temporary file for Spleeter processing
    temp_input = "temp_input.wav"
    sf.write(temp_input, wave_array, sr)

    # 2 stems means vocals + accompaniment
    separator = Separator('spleeter:2stems')
    # Load data using the AudioAdapter
    audio_loader = AudioAdapter.default()
    waveform, _ = audio_loader.load(temp_input, sample_rate=sr)
    
    # Extract vocal as NumPy array
    prediction = separator.separate(waveform)
    vocals = prediction['vocals']
    
    # Convert to mono if needed
    if vocals.ndim > 1:
        vocals = np.mean(vocals, axis=1)
    
    # Normalize the output
    max_val = np.max(np.abs(vocals)) + 1e-9  # Avoid division by zero
    vocals = vocals / max_val
    
    # Delete temp file
    if os.path.exists(temp_input):
        os.remove(temp_input)
    
    # Save extracted vocals if requested
    if save_output:
        sf.write(output_path, vocals, sr)
        print(f"The extracted vocals have been saved to: {output_path}")

    return vocals