"""
pitch_variance_analysis.py - Pitch variance analysis algorithm
"""

import time
import numpy as np
from cache_manager import cache_manager, cached
from compare import extract_segment
from audio_processor import audio_processor


# Core analysis functions
@cached("feature_cache", "pitch_variance")
def _pitch_variance(pitch):
    """Calculate variance of valid pitch values"""
    valid = pitch[~np.isnan(pitch)]
    return float(np.var(valid)) if len(valid) > 1 else 0.0


@cached("feature_cache", "pitch_fluctuations")
def _pitch_fluctuations(pitch, window=5):
    """Calculate sliding window variance for pitch data"""
    valid = pitch[~np.isnan(pitch)]
    if len(valid) <= window:
        return np.zeros(1)
    
    # Vectorized sliding window implementation
    strides = (valid.strides[0], valid.strides[0])
    views = np.lib.stride_tricks.as_strided(
        valid, shape=(len(valid) - window + 1, window), strides=strides
    )
    return np.var(views, axis=1)


def _chebyshev_bounds(var_arr, conf=0.8):
    """Calculate lower bound using Chebyshev's inequality"""
    mean, std = np.mean(var_arr), np.std(var_arr)
    if std == 0:
        return mean * 0.5
    k = np.sqrt(1 / (1 - conf))
    bound = mean - k * std * 0.8
    return max(bound, mean * 0.3)


def analyze_pitch_variance_chunked(
    live_path, 
    cd_path, 
    sr=44100,
    confidence=0.9,
    window_size=5,
    chunk_sec=15,
    shared_data=None
):
    """
    Check pitch variance consistency by chunks
    """
    start_time = time.time()
    shared_data = shared_data or {}
    
    # Get audio data from cache or shared data
    live_wave = shared_data.get("live_wave")
    cd_wave = shared_data.get("cd_wave")
    
    if live_wave is None or cd_wave is None:
        live_wave = cache_manager.get_shared_data("live_wave")
        cd_wave = cache_manager.get_shared_data("cd_wave")
    
    if live_wave is None or cd_wave is None:
        raise ValueError("Audio data not loaded. Call audio_processor.load_and_prepare_audio first.")
    
    # Get offset information
    offset_data = audio_processor._get_cached_offset(live_path, cd_path)
    base_offset = offset_data['offset'] if offset_data else 0
    
    # Determine which chunks to analyze and in what order
    chunk_size = int(chunk_sec * sr)
    num_chunks = len(live_wave) // chunk_size
    
    detected_chunk = shared_data.get("detected_chunk")
    if detected_chunk is None:
        detected_chunk = cache_manager.get_shared_data("detected_chunk", -1)
    
    chunk_indices = [detected_chunk] + [i for i in range(num_chunks + 1) if i != detected_chunk] if detected_chunk >= 0 else list(range(num_chunks + 1))
    
    # Process each chunk
    for idx in chunk_indices:
        start_sample = idx * chunk_size
        live_chunk = extract_segment(live_wave, sr, start_sample, desired_length_sec=chunk_sec, min_sec=10)
        if live_chunk is None:
            continue
        
        cd_start = base_offset + start_sample
        cd_chunk = extract_segment(cd_wave, sr, cd_start, desired_length=len(live_chunk), min_sec=10)
        if cd_chunk is None:
            continue
        
        # Flag important chunks (first or detected)
        is_important_chunk = (idx == 0 or idx == detected_chunk)
        
        # Extract vocals
        from extract_vocal import extract_vocals as extractor_func
        
        # For important chunks, try to get from shared_data or global cache first
        live_vocals = None
        cd_vocals = None
        
        if is_important_chunk:
            if "live_vocals" in shared_data:
                live_vocals = shared_data["live_vocals"]
            elif cache_manager.get_shared_data("live_vocals") is not None:
                live_vocals = cache_manager.get_shared_data("live_vocals")
                
            if "cd_vocals" in shared_data:
                cd_vocals = shared_data["cd_vocals"]
            elif cache_manager.get_shared_data("cd_vocals") is not None:
                cd_vocals = cache_manager.get_shared_data("cd_vocals")
        
        # If not found in cache, extract vocals
        if live_vocals is None:
            live_vocals = cache_manager.extract_vocals(live_chunk, sr=sr, extractor_func=extractor_func)
        
        if cd_vocals is None:
            cd_vocals = cache_manager.extract_vocals(cd_chunk, sr=sr, extractor_func=extractor_func)
        
        # Get pitch data
        f0_live = None
        f0_cd = None
        
        if is_important_chunk:
            if "f0_live" in shared_data:
                f0_live = shared_data["f0_live"]
            elif cache_manager.get_shared_data("f0_live") is not None:
                f0_live = cache_manager.get_shared_data("f0_live")
                
            if "f0_cd" in shared_data:
                f0_cd = shared_data["f0_cd"]
            elif cache_manager.get_shared_data("f0_cd") is not None:
                f0_cd = cache_manager.get_shared_data("f0_cd")
        
        if f0_live is None or f0_cd is None:
            _, f0_live = cache_manager.get_pitch_data(live_vocals, sr=sr)
            _, f0_cd = cache_manager.get_pitch_data(cd_vocals, sr=sr)
                
        # Actual variance analysis
        var_live = _pitch_variance(f0_live)
        var_cd = _pitch_variance(f0_cd)
        fluc_live = _pitch_fluctuations(f0_live, window_size)
        lower_bound = _chebyshev_bounds(fluc_live, confidence)
        
        # Calculate variance ratio and determine if fake
        ratio = var_live / var_cd if var_cd > 0 else 1.0
        is_fake = (var_live < lower_bound) or (0.95 < ratio < 1.05)
        
        if is_fake:
            explanation = f"Variance ratio {ratio:.3f} (Lip-sync)"
            
            # Update both shared data and global cache
            if shared_data is not None:
                shared_data["detected_chunk_var"] = idx
                shared_data["detected_ratio_var"] = ratio
            
            cache_manager.update_shared_data({
                "detected_chunk_var": idx,
                "detected_ratio_var": ratio
            })
            
            return True, idx, ratio, explanation, time.time() - start_time
    
    # No suspicious patterns detected in any chunk
    explanation = "No suspicious variance pattern detected (Real singing)"
    return False, -1, 0.0, explanation, time.time() - start_time