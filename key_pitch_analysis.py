"""
key_pitch_analysis.py - Key-based pitch analysis algorithm
"""

import numpy as np
import time
from cache_manager import cache_manager, cached, chunk_data_cache
from compare import extract_segment, find_best_offset
from get_spectrum import plot_key_comparison, get_pitch_and_times

# Music theory constants
KEY_TO_FREQUENCY = {
    'C': 261.63, 'C#': 277.18, 'Db': 277.18, 'D': 293.66,
    'D#': 311.13, 'Eb': 311.13, 'E': 329.63, 'F': 349.23,
    'F#': 369.99, 'Gb': 369.99, 'G': 392.00, 'G#': 415.30, 
    'Ab': 415.30, 'A': 440.00, 'A#': 466.16, 'Bb': 466.16, 'B': 493.88
}

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]

# Utility function to calculate pitch range
def calculate_pitch_range(pitch_data):
    """Calculate effective pitch range after removing outliers"""
    valid_pitch = pitch_data[~np.isnan(pitch_data)]
    if len(valid_pitch) == 0:
        return (82, 659)  # Default range if no valid pitch
        
    valid_pitch_no_outliers = valid_pitch[
        (valid_pitch > np.percentile(valid_pitch, 1)) & 
        (valid_pitch < np.percentile(valid_pitch, 99))
    ]
    
    if len(valid_pitch_no_outliers) == 0:
        return (82, 659)  # Default range if filtering removed all data
        
    min_pitch = np.min(valid_pitch_no_outliers) * 0.8
    max_pitch = np.max(valid_pitch_no_outliers) * 1.2
    
    # Ensure min_pitch is at least 82 Hz (approximately E2)
    min_pitch = max(82, min_pitch)
    
    return (min_pitch, max_pitch)

# Core functions
@cached("feature_cache", "scale_freq")
def get_scale_frequencies(key, pitch_range=None, include_microtones=False):
    """Calculate all scale frequencies for a given key"""
    if pitch_range is None:
        min_freq, max_freq = 82, 659
    else:
        min_freq, max_freq = pitch_range
    
    min_freq = max(82, min_freq)
    
    is_minor = False
    if key.endswith('m'):
        is_minor = True
        key = key[:-1]
    
    if key not in KEY_TO_FREQUENCY:
        raise ValueError(f"Unknown key: {key}")
    
    root_freq = KEY_TO_FREQUENCY[key]
    scale_intervals = MINOR_SCALE if is_minor else MAJOR_SCALE
    
    octave_min = int(np.floor(np.log2(min_freq / root_freq) * 12) // 12) - 1
    octave_max = int(np.ceil(np.log2(max_freq / root_freq) * 12) // 12) + 2
    
    max_octaves = octave_max - octave_min
    octave_range = np.arange(octave_min, octave_min + max_octaves)
    
    all_semitones = np.array([interval + (octave * 12) for octave in octave_range for interval in scale_intervals])
    frequencies = root_freq * (2 ** (all_semitones / 12))
    
    valid_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
    frequencies = frequencies[valid_mask]
    
    return sorted(frequencies.tolist())


@cached("feature_cache", "pitch_on_scale")
def is_pitch_on_scale_vectorized(scale_frequencies, pitches, error_threshold=0.05):
    """Check if pitch sequence is on scale (vectorized)"""
    results = np.zeros(len(pitches), dtype=bool)
    scale_freqs = np.array(scale_frequencies)
    
    valid_indices = ~np.isnan(pitches) & (pitches > 0)
    valid_pitches = pitches[valid_indices]
    
    if len(valid_pitches) == 0:
        return results
    
    insertion_points = np.searchsorted(scale_freqs, valid_pitches)
    
    at_start = insertion_points == 0
    at_end = insertion_points == len(scale_freqs)
    in_middle = ~(at_start | at_end)
    
    relative_errors = np.ones_like(valid_pitches)
    
    if np.any(at_start):
        start_errors = np.abs(valid_pitches[at_start] - scale_freqs[0]) / valid_pitches[at_start]
        relative_errors[at_start] = start_errors
    
    if np.any(at_end):
        end_errors = np.abs(valid_pitches[at_end] - scale_freqs[-1]) / valid_pitches[at_end]
        relative_errors[at_end] = end_errors
    
    if np.any(in_middle):
        mid_indices = insertion_points[in_middle]
        before = scale_freqs[mid_indices - 1]
        after = scale_freqs[mid_indices]
        
        error_before = np.abs(valid_pitches[in_middle] - before) / valid_pitches[in_middle]
        error_after = np.abs(valid_pitches[in_middle] - after) / valid_pitches[in_middle]
        
        min_errors = np.minimum(error_before, error_after)
        relative_errors[in_middle] = min_errors
    
    within_threshold = relative_errors <= error_threshold
    results[valid_indices] = within_threshold
    
    return results


@cached("feature_cache", "key_similarity")
def key_based_similarity(live_pitch, key):
    """Calculate similarity between pitch data and a specific key"""
    if live_pitch is None or len(live_pitch) == 0:
        return 0.0
    
    # Get valid pitch data
    valid_indices = ~np.isnan(live_pitch)
    valid_pitch = live_pitch[valid_indices]
    
    if len(valid_pitch) < 10:
        return 0.0
    
    # Calculate pitch range
    pitch_range = calculate_pitch_range(live_pitch)
    
    # Get scale frequencies 
    scale_frequencies = get_scale_frequencies(key, pitch_range, include_microtones=False)
    
    # Divide the pitch range into three parts: low, mid, high
    min_pitch, max_pitch = pitch_range
    range_size = max_pitch - min_pitch
    low_mid_boundary = min_pitch + range_size / 3
    mid_high_boundary = min_pitch + 2 * range_size / 3
    
    # Separate low, mid, and high pitch regions
    low_pitch_mask = valid_pitch <= low_mid_boundary
    mid_pitch_mask = (valid_pitch > low_mid_boundary) & (valid_pitch <= mid_high_boundary)
    high_pitch_mask = valid_pitch > mid_high_boundary
    
    low_pitch = valid_pitch[low_pitch_mask]
    mid_pitch = valid_pitch[mid_pitch_mask]
    high_pitch = valid_pitch[high_pitch_mask]
    
    # Use all data, no sampling
    sampled_low_pitch = low_pitch
    sampled_mid_pitch = mid_pitch
    sampled_high_pitch = high_pitch
    
    # Define three pitch regions, with their tolerance (threshold), weight, and sampled data
    regions = {
        "low":  dict(threshold=0.016, weight=0.20, data=sampled_low_pitch),   # low-register notes
        "mid":  dict(threshold=0.013, weight=0.30, data=sampled_mid_pitch),  # middle-register notes
        "high": dict(threshold=0.014, weight=0.50, data=sampled_high_pitch),  # high-register notes
    }

    weighted_sum = 0.0  # accumulate weighted similarities
    total_weight = 0.0  # accumulate weights actually used (skip empty regions)

    for cfg in regions.values():
        pitches = cfg["data"]
        # if this register has no valid pitch samples, skip it
        if len(pitches) == 0:
            continue  
                             
        # -- count how many pitches in this region sit on the song’s scale
        matches = np.sum(
            is_pitch_on_scale_vectorized(scale_frequencies, pitches, cfg["threshold"])
        )
        
        # similarity = (matching notes) / (total notes) for this region
        similarity = matches / len(pitches)
        
        # accumulate weighted similarity
        weighted_sum += similarity * cfg["weight"]
        total_weight += cfg["weight"]
        
    # final weighted similarity over all regions that contained data
    final_similarity = weighted_sum / total_weight if total_weight else 0.0
    return float(final_similarity)

def generate_key_comparison_plot(times, pitch, key, chunk_idx=-1, similarity_value=None, save_plot=True):
    """Generate key analysis visualization"""

    pitch_range = calculate_pitch_range(pitch)
    scale_frequencies = get_scale_frequencies(key, pitch_range, include_microtones=False)
    output_image = f"key_comparison_{chunk_idx}.png" if save_plot else None
    
    # Generate plot
    return plot_key_comparison(
        times=times,
        pitch=pitch,
        key=key,
        scale_frequencies=scale_frequencies,
        pitch_range=pitch_range,
        similarity_value=similarity_value,
        output_image=output_image
    )


def analyze_key_based_pitch_chunked(live_path, cd_path, key, sr=44100, threshold=0.55, chunk_sec=15, shared_data=None):
    """
    Main function for key-based pitch analysis
    Analyzes chunks to find segments highly matched with a specific key
    """
    total_start_time = time.time()
    shared_data = shared_data or {}
    
    # Get audio data
    live_wave = shared_data.get('live_wave')
    cd_wave = shared_data.get('cd_wave')
    
    if live_wave is None or cd_wave is None:
        live_wave = cache_manager.get_shared_data('live_wave')
        cd_wave = cache_manager.get_shared_data('cd_wave')
    
    if live_wave is None or cd_wave is None:
        raise ValueError("Audio data not loaded. Call audio_processor.load_and_prepare_audio first.")
    
    # Get offset information
    base_offset = shared_data.get('offset')
    if base_offset is None:
        base_offset = cache_manager.get_shared_data('offset')
        if base_offset is None:
            # If not in cache, find offset
            live_first_chunk = extract_segment(
                wave_data=live_wave,
                sr=sr,
                start_sample=0,
                desired_length_sec=chunk_sec,
                min_sec=10
            )
            if live_first_chunk is not None:
                base_offset = find_best_offset(
                    live_first_chunk, 
                    cd_wave, 
                    use_chroma=True,
                    live_path=live_path,
                    cd_path=cd_path
                )
            else:
                base_offset = 0
    
    # Determine analysis order
    chunk_size = int(chunk_sec * sr)
    num_chunks = len(live_wave) // chunk_size
    
    detected_chunk = shared_data.get('detected_chunk')
    if detected_chunk is None:
        detected_chunk = cache_manager.get_shared_data('detected_chunk', -1)
    
    # Prioritize chunk detected by basic algorithm
    if detected_chunk >= 0:
        chunks_to_analyze = [detected_chunk] + [i for i in range(num_chunks + 1) if i != detected_chunk]
    else:
        chunks_to_analyze = list(range(num_chunks + 1))
    
    # Analyze each chunk
    for i in chunks_to_analyze:
        # Extract live recording chunk
        start_sample = i * chunk_size
        live_chunk = extract_segment(
            wave_data=live_wave,
            sr=sr,
            start_sample=start_sample,
            desired_length_sec=chunk_sec,
            min_sec=10
        )
        
        if live_chunk is None:
            continue
        
        # Check if this is the baseline detection chunk
        if i == detected_chunk:
            # Try to get pitch data from cache
            times_live = None
            f0_live = None
            
            if 'times_live' in shared_data and 'f0_live' in shared_data:
                times_live = shared_data['times_live']
                f0_live = shared_data['f0_live']
            
            if times_live is None or f0_live is None:
                times_live = cache_manager.get_shared_data('times_live')
                f0_live = cache_manager.get_shared_data('f0_live')
            
            # If we have pitch data, calculate similarity directly
            if times_live is not None and f0_live is not None:
                key_sim = key_based_similarity(f0_live, key)
                is_fake = key_sim >= threshold
                
                if is_fake:
                    explanation = f"Key similarity: {key_sim:.3f} ≥ {threshold} (Lip-syncing)"
                    plot_path = generate_key_comparison_plot(times_live, f0_live, key, chunk_idx=i, similarity_value=key_sim)
                    
                    # Update cache
                    result_data = {
                        'detected_chunk_key': i,
                        'detected_sim_key': key_sim
                    }
                    
                    if plot_path:
                        result_data['key_plot_path'] = plot_path
                    
                    if shared_data is not None:
                        shared_data.update(result_data)
                    
                    cache_manager.update_shared_data(result_data)
                    
                    total_time = time.time() - total_start_time
                    return is_fake, i, key_sim, explanation, total_time, plot_path
                
                continue
        
        # Calculate CD recording offset for this chunk
        offset = base_offset + i * chunk_size if base_offset is not None else i * chunk_size
        
        # Extract CD recording chunk
        live_chunk_len = len(live_chunk)
        cd_chunk = extract_segment(
            wave_data=cd_wave,
            sr=sr,
            start_sample=offset,
            desired_length=live_chunk_len,
            min_sec=10
        )
        
        if cd_chunk is None:
            continue
        
        # Extract vocals
        from extract_vocal import extract_vocals as extractor_func
        
        live_vocals = cache_manager.extract_vocals(
            live_chunk, 
            sr=sr, 
            extractor_func=extractor_func
        )
        
        # Ensure vocals are mono
        if live_vocals.ndim > 1:
            live_vocals = np.mean(live_vocals, axis=1)
        
        # Extract pitch data
        times_live, f0_live = cache_manager.get_pitch_data(live_vocals, sr=sr)
        
        # Calculate key similarity
        key_sim = key_based_similarity(f0_live, key)
        is_fake = key_sim >= threshold
        
        if is_fake:
            explanation = f"Key similarity: {key_sim:.3f} ≥ {threshold} (Lip-syncing)"
            plot_path = generate_key_comparison_plot(times_live, f0_live, key, chunk_idx=i, similarity_value=key_sim)
            
            # Update cache
            result_data = {
                'detected_chunk_key': i,
                'detected_sim_key': key_sim
            }
            
            if plot_path:
                result_data['key_plot_path'] = plot_path
            
            if shared_data is not None:
                shared_data.update(result_data)
            
            cache_manager.update_shared_data(result_data)
            
            total_time = time.time() - total_start_time
            return is_fake, i, key_sim, explanation, total_time, plot_path
    
    total_time = time.time() - total_start_time
    explanation = f"Key similarity: below {threshold} (Real singing)"
    return False, -1, 0.0, explanation, total_time, None