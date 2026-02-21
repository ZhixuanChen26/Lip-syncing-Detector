"""
compare.py - Audio comparison for lip-syncing detection
"""

import numpy as np
from scipy.signal import correlate
import time

from cache_manager import cache_manager, cached
from audio_processor import audio_processor as _ap

@cached('feature_cache', 'extract_segment')
def extract_segment(wave_data, sr, start_sample, desired_length=None, desired_length_sec=None, min_sec=10):
    """Extract segment from audio wave"""
    total_len = len(wave_data)
    if desired_length is not None:
        target_len_samples = desired_length
    elif desired_length_sec is not None:
        target_len_samples = int(desired_length_sec * sr)
    else:
        raise ValueError("Either desired_length or desired_length_sec must be specified.")

    start = start_sample
    end = start + target_len_samples
    
    if end > total_len:
        remain = total_len - start
        remain_sec = remain / sr
        if remain_sec >= min_sec:
            return wave_data[start:]
        else:
            return None
    else:
        return wave_data[start:end]

@cached('feature_cache', 'find_offset')
def find_best_offset(live_chunk_wave, cd_wave, use_chroma=True, live_path=None, cd_path=None):
    """Find best offset between live and CD audio"""
    if use_chroma:
        try:
            import chroma_alignment
            result = chroma_alignment.find_offset_with_chroma(
                live_chunk_wave, 
                cd_wave, 
                sr=44100,
                live_path=live_path,
                cd_path=cd_path
            )
            return result['offset']
        except Exception:
            pass
    
    # Fallback to cross-correlation
    corr = correlate(cd_wave, live_chunk_wave, mode='valid')
    offset = np.argmax(corr)
    return offset

@cached('feature_cache', 'pitch_similarity')
def pitch_cosine_similarity(live_array, cd_array, sr=44100):
    """Calculate pitch cosine similarity between vocals"""
    # Check global cache first
    times_live = cache_manager.get_shared_data('times_live')
    f0_live = cache_manager.get_shared_data('f0_live')
    times_cd = cache_manager.get_shared_data('times_cd')
    f0_cd = cache_manager.get_shared_data('f0_cd')
    
    # If not in global cache, compute them
    if times_live is None or f0_live is None:
        times_live, f0_live_plot = cache_manager.get_pitch_data(live_array, sr=sr)
    else:
        f0_live_plot = f0_live
    
    if times_cd is None or f0_cd is None:
        times_cd, f0_cd_plot = cache_manager.get_pitch_data(cd_array, sr=sr)
    else:
        f0_cd_plot = f0_cd
    
    # Update global cache
    cache_manager.update_shared_data({
        'times_live': times_live,
        'f0_live': f0_live_plot,
        'times_cd': times_cd,
        'f0_cd': f0_cd_plot
    })
    
    # Align pitch array lengths
    min_len = min(len(f0_live_plot), len(f0_cd_plot))
    f0_live_plot = f0_live_plot[:min_len]
    f0_cd_plot = f0_cd_plot[:min_len]
    times_live = times_live[:min_len]
    times_cd = times_cd[:min_len]
    
    # Create NaN masks
    live_nan_mask = np.isnan(f0_live_plot)
    cd_nan_mask = np.isnan(f0_cd_plot)
    
    # Calculate NaN pattern similarity
    nan_pattern_match = np.mean(live_nan_mask == cd_nan_mask) if min_len > 0 else 0
    
    # Find regions where both signals have valid values
    both_valid_mask = (~live_nan_mask) & (~cd_nan_mask)
    valid_indices = np.where(both_valid_mask)[0]
    
    # If too few valid points, cannot calculate similarity
    if len(valid_indices) < 10:
        return 0.0, times_live, f0_live_plot, times_cd, f0_cd_plot
    
    # Extract valid pitch values
    valid_live = f0_live_plot[valid_indices]
    valid_cd = f0_cd_plot[valid_indices]
    
    # 1. Calculate pitch cosine similarity
    dot = np.sum(valid_live * valid_cd)
    norm_live = np.sqrt(np.sum(valid_live**2))
    norm_cd = np.sqrt(np.sum(valid_cd**2))
    
    if norm_live < 1e-10 or norm_cd < 1e-10:
        pitch_sim = 0.0
    else:
        pitch_sim = dot / (norm_live * norm_cd)
    
    # 2. Calculate pitch variation rate similarity
    if len(valid_indices) >= 2:
        # Find consecutive valid point index pairs
        consecutive_pairs = []
        for i in range(len(valid_indices)-1):
            if valid_indices[i+1] == valid_indices[i]+1:
                consecutive_pairs.append(i)
        
        if len(consecutive_pairs) >= 5:
            # Extract pitch values for these consecutive pairs
            consecutive_live = valid_live[consecutive_pairs]
            consecutive_live_next = valid_live[[p+1 for p in consecutive_pairs]]
            consecutive_cd = valid_cd[consecutive_pairs]
            consecutive_cd_next = valid_cd[[p+1 for p in consecutive_pairs]]
            
            # Calculate pitch variation rates
            live_diff = consecutive_live_next - consecutive_live
            cd_diff = consecutive_cd_next - consecutive_cd
            
            # Calculate cosine similarity of variation rates
            diff_dot = np.sum(live_diff * cd_diff)
            diff_norm_live = np.sqrt(np.sum(live_diff**2))
            diff_norm_cd = np.sqrt(np.sum(cd_diff**2))
            
            if diff_norm_live < 1e-10 or diff_norm_cd < 1e-10:
                diff_sim = 0.0
            else:
                diff_sim = diff_dot / (diff_norm_live * diff_norm_cd)
        else:
            diff_sim = 0.0
    else:
        diff_sim = 0.0
    
    # 3. Combine metrics
    combined_sim = 0.6 * pitch_sim + 0.3 * diff_sim + 0.1 * nan_pattern_match
    
    return combined_sim, times_live, f0_live_plot, times_cd, f0_cd_plot

def compare(live_path, cd_path, sr=44100, chunk_sec=15, threshold=0.73, shared_data=None, use_chroma=True):
    """
    Main function to compare audio files and detect lip-syncing
    Returns: (is_fake, detected_chunk_index, similarity_score)
    """
    
    # Get audio data from cache
    global_live_wave = cache_manager.get_shared_data('live_wave')
    global_cd_wave = cache_manager.get_shared_data('cd_wave')
    
    # Check if shared_data has audio data
    if shared_data and 'live_wave' in shared_data and 'cd_wave' in shared_data:
        live_wave = shared_data['live_wave']
        cd_wave = shared_data['cd_wave']
    elif global_live_wave is not None and global_cd_wave is not None:
        live_wave = global_live_wave
        cd_wave = global_cd_wave
        
        # Update shared_data if provided
        if shared_data is not None:
            shared_data['live_wave'] = live_wave
            shared_data['cd_wave'] = cd_wave
            shared_data['sr'] = cache_manager.get_shared_data('sr', sr)
    else:
        # Load audio
        live_wave, _ = cache_manager.load_audio(live_path, sr=sr)
        cd_wave, _ = cache_manager.load_audio(cd_path, sr=sr)
        
        # Save to shared_data
        if shared_data is not None:
            shared_data['live_wave'] = live_wave
            shared_data['cd_wave'] = cd_wave
            shared_data['sr'] = sr
        
        # Save to global cache
        cache_manager.update_shared_data({
            'live_wave': live_wave,
            'cd_wave': cd_wave,
            'sr': sr
        })

    # Chunk size in samples
    chunk_size = int(chunk_sec * sr)
    num_chunks = len(live_wave) // chunk_size
    base_offset = None

    # Check for cached offset
    if use_chroma:
        offset_data = _ap._get_cached_offset(live_path, cd_path)
        
        if offset_data is not None:
            base_offset = offset_data['offset']
            
            # Update shared_data
            if shared_data is not None:
                shared_data['offset'] = base_offset
                shared_data['offset_confidence'] = offset_data['confidence']
                shared_data['offset_method'] = offset_data['method']

    if use_chroma and base_offset is None:
        try:
            import chroma_alignment
            
            # Use first chunk to find initial offset
            live_first_chunk = extract_segment(
                wave_data=live_wave,
                sr=sr,
                start_sample=0,
                desired_length_sec=chunk_sec,
                min_sec=10
            )
            
            if live_first_chunk is not None:
                alignment_result = chroma_alignment.find_offset_with_chroma(
                    live_first_chunk,
                    cd_wave,
                    sr=sr,
                    live_path=live_path,
                    cd_path=cd_path
                )
                
                base_offset = alignment_result['offset']
                
                # Store offset info
                if shared_data is not None:
                    shared_data['offset'] = base_offset
                    shared_data['offset_confidence'] = alignment_result['confidence']
                    shared_data['offset_method'] = 'chroma'
                
                # Store in global cache
                cache_manager.update_shared_data({
                    'offset': base_offset,
                    'offset_confidence': alignment_result['confidence'],
                    'offset_method': 'chroma'
                })
                    
        except Exception:
            use_chroma = False
    
    # Process each chunk
    for i in range(num_chunks + 1):
        # Define starting sample for this chunk
        start_sample = i * chunk_size
        
        # Extract segment from live recording
        live_chunk = extract_segment(
            wave_data=live_wave,
            sr=sr,
            start_sample=start_sample,
            desired_length_sec=chunk_sec,
            min_sec=10
        )
        
        if live_chunk is None:
            break  # Skip if segment is too short

        # Find best offset for first chunk
        if i == 0 and base_offset is None:
            if use_chroma:
                try:
                    import chroma_alignment
                    alignment_result = chroma_alignment.find_offset_with_chroma(
                        live_chunk, cd_wave, sr=sr,
                        live_path=live_path, cd_path=cd_path
                    )
                    offset = alignment_result['offset']
                    
                    if shared_data is not None:
                        shared_data['offset_method'] = 'chroma'
                        shared_data['offset_confidence'] = alignment_result['confidence']
                    
                    cache_manager.update_shared_data({
                        'offset': offset,
                        'offset_method': 'chroma',
                        'offset_confidence': alignment_result['confidence']
                    })
                except Exception:
                    # Fall back to cross-correlation
                    offset = find_best_offset(live_chunk, cd_wave, use_chroma=False)
                    
                    if shared_data is not None:
                        shared_data['offset_method'] = "xcorr_fallback"
                    
                    cache_manager.update_shared_data({
                        'offset': offset,
                        'offset_method': 'xcorr_fallback'
                    })
            else:
                offset = find_best_offset(live_chunk, cd_wave, use_chroma=False)
                
                if shared_data is not None:
                    shared_data['offset_method'] = "xcorr"
                
                cache_manager.update_shared_data({
                    'offset': offset,
                    'offset_method': 'xcorr'
                })
            
            base_offset = offset
            
            # Store offset
            if shared_data is not None:
                shared_data['offset'] = base_offset
            
            cache_manager.update_shared_data({
                'offset': base_offset
            })
        else:
            if base_offset is not None:
                # Use base_offset plus chunk offset
                offset = base_offset + i * chunk_size
            else:
                # Should not happen unless first chunk was skipped
                if use_chroma:
                    try:
                        import chroma_alignment
                        alignment_result = chroma_alignment.find_offset_with_chroma(
                            live_chunk, cd_wave, sr=sr,
                            live_path=live_path, cd_path=cd_path
                        )
                        offset = alignment_result['offset']
                    except Exception:
                        offset = find_best_offset(live_chunk, cd_wave, use_chroma=False)
                else:
                    offset = find_best_offset(live_chunk, cd_wave, use_chroma=False)

        # Extract corresponding segment from CD recording
        live_chunk_len = len(live_chunk)
        cd_chunk = extract_segment(
            wave_data=cd_wave,
            sr=sr,
            start_sample=offset,
            desired_length=live_chunk_len,
            min_sec=10
        )
        
        if cd_chunk is None:
            continue  # Skip if no valid CD segment
        
        # Extract vocals
        live_vocals = None
        cd_vocals = None
        
        if i == 0:  # Only check global cache for first chunk
            live_vocals = cache_manager.get_shared_data('live_vocals')
            cd_vocals = cache_manager.get_shared_data('cd_vocals')
        
        # Extract vocals if needed
        if live_vocals is None:
            from extract_vocal import extract_vocals as extractor_func
            
            live_vocals = cache_manager.extract_vocals(
                live_chunk, 
                sr=sr, 
                extractor_func=extractor_func
            )
        
        if cd_vocals is None:
            from extract_vocal import extract_vocals as extractor_func
            
            cd_vocals = cache_manager.extract_vocals(
                cd_chunk, 
                sr=sr, 
                extractor_func=extractor_func
            )
        
        # Update global cache for first chunk
        if i == 0:
            cache_manager.update_shared_data({
                'live_vocals': live_vocals,
                'cd_vocals': cd_vocals
            })
            
            if shared_data is not None:
                shared_data['live_vocals'] = live_vocals
                shared_data['cd_vocals'] = cd_vocals
        
        # Ensure vocals are mono
        if live_vocals.ndim > 1:
            live_vocals = np.mean(live_vocals, axis=1)
        if cd_vocals.ndim > 1:
            cd_vocals = np.mean(cd_vocals, axis=1)

        # Compute pitch similarity
        times_live = None
        f0_live = None
        times_cd = None
        f0_cd = None
        
        if i == 0:  # Only use cache for first chunk
            times_live = cache_manager.get_shared_data('times_live')
            f0_live = cache_manager.get_shared_data('f0_live')
            times_cd = cache_manager.get_shared_data('times_cd')
            f0_cd = cache_manager.get_shared_data('f0_cd')
            
            if times_live is not None and f0_live is not None and times_cd is not None and f0_cd is not None:
                sim = pitch_cosine_similarity(live_vocals, cd_vocals, sr=sr)[0]
            else:
                sim, times_live, f0_live, times_cd, f0_cd = pitch_cosine_similarity(
                    live_vocals, cd_vocals, sr=sr
                )
        else:
            sim, times_live, f0_live, times_cd, f0_cd = pitch_cosine_similarity(
                live_vocals, cd_vocals, sr=sr
            )

        # Save pitch data
        if shared_data is not None:
            shared_data['times_live'] = times_live
            shared_data['f0_live'] = f0_live
            shared_data['times_cd'] = times_cd
            shared_data['f0_cd'] = f0_cd
        
        # Update global cache for first chunk
        if i == 0:
            cache_manager.update_shared_data({
                'times_live': times_live,
                'f0_live': f0_live,
                'times_cd': times_cd,
                'f0_cd': f0_cd
            })

        # Check if similarity exceeds threshold
        if sim >= threshold:
            from get_spectrum import plot_compare_pitch_contours
            
            decision = "Fake"
            
            plot_compare_pitch_contours(
                times_live, f0_live,
                times_cd, f0_cd,
                output_image=f"fake_chunk_{i}.png",
                sim=sim,
                decision=decision
            )
            
            # Store detection data
            if shared_data is not None:
                shared_data['detected_chunk'] = i
                shared_data['detected_sim'] = sim
            
            cache_manager.update_shared_data({
                'detected_chunk': i,
                'detected_sim': sim
            })
            
            return True, i, sim

    return False, -1, 0.0