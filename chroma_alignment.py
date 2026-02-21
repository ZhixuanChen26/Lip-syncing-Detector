#chroma_alignment.py - Find the best offset

import numpy as np

try:
    from cache_manager import cache_manager
    use_global_cache = True
except ImportError:
    use_global_cache = False

def extract_chroma_fingerprint(audio, sr=44100, hop_length=512):
    """Extract chromagram features from audio"""
    import librosa
        
    # Extract chroma features
    chroma = librosa.feature.chroma_cqt(
        y=audio, 
        sr=sr, 
        hop_length=hop_length,
        n_chroma=12,
        bins_per_octave=12  
    )
        
    # Normalize
    chroma_norm = librosa.util.normalize(chroma, norm=2, axis=0)
    
    return chroma_norm

def _save_offset_to_cache(live_path, cd_path, offset, confidence):
    """Save offset to cache"""
    if use_global_cache and live_path and cd_path:
        cache_manager.offset_cache.set_offset(
            live_path, cd_path, 
            offset, confidence, 
            method='chroma'
        )

def find_offset_with_chroma(live_chunk, cd_wave, sr=44100, hop_length=512, live_path=None, cd_path=None):
    """
    Find best offset using chroma fingerprints
    Returns offset in samples and confidence score
    """
    # Check for cached offset
    if use_global_cache and live_path and cd_path:
        cached_offset = cache_manager.offset_cache.get_offset(live_path, cd_path, method='chroma')
        if cached_offset is not None:
            return {
                'offset': cached_offset['offset'],
                'confidence': cached_offset['confidence'],
                'similarity': cached_offset.get('similarity', 0.0),
                'processing_time': 0.0
            }
    
    import time
    start_time = time.time()
    
    # Ensure audio is mono
    if len(live_chunk.shape) > 1:
        live_chunk = np.mean(live_chunk, axis=1)
    if len(cd_wave.shape) > 1:
        cd_wave = np.mean(cd_wave, axis=1)
    
    # Extract live chunk chroma features
    live_chroma = extract_chroma_fingerprint(live_chunk, sr=sr, hop_length=hop_length)
    
    # Compute window parameters
    cd_duration = len(cd_wave) / sr
    window_duration = min(60, cd_duration)
    window_samples = int(window_duration * sr)
    step_samples = int(20 * sr)  
    
    best_offset = 0
    best_confidence = 0
    best_similarity = -1
    
    if cd_duration > window_duration:
        # Pre-extract full CD chroma features
        full_cd_chroma = extract_chroma_fingerprint(cd_wave, sr=sr, hop_length=hop_length)
        step_frames = int(step_samples / hop_length)
        
        # Iterate through CD frames
        for start_frame in range(0, full_cd_chroma.shape[1] - int(window_samples / hop_length) + 1, step_frames):
            # Slice pre-computed chroma matrix
            window_chroma = full_cd_chroma[:, start_frame:start_frame + int(window_samples / hop_length)]
            
            # Find best match
            offset, confidence, similarity = _find_best_match(
                live_chroma, window_chroma
            )
            
            # Convert to global offset
            global_offset = int(start_frame * hop_length) + offset * hop_length
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_offset = global_offset
                best_confidence = confidence
                
                # Exit early if good match found
                if best_confidence > 0.8:
                    break
    else:
        # Extract chroma for entire CD
        cd_chroma = extract_chroma_fingerprint(cd_wave, sr=sr, hop_length=hop_length)
        
        # Find best match
        offset, confidence, similarity = _find_best_match(
            live_chroma, cd_chroma
        )
        
        best_offset = offset * hop_length
        best_confidence = confidence
        best_similarity = similarity
    
    processing_time = time.time() - start_time
    
    result = {
        'offset': best_offset,
        'confidence': best_confidence,
        'similarity': best_similarity,
        'processing_time': processing_time
    }
    
    _save_offset_to_cache(live_path, cd_path, best_offset, best_confidence)
    
    return result

def _find_best_match(live_chroma, cd_chroma):
    """Find best match position between two chroma sequences"""
    # Ensure live_chroma is not longer than cd_chroma
    if live_chroma.shape[1] > cd_chroma.shape[1]:
        return 0, 0.0, 0.0
    
    # Create similarity array
    S = np.zeros(cd_chroma.shape[1] - live_chroma.shape[1] + 1)
    
    # Compute similarity for each possible offset
    for i in range(len(S)):
        cd_slice = cd_chroma[:, i:i+live_chroma.shape[1]]
        # Calculate cosine similarity
        frame_sim = np.sum(live_chroma * cd_slice) / (
            np.sqrt(np.sum(live_chroma**2)) * np.sqrt(np.sum(cd_slice**2)) + 1e-8
        )
        S[i] = frame_sim
    
    # Find best match position
    best_offset = np.argmax(S)
    similarity = S[best_offset]
    
    # Calculate confidence based on local consistency
    window = 10 
    start = max(0, best_offset - window)
    end = min(len(S), best_offset + window + 1)
    
    # Count local peaks
    local_peaks = 0
    for i in range(start+1, end-1):
        if S[i] > S[i-1] and S[i] > S[i+1]:
            local_peaks += 1
    
    # Many local peaks indicate unstable match
    peak_penalty = min(1.0, local_peaks / 5)
    
    # Calculate standard deviation - lower means more stable match
    std_dev = np.std(S[start:end])
    std_factor = 1.0 - min(1.0, std_dev * 5)
    
    # Combine similarity and stability for confidence score
    confidence = similarity * (1.0 - peak_penalty) * std_factor
    
    return best_offset, float(confidence), float(similarity)