"""
audio_processor.py - Audio processing and offset caching
"""

import numpy as np
import time
from cache_manager import cache_manager

class AudioProcessor:
    def __init__(self):
        self.shared_data = {}
    
    def _get_cached_offset(self, live_path=None, cd_path=None):
        """Get offset from cache sources in priority order"""
        # Check local shared data first
        if 'offset' in self.shared_data:
            return {
                'offset': self.shared_data['offset'],
                'confidence': self.shared_data.get('offset_confidence', 0.0),
                'method': self.shared_data.get('offset_method', 'cached')
            }
        
        # Check global cache
        global_offset = cache_manager.get_shared_data('offset')
        if global_offset is not None:
            offset_data = {
                'offset': global_offset,
                'confidence': cache_manager.get_shared_data('offset_confidence', 0.0),
                'method': cache_manager.get_shared_data('offset_method', 'cached')
            }
            # Update local cache
            self.shared_data.update({
                'offset': global_offset,
                'offset_confidence': cache_manager.get_shared_data('offset_confidence', 0.0),
                'offset_method': cache_manager.get_shared_data('offset_method', 'cached')
            })
            return offset_data
        
        # Check offset cache if paths provided
        if live_path and cd_path:
            cached_offset = cache_manager.offset_cache.get_offset(
                live_path, cd_path, method='chroma'
            )
            if cached_offset is not None:
                offset_data = {
                    'offset': cached_offset['offset'],
                    'confidence': cached_offset['confidence'],
                    'method': 'chroma_cached'
                }
                # Update both local and global cache
                self.shared_data.update({
                    'offset': cached_offset['offset'],
                    'offset_confidence': cached_offset['confidence'],
                    'offset_method': 'chroma_cached'
                })
                cache_manager.update_shared_data({
                    'offset': cached_offset['offset'],
                    'offset_confidence': cached_offset['confidence'],
                    'offset_method': 'chroma_cached'
                })
                return offset_data
        
        # No cached offset found
        return None
    
    def _sync_cache(self, from_global=False, to_global=False):
        """Sync data between local and global cache"""
        important_keys = [
            'offset', 'offset_confidence', 'offset_method',
            'detected_chunk', 'detected_sim',
            'times_live', 'f0_live', 'times_cd', 'f0_cd',
            'live_vocals', 'cd_vocals',
            'live_wave', 'cd_wave', 'sr'
        ]
        
        if from_global:
            # Sync from global to local
            for key in important_keys:
                value = cache_manager.get_shared_data(key)
                if value is not None:
                    self.shared_data[key] = value
        
        if to_global:
            # Sync from local to global
            update_dict = {}
            for key in important_keys:
                if key in self.shared_data:
                    update_dict[key] = self.shared_data[key]
            
            if update_dict:
                cache_manager.update_shared_data(update_dict)
    
    def load_and_prepare_audio(self, live_path, cd_path, sr=44100):
        """Load audio files and prepare for analysis"""
        live_wave, _ = cache_manager.load_audio(live_path, sr)
        cd_wave, _ = cache_manager.load_audio(cd_path, sr)
        
        self.shared_data['live_wave'] = live_wave
        self.shared_data['cd_wave'] = cd_wave
        self.shared_data['sr'] = sr
        self.shared_data['live_path'] = live_path
        self.shared_data['cd_path'] = cd_path
        
        # Store data in the global cache
        cache_manager.update_shared_data({
            'live_wave': live_wave,
            'cd_wave': cd_wave,
            'sr': sr,
            'live_path': live_path,
            'cd_path': cd_path
        })
        
        # Try to load cached offset
        cached_offset = self._get_cached_offset(live_path, cd_path)
        if cached_offset is not None:
            self.shared_data['offset'] = cached_offset['offset']
            self.shared_data['offset_confidence'] = cached_offset['confidence']
            self.shared_data['offset_method'] = cached_offset['method']
            
            cache_manager.update_shared_data({
                'offset': cached_offset['offset'],
                'offset_confidence': cached_offset['confidence'],
                'offset_method': cached_offset['method']
            })
    
    def align_audio_segments(self, live_wave, cd_wave, chunk_sec=15, sr=44100, use_chroma=True):
        """Align audio segments using chroma or cross-correlation"""
        # Check for cached offset first
        offset_data = self._get_cached_offset(
            self.shared_data.get('live_path'),
            self.shared_data.get('cd_path')
        )
        
        if offset_data is not None:
            return offset_data
        
        from compare import find_best_offset
        
        # Extract first chunk of live audio
        chunk_size = int(chunk_sec * sr)
        live_chunk = live_wave[:min(chunk_size, len(live_wave))]
        
        offset = find_best_offset(
            live_chunk, 
            cd_wave, 
            use_chroma=use_chroma,
            live_path=self.shared_data.get('live_path'),
            cd_path=self.shared_data.get('cd_path')
        )
        
        correlation = 0.0
        
        # Update both local and global cache
        self.shared_data.update({
            'offset': offset,
            'offset_confidence': float(correlation),
            'offset_method': 'chroma' if use_chroma else 'cross-correlation'
        })
        cache_manager.update_shared_data({
            'offset': offset,
            'offset_confidence': float(correlation),
            'offset_method': 'chroma' if use_chroma else 'cross-correlation'
        })
        
        return {
            'offset': offset,
            'correlation': float(correlation),
            'confidence': float(correlation),
            'method': 'chroma' if use_chroma else 'cross-correlation'
        }
    
    def run_basic_algorithm(self, live_path, cd_path, sr=44100, threshold=0.71, use_chroma=True):
        """Run basic lip-sync detection using pitch contour similarity"""
        start_time = time.time()
        
        from compare import compare
        from cache_manager import chunk_data_cache
    
        is_fake_basic, chunk_idx, sim_val = compare(
            live_path=live_path,
            cd_path=cd_path,
            sr=sr,
            chunk_sec=15,
            threshold=threshold,
            shared_data=self.shared_data,
            use_chroma=use_chroma
        )
        
        processing_time = time.time() - start_time
        
        # Get alignment method information
        alignment_method = self.shared_data.get('offset_method', 'cross-correlation')
        if 'offset_confidence' in self.shared_data:
            alignment_info = f" (confidence: {self.shared_data['offset_confidence']:.2f})"
        else:
            alignment_info = ""
        
        # Create explanation
        explanation = f"Alignment: {alignment_method}{alignment_info}. Similarity: {sim_val:.3f}" + (
            " ≥ 0.71 (Lip-syncing)" if is_fake_basic else " < 0.71 (Real singing)"
        )
        
        result = {
            "name": "Basic Algorithm",
            "is_fake": is_fake_basic,
            "score": sim_val,
            "processing_time": processing_time,
            "explanation": explanation,
            "plot_path": f"{'fake' if is_fake_basic else 'true'}_chunk_{chunk_idx}.png" if chunk_idx >= 0 else None
        }
        
        # Sync cache
        self._sync_cache(to_global=True)
        
        # Cache the first chunk data
        if 'sr' in self.shared_data and 'live_path' in self.shared_data and 'cd_path' in self.shared_data:
            sr_value = self.shared_data['sr']
            chunk_size = int(15 * sr_value)
            
            # Cache vocals if available
            if 'live_vocals' in self.shared_data:
                chunk_data_cache.cache_chunk_vocals(
                    self.shared_data['live_path'], 
                    0, 
                    chunk_size,
                    self.shared_data['live_vocals'], 
                    sr_value
                )
            
            if 'cd_vocals' in self.shared_data:
                offset = self.shared_data.get('offset', 0)
                chunk_data_cache.cache_chunk_vocals(
                    self.shared_data['cd_path'], 
                    offset, 
                    chunk_size,
                    self.shared_data['cd_vocals'], 
                    sr_value
                )
            
            # Cache pitch data if available
            if 'times_live' in self.shared_data and 'f0_live' in self.shared_data:
                chunk_data_cache.cache_chunk_pitch(
                    self.shared_data['live_path'], 
                    0, 
                    chunk_size,
                    self.shared_data['times_live'], 
                    self.shared_data['f0_live'], 
                    sr_value
                )
            
            if 'times_cd' in self.shared_data and 'f0_cd' in self.shared_data:
                offset = self.shared_data.get('offset', 0)
                chunk_data_cache.cache_chunk_pitch(
                    self.shared_data['cd_path'], 
                    offset, 
                    chunk_size,
                    self.shared_data['times_cd'], 
                    self.shared_data['f0_cd'], 
                    sr_value
                )
        
        # Store result data in global cache
        cache_manager.update_shared_data({
            'basic_result': result,
            'detected_chunk': self.shared_data.get('detected_chunk', -1),
            'times_live': self.shared_data.get('times_live'),
            'f0_live': self.shared_data.get('f0_live'),
            'times_cd': self.shared_data.get('times_cd'),
            'f0_cd': self.shared_data.get('f0_cd'),
            'live_vocals': self.shared_data.get('live_vocals'),
            'cd_vocals': self.shared_data.get('cd_vocals')
        })
        
        return result

    def run_key_based_algorithm(self, key, threshold=0.86):
        """Run key-based pitch analysis for lip-sync detection"""
        # Get audio paths
        if 'live_path' not in self.shared_data or 'cd_path' not in self.shared_data:
            live_path = cache_manager.get_shared_data('live_path')
            cd_path = cache_manager.get_shared_data('cd_path')
            
            if not live_path or not cd_path:
                raise ValueError("Audio paths not set. Call load_and_prepare_audio first.")
                
            self.shared_data.update({
                'live_path': live_path,
                'cd_path': cd_path
            })
        else:
            live_path = self.shared_data['live_path']
            cd_path = self.shared_data['cd_path']
        
        # Sync from global cache
        self._sync_cache(from_global=True)
        
        import os
        try:
            from key_pitch_analysis import analyze_key_based_pitch_chunked
        except ImportError:
            from key_pitch_analysis import analyze_key_based_pitch_chunked
        
        is_fake_key, chunk_idx, key_sim, key_explanation, processing_time, plot_path = analyze_key_based_pitch_chunked(
            live_path=live_path,
            cd_path=cd_path,
            key=key,
            sr=self.shared_data.get('sr', 44100),
            threshold=threshold,
            chunk_sec=15,
            shared_data=self.shared_data
        )
        
        result = {
            "name": "Key-based Analysis",
            "is_fake": is_fake_key,
            "score": key_sim,
            "processing_time": processing_time,
            "explanation": key_explanation,
            "plot_path": plot_path if plot_path and os.path.exists(plot_path) else (
                        f"key_comparison_{chunk_idx}.png" if chunk_idx >= 0 else None)
        }
        
        # Update global cache
        self._sync_cache(to_global=True)
        
        cache_manager.update_shared_data({
            'key_result': result,
            'detected_chunk_key': self.shared_data.get('detected_chunk_key', -1),
            'detected_sim_key': self.shared_data.get('detected_sim_key', 0.0)
        })
        
        return result
    
    def run_variance_algorithm(self, confidence=0.9, window_size=5):
        """Run pitch variance analysis to detect unnatural consistency"""
        # Get audio paths
        if 'live_path' not in self.shared_data or 'cd_path' not in self.shared_data:
            live_path = cache_manager.get_shared_data('live_path')
            cd_path = cache_manager.get_shared_data('cd_path')
            
            if not live_path or not cd_path:
                raise ValueError("Audio paths not set. Call load_and_prepare_audio first.")
                
            self.shared_data.update({
                'live_path': live_path,
                'cd_path': cd_path
            })
        else:
            live_path = self.shared_data['live_path']
            cd_path = self.shared_data['cd_path']
        
        # Sync from global cache
        self._sync_cache(from_global=True)
        
        try:
            from pitch_variance_analysis import analyze_pitch_variance_chunked
        except ImportError:
            from pitch_variance_analysis import analyze_pitch_variance_chunked
        
        is_fake_var, chunk_idx, consistency_score, var_explanation, processing_time = analyze_pitch_variance_chunked(
            live_path=live_path,
            cd_path=cd_path,
            sr=self.shared_data.get('sr', 44100),
            confidence=confidence,
            window_size=window_size,
            chunk_sec=15,
            shared_data=self.shared_data
        )
        
        result = {
            "name": "Variance Analysis",
            "is_fake": is_fake_var,
            "score": consistency_score,
            "processing_time": processing_time,
            "explanation": var_explanation,
        }
        
        # Update global cache
        self._sync_cache(to_global=True)
        
        cache_manager.update_shared_data({
            'variance_result': result,
            'detected_chunk_var': self.shared_data.get('detected_chunk_var', -1),
            'detected_ratio_var': self.shared_data.get('detected_ratio_var', 0.0)
        })
        
        return result
    
    def extract_vocals(self, save_output=False):
        """Extract vocals from audio files"""
        if 'live_wave' not in self.shared_data or 'cd_wave' not in self.shared_data:
            live_wave = cache_manager.get_shared_data('live_wave')
            cd_wave = cache_manager.get_shared_data('cd_wave')
            
            if live_wave is None or cd_wave is None:
                raise ValueError("Audio data not loaded. Call load_and_prepare_audio first.")
                
            self.shared_data.update({
                'live_wave': live_wave,
                'cd_wave': cd_wave
            })
        else:
            live_wave = self.shared_data['live_wave']
            cd_wave = self.shared_data['cd_wave']
        
        from extract_vocal import extract_vocals as extractor_func
        
        sr = self.shared_data.get('sr', cache_manager.get_shared_data('sr', 44100))
        
        chunk_size = int(15 * sr)
        
        # Extract vocals for the first chunk
        live_chunk = live_wave[:min(chunk_size, len(live_wave))]
        
        # Get offset
        offset = self.shared_data.get('offset')
        if offset is None:
            offset = cache_manager.get_shared_data('offset')
            if offset is not None:
                self.shared_data['offset'] = offset
        
        if offset is None:
            # Align using chroma features
            result = self.align_audio_segments(live_wave, cd_wave, chunk_sec=15, sr=sr, use_chroma=True)
            offset = result['offset']
            self.shared_data['offset'] = offset
            self.shared_data['offset_method'] = result['method']
            self.shared_data['offset_confidence'] = result['confidence']
        
        cd_chunk = cd_wave[offset:offset + min(chunk_size, len(live_chunk))]
        
        # Check cached vocals
        live_vocals = cache_manager.get_shared_data('live_vocals')
        cd_vocals = cache_manager.get_shared_data('cd_vocals')
        
        if live_vocals is None:
            live_vocals = cache_manager.extract_vocals(
                live_chunk, 
                sr=sr, 
                extractor_func=extractor_func,
                save_output=save_output, 
                output_path="pre_extracted_live_vocals.wav" if save_output else None
            )
            
            cache_manager.set_shared_data('live_vocals', live_vocals)
            self.shared_data['live_vocals'] = live_vocals
        else:
            self.shared_data['live_vocals'] = live_vocals
        
        if cd_vocals is None:
            cd_vocals = cache_manager.extract_vocals(
                cd_chunk, 
                sr=sr, 
                extractor_func=extractor_func,
                save_output=save_output, 
                output_path="pre_extracted_cd_vocals.wav" if save_output else None
            )
            
            cache_manager.set_shared_data('cd_vocals', cd_vocals)
            self.shared_data['cd_vocals'] = cd_vocals
        else:
            self.shared_data['cd_vocals'] = cd_vocals
        
        # Ensure vocals are mono
        if live_vocals.ndim > 1:
            live_vocals = np.mean(live_vocals, axis=1)
            self.shared_data['live_vocals'] = live_vocals
            cache_manager.set_shared_data('live_vocals', live_vocals)
            
        if cd_vocals.ndim > 1:
            cd_vocals = np.mean(cd_vocals, axis=1)
            self.shared_data['cd_vocals'] = cd_vocals
            cache_manager.set_shared_data('cd_vocals', cd_vocals)
        
        # Sync with global cache
        self._sync_cache(to_global=True)
        
        return {
            'live_vocals': live_vocals,
            'cd_vocals': cd_vocals
        }
    
    def extract_pitch_contours(self):
        """Extract pitch contours from vocals"""
        # Check cached pitch data
        times_live = cache_manager.get_shared_data('times_live')
        f0_live = cache_manager.get_shared_data('f0_live')
        times_cd = cache_manager.get_shared_data('times_cd')
        f0_cd = cache_manager.get_shared_data('f0_cd')
        
        if times_live is not None and f0_live is not None and times_cd is not None and f0_cd is not None:
            self.shared_data.update({
                'times_live': times_live,
                'f0_live': f0_live,
                'times_cd': times_cd,
                'f0_cd': f0_cd
            })
            
            return {
                'times_live': times_live,
                'f0_live': f0_live,
                'times_cd': times_cd,
                'f0_cd': f0_cd
            }
        
        if 'live_vocals' not in self.shared_data or 'cd_vocals' not in self.shared_data:
            self.extract_vocals()
        
        sr = self.shared_data.get('sr', cache_manager.get_shared_data('sr', 44100))
        
        # Get vocals
        if 'live_vocals' not in self.shared_data:
            live_vocals = cache_manager.get_shared_data('live_vocals')
            if live_vocals is None:
                self.extract_vocals()
                live_vocals = self.shared_data.get('live_vocals')
            else:
                self.shared_data['live_vocals'] = live_vocals
        else:
            live_vocals = self.shared_data['live_vocals']
        
        if 'cd_vocals' not in self.shared_data:
            cd_vocals = cache_manager.get_shared_data('cd_vocals')
            if cd_vocals is None:
                self.extract_vocals()
                cd_vocals = self.shared_data.get('cd_vocals')
            else:
                self.shared_data['cd_vocals'] = cd_vocals
        else:
            cd_vocals = self.shared_data['cd_vocals']
        
        # Extract pitch contours
        times_live, f0_live = cache_manager.get_pitch_data(live_vocals, sr=sr)
        times_cd, f0_cd = cache_manager.get_pitch_data(cd_vocals, sr=sr)
        
        # Store pitch data
        self.shared_data.update({
            'times_live': times_live,
            'f0_live': f0_live,
            'times_cd': times_cd,
            'f0_cd': f0_cd
        })
        
        cache_manager.update_shared_data({
            'times_live': times_live,
            'f0_live': f0_live,
            'times_cd': times_cd,
            'f0_cd': f0_cd
        })
        
        return {
            'times_live': times_live,
            'f0_live': f0_live,
            'times_cd': times_cd,
            'f0_cd': f0_cd
        }
    
    def clear_shared_data(self, keep_audio=False):
        """Clear cached data"""
        keep_items = {}
        if keep_audio:
            keys_to_keep = ['live_wave', 'cd_wave', 'sr', 'live_path', 'cd_path']
            for key in keys_to_keep:
                if key in self.shared_data:
                    keep_items[key] = self.shared_data[key]
        
        self.shared_data.clear()
        self.shared_data.update(keep_items)
        
        # Clear global shared data
        if keep_audio:
            keep_keys = ['live_wave', 'cd_wave', 'sr', 'live_path', 'cd_path']
            other_keys = [k for k in cache_manager.shared_data_cache.keys() if k not in keep_keys]
            cache_manager.clear_shared_data(other_keys)
        else:
            cache_manager.clear_shared_data()

audio_processor = AudioProcessor()