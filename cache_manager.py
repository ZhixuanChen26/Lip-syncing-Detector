"""
cache_manager.py - Caching system
"""

import numpy as np
import hashlib
import json
import os
from functools import wraps

class OffsetCache:
    def __init__(self, cache_dir="./.offset_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, "offset_cache.json")
        self.cache = self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                # Create backup of corrupted file
                backup_file = f"{self.cache_file}.corrupt_{int(time.time())}"
                try:
                    import time
                    os.rename(self.cache_file, backup_file)
                except Exception:
                    pass
                return {}
            except Exception:
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            # Create a temporary file first
            temp_file = f"{self.cache_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
            
            # Rename temp file to actual cache file
            os.replace(temp_file, self.cache_file)
        except Exception:
            # Remove temporary file if it exists
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    def get_cache_key(self, live_path, cd_path, method='chroma'):
        """Generate a unique cache key for audio pair"""
        try:
            with open(live_path, 'rb') as f:
                live_hash = hashlib.md5(f.read(2 * 1024 * 1024)).hexdigest()  # First 2MB
            with open(cd_path, 'rb') as f:
                cd_hash = hashlib.md5(f.read(2 * 1024 * 1024)).hexdigest()  # First 2MB
            
            return f"{live_hash}_{cd_hash}_{method}"
        except Exception:
            return None
    
    def set_offset(self, live_path, cd_path, offset, confidence, method='chroma'):
        """Store offset information in cache"""
        key = self.get_cache_key(live_path, cd_path, method)
        if key:
            if hasattr(offset, 'item'): 
                offset = offset.item()  
            if hasattr(confidence, 'item'):
                confidence = confidence.item()
            
            import time
            self.cache[key] = {
                'offset': offset,
                'confidence': confidence,
                'method': method,
                'timestamp': time.time(),
                'live_path': live_path,
                'cd_path': cd_path
            }
            self._save_cache()
    
    def get_offset(self, live_path, cd_path, method='chroma'):
        """Retrieve offset information from cache"""
        key = self.get_cache_key(live_path, cd_path, method)
        if key and key in self.cache:
            return self.cache[key]
        return None
    
    def clear(self):
        """Clear all offset cache"""
        self.cache = {}
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
            except Exception:
                pass
        # Clear backup files as well
        try:
            cache_dir = os.path.dirname(self.cache_file)
            for file in os.listdir(cache_dir):
                if file.startswith("offset_cache.json.corrupt_"):
                    os.remove(os.path.join(cache_dir, file))
        except Exception:
            pass

class CacheManager:
    """Centralized cache manager for the lip-syncing detection project"""
    
    def __init__(self):
        # Main cache dictionaries
        self.audio_cache = {}  # For audio data
        self.vocal_cache = {}  # For extracted vocals
        self.pitch_cache = {}  # For pitch data
        self.feature_cache = {}  # For algorithm results
        
        # In-memory shared data cache
        self.shared_data_cache = {}
        
        # Add offset cache
        self.offset_cache = OffsetCache()
    
    def get_cached_data(self, cache_dict, key, factory_func, *args, **kwargs):
        """Get data from cache or compute and cache it"""
        if key in cache_dict:
            return cache_dict[key]
        
        data = factory_func(*args, **kwargs)
        cache_dict[key] = data
        return data
    
    def generate_audio_key(self, audio_path):
        """Generate a cache key for an audio file"""
        try:
            with open(audio_path, "rb") as f:
                file_hash = hashlib.md5(f.read(1024 * 1024)).hexdigest()  # Hash first 1MB
            return f"audio_{file_hash}"
        except Exception:
            return f"audio_{hash(audio_path)}"
    
    def generate_data_key(self, data, prefix="data"):
        """Generate a cache key for numpy array data"""
        try:
            # Use the first 1000 bytes of the data to create a hash
            if isinstance(data, np.ndarray):
                data_sample = data[:min(1000, len(data))].tobytes()
            else:
                import pickle
                data_sample = pickle.dumps(data)[:1000]
            
            key = f"{prefix}_{hashlib.md5(data_sample).hexdigest()}"
            return key
        except Exception:
            return f"{prefix}_{hash(str(data))}"
    
    def load_audio(self, audio_path, sr=44100):
        """Load audio data with caching"""
        key = self.generate_audio_key(audio_path)
        
        def _load_audio():
            import librosa
            audio, sr_value = librosa.load(audio_path, sr=sr, mono=True)
            return audio, sr_value
        
        return self.get_cached_data(self.audio_cache, key, _load_audio)
    
    def extract_vocals(self, audio_data, sr=44100, extractor_func=None, save_output=False, output_path=None):
        """Extract vocals with caching"""
        key = self.generate_data_key(audio_data, prefix="vocals")
        
        def _extract_vocals():
            if extractor_func is None:
                from extract_vocal import extract_vocals as default_extractor
                vocals = default_extractor(audio_data, sr, save_output, output_path)
            else:
                vocals = extractor_func(audio_data, sr, save_output, output_path)
            
            return vocals
        
        return self.get_cached_data(self.vocal_cache, key, _extract_vocals)
    
    def get_pitch_data(self, audio_data, sr=44100, pitch_extractor_func=None):
        """Extract pitch data with caching"""
        key = self.generate_data_key(audio_data, prefix="pitch")
        
        def _extract_pitch():
            if pitch_extractor_func is None:
                from get_spectrum import get_pitch_and_times
                times, pitch = get_pitch_and_times(audio_data, sr=sr)
            else:
                times, pitch = pitch_extractor_func(audio_data, sr=sr)
            
            return times, pitch
        
        return self.get_cached_data(self.pitch_cache, key, _extract_pitch)
    
    def update_shared_data(self, data_dict):
        """Update shared data with a dictionary"""
        self.shared_data_cache.update(data_dict)
    
    def get_shared_data(self, key, default=None):
        """Get shared data between algorithms"""
        return self.shared_data_cache.get(key, default)
    
    def set_shared_data(self, key, value):
        """Set shared data between algorithms"""
        self.shared_data_cache[key] = value
    
    def clear_shared_data(self, keys=None):
        """Clear specific or all shared data"""
        if keys is None:
            self.shared_data_cache.clear()
        else:
            for key in keys:
                if key in self.shared_data_cache:
                    del self.shared_data_cache[key]
    
    def clear_cache(self, cache_type=None):
        """Clear specified cache or all caches"""
        if cache_type is None or cache_type == "all":
            self.audio_cache.clear()
            self.vocal_cache.clear()
            self.pitch_cache.clear()
            self.feature_cache.clear()
            self.offset_cache.clear()
            self.shared_data_cache.clear()
        elif cache_type == "audio":
            self.audio_cache.clear()
        elif cache_type == "vocals":
            self.vocal_cache.clear()
        elif cache_type == "pitch":
            self.pitch_cache.clear()
        elif cache_type == "features":
            self.feature_cache.clear()
        elif cache_type == "offset":
            self.offset_cache.clear()
        elif cache_type == "shared":
            self.shared_data_cache.clear()

# Global instance for use across modules
cache_manager = CacheManager()

class ChunkDataCache:
    def __init__(self):
        self.chunk_cache = {}
    
    def get_chunk_key(self, audio_path, start_sample, length, sr=44100):
        file_hash = hashlib.md5(audio_path.encode()).hexdigest()[:8]
        return f"{file_hash}_{start_sample}_{length}_{sr}"
    
    def has_chunk_vocals(self, audio_path, start_sample, length, sr=44100):
        key = self.get_chunk_key(audio_path, start_sample, length, sr)
        return key in self.chunk_cache and 'vocals' in self.chunk_cache[key]
    
    def get_chunk_vocals(self, audio_path, start_sample, length, sr=44100):
        key = self.get_chunk_key(audio_path, start_sample, length, sr)
        if key in self.chunk_cache and 'vocals' in self.chunk_cache[key]:
            return self.chunk_cache[key]['vocals']
        return None
    
    def cache_chunk_vocals(self, audio_path, start_sample, length, vocals, sr=44100):
        key = self.get_chunk_key(audio_path, start_sample, length, sr)
        if key not in self.chunk_cache:
            self.chunk_cache[key] = {}
        self.chunk_cache[key]['vocals'] = vocals
    
    def has_chunk_pitch(self, audio_path, start_sample, length, sr=44100):
        key = self.get_chunk_key(audio_path, start_sample, length, sr)
        return key in self.chunk_cache and 'times' in self.chunk_cache[key] and 'pitch' in self.chunk_cache[key]
    
    def get_chunk_pitch(self, audio_path, start_sample, length, sr=44100):
        key = self.get_chunk_key(audio_path, start_sample, length, sr)
        if key in self.chunk_cache and 'times' in self.chunk_cache[key] and 'pitch' in self.chunk_cache[key]:
            return self.chunk_cache[key]['times'], self.chunk_cache[key]['pitch']
        return None, None
    
    def cache_chunk_pitch(self, audio_path, start_sample, length, times, pitch, sr=44100):
        key = self.get_chunk_key(audio_path, start_sample, length, sr)
        if key not in self.chunk_cache:
            self.chunk_cache[key] = {}
        self.chunk_cache[key]['times'] = times
        self.chunk_cache[key]['pitch'] = pitch
    
    def clear(self):
        self.chunk_cache.clear()

chunk_data_cache = ChunkDataCache()

# Decorator for caching function results
def cached(cache_dict_name, key_prefix):
    """Decorator to cache function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate a key from function name, args and kwargs
            import pickle
            key_data = pickle.dumps((func.__name__, args, kwargs))
            key = f"{key_prefix}_{hashlib.md5(key_data).hexdigest()}"
            
            # Get the cache dictionary
            cache_dict = getattr(cache_manager, cache_dict_name)
            
            # Check if result is in cache
            if key in cache_dict:
                return cache_dict[key]
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Cache result
            cache_dict[key] = result
            
            return result
        return wrapper
    return decorator