#!/usr/bin/env python3
"""
main.py - Command-line interface for lip-syncing detection system
"""

import os
import sys
import time
import argparse
import librosa

# Add directory containing this script to path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from audio_processor import audio_processor
from cache_manager import cache_manager

def main():
    """Main entry point for the CLI application"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Lip-sync detection with enhanced algorithms')
    parser.add_argument('--live', type=str, required=True, help='Path to live audio file')
    parser.add_argument('--cd', type=str, required=True, help='Path to CD audio file')
    parser.add_argument('--use-key-analysis', action='store_true', help='Enable key-based pitch analysis')
    parser.add_argument('--use-variance-analysis', action='store_true', help='Enable pitch variance analysis')
    parser.add_argument('--key', type=str, default=None, help='Song key (e.g., "C", "D#") for key-based analysis')
    parser.add_argument('--clear-cache', action='store_true', help='Clear all caches before running')
    parser.add_argument('--save-output', action='store_true', help='Save intermediate files and plots')
    
    args = parser.parse_args()
    
    # Get audio file paths
    live_audio = args.live
    cd_audio = args.cd
    
    # Validate arguments
    if args.use_key_analysis and not args.key:
        print("Error: --key argument is required when using key-based analysis")
        return 1
    
    start_time = time.time()
    
    try:
        # Clear caches if requested
        if args.clear_cache:
            print("Clearing cache...")
            audio_processor.clear_shared_data()
            cache_manager.clear_cache()
        
        # 1. Check audio length
        print("Checking audio length...")
        live_duration = librosa.get_duration(filename=live_audio)
        
        if live_duration < 15:
            print("Error: Audio is too short (<15 seconds), cannot perform detection.")
            return 1
        if live_duration > 60:
            print("Error: Audio is too long (>1 minute), not supported.")
            return 1
        
        # 2. Load and prepare audio data
        print("Loading and preparing audio data...")
        audio_processor.load_and_prepare_audio(live_audio, cd_audio, sr=44100)
        
        # Pre-extract vocals and pitch data if multiple algorithms will be used
        if args.use_key_analysis or args.use_variance_analysis:
            print("Pre-extracting common data for all algorithms...")
            audio_processor.extract_vocals(save_output=args.save_output)
            audio_processor.extract_pitch_contours()
        
        # 3. Run basic algorithm
        print("\n=== Running Basic Algorithm ===")
        basic_result = audio_processor.run_basic_algorithm(
            live_path=live_audio,
            cd_path=cd_audio,
            sr=44100,
            threshold=0.71
        )
        
        print(f"Basic algorithm result: {'Lip-syncing detected' if basic_result['is_fake'] else 'Real singing'}")
        print(f"Score: {basic_result['score']:.3f}")
        print(f"Explanation: {basic_result['explanation']}")
        
        results = [basic_result]
        
        # 4. Run enhanced algorithms if selected
        if args.use_key_analysis:
            print("\n=== Running Key-based Pitch Analysis ===")
            key_result = audio_processor.run_key_based_algorithm(
                key=args.key,
                threshold=0.86
            )
            
            print(f"Key-based analysis result: {'Lip-syncing detected' if key_result['is_fake'] else 'Real singing'}")
            print(f"Score: {key_result['score']:.3f}")
            print(f"Explanation: {key_result['explanation']}")
            
            results.append(key_result)
        
        if args.use_variance_analysis:
            print("\n=== Running Pitch Variance Analysis ===")
            variance_result = audio_processor.run_variance_algorithm(
                confidence=0.9,
                window_size=5
            )
            
            print(f"Variance analysis result: {'Lip-syncing detected' if variance_result['is_fake'] else 'Real singing'}")
            print(f"Score: {variance_result['score']:.3f}")
            print(f"Explanation: {variance_result['explanation']}")
            
            results.append(variance_result)
        
        # 5. Aggregate results
        any_fake = any(result["is_fake"] for result in results)
        
        print("\n" + "="*50)
        print("Final Result:")
        if any_fake:
            print("LIP-SYNCING DETECTED!")
            # Identify which algorithms detected lip-syncing
            for result in results:
                if result["is_fake"]:
                    print(f"- {result['name']}: Detected lip-syncing (score: {result['score']:.3f})")
        else:
            print("No lip-syncing detected. The singing appears to be genuine.")
        
        print("="*50)
        
        # 6. Clean up
        if not args.save_output:
            audio_processor.clear_shared_data(keep_audio=False)
        
        total_time = time.time() - start_time
        print(f"\nAnalysis completed in {total_time:.2f} seconds")
        
        return 0
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())