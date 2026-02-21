#get_spectrum.py

import numpy as np
import librosa

def get_pitch_and_times(wave, sr=44100):
    """Extract pitch and corresponding times from audio"""
    f0, vf, _ = librosa.pyin(
        wave,
        sr=sr,
        fmin=librosa.note_to_hz('E2'),  # approximately 82Hz
        fmax=librosa.note_to_hz('E5')   # approximately 659Hz
    )
    # Set undetected portions to NaN
    f0[~vf] = np.nan

    # Generate the time axis
    times = librosa.times_like(f0, sr=sr)
    return times, f0


def plot_compare_pitch_contours(times_live, f0_live,
                               times_cd,   f0_cd,
                               output_image="compare_pitch.png",
                               sim=None,
                               decision=None):
    """Plot pitch contours for comparison between live and CD audio"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    
    # Plot the CD's pitch contour (blue)
    plt.plot(times_cd, f0_cd, color='blue', linewidth=1.5, marker='o', markersize=1.5, 
             label='Record', zorder=1)
    # Plot the Live pitch contour (red)
    plt.plot(times_live, f0_live, color='red', linewidth=0.8, marker='o', markersize=0.8, 
             label='Live', zorder=2)

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Pitch Contour (Live vs. Record)")

    # Place the legend in the upper right corner
    plt.legend(loc='upper right')

    # If similarity and results are provided, annotate the plot
    if sim is not None and decision is not None:
        plt.gca().text(
            0.01, 0.95,
            f"Similarity: {sim*100:.1f}%\n{decision}",
            transform=plt.gca().transAxes,
            fontsize=12, color='black',
            verticalalignment='top',
            horizontalalignment='left'
        )

    plt.tight_layout()
    plt.savefig(output_image, dpi=200)
    plt.close()  
    
    print(f"Comparison pitch contour plot saved: {output_image}")


def plot_key_comparison(times, pitch, key, scale_frequencies, pitch_range=None, 
                       similarity_value=None, output_image="key_comparison.png"):
    """Plot pitch compared to key scale frequencies"""
    try:
        import matplotlib.pyplot as plt
        
        # Calculate effective pitch range
        if pitch_range is None:
            valid_pitch = pitch[~np.isnan(pitch)]
            if len(valid_pitch) == 0:
                return None
                
            valid_pitch_no_outliers = valid_pitch[
                (valid_pitch > np.percentile(valid_pitch, 1)) & 
                (valid_pitch < np.percentile(valid_pitch, 99))
            ]
            
            if len(valid_pitch_no_outliers) == 0:
                return None
                
            min_pitch = np.min(valid_pitch_no_outliers) * 0.8
            max_pitch = np.max(valid_pitch_no_outliers) * 1.2
            pitch_range = (min_pitch, max_pitch)
        else:
            min_pitch, max_pitch = pitch_range
        
        plt.figure(figsize=(10, 4))
        
        # Plot singer's pitch contour
        plt.plot(times, pitch, color='red', linewidth=0.8, marker='o', markersize=1.0, 
                label='Live')
        
        # Select main scale tones (every 7th note)
        main_scale = scale_frequencies[::7]
        
        # Track if Key label was added
        key_label_added = False
        
        # Draw all frequency lines, thicker for main scale tones
        for freq in scale_frequencies:
            if min_pitch <= freq <= max_pitch:
                # Check if main scale frequency
                is_main_scale = freq in main_scale
                
                # Set line width
                linewidth = 0.8 if is_main_scale else 0.5
                
                # Add label for main scale, but only once
                label = ""
                if is_main_scale and not key_label_added:
                    label = "Key"
                    key_label_added = True
                
                plt.axhline(y=freq, color='#0055CC', linestyle='-', 
                           linewidth=linewidth, alpha=0.5, label=label)
        
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(f"Pitch Alignment with {key} Key")
        
        # Show similarity value
        sim_value = similarity_value if similarity_value is not None else 0.0
        plt.gca().text(
            0.01, 0.95,
            f"Similarity: {sim_value*100:.1f}%\n{'Fake' if sim_value >= 0.86 else 'Real'}",
            transform=plt.gca().transAxes,
            fontsize=12, color='black',
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=5)
        )
        
        plt.legend(loc='upper right')
        
        # Set y-axis range
        plt.ylim(min_pitch, max_pitch)
        
        plt.tight_layout()
        plt.savefig(output_image, dpi=200)
        plt.close()
        
        return output_image
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None