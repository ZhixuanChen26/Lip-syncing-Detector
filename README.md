# Lip-Sync Detection System

**By Zhixuan Chen**

---

## Overview

This system detects lip-syncing in live performances by comparing live recordings with studio versions. It uses vocal extraction, pitch contour analysis, and machine learning techniques to identify playback.

---

## System Requirements

- Python 3.8 (recommended)
- At least 4GB of RAM
- 2GB of free disk space
- Windows, macOS, or Linux

---

## Installation

### 1. Set up a Python environment

It's recommended to use Python 3.8 and create a virtual environment:

```bash
python3.8 -m venv lipsync-env
```

Activate the environment:

```bash
# Windows
lipsync-env\Scripts\activate

# macOS/Linux
source lipsync-env/bin/activate
```

### 2. Install dependencies

```bash
pip install numpy scipy matplotlib librosa spleeter soundfile tkinter
```

> **Note:** If you encounter issues with Spleeter, install it separately with `pip install spleeter`. Spleeter requires TensorFlow, which will be installed automatically as a dependency.

---

## Running the Application

### GUI

```bash
python simple_gui.py
```

### Command Line

```bash
python main.py --live [PATH_TO_LIVE_AUDIO] --cd [PATH_TO_CD_AUDIO] [OPTIONS]
```

**Options:**

| Flag | Description |
|---|---|
| `--use-key-analysis` | Enable key-based pitch analysis (requires `--key`) |
| `--key` | Song key (e.g., `"C"`, `"D#"`, `"Ab"`) |
| `--use-variance-analysis` | Enable pitch variance analysis |
| `--clear-cache` | Clear cached data before running |
| `--save-output` | Save intermediate files and plots |

**Example:**

```bash
python main.py --live ./Test\ data\ samples/fake/sample1_live.wav \
               --cd ./Test\ data\ samples/fake/sample1_cd.wav \
               --use-key-analysis --key "G" --use-variance-analysis
```

---

## Test Data

The provided test dataset contains:

- `fake/` — 6 samples of confirmed lip-synced performances
- `real/` — 4 samples of authentic live performances
- `uncertain/` — 2 samples of ambiguous cases
- `Key.docx` — Musical key for each sample

---

## Sample Testing Procedure

1. Select a pair of files (live and CD versions) from one of the test folders
2. Enter or browse to select both files in the GUI
3. Enable key-based analysis and enter the correct key from `Key.docx`
4. Run the analysis
5. View the detailed results

---

## Notes

- Audio files should be **15 seconds to 1 minute** for optimal analysis
- **WAV preferred over MP3** for better accuracy
- Initial analysis may be slow due to model loading and vocal separation
- The system creates a cache folder `./.offset_cache` — clear it if you encounter issues
- After testing multiple files, memory usage may increase; use **"Clear All Caches"** in the GUI or restart periodically
- Fake singing detection: ~15–25s; real singing takes longer as it is tested more thoroughly

---

## Troubleshooting

- **App fails to start:** Check Python version and dependencies
- **Vocal extraction fails:** Ensure Spleeter is properly installed
- **File not found errors:** Check file paths and permissions
- **System appears unresponsive:** It may still be processing — vocal separation is computationally intensive

---

## Contact

For questions or technical support: diangee470399@hotmail.com
