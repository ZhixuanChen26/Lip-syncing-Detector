Lip-Sync Detection System

By Zhixuan Chen
------------------------------------------------
Overview

This system detects lip-syncing in live performances by comparing live recordings with studio versions. It uses vocal extraction, pitch contour analysis, and machine learning techniques to identify playback.
System Requirements

Python 3.8 (recommended)
At least 4GB of RAM
2GB of free disk space
Windows, macOS, or Linux operating system
------------------------------------------------
Installation

1. Set up a Python environment
It's recommended to use Python 3.8 and create a virtual environment:

python3.8 -m venv lipsync-env

Activate the environment:

Windows: lipsync-env\Scripts\activate
macOS/Linux: source lipsync-env/bin/activate

2. Install dependencies
Install all required packages:

pip install numpy scipy matplotlib librosa spleeter soundfile tkinter

If you encounter issues with Spleeter installation, you can install it separately:

pip install spleeter

Note: Spleeter requires TensorFlow, which will be installed automatically as a dependency.
------------------------------------------------
Running the Application

1. Launch the GUI

python simple_gui.py

2. Using the command line interface

python main.py --live [PATH_TO_LIVE_AUDIO] --cd [PATH_TO_CD_AUDIO] [OPTIONS]

Options:

--use-key-analysis: Enable key-based pitch analysis (requires --key)
--key: Song key (e.g., "C", "D#", "Ab")
--use-variance-analysis: Enable pitch variance analysis
--clear-cache: Clear cached data before running
--save-output: Save intermediate files and plots

Example:

python main.py --live ./Test\ data\ samples/fake/sample1_live.wav --cd ./Test\ data\ samples/fake/sample1_cd.wav --use-key-analysis --key "G" --use-variance-analysis
------------------------------------------------
Test Data

The provided test dataset contains:

fake/: 6 samples of confirmed lip-synced performances
real/: 4 samples of authentic live performances
uncertain/: 2 samples of ambiguous cases
Key.docx: Document containing the musical key for each sample
------------------------------------------------
Sample Testing Procedure

Select a pair of files (live and CD versions) from one of the test folders
Enter or browse to select both files in the GUI
Enable the key-based analysis and enter the correct key from Key.docx
Run the analysis
View the detailed results
------------------------------------------------
Notes and Considerations

Audio files should be between 15 seconds and 1 minute in length for optimal analysis
Higher quality audio files yield better results (WAV preferred over MP3)
Initial analysis may be slow due to model loading and vocal separation
For better results with key-based analysis, ensure you enter the correct musical key
The system creates a cache folder ./.offset_cache - if you encounter issues, try clearing the cache
Important: After testing multiple audio files, memory usage may increase. Use the "Clear All Caches" button in the GUI or restart the application periodically to free up memory
------------------------------------------------
Troubleshooting

If the application fails to start, check Python version and dependencies
If vocal extraction fails, ensure Spleeter is properly installed
For "file not found" errors, check file paths and permissions
If the system appears unresponsive, it may be processing (vocal separation is computationally intensive)
Fake singing costs 15-25s to detect, while real singing costs more time since it will be tested thoroughly
------------------------------------------------
Contact
For questions or technical support, please contact: diangee470399@hotmail.com