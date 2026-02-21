"""
simple_gui.py - GUI application
"""

import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
import os
import threading
import sys
import librosa
import time
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['figure.max_open_warning'] = 0
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import traceback
from PIL import Image, ImageTk

# Add directory containing this script to path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from cache_manager import cache_manager
from audio_processor import audio_processor

def clear_offset_cache():
    """Clear offset cache files and directory"""
    offset_cache_dir = "./.offset_cache"
    if os.path.exists(offset_cache_dir):
        try:
            import shutil
            import glob
            files = glob.glob(os.path.join(offset_cache_dir, "*"))
            for f in files:
                try:
                    os.remove(f)
                except Exception as file_error:
                    print(f"Error removing file {f}: {file_error}")
        
            # Remove and recreate the directory
            try:
                shutil.rmtree(offset_cache_dir)
                os.makedirs(offset_cache_dir)
                print(f"Cleared offset cache directory: {offset_cache_dir}")
            except Exception as dir_error:
                print(f"Error clearing directory structure: {dir_error}")
        except Exception as e:
            print(f"Error during cache directory cleanup: {e}")

    # Clear corrupted cache files
    try:
        import glob
        corrupted_files = glob.glob("./.offset_cache/*.corrupt_*")
        for f in corrupted_files:
            try:
                os.remove(f)
            except Exception as file_error:
                print(f"Error removing corrupted file {f}: {file_error}")
    except Exception as e:
        print(f"Error cleaning corrupted files: {e}")

class OptimizedLipSyncDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lip-Sync Detector")
        self.root.geometry("1000x820")
        
        self.root.attributes('-topmost', True)
        self.root.update()
        self.root.attributes('-topmost', False)
        
        self.root.focus_force()
        
        self.main_frame = ttk.Frame(root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(self.main_frame, text="Audio Analysis for Lip-Syncing Detection", 
                                font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 15))
        
        self.create_file_selection_frame()
        self.create_algorithm_selection_frame()
        
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(pady=10)
        
        self.run_button = ttk.Button(button_frame, text="Run Analysis", command=self.run_analysis)
        self.run_button.pack(pady=5, padx=20, fill=tk.X)
        
        self.clear_all_cache_button = ttk.Button(button_frame, text="Clear All Caches", command=self.clear_all_caches)
        self.clear_all_cache_button.pack(pady=5, padx=20, fill=tk.X)
        
        self.create_results_area()
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.progress_var = tk.DoubleVar()
        self.progress_var.set(0)
        self.progress_bar = ttk.Progressbar(self.main_frame, variable=self.progress_var, 
                                           mode='determinate', length=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.analysis_results = []
        self.plot_canvases = []
        self.result_plots = {}
        self.combined_plot = None
        
        self.current_algorithm = None
        self.analysis_thread = None
        
        self.root.deiconify()
    
    def create_file_selection_frame(self):
        """Create the file selection section of the GUI"""
        file_frame = ttk.LabelFrame(self.main_frame, text="Audio File Selection", padding=10)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        description = ttk.Label(file_frame, text="Select both the live performance audio and the studio recording for comparison.",
                               wraplength=700)
        description.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        
        ttk.Label(file_frame, text="Live Audio:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.live_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.live_path_var, width=60).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_live_audio).grid(row=1, column=2, padx=5, pady=5)
        
        ttk.Label(file_frame, text="CD Audio:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.cd_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.cd_path_var, width=60).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_cd_audio).grid(row=2, column=2, padx=5, pady=5)
        
        note_label = ttk.Label(file_frame, text="Note: Live audio should be between 15 seconds and 1 minute.",
                              foreground="#555555", wraplength=700)
        note_label.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
    
    def create_algorithm_selection_frame(self):
        """Create the algorithm selection section of the GUI"""
        algo_frame = ttk.LabelFrame(self.main_frame, text="Detection Algorithms", padding=10)
        algo_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Label(algo_frame, text="Basic Algorithm:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Label(algo_frame, text="Always Enabled (Pitch Contour Comparing)").grid(row=0, column=1, columnspan=3, sticky=tk.W, pady=5)
        
        ttk.Label(algo_frame, text="Enhanced Algorithm 1:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.use_key_analysis_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(algo_frame, text="Key-based Pitch Analysis", variable=self.use_key_analysis_var).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(algo_frame, text="Song Key (e.g. C, Db, G#):").grid(row=1, column=2, sticky=tk.W, padx=10, pady=5)
        self.song_key_var = tk.StringVar()
        ttk.Entry(algo_frame, textvariable=self.song_key_var, width=5).grid(row=1, column=3, sticky=tk.W, pady=5)
        
        ttk.Label(algo_frame, text="Enhanced Algorithm 2:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.use_variance_analysis_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(algo_frame, text="Pitch Variance Analysis", 
                        variable=self.use_variance_analysis_var).grid(row=2, column=1, columnspan=3, sticky=tk.W, pady=5)
    
    def create_results_area(self):
        """Create the results display area of the GUI"""
        results_frame = ttk.LabelFrame(self.main_frame, text="Analysis Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.detail_button_frame = ttk.Frame(results_frame)
        self.detail_button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.view_detail_button = ttk.Button(
            self.detail_button_frame, 
            text="View Detailed Results", 
            command=self.show_detailed_visualization,
            state=tk.DISABLED
        )
        self.view_detail_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        self.save_plots_button = ttk.Button(
            self.detail_button_frame,
            text="Save All Plots",
            command=self.save_all_plots,
            state=tk.DISABLED
        )
        self.save_plots_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        self.plot_frame = ttk.Frame(results_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def browse_live_audio(self):
        """Open file dialog to select live audio file"""
        filepath = filedialog.askopenfilename(
            title="Select Live Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.ogg *.flac"), ("All Files", "*.*")]
        )
        if filepath:
            self.live_path_var.set(filepath)
    
    def browse_cd_audio(self):
        """Open file dialog to select CD audio file"""
        filepath = filedialog.askopenfilename(
            title="Select CD Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.ogg *.flac"), ("All Files", "*.*")]
        )
        if filepath:
            self.cd_path_var.set(filepath)
    
    def clear_all_caches(self):
        """Clear all cache data"""
        cache_manager.clear_cache()
        audio_processor.clear_shared_data()
        clear_offset_cache()
    
        self.update_status("All caches cleared")
        messagebox.showinfo("Cache Cleared", "All caches have been cleared.")
    
    def _update_progress(self, current_step, total_steps):
        """Update progress bar"""
        self.root.after(0, lambda: self.progress_var.set(current_step * 100 / total_steps))
    
    def run_analysis(self):
        """Main function to run the lip-sync detection analysis"""
        live_path = self.live_path_var.get()
        cd_path = self.cd_path_var.get()
        
        if not live_path or not os.path.exists(live_path):
            messagebox.showerror("Error", "Live audio file not found!")
            return
        
        if not cd_path or not os.path.exists(cd_path):
            messagebox.showerror("Error", "CD audio file not found!")
            return
        
        if self.use_key_analysis_var.get() and not self.song_key_var.get().strip():
            messagebox.showwarning("Warning", "Song key is required for Key-based Analysis!")
            return
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Starting analysis...\n\n")
        
        for canvas in self.plot_canvases:
            canvas.get_tk_widget().destroy()
        self.plot_canvases = []
        
        self.analysis_results = []
        self.result_plots = {}
        self.combined_plot = None
        
        self.progress_var.set(0)
        
        self.view_detail_button.configure(state=tk.DISABLED)
        self.save_plots_button.configure(state=tk.DISABLED)
        
        self.status_var.set("Running analysis...")
        self.run_button.configure(state="disabled")
        
        self.analysis_thread = threading.Thread(target=self.perform_analysis, daemon=True)
        self.analysis_thread.start()
    
    def perform_analysis(self):
        """Execute the lip-sync detection analysis in a separate thread"""
        try:
            live_path = self.live_path_var.get()
            cd_path = self.cd_path_var.get()
            use_key_analysis = self.use_key_analysis_var.get()
            use_variance_analysis = self.use_variance_analysis_var.get()
            song_key = self.song_key_var.get() if use_key_analysis else None
            
            total_steps = 1
            if use_key_analysis:
                total_steps += 1
            if use_variance_analysis:
                total_steps += 1
            
            current_step = 0
            
            self.root.after(0, lambda: self.update_status("Checking audio duration..."))
            live_duration = librosa.get_duration(filename=live_path)
            
            if live_duration < 15:
                self.root.after(0, lambda: self.display_error("Sorry, the audio is too short (<15 seconds)."))
                return
                
            if live_duration > 60:
                self.root.after(0, lambda: self.display_error("Sorry, the audio is too long (>1 minute). Please submit shorter segments."))
                return
            
            self.root.after(0, lambda: self.update_status("Loading and preprocessing audio data..."))
            
            # Load and prepare audio data
            audio_processor.load_and_prepare_audio(live_path, cd_path, sr=44100)
            
            # Pre-extract vocals and pitch data if needed
            if use_key_analysis or use_variance_analysis:
                self.root.after(0, lambda: self.update_status("Pre-extracting common data for all algorithms..."))
                audio_processor.extract_vocals()
                audio_processor.extract_pitch_contours()
            
            self.root.after(0, lambda: self.update_status("Running basic algorithm..."))
            self.current_algorithm = "basic"
            
            basic_result = audio_processor.run_basic_algorithm(
                live_path=live_path,
                cd_path=cd_path,
                sr=44100,
                threshold=0.73
            )
            
            if basic_result["is_fake"]:
                sim_value = basic_result["score"]
                basic_result["explanation"] = f"Similarity: {sim_value:.3f} ≥ 0.73 (Lip-syncing)"
            else:
                basic_result["explanation"] = f"Similarity: < 0.73 (Real singing)"
            
            self.analysis_results.append(basic_result)
            
            if basic_result["plot_path"] and os.path.exists(basic_result["plot_path"]):
                self.result_plots["basic"] = basic_result["plot_path"]
            
            current_step += 1
            self._update_progress(current_step, total_steps)
            
            # Ensure shared data is synchronized
            if use_key_analysis or use_variance_analysis:
                audio_processor._sync_cache(to_global=True)
            
            if use_key_analysis:
                self.root.after(0, lambda: self.update_status("Running key-based pitch analysis..."))
                self.current_algorithm = "key"
                
                # Sync to audio processor
                audio_processor._sync_cache(to_global=True)
                
                key_result = audio_processor.run_key_based_algorithm(
                    key=song_key,
                    threshold=0.55
                )
                
                if key_result["is_fake"] and key_result["plot_path"] and os.path.exists(key_result["plot_path"]):
                    self.result_plots["key"] = key_result["plot_path"]
                
                self.analysis_results.append(key_result)
                
                current_step += 1
                self._update_progress(current_step, total_steps)
            
            if use_variance_analysis:
                self.root.after(0, lambda: self.update_status("Running pitch variance analysis..."))
                self.current_algorithm = "variance"
                
                # Sync to audio processor
                audio_processor._sync_cache(to_global=True)
                
                variance_result = audio_processor.run_variance_algorithm(
                    confidence=0.9,
                    window_size=5
                )
                
                self.analysis_results.append(variance_result)
                
                current_step += 1
                self._update_progress(current_step, total_steps)
            
            any_fake = any(result["is_fake"] for result in self.analysis_results)
            final_result = "LIP-SYNCING DETECTED!" if any_fake else "No lip-syncing detected. The singing appears to be genuine."
            
            self.root.after(0, lambda: self.display_results(final_result))
            
            has_plots = bool(self.result_plots) or bool(self.combined_plot)
            if has_plots:
                self.root.after(0, lambda: self.view_detail_button.configure(state=tk.NORMAL))
                self.root.after(0, lambda: self.save_plots_button.configure(state=tk.NORMAL))
            
            audio_processor.clear_shared_data(keep_audio=False)
        
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            self.root.after(0, lambda error=error_msg: self.display_error(f"Error during analysis: {error}"))
        finally:
            self.root.after(0, lambda: self.finish_analysis())
            self.current_algorithm = None
    
    def update_status(self, message):
        """Update status display"""
        self.status_var.set(message)
        self.results_text.insert(tk.END, f"{message}\n")
        self.results_text.see(tk.END)
    
    def display_error(self, message):
        """Display error message"""
        messagebox.showerror("Error", message)
        self.status_var.set("Error")
        self.run_button.configure(state="normal")
    
    def display_results(self, final_result):
        """Display analysis results"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "===== ANALYSIS RESULTS =====\n\n")
        
        self.results_text.insert(tk.END, "FINAL RESULT: " + final_result + "\n\n")
        
        self.results_text.insert(tk.END, "Individual Algorithm Results:\n")
        for result in self.analysis_results:
            self.results_text.insert(tk.END, f"- {result['name']}: ")
            self.results_text.insert(tk.END, f"{'Fake' if result['is_fake'] else 'Real'}\n")
            self.results_text.insert(tk.END, f"  {result['explanation']}\n\n")
        
        if self.result_plots or self.combined_plot:
            self.results_text.insert(tk.END, "Click 'View Detailed Results' for full analysis plots.")
    
    def finish_analysis(self):
        """Complete the analysis process"""
        self.run_button.configure(state="normal")
        self.status_var.set("Analysis complete")
        self.progress_var.set(100)

    def show_detailed_visualization(self):
        """Show detailed visualization window with plots"""
        if not (self.result_plots or self.combined_plot):
            messagebox.showinfo("No Results", "No visualization results available.")
            return
            
        detail_window = tk.Toplevel(self.root)
        detail_window.title("Detailed Lip-Sync Analysis")
        detail_window.geometry("1400x900")
        
        detail_window.attributes('-topmost', True)
        detail_window.update()
        detail_window.attributes('-topmost', False)
        detail_window.focus_force()
    
        main_frame = ttk.Frame(detail_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
        title_label = ttk.Label(main_frame, text="Lip-Sync Detection Analysis", font=("Arial", 16, "bold"))
        title_label.pack(pady=(5, 10))
    
        fake_results = [r for r in self.analysis_results if r["is_fake"]]
        if fake_results:
            result_label = ttk.Label(main_frame, text="LIP-SYNCING DETECTED", 
                                    font=("Arial", 14, "bold"), foreground="red")
            result_label.pack(pady=(0, 10))
        else:
            result_label = ttk.Label(main_frame, text="No lip-syncing detected", 
                                    font=("Arial", 14))
            result_label.pack(pady=(0, 10))
    
        plot_outer_frame = ttk.Frame(main_frame)
        plot_outer_frame.pack(fill=tk.BOTH, expand=True)
    
        plot_canvas = tk.Canvas(plot_outer_frame)
        v_scrollbar = ttk.Scrollbar(plot_outer_frame, orient="vertical", command=plot_canvas.yview)
        plot_canvas.configure(yscrollcommand=v_scrollbar.set)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
        plot_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
        plot_frame = ttk.Frame(plot_canvas)
        window_id = plot_canvas.create_window((0, 0), window=plot_frame, anchor="nw", width=1300)

        for key, plot_path in self.result_plots.items():
            if key == "variance":  # Skip variance plots
                continue
                    
            if key == "basic":
                title = "Basic Algorithm - Pitch Contour Matching"
            elif key == "key":
                title = f"Key-based Analysis - Key: {self.song_key_var.get()}"
            
            self._add_plot_with_pil(plot_frame, plot_path, title)
    
        plot_frame.update_idletasks()
        plot_canvas.config(scrollregion=plot_canvas.bbox("all"))
    
        explanation_frame = ttk.LabelFrame(main_frame, text="Analysis Details")
        explanation_frame.pack(fill=tk.X, padx=5, pady=10)
    
        explanation_text = scrolledtext.ScrolledText(explanation_frame, wrap=tk.WORD, height=8)
        explanation_text.pack(fill=tk.X, padx=5, pady=5)
    
        explanation_text.insert(tk.END, "DETAILED ANALYSIS RESULTS\n\n")
    
        for result in self.analysis_results:
            explanation_text.insert(tk.END, f"{result['name']}:\n")
            explanation_text.insert(tk.END, f"Result: {'Lip-syncing detected' if result['is_fake'] else 'Real singing'}\n")
            explanation_text.insert(tk.END, f"Explanation: {result['explanation']}\n\n")
    
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
    
        close_button = ttk.Button(button_frame, text="Close", command=detail_window.destroy)
        close_button.pack(side=tk.RIGHT, padx=20)
    
        save_button = ttk.Button(button_frame, text="Save All Plots", 
                                command=lambda: self.save_all_plots())
        save_button.pack(side=tk.RIGHT, padx=5)   

    def _add_plot_with_pil(self, parent_frame, plot_path, title):
        """Add plot to GUI using PIL"""
        if not (plot_path and os.path.exists(plot_path)):
            return False
        
        plot_frame = ttk.LabelFrame(parent_frame, text=title)
        plot_frame.pack(fill=tk.X, padx=5, pady=10, expand=True)
    
        try:
            # Use PIL instead of matplotlib
            img = Image.open(plot_path)
            
            # Calculate size to fit window
            target_width = 1260
            width, height = img.size
            
            # Resize if too large
            if width > target_width:
                ratio = target_width / width
                new_width = target_width
                new_height = int(height * ratio)
                img = img.resize((new_width, new_height), Image.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            
            # Display image
            image_frame = ttk.Frame(plot_frame, width=target_width)
            image_frame.pack(fill=tk.X, padx=5, pady=5)
            
            image_label = ttk.Label(image_frame, image=photo, anchor=tk.CENTER)
            image_label.image = photo  # Keep reference
            image_label.pack(fill=tk.X, padx=0, pady=5)
            
            return True
        except Exception as e:
            print(f"Error displaying plot: {e}")
            traceback.print_exc()
            return False

    def save_all_plots(self):
        """Save all plots to chosen directory"""
        save_dir = filedialog.askdirectory(title="Select Directory to Save Plots")
        if not save_dir:
            return
            
        try:
            import shutil
            saved_count = 0
            
            if self.combined_plot and os.path.exists(self.combined_plot):
                dest_path = os.path.join(save_dir, f"lipsyncing_combined_analysis_{int(time.time())}.png")
                shutil.copy2(self.combined_plot, dest_path)
                saved_count += 1
            
            for key, plot_path in self.result_plots.items():
                if key == "variance":  # Skip variance plots
                    continue
                    
                if plot_path and os.path.exists(plot_path):
                    dest_path = os.path.join(save_dir, f"lipsyncing_{key}_analysis_{int(time.time())}.png")
                    shutil.copy2(plot_path, dest_path)
                    saved_count += 1
            
            if saved_count > 0:
                messagebox.showinfo("Plots Saved", f"{saved_count} plots have been saved.")
            else:
                messagebox.showwarning("No Plots Saved", "No plots were available to save.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error saving plots: {str(e)}")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        root.withdraw() 
        app = OptimizedLipSyncDetectorApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        traceback.print_exc()