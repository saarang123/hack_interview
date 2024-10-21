import tkinter as tk
from tkinter import ttk, messagebox
import threading
import asyncio

from src.threads import (
    start_transcription, 
    process_audio, 
    handle_transcription, 
    stop_transcription,
    generate_answer
)

from src.constants import APPLICATION_WIDTH, OFF_IMAGE, ON_IMAGE
from loguru import logger

logger.add("debug.log", level="DEBUG", rotation="3 MB", compression="zip")

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title('Keyboard Test')
        self.root.geometry(f"{APPLICATION_WIDTH}x800")

        self.recording = False
        self.recording_thread = None
        self.transcribe_thread = None
        self.quick_answer_thread = None
        self.full_answer_thread = None

        self.audio_transcript = None

        self.initUI()

    def initUI(self):
        # Create and configure widgets
        self.record_status_button = ttk.Button(self.root, text="Start Recording", command=self.toggle_recording)
        self.analyze_button = ttk.Button(self.root, text="Analyze Recording", command=self.handle_transcription_done)

        self.info_label = ttk.Label(self.root, text="Use buttons to control the recording and analyze audio.")
        self.analyzed_text_label = ttk.Label(self.root, text="")
        self.quick_chat_gpt_answer = tk.Text(self.root, height=5, width=50, state='disabled')
        self.full_chat_gpt_answer = tk.Text(self.root, height=5, width=50, state='disabled')

        # Create layout
        self.info_label.pack(pady=10)
        self.record_status_button.pack(pady=10)
        self.analyze_button.pack(pady=10)
        
        ttk.Label(self.root, text="Analysis Result:").pack(pady=5)
        self.analyzed_text_label.pack(pady=5)
        
        ttk.Label(self.root, text="Short answer:").pack(pady=5)
        self.quick_chat_gpt_answer.pack(pady=5)

        ttk.Label(self.root, text="Full answer:").pack(pady=5)
        self.full_chat_gpt_answer.pack(pady=5)

    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            self.record_status_button.config(text="Recording... Click to Stop", style="danger.TButton")
            logger.debug("Starting recording...")
            self.start_recording_thread()
        else:
            self.record_status_button.config(text="Start Recording", style="")
            logger.debug("Stopping recording...")
            self.stop_recording_thread()

    def start_recording_thread(self):
        if not self.recording_thread:
            self.recording_thread = True
            start_transcription()
            asyncio.run(process_audio())

    def run_recording_process(self):
        start_transcription()
        asyncio.run(process_audio())

    def stop_recording_thread(self):
        if self.recording_thread:
            self.audio_transcript = stop_transcription()
            self.recording_thread = False

    def handle_transcription_done(self):
        if self.audio_transcript is None:
            messagebox.showerror("Error", "No transcription available!")
            return
        self.analyzed_text_label.config(text=self.audio_transcript)

        # Start threads for quick and full answers
        self.generate_quick_answer()

        self.generate_full_answer()

    def generate_quick_answer(self):
        self.quick_chat_gpt_answer.config(state='normal')
        self.quick_chat_gpt_answer.delete(1.0, tk.END)
        self.quick_chat_gpt_answer.insert(tk.END, "ChatGPT is working...")
        self.quick_chat_gpt_answer.config(state='disabled')

        quick_answer = generate_answer(self.audio_transcript, short_answer=True, temperature=0.2)

        self.quick_chat_gpt_answer.config(state='normal')
        self.quick_chat_gpt_answer.delete(1.0, tk.END)
        self.quick_chat_gpt_answer.insert(tk.END, quick_answer)
        self.quick_chat_gpt_answer.config(state='disabled')

    def generate_full_answer(self):
        self.full_chat_gpt_answer.config(state='normal')
        self.full_chat_gpt_answer.delete(1.0, tk.END)
        self.full_chat_gpt_answer.insert(tk.END, "ChatGPT is working...")
        self.full_chat_gpt_answer.config(state='disabled')

        quick_answer = generate_answer(self.audio_transcript, short_answer=True, temperature=0.2)

        self.full_chat_gpt_answer.config(state='normal')
        self.full_chat_gpt_answer.delete(1.0, tk.END)
        self.full_chat_gpt_answer.insert(tk.END, full_answer)
        self.full_chat_gpt_answer.config(state='disabled')

    def close(self):
        if self.recording_thread:
            stop_transcription()

# Main app window creation
if __name__ == '__main__':
    root = tk.Tk()
    app = MainWindow(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()

# import sys
# from PyQt5.QtWidgets import QApplication
# from src.gui import MainWindow
# from loguru import logger

# logger.add("debug.log", level="DEBUG", rotation="3 MB", compression="zip")

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())
