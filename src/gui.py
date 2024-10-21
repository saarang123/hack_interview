from PyQt5.QtWidgets import (
    QWidget, QPushButton, QLabel, QTextEdit,
    QVBoxLayout, QHBoxLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap
from loguru import logger
import asyncio

from .threads import (
    start_transcription, 
    process_audio, 
    handle_transcription, 
    stop_transcription
)
from .threads import ChatGPTThread


from .llm import LLMInference
from .constants import APPLICATION_WIDTH, OFF_IMAGE, ON_IMAGE

logger.add("debug.log", level="DEBUG", rotation="3 MB", compression="zip")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.recording = False
        self.recording_thread = None
        self.transcribe_thread = None
        self.quick_answer_thread = None
        self.full_answer_thread = None

        self.llm = LLMInference()

        self.audio_transcript = None

        self.initUI()

    def initUI(self):
        # Set window title and size
        self.setWindowTitle('Keyboard Test')
        self.setGeometry(100, 100, APPLICATION_WIDTH, 800)

        # Create the record_status_button
        self.record_status_button = QPushButton("Start Recording")
        self.off_pixmap = QPixmap(OFF_IMAGE)
        self.on_pixmap = QPixmap(ON_IMAGE)

        # Check if the pixmaps are successfully loaded
        if self.off_pixmap.isNull():
            logger.error("Failed to load OFF_IMAGE.")
        if self.on_pixmap.isNull():
            logger.error("Failed to load ON_IMAGE.")

        # Set initial icon for the button
        self.record_status_button.setIcon(QIcon(self.off_pixmap) if not self.off_pixmap.isNull() else QIcon())
        self.record_status_button.setIconSize(self.off_pixmap.size() if not self.off_pixmap.isNull() else self.record_status_button.size())
        self.record_status_button.setFlat(True)
        self.record_status_button.clicked.connect(self.toggle_recording)

        # Create the analyze button
        self.analyze_button = QPushButton("Analyze Recording")
        self.analyze_button.clicked.connect(self.handle_transcription_done)

        # Create labels and text areas
        self.info_label = QLabel("Use buttons to control the recording and analyze audio.")
        self.info_label.setFixedHeight(50)
        self.analyzed_text_label = QLabel("")
        self.analyzed_text_label.setFixedHeight(50)
        self.quick_chat_gpt_answer = QTextEdit()
        self.quick_chat_gpt_answer.setReadOnly(True)
        self.full_chat_gpt_answer = QTextEdit()
        self.full_chat_gpt_answer.setReadOnly(True)

        # Layouts
        main_layout = QVBoxLayout()
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.info_label)
        h_layout.addWidget(self.record_status_button)
        h_layout.addWidget(self.analyze_button)
        main_layout.addLayout(h_layout)

        main_layout.addWidget(QLabel("Analysis Result:"))
        main_layout.addWidget(self.analyzed_text_label)
        main_layout.addWidget(QLabel("Short answer:"))
        main_layout.addWidget(self.quick_chat_gpt_answer)
        main_layout.addWidget(QLabel("Full answer:"))
        main_layout.addWidget(self.full_chat_gpt_answer)

        self.setLayout(main_layout)

    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            self.record_status_button.setIcon(QIcon(self.on_pixmap) if not self.on_pixmap.isNull() else QIcon())
            self.record_status_button.setText("Recording... Click to Stop")
            self.record_status_button.setStyleSheet("background-color: red; color: white;")  # Make button stand out
            logger.debug("Starting recording...")
            self.start_recording_thread()
        else:
            self.record_status_button.setIcon(QIcon(self.off_pixmap) if not self.off_pixmap.isNull() else QIcon())
            self.record_status_button.setText("Start Recording")
            self.record_status_button.setStyleSheet("")  # Reset to default style
            logger.debug("Stopping recording...")
            self.stop_recording_thread()

    def start_recording_thread(self):
        if not self.recording_thread:
            self.recording_thread = True
            start_transcription()
            asyncio.run(process_audio())

    def stop_recording_thread(self):
        if self.recording_thread:
            self.audio_transcript = stop_transcription()
            self.recording_thread = False

    def handle_transcription_done(self):
        if self.audio_transcript == None:
            return
        self.analyzed_text_label.setText(self.audio_transcript)

        # Generate quick answer:
        self.quick_chat_gpt_answer.setText("ChatGPT is working...")
        self.quick_answer_thread = ChatGPTThread(
            self.llm, audio_transcript, short_answer=True, temperature=0.2
        )
        self.quick_answer_thread.answer_ready.connect(self.handle_quick_answer_ready)
        self.quick_answer_thread.start()

        # Generate full answer:
        self.full_chat_gpt_answer.setText("ChatGPT is working...")
        self.full_answer_thread = ChatGPTThread(
            self.llm, self.audio_transcript, short_answer=False, temperature=0.7
        )
        self.full_answer_thread.answer_ready.connect(self.handle_full_answer_ready)
        self.full_answer_thread.start()

    def handle_quick_answer_ready(self, answer):
        self.quick_chat_gpt_answer.setText(answer)

    def handle_full_answer_ready(self, answer):
        self.full_chat_gpt_answer.setText(answer)

    def closeEvent(self, event):
        # Clean up threads when closing
        if self.recording_thread:
            stop_transcription()
        if self.transcribe_thread and self.transcribe_thread.isRunning():
            self.transcribe_thread.quit()
        if self.quick_answer_thread and self.quick_answer_thread.isRunning():
            self.quick_answer_thread.quit()
        if self.full_answer_thread and self.full_answer_thread.isRunning():
            self.full_answer_thread.quit()
        event.accept()
