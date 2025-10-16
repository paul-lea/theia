# Whisper Real-Time Transcriber with Tkinter
# Requirements: openai-whisper, sounddevice, numpy, tkinter
# Install with: pip install openai-whisper sounddevice numpy

import threading
import queue
import sounddevice as sd
import numpy as np
import whisper
import tkinter as tk

# Parameters
SAMPLE_RATE = 16000
BLOCK_DURATION = 1  # seconds per transcription block
MODEL_SIZE = "small"  # Change to "small", "medium", "large" for better accuracy

class TranscriberApp:
    def __init__(self, master):
        self.master = master
        master.title("Whisper Real-Time Transcriber")
        self.text = tk.Text(master, height=20, width=80, font=("Arial", 16))
        self.text.pack()
        self.text.insert(tk.END, "Listening...\n")
        self.audio_queue = queue.Queue()
        self.transcript_queue = queue.Queue()
        self.model = whisper.load_model(MODEL_SIZE)
        self.running = True
        threading.Thread(target=self.audio_thread, daemon=True).start()
        threading.Thread(target=self.transcribe_thread, daemon=True).start()
        self.update_gui()

    def audio_thread(self):
        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.audio_queue.put(indata.copy())
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=callback):
            while self.running:
                sd.sleep(int(BLOCK_DURATION * 1000))

    def transcribe_thread(self):
        buffer = np.empty((0, 1), dtype='float32')
        while self.running:
            try:
                while not self.audio_queue.empty():
                    data = self.audio_queue.get()
                    buffer = np.concatenate((buffer, data), axis=0)
                if buffer.shape[0] >= SAMPLE_RATE * BLOCK_DURATION:
                    audio_block = buffer[:SAMPLE_RATE * BLOCK_DURATION]
                    buffer = buffer[SAMPLE_RATE * BLOCK_DURATION:]
                    audio_block = np.squeeze(audio_block)
                    result = self.model.transcribe(audio_block, language='en', fp16=False)
                    self.transcript_queue.put(result['text'])
            except Exception as e:
                self.transcript_queue.put(f"Error: {e}")

    def update_gui(self):
        while not self.transcript_queue.empty():
            text = self.transcript_queue.get()
            self.text.insert(tk.END, text + "\n")
            self.text.see(tk.END)
        self.master.after(100, self.update_gui)

    def on_close(self):
        self.running = False
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriberApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
