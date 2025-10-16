# Whisper Real-Time Transcriber with Tkinter
# Requirements: openai-whisper, sounddevice, numpy, tkinter
# Install with: pip install openai-whisper sounddevice numpy

import threading
import queue
import sounddevice as sd
import numpy as np
import whisper
import tkinter as tk
import math

# Parameters
SAMPLE_RATE = 16000
BLOCK_DURATION = 5  # seconds per transcription block
MODEL_SIZE = "base"  # Change to "small", "medium", "large" for better accuracy

class TranscriberApp:
    def __init__(self, master):
        self.master = master
        master.title("Whisper Real-Time Transcriber")

        # Main transcript area: make it expand and center text
        self.text = tk.Text(master, height=20, width=80, font=("Arial", 16), wrap='word')
        self.text.pack(expand=True, fill=tk.BOTH)
        # Configure a tag for centered text
        self.text.tag_configure("center", justify='center')
        # Insert initial line with center tag
        self.text.insert(tk.END, "Listening...\n", "center")

        # Volume meter setup
        self.meter_canvas = tk.Canvas(master, height=36, bg="#222222")
        self.meter_canvas.pack(fill=tk.X, side=tk.BOTTOM)
        # We'll compute sizes dynamically from canvas width
        self.meter_width = None
        self.meter_height = 36
        # Draw meter background and initial bar (coords updated later)
        self.meter_bg = self.meter_canvas.create_rectangle(2, 2, 10, self.meter_height-2, fill="#444444", outline="#555555")
        self.meter_bar = self.meter_canvas.create_rectangle(4, 4, 4, self.meter_height-4, fill="#00ff00", outline="")
        # Peak hold indicator (x position)
        self.peak_x = 4
        self.peak_line = self.meter_canvas.create_line(self.peak_x, 2, self.peak_x, self.meter_height-2, fill="#ffffff")
        # dB numeric display
        self.db_label = tk.Label(master, text="-inf dB", bg="#222222", fg="#ffffff")
        self.db_label.place(relx=1.0, rely=1.0, x=-8, y=-8, anchor="se")
        self.volume_queue = queue.Queue()
        # EMA smoothing state
        self.ema_level = 0.0
        self.ema_alpha = 0.25  # smoothing factor (0..1), larger => less smoothing
        # Peak hold state
        self.peak_level = 0.0
        self.peak_decay = 0.02  # decay per update

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
            data = indata.copy()
            self.audio_queue.put(data)
            # compute RMS for volume meter and put into queue (non-blocking)
            try:
                rms = float(np.sqrt(np.mean(np.square(data.astype('float64')))))
                # convert rms to db-like 0..1 scale
                db = 20 * math.log10(rms + 1e-10)
                # map db range (-80..0) to 0..1
                level = min(max((db + 80) / 80, 0.0), 1.0)
            except Exception:
                level = 0.0
            # keep only the latest value to avoid backlog
            if not self.volume_queue.empty():
                try:
                    _ = self.volume_queue.get_nowait()
                except Exception:
                    pass
            self.volume_queue.put(level)
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
            # Insert new transcript line with the 'center' tag so lines are horizontally centered
            self.text.insert(tk.END, text + "\n", "center")
            # Keep view at the end so new lines push older lines upward (scroll up effect)
            self.text.see(tk.END)
        # Update volume meter if we have a recent level
        try:
            level = None
            while not self.volume_queue.empty():
                level = self.volume_queue.get_nowait()
        except Exception:
            level = None
        if level is not None:
            # Apply EMA smoothing
            self.ema_level = (self.ema_alpha * level) + ((1 - self.ema_alpha) * self.ema_level)
            display_level = self.ema_level
            # Update peak hold
            if display_level > self.peak_level:
                self.peak_level = display_level
            else:
                # decay peak
                self.peak_level = max(0.0, self.peak_level - self.peak_decay)

            # get canvas width (dynamic)
            canvas_width = self.meter_canvas.winfo_width()
            if canvas_width <= 1:
                # sometimes width not ready; fall back to 400
                canvas_width = 400
            self.meter_width = canvas_width
            # compute bar width based on level
            width = 4 + int((self.meter_width - 16) * display_level)
            # color gradient from green to red
            r = int(255 * min(max((display_level - 0.6) / 0.4, 0.0), 1.0))
            g = int(255 * min(max((0.6 - display_level) / 0.6, 0.0), 1.0))
            color = f"#{r:02x}{g:02x}00"
            # update bar
            self.meter_canvas.coords(self.meter_bar, 4, 4, width, self.meter_height-4)
            self.meter_canvas.itemconfig(self.meter_bar, fill=color)

            # Update peak line position
            peak_x = 4 + int((self.meter_width - 16) * self.peak_level)
            self.meter_canvas.coords(self.peak_line, peak_x, 2, peak_x, self.meter_height-2)

            # Update numeric dB label using the raw level -> db mapping
            # reverse map: level 0..1 -> db -80..0
            db_value = -80.0 + (display_level * 80.0)
            if display_level <= 0.0:
                db_text = "-inf dB"
            else:
                db_text = f"{db_value:.1f} dB"
            self.db_label.config(text=db_text)

        self.master.after(100, self.update_gui)

    def on_close(self):
        self.running = False
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriberApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
