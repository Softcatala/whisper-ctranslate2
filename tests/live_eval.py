"""
live_wer_eval.py
Evaluate live transcription WER using LibriSpeech (HF).
"""

import numpy as np
from datasets import load_dataset
from jiwer import wer
from typing import List

# Import your Live class
from src.whisper_ctranslate2.live import Live, BlockSize
from src.whisper_ctranslate2.commandline import CommandLine
from src.whisper_ctranslate2.transcribe import TranscriptionOptions, Transcribe
from src.whisper_ctranslate2.whisper_ctranslate2 import get_transcription_options


# -----------------------------
# Test wrapper around Live
# -----------------------------
class LiveWER(Live):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collected_text: List[str] = []

    def transcribed_text(self, text):
        print(f"**** transcribed_text: {text}")
        if text:
            self.collected_text.append(text)


# -----------------------------
# Streaming simulation
# -----------------------------
def stream_audio(live: LiveWER, audio: np.ndarray, sample_rate: int):
    block_size = int(sample_rate * BlockSize / 1000)
    for i in range(0, len(audio), block_size):
        block = audio[i : i + block_size]
        if block.ndim == 1:
            block = block[:, None]
        live.callback(block, len(block), None, None)
        live.prevblock = block  # Update prevblock for context
        live.process()

    # Send 4 seconds of silence to trigger end-of-speech detection
    silence_duration = 4.0  # seconds
    silence_samples = int(sample_rate * silence_duration)
    silence = np.zeros((silence_samples, 1), dtype=audio.dtype)

    for i in range(0, len(silence), block_size):
        block = silence[i : i + block_size]
        live.callback(block, len(block), None, None)
        live.prevblock = block
        live.process()


# -----------------------------
# Main evaluation loop
# -----------------------------
def main():
    args = CommandLine().read_command_line()
    options = get_transcription_options(args)

    dataset = load_dataset(
        "facebook/voxpopuli",
        "en",  # or other language codes
        split="test",
        streaming=True,
    )

    total_wer = []
    max_samples = 50  # adjust for speed

    live = LiveWER(
        model_path="medium",
        cache_directory="cache",
        local_files_only=False,
        task="transcribe",
        language="en",
        threads=4,
        device="cpu",
        device_index=0,
        compute_type="float32",
        verbose=False,
        threshold=0.01,
        input_device=None,
        input_device_sample_rate=16000,
        options=options,
    )

    for idx, sample in enumerate(dataset):
        if idx >= max_samples:
            break

        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        reference = sample["normalized_text"].lower().strip()

        # Reset state between samples
        live.collected_text.clear()
        live.buffer = np.zeros((0, 1))
        live.prevblock = np.zeros((0, 1))
        live.buffers_to_process.clear()
        live.speaking = False
        live.waiting = 0
        live.blocks_speaking = 0

        stream_audio(live, audio, sr)

        hypothesis = " ".join(live.collected_text).lower().strip()
        sample_wer = wer(reference, hypothesis)
        total_wer.append(sample_wer)

        print("-" * 60)
        print(f"REF: {reference}")
        print(f"HYP: {hypothesis}")
        print(f"[{idx:03d}] WER={sample_wer:.3f}")

    avg_wer = sum(total_wer) / len(total_wer)
    print(f"\nAverage WER over {len(total_wer)} samples: {avg_wer:.3f}")


if __name__ == "__main__":
    main()
