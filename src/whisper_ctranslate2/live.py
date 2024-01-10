# Based on code from https://github.com/Nikorasu/LiveWhisper/blob/main/livewhisper.py

import numpy as np
from .transcribe import Transcribe, TranscriptionOptions
from typing import Union, List

SampleRate = 16000  # Stream device recording frequency per second
BlockSize = 30  # Block size in milliseconds
Vocals = [50, 1000]  # Frequency range to detect sounds that could be speech
EndBlocks = 33 * 2  # Number of blocks to wait before sending (30 ms is block)
FlushBlocks = 33 * 10  # Number of blocks to wait before sending

try:
    import sounddevice as sd

    sounddevice_available = True

except Exception as e:
    sounddevice_available = False
    sounddevice_exception = e


class Live:
    def __init__(
        self,
        model_path: str,
        cache_directory: str,
        local_files_only: bool,
        task: str,
        language: str,
        threads: int,
        device: str,
        device_index: Union[int, List[int]],
        compute_type: str,
        verbose: bool,
        threshold: float,
        input_device: int,
        options: TranscriptionOptions,
    ):
        self.model_path = model_path
        self.cache_directory = cache_directory
        self.local_files_only = local_files_only
        self.task = task
        self.language = language
        self.threads = threads
        self.device = device
        self.device_index = device_index
        self.compute_type = compute_type
        self.verbose = verbose
        self.threshold = threshold
        self.input_device = input_device
        self.options = options

        self.running = True
        self.waiting = 0
        self.prevblock = self.buffer = np.zeros((0, 1))
        self.speaking = False
        self.blocks_speaking = 0
        self.buffers_to_process = []
        self.transcribe = None

    @staticmethod
    def is_available():
        return sounddevice_available

    @staticmethod
    def force_not_available_exception():
        raise (sounddevice_exception)

    def _is_there_voice(self, indata, frames):
        freq = np.argmax(np.abs(np.fft.rfft(indata[:, 0]))) * SampleRate / frames
        volume = np.sqrt(np.mean(indata**2))

        return volume > self.threshold and Vocals[0] <= freq <= Vocals[1]

    def _save_to_process(self):
        self.buffers_to_process.append(self.buffer.copy())
        self.buffer = np.zeros((0, 1))
        self.speaking = False

    def callback(self, indata, frames, _time, status):
        if not any(indata):
            return

        voice = self._is_there_voice(indata, frames)

        # Silence and no nobody has started speaking
        if not voice and not self.speaking:
            return

        if voice:  # User speaking
            if self.verbose:
                print(".", end="", flush=True)
            if self.waiting < 1:
                self.buffer = self.prevblock.copy()

            self.buffer = np.concatenate((self.buffer, indata))
            self.waiting = EndBlocks

            if not self.speaking:
                self.blocks_speaking = FlushBlocks

            self.speaking = True
        else:  # Silence after user has spoken
            self.waiting -= 1
            if self.waiting < 1:
                self._save_to_process()
                return
            else:
                self.buffer = np.concatenate((self.buffer, indata))

        self.blocks_speaking -= 1
        # User spoken for a long time and we need to flush
        if self.blocks_speaking < 1:
            self._save_to_process()

    def process(self):
        if len(self.buffers_to_process) > 0:
            _buffer = self.buffers_to_process.pop()
            if self.verbose:
                print("\n\033[90mTranscribing..\033[0m")

            if not self.transcribe:
                self.transcribe = Transcribe(
                    self.model_path,
                    self.device,
                    self.device_index,
                    self.compute_type,
                    self.threads,
                    self.cache_directory,
                    self.local_files_only,
                )

            result = self.transcribe.inference(
                audio=_buffer.flatten().astype("float32"),
                task=self.task,
                language=self.language,
                verbose=self.verbose,
                live=True,
                options=self.options,
            )
            print(f"\033[1A\033[2K\033[0G{result['text']}")
            if not self.verbose:
                print("")

    def listen(self):
        show_device = (
            self.input_device if self.input_device is not None else sd.default.device[0]
        )
        print(
            f"\033[32mLive stream device: \033[37m{sd.query_devices(device=show_device)['name']}\033[0m"
        )
        print("\033[32mListening.. \033[37m(Ctrl+C to Quit)\033[0m")
        with sd.InputStream(
            channels=1,
            callback=self.callback,
            blocksize=int(SampleRate * BlockSize / 1000),
            samplerate=SampleRate,
            device=self.input_device,
        ):
            while self.running:
                self.process()

    def inference(self):
        try:
            self.listen()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            print("\n\033[93mQuitting..\033[0m")
