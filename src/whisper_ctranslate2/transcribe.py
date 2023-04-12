from .writers import format_timestamp
from typing import NamedTuple, Optional, List, Union
import tqdm
import sys
from faster_whisper import WhisperModel
from .languages import LANGUAGES
from typing import BinaryIO
import numpy as np


system_encoding = sys.getdefaultencoding()

if system_encoding != "utf-8":

    def make_safe(string):
        return string.encode(system_encoding, errors="replace").decode(system_encoding)

else:

    def make_safe(string):
        return string


class TranscriptionOptions(NamedTuple):
    beam_size: int
    best_of: int
    patience: float
    length_penalty: float
    log_prob_threshold: Optional[float]
    no_speech_threshold: Optional[float]
    compression_ratio_threshold: Optional[float]
    condition_on_previous_text: bool
    temperature: List[float]
    initial_prompt: Optional[str]
    #    prefix: Optional[str]
    #    suppress_blank: bool
    suppress_tokens: Optional[List[int]]
    #    without_timestamps: bool
    #    max_initial_timestamp: float
    word_timestamps: bool
    print_colors: bool
    prepend_punctuations: str
    append_punctuations: str
    vad_filter: bool
    vad_min_silence_duration_ms: Optional[int]


class Transcribe:
    def _get_colored_text(self, words):
        k_colors = [
            "\033[38;5;196m",
            "\033[38;5;202m",
            "\033[38;5;208m",
            "\033[38;5;214m",
            "\033[38;5;220m",
            "\033[38;5;226m",
            "\033[38;5;190m",
            "\033[38;5;154m",
            "\033[38;5;118m",
            "\033[38;5;82m",
        ]

        text_words = ""

        n_colors = len(k_colors)
        for word in words:
            p = word.probability
            col = max(0, min(n_colors - 1, (int)(pow(p, 3) * n_colors)))
            end_mark = "\033[0m"
            text_words += f"{k_colors[col]}{word.word}{end_mark}"

        return text_words

    def inference(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        model_path: str,
        task: str,
        language: str,
        threads: int,
        device: str,
        device_index: Union[int, List[int]],
        compute_type: str,
        verbose: bool,
        live: bool,
        options: TranscriptionOptions,
    ):
        model = WhisperModel(
            model_path,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            cpu_threads=threads,
        )

        segments, info = model.transcribe(
            audio=audio,
            language=language,
            task=task,
            beam_size=options.beam_size,
            best_of=options.best_of,
            patience=options.patience,
            length_penalty=options.length_penalty,
            temperature=options.temperature,
            compression_ratio_threshold=options.compression_ratio_threshold,
            log_prob_threshold=options.log_prob_threshold,
            no_speech_threshold=options.no_speech_threshold,
            condition_on_previous_text=options.condition_on_previous_text,
            initial_prompt=options.initial_prompt,
            suppress_tokens=options.suppress_tokens,
            word_timestamps=True if options.print_colors else options.word_timestamps,
            prepend_punctuations=options.prepend_punctuations,
            append_punctuations=options.append_punctuations,
            vad_filter=options.vad_filter,
            vad_parameters={
                "min_silence_duration_ms": options.vad_min_silence_duration_ms
            },
        )

        language_name = LANGUAGES[info.language].title()
        if not live:
            print(
                "Detected language '%s' with probability %f"
                % (language_name, info.language_probability)
            )

        list_segments = []
        last_pos = 0
        accumated_inc = 0
        all_text = ""
        with tqdm.tqdm(
            total=info.duration, unit="seconds", disable=verbose or live is not False
        ) as pbar:
            for segment in segments:
                start, end, text = segment.start, segment.end, segment.text
                all_text += segment.text

                if verbose or options.print_colors:
                    if options.print_colors and segment.words:
                        text = self._get_colored_text(segment.words)
                    else:
                        text = segment.text

                    if not live:
                        line = f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"
                        print(make_safe(line))

                segment_dict = segment._asdict()
                if segment.words:
                    segment_dict["words"] = [word._asdict() for word in segment.words]

                list_segments.append(segment_dict)
                duration = segment.end - last_pos
                increment = (
                    duration
                    if accumated_inc + duration < info.duration
                    else info.duration - accumated_inc
                )
                accumated_inc += increment
                last_pos = segment.end
                pbar.update(increment)

        return dict(
            text=all_text,
            segments=list_segments,
            language=language_name,
        )
