from .writers import get_writer
from typing import NamedTuple, Optional, List

try:
    from faster_whisper import WhisperModel
except ModuleNotFoundError as e:
    print("Error faster_whisper dependency not found")
    print("Install faster_whisper dependency by typing: pip install git+https://github.com/guillaumekln/faster-whisper")
    exit()

class TranscriptionOptions(NamedTuple):
    beam_size: int
    best_of: int
    patience: float
    length_penalty: float
    log_prob_threshold: Optional[float]
    no_speech_threshold: Optional[float]
    compression_ratio_threshold: Optional[float]
    condition_on_previous_text: bool
    temperatures: List[float]
    initial_prompt: Optional[str]
    #    prefix: Optional[str]
    #    suppress_blank: bool
    suppress_tokens: Optional[List[int]]
    #    without_timestamps: bool
    #    max_initial_timestamp: float
    word_timestamps: bool


#    prepend_punctuations: str
#    append_punctuations: str


class Transcribe:
    def inference(
        self,
        audio: str,
        model_path: str,
        output_dir: str,
        output_format: str,
        task: str,
        language: str,
        threads: int,
        device: str,
        compute_type: str,
        options: TranscriptionOptions,
    ):
        print(f"model_path: {model_path}")
        model = WhisperModel(
            model_path, device=device, compute_type=compute_type, cpu_threads=threads
        )

        segments, info = model.transcribe(
            audio=audio,
            language=language,
            task=task,
            beam_size=options.beam_size,
            best_of=options.best_of,
            patience=options.patience,
            length_penalty=options.length_penalty,
            temperature=options.temperatures,
            compression_ratio_threshold=options.compression_ratio_threshold,
            log_prob_threshold=options.log_prob_threshold,
            no_speech_threshold=options.no_speech_threshold,
            condition_on_previous_text=options.condition_on_previous_text,
            initial_prompt=options.initial_prompt,
            # suppress_tokens = options.suppress_tokens,
            word_timestamps=options.word_timestamps,
        )

        print(
            "Detected language '%s' with probability %f"
            % (info.language, info.language_probability)
        )

        writer = get_writer(output_format, output_dir)
        results = {}
        results["segments"] = list(segments)
        writer(results, audio)
