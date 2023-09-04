import argparse
import os
from .transcribe import Transcribe, TranscriptionOptions
from .languages import LANGUAGES, TO_LANGUAGE_CODE, from_language_to_iso_code
import numpy as np
import warnings
from typing import Union, List
from .writers import get_writer
from .version import __version__
from .live import Live
import sys

MODEL_NAMES = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
]


def optional_int(string):
    return None if string == "None" else int(string)


def str2bool(string):
    str2val = {"true": True, "false": False}
    if string and string.lower() in str2val:
        return str2val[string.lower()]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")


def optional_float(string):
    return None if string == "None" else float(string)


def read_command_line():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "audio", nargs="*", type=str, help="audio file(s) to transcribe"
    )

    model_args = parser.add_argument_group("Model selection options")

    model_args.add_argument(
        "--model",
        default="small",
        choices=MODEL_NAMES,
        help="name of the Whisper model to use",
    )

    model_args.add_argument(
        "--model_directory",
        type=str,
        default=None,
        help="directory where to find a CTranslate Whisper model (e.g. fine-tuned model)",
    )

    caching_args = parser.add_argument_group("Model caching control options")

    caching_args.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="the path to save model files; uses ~/.cache/huggingface/ by default",
    )

    caching_args.add_argument(
        "--local_files_only",
        type=str2bool,
        default=False,
        help="use models in cache without connecting to Internet to check if there are newer versions",
    )

    outputs_args = parser.add_argument_group(
        "Configuration options to control generated outputs"
    )

    outputs_args.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=".",
        help="directory to save the outputs",
    )
    outputs_args.add_argument(
        "--output_format",
        "-f",
        type=str,
        default="all",
        choices=["txt", "vtt", "srt", "tsv", "json", "all"],
        help="format of the output file; if not specified, all available formats will be produced",
    )

    outputs_args.add_argument(
        "--pretty_json",
        "-p",
        type=str2bool,
        default=False,
        help="produce json in a human redable format",
    )

    outputs_args.add_argument(
        "--print_colors",
        type=str2bool,
        default=False,
        help="print the transcribed text using an experimental color coding strategy to highlight words with high or low confidence",
    )

    outputs_args.add_argument(
        "--verbose",
        type=str2bool,
        default=True,
        help="whether to print out the progress and debug messages",
    )

    outputs_args.add_argument(
        "--highlight_words",
        type=str2bool,
        default=False,
        help="underline each word as it is spoken in srt and vtt output formats (requires --word_timestamps True)",
    )

    outputs_args.add_argument(
        "--max_line_width",
        type=optional_int,
        default=None,
        help="the maximum number of characters in a line before breaking the line in srt and vtt output formats (requires --word_timestamps True)",
    )
    outputs_args.add_argument(
        "--max_line_count",
        type=optional_int,
        default=None,
        help="the maximum number of lines in a segment in srt and vtt output formats (requires --word_timestamps True)",
    )

    computing_args = parser.add_argument_group("Computing configuration options")

    computing_args.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
        help="device to use for CTranslate2 inference",
    )

    computing_args.add_argument(
        "--threads",
        type=optional_int,
        default=0,
        help="number of threads used for CPU inference",
    )

    computing_args.add_argument(
        "--device_index",
        nargs="*",
        type=int,
        default=0,
        help="device IDs where to place this model on",
    )

    computing_args.add_argument(
        "--compute_type",
        choices=[
            "default",
            "auto",
            "int8",
            "int8_float16",
            "int16",
            "float16",
            "float32",
        ],
        default="auto",
        help="Type of quantization to use (see https://opennmt.net/CTranslate2/quantization.html)",
    )

    algorithm_args = parser.add_argument_group("Algorithm execution options")

    algorithm_args.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')",
    )
    algorithm_args.add_argument(
        "--language",
        type=str,
        default=None,
        choices=sorted(LANGUAGES.keys())
        + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
        help="language spoken in the audio, specify None to perform language detection",
    )

    algorithm_args.add_argument(
        "--temperature", type=float, default=0, help="temperature to use for sampling"
    )

    algorithm_args.add_argument(
        "--temperature_increment_on_fallback",
        type=optional_float,
        default=0.2,
        help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below",
    )

    algorithm_args.add_argument(
        "--prompt_reset_on_temperature",
        type=int,
        default=0,
        help="resets prompt if temperature is above this value. Arg has effect only if condition_on_previous_text is True",
    )

    algorithm_args.add_argument(
        "--best_of",
        type=optional_int,
        default=5,
        help="number of candidates when sampling with non-zero temperature",
    )
    algorithm_args.add_argument(
        "--beam_size",
        type=optional_int,
        default=5,
        help="number of beams in beam search, only applicable when temperature is zero",
    )
    algorithm_args.add_argument(
        "--patience",
        type=float,
        default=1.0,
        help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search",
    )
    algorithm_args.add_argument(
        "--length_penalty",
        type=float,
        default=1.0,
        help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default",
    )

    algorithm_args.add_argument(
        "--suppress_tokens",
        type=str,
        default="-1",
        help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations",
    )
    algorithm_args.add_argument(
        "--initial_prompt",
        type=str,
        default=None,
        help="optional text to provide as a prompt for the first window.",
    )
    algorithm_args.add_argument(
        "--condition_on_previous_text",
        type=str2bool,
        default=True,
        help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop",
    )
    algorithm_args.add_argument(
        "--compression_ratio_threshold",
        type=optional_float,
        default=2.4,
        help="if the gzip compression ratio is higher than this value, treat the decoding as failed",
    )
    algorithm_args.add_argument(
        "--logprob_threshold",
        type=optional_float,
        default=-1.0,
        help="if the average log probability is lower than this value, treat the decoding as failed",
    )
    algorithm_args.add_argument(
        "--no_speech_threshold",
        type=optional_float,
        default=0.6,
        help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence",
    )
    algorithm_args.add_argument(
        "--word_timestamps",
        type=str2bool,
        default=False,
        help="(experimental) extract word-level timestamps and refine the results based on them",
    )

    algorithm_args.add_argument(
        "--prepend_punctuations",
        type=str,
        default="\"'“¿([{-",
        help="if word_timestamps is True, merge these punctuation symbols with the next word",
    )
    algorithm_args.add_argument(
        "--append_punctuations",
        type=str,
        default="\"'.。,，!！?？:：”)]}、",
        help="if word_timestamps is True, merge these punctuation symbols with the previous word",
    )
    algorithm_args.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="prevent repetitions of ngrams with this size (set 0 to disable)",
    )
    algorithm_args.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=0,
        help="Penalty applied to the score of previously generated tokens (set > 1 to penalize)",
    )
    vad_args = parser.add_argument_group("VAD filter arguments")

    vad_args.add_argument(
        "--vad_filter",
        type=str2bool,
        default=False,
        help="enable the voice activity detection (VAD) to filter out parts of the audio without speech. This step is using the Silero VAD model https://github.com/snakers4/silero-vad.",
    )

    vad_args.add_argument(
        "--vad_threshold",
        type=float,
        default=None,
        help="when `vad_filter` is enabled, probabilities above this value are considered as speech.",
    )

    vad_args.add_argument(
        "--vad_min_speech_duration_ms",
        type=int,
        default=None,
        help="when `vad_filter` is enabled, final speech chunks shorter min_speech_duration_ms are thrown out.",
    )

    vad_args.add_argument(
        "--vad_max_speech_duration_s",
        type=int,
        default=None,
        help="when `vad_filter` is enabled, Maximum duration of speech chunks in seconds. Longer will be split at the timestamp of the last silence.",
    )

    vad_args.add_argument(
        "--vad_min_silence_duration_ms",
        type=int,
        default=None,
        help="when `vad_filter` is enabled, in the end of each speech chunk time to wait before separating it.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
        help="show program's version number and exit",
    )

    live_args = parser.add_argument_group("Live transcribe options")

    live_args.add_argument(
        "--live_transcribe", type=str2bool, default=False, help="live transcribe mode"
    )

    live_args.add_argument(
        "--live_volume_threshold",
        type=float,
        default=0.2,
        help="minimum volume threshold to activate listening in live transcribe mode",
    )

    live_args.add_argument(
        "--live_input_device",
        type=int,
        default=None,
        help="Set live stream input device (python -m sounddevice)",
    )

    return parser.parse_args().__dict__


def _does_old_cache_dir_has_files():
    default = os.path.join(os.path.expanduser("~"), ".cache")
    cache_dir = os.path.join(
        os.getenv("XDG_CACHE_HOME", default), "whisper-ctranslate2"
    )
    return cache_dir, os.path.exists(cache_dir)


def main():
    args = read_command_line()
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    os.makedirs(output_dir, exist_ok=True)
    model: str = args.pop("model")
    threads: int = args.pop("threads")
    language: str = args.pop("language")
    task: str = args.pop("task")
    device: str = args.pop("device")
    compute_type: str = args.pop("compute_type")
    verbose: bool = args.pop("verbose")
    model_directory: str = args.pop("model_directory")
    cache_directory: str = args.pop("model_dir")
    device_index: Union[int, List[int]] = args.pop("device_index")
    suppress_tokens: str = args.pop("suppress_tokens")
    live_transcribe: bool = args.pop("live_transcribe")
    audio: str = args.pop("audio")
    local_files_only: bool = args.pop("local_files_only")
    live_volume_threshold: float = args.pop("live_volume_threshold")
    live_input_device: int = args.pop("live_input_device")
    temperature = args.pop("temperature")

    if (increment := args.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    language = from_language_to_iso_code(language)

    if (
        not model_directory
        and model.endswith(".en")
        and language not in {"en", "English"}
    ):
        if language is not None:
            warnings.warn(
                f"{model} is an English-only model but receipted '{language}'; using English instead."
            )
        language = "en"

    suppress_tokens = [int(t) for t in suppress_tokens.split(",")]
    options = TranscriptionOptions(
        beam_size=args.pop("beam_size"),
        best_of=args.pop("best_of"),
        patience=args.pop("patience"),
        length_penalty=args.pop("length_penalty"),
        repetition_penalty=args.pop("repetition_penalty"),
        no_repeat_ngram_size=args.pop("no_repeat_ngram_size"),
        log_prob_threshold=args.pop("logprob_threshold"),
        no_speech_threshold=args.pop("no_speech_threshold"),
        compression_ratio_threshold=args.pop("compression_ratio_threshold"),
        condition_on_previous_text=args.pop("condition_on_previous_text"),
        temperature=temperature,
        prompt_reset_on_temperature=args.pop("prompt_reset_on_temperature"),
        initial_prompt=args.pop("initial_prompt"),
        suppress_tokens=suppress_tokens,
        word_timestamps=args.pop("word_timestamps"),
        prepend_punctuations=args.pop("prepend_punctuations"),
        append_punctuations=args.pop("append_punctuations"),
        print_colors=args.pop("print_colors"),
        vad_filter=args.pop("vad_filter"),
        vad_threshold=args.pop("vad_threshold"),
        vad_min_speech_duration_ms=args.pop("vad_min_speech_duration_ms"),
        vad_max_speech_duration_s=args.pop("vad_max_speech_duration_s"),
        vad_min_silence_duration_ms=args.pop("vad_min_silence_duration_ms"),
    )

    if not live_transcribe and len(audio) == 0:
        sys.stderr.write("You need to specify one or more audio files\n")
        sys.stderr.write(
            "Use `whisper-ctranslate2 --help` to see the available options.\n"
        )
        return

    word_options = ["highlight_words", "max_line_count", "max_line_width"]
    if not options.word_timestamps:
        for option in word_options:
            if args[option]:
                sys.stderr.write(f"--{option} requires --word_timestamps True\n")
                return

    if args["max_line_count"] and not args["max_line_width"]:
        warnings.warn("--max_line_count has no effect without --max_line_width")

    writer_options = list(word_options)
    writer_options.append("pretty_json")
    writer_args = {arg: args.pop(arg) for arg in writer_options}

    if verbose:
        cache_dir, exists = _does_old_cache_dir_has_files()
        if exists:
            print(
                f"There are old cache files at `{cache_dir}` which are no longer used. Consider deleting them"
            )

    if not verbose and options.print_colors:
        sys.stderr.write("You cannot disable verbose and enable print colors\n")
        return

    if live_transcribe and not Live.is_available():
        Live.force_not_available_exception()

    if verbose and not language:
        if live_transcribe:
            print(
                "Consider specifying the language using `--language`. It improves significantly prediction in live transcription."
            )
        else:
            print(
                "Detecting language using up to the first 30 seconds. Use `--language` to specify the language"
            )

    if options.print_colors and output_dir and not options.word_timestamps:
        print(
            "Print colors requires word-level time stamps. Generated files in output directory will have word-level timestamps"
        )

    output_dir = os.path.abspath(output_dir)
    if model_directory:
        model_filename = os.path.join(model_directory, "model.bin")
        if not os.path.exists(model_filename):
            raise RuntimeError(f"Model file '{model_filename}' does not exists")
        model_dir = model_directory
    else:
        model_dir = model

    if live_transcribe:
        Live(
            model_dir,
            cache_directory,
            local_files_only,
            task,
            language,
            threads,
            device,
            device_index,
            compute_type,
            verbose,
            live_volume_threshold,
            live_input_device,
            options,
        ).inference()

        return

    transcribe = Transcribe(
        model_dir,
        device,
        device_index,
        compute_type,
        threads,
        cache_directory,
        local_files_only,
    )

    for audio_path in audio:
        result = transcribe.inference(
            audio_path,
            task,
            language,
            verbose,
            False,
            options,
        )
        writer = get_writer(output_format, output_dir)
        writer(result, audio_path, writer_args)

    if verbose:
        print(f"Transcription results written to '{output_dir}' directory")


if __name__ == "__main__":
    main()
