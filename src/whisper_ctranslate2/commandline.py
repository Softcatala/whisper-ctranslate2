import argparse
from .languages import LANGUAGES, TO_LANGUAGE_CODE
from .version import __version__

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
    "large-v3",
    "large-v3-turbo",
    "distil-large-v2",
    "distil-large-v3",
    "distil-medium.en",
    "distil-small.en",
]


class CommandLine:
    @staticmethod
    def _optional_int(string):
        return None if string == "None" else int(string)

    @staticmethod
    def _str2bool(string):
        str2val = {"true": True, "false": False}
        if string and string.lower() in str2val:
            return str2val[string.lower()]
        else:
            raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")

    @staticmethod
    def _optional_float(string):
        return None if string == "None" else float(string)

    @staticmethod
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
            help="directory where to find a CTranslate2 Whisper model (e.g. fine-tuned model)",
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
            type=CommandLine()._str2bool,
            default=False,
            help="use only models in cache without connecting to Internet to check if there are newer versions",
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
            type=CommandLine()._str2bool,
            default=False,
            help="produce json in a human readable format",
        )

        outputs_args.add_argument(
            "--print_colors",
            type=CommandLine()._str2bool,
            default=False,
            help="print the transcribed text using an experimental color coding strategy to highlight words with high or low confidence",
        )

        outputs_args.add_argument(
            "--verbose",
            type=CommandLine()._str2bool,
            default=True,
            help="whether to print out the progress and debug messages",
        )

        outputs_args.add_argument(
            "--highlight_words",
            type=CommandLine()._str2bool,
            default=False,
            help="underline each word as it is spoken in srt and vtt output formats (requires --word_timestamps True)",
        )

        outputs_args.add_argument(
            "--max_line_width",
            type=CommandLine()._optional_int,
            default=None,
            help="the maximum number of characters in a line before breaking the line in srt and vtt output formats (requires --word_timestamps True)",
        )
        outputs_args.add_argument(
            "--max_line_count",
            type=CommandLine()._optional_int,
            default=None,
            help="the maximum number of lines in a segment in srt and vtt output formats (requires --word_timestamps True)",
        )
        outputs_args.add_argument(
            "--max_words_per_line",
            type=CommandLine()._optional_int,
            default=None,
            help="(requires --word_timestamps True, no effect with --max_line_width) the maximum number of words in a segment",
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
            type=CommandLine()._optional_int,
            default=0,
            help="number of threads used for CPU inference",
        )

        computing_args.add_argument(
            "--device_index",
            type=int,
            default=0,
            help="device ID where to place this model on",
        )

        computing_args.add_argument(
            "--compute_type",
            choices=[
                "default",
                "auto",
                "int8",
                "int8_float16",
                "int8_bfloat16",
                "int8_float32",
                "int16",
                "float16",
                "float32",
                "bfloat16",
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
            "--temperature",
            type=float,
            default=0,
            help="temperature to use for sampling",
        )

        algorithm_args.add_argument(
            "--temperature_increment_on_fallback",
            type=CommandLine()._optional_float,
            default=0.2,
            help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below",
        )

        algorithm_args.add_argument(
            "--prompt_reset_on_temperature",
            type=float,
            default=0.5,
            help="resets prompt if temperature is above this value. Arg has effect only if condition_on_previous_text is True",
        )

        algorithm_args.add_argument(
            "--prefix",
            type=str,
            default=None,
            help="optional text to provide as a prefix for the first window",
        )

        algorithm_args.add_argument(
            "--best_of",
            type=CommandLine()._optional_int,
            default=5,
            help="number of candidates when sampling with non-zero temperature",
        )
        algorithm_args.add_argument(
            "--beam_size",
            type=CommandLine()._optional_int,
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
            "--suppress_blank",
            type=CommandLine()._str2bool,
            default="True",
            help="suppress blank outputs at the beginning of the sampling",
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
            type=CommandLine()._str2bool,
            default=True,
            help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop",
        )
        algorithm_args.add_argument(
            "--compression_ratio_threshold",
            type=CommandLine()._optional_float,
            default=2.4,
            help="if the gzip compression ratio is higher than this value, treat the decoding as failed",
        )
        algorithm_args.add_argument(
            "--logprob_threshold",
            type=CommandLine()._optional_float,
            default=-1.0,
            help="if the average log probability is lower than this value, treat the decoding as failed",
        )
        algorithm_args.add_argument(
            "--no_speech_threshold",
            type=CommandLine()._optional_float,
            default=0.6,
            help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence",
        )
        algorithm_args.add_argument(
            "--word_timestamps",
            type=CommandLine()._str2bool,
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
            help="penalty applied to the score of previously generated tokens (set > 1 to penalize)",
        )
        algorithm_args.add_argument(
            "--no_repeat_ngram_size",
            type=int,
            default=0,
            help="prevent repetitions of ngrams with this size (set 0 to disable)",
        )
        algorithm_args.add_argument(
            "--hallucination_silence_threshold",
            type=CommandLine()._optional_float,
            default=None,
            help="When word_timestamps is True, skip silent periods longer than this threshold (in seconds) when a possible hallucination is detected",
        )
        algorithm_args.add_argument(
            "--hotwords",
            type=str,
            default=None,
            help="Hotwords/hint phrases to the model. Useful for names you want the model to priotize",
        )

        vad_args = parser.add_argument_group("VAD filter arguments")

        vad_args.add_argument(
            "--vad_filter",
            type=CommandLine()._str2bool,
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

        diarization_args = parser.add_argument_group("Diarization options")
        diarization_args.add_argument(
            "--hf_token",
            type=str,
            default="",
            help="HuggingFace token which enables to download the diarization models.",
        )

        diarization_args.add_argument(
            "--speaker_name",
            type=str,
            default="SPEAKER",
            help="Name to use to identify the speaker (e.g. SPEAKER_00).",
        )

        live_args = parser.add_argument_group("Live transcribe options")

        live_args.add_argument(
            "--live_transcribe",
            type=CommandLine()._str2bool,
            default=False,
            help="live transcribe mode",
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
            help="Set live stream input device ID (see python -m sounddevice for a list)",
        )

        return parser.parse_args().__dict__
