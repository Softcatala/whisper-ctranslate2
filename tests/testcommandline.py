import io
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from tempfile import TemporaryDirectory
from unittest.mock import patch

from whisper_ctranslate2.whisper_ctranslate2 import main


class TestCommandLine(unittest.TestCase):
    def test_output_dir_controls_file_creation(self):
        result = {
            "text": "Hello",
            "segments": [{"start": 0, "end": 1, "text": "Hello", "words": []}],
        }
        formats = ("txt", "vtt", "srt", "tsv", "json")
        for output_format in (None, *formats, "all"):
            with (
                self.subTest(output_format=output_format),
                TemporaryDirectory() as directory,
            ):
                previous_directory = os.getcwd()
                try:
                    os.chdir(directory)
                    for output_dir in ("", "output"):
                        with self.subTest(output_dir=output_dir):
                            argv = [
                                "whisper-ctranslate2",
                                "audio.mp3",
                                "--output_dir",
                                output_dir,
                            ]
                            if output_format is not None:
                                argv.extend(["--output_format", output_format])
                            stderr = io.StringIO()
                            with (
                                patch("sys.argv", argv),
                                patch(
                                    "whisper_ctranslate2.whisper_ctranslate2.Transcribe"
                                ) as transcribe,
                                redirect_stdout(io.StringIO()),
                                redirect_stderr(stderr),
                            ):
                                transcribe.return_value.inference.return_value = result
                                main()

                            transcribe.return_value.inference.assert_called_once()
                            self.assertEqual("", stderr.getvalue())
                            if not output_dir:
                                self.assertEqual([], os.listdir(directory))
                            else:
                                self.assertEqual(["output"], os.listdir(directory))
                                expected_formats = (
                                    formats
                                    if output_format in (None, "all")
                                    else (output_format,)
                                )
                                self.assertCountEqual(
                                    [f"audio.{fmt}" for fmt in expected_formats],
                                    os.listdir(output_dir),
                                )
                                for filename in os.listdir(output_dir):
                                    with open(
                                        os.path.join(output_dir, filename),
                                        encoding="utf-8",
                                    ) as output_file:
                                        self.assertIn("Hello", output_file.read())
                finally:
                    os.chdir(previous_directory)


if __name__ == "__main__":
    unittest.main()
