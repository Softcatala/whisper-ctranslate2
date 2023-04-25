import tempfile
import unittest
import os
import json


class TestCmd(unittest.TestCase):
    def _remove_fields_from_json(self, json_file):
        _dict = json.loads(json_file)
        for segment in _dict["segments"]:
            for del_item in [
                "id",
                "seek",
                "tokens",
                "temperature",
                "avg_log_prob",
                "avg_logprob",
                "compression_ratio",
                "no_speech_prob",
            ]:
                if del_item in segment:
                    del segment[del_item]

            if segment["words"]:
                for word in segment["words"]:
                    del word["probability"]

        return str(_dict)

    def _check_ref_small(self, hyp_dir, _file, ref_dir, option):
        extensions = [".txt", ".json", ".srt", ".tsv", ".vtt"]

        self.maxDiff = None
        for ext in extensions:
            hyp = os.path.join(hyp_dir, _file) + ext
            ref = os.path.join(ref_dir, _file) + ext

            with open(ref, "r") as fh_r, open(hyp, "r") as fh_h:
                r = fh_r.read()
                h = fh_h.read()

                if ext == ".json":
                    r = self._remove_fields_from_json(r)
                    h = self._remove_fields_from_json(h)
                self.assertEqual(r, h, f"failed {ext} on {option}")

    def test_options_transcribe(self):
        full_path = os.path.realpath(__file__)
        path, filename = os.path.split(full_path)

        options = [
            "",
            "--model_dir tmp/",
            "--verbose False",
            "--verbose True",
            "--print_colors False",
            "--threads 4",
        ]

        for option in options:
            with tempfile.TemporaryDirectory() as directory:
                _file = "gossos"
                cmd = f"cd {directory} && whisper-ctranslate2 {path}/{_file}.mp3 --device cpu --compute_type float32 {option}"
                os.system(cmd)
                self._check_ref_small(
                    f"{directory}", _file, "e2e-tests/ref-small-transcribe/", option
                )

    def test_options_transcribe_timestamps(self):
        full_path = os.path.realpath(__file__)
        path, filename = os.path.split(full_path)

        options = [
            "--word_timestamps True",
            "--print_colors True",
        ]

        for option in options:
            with tempfile.TemporaryDirectory() as directory:
                _file = "gossos"
                cmd = f"cd {directory} && whisper-ctranslate2 {path}/{_file}.mp3 --device cpu --compute_type float32 {option}"
                os.system(cmd)
                self._check_ref_small(
                    f"{directory}",
                    _file,
                    "e2e-tests/ref-small-transcribe-word-stamps/",
                    option,
                )

    def test_options_translate(self):
        full_path = os.path.realpath(__file__)
        path, filename = os.path.split(full_path)

        options = [
            "",
            "--model_dir tmp/",
            "--verbose False",
            "--verbose True",
            "--print_colors False",
        ]

        for option in options:
            with tempfile.TemporaryDirectory() as directory:
                _file = "gossos"
                cmd = f"cd {directory} && whisper-ctranslate2 {path}/{_file}.mp3 --device cpu --task translate --model medium --compute_type float32 {option}"
                os.system(cmd)
                self._check_ref_small(
                    f"{directory}", _file, "e2e-tests/ref-medium-translate/", option
                )


if __name__ == "__main__":
    unittest.main()
