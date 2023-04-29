import unittest
import os
from src.whisper_ctranslate2.writers import WriteTXT, WriteSRT, WriteTSV, WriteVTT
from tempfile import NamedTemporaryFile
from faster_whisper.transcribe import Segment, Word


class TestCmd(unittest.TestCase):
    def _get_segment(self, text, start=0, end=1):
        return Segment(
            start=start,
            end=end,
            text=text,
            words=[],
            avg_logprob=0,
            no_speech_prob=0,
            id=0,
            seek=0,
            tokens=[],
            temperature=0,
            compression_ratio=0,
        )._asdict()

    def _read_subtitles(self, filename):
        with open(filename, "r") as fh_r:
            return fh_r.readlines()

    def _get_temp_file_name_dir(self):
        f = NamedTemporaryFile()
        return f.name, os.path.dirname(f.name)

    def test_writetext(self):
        segments = [
            self._get_segment("Hello my friends."),
            self._get_segment("How are you?"),
        ]
        results = {"text": "all text", "segments": segments}

        filename, dirname = self._get_temp_file_name_dir()
        subtitlesWriter = WriteTXT(output_dir=dirname)
        subtitlesWriter(results, filename, dict())

        r = self._read_subtitles(filename + ".txt")

        self.assertEqual(2, len(r), "len")
        self.assertEqual("Hello my friends.\n", r[0], "text")
        self.assertEqual("How are you?\n", r[1], "text")

    def test_write_srt(self):
        segments = [
            self._get_segment("Hello my friends.", start=1, end=5),
            self._get_segment("How are you?", start=6.5, end=8),
        ]
        results = {"text": "all text", "segments": segments}

        filename, dirname = self._get_temp_file_name_dir()
        subtitlesWriter = WriteSRT(output_dir=dirname)
        subtitlesWriter(results, filename, {})
        r = self._read_subtitles(filename + ".srt")

        self.assertEqual(8, len(r), "text")
        self.assertEqual("1\n", r[0], "text")
        self.assertEqual("00:00:01,000 --> 00:00:05,000\n", r[1], "text")
        self.assertEqual("Hello my friends.\n", r[2], "text")
        self.assertEqual("\n", r[3], "text")
        self.assertEqual("2\n", r[4], "text")
        self.assertEqual("00:00:06,500 --> 00:00:08,000\n", r[5], "text")
        self.assertEqual("How are you?\n", r[6], "text")
        self.assertEqual("\n", r[7], "text")

    def test_write_tsv(self):
        segments = [
            self._get_segment("Hello my friends.", start=1, end=5),
            self._get_segment("How are you?", start=6.5, end=8),
        ]
        results = {"text": "all text", "segments": segments}

        filename, dirname = self._get_temp_file_name_dir()
        subtitlesWriter = WriteTSV(output_dir=dirname)
        subtitlesWriter(results, filename, {})
        r = self._read_subtitles(filename + ".tsv")
        self.assertEqual(3, len(r), "text")
        self.assertEqual("start\tend\ttext\n", r[0], "text")
        self.assertEqual("1000\t5000\tHello my friends.\n", r[1], "text")
        self.assertEqual("6500\t8000\tHow are you?\n", r[2], "text")

    def test_write_vtt(self):
        segments = [
            self._get_segment("Hello my friends.", start=1, end=5),
            self._get_segment("How are you?", start=6.5, end=8),
        ]
        results = {"text": "all text", "segments": segments}

        filename, dirname = self._get_temp_file_name_dir()
        subtitlesWriter = WriteVTT(output_dir=dirname)
        subtitlesWriter(results, filename, {})
        r = self._read_subtitles(filename + ".vtt")
        self.assertEqual(8, len(r), "text")
        self.assertEqual("WEBVTT\n", r[0], "text")
        self.assertEqual("\n", r[1], "text")
        self.assertEqual("00:01.000 --> 00:05.000\n", r[2], "text")
        self.assertEqual("Hello my friends.\n", r[3], "text")
        self.assertEqual("\n", r[4], "text")
        self.assertEqual("00:06.500 --> 00:08.000\n", r[5], "text")
        self.assertEqual("How are you?\n", r[6], "text")
        self.assertEqual("\n", r[7], "text")

    def test_write_srt_words(self):
        segment = self._get_segment("Hello", start=1, end=5)
        segments = [segment]
        segments[0]["words"] = [
            Word(start=1, end=2, word="Hello", probability=0)._asdict(),
        ]

        results = {"text": "all text", "segments": segments}

        filename, dirname = self._get_temp_file_name_dir()
        subtitlesWriter = WriteSRT(output_dir=dirname)
        subtitlesWriter(results, filename, {"highlight_words": True})
        r = self._read_subtitles(filename + ".srt")

        self.assertEqual(4, len(r), "text")
        self.assertEqual("1\n", r[0], "text")
        self.assertEqual("00:00:01,000 --> 00:00:02,000\n", r[1], "text")
        self.assertEqual("<u>Hello</u>\n", r[2], "text")

    def test_write_srt_words_max_line_width(self):
        segment = self._get_segment("Hello friends", start=1, end=5)
        segments = [segment]
        segments[0]["words"] = [
            Word(start=1, end=2, word="Hello", probability=0)._asdict(),
            Word(start=4, end=6, word="friends", probability=0)._asdict(),
        ]

        results = {"text": "all text", "segments": segments}

        filename, dirname = self._get_temp_file_name_dir()
        subtitlesWriter = WriteSRT(output_dir=dirname)
        subtitlesWriter(results, filename, {"max_line_width": 5})
        r = self._read_subtitles(filename + ".srt")
        self.assertEqual(5, len(r), "text")
        self.assertEqual("1\n", r[0], "text")
        self.assertEqual("00:00:01,000 --> 00:00:06,000\n", r[1], "text")
        self.assertEqual("Hello\n", r[2], "text")
        self.assertEqual("friends\n", r[3], "text")
        self.assertEqual("\n", r[4], "text")

    def test_write_srt_words_max_line_count(self):
        segment = self._get_segment("Hello friends", start=1, end=5)
        segments = [segment]
        segments[0]["words"] = [
            Word(start=1, end=2, word="Hello", probability=0)._asdict(),
            Word(start=4, end=6, word="friends", probability=0)._asdict(),
        ]

        results = {"text": "all text", "segments": segments}

        filename, dirname = self._get_temp_file_name_dir()
        subtitlesWriter = WriteSRT(output_dir=dirname)
        subtitlesWriter(results, filename, {"max_line_width": 5, "max_line_count": 1})
        r = self._read_subtitles(filename + ".srt")
        self.assertEqual(8, len(r), "text")
        self.assertEqual("1\n", r[0], "text")
        self.assertEqual("00:00:01,000 --> 00:00:02,000\n", r[1], "text")
        self.assertEqual("Hello\n", r[2], "text")
        self.assertEqual("\n", r[3], "text")
        self.assertEqual("2\n", r[4], "text")
        self.assertEqual("00:00:04,000 --> 00:00:06,000\n", r[5], "text")
        self.assertEqual("friends\n", r[6], "text")
        self.assertEqual("\n", r[7], "text")


if __name__ == "__main__":
    unittest.main()
