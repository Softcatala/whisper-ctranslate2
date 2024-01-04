import unittest
from src.whisper_ctranslate2.diarization import Diarization


class TestDiarization(unittest.TestCase):
    class Segment:
        start: float = 0.0
        end: float = 0.0

        def __init__(self, start, end):
            self.start = start
            self.end = end

    def test_no_speaker(self):
        pyannote_data = [(TestDiarization.Segment(10, 20), "A", "SPEAKER_00")]

        segment = {"start": 1, "end": 5}
        segments = [segment]

        segments = Diarization()._do_assign_speakers_to_segments(
            pyannote_data, {"segments": segments}, None
        )

        self.assertFalse("speaker" in segment)

    def test_single_speaker(self):
        pyannote_data = [(TestDiarization.Segment(2, 4), "A", "SPEAKER_00")]

        segment = {"start": 1, "end": 5}
        segments = [segment]

        segments = Diarization()._do_assign_speakers_to_segments(
            pyannote_data, {"segments": segments}, None
        )

        self.assertEqual("SPEAKER_00", segment["speaker"])

    def test_two_speakers(self):
        pyannote_data = [
            (TestDiarization.Segment(1, 5), "A", "SPEAKER_00"),
            (TestDiarization.Segment(5, 7), "B", "SPEAKER_01"),
        ]

        segment = {"start": 4, "end": 10}
        segments = [segment]

        segments = Diarization()._do_assign_speakers_to_segments(
            pyannote_data, {"segments": segments}, None
        )

        self.assertEqual("SPEAKER_01", segment["speaker"])

    def test_single_speaker_with_speakername(self):
        SPEAKER_NAME = "PARLANT"
        pyannote_data = [(TestDiarization.Segment(2, 4), "A", "SPEAKER_00")]

        segment = {"start": 1, "end": 5}
        segments = [segment]

        segments = Diarization()._do_assign_speakers_to_segments(
            pyannote_data, {"segments": segments}, SPEAKER_NAME
        )

        self.assertEqual("PARLANT_00", segment["speaker"])


if __name__ == "__main__":
    unittest.main()
