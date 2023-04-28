import unittest
from src.whisper_ctranslate2.live import Live


class TestLive(unittest.TestCase):
    def test_constructor(self):
        Live(
            "path",
            True,
            False,
            "translate",
            "ca",
            "10",
            "auto",
            0,
            "int8",
            False,
            0,
            None,
        )


if __name__ == "__main__":
    unittest.main()
