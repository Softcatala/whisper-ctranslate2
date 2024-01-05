import unittest
from src.whisper_ctranslate2.live import Live


class TestLive(unittest.TestCase):
    def test_constructor(self):
        live = Live(
            model_path="path",
            cache_directory="directory",
            local_files_only=False,
            task="translate",
            language="ca",
            threads="10",
            device="auto",
            device_index=0,
            compute_type="int8",
            verbose=False,
            threshold=0.2,
            input_device=0,
            options=None,
        )
        self.assertNotEqual(None, live)


if __name__ == "__main__":
    unittest.main()
