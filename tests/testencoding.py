import os
import subprocess
import sys
import unittest


class TestEncoding(unittest.TestCase):
    def test_help_with_non_utf8_stdout(self):
        # A redirected or piped stdout on Windows uses the ANSI code page, which
        # cannot encode every character of the help text.
        env = dict(os.environ, PYTHONIOENCODING="cp1252")
        env.pop("PYTHONUTF8", None)

        result = subprocess.run(
            [sys.executable, "-m", "whisper_ctranslate2.whisper_ctranslate2", "--help"],
            env=env,
            capture_output=True,
        )

        self.assertEqual(0, result.returncode, result.stderr.decode(errors="replace"))
        self.assertIn(b"--append_punctuations", result.stdout)


if __name__ == "__main__":
    unittest.main()
