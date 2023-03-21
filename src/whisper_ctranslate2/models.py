#
# Based on code from https://github.com/openai/whisper
#
import hashlib
import os
import urllib.request
from tqdm import tqdm


class Models:
    _MODELS = {
        "tiny": "https://huggingface.co/jordimas/whisper-ct2/resolve/main/1e2aa48a43d45b9304a22aca730c5b5177a923f50008bffd26140e3b6ff19fdf.tiny/",
        "tiny.en": "https://huggingface.co/jordimas/whisper-ct2/resolve/main/d82547957fb5dffbe279d22292980ab3ad33675a231faadbc9f2c908b909b42d.tiny-en/",
        "base": "https://huggingface.co/jordimas/whisper-ct2/resolve/main/2b3a5459a344d5cfc6959018fb01f78ad610460d184ddf6a3282c187e7ddeceb.base/",
        "base.en": "https://huggingface.co/jordimas/whisper-ct2/resolve/main/67b9ae74793ba78292c0c11db4bf296169bbfbc8bd9f81dadf22546b9fa0e809.base-en/",
        "small": "https://huggingface.co/jordimas/whisper-ct2/resolve/main/64c8c26522e710cfad21c44c037a806cccf0dda756ce0a222b5f3a938d87de43.small/",
        "small.en": "https://huggingface.co/jordimas/whisper-ct2/resolve/main/ee51a5c0132eb205931bfdd635fae2ada72f5580c812ab113ca2d735ea2707e9.small-en/",
        "medium": "https://huggingface.co/jordimas/whisper-ct2/resolve/main/309bb6f6fc40dedef3526916b4110288796ca6a873ed73dfc9f3b7bbbf5ecccd.medium/",
        "medium.en": "https://huggingface.co/jordimas/whisper-ct2/resolve/main/4f2c05f42abebfd9f5f9027e26966153b38995722a066ee8da14e6dbdc976622.medium-en/",
        "large": "https://huggingface.co/jordimas/whisper-ct2/resolve/main/d2bb7bcdf195976460f6ba94f4222d3ba8d6b306d0a7d98447f3d58eecac4a28.large/",
    }

    _FILES = ["model.bin", "config.json", "vocabulary.txt"]

    def __init__(self):
        default = os.path.join(os.path.expanduser("~"), ".cache")
        self.download_root = os.path.join(
            os.getenv("XDG_CACHE_HOME", default), "whisper-ctranslate2"
        )

    def get_list(self) -> str:
        return list(self._MODELS.keys())

    def get_model_path(self, model) -> str:
        url = self._MODELS[model]
        model_dir = url.split("/")[-2]
        return os.path.join(self.download_root, model_dir)

    def get_model_dir(self, model):
        if model not in self._MODELS:
            raise RuntimeError(f"Model {model} not supported")

        model_path = self.get_model_path(model)

        if self._are_localfile_checksums_correct(model, self._MODELS[model]):
            return model_path

        for _file in self._FILES:
            self._download(model, self._MODELS[model] + _file)

        if self._are_localfile_checksums_correct(model, self._MODELS[model]):
            return model_path

        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match. Please retry downloading the model."
        )

    def _are_localfile_checksums_correct(self, model: str, url: str) -> bool:
        model_path = self.get_model_path(model)
        model_bytes = bytes()
        expected_sha256 = url.split("/")[-2].rsplit(".", 1)[0]

        for _file in self._FILES:
            _file = os.path.join(model_path, _file)

            if not os.path.exists(_file):
                return False

            with open(_file, "rb") as f:
                model_bytes += f.read()

        got = hashlib.sha256(model_bytes).hexdigest()
        print(f"expected: {expected_sha256}")
        print(f"got: {got}")
        return got == expected_sha256

    def _download(self, model: str, url: str) -> str:
        model_path = self.get_model_path(model)
        os.makedirs(model_path, exist_ok=True)

        expected_sha256 = url.split("/")[-2].rsplit(".", 1)[0]
        download_target = os.path.join(model_path, os.path.basename(url))

        print(f"*** Download {download_target}")

        if os.path.exists(download_target) and not os.path.isfile(download_target):
            raise RuntimeError(f"{download_target} exists and is not a regular file")

        with urllib.request.urlopen(url) as source, open(
            download_target, "wb"
        ) as output:
            with tqdm(
                total=int(source.info().get("Content-Length")),
                ncols=80,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))
