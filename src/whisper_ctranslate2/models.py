#
# Based on code from https://github.com/openai/whisper
#
import hashlib
import os
import urllib.request
from tqdm import tqdm


class Models:
    _MODELS = {
        "tiny": "https://huggingface.co/datasets/jordimas/whisper-ct2-v2/resolve/main/dc6a7765cdc8ae7822b1c3068a2f966eddc2549eda7e67406cae915a8c19430c.tiny/",
        "tiny.en": "https://huggingface.co/datasets/jordimas/whisper-ct2-v2/resolve/main/c4d1be0a2003ce00bb721abd23e7a34925a6f0c0d21d5c416f11c763ee7f7b15.tiny-en/",
        "base": "https://huggingface.co/datasets/jordimas/whisper-ct2-v2/resolve/main/d7c4df31737340263ce37933bda1e77a38367dbb09cda7433ce1ee0c58ce1a60.base/",
        "base.en": "https://huggingface.co/datasets/jordimas/whisper-ct2-v2/resolve/main/2ca96777261aeadd48e30bc5f26fd3ee462f4921dbc1f38dfd826a67c761c9b2.base-en/",
        "small": "https://huggingface.co/datasets/jordimas/whisper-ct2-v2/resolve/main/936cd99363be80fa388c0c5006e846d8cd42834c6cf8156a7c300723a3bf929f.small/",
        "small.en": "https://huggingface.co/datasets/jordimas/whisper-ct2-v2/resolve/main/e2c14bb0f6a8a69afe12fbe1d82fa0c41494d4bde9615bdf399da5665f43cbc4.small-en/",
        "medium": "https://huggingface.co/datasets/jordimas/whisper-ct2-v2/resolve/main/d8b91e278db3041c3b41bf879716281edf5cfa7b0025823cc174b5429877d2bc.medium/",
        "medium.en": "https://huggingface.co/datasets/jordimas/whisper-ct2-v2/resolve/main/22d42d3e69ce9149bfd52c07d357e4cb72b992fb602805d6bb39f331400d6742.mediu-en/",
        "large-v2": "https://huggingface.co/datasets/jordimas/whisper-ct2-v2/resolve/main/ea44e7a9609bd21a604b28880b85fc7dc2c373876c627a7553ce5440a8c406c1.large/",
    }

    _FILES = ["model.bin", "config.json", "vocabulary.txt", "tokenizer.json"]

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
        return got == expected_sha256

    def _download(self, model: str, url: str) -> str:
        model_path = self.get_model_path(model)
        os.makedirs(model_path, exist_ok=True)

        download_target = os.path.join(model_path, os.path.basename(url))

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
