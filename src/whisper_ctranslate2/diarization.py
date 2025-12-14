from collections import OrderedDict

import numpy as np

from faster_whisper.audio import decode_audio

try:
    import torch
except Exception as e:
    print(f"Unable to import torch library. Make sure that it's installed. Error: {e}")

try:
    from pyannote.audio import Pipeline
except Exception as e:
    print(
        f"Unable to import pyannote.audio library. Make sure that it's installed. Error: {e}"
    )


class Diarization:
    def __init__(
        self,
        token=None,
        device: str = "cpu",
        num_speakers=2,
    ):
        self.device = device
        self.token = token
        self.num_speakers = num_speakers

    def set_threads(self, threads):
        torch.set_num_threads(threads)

    def unload_model(self):
        del self.model
        torch.cuda.empty_cache()

    def _load_model(self):
        model_name = "pyannote/speaker-diarization-community-1"
        device = torch.device(self.device)
        model_handle = Pipeline.from_pretrained(model_name, token=self.token)
        if model_handle is None:
            raise ValueError(
                f"The token Hugging Face token '{self.token}' for diarization is not valid or you did not accept the EULAs for the necessary models. See https://github.com/Softcatala/whisper-ctranslate2#diarization-speaker-identification"
            )

        self.model = model_handle.to(device)

    def run_model(self, audio: str):
        self._load_model()
        audio = decode_audio(audio)
        audio_data = {
            "waveform": torch.from_numpy(audio[None, :]),
            "sample_rate": 16000,
        }
        segments = self.model(audio_data, num_speakers=self.num_speakers)
        return segments

    def assign_speakers_to_segments(self, segments, transcript_result, speaker_name):
        diarize_data = []
        for turn, speaker in segments.speaker_diarization:
            diarize_data.append((turn, None, speaker))

        return self._do_assign_speakers_to_segments(
            diarize_data, transcript_result, speaker_name
        )

    def _do_assign_speakers_to_segments(
        self, diarize_data, transcript_result, speaker_name
    ):
        diarize_df = np.array(
            diarize_data,
            dtype=[("segment", object), ("label", object), ("speaker", object)],
        )

        diarize_df = np.core.records.fromarrays(
            [
                diarize_df["segment"],
                diarize_df["label"],
                diarize_df["speaker"],
                np.array([seg.start for seg in diarize_df["segment"]]),
                np.array([seg.end for seg in diarize_df["segment"]]),
                np.zeros(len(diarize_df)),
            ],
            names="segment, label, speaker, start, end, intersection",
        )

        for seg in transcript_result["segments"]:
            intersection = np.minimum(diarize_df["end"], seg["end"]) - np.maximum(
                diarize_df["start"], seg["start"]
            )
            diarize_df["intersection"] = intersection
            dia_segment = diarize_df[diarize_df["intersection"] > 0]
            if len(dia_segment) > 0:
                speakers = {}
                for item in dia_segment:
                    speaker = item["speaker"]
                    old_i = speakers.get(speaker, 0)
                    speakers[speaker] = old_i + item["intersection"]

                sorted_dict = OrderedDict(
                    sorted(speakers.items(), key=lambda x: x[1], reverse=True)
                )
                first_item = next(iter(sorted_dict.items()))
                if first_item:
                    speaker = first_item[0]
                    if speaker_name:
                        speaker = speaker.replace("SPEAKER", speaker_name)
                    seg["speaker"] = speaker

        return transcript_result
