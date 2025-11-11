from collections import OrderedDict

import numpy as np

from faster_whisper.audio import decode_audio

try:
    import torch
except Exception:
    print("Unable to import torch library. Make sure that it's installed")

try:
    from pyannote.audio import Pipeline
except Exception:
    print("Unable to import pyannote.audio library. Make sure that it's installed")


class Diarization:
    def __init__(
        self,
        use_auth_token=None,
        device: str = "cpu",
    ):
        self.device = device
        self.use_auth_token = use_auth_token

    def set_threads(self, threads):
        torch.set_num_threads(threads)

    def unload_model(self):
        del self.model
        torch.cuda.empty_cache()

    def _load_model(self) -> "Pipeline":
        model_name = "pyannote/speaker-diarization-3.1"
        device = torch.device(self.device)
        model_handle = Pipeline.from_pretrained(
            model_name, token=self.use_auth_token
        )
        if model_handle is None:
            raise ValueError(
                f"The token Hugging Face token '{self.use_auth_token}' for diarization is not valid or you did not accept the EULA"
            )

        self.model = model_handle.to(device)

    def run_model(self, audio: str) -> "Pipeline":
        self._load_model()
        audio = decode_audio(audio)
        audio_data = {
            "waveform": torch.from_numpy(audio[None, :]),
            "sample_rate": 16000,
        }
        segments = self.model(audio_data)
        return segments

    def assign_speakers_to_segments(
        self,
        segments,
        transcript_result,
        speaker_name
    ):
        diarize_data = self.diarize_chunks_to_records(segments)
        return self._do_assign_speakers_to_segments(
            diarize_data,
            transcript_result,
            speaker_name
        )

    def diarize_chunks_to_records(self, segments):
        diarize_data = list(
            segments.itertracks(yield_label=True)
        )
        date_frame = np.array(
            diarize_data,
            dtype=[
                ("segment", object),
                ("label", object),
                ("speaker", object)
            ],
        )

        segments_as_records = np.core.records.fromarrays(
            [
                date_frame["segment"],
                date_frame["label"],
                date_frame["speaker"],
                np.array([seg.start for seg in date_frame["segment"]]),
                np.array([seg.end for seg in date_frame["segment"]]),
                np.zeros(len(date_frame)),
            ],
            names="segment, label, speaker, start, end, intersection",
        )

        return segments_as_records

    def assign_speaker_to_segment(
        self,
        segment,
        diarize_df,
        speaker_name
    ):
        # Create a copy of the incoming segment
        segment_with_speaker = segment.copy()

        intersection = np.minimum(
            diarize_df["end"],
            segment["end"]) - np.maximum(diarize_df["start"],
            segment["start"]
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
                segment_with_speaker["speaker"] = speaker

        return segment_with_speaker

    def _do_assign_speakers_to_segments(
        self,
        diarize_df,
        transcript_result,
        speaker_name
    ):
        diarized_segments = []
        for seg in transcript_result["segments"]:
            new_segment = self.assign_speaker_to_segment(
                seg,
                diarize_df,
                speaker_name
            )
            diarized_segments.append(new_segment)

        transcript_result["segments"] = diarized_segments
        return transcript_result
