#
# Based on code from https://github.com/openai/whisper
#

import os
import json
import re
import copy
from typing import Callable, TextIO, Optional


def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


class ResultWriter:
    extension: str

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def __call__(self, result: dict, audio_path: str, options: dict):
        audio_basename = os.path.basename(audio_path)
        audio_basename = os.path.splitext(audio_basename)[0]
        output_path = os.path.join(
            self.output_dir, audio_basename + "." + self.extension
        )

        with open(output_path, "w", encoding="utf-8") as f:
            self.write_result(result, f, options)

    def write_result(self, result: dict, file: TextIO, options: dict):
        raise NotImplementedError


class SubtitlesWriter(ResultWriter):
    always_include_hours: bool
    decimal_marker: str

    def iterate_result(self, result: dict, options: dict):
        raw_max_line_width: Optional[int] = options.get("max_line_width", None)
        max_line_count: Optional[int] = options.get("max_line_count", None)
        highlight_words = options.get("highlight_words", False)
        max_line_width = 1000 if raw_max_line_width is None else raw_max_line_width
        preserve_segments = max_line_count is None or raw_max_line_width is None

        # Splits a line based on commas or word gaps
        def split_lineIfNeeded(words, max_splits=12):
            # If there are no words or we have no more splits left, return an empty list
            if not words or max_splits <= 0:
                return [{'words': words}]
            
            # If the length of the words is less or equal to n, return the words as they are.
            if len("".join([word['word'] for word in words])) <= max_line_width:
                return [{'words': words}]

            # Find the index of the comma closest to the middle of the line
            middle = len(words) // 2
            comma_indices = [i for i, word in enumerate(words[:-1]) if ',' in word['word']]
            closest_comma_index = min(comma_indices, key=lambda idx: abs(middle - idx), default=None)

            #discard comma index if it's too close to the beginning or end of the line
            if closest_comma_index is not None and closest_comma_index < len(words) // 5:
                closest_comma_index = None

            # If there's no comma, find the largest gap among approximately 20% of words (/5) around the center
            if closest_comma_index is None:
                window_start = max(0, middle - len(words) // 5)
                window_end = min(len(words), middle + len(words) // 5)
                max_gap_size = 0
                for i in range(window_start, window_end - 1):
                    gap_size = words[i + 1]['start'] - words[i]['end']
                    if gap_size > max_gap_size:
                        max_gap_size = gap_size
                        closest_comma_index = i

            # If there's still no suitable split point (no comma and no gap found), split at the middle
            if closest_comma_index is None:
                closest_comma_index = middle

            # Splitting the line at the closest comma or the largest gap
            left_part = words[:closest_comma_index + 1]
            right_part = words[closest_comma_index + 1:]

            # Recursively check if the split parts need further splitting
            split_left_part = split_lineIfNeeded(left_part, max_splits-1)
            split_right_part = split_lineIfNeeded(right_part, max_splits-1)

            return split_left_part + split_right_part
        
        #Yield successive chunks from lst of size chunk_size.
        def chunks(lst, chunk_size):
            for i in range(0, len(lst), chunk_size):
                yield lst[i:i + chunk_size]

        def iterate_subtitles():
            output_subtitles = [] # All of the output subtitles
            result_copy = copy.deepcopy(result) # Copy of the result to be modified
            for segment in result_copy["segments"]:
                output_subtitles_buffer = [] # All the split subtitles resulting from the current segment
                speaker = f"[{segment['speaker']}]: " if "speaker" in segment else "" # Speaker for the current segment

                for group_of_lines in chunks(split_lineIfNeeded(segment["words"]), max_line_count):
                    output_line = [] # A single subtitle, composed of multiple raw_line in a group_of_lines, joined with a linebreak.
                    for raw_line in group_of_lines: # A raw_line is a single line. Multiple raw_line make up a subtitle, concatenated with linebreaks inserted in words where needed.
                        is_last_line = raw_line == group_of_lines[-1]
                        if not is_last_line: # All but the last line should have a linebreak at the end
                            raw_line["words"][0]["word"] = raw_line["words"][0]["word"].lstrip() # Remove leading space from first word.
                            raw_line["words"][-1]["word"] = raw_line["words"][-1]["word"].rstrip() + "\n" # Replace trailing space in last word with linebreak
                        else: #The last line should not have a linebreak at the end
                            raw_line["words"][0]["word"] = raw_line["words"][0]["word"].lstrip() # Remove leading space from first word.
                            raw_line["words"][-1]["word"] = raw_line["words"][-1]["word"].rstrip() # Remove trailing space from last word.
                        output_line.extend(raw_line["words"])

                    word_already_has_speaker = output_line[0]["word"].startswith(speaker) # Ensure the speaker is only added once
                    if not word_already_has_speaker:
                        output_line[0]["word"] = speaker + output_line[0]["word"].lstrip() # Add speaker to first word of subtitle

                    output_subtitles_buffer.append(output_line)
                output_subtitles.extend(output_subtitles_buffer)
            return output_subtitles
                                
        if (
            len(result["segments"]) > 0
            and "words" in result["segments"][0]
            and result["segments"][0]["words"]
        ):
            for subtitle in iterate_subtitles():
                subtitle_start = self.format_timestamp(subtitle[0]["start"])
                subtitle_end = self.format_timestamp(subtitle[-1]["end"])
                subtitle_text = "".join([word["word"] for word in subtitle]).strip()
                if highlight_words:
                    last = subtitle_start
                    all_words = [timing["word"] for timing in subtitle]
                    for i, this_word in enumerate(subtitle):
                        start = self.format_timestamp(this_word["start"])
                        end = self.format_timestamp(this_word["end"])
                        if last != start:
                            yield last, start, subtitle_text

                        yield start, end, "".join(
                            [
                                re.sub(r"^(\s*)(.*)$", r"\1<u>\2</u>", word)
                                if j == i
                                else word
                                for j, word in enumerate(all_words)
                            ]
                        )
                        last = end
                else:
                    yield subtitle_start, subtitle_end, subtitle_text
        else:
            for segment in result["segments"]:
                speaker = f"[{segment['speaker']}]: " if "speaker" in segment else ""
                segment_start = self.format_timestamp(segment["start"])
                segment_end = self.format_timestamp(segment["end"])
                segment_text = speaker + segment["text"].strip().replace("-->", "->")
                yield segment_start, segment_end, segment_text

    def format_timestamp(self, seconds: float):
        return format_timestamp(
            seconds=seconds,
            always_include_hours=self.always_include_hours,
            decimal_marker=self.decimal_marker,
        )


class WriteTXT(ResultWriter):
    extension: str = "txt"

    def write_result(self, result: dict, file: TextIO, options: dict):
        for segment in result["segments"]:
            speaker = f"[{segment['speaker']}]: " if "speaker" in segment else ""
            print(speaker + segment["text"].strip(), file=file, flush=True)


class WriteSRT(SubtitlesWriter):
    extension: str = "srt"
    always_include_hours: bool = True
    decimal_marker: str = ","

    def write_result(self, result: dict, file: TextIO, options: dict):
        for i, (start, end, text) in enumerate(
            self.iterate_result(result, options), start=1
        ):
            print(f"{i}\n{start} --> {end}\n{text}\n", file=file, flush=True)


class WriteVTT(SubtitlesWriter):
    extension: str = "vtt"
    always_include_hours: bool = False
    decimal_marker: str = "."

    def write_result(self, result: dict, file: TextIO, options: dict):
        print("WEBVTT\n", file=file)
        for start, end, text in self.iterate_result(result, options):
            print(f"{start} --> {end}\n{text}\n", file=file, flush=True)


class WriteTSV(ResultWriter):
    """
    Write a transcript to a file in TSV (tab-separated values) format containing lines like:
    <start time in integer milliseconds>\t<end time in integer milliseconds>\t<transcript text>

    Using integer milliseconds as start and end times means there's no chance of interference from
    an environment setting a language encoding that causes the decimal in a floating point number
    to appear as a comma; also is faster and more efficient to parse & store, e.g., in C++.
    """

    extension: str = "tsv"

    def write_result(self, result: dict, file: TextIO, options: dict):
        print("start", "end", "text", sep="\t", file=file)
        for segment in result["segments"]:
            print(round(1000 * segment["start"]), file=file, end="\t")
            print(round(1000 * segment["end"]), file=file, end="\t")
            print(segment["text"].strip().replace("\t", " "), file=file, flush=True)


class WriteJSON(ResultWriter):
    extension: str = "json"

    def write_result(self, result: dict, file: TextIO, options: dict):
        pretty_json: bool = options.get("pretty_json", False)

        if pretty_json:
            json.dump(result, file, indent=4, ensure_ascii=False)
        else:
            json.dump(result, file)


def get_writer(
    output_format: str, output_dir: str
) -> Callable[[dict, TextIO, dict], None]:
    writers = {
        "txt": WriteTXT,
        "vtt": WriteVTT,
        "srt": WriteSRT,
        "tsv": WriteTSV,
        "json": WriteJSON,
    }

    if output_format == "all":
        all_writers = [writer(output_dir) for writer in writers.values()]

        def write_all(result: dict, file: TextIO, options: dict):
            for writer in all_writers:
                writer(result, file, options)

        return write_all

    return writers[output_format](output_dir)
