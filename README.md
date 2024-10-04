[![PyPI version](https://img.shields.io/pypi/v/whisper-ctranslate2.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/whisper-ctranslate2/)
[![PyPI downloads](https://img.shields.io/pypi/dm/whisper-ctranslate2.svg)](https://pypistats.org/packages/whisper-ctranslate2)

# Introduction

Whisper command line client compatible with original [OpenAI client](https://github.com/openai/whisper) based on CTranslate2.

It uses [CTranslate2](https://github.com/OpenNMT/CTranslate2/) and [Faster-whisper](https://github.com/SYSTRAN/faster-whisper) Whisper implementation that is up to 4 times faster than openai/whisper for the same accuracy while using less memory.

Goals of the project:
* Provide an easy way to use the CTranslate2 Whisper implementation
* Ease the migration for people using OpenAI Whisper CLI

# 🚀 **NEW PROJECT LAUNCHED!** 🚀

**Open dubbing** is an AI dubbing system which uses machine learning models to automatically translate and synchronize audio dialogue into different languages ! 🎉

### **🔥 Check it out now: [*open-dubbing*](https://github.com/jordimas/open-dubbing) 🔥**


# Installation

To install the latest stable version, just type:

    pip install -U whisper-ctranslate2

Alternatively, if you are interested in the latest development (non-stable) version from this repository, just type:

    pip install git+https://github.com/Softcatala/whisper-ctranslate2

# CPU and GPU support

GPU and CPU support are provided by [CTranslate2](https://github.com/OpenNMT/CTranslate2/).

It has compatibility with x86-64 and AArch64/ARM64 CPU and integrates multiple backends that are optimized for these platforms: Intel MKL, oneDNN, OpenBLAS, Ruy, and Apple Accelerate.

GPU execution requires the NVIDIA libraries cuBLAS 11.x and cuDNN 8.x to be installed on the system. Please refer to the [CTranslate2 documentation](https://opennmt.net/CTranslate2/installation.html)

By default the best hardware available is selected for inference. You can use the options `--device` and `--device_index` to control manually the selection.
    
# Usage

Same command line as OpenAI Whisper.

To transcribe:

    whisper-ctranslate2 inaguracio2011.mp3 --model medium
    
<img alt="image" src="https://user-images.githubusercontent.com/309265/226923541-8326c575-7f43-4bba-8235-2a4a8bdfb161.png">

To translate:

    whisper-ctranslate2 inaguracio2011.mp3 --model medium --task translate

<img alt="image" src="https://user-images.githubusercontent.com/309265/226923535-b6583536-2486-4127-b17b-c58d85cdb90f.png">

Whisper translate task translates the transcription from the source language to English (the only target language supported).

Additionally using:

    whisper-ctranslate2 --help

All the supported options with their help are shown.

# CTranslate2 specific options

On top of the OpenAI Whisper command line options, there are some specific options provided by CTranslate2 or whiper-ctranslate2.

## Quantization

`--compute_type` option which accepts _default,auto,int8,int8_float16,int16,float16,float32_ values indicates the type of [quantization](https://opennmt.net/CTranslate2/quantization.html) to use. On CPU _int8_ will give the best performance:

    whisper-ctranslate2 myfile.mp3 --compute_type int8

## Loading the model from a directory

`--model_directory` option allows to specify the directory from which you want to load a CTranslate2 Whisper model. For example, if you want to load your own quantified [Whisper model](https://opennmt.net/CTranslate2/conversion.html) version or using your own [Whisper fine-tunned](https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event) version. The model must be in CTranslate2 format.

## Using Voice Activity Detection (VAD) filter

`--vad_filter` option enables the voice activity detection (VAD) to filter out parts of the audio without speech. This step uses the [Silero VAD model](https://github.com/snakers4/silero-vad):

    whisper-ctranslate2 myfile.mp3 --vad_filter True

The VAD filter accepts multiple additional options to determine the filter behavior:

    --vad_threshold VALUE (float)

Probabilities above this value are considered as speech.

    --vad_min_speech_duration_ms (int)

Final speech chunks shorter min_speech_duration_ms are thrown out.

    --vad_max_speech_duration_s VALUE (int)

Maximum duration of speech chunks in seconds. Longer will be split at the timestamp of the last silence.


## Print colors

`--print_colors True` options prints the transcribed text using an experimental color coding strategy based on [whisper.cpp](https://github.com/ggerganov/whisper.cpp) to highlight words with high or low confidence:

    whisper-ctranslate2 myfile.mp3 --print_colors True

<img alt="image" src="https://user-images.githubusercontent.com/309265/228054378-48ac6af4-ce4b-44da-b4ec-70ce9f2f2a6c.png">

## Live transcribe from your microphone

`--live_transcribe True` option activates the live transcription mode from your microphone:

    whisper-ctranslate2 --live_transcribe True --language en

https://user-images.githubusercontent.com/309265/231533784-e58c4b92-e9fb-4256-b4cd-12f1864131d9.mov

## Diarization (speaker identification)

There is experimental diarization support using [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) to identify speakers. At the moment, the support is a segment level.

To enable diarization you need to follow these steps:

1. Install [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) with `pip install pyannote.audio`
2. Accept [`pyannote/segmentation-3.0`](https://hf.co/pyannote/segmentation-3.0) user conditions
3. Accept [`pyannote/speaker-diarization-3.1`](https://hf.co/pyannote/speaker-diarization-3.1) user conditions
4. Create access token at [`hf.co/settings/tokens`](https://hf.co/settings/tokens).

And then execute passing the HuggingFace API token as parameter to enable diarization:

    whisper-ctranslate2 --hf_token YOUR_HF_TOKEN

and then the name of the speaker is added in the output files (e.g. JSON, VTT and STR files):

_[SPEAKER_00]: There is a lot of people in this room_

The option `--speaker_name SPEAKER_NAME` allows to use your own string to identify the speaker.


# Need help?

Check our [frequently asked questions](FAQ.md) for common questions.

# Contact

Jordi Mas <jmas@softcatala.org>
