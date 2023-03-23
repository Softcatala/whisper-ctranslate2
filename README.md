[![PyPI version](https://img.shields.io/pypi/v/whisper-ctranslate2.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/whisper-ctranslate2/)
[![PyPI downloads](https://img.shields.io/pypi/dm/whisper-ctranslate2.svg)](https://pypistats.org/packages/whisper-ctranslate2)

# Introduction

Whisper command line client compatible with original [OpenAI client](https://github.com/openai/whisper) based on CTranslate2.

It uses [CTranslate2](https://github.com/OpenNMT/CTranslate2/) and [Faster-whisper](https://github.com/guillaumekln/faster-whisper) Whisper implementation that is up to 4 times faster than openai/whisper for the same accuracy while using less memory.  

Goals of the project:
* Provide an easy way to use the CTranslate2 Whisper implementation
* Easy the migration for people using OpenAI Whisper CLI

# Installation

Just type:

    pip install -U whisper-ctranslate2

Alternatively, the following command will pull and install the latest commit from this repository, along with its Python dependencies:

    pip install https://github.com/jordimas/whisper-ctranslate2
    
# Usage

Same command line that OpenAI whisper.

To transcribe:

    whisper-ctranslate2 inaguracio2011.mp3 --model medium
    
<img alt="image" src="https://user-images.githubusercontent.com/309265/226923541-8326c575-7f43-4bba-8235-2a4a8bdfb161.png">

To translate:

    whisper-ctranslate2 inaguracio2011.mp3 --model medium --task translate

<img alt="image" src="https://user-images.githubusercontent.com/309265/226923535-b6583536-2486-4127-b17b-c58d85cdb90f.png">

# Contact

Jordi Mas <jmas@softcatala.org>
