# Introduction

Whisper command line client compatible with original [OpenAI client](https://github.com/openai/whisper) based on CTranslate2.

It uses [CTranslate2](https://github.com/OpenNMT/CTranslate2/) and [Faster-whisper](https://github.com/guillaumekln/faster-whisper) Whisper implementation that is up to 4 times faster than openai/whisper for the same accuracy while using less memory.  

Goals of the project:
* Provide an easy way to use the CTranslate2 Whisper implementation
* Easy the migration for people using OpenAI Whisper CLI

# Installation

You need to install this dependency first:

    pip install git+https://github.com/guillaumekln/faster-whisper

And then, just type:

    pip install -U whisper-ctranslate2

Alternatively, the following command will pull and install the latest commit from this repository, along with its Python dependencies:

    pip install https://github.com/jordimas/whisper-ctranslate2
