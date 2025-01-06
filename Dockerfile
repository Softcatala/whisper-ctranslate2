FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip

WORKDIR /app
COPY . /app

RUN pip3 install --no-cache-dir -U .
RUN python3 -c 'from faster_whisper import WhisperModel; WhisperModel("small"); WhisperModel("medium");  WhisperModel("large-v2")'

ENTRYPOINT ["whisper-ctranslate2"]
