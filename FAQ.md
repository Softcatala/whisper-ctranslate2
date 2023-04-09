
# Problems loading CUDA drivers.

If executing whisper-ctranslate2 gives errors like:

*Could not load library cudnn_ops_infer64_8.dll. Error code 126*

Make sure that the environment variable *LD_LIBRARY_PATH* includes the path where you libraries are installed.

# Executing the program more than once produces different transcriptions

Some audio files can trigger the "temperature fallback" which is based on random sampling. So it is expected that the transcription is different each time. See [openai/whisper#81](https://github.com/openai/whisper/discussions/81) for more discussion about this.

# Transcribed text changes when using the vad filter 

If you execute whisper-ctranslate2 with or without the Vad filter options it produces different transcriptions.

This is expected since VAD will change the model input. 

