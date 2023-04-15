# Problems using GPUs and CUDA drivers

GPU execution requires the NVIDIA libraries cuBLAS 11.x and cuDNN 8.x to be installed on the system. Please refer to the [CTranslate2 documentation](https://opennmt.net/CTranslate2/installation.html).

If executing whisper-ctranslate2 gives errors like:

*Could not load library cudnn_ops_infer64_8.dll. Error code 126*

Make sure that the environment variable *LD_LIBRARY_PATH* includes the path where your libraries are installed.

# Executing the program more than once produces different transcriptions

Some audio files can trigger the "temperature fallback" which is based on random sampling. So it is expected that the transcription is different each time. See [openai/whisper#81](https://github.com/openai/whisper/discussions/81) for more discussion about this.

# Transcribed text changes when using the vad filter 

If you execute whisper-ctranslate2 with or without the Vad filter options it produces different transcriptions.

This is expected since VAD will change the model input. 

# Transcription file is not written

If you have little memory the program can finish without showing any message.

Trying using a smaller model or a shorter file and watch out your available memory.

# Live transcription does not work

Make sure that your terminal has permissions to access the microphone. In some operating systems like mac OS access can be restricted because of privacy reasons.

# OSError: PortAudio library not found

This is a _sounddevice_ library dependency needed for the live transcription functionality. If you are running Ubuntu, you can install the necessary library by running:

    sudo apt-get install libportaudio2
