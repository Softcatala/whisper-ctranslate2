.PHONY: run install-dependencies-e2e-tests run-e2e-tests run-tests publish-release dev docker-build docker-run

docker-build:
	docker build  -t whisper-ctranslate2 . -f Dockerfile
	docker image ls | grep whisper-ctranslate2

docker-run:
	docker run --gpus "device=0" -v "$(shell pwd)":/srv/files/ -it --rm whisper-ctranslate2 /srv/files/e2e-tests/gossos.mp3 --output_dir /srv/files/

run:
	python3 setup.py sdist bdist_wheel
	pip3 install --force-reinstall .

install-dependencies-e2e-tests:
	pip install --force-reinstall faster-whisper==1.2.0 "ctranslate2==4.0.0"
	pip install --force-reinstall "pyannote.audio<4.0.0" "torchaudio<2.9.0"

run-e2e-tests:
	CT2_USE_MKL="False" CT2_FORCE_CPU_ISA='GENERIC' KMP_DUPLICATE_LIB_OK="TRUE" nose2 -s e2e-tests

run-tests:
	nose2 -s tests

E2E_COMMAND=CT2_USE_MKL="False" CT2_FORCE_CPU_ISA="GENERIC" whisper-ctranslate2
update-e2e-tests:
	$(E2E_COMMAND) e2e-tests/gossos.mp3 --device cpu --compute_type float32 --word_timestamps True --output_dir e2e-tests/ref-small-transcribe-word-stamps/
	$(E2E_COMMAND) e2e-tests/gossos.mp3 --device cpu --task translate --model medium --compute_type float32 --output_dir e2e-tests/ref-medium-translate/
	$(E2E_COMMAND) e2e-tests/gossos.mp3 --device cpu  --compute_type float32 --output_dir e2e-tests/ref-small-transcribe/
	$(E2E_COMMAND) e2e-tests/dosparlants.mp3 --temperature_increment_on_fallback None  --device cpu --model medium --compute_type float32 --output_dir e2e-tests/ref-medium-diarization/ --hf_token ${HF_TOKEN}
	$(E2E_COMMAND) e2e-tests/gossos.mp3 --device cpu --compute_type float32 --max_words_per_line 5 --word_timestamps True --output_dir e2e-tests/ref-small-transcribe-line-max-words/
	$(E2E_COMMAND) e2e-tests/gossos.mp3 --device cpu --compute_type float32 --vad_filter True --vad_onset 0.5 --vad_min_speech_duration_ms 2000 --vad_max_speech_duration_s 50000 --output_dir e2e-tests/ref-small-transcribe-vad/

PATHS = src/ tests/ e2e-tests/

dev:
	python -m black $(PATHS)
	python -m flake8 $(PATHS)
	python -m isort $(PATHS)

publish-release:
	rm dist/ -r -f
	python3 setup.py sdist bdist_wheel
	python3 -m  twine upload -u "${PYPI_USERNAME}" -p "${PYPI_PASSWORD}" --repository-url https://upload.pypi.org/legacy/ dist/*
	@echo 'Do -> git tag -m "0.X.Y" -a 0.X.Y && git push --tags'
