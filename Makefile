.PHONY: run install-dependencies-e2e-tests run-e2e-tests run-tests publish-release dev

run:
	python3 setup.py sdist bdist_wheel
	pip3 install --force-reinstall .

install-dependencies-e2e-tests:
	echo ctranslate2==4.0.0 > constraints.txt
	pip install --force-reinstall -c constraints.txt faster-whisper==1.1.0
	echo numpy==1.26 > constraints.txt
	pip install --force-reinstall -c constraints.txt pyannote.audio==3.3.1

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
