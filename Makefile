.PHONY: run install-dependencies-e2e-tests run-e2e-tests run-tests publish-release

run:
	python3 setup.py sdist bdist_wheel
	pip3 install --force-reinstall .

install-dependencies-e2e-tests:
	echo ctranslate2==3.22.0 > constraints.txt
	pip install --force-reinstall -c constraints.txt https://github.com/SYSTRAN/faster-whisper/archive/refs/tags/0.10.0.tar.gz faster-whisper
	pip install --force-reinstall pyannote.audio==3.1.1

run-e2e-tests:
	CT2_USE_MKL="False" CT2_FORCE_CPU_ISA='GENERIC' nose2 -s e2e-tests

run-tests:
	nose2 -s tests

E2E_COMMAND=CT2_USE_MKL="False" CT2_FORCE_CPU_ISA="GENERIC" whisper-ctranslate2
update-e2e-tests:
	$(E2E_COMMAND) e2e-tests/gossos.mp3 --device cpu --compute_type float32 --word_timestamps True --output_dir e2e-tests/ref-small-transcribe-word-stamps/
	$(E2E_COMMAND) e2e-tests/gossos.mp3 --device cpu --task translate --model medium --compute_type float32 --output_dir e2e-tests/ref-medium-translate/
	$(E2E_COMMAND) e2e-tests/gossos.mp3 --device cpu  --compute_type float32 --output_dir e2e-tests/ref-small-transcribe/
	$(E2E_COMMAND) e2e-tests/dosparlants.mp3 --device cpu --model medium --compute_type float32 --output_dir e2e-tests/ref-medium-diarization/ --hf_token ${HF_TOKEN}


publish-release:
	rm dist/ -r -f
	python3 setup.py sdist bdist_wheel
	python3 -m  twine upload -u "${PYPI_USERNAME}" -p "${PYPI_PASSWORD}" --repository-url https://upload.pypi.org/legacy/ dist/*
	@echo 'Do -> git tag -m "0.X.Y" -a 0.X.Y && git push --tags'
