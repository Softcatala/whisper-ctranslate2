.PHONY: run run-e2e-test

run:
	python3 setup.py sdist bdist_wheel
	pip3 install --force-reinstall --user dist/whisper_ctranslate2-0.1.6-py3-none-any.whl
	
run-e2e-test:
	pip install --force-reinstall ctranslate2==3.11
	pip install --force-reinstall faster-whisper==0.4.1
	CT2_USE_MKL="False" CT2_FORCE_CPU_ISA='GENERIC' nose2
