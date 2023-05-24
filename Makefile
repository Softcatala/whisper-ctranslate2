.PHONY: run run-e2e-test run-tests

run:
	python3 setup.py sdist bdist_wheel
	pip3 install --force-reinstall .
	
run-e2e-tests:
	pip install --force-reinstall ctranslate2==3.13.0
	pip install --force-reinstall faster-whisper==0.6.0
	CT2_USE_MKL="False" CT2_FORCE_CPU_ISA='GENERIC' nose2 -s e2e-tests

run-tests:
	nose2 -s tests
