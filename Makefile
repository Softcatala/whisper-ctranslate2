.PHONY: run run-e2e-test run-tests publish-release

run:
	python3 setup.py sdist bdist_wheel
	pip3 install --force-reinstall .
	
run-e2e-tests:
	pip install --force-reinstall ctranslate2==3.14.0
	pip install --force-reinstall faster-whisper==0.6.0
	CT2_USE_MKL="False" CT2_FORCE_CPU_ISA='GENERIC' nose2 -s e2e-tests

run-tests:
	nose2 -s tests

publish-release:
	rm dist/ -r -f
	python3 setup.py sdist bdist_wheel
	python3 -m  twine upload -u "${PYPI_USERNAME}" -p "${PYPI_PASSWORD}" --repository-url https://upload.pypi.org/legacy/ dist/*
	@echo 'Do -> git tag -m "0.X.Y" -a 0.X.Y && git push --tags'
