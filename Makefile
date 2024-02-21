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

publish-release:
	rm dist/ -r -f
	python3 setup.py sdist bdist_wheel
	python3 -m  twine upload -u "${PYPI_USERNAME}" -p "${PYPI_PASSWORD}" --repository-url https://upload.pypi.org/legacy/ dist/*
	@echo 'Do -> git tag -m "0.X.Y" -a 0.X.Y && git push --tags'
