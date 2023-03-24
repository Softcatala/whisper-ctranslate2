import pathlib
from setuptools import setup
import pkg_resources
import os

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="whisper-ctranslate2",
    version="0.0.5",
    description="Whisper command line client that uses CTranslate2",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/jordimas/whisper-ctranslate2",
    author="Jordi Mas",
    author_email="jmas@softcatala.org",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        'Programming Language :: Python :: 3.11',
    ],
    packages=["src/whisper_ctranslate2"],
    include_package_data=True,
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    entry_points={
        "console_scripts": [
            "whisper-ctranslate2=src.whisper_ctranslate2.whisper_ctranslate2:main",
        ]
    },
)

