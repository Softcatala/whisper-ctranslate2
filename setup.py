import pathlib
from setuptools import setup
#import pkg_resources
import os

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()


def read_version(fname="src/whisper_ctranslate2/version.py"):
    version = {}
    exec(compile(open(fname).read(), fname, "exec"), version)
    return version["__version__"]

with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as f:
    requirements = f.read().splitlines()

setup(
    name="whisper-ctranslate2",
    version=read_version(),
    description="Whisper command line client that uses CTranslate2 and faster-whisper",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Softcatala/whisper-ctranslate2",
    author="Jordi Mas",
    author_email="jmas@softcatala.org",
    #license="MIT",
    # classifiers=[
    #     "License :: OSI Approved :: MIT License",
    #     "Programming Language :: Python :: 3.6",
    #     "Programming Language :: Python :: 3.7",
    #     "Programming Language :: Python :: 3.8",
    #     "Programming Language :: Python :: 3.9",
    #     "Programming Language :: Python :: 3.10",
    #     "Programming Language :: Python :: 3.11",
    #     "Programming Language :: Python :: 3.12",
    #     "Programming Language :: Python :: 3.13",
    #     "Programming Language :: Python :: 3.14",
    #     "Topic :: Scientific/Engineering :: Artificial Intelligence",
    # ],
    packages=["src/whisper_ctranslate2"],
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "dev": ["flake8==7.*", "black==24.*", "isort==5.13", "nose2", "twine"],
    },
    entry_points={
        "console_scripts": [
            "whisper-ctranslate2=whisper_ctranslate2.whisper_ctranslate2:main",
        ]
    },
)
