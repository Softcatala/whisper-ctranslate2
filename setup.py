import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

def read_version(fname="src/whisper_ctranslate2/version.py"):
    version = {}
    exec(compile(open(fname).read(), fname, "exec"), version)
    return version["__version__"]

def read_requirements(fname="requirements.txt"):
    """Read requirements from file and return as list."""
    requirements_file = HERE / fname
    with open(requirements_file) as f:
        return [line.strip() for line in f
                if line.strip() and not line.startswith('#')]

setup(
    name="whisper-ctranslate2",
    version=read_version(),
    description="Whisper command line client that uses CTranslate2 and faster-whisper",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Softcatala/whisper-ctranslate2",
    author="Jordi Mas",
    author_email="jmas@softcatala.org",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=read_requirements(),
    extras_require={
        "dev": ["flake8==7.*", "black==24.*", "isort==5.13", "nose2", "twine"],
    },
    entry_points={
        "console_scripts": [
            "whisper-ctranslate2=whisper_ctranslate2.whisper_ctranslate2:main",
        ]
    },
)
