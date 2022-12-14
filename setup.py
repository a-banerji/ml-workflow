import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="mle-workflow",
    version="0.3",
    description="MLE Training Tiger Analytics",
    long_description=long_description,
    url="https://github.com/a-banerji/mle-training",
    author="Arpit Banerji, a Tiger MLE",
    author_email="arpit.banerji@tigeranalytics.com",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "scikit-learn", "matplotlib"],
    entry_points={
        "console_scripts": [
            "ml-workflow=train:main",
        ],
    },
)
