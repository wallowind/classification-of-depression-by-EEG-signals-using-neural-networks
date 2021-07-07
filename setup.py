from setuptools import setup, find_packages

setup(
    name="Classification of depression by EEG signals using neural networks",
    version="0.0.1",
    author="Viktor Sapunov",
    author_email="wallowind@gmail.com",
    package_dir={"": "cdenn"},
    packages=find_packages(where="cdenn"),
    url="https://github.com/wallowind/classification-of-depression-by-EEG-signals-using-neural-networks",
    description="A collection of neural networks for the classification of an open EEG dataset in depression.",
    license="MIT",
    install_requires=["torch == 1.4.0",
                      "mne >= 0.22.1",
                      "numpy >= 1.19.1",
                      "tqdm >= 4.48.0"]
)
