### AudioClaasification

## General
Torchaudio is a library for audio and signal processing with PyTorch. It provides I/O, signal and data processing functions, datasets, model implementations and application components.
#
## An audio classification example
 This code first pre-processes the audio files and trains an audio segment classifier, given a set of WAV files stored in folders (each folder represents a different class).
Then, using the Telegram bot, she takes the voice of the person and says the name of the class

![photo_2023-01-11_08-34-04](https://user-images.githubusercontent.com/80622132/211721985-0475293c-6f74-4a79-ab59-315c522c9eca.jpg)

## Installation
Install dependencies: pip install -r ./requirements.txt 
#
## Dataset
Dataset contains voices from 11 classes
# Accuracy |
| :---         |     :---:      |
| Test data  | 0.9495   |
