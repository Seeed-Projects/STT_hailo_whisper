# Automatic Speech Recognition with Whisper model

This application performs a speech-to-text transcription using OpenAI's *Whisper-tiny* and *Whisper-base* model on the Hailo-8/8L/10H AI accelerators.

## Prerequisites

Ensure your system matches the following requirements before proceeding:

- Platforms tested: x86, Raspberry Pi 5
- OS: Ubuntu 22 (x86) or Raspberry OS.
- **HailoRT 4.20 or 4.21** and the corresponding **PCIe driver** must be installed. You can download them from the [Hailo Developer Zone](https://hailo.ai/developer-zone/)
- **ffmpeg** and **libportaudio2** installed for audio processing.
  ```
  sudo apt update
  sudo apt install ffmpeg
  sudo apt install libportaudio2
  sudo apt install portaudio19-dev python3-pyaudio
  sudo apt install hailo-all
  ```
- **Python 3.10 or 3.11** installed.

## Hardware prepare [one of them]

### [reComputer AI R2140-12](https://www.seeedstudio.com/reComputer-AI-R2140-12-p-6431.html?qid=BEN48Y_oo22igmt_1760944323400)

<div align='center'><img width={600} src='https://media-cdn.seeedstudio.com/media/catalog/product/cache/bb49d3ec4ee05b6f018e93f896b8a25d/q/q/qq_1.jpg'></div>

### [reComputer Industrial R2045-12](https://www.seeedstudio.com/reComputer-Industrial-R2045-12-p-6544.html)
<div align='center'><img width={600} src='https://media-cdn.seeedstudio.com/media/catalog/product/cache/bb49d3ec4ee05b6f018e93f896b8a25d/1/-/1-recomputer-industrail-r2000_1.jpg'></div>

## Microphone Array

### [ReSpeaker Mic Array v3.0](https://www.seeedstudio.com/ReSpeaker-Mic-Array-v3-0.html)
<div align='center'><img width={600} src='https://media-cdn.seeedstudio.com/media/catalog/product/cache/bb49d3ec4ee05b6f018e93f896b8a25d/h/t/httpsstatics3.seeedstudio.comseeedfile2018-05bazaar820383_micarrayv2.jpg'></div>

## Installation - Inference only

Follow these steps to set up the environment and install dependencies for inference:

1. Clone this repository:

   ```sh
   https://github.com/Seeed-Projects/STT_hailo_whisper
   ```
   If you have any authentication issues, add your SSH key or download the zip.

2. Activate the virtual environment from the repository root folder:

   ```sh
   python -m venv .env --system-site-packages && source .env/bin/activate
   ```

3. Install necessary model

   ```sh
   cd app && python download_resources.py
   ```

4. Install necessary lib

   ```sh
   cd .. && pip install -r requirements.txt
   ```

## Run this job

1. Run whisper for real-time STT

   ```sh
   python hailo_whisper.py --hw-arch hailo8 --variant base --udp-host 0.0.0.0 --udp-port 12345
   ```
   You can also run `python hailo_whisper.py --help` to check more information.

2. Run UDP reciver

   ```sh
   cd test & python recive_message.py --host 0.0.0.0 --port 12345 --stats-interval 5
   ```
   You can also run `python recive_message.py --help` to check more information. 
