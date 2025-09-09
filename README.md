# 🎙️ Transcription with Diarization and Confidence Scoring

This project provides a **complete audio processing pipeline** that performs:
- **Speech Transcription** (via [WhisperX](https://github.com/m-bain/whisperX))
- **Speaker Diarization** (who spoke when, via [pyannote.audio](https://github.com/pyannote/pyannote-audio))
- **Confidence Scoring** (word-level scores)
- **Audio Enhancement** (denoising, resampling)

It is containerized with **Docker** for reproducibility and uses **GPU acceleration** (CUDA) for fast inference.

---

## 🖥️ System Requirements

### Hardware
- **GPU:** NVIDIA GPU with CUDA support (tested on CUDA **12.4**)
- **VRAM:** Minimum **8 GB** (16 GB+ recommended for large models like `large-v2`)
- **RAM:** 16 GB system memory or more
- **Disk:** At least 5 GB free for models, logs, and outputs

### Software
- **OS:** Ubuntu 22.04 (native or via WSL2)
- **Docker Desktop** (with WSL2 integration enabled)
- **NVIDIA Drivers** (latest, must match CUDA version)
- **NVIDIA Container Toolkit** (installed automatically by setup script)

---

## 🚀 Quick Start

### 1. Clone repository
```bash
git clone https://github.com/yourusername/audio-td.git
cd audio-td
```

### 2. Setup environment
Run the automated setup script:


```bash
chmod +x setup.sh
./setup.sh
```

### This will:
Verify NVIDIA GPU availability (nvidia-smi)
Check Docker installation and user permissions
Install NVIDIA Container Toolkit for GPU passthrough
Create project folders (input/, output/, models/, logs/)
Generate .env for secrets
Build the Docker image with CUDA support

### 3. Configure environment variables
Hugging Face Token
Open .env (created automatically) and add your token:

```bash
HUGGING_FACE_TOKEN=your_hf_token_here
```
#### 👉 You can create a free token at Hugging Face Settings.

#### Whisper Model
Choose a Whisper model size (tradeoff between speed & accuracy):

```bash
export WHISPER_MODEL=large-v2
```

### 4. Prepare your audio
Put your input file (.wav, .mp3, etc.) in the input/ directory.
#### Example:

```bash
input/noisy_audio.mp3
```

### 5. Run transcription with diarization
Build and run the pipeline:

```bash
docker compose build
docker compose run --rm audio-td python main.py "input/noisy_audio.mp3" --min_speakers 4 --max_speakers 5
```
#### Arguments:

--min_speakers → minimum expected speakers

--max_speakers → maximum expected speakers

## Outputs
After processing, check the output/ folder:
enhanced_for_asr.wav → audio cleaned & resampled
original_for_diarization.wav → original audio for diarization
diarized_transcription.json → final structured result

### Example JSON line:
```bash
{
  "speaker": "SPEAKER_01",
  "word": "Hello",
  "start": "00:00.500",
  "end": "00:01.200",
  "confidence": 0.94
}
```
#### Each entry contains:

Speaker label
Word spoken
Start & end timestamps (mm:ss.sss format)
Confidence score

##  Development
Project Structure
```bash
├── setup.sh              # Automated setup script
├── docker-compose.yml    # Docker configuration
├── Dockerfile            # Base image and environment
├── requirements.txt      # Python dependencies
├── main.py               # Main pipeline (recommended entry point)
├── main_01.py            # Alternative pipeline (simpler version)
├── src/                  # Core processing modules
│   ├── asr_pipeline.py       # Automatic Speech Recognition pipeline
│   ├── audio_enhancer.py     # Noise reduction & audio enhancement
│   └── diarization_pipeline.py # Speaker diarization pipeline
├── input/                # Place your input audio files here
├── output/               # Generated outputs
├── models/               # Downloaded ML models
├── logs/                 # Runtime logs
└── .env                  # Environment variables (Hugging Face token)

main.py → full pipeline with enhanced error handling, JSONL output (line-by-line JSON), and flexible speaker constraints.
```


## Dependencies
Defined in requirements.txt:

Core ML: torch, torchaudio, whisperx, pyannote.audio

Audio Processing: pydub, librosa, demucs, speechbrain

Data & Utils: datasets, pandas, huggingface-hub, tqdm


## Notes & Tips
GPU is mandatory for performance — CPU-only mode is not supported for long audios.
Adjust --min_speakers / --max_speakers for more accurate diarization.
Large Whisper models (large-v2) need ≥16GB VRAM.
.gitignore ensures input/, output/, and .env are not tracked.
For debugging, check logs inside logs/.

## License
MIT License (update if needed)

## Acknowledgements
OpenAI Whisper

WhisperX

pyannote.audio

Speechbrain


---





