# 🎙️ Audio Transcription & Speaker Diarization

A GPU-accelerated audio processing pipeline for **speech enhancement, transcription, speaker diarization, word-level confidence scoring, and structured output**.

The project combines **Demucs, DeepFilterNet, WhisperX, and pyannote.audio** into a single processing pipeline and exposes the system through a **FastAPI REST API** running inside a CUDA-enabled Docker container.

---

## 🚀 What This Project Does

Given an audio file, the system:

1. Loads and analyzes the input audio.
2. Creates task-specific enhanced audio versions.
3. Transcribes speech using **WhisperX**.
4. Aligns the transcription to obtain word-level timestamps and confidence scores.
5. Performs **speaker diarization** using pyannote.audio.
6. Assigns speakers to individual words and segments.
7. Produces a structured JSON transcription.
8. Exposes the pipeline through a REST API.

The system is designed as a reusable backend service rather than a one-off transcription script.

---

## 🏗️ Architecture

```text
                    Audio File
                        │
                        ▼
                ┌───────────────┐
                │    FastAPI    │
                │   REST API    │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │ AudioPipeline │
                └───────┬───────┘
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
       ┌─────────────┐     ┌───────────────┐
       │   Demucs    │     │ DeepFilterNet │
       │ ASR Audio   │     │ Diarization   │
       └──────┬──────┘     └───────┬───────┘
              │                    │
              ▼                    ▼
       ┌─────────────┐     ┌───────────────┐
       │  WhisperX   │     │   pyannote    │
       │ ASR + Align │     │ Diarization   │
       └──────┬──────┘     └───────┬───────┘
              │                    │
              └─────────┬──────────┘
                        ▼
               Speaker Attribution
                        │
                        ▼
                Structured JSON
```

---

## 🔍 Why Two Audio Enhancement Paths?

The pipeline does not use the same preprocessing strategy for every downstream task.

Instead, it creates task-specific audio:

- **Demucs** → preprocessing for automatic speech recognition.
- **DeepFilterNet** → preprocessing for speaker diarization.

This separation allows each downstream model to receive audio optimized for its specific task.

The enhanced audio is also saved during processing so that the results can be inspected and compared during development.

---

## 🧠 Core Pipeline

The main orchestration logic is implemented in:

```text
src/pipeline/audio_pipeline.py
```

The pipeline follows this sequence:

```text
Input Audio
    ↓
Audio Loading
    ↓
Task-Specific Enhancement
    ├── Demucs → ASR path
    └── DeepFilterNet → Diarization path
    ↓
WhisperX Transcription
    ↓
Word-Level Alignment
    ↓
Speaker Diarization
    ↓
Speaker-to-Word Assignment
    ↓
Structured JSON Output
```

The different stages communicate through Python objects and in-memory audio representations rather than requiring intermediate model stages to communicate through temporary files.

---

## 🎯 Speaker Diarization

The diarization stage uses **pyannote.audio** to identify different speakers within the audio.

The API allows optional speaker constraints:

```text
min_speakers
max_speakers
```

For example:

```text
min_speakers = 2
max_speakers = 5
```

These parameters provide additional control when the approximate number of speakers is known.

The system also supports diarization presets so that the behavior can be configured without exposing every low-level model parameter through the API.

---

## 📝 Structured Output

The final result is saved as structured JSON.

Example:

```json
{
  "filename": "noisy_audio",
  "duration": "00:04:45.344",
  "segments": [
    {
      "speaker": "Unknown Speaker",
      "start": "00:00:01.991",
      "end": "00:00:03.511",
      "text": "We're gonna need it for the game.",
      "words": [
        {
          "word": "We're",
          "start": "00:00:01.991",
          "end": "00:00:02.331",
          "confidence": 0.423
        }
      ]
    }
  ]
}
```

Each word can contain:

- Word text
- Start timestamp
- End timestamp
- Confidence score

Each segment contains:

- Speaker
- Start timestamp
- End timestamp
- Transcribed text
- Word-level information

This structured format makes the output suitable for downstream applications.

---

# 🌐 REST API

The processing pipeline is exposed through **FastAPI**.

## Available Endpoints

### Health Check

```http
GET /health
```

Returns:

```json
{
  "status": "ok"
}
```

---

### Transcribe Audio

```http
POST /transcribe
```

Accepts an audio file and optional speaker constraints.

Supported formats:

```text
.wav
.mp3
.m4a
.flac
.ogg
.mp4
```

Optional parameters:

```text
min_speakers
max_speakers
diarization_preset
```

Example response:

```json
{
  "status": "completed",
  "filename": "noisy_audio.mp3",
  "duration": 285.344,
  "transcription_url": "/transcribe/noisy_audio"
}
```

The API validates the request before starting the GPU-heavy processing pipeline.

---

### Retrieve Transcription

```http
GET /transcribe/{filename}
```

Returns the generated speaker-attributed JSON file.

---

### Interactive API Documentation

FastAPI automatically provides interactive documentation at:

```text
http://localhost:8000/docs
```

OpenAPI schema is available at:

```text
http://localhost:8000/openapi.json
```

---

# 🐳 Docker & GPU Setup

The application is containerized using Docker and runs with NVIDIA GPU support.

The Docker environment contains:

- CUDA runtime/development environment
- Python
- PyTorch
- WhisperX
- pyannote.audio
- Demucs
- DeepFilterNet
- FastAPI
- Uvicorn
- FFmpeg and audio processing dependencies

## Start the Application

Build and start the service with:

```bash
docker compose up --build
```

The API will be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

## ⚡ Model Caching

Large machine learning models are downloaded during their first use.

The Docker Compose configuration uses persistent volumes for model caches:

```text
torch_cache
huggingface_cache
deepfilternet_cache
```

This means that recreating the application container does not require downloading all models again.

Once the service is running, the models are initialized when the application starts and can be reused across API requests.

This avoids repeatedly loading expensive models for every request.

---

# 💻 Hardware

The project has been developed and tested on an **NVIDIA RTX 4050 laptop GPU with 6 GB VRAM**.

Because of the limited VRAM, the project uses a smaller Whisper model configuration:

```text
WHISPER_MODEL=small.en
```

The exact performance depends on the audio duration, model configuration, and available GPU memory.

---

# 📁 Project Structure

```text
Audio_TD/
│
├── src/
│   ├── api/
│   │   └── main.py
│   │
│   ├── pipeline/
│   │   ├── audio_pipeline.py
│   │   └── result.py
│   │
│   ├── output/
│   │   └── formatter.py
│   │
│   ├── asr_pipeline.py
│   ├── diarization_pipeline.py
│   └── audio_enhancer.py
│
├── main.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

# 🔧 Technology Stack

### Machine Learning

- PyTorch
- WhisperX
- pyannote.audio
- Demucs
- DeepFilterNet

### Audio Processing

- Pydub
- Librosa
- SoundFile
- SciPy

### Backend

- FastAPI
- Uvicorn
- Python

### Infrastructure

- Docker
- NVIDIA CUDA
- Docker Compose

---

# 🧩 Key Engineering Decisions

| Decision | Reason |
|---|---|
| FastAPI | Exposes the ML pipeline as a reusable service |
| Task-specific enhancement | Different downstream tasks benefit from different preprocessing |
| In-memory model pipeline | Reduces unnecessary intermediate file I/O between ML stages |
| Models initialized once | Avoids repeatedly loading large models for every request |
| Persistent model caches | Prevents repeated model downloads when containers are recreated |
| Structured JSON output | Makes results easier to consume programmatically |
| API-level validation | Prevents invalid requests from reaching expensive GPU processing |
| Docker + CUDA | Provides a reproducible GPU execution environment |
| Optional speaker constraints | Allows users to provide prior knowledge about the recording |

---

# 🛡️ API Validation

The API performs basic validation before starting the processing pipeline.

Examples include:

- Unsupported audio formats are rejected.
- Missing filenames are rejected.
- `min_speakers` must be at least `1`.
- `max_speakers` must be at least `1`.
- `min_speakers` cannot be greater than `max_speakers`.

This prevents avoidable errors before expensive GPU inference begins.

---

# 📊 Example Use Case

The pipeline is intended for scenarios where a raw recording needs to be converted into structured, speaker-attributed information.

For example:

```text
Raw Meeting / Conversation
          ↓
Noise / Source Enhancement
          ↓
Speech Recognition
          ↓
Word-Level Alignment
          ↓
Speaker Diarization
          ↓
Speaker Attribution
          ↓
Structured JSON
```

The resulting JSON can then be consumed by another application, stored in a database, searched, summarized, or passed to a downstream NLP/LLM system.

---

# ⚠️ Current Limitations

This project is currently optimized for local GPU execution rather than large-scale production deployment.

Current limitations include:

- GPU acceleration is strongly recommended.
- Large audio files require significant processing time and GPU memory.
- Model downloads can be large during first-time setup.
- The current API stores generated transcription files locally.
- The project currently focuses on the core inference pipeline rather than distributed processing.

These limitations are intentional for the current portfolio version of the project.

---

# 🚧 Future Improvements

Potential future improvements include:

- Lightweight web frontend for uploading audio and viewing results.
- Public demonstration using precomputed example outputs.
- More robust job management for long-running audio files.
- Background processing for asynchronous requests.
- Persistent database storage for transcription metadata.
- Cloud deployment when GPU infrastructure is available.
- Additional evaluation metrics for transcription and diarization quality.

---

# 🎬 Demo

A lightweight demonstration will be provided using example audio and precomputed results.

The full GPU inference pipeline can be run locally using Docker and an NVIDIA GPU.

---

# 📚 Acknowledgements

This project builds upon several open-source machine learning and audio-processing projects:

- [WhisperX](https://github.com/m-bain/whisperX)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [Demucs](https://github.com/facebookresearch/demucs)
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- [PyTorch](https://pytorch.org/)

---

# 👩‍💻 Project Focus

This project focuses on the engineering challenges involved in turning multiple deep-learning audio models into a reusable processing service.

The main areas demonstrated are:

- Audio preprocessing
- Speech recognition
- Word-level alignment
- Speaker diarization
- Model orchestration
- GPU inference
- REST API design
- Docker-based deployment
- Model caching
- Structured ML outputs
- Input validation and error handling