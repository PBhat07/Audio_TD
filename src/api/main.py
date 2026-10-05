import os 
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from pathlib import Path
import tempfile

from fastapi import HTTPException

from src.pipeline.audio_pipeline import AudioPipeline

ALLOWED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".mp4",
}

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
WHISPER_MODEL = os.getenv("WHISPER_MODEL")

pipeline = AudioPipeline(
    hf_token=HUGGING_FACE_TOKEN,
    whisper_model=WHISPER_MODEL,
)

app = FastAPI(
    title="Audio Transcription API",
    description="API for audio transcription, speaker diarization, and confidence scoring.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    """Return the service health status."""
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
        file: UploadFile = File(
        ...,
        description="Audio file to transcribe and diarize.",
    ),
    min_speakers: int | None = Form(
        None,
        description="Optional minimum number of speakers.",
    ),
    max_speakers: int | None = Form(
        None,
        description="Optional maximum number of speakers.",
    ),
    diarization_preset: str = Form(
        "high_sensitivity",
        description="Diarization sensitivity preset.",
    ),
 ):
    """
Transcribe an audio file, perform speaker diarization,
and save the final speaker-attributed transcription as JSON.
"""

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    suffix = Path(file.filename).suffix or ".wav"
    
    if suffix.lower() not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported audio format: {suffix}. "
                f"Supported formats: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
            ),
        )
        
    if min_speakers is not None and min_speakers < 1:
        raise HTTPException(
            status_code=400,
            detail="min_speakers must be at least 1.",
        )

    if max_speakers is not None and max_speakers < 1:
        raise HTTPException(
            status_code=400,
            detail="max_speakers must be at least 1.",
        )

    if (
        min_speakers is not None
        and max_speakers is not None
        and min_speakers > max_speakers
    ):
        raise HTTPException(
            status_code=400,
            detail="min_speakers cannot be greater than max_speakers.",
        )    

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
            content = await file.read()
            temp_file.write(content)

        result = pipeline.process(
            audio_file_path=temp_path,
            base_filename=Path(file.filename).stem,
            diar_preset=diarization_preset,
            min_speakers=min_speakers or 2,
            max_speakers=max_speakers or 5,
        )

        return {
            "status": "completed",
            "filename": file.filename,
            "duration": result.original_duration,
            "transcription_url": f"/transcribe/{Path(file.filename).stem}",
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Audio processing failed. Please try again later.",
        )

    finally:
        if "temp_path" in locals():
            Path(temp_path).unlink(missing_ok=True)
            
@app.get("/transcribe/{filename}")
def download_transcription(filename: str):
    """Retrieve the generated speaker-attributed transcription JSON."""

    output_path = Path("output") / f"{Path(filename).stem}_diarized_transcription.json"

    if not output_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Transcription file not found.",
        )

    return FileResponse(
        path=output_path,
        media_type="application/json",
        filename=output_path.name,
    )            