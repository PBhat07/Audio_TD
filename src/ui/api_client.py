import os
import json
import tempfile
from mimetypes import guess_type
from pathlib import Path

import requests


FASTAPI_URL = os.getenv(
    "FASTAPI_URL",
    "http://localhost:8000",
)


def transcribe_audio(
    audio_file: str,
    min_speakers: int,
    max_speakers: int,
) -> tuple[dict, dict, str]:
    """
    Send an audio file to the FastAPI transcription endpoint.

    Returns:
        API response metadata,
        full transcription JSON,
        path to downloadable JSON file.
    """

    mime_type, _ = guess_type(audio_file)

    if mime_type is None:
        mime_type = "application/octet-stream"

    with open(audio_file, "rb") as file:
        response = requests.post(
            f"{FASTAPI_URL}/transcribe",
            files={
                "file": (
                    Path(audio_file).name,
                    file,
                    mime_type,
                )
            },
            data={
                "min_speakers": int(min_speakers),
                "max_speakers": int(max_speakers),
            },
            timeout=1800,
        )

    response.raise_for_status()

    data = response.json()

    transcription_url = data["transcription_url"]

    result_response = requests.get(
        f"{FASTAPI_URL}{transcription_url}",
        timeout=30,
    )

    result_response.raise_for_status()

    transcription = result_response.json()

    # Save the exact JSON returned by FastAPI so it can be
    # downloaded from the Gradio interface.
    base_filename = Path(audio_file).stem

    json_path = Path(tempfile.gettempdir()) / (
        f"{base_filename}_diarized_transcription.json"
    )

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(
            transcription,
            json_file,
            indent=2,
            ensure_ascii=False,
        )

    return data, transcription, str(json_path)