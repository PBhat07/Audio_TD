from dataclasses import dataclass
from typing import Any


@dataclass
class PipelineResult:
    """Results produced by the audio processing pipeline."""

    aligned_result: dict[str, Any]
    diarization_result: Any
    merged_result: dict[str, Any]
    output_file: str

    base_filename: str
    original_duration: float
