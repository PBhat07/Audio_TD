import json
import os


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm."""

    milliseconds = round(seconds * 1000)

    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds_part, milliseconds_part = divmod(remainder, 1_000)

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds_part:02d}."
        f"{milliseconds_part:03d}"
    )


def save_diarized_transcription(
    merged_result: dict,
    output_dir: str,
    base_filename: str,
    original_duration: float,
) -> str:
    """Save the final speaker-attributed transcription as structured JSON."""

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        f"{base_filename}_diarized_transcription.json",
    )

    formatted_segments = []

    for segment in merged_result.get("segments", []):
        formatted_words = []

        for word_info in segment.get("words", []):
            formatted_words.append(
                {
                    "word": word_info.get("word", ""),
                    "start": format_timestamp(
                        word_info.get("start", 0)
                    ),
                    "end": format_timestamp(
                        word_info.get("end", 0)
                    ),
                    "confidence": word_info.get(
                        "score",
                        "N/A",
                    ),
                }
            )

        formatted_segments.append(
            {
                "speaker": segment.get(
                    "speaker",
                    "Unknown Speaker",
                ),
                "start": format_timestamp(
                    segment.get("start", 0)
                ),
                "end": format_timestamp(
                    segment.get("end", 0)
                ),
                "text": segment.get("text", "").strip(),
                "words": formatted_words,
            }
        )

    structured_output = {
    "filename": f"{base_filename}",
    "duration": format_timestamp(original_duration),
    "segments": formatted_segments,
}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            structured_output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return output_file