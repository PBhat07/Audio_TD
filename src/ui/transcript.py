import html


def format_timestamp(timestamp: str) -> str:
    """Return a timestamp in a compact display format."""

    if timestamp.startswith("00:"):
        return timestamp[3:]

    return timestamp


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert HH:MM:SS.mmm timestamp into seconds."""

    parts = timestamp.split(":")

    if len(parts) != 3:
        return 0.0

    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])

    return (
        hours * 3600
        + minutes * 60
        + seconds
    )


def render_transcript(transcription: dict) -> str:
    """
    Convert transcription JSON into interactive HTML.

    Each transcript segment contains:
    - speaker
    - timestamp
    - text
    - audio start position
    """

    segments = transcription.get("segments", [])

    if not segments:
        return """
        <div class="transcript-empty">
            No transcript segments were found.
        </div>
        """

    transcript_parts = []

    speaker_classes = {}
    speaker_index = 0

    for index, segment in enumerate(segments):

        raw_speaker = str(
            segment.get(
                "speaker",
                "Unknown Speaker",
            )
        )

        if raw_speaker not in speaker_classes:
            speaker_classes[raw_speaker] = (
                f"speaker-{speaker_index}"
            )
            speaker_index += 1

        speaker_class = speaker_classes[raw_speaker]

        speaker = html.escape(raw_speaker)

        text = html.escape(
            str(
                segment.get(
                    "text",
                    "",
                )
            ).strip()
        )

        raw_start = str(
            segment.get(
                "start",
                "00:00:00.000",
            )
        )

        start = html.escape(
            format_timestamp(raw_start)
        )

        start_seconds = timestamp_to_seconds(
            raw_start
        )

        transcript_parts.append(
            f"""
            <div
                class="transcript-segment {speaker_class}"
                data-segment-index="{index}"
                data-start="{start_seconds}"
                data-speaker="{speaker}"
                onclick="window.seekToTranscriptTime({start_seconds})"
            >

                <div class="transcript-segment-header">

                    <span
                        class="transcript-speaker"
                        onclick="event.stopPropagation(); window.renameSpeaker(this)"
                    >
                        {speaker}
                    </span>

                    <span class="transcript-timestamp">
                        {start}
                    </span>

                </div>

                <div class="transcript-text">
                    {text}
                </div>

            </div>
            """
        )

    return f"""
    <div class="transcript-container">
        {''.join(transcript_parts)}
    </div>
    """