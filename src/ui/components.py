import gradio as gr


def create_audio_input_section() -> dict:
    """Create the audio upload and processing controls."""

    with gr.Group(elem_classes=["input-card"]):

        gr.Markdown(
            """
            ## Upload & Settings

            Upload an audio file and configure the expected speaker range.
            """
        )

        audio_input = gr.File(
            label="Audio file",
            file_types=[
                ".wav",
                ".mp3",
                ".m4a",
                ".flac",
                ".ogg",
                ".mp4",
            ],
            type="filepath",
            elem_classes=["audio-upload"],
        )

        gr.Markdown(
            """
            **Speaker settings**

            Define the minimum and maximum number of speakers expected
            in the recording.
            """
        )

        with gr.Row(elem_classes=["speaker-settings"]):

            min_speakers = gr.Number(
                label="Minimum speakers",
                value=2,
                precision=0,
                minimum=1,
                maximum=20,
            )

            max_speakers = gr.Number(
                label="Maximum speakers",
                value=5,
                precision=0,
                minimum=1,
                maximum=20,
            )

        transcribe_button = gr.Button(
            "Transcribe audio",
            variant="primary",
            elem_classes=["transcribe-button"],
        )

        status = gr.Markdown(
            elem_classes=["processing-status"],
        )

    return {
        "audio_input": audio_input,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "transcribe_button": transcribe_button,
        "status": status,
    }


def create_results_section() -> dict:
    """Create the audio player and transcript review area."""

    with gr.Group(elem_classes=["results-card"]):

        gr.Markdown(
            """
            ## Audio

            Listen to the original recording while reviewing the transcript.
            """
        )

        result_audio = gr.Audio(
            label="Audio",
            type="filepath",
            interactive=False,
            elem_classes=["audio-player"],
        )

        gr.Markdown(
            """
            ## Transcript

            Review the conversation, search the transcript, and click
            a transcript segment to jump to that point in the audio.

            <span class="speaker-help">
            Click a speaker name <strong>✎</strong> to rename it.
            </span>
            """
        )

        transcript_search = gr.Textbox(
            label="Search transcript",
            placeholder="Search speaker or transcript text...",
            interactive=True,
            elem_classes=["transcript-search"],
        )

        transcript = gr.HTML(
            value="""
            <div class="transcript-empty">
                <p>
                    Your transcript will appear here after processing.
                </p>
            </div>
            """,
            elem_classes=["transcript-output"],
        )

    return {
        "result_audio": result_audio,
        "transcript_search": transcript_search,
        "transcript": transcript,
    }


def create_export_section() -> dict:
    """Create the transcription export area."""

    with gr.Group(elem_classes=["export-card"]):

        gr.Markdown(
            """
            ## Export

            Download the complete transcription for detailed review
            or further analysis.

            The JSON includes speaker labels, segment timestamps,
            **word-level timestamps**, and **word-level confidence scores**.
            """
        )

        download_json = gr.File(
            label="Download transcription JSON",
            interactive=False,
            elem_classes=["download-json"],
        )

    return {
        "download_json": download_json,
    }