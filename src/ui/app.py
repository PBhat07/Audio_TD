import gradio as gr

from src.ui.api_client import transcribe_audio
from src.ui.components import (
    create_audio_input_section,
    create_results_section,
    create_export_section,
)
from src.ui.transcript import render_transcript
from src.ui.styles import CUSTOM_CSS
from src.ui.interactions import CUSTOM_JS


def build_app() -> gr.Blocks:
    """Build and configure the Gradio application."""

    with gr.Blocks(
        title="Audio Transcription Studio",
        theme=gr.themes.Base(
            primary_hue="violet",
            secondary_hue="cyan",
            neutral_hue="slate",
        ),
        css=CUSTOM_CSS,
        js=CUSTOM_JS,
    ) as demo:

        gr.Markdown(
            """
            # Audio Transcription Studio

            AI-powered transcription and speaker analysis for reviewing
            recorded conversations.
            """
        )

        # ============================================================
        # Input
        # ============================================================

        input_components = create_audio_input_section()

        # ============================================================
        # Results
        # ============================================================

        result_components = create_results_section()

        # ============================================================
        # Export
        # ============================================================

        export_components = create_export_section()

        # ============================================================
        # Transcript search
        # ============================================================

        result_components["transcript_search"].change(
            fn=None,
            inputs=result_components["transcript_search"],
            outputs=[],
            js="(query) => window.searchTranscript(query)",
        )

        # ============================================================
        # Processing
        # ============================================================

        def process_audio(
            audio_file,
            min_speakers,
            max_speakers,
        ):
            if audio_file is None:
                return (
                    "Please upload an audio file first.",
                    None,
                    None,
                    None,
                )

            try:
                metadata, transcription, json_path = transcribe_audio(
                    audio_file=audio_file,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )

                return (
                    (
                        "### ✓ Transcription completed\n\n"
                        f"**File:** {metadata['filename']}  \n"
                        f"**Duration:** "
                        f"{metadata['duration']:.2f} seconds"
                    ),
                    audio_file,
                    render_transcript(transcription),
                    json_path,
                )

            except Exception as exc:
                return (
                    f"### Processing failed\n\n{exc}",
                    None,
                    None,
                    None,
                )

        # ============================================================
        # Transcribe button
        # ============================================================

        input_components["transcribe_button"].click(
            fn=process_audio,
            inputs=[
                input_components["audio_input"],
                input_components["min_speakers"],
                input_components["max_speakers"],
            ],
            outputs=[
                input_components["status"],
                result_components["result_audio"],
                result_components["transcript"],
                export_components["download_json"],
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_app()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )