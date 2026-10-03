import argparse
import logging
import os
import sys

from src.logging_config import configure_logging
from src.pipeline.audio_pipeline import AudioPipeline


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the audio transcription and diarization pipeline."
    )

    parser.add_argument(
        "audio_file_path",
        help="Path to the input audio file.",
    )

    parser.add_argument(
        "--min_speakers",
        type=int,
        default=2,
        help="Minimum number of speakers to detect.",
    )

    parser.add_argument(
        "--max_speakers",
        type=int,
        default=5,
        help="Maximum number of speakers to detect.",
    )

    parser.add_argument(
        "--diar_preset",
        type=str,
        default="high_sensitivity",
        choices=[
            "similar_voices",
            "pitch_variation_robust",
            "high_sensitivity",
            "conservative",
        ],
        help="Diarization optimization preset.",
    )

    return parser.parse_args()


def main():
    """Application entry point."""
    configure_logging()

    logger = logging.getLogger(__name__)

    try:
        args = parse_args()

        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if not hf_token:
            logger.critical(
                "HUGGING_FACE_TOKEN environment variable is not set."
            )
            return 1

        whisper_model = os.getenv("WHISPER_MODEL")
        if not whisper_model:
            logger.critical(
                "WHISPER_MODEL environment variable is not set."
            )
            return 1

        pipeline = AudioPipeline(
            hf_token=hf_token,
            whisper_model=whisper_model,
        )

        result = pipeline.process(
            audio_file_path=args.audio_file_path,
            diar_preset=args.diar_preset,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )

        logger.info(
            "Pipeline completed successfully: %s",
            result.output_file,
        )

        return 0

    except Exception:
        logger.critical(
            "Pipeline execution failed.",
            exc_info=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())