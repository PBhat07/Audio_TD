import logging
import os
import torch
import whisperx
import tempfile
from pydub import AudioSegment
from typing import Dict, Any

# Configure logging for this module
logger = logging.getLogger(__name__)


class ASRPipeline:
    """
    An optimized pipeline for Automatic Speech Recognition (ASR) using
    the whisperx library. This class handles model loading,
    transcription, and returns word-level confidence scores.
    """

    def __init__(
        self,
        model_size: str = "medium.en",
        device: str = None,
        compute_type: str = "float16"
    ):
        """
        Initializes the ASR pipeline by loading the core Whisper model.
        Alignment models are loaded dynamically when needed.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(
            f"Initializing ASR pipeline with model: "
            f"{model_size} on device: {self.device}"
        )

        try:
            self.model = whisperx.load_model(
                model_size,
                device=self.device,
                compute_type=compute_type
            )

            # Alignment model is intentionally not loaded at startup.
            # It is loaded only after Whisper detects the language.
            self.align_model = None
            self.align_metadata = None
            self._current_align_lang = None

            logger.info(
                "Whisper ASR model loaded successfully. "
                "Alignment models will be loaded on the fly."
            )

        except Exception as e:
            self.model = None
            logger.critical(
                f"Failed to load Whisper model: {e}",
                exc_info=True
            )
            raise RuntimeError(
                "ASR model could not be initialized. "
                "Check your device and model path."
            )

    def transcribe_and_align(
        self,
        audio_path: str,
        batch_size: int = 4
    ) -> Dict[str, Any]:
        """
        Transcribes an audio file and performs word-level alignment.

        The alignment model is loaded dynamically based on the detected
        language and explicitly released after alignment to avoid keeping
        temporary alignment weights in GPU memory between requests.

        Args:
            audio_path (str): The path to the enhanced audio file.
            batch_size (int): The batch size for transcription.

        Returns:
            Dict[str, Any]: A dictionary containing the transcription and
                            aligned segments, or an empty dictionary if
                            the process fails.
        """
        if not self.model or not os.path.exists(audio_path):
            logger.error(
                "ASR model not loaded or audio file not found. "
                "Aborting transcription."
            )
            return {}

        try:
            logger.info(
                f"Starting transcription and alignment of audio file: "
                f"{audio_path}"
            )

            audio = whisperx.load_audio(audio_path)

            # Step 1: Transcribe the enhanced audio
            transcription_result = self.model.transcribe(
                audio,
                batch_size=batch_size
            )

            language_code = transcription_result.get("language")

            if not language_code:
                logger.error(
                    "Language detection failed. Cannot perform alignment."
                )
                return {}

            # Step 2: Load alignment model dynamically based on detected language
            if (
                self.align_model is None
                or language_code != self._current_align_lang
            ):
                logger.info(
                    f"Loading alignment model for detected language: "
                    f"{language_code}"
                )

                # Release a previous alignment model before loading
                # another language model.
                self.release_alignment_model()

                self.align_model, self.align_metadata = (
                    whisperx.load_align_model(
                        language_code=language_code,
                        device=self.device
                    )
                )

                self._current_align_lang = language_code

            else:
                logger.info(
                    f"Alignment model for {language_code} "
                    "already loaded. Skipping."
                )

            # Step 3: Align
            aligned_result = whisperx.align(
                transcription_result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                self.device
            )

            logger.info("Transcription and alignment complete.")

            return aligned_result

        except Exception as e:
            logger.error(
                f"An error occurred during transcription or alignment: {e}",
                exc_info=True
            )
            return {}

        finally:
            # The alignment model is temporary and should not remain
            # resident on the GPU after the request.
            self.release_alignment_model()

    def transcribe_and_align_in_memory(
        self,
        audio_segment: AudioSegment,
        batch_size: int = 2
    ) -> Dict[str, Any]:
        """
        Transcribes and aligns an in-memory AudioSegment by exporting
        it to a temporary WAV file first.

        The alignment model is explicitly released after alignment.

        Returns:
            Dict[str, Any]: The aligned ASR result.
        """
        if not self.model:
            logger.error(
                "ASR model not loaded. Aborting transcription."
            )
            return {}

        tmp_path = None

        try:
            # Export AudioSegment to a temporary WAV file
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ) as tmpfile:
                audio_segment.export(
                    tmpfile.name,
                    format="wav"
                )
                tmp_path = tmpfile.name

            # Load audio from file path
            audio = whisperx.load_audio(tmp_path)

            # Step 1: Transcribe
            #
            # Batch size is deliberately kept small because this pipeline
            # runs on a 6 GB RTX 4050.
            transcription_result = self.model.transcribe(
                audio,
                batch_size=batch_size
            )

            language_code = transcription_result.get("language")

            if not language_code:
                logger.error(
                    "Language detection failed. Cannot perform alignment."
                )
                return {}

            # Step 2: Load alignment model if needed
            if (
                self.align_model is None
                or language_code != self._current_align_lang
            ):
                logger.info(
                    f"Loading alignment model for detected language: "
                    f"{language_code}"
                )

                # Release any previous alignment model before loading
                # another one.
                self.release_alignment_model()

                self.align_model, self.align_metadata = (
                    whisperx.load_align_model(
                        language_code=language_code,
                        device=self.device
                    )
                )

                self._current_align_lang = language_code

            else:
                logger.info(
                    f"Alignment model for {language_code} "
                    "already loaded. Skipping."
                )

            # Step 3: Align
            aligned_result = whisperx.align(
                transcription_result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                self.device
            )

            logger.info(
                "In-memory transcription and alignment complete."
            )

            return aligned_result

        except Exception as e:
            logger.error(
                "Error during in-memory transcription or alignment: "
                f"{e}",
                exc_info=True
            )
            return {}

        finally:
            # Remove temporary WAV file
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

            # Release temporary alignment model and free CUDA cache.
            self.release_alignment_model()

    def release_alignment_model(self):
        """
        Explicitly release the temporary WhisperX alignment model.

        The main Whisper model remains loaded because it is reused across
        requests. Only the dynamically loaded alignment model is released.
        """
        if self.align_model is not None:
            logger.info("Releasing WhisperX alignment model.")
            del self.align_model
            self.align_model = None

        self.align_metadata = None
        self._current_align_lang = None

        if self.device == "cuda":
            torch.cuda.empty_cache()

            logger.info(
                "WhisperX alignment model released and CUDA cache cleared."
            )
