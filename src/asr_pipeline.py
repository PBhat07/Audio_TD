# filename: src/asr_pipeline.py

import logging
import os
import torch
import whisperx
import tempfile
from pydub import AudioSegment
from typing import Dict, Any

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ASRPipeline:
    """
    An optimized pipeline for Automatic Speech Recognition (ASR) using
    the whisperx library. This class handles model loading,
    transcription, and returns word-level confidence scores.
    """

    def __init__(self, model_size: str = "medium.en", device: str = None, compute_type: str = "float16"):
        """
        Initializes the ASR pipeline by loading the core Whisper model.
        Alignment models are loaded dynamically when needed.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logging.info(f"Initializing ASR pipeline with model: {model_size} on device: {self.device}")

        try:
            self.model = whisperx.load_model(
                model_size,
                device=self.device,
                compute_type=compute_type
            )
            self.align_model = None
            self.align_metadata = None
            self._current_align_lang = None
            logging.info("Whisper ASR model loaded successfully. Alignment models will be loaded on the fly.")
        except Exception as e:
            self.model = None
            logging.critical(f"Failed to load Whisper model: {e}", exc_info=True)
            raise RuntimeError("ASR model could not be initialized. Check your device and model path.")

    def transcribe_and_align(self, audio_path: str, batch_size: int = 16) -> Dict[str, Any]:
        """
        Transcribes an audio file and performs word-level alignment.
        This method dynamically loads the alignment model based on the detected language
        to ensure efficiency and save memory.

        Args:
            audio_path (str): The path to the enhanced audio file.
            batch_size (int): The batch size for transcription.

        Returns:
            Dict[str, Any]: A dictionary containing the transcription and aligned segments,
                            or an empty dictionary if the process fails.
        """
        if not self.model or not os.path.exists(audio_path):
            logging.error("ASR model not loaded or audio file not found. Aborting transcription.")
            return {}

        try:
            logging.info(f"Starting transcription and alignment of audio file: {audio_path}")
            audio = whisperx.load_audio(audio_path)

            # Step 1: Transcribe the enhanced audio
            transcription_result = self.model.transcribe(audio, batch_size=batch_size)
            language_code = transcription_result.get("language")

            if not language_code:
                logging.error("Language detection failed. Cannot perform alignment.")
                return {}

            # Step 2: Load alignment model dynamically based on detected language
            if self.align_model is None or language_code != self._current_align_lang:
                logging.info(f"Loading alignment model for detected language: {language_code}")

                if self.align_model is not None:
                    del self.align_model
                    self.align_model = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=language_code,
                    device=self.device
                )
                self._current_align_lang = language_code
            else:
                logging.info(f"Alignment model for {language_code} already loaded. Skipping.")

            # Step 3: Align with the loaded model
            aligned_result = whisperx.align(
                transcription_result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                self.device
            )

            logging.info("Transcription and alignment complete.")
            return aligned_result

        except Exception as e:
            logging.error(f"An error occurred during transcription or alignment: {e}", exc_info=True)
            return {}

    def transcribe_and_align_in_memory(self, audio_segment: AudioSegment, batch_size: int = 16) -> Dict[str, Any]:
        """
        Transcribes and aligns an in-memory AudioSegment by exporting
        it to a temporary WAV file first.
        Returns the aligned ASR result.
        """
        if not self.model:
            logging.error("ASR model not loaded. Aborting transcription.")
            return {}

        tmp_path = None
        try:
            # Export AudioSegment to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                audio_segment.export(tmpfile.name, format="wav")
                tmp_path = tmpfile.name

            # Load audio from file path
            audio = whisperx.load_audio(tmp_path)

            # Step 1: Transcribe
            transcription_result = self.model.transcribe(audio, batch_size=batch_size)
            language_code = transcription_result.get("language")
            if not language_code:
                logging.error("Language detection failed. Cannot perform alignment.")
                return {}

            # Step 2: Load alignment model if needed
            if self.align_model is None or language_code != self._current_align_lang:
                logging.info(f"Loading alignment model for detected language: {language_code}")

                if self.align_model is not None:
                    del self.align_model
                    self.align_model = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=language_code,
                    device=self.device
                )
                self._current_align_lang = language_code

            # Step 3: Align
            aligned_result = whisperx.align(
                transcription_result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                self.device
            )

            logging.info("In-memory transcription and alignment complete.")
            return aligned_result

        except Exception as e:
            logging.error(f"Error during in-memory transcription or alignment: {e}", exc_info=True)
            return {}

        finally:
            # Clean up temporary file
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
