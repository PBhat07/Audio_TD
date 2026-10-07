import logging
import os

import torch
import whisperx
from pydub import AudioSegment

from src.asr_pipeline import ASRPipeline
from src.audio_enhancer import ParallelAudioEnhancer
from src.diarization_pipeline import DiarizationPipeline
from src.output.formatter import save_diarized_transcription
from src.pipeline.result import PipelineResult


logger = logging.getLogger(__name__)
gpu_logger = logging.getLogger("gpu_memory")


class AudioPipeline:
    """Orchestrates audio enhancement, transcription, alignment, and diarization."""

    def __init__(
        self,
        hf_token: str,
        whisper_model: str,
        device: str | None = None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.enhancer = ParallelAudioEnhancer(
            device=self.device
        )

        self.asr_pipeline = ASRPipeline(
            model_size=whisper_model,
            device=self.device,
        )

        self.diarization_pipeline = DiarizationPipeline(
            hf_token=hf_token,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # GPU MEMORY OBSERVABILITY
    # ------------------------------------------------------------------

    def _cuda_available(self) -> bool:
        """Return True when the pipeline is configured to use CUDA."""

        return (
            self.device == "cuda"
            and torch.cuda.is_available()
        )

    def _reset_gpu_peak_memory(self) -> None:
        """Reset PyTorch peak-memory statistics for a new request."""

        if self._cuda_available():
            torch.cuda.reset_peak_memory_stats()

    def _log_gpu_memory(self, stage: str) -> None:
        """
        Log current and peak GPU memory usage.

        Reports:
            - current PyTorch allocated memory
            - current PyTorch reserved memory
            - peak PyTorch allocated memory
            - peak PyTorch reserved memory
            - driver-level free GPU memory
            - total visible GPU memory
        """

        if not self._cuda_available():
            return

        device = torch.cuda.current_device()

        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)

        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

        free_memory, total_memory = torch.cuda.mem_get_info(device)

        gpu_logger.info(
            "GPU MEMORY | stage=%s | "
            "allocated=%.0f MiB | "
            "reserved=%.0f MiB | "
            "peak_allocated=%.0f MiB | "
            "peak_reserved=%.0f MiB | "
            "free=%.0f MiB | "
            "total=%.0f MiB",
            stage,
            allocated / 1024**2,
            reserved / 1024**2,
            peak_allocated / 1024**2,
            peak_reserved / 1024**2,
            free_memory / 1024**2,
            total_memory / 1024**2,
        )

    def _log_gpu_peak_summary(self) -> None:
        """Log peak GPU memory reached during the request."""

        if not self._cuda_available():
            return

        device = torch.cuda.current_device()

        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

        gpu_logger.info(
            "GPU MEMORY SUMMARY | "
            "peak_allocated=%.0f MiB | "
            "peak_reserved=%.0f MiB",
            peak_allocated / 1024**2,
            peak_reserved / 1024**2,
        )

    # ------------------------------------------------------------------
    # MAIN PIPELINE
    # ------------------------------------------------------------------

    def process(
        self,
        audio_file_path: str,
        output_dir: str = "output",
        diar_preset: str = "high_sensitivity",
        min_speakers: int = 2,
        max_speakers: int = 5,
        base_filename: str | None = None,
    ) -> PipelineResult:
        """Run the complete audio processing pipeline."""

        logger.info(
            "Starting audio pipeline: %s",
            audio_file_path,
        )

        # Start a fresh GPU-memory measurement window.
        self._reset_gpu_peak_memory()
        self._log_gpu_memory("pipeline_start")

        try:
            # ----------------------------------------------------------
            # Validate and load audio
            # ----------------------------------------------------------

            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(
                    f"Audio file not found: {audio_file_path}"
                )

            full_audio = AudioSegment.from_file(
                audio_file_path
            )

            if base_filename is None:
                base_filename = os.path.splitext(
                    os.path.basename(audio_file_path)
                )[0]

            original_duration = full_audio.duration_seconds

            logger.info(
                "Audio loaded: %.2f seconds, %d Hz, %d channel(s)",
                original_duration,
                full_audio.frame_rate,
                full_audio.channels,
            )

            # ----------------------------------------------------------
            # Audio enhancement
            # ----------------------------------------------------------

            logger.info(
                "Starting ASR audio enhancement..."
            )

            asr_audio_segment = (
                self.enhancer.enhance_for_asr(
                    full_audio
                )
            )

            self._log_gpu_memory(
                "after_demucs_release"
            )

            logger.info(
                "Starting diarization audio enhancement..."
            )

            diarization_audio_segment = (
                self.enhancer.enhance_for_diarization(
                    full_audio
                )
            )

            self._log_gpu_memory(
                "after_deepfilternet_release"
            )

            # ----------------------------------------------------------
            # Save intermediate audio artifacts
            # ----------------------------------------------------------

            os.makedirs(
                output_dir,
                exist_ok=True,
            )

            full_audio.export(
                os.path.join(
                    output_dir,
                    f"{base_filename}_original.wav",
                ),
                format="wav",
            )

            asr_audio_segment.export(
                os.path.join(
                    output_dir,
                    f"{base_filename}_demucs_preprocessed.wav",
                ),
                format="wav",
            )

            diarization_audio_segment.export(
                os.path.join(
                    output_dir,
                    f"{base_filename}_deepfilternet_preprocessed.wav",
                ),
                format="wav",
            )

            # ----------------------------------------------------------
            # ASR + alignment
            # ----------------------------------------------------------

            logger.info(
                "Starting ASR transcription and alignment..."
            )

            self._log_gpu_memory("before_asr")

            aligned_result = (
                self.asr_pipeline.transcribe_and_align_in_memory(
                    asr_audio_segment
                )
            )

            self._log_gpu_memory(
                "after_asr_alignment"
            )

            if not aligned_result:
                raise RuntimeError(
                    "Transcription and alignment failed."
                )

            logger.info(
                "ASR transcription and alignment completed."
            )

            # ----------------------------------------------------------
            # Speaker diarization
            # ----------------------------------------------------------

            logger.info(
                "Starting speaker diarization with preset '%s'...",
                diar_preset,
            )

            self._log_gpu_memory(
                "before_diarization"
            )

            diarization_result = (
                self.diarization_pipeline.process_audio_with_preset(
                    diarization_audio_segment,
                    preset_name=diar_preset,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
            )

            self._log_gpu_memory(
                "after_diarization"
            )

            if (
                diarization_result is None
                or diarization_result.empty
            ):
                raise RuntimeError(
                    "Diarization failed or returned no segments."
                )

            logger.info(
                "Speaker diarization completed."
            )

            # ----------------------------------------------------------
            # Merge ASR and speaker information
            # ----------------------------------------------------------

            merged_result = whisperx.assign_word_speakers(
                diarization_result,
                aligned_result,
            )

            # ----------------------------------------------------------
            # Coverage statistics
            # ----------------------------------------------------------

            asr_words = [
                word
                for segment in aligned_result.get(
                    "segments",
                    [],
                )
                for word in segment.get(
                    "words",
                    [],
                )
            ]

            if asr_words:
                asr_end = max(
                    word.get("end", 0)
                    for word in asr_words
                )

                logger.info(
                    "ASR coverage: %.2f / %.2f seconds",
                    asr_end,
                    original_duration,
                )

            if not diarization_result.empty:
                diarization_end = (
                    diarization_result["end"].max()
                )

                logger.info(
                    "Diarization coverage: %.2f / %.2f seconds",
                    diarization_end,
                    original_duration,
                )

            # ----------------------------------------------------------
            # Save final output
            # ----------------------------------------------------------

            output_file = save_diarized_transcription(
                merged_result=merged_result,
                output_dir=output_dir,
                base_filename=base_filename,
                original_duration=original_duration,
            )

            logger.info(
                "Final transcription saved to: %s",
                output_file,
            )

            logger.info(
                "Audio pipeline completed successfully."
            )

            return PipelineResult(
                aligned_result=aligned_result,
                diarization_result=diarization_result,
                merged_result=merged_result,
                output_file=output_file,
                base_filename=base_filename,
                original_duration=original_duration,
            )

        finally:
            # Always record final GPU state, including failures.
            self._log_gpu_memory("pipeline_end")
            self._log_gpu_peak_summary()