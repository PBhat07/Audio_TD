import logging
import os

import torch
import whisperx


from pydub import AudioSegment
from src.asr_pipeline import ASRPipeline
from src.diarization_pipeline import DiarizationPipeline
from src.audio_enhancer import ParallelAudioEnhancer
from src.output.formatter import save_diarized_transcription
from src.pipeline.result import PipelineResult


logger = logging.getLogger(__name__)


class AudioPipeline:
    """Orchestrates the complete audio transcription pipeline."""

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

    def process(
        self,
        audio_file_path: str,
        output_dir: str = "output",
        diar_preset: str = "high_sensitivity",
        min_speakers: int = 2,
        max_speakers: int = 5,
    ) -> PipelineResult:
        """Load audio and run the enhancement stages."""
        
        logger.info("Starting audio pipeline: %s", audio_file_path)

        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(
                f"Audio file not found: {audio_file_path}"
            )

        # Load the original audio
        full_audio = AudioSegment.from_file(audio_file_path)

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

        # Create the two enhanced versions used by downstream models
        logger.info("Starting ASR audio enhancement...")
        
        asr_audio_segment = self.enhancer.enhance_for_asr(
            full_audio
        )
        
        logger.info("Starting diarization audio enhancement...")
        diarization_audio_segment = self.enhancer.enhance_for_diarization(
            full_audio
        )
        

        os.makedirs(output_dir, exist_ok=True)

        full_audio.export(
            os.path.join(
                output_dir,
                f"{base_filename}_original.wav"
            ),
            format="wav"
        )

        asr_audio_segment.export(
            os.path.join(
                output_dir,
                f"{base_filename}_demucs_preprocessed.wav"
            ),
            format="wav"
        )

        diarization_audio_segment.export(
            os.path.join(
                output_dir,
                f"{base_filename}_deepfilternet_preprocessed.wav"
            ),
            format="wav"
        )
        
        logger.info("Starting ASR transcription and alignment...")
        
        aligned_result = self.asr_pipeline.transcribe_and_align_in_memory(
            asr_audio_segment
        )
        

        if not aligned_result:
            raise RuntimeError(
                "Transcription and alignment failed."
            )
        logger.info("ASR transcription and alignment completed.")  
        
        logger.info("Starting speaker diarization with preset '%s'...",
                            diar_preset,
                        )  
            
        diarization_result = self.diarization_pipeline.process_audio_with_preset(
            diarization_audio_segment,
            preset_name=diar_preset,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        
        

        if diarization_result is None or diarization_result.empty:
            raise RuntimeError(
                "Diarization failed or returned no segments."
            )
            
        logger.info("Speaker diarization completed.")    
            
        merged_result = whisperx.assign_word_speakers(
            diarization_result,
            aligned_result,
        )
        
        asr_words = [
            word
            for segment in aligned_result.get("segments", [])
            for word in segment.get("words", [])
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
            diarization_end = diarization_result["end"].max()

            logger.info(
                "Diarization coverage: %.2f / %.2f seconds",
                diarization_end,
                original_duration,
            )
        
        output_file = save_diarized_transcription(
            merged_result=merged_result,
            output_dir=output_dir,
            base_filename=base_filename,
        )
        logger.info("Final transcription saved to: %s", output_file)
        
        logger.info("Audio pipeline completed successfully.")        
        
        return PipelineResult(
            aligned_result=aligned_result,
            diarization_result=diarization_result,
            merged_result=merged_result,
            output_file=output_file,
            base_filename=base_filename,
            original_duration=original_duration,
        )