import os
import sys
import json
import logging
import torch
import whisperx
import argparse
from pydub import AudioSegment
from src.asr_pipeline import ASRPipeline
from src.diarization_pipeline import DiarizationPipeline
from src.audio_enhancer import ParallelAudioEnhancer

# Configure logging for clear and consistent output.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
# The target sample rate for the ASR model.
TARGET_SR = 16000

# Rely on the environment variable for the Hugging Face token.
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
if not HUGGING_FACE_TOKEN:
    logging.critical("HUGGING_FACE_TOKEN environment variable is not set. Please set it.")
    sys.exit(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WHISPER_MODEL = os.getenv("WHISPER_MODEL")
if not WHISPER_MODEL:
    logging.critical("WHISPER_MODEL environment variable is not set. Please set it.")
    sys.exit(1)

# --- UTILITY FUNCTIONS ---
def format_time_to_mm_ss(seconds: float) -> str:
    """
    Converts a time in seconds to a string in mm:ss.sss format.
    """
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes:02d}:{remaining_seconds:02.0f}"

def log_audio_processing_stats(asr_result, diarization_result, original_duration):
    """Log processing statistics to verify full audio coverage."""
    # Check ASR coverage
    asr_words = [word for segment in asr_result.get("segments", []) for word in segment.get("words", [])]
    if asr_words:
        asr_end = max(word.get("end", 0) for word in asr_words)
        logging.info(f"ASR processed: 0 to {asr_end:.2f}s of {original_duration:.2f}s audio")

    # Check diarization coverage
    if diarization_result is not None and not diarization_result.empty:
        diar_end = diarization_result["end"].max()
        logging.info(f"Diarization processed: 0 to {diar_end:.2f}s of {original_duration:.2f}s audio")

def process_and_save_results(merged_result, diarization_result, output_dir, base_filename):
    """Formats and saves the final transcription and diarization results."""
    # Format and save transcription as JSON Lines
    output_file_name = os.path.join(output_dir, f"{base_filename}_diarized_transcription.json")
    with open(output_file_name, 'w', encoding='utf-8') as f:
        for segment in merged_result.get("segments", []):
            for word_info in segment.get("words", []):
                structured_output = {
                    "speaker": word_info.get("speaker", "Unknown Speaker"),
                    "word": word_info.get("word", ""),
                    "start": format_time_to_mm_ss(word_info.get("start", 0)),
                    "end": format_time_to_mm_ss(word_info.get("end", 0)),
                    "confidence": word_info.get("score", "N/A")
                }
                f.write(json.dumps(structured_output, ensure_ascii=False) + '\n')
    logging.info(f"Final structured transcription saved to: {output_file_name}")

    # No longer saving the detailed diarization JSON file as requested.
    if diarization_result is not None and not diarization_result.empty:
        unique_speakers = diarization_result['speaker'].unique()
        logging.info(f"Final result: {len(unique_speakers)} unique speakers detected: {list(unique_speakers)}")
        for speaker in unique_speakers:
            total_time = diarization_result[diarization_result['speaker'] == speaker]['duration'].sum()
            num_segments = len(diarization_result[diarization_result['speaker'] == speaker])
            logging.info(f"{speaker}: {num_segments} segments, {total_time:.2f}s total speaking time")

def main():
    """
    Main function to run the audio processing pipeline with advanced diarization presets.
    """
    try:
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        parser = argparse.ArgumentParser(
            description="Run the audio transcription and diarization pipeline with advanced presets.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Advanced Diarization Options:
Use presets for common scenarios:
    --diar_preset similar_voices         : For distinguishing similar male voices
    --diar_preset pitch_variation_robust : For handling pitch variations
    --diar_preset high_sensitivity       : For maximum sensitivity
    --diar_preset conservative           : For conservative detection
    --diar_preset silence_aggressive     : To aggressively mark long periods of silence
    --diar_preset speech_transition_robust: For robust detection of new sentences after silence
            """
        )
        parser.add_argument("audio_file_path", help="Path to the input audio file.")

        # Basic speaker parameters
        parser.add_argument("--min_speakers", type=int, default=2, help="Minimum number of speakers to detect.")
        parser.add_argument("--max_speakers", type=int, default=5, help="Maximum number of speakers to detect.")
        parser.add_argument("--diar_preset", type=str, default="speech_transition_robust",
                            choices=["similar_voices", "pitch_variation_robust", "high_sensitivity", "conservative", "silence_aggressive", "speech_transition_robust"],
                            help="Use predefined parameter preset for common scenarios, or None for default.")

        args = parser.parse_args()

        # Initialize pipelines once
        enhancer = ParallelAudioEnhancer(device=DEVICE)
        asr_pipeline = ASRPipeline(model_size=WHISPER_MODEL, device=DEVICE)
        diarization_pipeline = DiarizationPipeline(hf_token=HUGGING_FACE_TOKEN, device=DEVICE)

        # Load audio from command line input
        if not os.path.exists(args.audio_file_path):
            logging.critical(f"Audio file not found at: {args.audio_file_path}")
            sys.exit(1)

        full_audio = AudioSegment.from_file(args.audio_file_path)
        base_filename = os.path.splitext(os.path.basename(args.audio_file_path))[0]
        original_duration = full_audio.duration_seconds

        logging.info(f"Audio loaded with sample rate: {full_audio.frame_rate} Hz, Channels: {full_audio.channels}")

        # --- Parallel Audio Enhancement ---
        logging.info("Starting parallel audio enhancement...")
        asr_audio_segment = enhancer.enhance_for_asr(full_audio)
        diarization_audio_segment = enhancer.enhance_for_diarization(full_audio)

        # --- Save Original and Preprocessed Audio Files ---
        logging.info("Saving original and preprocessed audio files to output directory...")
        full_audio.export(os.path.join(output_dir, f"{base_filename}_original.wav"), format="wav")
        asr_audio_segment.export(os.path.join(output_dir, f"{base_filename}_demucs_preprocessed.wav"), format="wav")
        diarization_audio_segment.export(os.path.join(output_dir, f"{base_filename}_deepfilternet_preprocessed.wav"), format="wav")
        logging.info("Audio files saved successfully.")

        # --- ASR Transcription and Alignment ---
        logging.info("Starting ASR transcription and alignment on Demucs output...")
        aligned_result = asr_pipeline.transcribe_and_align_in_memory(asr_audio_segment)
        if not aligned_result:
            logging.error("Transcription and alignment failed.")
            sys.exit(1)

        # --- Speaker Diarization ---
        logging.info("Starting speaker diarization on DeepFilterNet output...")
        diarization_result = diarization_pipeline.process_audio_with_preset(
            diarization_audio_segment,
            preset_name=args.diar_preset,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers
        )

        if diarization_result is None or diarization_result.empty:
            logging.error("Diarization failed or returned no segments.")
            pass

        # Log processing stats
        log_audio_processing_stats(aligned_result, diarization_result, original_duration)

        # Merge ASR and diarization results
        logging.info("Merging ASR and diarization results...")
        if diarization_result is not None and not diarization_result.empty:
            merged_result = whisperx.assign_word_speakers(diarization_result, aligned_result)
        else:
            logging.warning("No diarization result to merge. Using un-diarized ASR output for final save.")
            merged_result = aligned_result
            diarization_result = None

        # Format and save final results
        process_and_save_results(merged_result, diarization_result, output_dir, base_filename)

    except Exception as e:
        logging.critical(f"An unrecoverable error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
