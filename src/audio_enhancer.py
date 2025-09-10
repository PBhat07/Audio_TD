import logging
import torch
import os
import tempfile
from pydub import AudioSegment
from typing import Optional
import numpy as np
import pyloudnorm
from functools import lru_cache

# Set up logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

AUDIO_LIBS_AVAILABLE = False
try:
    from df.enhance import enhance, init_df
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    # DeepFilterNet is highly sensitive to the sample rate
    DEEPFILTERNET_SR = 16000
    # Demucs models require specific sample rates
    DEMUCS_SR = 44100
    TARGET_SR = 16000

    _demucs_model = None
    AUDIO_LIBS_AVAILABLE = True

    @lru_cache(maxsize=1)
    def _get_deepfilternet_model(device: str):
        """Initializes and caches the DeepFilterNet model."""
        logging.info("Initializing DeepFilterNet model...")
        model, df_state, _ = init_df(model_base_dir=None, log_level='info')
        model = model.to(device)
        model.eval()
        logging.info("DeepFilterNet model initialized.")
        return model, df_state

    @lru_cache(maxsize=1)
    def _get_demucs_model(device: str):
        """Initializes and caches the Demucs model."""
        logging.info("Initializing Demucs model...")
        model = get_model(name='htdemucs_6s')
        model = model.to(device)
        model.eval()
        logging.info("Demucs model initialized.")
        return model

except ImportError as e:
    logging.critical(f"Required audio enhancement libraries not found: {e}. Please install them.")
    AUDIO_LIBS_AVAILABLE = False


class ParallelAudioEnhancer:
    """
    Performs parallel audio enhancement tailored for ASR and diarization.
    """
    def __init__(self, atten_lim_db: float = -30.0, device: str = "cpu"):
        """
        Initializes the enhancer.
        """
        if not AUDIO_LIBS_AVAILABLE:
            raise RuntimeError("Audio enhancement dependencies are not installed.")

        self.device = device
        self.atten_lim_db = atten_lim_db
        self.df_model, self.df_state = _get_deepfilternet_model(self.device)
        self.demucs_model = _get_demucs_model(self.device)

    def _convert_to_tensor(self, audio_segment: AudioSegment) -> torch.Tensor:
        """
        Converts an AudioSegment to a PyTorch tensor, ensuring it is 2-channel and
        properly formatted for Demucs.
        """
        # Ensure stereo (Demucs expects 2 channels)
        if audio_segment.channels == 1:
            audio_segment = audio_segment.set_channels(2)

        np_data = np.array(audio_segment.get_array_of_samples())
        max_val = np.iinfo(np_data.dtype).max if np.issubdtype(np_data.dtype, np.integer) else 1.0
        float_data = np_data.astype(np.float32) / max_val

        # Reshape to (channels, samples) and add batch dimension.
        audio_tensor = torch.from_numpy(float_data.reshape(-1, audio_segment.channels).T).unsqueeze(0)
        return audio_tensor.to(self.device)

    def _normalize_volume(self, audio_segment: AudioSegment, target_lufs: float = -18.0) -> AudioSegment:
        """
        Loudness normalization using pyloudnorm.
        """
        np_data = np.array(audio_segment.get_array_of_samples())
        max_val = np.iinfo(np_data.dtype).max if np.issubdtype(np_data.dtype, np.integer) else 1.0
        float_data = np_data.astype(np.float32) / max_val

        try:
            meter = pyloudnorm.Meter(audio_segment.frame_rate)
            loudness = meter.integrated_loudness(float_data)
            gain_to_apply = target_lufs - loudness
            normalized_audio = audio_segment.apply_gain(gain_to_apply)
            logging.info(f"Normalized from {loudness:.2f} LUFS → {target_lufs} LUFS.")
            return normalized_audio
        except Exception as e:
            logging.error(f"Volume normalization failed: {e}", exc_info=True)
            return audio_segment

    def enhance_for_asr(self, audio_segment: AudioSegment) -> AudioSegment:
        """
        Enhances audio for ASR using Demucs to isolate vocals.
        """
        if self.demucs_model is None:
            logging.error("Demucs model not loaded.")
            return audio_segment

        logging.info("Enhancing audio for ASR using Demucs...")
        try:
            # Step 1: Resample and convert to tensor for Demucs.
            processed_audio = audio_segment.set_frame_rate(DEMUCS_SR)
            audio_tensor = self._convert_to_tensor(processed_audio)

            # Step 2: Apply Demucs separation.
            with torch.no_grad():
                separated_stems = apply_model(self.demucs_model, audio_tensor, progress=True, split=True, overlap=0.25)

            # Step 3: Select vocals stem.
            # Demucs standard order is ['drums', 'bass', 'other', 'vocals'].
            vocals_tensor = separated_stems[0, 3, :, :]

            # Step 4: Convert back to pydub AudioSegment.
            vocals_numpy = vocals_tensor.detach().cpu().numpy()

            # Downmix to mono if stereo
            if vocals_numpy.ndim == 2 and vocals_numpy.shape[0] == 2:
                vocals_numpy = vocals_numpy.mean(axis=0)
            elif vocals_numpy.ndim == 2 and vocals_numpy.shape[0] == 1:
                vocals_numpy = vocals_numpy[0]

            # Clip and scale to int16
            clipped_np = np.clip(vocals_numpy, -1.0, 1.0)
            enhanced_int16 = (clipped_np * 32767).astype(np.int16)

            enhanced_audio = AudioSegment(
                enhanced_int16.tobytes(),
                frame_rate=DEMUCS_SR,
                sample_width=2,
                channels=1,
            )

            # Step 5: Normalize volume and downsample for the ASR model.
            normalized_audio = self._normalize_volume(enhanced_audio)
            final_audio = normalized_audio.set_frame_rate(TARGET_SR)

            logging.info("Demucs-based enhancement for ASR complete.")
            return final_audio

        except Exception as e:
            logging.error(f"Demucs enhancement failed: {e}")
            return audio_segment

    def enhance_for_diarization(self, audio_segment: AudioSegment) -> AudioSegment:
        """
        Enhances audio for diarization using DeepFilterNet for noise reduction.
        """
        if self.df_model is None:
            logging.error("DeepFilterNet model not loaded.")
            return audio_segment

        logging.info("Enhancing audio for diarization using DeepFilterNet...")

        # DeepFilterNet requires 16kHz mono audio
        input_audio_df = audio_segment.set_frame_rate(DEEPFILTERNET_SR).set_channels(1)

        # Convert pydub AudioSegment to a torch Tensor and move to device
        audio_tensor = (
            torch.frombuffer(input_audio_df.raw_data, dtype=torch.int16)
            .float()
            .div(32768.0)
            .unsqueeze(0)  # add batch dimension -> (1, length)
            .to(self.device)
        )

        with torch.no_grad():
            enhanced_tensor = enhance(
                self.df_model, self.df_state, audio_tensor.cpu(), atten_lim_db=self.atten_lim_db
            )

        # Convert back to pydub AudioSegment
        enhanced_audio = AudioSegment(
            (enhanced_tensor.squeeze().cpu().numpy() * 32767.0).astype("int16").tobytes(),
            frame_rate=DEEPFILTERNET_SR,
            sample_width=2,
            channels=1,
        )

        # Apply normalization after enhancement
        normalized_audio = self._normalize_volume(enhanced_audio)

        logging.info("DeepFilterNet-based enhancement for diarization complete.")
        return normalized_audio
