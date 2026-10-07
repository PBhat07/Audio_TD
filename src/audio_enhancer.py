import logging

import numpy as np
import pyloudnorm
import torch
from pydub import AudioSegment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio configuration
# ---------------------------------------------------------------------------

DEMUCS_SR = 44_100
DEEPFILTERNET_SR = 48_000
TARGET_SR = 16_000


# ---------------------------------------------------------------------------
# Optional dependency loading
# ---------------------------------------------------------------------------

try:
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from df.enhance import enhance, init_df

    AUDIO_LIBS_AVAILABLE = True

except ImportError as exc:
    AUDIO_LIBS_AVAILABLE = False

    logger.critical(
        "Required audio enhancement libraries are unavailable: %s",
        exc,
    )


# ---------------------------------------------------------------------------
# Audio enhancer
# ---------------------------------------------------------------------------


class ParallelAudioEnhancer:
    """
    Enhance audio for ASR and speaker diarization.

    GPU memory strategy
    -------------------
    Large enhancement models are kept on CPU when idle and staged onto the
    GPU only while they are actively processing audio.

    Demucs:
        CPU audio -> GPU tensor -> Demucs -> CPU audio

    DeepFilterNet:
        CPU audio -> CPU feature extraction -> device-side inference
        -> CPU audio -> 16 kHz mono audio

    DeepFilterNet performs part of its feature extraction through NumPy,
    so its time-domain input must remain on the CPU.
    """

    def __init__(
        self,
        atten_lim_db: float = -30.0,
        device: str = "cpu",
    ) -> None:
        if not AUDIO_LIBS_AVAILABLE:
            raise RuntimeError(
                "Audio enhancement dependencies are not installed."
            )

        self.device = device
        self.atten_lim_db = atten_lim_db

        # ------------------------------------------------------------------
        # DeepFilterNet
        # ------------------------------------------------------------------

        self.df_model, self.df_state, _ = init_df(
            model_base_dir=None,
            log_level="info",
            log_file=None,
        )

        self.df_model = self.df_model.to(self.device)
        self.df_model.eval()

        # Keep the explicit configuration constant for readability while
        # verifying that it matches the loaded DeepFilterNet state.
        model_sr = int(self.df_state.sr())

        if model_sr != DEEPFILTERNET_SR:
            raise RuntimeError(
                "DeepFilterNet sample-rate mismatch: "
                f"configured={DEEPFILTERNET_SR} Hz, "
                f"model={model_sr} Hz."
            )

        # ------------------------------------------------------------------
        # Demucs
        # ------------------------------------------------------------------

        # Demucs remains on CPU until ASR enhancement begins.
        self.demucs_model = get_model(
            name="htdemucs_6s"
        )
        self.demucs_model = self.demucs_model.to("cpu")
        self.demucs_model.eval()

        # Keep DeepFilterNet on CPU while idle when CUDA is being used.
        if self.device == "cuda":
            self._release_deepfilternet_gpu_memory()

        logger.info(
            "Audio enhancer initialized | device=%s | "
            "Demucs=CPU idle | DeepFilterNet=CPU idle | "
            "DeepFilterNet sample rate=%s Hz",
            self.device,
            DEEPFILTERNET_SR,
        )

    # ------------------------------------------------------------------
    # GPU lifecycle
    # ------------------------------------------------------------------

    def _move_model_to_gpu(
        self,
        model: torch.nn.Module | None,
        name: str,
    ) -> None:
        """Move an enhancement model to CUDA when GPU processing is enabled."""

        if self.device != "cuda" or model is None:
            return

        logger.info(
            "GPU STAGE | loading %s onto CUDA",
            name,
        )

        model.to("cuda")

    def _release_model_from_gpu(
        self,
        model: torch.nn.Module | None,
        name: str,
    ) -> None:
        """Move an enhancement model to CPU and release unused CUDA cache."""

        if self.device != "cuda" or model is None:
            return

        model.to("cpu")
        torch.cuda.empty_cache()

        logger.info(
            "GPU STAGE | %s moved to CPU; unused CUDA cache released",
            name,
        )

    def _release_demucs_gpu_memory(self) -> None:
        """Release Demucs from GPU after ASR enhancement."""

        self._release_model_from_gpu(
            self.demucs_model,
            "Demucs",
        )

    def _release_deepfilternet_gpu_memory(self) -> None:
        """Release DeepFilterNet from GPU after diarization enhancement."""

        self._release_model_from_gpu(
            self.df_model,
            "DeepFilterNet",
        )

    # ------------------------------------------------------------------
    # Audio conversion
    # ------------------------------------------------------------------

    def _convert_to_tensor(
        self,
        audio_segment: AudioSegment,
    ) -> torch.Tensor:
        """
        Convert a pydub AudioSegment to Demucs input format.

        Returns:
            Float32 tensor with shape:
            [batch, channels, samples]
        """

        if audio_segment.channels == 1:
            audio_segment = audio_segment.set_channels(2)

        samples = np.asarray(
            audio_segment.get_array_of_samples()
        )

        if np.issubdtype(samples.dtype, np.integer):
            scale = np.iinfo(samples.dtype).max
            samples = samples.astype(
                np.float32
            ) / scale
        else:
            samples = samples.astype(
                np.float32,
                copy=False,
            )

        # pydub stores multichannel audio as:
        #
        # [L, R, L, R, ...]
        #
        # Convert to:
        #
        # [channels, samples]
        audio_tensor = torch.from_numpy(
            samples.reshape(
                -1,
                audio_segment.channels,
            ).T
        ).unsqueeze(0)

        return audio_tensor.to(self.device)

    # ------------------------------------------------------------------
    # Audio normalization
    # ------------------------------------------------------------------

    def _normalize_volume(
        self,
        audio_segment: AudioSegment,
        target_lufs: float = -18.0,
    ) -> AudioSegment:
        """Normalize audio loudness to the requested integrated LUFS level."""

        samples = np.asarray(
            audio_segment.get_array_of_samples()
        )

        if np.issubdtype(samples.dtype, np.integer):
            scale = np.iinfo(samples.dtype).max
            float_data = samples.astype(
                np.float32
            ) / scale
        else:
            float_data = samples.astype(
                np.float32,
                copy=False,
            )

        try:
            meter = pyloudnorm.Meter(
                audio_segment.frame_rate
            )

            loudness = meter.integrated_loudness(
                float_data
            )

            gain_db = target_lufs - loudness

            normalized_audio = audio_segment.apply_gain(
                gain_db
            )

            logger.info(
                "Audio normalization | %.2f LUFS -> %.2f LUFS",
                loudness,
                target_lufs,
            )

            return normalized_audio

        except Exception:
            logger.exception(
                "Volume normalization failed; "
                "returning original audio."
            )

            return audio_segment

    # ------------------------------------------------------------------
    # Demucs enhancement
    # ------------------------------------------------------------------

    def enhance_for_asr(
        self,
        audio_segment: AudioSegment,
    ) -> AudioSegment:
        """
        Enhance audio for ASR using Demucs vocal separation.

        Processing:
            CPU audio -> GPU Demucs inference -> CPU audio
        """

        if self.demucs_model is None:
            logger.error(
                "Demucs model is unavailable; "
                "returning original audio."
            )
            return audio_segment

        logger.info(
            "ASR enhancement started | model=Demucs"
        )

        audio_tensor = None
        separated_stems = None
        vocals = None

        try:
            # --------------------------------------------------------------
            # 1. Stage Demucs on GPU
            # --------------------------------------------------------------

            self._move_model_to_gpu(
                self.demucs_model,
                "Demucs",
            )

            # --------------------------------------------------------------
            # 2. Prepare audio
            # --------------------------------------------------------------

            processed_audio = audio_segment.set_frame_rate(
                DEMUCS_SR
            )

            audio_tensor = self._convert_to_tensor(
                processed_audio
            )

            # --------------------------------------------------------------
            # 3. Run Demucs
            # --------------------------------------------------------------

            with torch.inference_mode():
                separated_stems = apply_model(
                    self.demucs_model,
                    audio_tensor,
                    progress=False,
                    split=True,
                    overlap=0.25,
                )

            # --------------------------------------------------------------
            # 4. Extract vocals
            # --------------------------------------------------------------
            #
            # htdemucs_6s stem order:
            # drums, bass, other, vocals, guitar, piano
            #

            vocals = separated_stems[
                0,
                3,
            ]

            vocals_numpy = (
                vocals
                .detach()
                .cpu()
                .numpy()
            )

            # --------------------------------------------------------------
            # 5. Convert to mono
            # --------------------------------------------------------------

            if vocals_numpy.ndim == 2:
                vocals_numpy = vocals_numpy.mean(
                    axis=0
                )

            # --------------------------------------------------------------
            # 6. Convert to PCM16
            # --------------------------------------------------------------

            vocals_numpy = np.clip(
                vocals_numpy,
                -1.0,
                1.0,
            )

            enhanced_int16 = (
                vocals_numpy * 32767.0
            ).astype(np.int16)

            enhanced_audio = AudioSegment(
                enhanced_int16.tobytes(),
                frame_rate=DEMUCS_SR,
                sample_width=2,
                channels=1,
            )

            # --------------------------------------------------------------
            # 7. Normalize and resample for ASR
            # --------------------------------------------------------------

            normalized_audio = self._normalize_volume(
                enhanced_audio
            )

            final_audio = normalized_audio.set_frame_rate(
                TARGET_SR
            )

            logger.info(
                "ASR enhancement completed | model=Demucs"
            )

            return final_audio

        except Exception:
            logger.exception(
                "Demucs ASR enhancement failed; "
                "returning original audio."
            )

            return audio_segment

        finally:
            # Release references to request-specific tensors before
            # moving the model back to CPU.
            audio_tensor = None
            separated_stems = None
            vocals = None

            self._release_demucs_gpu_memory()

    # ------------------------------------------------------------------
    # DeepFilterNet enhancement
    # ------------------------------------------------------------------

    def enhance_for_diarization(
        self,
        audio_segment: AudioSegment,
    ) -> AudioSegment:
        """
        Enhance audio for speaker diarization using DeepFilterNet.

        DeepFilterNet expects audio at its configured 48 kHz sample rate.

        Its feature-extraction path converts the time-domain input to NumPy,
        so the input tensor must remain on CPU. DeepFilterNet then performs
        its neural processing on the configured device.

        Processing:
            CPU audio
                -> CPU feature extraction
                -> device-side neural inference
                -> CPU audio
                -> 16 kHz mono output
        """

        if self.df_model is None:
            logger.error(
                "DeepFilterNet model is unavailable; "
                "returning original audio."
            )
            return audio_segment

        logger.info(
            "Diarization enhancement started | "
            "model=DeepFilterNet"
        )

        audio_tensor = None
        enhanced_tensor = None

        try:
            # --------------------------------------------------------------
            # 1. Stage DeepFilterNet on the configured device
            # --------------------------------------------------------------

            self._move_model_to_gpu(
                self.df_model,
                "DeepFilterNet",
            )

            # --------------------------------------------------------------
            # 2. Prepare 48 kHz mono audio
            # --------------------------------------------------------------

            input_audio = (
                audio_segment
                .set_frame_rate(DEEPFILTERNET_SR)
                .set_channels(1)
            )

            # DeepFilterNet's feature extraction calls .numpy() on the
            # time-domain input. Keep this tensor on CPU.
            audio_tensor = (
                torch.frombuffer(
                    input_audio.raw_data,
                    dtype=torch.int16,
                )
                .float()
                .div_(32768.0)
                .unsqueeze(0)
            )

            # --------------------------------------------------------------
            # 3. Run DeepFilterNet
            # --------------------------------------------------------------

            with torch.inference_mode():
                enhanced_tensor = enhance(
                    self.df_model,
                    self.df_state,
                    audio_tensor,
                    atten_lim_db=self.atten_lim_db,
                )

            # --------------------------------------------------------------
            # 4. Convert output to NumPy
            # --------------------------------------------------------------

            enhanced_numpy = (
                enhanced_tensor
                .detach()
                .cpu()
                .squeeze()
                .numpy()
            )

            enhanced_numpy = np.clip(
                enhanced_numpy,
                -1.0,
                1.0,
            )

            enhanced_int16 = (
                enhanced_numpy * 32767.0
            ).astype(np.int16)

            enhanced_audio = AudioSegment(
                enhanced_int16.tobytes(),
                frame_rate=DEEPFILTERNET_SR,
                sample_width=2,
                channels=1,
            )

            # --------------------------------------------------------------
            # 5. Normalize and resample for diarization
            # --------------------------------------------------------------

            normalized_audio = self._normalize_volume(
                enhanced_audio
            )

            final_audio = normalized_audio.set_frame_rate(
                TARGET_SR
            )

            logger.info(
                "Diarization enhancement completed | "
                "model=DeepFilterNet"
            )

            return final_audio

        except Exception:
            logger.exception(
                "DeepFilterNet diarization enhancement failed; "
                "returning original audio."
            )

            return audio_segment

        finally:
            # Release request-specific tensors before moving the model
            # back to CPU.
            audio_tensor = None
            enhanced_tensor = None

            self._release_deepfilternet_gpu_memory()