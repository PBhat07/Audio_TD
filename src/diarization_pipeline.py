import logging
import torch
import io
import tempfile
import os
import pandas as pd
from typing import Dict, Any, Optional, Union
from pyannote.audio import Pipeline
from pydub import AudioSegment

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DiarizationPipeline:
    """
    A class to handle speaker diarization using the pyannote.audio library.
    It supports advanced parameter tuning and can process both audio files
    and in-memory pydub.AudioSegment objects.
    """

    def __init__(self, hf_token: str, device: str = None):
        """
        Initializes the diarization pipeline by loading the Pyannote model.

        Args:
            hf_token (str): Hugging Face user access token.
            device (str): The device to run the model on (e.g., "cuda" or "cpu").
        """
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Initializing DiarizationPipeline on device: {self.device}")

        try:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            self.pipeline.to(torch.device(self.device))
            logging.info("Pyannote speaker diarization model loaded successfully.")
        except Exception as e:
            logging.critical(
                f"Failed to load Pyannote diarization model. Check your Hugging Face token and model access: {e}",
                exc_info=True
            )
            self.pipeline = None

    def _configure_parameters(self, params: Dict[str, Any]):
        """
        Configures the pipeline with a given dictionary of parameters.
        """
        if not self.pipeline:
            logging.error("Pipeline not loaded. Cannot configure parameters.")
            return

        for component_name, component_params in params.items():
            try:
                if hasattr(self.pipeline, f'_{component_name}'):
                    component = getattr(self.pipeline, f'_{component_name}')
                    for param, value in component_params.items():
                        if hasattr(component, param):
                            setattr(component, param, value)
                            logging.info(f"Set '{component_name}.{param}' to {value}")
            except Exception as e:
                logging.warning(f"Failed to set parameter '{component_name}.{param}': {e}")

    def process_audio(self,
                    audio_source: Union[str, AudioSegment],
                    min_speakers: int = None,
                    max_speakers: int = None,
                      **kwargs) -> Optional[pd.DataFrame]:
        """
        Applies the diarization pipeline to an audio source (file path or in-memory object).
        """
        result = None  # default

        if not self.pipeline:
            logging.error("Diarization model not loaded. Cannot process audio.")
            return result

        # Configure parameters
        param_groups = self._group_params_by_component(kwargs)
        if param_groups:
            self._configure_parameters(param_groups)

        logging.info("Starting speaker diarization...")

        tmp_path = None
        try:
            # Handle both file paths and in-memory AudioSegment objects
            if isinstance(audio_source, AudioSegment):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    audio_source.export(tmpfile.name, format="wav")
                    tmp_path = tmpfile.name
                pipeline_kwargs = {"file": tmp_path}
            else:
                pipeline_kwargs = {"file": audio_source}

            if min_speakers is not None:
                pipeline_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                pipeline_kwargs["max_speakers"] = max_speakers

            # Run diarization
            diarization_result = self.pipeline(**pipeline_kwargs)
            logging.info("Diarization complete.")

            # Convert Annotation -> DataFrame
            data = [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": speaker,
                    "duration": segment.end - segment.start
                }
                for segment, _, speaker in diarization_result.itertracks(yield_label=True)
            ]
            df = pd.DataFrame(data)

            if not df.empty:
                result = df
            else:
                logging.warning("Diarization returned no segments.")

        except Exception as e:
            logging.error(f"An error occurred during diarization: {e}", exc_info=True)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        return result

    def get_optimization_presets(self) -> Dict[str, Dict[str, Any]]:
        """
        Get predefined parameter presets for different scenarios.
        """
        return {
            "high_sensitivity": {
                "segmentation": {"onset": 0.4, "offset": 0.2, "min_duration_on": 0.05, "min_duration_off": 0.02},
                "clustering": {"threshold": 0.5},
                "voice_activity_detection": {"onset": 0.3, "offset": 0.2}
            },
            "conservative": {
                "segmentation": {"onset": 0.6, "offset": 0.5, "min_duration_on": 0.2, "min_duration_off": 0.1},
                "clustering": {"threshold": 0.8},
                "voice_activity_detection": {"onset": 0.6, "offset": 0.5}
            },
            "similar_voices": {
                "segmentation": {"onset": 0.35, "offset": 0.25, "min_duration_on": 0.08, "min_duration_off": 0.03},
                "clustering": {"threshold": 0.45},
                "voice_activity_detection": {"onset": 0.35, "offset": 0.25}
            },
            "pitch_variation_robust": {
                "segmentation": {"onset": 0.45, "offset": 0.55, "min_duration_on": 0.01, "min_duration_off": 0.05},
                "clustering": {"threshold": 0.45},
                "voice_activity_detection": {"onset": 0.01, "offset": 0.01}
            }
        }

    def _group_params_by_component(self, params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Internal helper to group flat parameters into component-based dictionaries.
        """
        grouped = {
            "segmentation": {},
            "clustering": {},
            "voice_activity_detection": {}
        }
        for key, value in params.items():
            if key in ["onset", "offset", "min_duration_on", "min_duration_off"]:
                grouped["segmentation"][key] = value
            elif key in ["threshold"]:
                grouped["clustering"][key] = value
            elif key.startswith("vad_"):
                grouped["voice_activity_detection"][key.replace("vad_", "")] = value
        return grouped

    def process_audio_with_preset(self, audio_source: Union[str, AudioSegment], preset_name: str,
                                min_speakers: int = None, max_speakers: int = None) -> Optional[pd.DataFrame]:
        """
        Process audio using a predefined parameter preset.

        Args:
            audio_source (Union[str, AudioSegment]): The audio file path or in-memory object.
            preset_name (str): Name of the preset to use.
            min_speakers (int): Minimum number of speakers.
            max_speakers (int): Maximum number of speakers.

        Returns:
            pd.DataFrame: Diarization results or None on failure.
        """
        presets = self.get_optimization_presets()
        if preset_name not in presets:
            logging.error(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
            return None

        preset_params_nested = presets[preset_name]
        logging.info(f"Using preset '{preset_name}' with parameters: {preset_params_nested}")

        # flatten and rename parameters to match _group_params_by_component's logic
        all_params = {}
        for component, params in preset_params_nested.items():
            for key, value in params.items():
                if component == "voice_activity_detection":
                    all_params[f"vad_{key}"] = value
                else:
                    all_params[key] = value

        return self.process_audio(
            audio_source=audio_source,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            **all_params
        )