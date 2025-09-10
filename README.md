Technical Analysis: End-to-End ASR & Diarization Pipeline
1. Introduction
This document provides a technical analysis of the end-to-end ASR and speaker diarization pipeline built to process a complex, noisy audio file. The project utilizes containerization for reproducibility and a modular architecture to allow for iterative improvements and model selection.

2. Pipeline Architecture
The final pipeline architecture is a parallelized system designed to leverage the strengths of different models for each task.

Path 1 (for ASR): The original audio is passed through Demucs, a source separation model, to isolate speech from a complex background of in-vehicle, instrument, and public noise. The enhanced audio is then sent to WhisperX for transcription.

Path 2 (for Diarization): The original, unprocessed audio is passed directly to the pyannote.audio diarization pipeline. This was a critical adjustment to ensure diarization accuracy.

Final Output: The results from both paths are then merged to produce a structured output with word-level timestamps, speaker labels, and confidence scores.

3. Iterative Problem-Solving
The final pipeline was the result of an iterative process to overcome specific challenges presented by the audio sample.

Initial Challenge: My first attempt used SpeechBrain for waveform enhancement. While it reduced background noise, it often aggressively cut out parts of the speech, negatively impacting ASR.

Switch to Demucs: I switched to Demucs, a source separation model, to better handle the complex noise profile. This produced a smoother audio output with less speech truncation, leading to better ASR results, especially at the beginning of the audio.

The Diarization Challenge: I found that preprocessing the audio with Demucs for ASR negatively impacted the diarization model. Demucs, by separating sources, altered the voice quality in a way that made it difficult for pyannote to distinguish between speakers. This confirmed that preserving the original voice characteristics was crucial for accurate diarization. The current pipeline's parallel approach, which feeds the original audio directly to the diarization model, significantly improved results compared to the initial attempts with enhanced audio.

Persistent Diarization Issues: Even with the refined parallel pipeline, the diarization model still struggles with two specific, persistent issues:

It often fails to differentiate between two different male voices that have a similar pitch and tone.

A single male speaker is incorrectly assigned a new speaker label when their pitch or tone shifts significantly (e.g., due to emphasis or a change in emotion), treating them as two separate individuals.

Fine-Tuning: I further improved the diarization results by experimenting with pyannote's fine-tuning parameters, such as the clustering_threshold, to balance the need for speaker separation with the risk of mislabeling.

4. Limitations & Future Work
Despite the improvements, the results are still not entirely accurate due to the highly challenging nature of the audio. The ASR still struggles with sections of heavy noise and reverberation.

My next step is to replace the pre-processing step with CleanMel, a model specifically designed to improve Mel-spectrograms. My hypothesis is that this approach will provide a cleaner input for both the ASR and diarization models, leading to better overall accuracy without the trade-offs observed with waveform-level enhancement.