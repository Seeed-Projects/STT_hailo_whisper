"""
Preprocessing functions for Whisper audio data.

This module provides functions for preparing audio data for Whisper model inference.
It handles chunking long audio into appropriate segments, converting audio to mel
spectrograms, and formatting the data for neural network processing.

The main preprocessing pipeline includes:
- Audio chunking to appropriate lengths for model input
- Conversion to log-Mel spectrograms using standard parameters
- Reshaping for neural network input formats (NHWC vs NCHW)
- Voice activity detection to identify speech segments
- Audio gain adjustment for low-level signals
"""

import common.audio_utils
import numpy as np
import logging


def preprocess(audio, is_nhwc=False, chunk_length=10, chunk_offset=0, max_duration=60, overlap=0.0):
    """
    Generate mel spectrograms from audio for Whisper model input.

    This function splits the input audio into overlapping chunks, converts each chunk
    to a log-Mel spectrogram, and formats the data for neural network processing.
    It handles audio segmentation to match model input requirements and applies
    appropriate reshaping for different neural network architectures.

    Args:
        audio (numpy.ndarray): Input audio waveform as a 1D array
        is_nhwc (bool): Whether to use NHWC format instead of NCHW (default: False)
        chunk_length (int): Length in seconds of each audio chunk to process (default: 10)
        chunk_offset (float): Position in seconds to start processing (default: 0)
        max_duration (int): Maximum duration of audio to process in seconds (default: 60)
        overlap (float): Overlap ratio between chunks (default: 0.0)

    Returns:
        list: List of mel spectrogram arrays, one for each audio chunk
    """
    # Get the sample rate from audio utils to calculate time-based operations
    sample_rate = common.audio_utils.SAMPLE_RATE
    
    # Calculate maximum number of samples based on max duration
    max_samples = max_duration * sample_rate
    
    # Calculate offset in samples based on chunk offset in seconds
    offset = int(chunk_offset * sample_rate)

    # Define parameters for audio chunking
    segment_duration = chunk_length  # Duration of each segment in seconds
    segment_samples = segment_duration * sample_rate  # Number of samples per segment
    step = int(segment_samples * (1 - overlap))  # Step size between segments (accounts for overlap)

    # Extract the portion of audio to process based on offset and maximum duration
    audio = audio[offset:max_samples]
    
    # List to hold the mel spectrograms for each chunk
    mel_spectrograms = []

    # Process the audio in overlapping chunks
    for start in range(0, len(audio), step):
        end = int(start + segment_samples)
        
        # Skip if the start position exceeds audio length
        if start >= len(audio):
            break
            
        # Extract the current chunk
        chunk = audio[start:end]

        # Ensure the chunk has the required duration by padding or trimming
        # This is necessary because Whisper models expect fixed-length inputs
        chunk = common.audio_utils.pad_or_trim(chunk, int(segment_duration * sample_rate))

        # Convert the audio chunk to log-Mel spectrogram representation
        mel = common.audio_utils.log_mel_spectrogram(chunk).to("cpu")
        # The result is moved to CPU as the Hailo pipeline expects CPU tensors

        # Add batch dimension (axis 0) to create shape (1, n_mels, n_frames)
        mel = np.expand_dims(mel, axis=0)
        
        # Add another dimension at axis 2 to match expected input shape (1, 80, 1, 1000)
        mel = np.expand_dims(mel, axis=2) 

        # If NHWC format is requested, transpose from (batch, channels, height, width) 
        # to (batch, height, width, channels)
        if is_nhwc:
            mel = np.transpose(mel, [0, 2, 3, 1])

        # Add the processed mel spectrogram to the list
        mel_spectrograms.append(mel)

    return mel_spectrograms


def apply_gain(audio, gain_db):
    """
    Apply gain to the audio signal to adjust its amplitude.

    This function increases or decreases the amplitude of an audio signal
    using a gain factor specified in decibels. Positive gain values amplify
    the signal, while negative values attenuate it.

    Args:
        audio (numpy.ndarray): Input audio waveform as a 1D array
        gain_db (float): Gain to apply in decibels (dB)

    Returns:
        numpy.ndarray: Audio waveform with gain applied
    """
    # Convert gain from decibels to linear scale
    # Formula: linear_gain = 10^(dB_gain/20) for amplitude scaling
    gain_linear = 10 ** (gain_db / 20)
    
    # Apply the gain by multiplying each sample by the linear gain factor
    return audio * gain_linear


def improve_input_audio(audio, vad=True, low_audio_gain=True):
    """
    Improve the input audio by applying gain adjustment and detecting speech.

    This function provides preprocessing to enhance audio quality before
    Whisper inference. It can apply gain to low-level signals and detect
    the start time of speech using voice activity detection.

    Args:
        audio (numpy.ndarray): Input audio waveform as a 1D array
        vad (bool): Whether to perform voice activity detection (default: True)
        low_audio_gain (bool): Whether to apply gain to low-level signals (default: True)

    Returns:
        tuple: (improved_audio, start_time) where start_time is the time of
               first detected speech (0 if VAD is disabled)
    """
    
    # Check if gain adjustment is enabled and audio level is low
    if (low_audio_gain == True) and (np.max(audio) < 0.1):
        # Apply different gain levels based on current max amplitude
        if np.max(audio) < 0.1:
            # Apply 20 dB gain for very low signals
            audio = apply_gain(audio, gain_db=20)  # Increase by 20 dB
        elif np.max(audio) < 0.2:
            # Apply 10 dB gain for somewhat low signals
            audio = apply_gain(audio, gain_db=10)  # Increase by 10 dB
        print(f"New max audio level: {np.max(audio)}")

    # Initialize the start time to 0
    start_time = 0
    
    # Perform voice activity detection if enabled
    if vad:
        # Detect when speech first occurs in the audio
        start_time = detect_first_speech(
            audio, 
            common.audio_utils.SAMPLE_RATE, 
            threshold=0.2, 
            frame_duration=0.2
        )
        
        # Log the detection result
        if start_time is not None:
            logging.info(f"Speech detected at {start_time:.2f} seconds.")
        else:
            logging.info("No speech detected.")
    
    # Return the potentially enhanced audio and start time
    return audio, start_time


def detect_first_speech(audio_data, sample_rate, threshold=0.2, frame_duration=0.02):
    """
    Detect the first time when human speech occurs in audio data using energy-based VAD.

    This function implements a simple voice activity detection (VAD) algorithm by
    analyzing the energy level of audio frames. It processes the audio in short
    segments and identifies the first segment with energy above a threshold,
    indicating the presence of speech.

    Args:
        audio_data (numpy.ndarray): Audio samples as a 1D or 2D array
        sample_rate (int): Sample rate of the audio in Hz
        threshold (float): Energy threshold for speech detection (default: 0.2)
        frame_duration (float): Duration of each analysis frame in seconds (default: 0.02)

    Returns:
        float or None: Time in seconds when speech is first detected, or None if no speech found
    """
    # Convert stereo audio to mono by averaging channels if necessary
    if len(audio_data.shape) == 2:
        audio_data = np.mean(audio_data, axis=1)

    # Calculate frame size in samples based on duration and sample rate
    frame_size = int(frame_duration * sample_rate)

    # Split the audio into overlapping frames for analysis
    frames = [audio_data[i:i + frame_size] for i in range(0, len(audio_data), frame_size)]

    # Calculate the average energy of each frame using root-mean-square method
    # Energy = sum of squared samples divided by number of samples in frame
    energy = [np.sum(np.abs(frame)**2) / len(frame) for frame in frames]

    # Normalize energy values to range [0, 1] for consistent thresholding
    max_energy = max(energy)
    if max_energy > 0:
        energy = [e / max_energy for e in energy]
    
    #print(energy)  # Debug print (commented out in production)
    
    # Detect the first frame with energy above the threshold
    # TODO: Add noise floor estimation for the threshold to improve accuracy
    for i, e in enumerate(energy):
        if e > threshold:
            # Calculate the start time based on frame index and duration
            start_time = i * frame_duration
            # Round to 1 decimal place for consistency
            start_time_rounded = round(start_time, 1)
            return start_time_rounded

    return None  # Return None if no speech detected above threshold