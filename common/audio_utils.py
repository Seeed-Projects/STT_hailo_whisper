"""
Audio utilities for the Whisper speech recognition pipeline.

This module provides functions for audio processing that are copied from the
original Whisper repository. These utilities handle core audio operations:
- Loading and resampling audio files
- Converting audio to appropriate formats for neural network processing
- Computing mel-spectrograms from audio waveforms
- Padding/trimming audio to required lengths

The constants and functions are designed to match the preprocessing used in
the original Whisper model to ensure compatibility with pre-trained models.
"""
# copied from Whisper repo
import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F


def exact_div(x, y):
    """
    Perform exact integer division with assertion that remainder is zero.

    This function ensures that x is evenly divisible by y before performing
    the division. It's used for calculating audio processing parameters that
    must be whole numbers.

    Args:
        x (int): Dividend
        y (int): Divisor

    Returns:
        int: Result of x // y

    Raises:
        AssertionError: If x is not evenly divisible by y
    """
    assert x % y == 0
    return x // y

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Load an audio file and convert it to a mono waveform at the specified sample rate.

    This function uses ffmpeg to decode various audio formats, downmix to mono,
    and resample to the target sample rate. The resulting audio is normalized
    to floating-point values between -1.0 and 1.0.

    Args:
        file (str): Path to the audio file to load
        sr (int): Target sample rate for resampling (default: 16000)

    Returns:
        numpy.ndarray: 1D array containing the audio waveform as float32 values in [-1.0, 1.0]

    Raises:
        RuntimeError: If ffmpeg fails to decode the audio file
    """
    # Launch ffmpeg subprocess to decode and resample audio
    # This command does the following:
    # - Disables stdin to prevent interactive prompts
    # - Uses all available CPU threads for decoding
    # - Reads from input file
    # - Outputs raw 16-bit little-endian samples
    # - Downmixes to mono channel
    # - Uses PCM signed 16-bit codec
    # - Resamples to target sample rate
    # - Outputs to stdout for capture
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    
    try:
        # Execute the command and capture output
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        # Raise an error if ffmpeg fails
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    # Convert raw 16-bit samples to float32 in range [-1.0, 1.0]
    # 1. Convert from bytes to int16 array
    # 2. Flatten to 1D array
    # 3. Convert to float32
    # 4. Normalize by dividing by max int16 value (32768)
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to the expected length as required by the Whisper encoder.

    This function ensures that audio arrays have exactly the right length for model input.
    If the array is longer than the target length, it's trimmed. If shorter, it's padded
    with zeros. This maintains compatibility with the fixed-size input expected by Whisper.

    Args:
        array (numpy.ndarray or torch.Tensor): Input audio array to adjust
        length (int): Target length (default: N_SAMPLES = 480,000 for 30 seconds of 16kHz audio)
        axis (int): Axis along which to perform padding/trimming (default: -1, last axis)

    Returns:
        numpy.ndarray or torch.Tensor: Array with adjusted length, same type as input
    """
    if torch.is_tensor(array):
        # Handle PyTorch tensors
        if array.shape[axis] > length:
            # Trim the tensor along the specified axis to the target length
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            # Pad the tensor with zeros to reach the target length
            pad_widths = [(0, 0)] * array.ndim  # Start with no padding on all dimensions
            pad_widths[axis] = (0, length - array.shape[axis])  # Add padding only on target axis
            # Reorder padding dimensions for PyTorch's F.pad function (reversed order)
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        # Handle NumPy arrays
        if array.shape[axis] > length:
            # Trim the array along the specified axis to the target length
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            # Pad the array with zeros to reach the target length
            pad_widths = [(0, 0)] * array.ndim  # Start with no padding on all dimensions
            pad_widths[axis] = (0, length - array.shape[axis])  # Add padding only on target axis
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    Load the mel filterbank matrix for projecting STFT into a Mel spectrogram.

    This function caches the loaded filters to avoid repeated file I/O. The mel filterbank
    matrix transforms frequency-domain representations (STFT) to mel-scale frequencies,
    which better match human auditory perception. The filters are precomputed and stored
    to avoid depending on librosa at runtime.

    The filter files were originally generated using librosa with the following command:
        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )

    Args:
        device: PyTorch device to place the loaded filters on (e.g., 'cpu', 'cuda')
        n_mels (int): Number of mel filters to use (currently only 80 and 128 supported)

    Returns:
        torch.Tensor: Mel filterbank matrix of shape (n_mels, n_freq_bins) on the specified device

    Raises:
        AssertionError: If n_mels is not 80 or 128
        FileNotFoundError: If the mel filters file is not found
    """
    # Verify that the requested number of mel filters is supported
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    # Construct path to the mel filters file
    filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    
    # Load the precomputed mel filters from the .npz file
    with np.load(filters_path, allow_pickle=False) as f:
        # Extract the requested filter bank and convert from numpy to PyTorch tensor
        # Move the tensor to the specified device (GPU/CPU)
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of an audio signal for Whisper model input.

    This function performs the complete audio preprocessing pipeline to convert
    raw audio waveforms into log-Mel spectrograms, which are the standard input
    format for Whisper models. The process includes:
    1. Loading audio if provided as a file path
    2. Applying Short-Time Fourier Transform (STFT)
    3. Converting to mel-scale frequencies using mel filterbank
    4. Taking logarithm and applying normalization

    Args:
        audio (Union[str, np.ndarray, torch.Tensor]): Input audio as file path, NumPy array, or PyTorch tensor
        n_mels (int): Number of mel filters to use (default: 80, options: 80 or 128)
        padding (int): Number of zero samples to pad to the right (default: 0)
        device (Optional[Union[str, torch.device]]): PyTorch device for computation (default: None)

    Returns:
        torch.Tensor: Log-Mel spectrogram of shape (n_mels, n_frames) suitable for Whisper input

    Raises:
        FileNotFoundError: If audio file path is invalid
        AssertionError: If n_mels is not supported
    """
    # Convert input audio to PyTorch tensor if it's not already
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            # If audio is provided as a file path, load it first
            audio = load_audio(audio)
        # Convert NumPy array to PyTorch tensor
        audio = torch.from_numpy(audio)

    # Move audio tensor to specified device if provided
    if device is not None:
        audio = audio.to(device)
    
    # Add zero padding to the right of the audio if specified
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    
    # Create Hann window function for STFT (reduces spectral leakage)
    window = torch.hann_window(N_FFT).to(audio.device)
    
    # Compute Short-Time Fourier Transform (STFT) of the audio signal
    # Returns complex-valued spectrogram with shape (n_freq_bins, n_time_frames)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)

    # Compute power spectrogram (magnitude squared, excluding last frame to match expected dimensions)
    magnitudes = stft[..., :-1].abs() ** 2

    # Load mel filterbank matrix for converting frequencies to mel scale
    filters = mel_filters(audio.device, n_mels)
    
    # Apply mel filterbank to convert frequency representation to mel scale
    # Result shape: (n_mels, n_time_frames)
    mel_spec = filters @ magnitudes

    # Convert to log scale and clamp very small values to prevent log(0)
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    
    # Apply additional clamping to prevent extreme values
    # Ensures no value is more than 8 units below the maximum
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    
    # Normalize the log-mel spectrogram to the range typically expected by Whisper
    # This normalization helps with numerical stability during inference
    log_spec = (log_spec + 4.0) / 4.0
    
    return log_spec
