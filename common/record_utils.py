"""
Audio recording utilities for the Whisper speech recognition pipeline.

This module provides functions for recording audio from the microphone in real-time.
It supports interactive recording with early termination via keyboard input and
saves the recorded audio in the format expected by Whisper models (16kHz mono).

The main function handles audio streaming, buffering, and file output while
allowing the user to stop recording early by pressing Enter.
"""

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import select
import sys
import queue
import time


# Whisper expects 16kHz mono audio
SAMPLE_RATE = 16000  # Sample rate in Hz to match Whisper model requirements
CHANNELS = 1          # Mono audio channel configuration

def enter_pressed():
    """
    Check if Enter key has been pressed without blocking execution.

    This function uses select to check if there's input available on stdin
    without blocking the main thread. It enables non-blocking detection of
    user input during audio recording.

    Returns:
        list: List containing stdin if input is available, empty list otherwise
    """
    return select.select([sys.stdin], [], [], 0.0)[0]

def record_audio(duration, audio_path):
    """
    Record audio from the microphone and save it as a WAV file.

    This function provides an interactive audio recording capability that:
    - Records audio at 16kHz sample rate and mono channel configuration
    - Allows early termination by pressing Enter key
    - Saves the recording to the specified file path
    - Returns the raw audio data for further processing

    Args:
        duration (int): Maximum duration of the recording in seconds
        audio_path (str): File path where the recorded audio will be saved

    Returns:
        numpy.ndarray: Recorded audio data as a 1D array of float32 values in [-1.0, 1.0]
    """
    # Create a queue for thread-safe audio frame buffering
    q = queue.Queue()
    
    # List to store recorded audio frames
    recorded_frames = []

    def audio_callback(indata, frames, time_info, status):
        """
        Callback function called by the audio input stream for each audio chunk.
        
        Args:
            indata: Array containing the input audio data
            frames: Number of frames in the input data
            time_info: Timestamp information for the audio data
            status: Status flags for the audio stream
        """
        # Print any audio status warnings/errors
        if status:
            print("Status:", status)
        # Add a copy of the incoming audio data to the queue
        q.put(indata.copy())

    print(f"Recording for up to {duration} seconds. Press Enter to stop early...")

    # Record the start time to track recording duration
    start_time = time.time()
    
    # Create an input stream with the specified configuration
    with sd.InputStream(samplerate=SAMPLE_RATE,
                        channels=CHANNELS,
                        dtype="float32",
                        callback=audio_callback):
        # Set stdin to non-blocking line-buffered mode to enable early stopping
        sys.stdin = open('/dev/stdin')
        
        # Main recording loop - continues until max duration or early stop
        while True:
            # Check if maximum duration has been reached
            if time.time() - start_time >= duration:
                print("Max duration reached.")
                break
                
            # Check if user has pressed Enter to stop early
            if enter_pressed():
                sys.stdin.read(1)  # consume the newline character
                print("Early stop requested.")
                break
                
            # Get audio frames from the queue and add to recorded frames
            try:
                frame = q.get(timeout=0.1)
                recorded_frames.append(frame)
            except queue.Empty:
                # Continue if no new frames are available (timeout)
                continue

    print("Recording finished. Processing...")

    # Combine all recorded frames into a single array
    audio_data = np.concatenate(recorded_frames, axis=0)
    
    # Convert stereo to mono if necessary by averaging channels
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Save the audio data to a WAV file
    # Convert from float32 [-1.0, 1.0] to int16 for WAV format
    wav.write(audio_path, SAMPLE_RATE, (audio_data * 32767).astype(np.int16))
    
    # Return the raw audio data for potential further processing
    return audio_data
