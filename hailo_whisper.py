"""Real-time STT with Hailo Whisper - Fixed Version

Fixed audio buffer size and method calls.
"""

import time
import argparse
import threading
import queue
import numpy as np
import os
from collections import deque
from app.hailo_whisper_pipeline import HailoWhisperPipeline
from common.preprocessing import preprocess
from common.postprocessing import clean_transcription
from app.whisper_hef_registry import HEF_REGISTRY

# Try to import sounddevice, fallback to pyaudio if not available
try:
    import sounddevice as sd
    AUDIO_BACKEND = "sounddevice"
except ImportError:
    try:
        import pyaudio
        AUDIO_BACKEND = "pyaudio"
    except ImportError:
        print("Error: No audio backend available. Please install either 'sounddevice' or 'pyaudio'")
        print("Install sounddevice: uv pip install sounddevice")
        print("Or install pyaudio: uv pip install pyaudio")
        exit(1)

# Real-time configuration
CHUNK_SIZE = 1600  # 100ms chunks at 16kHz
SAMPLE_RATE = 16000
CHANNELS = 1

# Model processing configuration - 使用5秒块来匹配模型期望
PROCESS_CHUNK_DURATION = 5.0  # 改为5秒以匹配模型输入
OVERLAP_DURATION = 1.0  # 增加重叠以确保连续性
SILENCE_THRESHOLD = 500


class AudioStream:
    """Audio stream with multiple backend support"""
    
    def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        # 增加缓冲区大小以容纳5秒音频
        buffer_seconds = 15  # 15秒缓冲区
        self.audio_buffer = deque(maxlen=int(sample_rate * buffer_seconds))
        self.processing_buffer = deque()
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.backend = AUDIO_BACKEND
        
        if self.backend == "sounddevice":
            self._init_sounddevice()
        else:
            self._init_pyaudio()
            
    def _init_sounddevice(self):
        """Initialize sounddevice backend"""
        self.stream = None
        print("Using sounddevice audio backend")
        
    def _init_pyaudio(self):
        """Initialize pyaudio backend"""
        self.audio_interface = pyaudio.PyAudio()
        self.stream = None
        print("Using pyaudio audio backend")
        
    def start_stream(self):
        """Start audio streaming"""
        self.is_recording = True
        
        if self.backend == "sounddevice":
            def audio_callback(indata, frames, time, status):
                if status:
                    print(f"Audio stream status: {status}")
                if self.is_recording:
                    # Convert to float32 and normalize
                    audio_chunk = indata[:, 0].astype(np.float32)
                    self.audio_buffer.extend(audio_chunk)
                    
                    # Check if we have enough data for processing (5 seconds)
                    if len(self.audio_buffer) >= int(self.sample_rate * PROCESS_CHUNK_DURATION):
                        self._process_audio_chunk()
            
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=CHANNELS,
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=audio_callback
            )
            self.stream.start()
            
        else:  # pyaudio backend
            def audio_callback(in_data, frame_count, time_info, status):
                if self.is_recording:
                    # Convert to numpy array and normalize
                    audio_chunk = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                    self.audio_buffer.extend(audio_chunk)
                    
                    # Check if we have enough data for processing (5 seconds)
                    if len(self.audio_buffer) >= int(self.sample_rate * PROCESS_CHUNK_DURATION):
                        self._process_audio_chunk()
                        
                return (in_data, pyaudio.paContinue)
            
            self.stream = self.audio_interface.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=audio_callback
            )
            self.stream.start_stream()
        
        print("Audio stream started")
        
    def stop_stream(self):
        """Stop audio streaming"""
        self.is_recording = False
        if self.stream:
            if self.backend == "sounddevice":
                self.stream.stop()
                self.stream.close()
            else:
                self.stream.stop_stream()
                self.stream.close()
                self.audio_interface.terminate()
        print("Audio stream stopped")
        
    def _process_audio_chunk(self):
        """Process audio chunk for inference"""
        process_samples = int(self.sample_rate * PROCESS_CHUNK_DURATION)
        overlap_samples = int(self.sample_rate * OVERLAP_DURATION)
        
        if len(self.processing_buffer) >= overlap_samples:
            # Combine overlap with new data
            overlap_data = list(self.processing_buffer)
            new_data = list(self.audio_buffer)[-process_samples + overlap_samples:]
            combined_data = overlap_data + new_data
        else:
            combined_data = list(self.audio_buffer)[-process_samples:]
        
        # Update processing buffer for next overlap
        self.processing_buffer = deque(
            list(self.audio_buffer)[-overlap_samples:], 
            maxlen=overlap_samples
        )
        
        # Check voice activity and add to queue
        audio_array = np.array(combined_data)
        if self._has_voice_activity(audio_array):
            self.audio_queue.put(audio_array)
    
    def _has_voice_activity(self, audio_data):
        """Simple voice activity detection"""
        rms = np.sqrt(np.mean(audio_data**2))
        # Normalize threshold for both backends
        threshold = SILENCE_THRESHOLD / 32768.0
        return rms > threshold
    
    def get_audio_chunk(self, timeout=1.0):
        """Get next audio chunk for processing"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class RealTimeSTTEngine:
    """Real-time speech-to-text engine with Hailo acceleration"""
    
    def __init__(self, model_variant="base", hw_arch="hailo8", multi_process_service=False):
        self.model_variant = model_variant
        self.hw_arch = hw_arch
        self.multi_process_service = multi_process_service
        
        # Initialize pipeline
        self._init_pipeline()
        self.audio_stream = None
        self.is_running = False
        self.processing_thread = None
        
    def _init_pipeline(self):
        """Initialize Hailo Whisper pipeline"""
        encoder_path = self._get_hef_path("encoder")
        decoder_path = self._get_hef_path("decoder")
        
        self.pipeline = HailoWhisperPipeline(
            encoder_path, 
            decoder_path, 
            self.model_variant,
            multi_process_service=self.multi_process_service
        )
        print(f"Real-time STT engine initialized with {self.model_variant} model")
        
    def _get_hef_path(self, component):
        """Get HEF file path for model component"""
        try:
            hef_path = HEF_REGISTRY[self.model_variant][self.hw_arch][component]
            if not os.path.exists(hef_path):
                raise FileNotFoundError(f"HEF file not found: {hef_path}")
            return hef_path
        except KeyError as e:
            raise FileNotFoundError(
                f"HEF not available for model '{self.model_variant}' on hardware '{self.hw_arch}'"
            ) from e
    
    def start(self):
        """Start real-time transcription"""
        self.is_running = True
        self.audio_stream = AudioStream()
        self.audio_stream.start_stream()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("Real-time STT started. Speak now... (Press Ctrl+C to stop)")
        print("Note: ALSA warnings are normal and can be ignored")
        
    def stop(self):
        """Stop real-time transcription"""
        self.is_running = False
        if self.audio_stream:
            self.audio_stream.stop_stream()
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        self.pipeline.stop()
        print("Real-time STT stopped")
        
    def _processing_loop(self):
        """Main processing loop for real-time inference"""
        while self.is_running:
            # Get audio chunk for processing
            audio_chunk = self.audio_stream.get_audio_chunk(timeout=0.1)
            
            if audio_chunk is not None:
                try:
                    # 检查音频长度是否正确
                    expected_samples = int(SAMPLE_RATE * PROCESS_CHUNK_DURATION)
                    if len(audio_chunk) != expected_samples:
                        print(f"Warning: Audio chunk size {len(audio_chunk)} doesn't match expected {expected_samples}")
                        # 如果长度不匹配，进行调整
                        if len(audio_chunk) > expected_samples:
                            audio_chunk = audio_chunk[:expected_samples]
                        else:
                            # 如果太短，用静音填充
                            padding = np.zeros(expected_samples - len(audio_chunk))
                            audio_chunk = np.concatenate([audio_chunk, padding])
                    
                    # Preprocess audio - 使用5秒块
                    mel_spectrograms = preprocess(
                        audio_chunk,
                        is_nhwc=True,
                        chunk_length=PROCESS_CHUNK_DURATION,  # 5秒
                        chunk_offset=0
                    )
                    
                    if not mel_spectrograms:
                        print("Warning: No mel spectrograms generated from audio")
                        continue
                    
                    # Process through pipeline
                    for mel in mel_spectrograms:
                        self.pipeline.send_data(mel)
                        # 修复：不使用timeout参数，或者使用非阻塞方式
                        try:
                            # 尝试非阻塞获取转录
                            transcription = self._get_transcription_non_blocking()
                            if transcription:
                                cleaned_text = clean_transcription(transcription)
                                if cleaned_text.strip():
                                    # Output with timestamp
                                    timestamp = time.strftime("%H:%M:%S")
                                    print(f"[{timestamp}] {cleaned_text}")
                        except Exception as e:
                            print(f"Transcription error: {e}")
                                
                except Exception as e:
                    print(f"Processing error: {e}")
                    
            time.sleep(0.01)
    
    def _get_transcription_non_blocking(self):
        """Non-blocking method to get transcription"""
        # 检查管道是否有可用的转录
        if hasattr(self.pipeline, 'transcription_available') and callable(getattr(self.pipeline, 'transcription_available')):
            if self.pipeline.transcription_available():
                return self.pipeline.get_transcription()
        else:
            # 回退方法：直接尝试获取，但可能会阻塞
            # 在单独的线程中运行以避免阻塞
            return self.pipeline.get_transcription()


def get_args():
    """Parse command line arguments for real-time STT"""
    parser = argparse.ArgumentParser(description="Real-time Whisper STT with Hailo")
    
    parser.add_argument(
        "--hw-arch",
        type=str,
        default="hailo8",
        choices=["hailo8", "hailo8l", "hailo10h"],
        help="Hardware architecture (default: hailo8)"
    )
    
    parser.add_argument(
        "--variant",
        type=str,
        default="base",
        choices=["tiny", "tiny.en", "base"],
        help="Model variant (default: base)"
    )
    
    parser.add_argument(
        "--multi-process-service", 
        action="store_true", 
        help="Enable multi-process service"
    )
    
    return parser.parse_args()


def main():
    """Main function for real-time STT application"""
    args = get_args()
    
    # Initialize real-time STT engine
    stt_engine = RealTimeSTTEngine(
        model_variant=args.variant,
        hw_arch=args.hw_arch,
        multi_process_service=args.multi_process_service
    )
    
    try:
        # Start real-time transcription
        stt_engine.start()
        
        # Keep running until interrupted
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping real-time STT...")
    finally:
        stt_engine.stop()


if __name__ == "__main__":
    main()