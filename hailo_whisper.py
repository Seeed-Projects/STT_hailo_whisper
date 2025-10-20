"""Real-time STT with Hailo Whisper - Overlapping Chunk Processing with UDP Streaming

Processes audio in 5-second chunks with 1-second overlap between consecutive chunks.
Sends transcription results via UDP to a remote server.
"""

import time
import argparse
import threading
import queue
import numpy as np
import os
import socket
import json
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

# Model processing configuration
PROCESS_CHUNK_DURATION = 5.0  # 5-second processing chunks
OVERLAP_DURATION = 1.0  # 1-second overlap between consecutive chunks
SILENCE_THRESHOLD = 600


class UDPStreamer:
    """Handles UDP streaming of transcription results to remote server"""
    
    def __init__(self, remote_host, remote_port):
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.socket = None
        self._initialize_socket()
        
    def _initialize_socket(self):
        """Initialize UDP socket"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"UDP socket initialized for {self.remote_host}:{self.remote_port}")
        except Exception as e:
            print(f"Failed to initialize UDP socket: {e}")
            self.socket = None
            
    def send_transcription(self, transcription_data):
        """Send transcription data via UDP"""
        if self.socket is None:
            print("UDP socket not available, cannot send transcription")
            return False
            
        try:
            # Convert data to JSON string
            json_data = json.dumps(transcription_data, ensure_ascii=False)
            # Encode to bytes
            message = json_data.encode('utf-8')
            
            # Send via UDP
            self.socket.sendto(message, (self.remote_host, self.remote_port))
            return True
            
        except Exception as e:
            print(f"Failed to send UDP message: {e}")
            return False
            
    def close(self):
        """Close UDP socket"""
        if self.socket:
            self.socket.close()
            self.socket = None


class OverlappingAudioProcessor:
    """Processes audio in 5-second chunks with 1-second overlap"""
    
    def __init__(self, sample_rate=SAMPLE_RATE, udp_streamer=None):
        self.sample_rate = sample_rate
        self.required_samples = int(sample_rate * PROCESS_CHUNK_DURATION)
        self.overlap_samples = int(sample_rate * OVERLAP_DURATION)
        self.step_samples = self.required_samples - self.overlap_samples  # 4 seconds of new audio per chunk
        
        # Audio buffer to accumulate samples
        self.audio_buffer = deque(maxlen=self.required_samples * 3)  # 3 chunks buffer
        
        # Processing queues
        self.raw_audio_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=10)
        
        # UDP streaming
        self.udp_streamer = udp_streamer
        
        self.chunk_counter = 0
        self.is_processing = False
        self.processing_thread = None
        self.last_processed_position = 0  # Track position in the audio stream
        
    def add_audio_data(self, audio_data):
        """Add new audio data to buffer"""
        self.audio_buffer.extend(audio_data)
        
    def extract_next_chunk(self):
        """Extract the next 5-second chunk with 1-second overlap"""
        # We need enough data for a full chunk starting from the last processed position
        required_from_position = self.last_processed_position + self.required_samples
        
        if len(self.audio_buffer) >= required_from_position:
            # Extract chunk from last_processed_position to last_processed_position + required_samples
            start_idx = self.last_processed_position
            end_idx = start_idx + self.required_samples
            
            # Convert deque to list for slicing
            buffer_list = list(self.audio_buffer)
            
            if end_idx <= len(buffer_list):
                chunk_data = buffer_list[start_idx:end_idx]
                
                # Update position for next chunk (move forward by step_samples)
                self.last_processed_position += self.step_samples
                
                # If we're getting too far ahead, reset the buffer and position
                if self.last_processed_position > len(self.audio_buffer) // 2:
                    # Keep only the recent data
                    keep_from = max(0, len(self.audio_buffer) - self.required_samples)
                    self.audio_buffer = deque(
                        list(self.audio_buffer)[keep_from:], 
                        maxlen=self.audio_buffer.maxlen
                    )
                    self.last_processed_position = max(0, self.last_processed_position - keep_from)
                
                return np.array(chunk_data)
        
        return None
        
    def start_processing(self, pipeline):
        """Start the overlapping processing loop"""
        self.is_processing = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop, 
            args=(pipeline,)
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop_processing(self):
        """Stop the processing loop"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            
    def _processing_loop(self, pipeline):
        """Main processing loop - processes 5-second chunks with 1-second overlap"""
        while self.is_processing:
            # Try to extract the next 5-second chunk
            audio_chunk = self.extract_next_chunk()
            
            if audio_chunk is not None:
                try:
                    # Process this chunk
                    self._process_single_chunk(audio_chunk, pipeline)
                    
                except Exception as e:
                    print(f"Chunk processing error: {e}")
                    
            # Small delay to prevent busy waiting
            time.sleep(0.01)
            
    def _process_single_chunk(self, audio_chunk, pipeline):
        """Process a single 5-second audio chunk"""
        chunk_id = self.chunk_counter
        self.chunk_counter += 1
        
        # Ensure correct length
        if len(audio_chunk) != self.required_samples:
            print(f"Warning: Chunk {chunk_id} has incorrect length {len(audio_chunk)}, expected {self.required_samples}")
            return
            
        # Check voice activity
        if not self._has_voice_activity(audio_chunk):
            print(f"Chunk {chunk_id}: No voice activity detected, skipping")
            return
            
        # Preprocess audio
        mel_spectrograms = preprocess(
            audio_chunk,
            is_nhwc=True,
            chunk_length=PROCESS_CHUNK_DURATION,
            chunk_offset=0
        )
        
        if not mel_spectrograms:
            print(f"Chunk {chunk_id}: No mel spectrograms generated")
            return
            
        # Process through pipeline
        for mel in mel_spectrograms:
            pipeline.send_data(mel)
            
            # Wait for transcription result
            try:
                transcription = pipeline.get_transcription()
                
                if transcription:
                    cleaned_text = clean_transcription(transcription)
                    if cleaned_text.strip():
                        # Calculate timestamp based on chunk sequence and overlap
                        start_time = chunk_id * self.step_samples / self.sample_rate
                        end_time = start_time + PROCESS_CHUNK_DURATION
                        
                        result = {
                            'chunk_id': chunk_id,
                            'start_time': start_time,
                            'end_time': end_time,
                            'transcription': cleaned_text,
                            'timestamp': time.time()
                        }
                        self.result_queue.put(result)
                        
                        # Send via UDP if streamer is available
                        if self.udp_streamer:
                            udp_data = {
                                'chunk_id': chunk_id,
                                'start_time': round(start_time, 2),
                                'end_time': round(end_time, 2),
                                'transcription': cleaned_text,
                                'timestamp': time.time(),
                                'sample_rate': self.sample_rate,
                                'chunk_duration': PROCESS_CHUNK_DURATION,
                                'overlap': OVERLAP_DURATION
                            }
                            success = self.udp_streamer.send_transcription(udp_data)
                            if success:
                                print(f"Chunk {chunk_id} sent via UDP")
                        
            except Exception as e:
                print(f"Transcription error for chunk {chunk_id}: {e}")
                
    def _has_voice_activity(self, audio_data):
        """Simple voice activity detection"""
        rms = np.sqrt(np.mean(audio_data**2))
        threshold = SILENCE_THRESHOLD / 32768.0
        return rms > threshold
        
    def get_result(self, timeout=1.0):
        """Get next processing result"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class AudioStream:
    """Audio stream with overlapping chunk processing and UDP streaming"""
    
    def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE, udp_streamer=None):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_processor = OverlappingAudioProcessor(sample_rate, udp_streamer)
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
        
    def start_stream(self, pipeline):
        """Start audio streaming and processing"""
        self.is_recording = True
        self.audio_processor.start_processing(pipeline)
        
        if self.backend == "sounddevice":
            def audio_callback(indata, frames, time, status):
                if status:
                    print(f"Audio stream status: {status}")
                if self.is_recording:
                    # Convert to float32 and normalize
                    audio_chunk = indata[:, 0].astype(np.float32)
                    self.audio_processor.add_audio_data(audio_chunk)
            
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
                    self.audio_processor.add_audio_data(audio_chunk)
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
        
        print("Audio stream and overlapping processing started")
        print(f"Processing {PROCESS_CHUNK_DURATION}-second chunks with {OVERLAP_DURATION}-second overlap")
        
    def stop_stream(self):
        """Stop audio streaming and processing"""
        self.is_recording = False
        if self.stream:
            if self.backend == "sounddevice":
                self.stream.stop()
                self.stream.close()
            else:
                self.stream.stop_stream()
                self.stream.close()
                self.audio_interface.terminate()
        
        self.audio_processor.stop_processing()
        print("Audio stream and processing stopped")
        
    def get_transcription(self, timeout=1.0):
        """Get next transcription result"""
        return self.audio_processor.get_result(timeout)


class RealTimeSTTEngine:
    """Real-time speech-to-text engine with overlapping chunk processing and UDP streaming"""
    
    def __init__(self, model_variant="base", hw_arch="hailo8", multi_process_service=False, 
                 udp_host=None, udp_port=None):
        self.model_variant = model_variant
        self.hw_arch = hw_arch
        self.multi_process_service = multi_process_service
        
        # Initialize UDP streaming if host and port are provided
        self.udp_streamer = None
        if udp_host and udp_port:
            self.udp_streamer = UDPStreamer(udp_host, udp_port)
        
        # Initialize pipeline
        self._init_pipeline()
        self.audio_stream = None
        self.is_running = False
        
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
        self.audio_stream = AudioStream(udp_streamer=self.udp_streamer)
        self.audio_stream.start_stream(self.pipeline)
        
        print("Real-time STT started. Speak now... (Press Ctrl+C to stop)")
        print(f"Processing {PROCESS_CHUNK_DURATION}-second chunks with {OVERLAP_DURATION}-second overlap")
        print(f"Chunk sequence: 0-5s, 4-9s, 8-13s, etc.")
        if self.udp_streamer:
            print(f"UDP streaming enabled to {self.udp_streamer.remote_host}:{self.udp_streamer.remote_port}")
        print("Note: ALSA warnings are normal and can be ignored")
        
        # Main result output loop
        try:
            while self.is_running:
                result = self.audio_stream.get_transcription(timeout=0.5)
                if result is not None:
                    # Format output with timing information
                    chunk_id = result['chunk_id']
                    start_time = result['start_time']
                    end_time = result['end_time']
                    transcription = result['transcription']
                    timestamp = time.strftime("%H:%M:%S")
                    
                    print(f"[{timestamp}] Chunk_{chunk_id:03d} ({start_time:05.1f}s-{end_time:05.1f}s): {transcription}")
                    
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            self.stop()
        
    def stop(self):
        """Stop real-time transcription"""
        self.is_running = False
        if self.audio_stream:
            self.audio_stream.stop_stream()
        self.pipeline.stop()
        if self.udp_streamer:
            self.udp_streamer.close()
        print("Real-time STT stopped")


def get_args():
    """Parse command line arguments for real-time STT"""
    parser = argparse.ArgumentParser(description="Real-time Whisper STT with Hailo and UDP Streaming")
    
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
    
    parser.add_argument(
        "--udp-host",
        type=str,
        help="Remote host for UDP streaming (e.g., 192.168.1.100)"
    )
    
    parser.add_argument(
        "--udp-port",
        type=int,
        default=12345,
        help="Remote port for UDP streaming (default: 12345)"
    )
    
    return parser.parse_args()


def main():
    """Main function for real-time STT application"""
    args = get_args()
    
    # Validate UDP arguments
    if args.udp_host and not args.udp_port:
        print("Error: --udp-port is required when --udp-host is specified")
        return
    
    # Initialize real-time STT engine
    stt_engine = RealTimeSTTEngine(
        model_variant=args.variant,
        hw_arch=args.hw_arch,
        multi_process_service=args.multi_process_service,
        udp_host=args.udp_host,
        udp_port=args.udp_port
    )
    
    try:
        # Start real-time transcription
        stt_engine.start()
            
    except KeyboardInterrupt:
        print("\nStopping real-time STT...")
    finally:
        stt_engine.stop()


if __name__ == "__main__":
    main()