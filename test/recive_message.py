#!/usr/bin/env python3
"""
UDP Receiver for Hailo Whisper Transcription Results

This script listens for UDP packets containing transcription results
from the Hailo Whisper real-time STT engine and displays them.
"""

import socket
import json
import threading
import time
import argparse
from datetime import datetime


class TranscriptionReceiver:
    """Receives and displays transcription results via UDP"""
    
    def __init__(self, host='0.0.0.0', port=12345):
        self.host = host
        self.port = port
        self.socket = None
        self.is_receiving = False
        self.receive_thread = None
        self.statistics = {
            'total_packets': 0,
            'last_received': None,
            'start_time': None
        }
        
    def start_receiving(self):
        """Start the UDP receiver"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind((self.host, self.port))
            print(f"UDP receiver started on {self.host}:{self.port}")
            print("Waiting for transcription data...")
            print("-" * 80)
            
        except Exception as e:
            print(f"Failed to start UDP receiver: {e}")
            return False
            
        self.is_receiving = True
        self.statistics['start_time'] = time.time()
        self.receive_thread = threading.Thread(target=self._receive_loop)
        self.receive_thread.daemon = True
        self.receive_thread.start()
        return True
        
    def stop_receiving(self):
        """Stop the UDP receiver"""
        self.is_receiving = False
        if self.socket:
            self.socket.close()
        if self.receive_thread:
            self.receive_thread.join(timeout=2.0)
        print("\nUDP receiver stopped")
        
    def _receive_loop(self):
        """Main receive loop"""
        while self.is_receiving:
            try:
                # Set timeout to allow checking is_receiving flag
                self.socket.settimeout(1.0)
                data, addr = self.socket.recvfrom(4096)  # Buffer size 4KB
                
                # Process received data
                self._process_data(data, addr)
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_receiving:  # Only print errors if we're supposed to be receiving
                    print(f"Error receiving data: {e}")
                    
    def _process_data(self, data, addr):
        """Process received UDP data"""
        try:
            # Decode JSON data
            json_str = data.decode('utf-8')
            transcription_data = json.loads(json_str)
            
            # Update statistics
            self.statistics['total_packets'] += 1
            self.statistics['last_received'] = time.time()
            
            # Display the transcription
            self._display_transcription(transcription_data, addr)
            
        except json.JSONDecodeError as e:
            print(f"Invalid JSON received from {addr}: {e}")
            print(f"Raw data: {data[:100]}...")  # Print first 100 chars
        except UnicodeDecodeError as e:
            print(f"Invalid encoding from {addr}: {e}")
        except Exception as e:
            print(f"Error processing data from {addr}: {e}")
            
    def _display_transcription(self, data, addr):
        """Display transcription data in a formatted way"""
        # Extract fields with defaults for backward compatibility
        chunk_id = data.get('chunk_id', 'N/A')
        start_time = data.get('start_time', 0)
        end_time = data.get('end_time', 0)
        transcription = data.get('transcription', '')
        timestamp = data.get('timestamp', time.time())
        sample_rate = data.get('sample_rate', 16000)
        chunk_duration = data.get('chunk_duration', 5.0)
        overlap = data.get('overlap', 1.0)
        
        # Convert timestamp to readable format
        time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
        
        # Format output
        print(f"\n[{time_str}] From: {addr[0]}:{addr[1]}")
        print(f"Chunk: {chunk_id} | Time: {start_time:.1f}s - {end_time:.1f}s")
        print(f"Duration: {chunk_duration}s | Overlap: {overlap}s | Sample Rate: {sample_rate}Hz")
        print(f"Transcription: {transcription}")
        print("-" * 80)
        
    def print_statistics(self):
        """Print reception statistics"""
        if self.statistics['start_time']:
            uptime = time.time() - self.statistics['start_time']
            packets_per_second = self.statistics['total_packets'] / uptime if uptime > 0 else 0
            
            print(f"\n{'='*50}")
            print("RECEPTION STATISTICS")
            print(f"{'='*50}")
            print(f"Total packets received: {self.statistics['total_packets']}")
            print(f"Uptime: {uptime:.1f} seconds")
            print(f"Packets per second: {packets_per_second:.2f}")
            
            if self.statistics['last_received']:
                time_since_last = time.time() - self.statistics['last_received']
                print(f"Time since last packet: {time_since_last:.1f} seconds")
            print(f"{'='*50}")


def main():
    """Main function for UDP receiver"""
    parser = argparse.ArgumentParser(description='UDP Receiver for Hailo Whisper Transcription Results')
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host IP to bind to (default: 0.0.0.0 - all interfaces)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=12345,
        help='Port to listen on (default: 12345)'
    )
    parser.add_argument(
        '--stats-interval',
        type=int,
        default=0,
        help='Print statistics every N seconds (0 to disable, default: 0)'
    )
    
    args = parser.parse_args()
    
    # Create and start receiver
    receiver = TranscriptionReceiver(args.host, args.port)
    
    if not receiver.start_receiving():
        return
    
    # Statistics thread if interval is specified
    stats_thread = None
    if args.stats_interval > 0:
        def stats_loop():
            while receiver.is_receiving:
                time.sleep(args.stats_interval)
                receiver.print_statistics()
        
        stats_thread = threading.Thread(target=stats_loop)
        stats_thread.daemon = True
        stats_thread.start()
        print(f"Statistics will be printed every {args.stats_interval} seconds")
    
    try:
        # Keep main thread alive
        while receiver.is_receiving:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping UDP receiver...")
    finally:
        receiver.stop_receiving()
        receiver.print_statistics()


if __name__ == "__main__":
    main()