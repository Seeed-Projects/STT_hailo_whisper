"""
HailoWhisperPipeline - A pipeline for running Whisper inference on Hailo hardware.

This module implements a threaded pipeline for Whisper speech recognition that:
- Runs encoder and decoder models on Hailo hardware accelerators
- Handles tokenization and model input/output processing
- Manages data flow between preprocessing and inference components
- Implements iterative decoding for generating transcriptions
"""

import numpy as np
import os
from hailo_platform import (HEF, VDevice, HailoSchedulingAlgorithm, FormatType)
from transformers import AutoTokenizer
from queue import Queue, Empty
from threading import Thread
from common.postprocessing import apply_repetition_penalty


class HailoWhisperPipeline:
    """
    A pipeline for running Whisper speech recognition inference on Hailo hardware.

    This class manages the entire inference process including:
    - Loading and configuring Hailo HEF models (encoder and decoder)
    - Tokenization and embedding lookup
    - Processing audio mel-spectrograms through the model
    - Iterative decoding to generate text transcriptions
    - Threaded execution for efficient processing
    """

    def __init__(self, encoder_model_path: str, decoder_model_path: str, variant="tiny", host="arm64", multi_process_service=False):
        """
        Initialize the Hailo Whisper pipeline with encoder and decoder models.

        Args:
            encoder_model_path (str): Path to the encoder HEF model file
            decoder_model_path (str): Path to the decoder HEF model file
            variant (str): Whisper model variant to load (e.g., "tiny", "base")
            host (str): Target host architecture (currently not used)
            multi_process_service (bool): Enable multi-process service for concurrent model execution
        """
        # Store paths to the encoder and decoder HEF models
        self.encoder_model_path = encoder_model_path
        self.decoder_model_path = decoder_model_path
        
        # Timeout value for inference operations (in milliseconds)
        self.timeout_ms = 100000000
        
        # Store the model variant to determine behavior and assets
        self.variant = variant

        # Set decoding sequence length based on model variant (tiny models can handle longer sequences)
        self.decoding_sequence_length = 32 if ("tiny" in self.variant) else 24
        
        # Target host architecture (reserved for future use)
        self.host = host  # not used in this version
        
        # Flag to enable multi-process service for concurrent model execution
        self.multi_process_service = multi_process_service

        # Load precomputed token embedding weights and bias for decoder input processing
        self.token_embedding_weight = self._load_token_embedding_weight()
        self.onnx_add_input = self._load_onnx_add_input()

        # Constant for unsqueeze operation axis (used in tensor reshaping)
        self.constant_output_0 = np.array([1])  # Unsqueeze axis
        
        # Load the tokenizer for converting token IDs to text
        self._load_tokenizer()

        # Queues for thread-safe communication between main process and inference thread
        self.data_queue = Queue()      # Input data queue for mel-spectrogram chunks
        self.results_queue = Queue()   # Output queue for transcription results
        
        # Thread control flag and inference thread
        self.running = True
        self.thread = Thread(target=self._inference_loop)
        self.thread.start()

    def _load_token_embedding_weight(self):
        """
        Load precomputed token embedding weights for the decoder.

        These weights are used to convert token IDs to dense vector representations
        that can be processed by the neural network. The embeddings are precomputed
        and stored as numpy arrays to avoid recomputation during inference.

        Returns:
            numpy.ndarray: Token embedding weights for the specified model variant.
        """
        base_path = os.path.dirname(os.path.abspath(__file__))
        # Construct path to the embedding weights file based on model variant
        file_path = os.path.join(base_path,
                                 f"decoder_assets/{self.variant}/decoder_tokenization/token_embedding_weight_{self.variant}.npy")
        return np.load(file_path)

    def _load_onnx_add_input(self):
        """
        Load precomputed bias values for the decoder tokenization process.

        These values are added to the embedded tokens to provide additional bias
        information during the tokenization process, which is part of the model's
        preprocessing pipeline.

        Returns:
            numpy.ndarray: Bias values for decoder input processing.
        """
        base_path = os.path.dirname(os.path.abspath(__file__))
        # Construct path to the bias values file based on model variant
        file_path = os.path.join(base_path,
                                 f"decoder_assets/{self.variant}/decoder_tokenization/onnx_add_input_{self.variant}.npy")
        return np.load(file_path)

    def _load_tokenizer(self):
        """
        Load the HuggingFace tokenizer for the specified Whisper model variant.

        The tokenizer is responsible for converting between text tokens and token IDs
        during the transcription process. It's used to decode the model's output
        token IDs back into readable text.

        Sets:
            self.tokenizer: An instance of AutoTokenizer loaded from HuggingFace.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(f"openai/whisper-{self.variant}")

    def _tokenization(self, decoder_input_ids):
        """
        Perform tokenization operations for the decoder input.

        This function converts token IDs to the appropriate tensor format for
        the decoder model. It performs embedding lookup, adds bias values,
        reshapes the tensor, and prepares it for neural network processing.

        Args:
            decoder_input_ids (numpy.ndarray): Input token IDs for the decoder.

        Returns:
            numpy.ndarray: Reshaped and processed tensor ready for decoder input.
        """
        # Perform embedding lookup to convert token IDs to dense vectors
        # Shape: (len(decoder_input_ids), embedding_dim) where embedding_dim is typically 384
        gather_output = self.token_embedding_weight[decoder_input_ids]  
        
        # Add bias values to the embedded tokens
        # Broadcasting with shape (sequence_length, 384) + (32, 384) -> (sequence_length, 384)
        add_output = gather_output + self.onnx_add_input  
        
        # Insert a new dimension at axis=1 to match expected input shape
        # Shape: (sequence_length, 1, 384) after expanding at axis 1
        unsqueeze_output = np.expand_dims(add_output, axis=int(self.constant_output_0[0]))  
        
        # Transpose the tensor to match the expected input format for Hailo hardware
        # Changes format from (sequence_length, 1, 384) to (sequence_length, 384, 1, 1) equivalent
        # This is needed to match the NHWC format (batch, height, width, channels)
        transpose_output = np.transpose(unsqueeze_output, (0, 2, 1, 3))

        return transpose_output

    def _inference_loop(self):
        """
        Main inference loop for processing input data and generating transcriptions.

        This threaded method handles the complete inference pipeline for Whisper:
        1. Sets up Hailo hardware devices and model configurations
        2. Processes audio mel-spectrograms through the encoder model
        3. Performs iterative decoding with the decoder model
        4. Applies post-processing (repetition penalty) to improve quality
        5. Converts token IDs to readable text using the tokenizer
        6. Puts results in the output queue for retrieval by the main thread

        The method runs continuously while self.running is True, processing data
        from self.data_queue and putting results in self.results_queue.
        """
        # Create device parameters for Hailo hardware configuration
        params = VDevice.create_params()
        # Use round-robin scheduling for multiple models or concurrent requests
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        
        # Configure multi-process service if enabled for concurrent model execution
        if self.multi_process_service:
            params.multi_process_service = True
            params.group_id = "SHARED"

        # Load decoder HEF file to get model information
        decoder_hef = HEF(self.decoder_model_path)
        # Get sorted list of output tensor names for the decoder model
        sorted_output_names = decoder_hef.get_sorted_output_names()
        # Get the decoder network group name (used for input/output binding)
        decoder_model_name = decoder_hef.get_network_group_names()[0]

        # Initialize the Hailo VDevice which manages the hardware accelerator
        with VDevice(params) as vdevice:
            # Create inference models for both encoder and decoder
            encoder_infer_model = vdevice.create_infer_model(self.encoder_model_path)
            decoder_infer_model = vdevice.create_infer_model(self.decoder_model_path)
            
            # Configure input and output data formats (all as FLOAT32)
            encoder_infer_model.input().set_format_type(FormatType.FLOAT32)
            encoder_infer_model.output().set_format_type(FormatType.FLOAT32)
            
            # Configure decoder input layers with FLOAT32 format
            decoder_infer_model.input(f"{decoder_model_name}/input_layer1").set_format_type(FormatType.FLOAT32)
            decoder_infer_model.input(f"{decoder_model_name}/input_layer2").set_format_type(FormatType.FLOAT32)

            # Configure all decoder outputs to use FLOAT32 format
            # Multiple output tensors will be concatenated on the host
            for output_name in sorted_output_names:
                decoder_infer_model.output(output_name).set_format_type(FormatType.FLOAT32)

            # Configure and bind the encoder and decoder models
            with encoder_infer_model.configure() as encoder_configured_infer_model:
                with decoder_infer_model.configure() as decoder_configured_infer_model:
                    # Create input/output bindings for efficient data transfer
                    encoder_bindings = encoder_configured_infer_model.create_bindings()
                    decoder_bindings = decoder_configured_infer_model.create_bindings()

                    # Main processing loop - continues until self.running is False
                    while self.running:
                        try:
                            # Wait for new mel-spectrogram data from the input queue with a 1-second timeout
                            # This allows the loop to check self.running periodically for graceful shutdown
                            input_mel = self.data_queue.get(timeout=1)

                            transcriptions = []  # Store transcriptions for this input
                            
                            # Ensure the input data is in contiguous memory layout for efficient processing
                            input_mel = np.ascontiguousarray(input_mel)
                            
                            # Set the input buffer for the encoder with the mel-spectrogram data
                            encoder_bindings.input().set_buffer(input_mel)
                            
                            # Create output buffer for encoder results (initialize with zeros)
                            buffer = np.zeros(encoder_infer_model.output().shape).astype(np.float32)
                            encoder_bindings.output().set_buffer(buffer)

                            # Run the encoder model on Hailo hardware
                            encoder_configured_infer_model.run([encoder_bindings], self.timeout_ms)
                            # Get the encoded audio features from the encoder output
                            encoded_features = encoder_bindings.output().get_buffer()

                            # Decoder initialization: start with the special start token (50258 is the start token ID for Whisper)
                            start_token_id = [50258]
                            decoder_input_ids = np.array(
                                [[start_token_id[0]]], dtype=np.int64
                            )  # Shape (1,1) - single batch, single token
                            
                            # Pad the decoder input with zeros to the full sequence length
                            decoder_input_ids = np.concatenate(
                                [decoder_input_ids, np.zeros((1, self.decoding_sequence_length - 1), dtype=np.int64)], axis=1
                            )

                            # Track generated tokens for this transcription
                            generated_tokens = []
                            decoder_outputs = None
                            
                            # Iterative decoding loop - generate one token at a time
                            for i in range(self.decoding_sequence_length - 1):
                                # Process the decoder input IDs through the tokenization pipeline
                                tokenized_ids = self._tokenization(decoder_input_ids)

                                # Set encoder features as first input to decoder
                                decoder_bindings.input(f"{decoder_model_name}/input_layer1").set_buffer(encoded_features)
                                # Set processed tokenized IDs as second input to decoder
                                decoder_bindings.input(f"{decoder_model_name}/input_layer2").set_buffer(tokenized_ids)

                                # Prepare output buffers for all decoder outputs
                                buffers = [
                                    np.zeros(decoder_infer_model.output(name).shape).astype(np.float32) 
                                    for name in sorted_output_names
                                ]

                                # Bind output buffers to decoder model
                                for name, buffer in zip(sorted_output_names, buffers):
                                    decoder_bindings.output(name).set_buffer(buffer)

                                # Run the decoder model on Hailo hardware
                                decoder_configured_infer_model.run([decoder_bindings], self.timeout_ms)

                                # Concatenate all decoder outputs along the feature dimension (axis=2)
                                decoder_outputs = np.concatenate(
                                    [decoder_bindings.output(name).get_buffer() for name in sorted_output_names], axis=2
                                )
                                
                                # Apply repetition penalty to discourage repetitive text generation
                                repetition_penalty = 1.5
                                logits = apply_repetition_penalty(
                                    decoder_outputs[:, i], generated_tokens, penalty=repetition_penalty
                                )
                                # Select the next token as the one with the highest probability
                                next_token = np.argmax(logits)

                                # Add the generated token to the sequence
                                generated_tokens.append(next_token)
                                # Update the decoder input for the next iteration
                                decoder_input_ids[0][i + 1] = next_token

                                # Stop decoding if we've reached the end-of-sequence token
                                if next_token == self.tokenizer.eos_token_id:
                                    break

                            # Convert the generated token IDs to readable text using the tokenizer
                            transcription = self.tokenizer.decode(
                                generated_tokens, skip_special_tokens=True
                            )
                            
                            # Add the final transcription to the results queue
                            self.results_queue.put(transcription)
                            transcriptions.append(transcription)
                            
                        except Empty:
                            # No data in queue yet, continue looping to check self.running flag
                            pass  # No data yet, continue looping

    def send_data(self, data):
        """
        Send new mel-spectrogram data to the inference pipeline for processing.

        This method adds the input data to the internal queue where it will be
        processed by the inference thread. The data should be a mel-spectrogram
        tensor in the format expected by the Whisper encoder.

        Args:
            data (numpy.ndarray): Mel-spectrogram data to process, with shape 
                                appropriate for the encoder input (typically [n_mels, n_frames])
        """
        self.data_queue.put(data)

    def get_transcription(self):
        """
        Retrieve the next transcription result from the output queue.

        This method blocks until a transcription result is available in the queue.
        It should be called after sending data via send_data() to obtain the
        corresponding transcription.

        Returns:
            str: The transcribed text result from the Whisper model.
        """
        return self.results_queue.get()

    def stop(self):
        """
        Stop the inference processing and clean up resources.

        This method signals the inference thread to stop processing and waits
        for it to complete before returning. It should be called when the
        pipeline is no longer needed to ensure proper cleanup of Hailo hardware
        resources and threads.
        """
        # Signal the inference thread to stop processing
        self.running = False
        # Wait for the inference thread to complete and exit
        self.thread.join()

