"""
Postprocessing functions for Whisper-generated transcriptions.

This module provides functions for improving the quality of Whisper-generated
transcriptions through various techniques:
- Repetition penalty to discourage repeated tokens
- Temperature sampling for diversity in generation
- Text cleaning to remove duplicate sentences and improve readability

These functions help refine the raw output from the Whisper model to produce
more natural and readable transcriptions.
"""

import numpy as np
import re


# List of token IDs that correspond to punctuation marks that should not be penalized
# Token 11 and 13 typically correspond to common punctuation like commas and periods
excluded_tokens = [11, 13]  # Punctuation tokens to exclude from repetition penalty

def apply_repetition_penalty(logits, generated_tokens, penalty=1.5, last_window=8):
    """
    Apply repetition penalty to discourage generating tokens that appeared recently.

    This function reduces the probability of tokens that have been recently generated,
    helping to prevent repetitive text in the output. It works by dividing the logits
    of recently generated tokens by the penalty factor, making them less likely to be selected.

    Args:
        logits (numpy.ndarray): Model output logits with shape (1, vocab_size) or (vocab_size,)
        generated_tokens (list): List of token IDs that have already been generated
        penalty (float): Penalty factor (default: 1.5, values > 1.0 reduce repetition)
        last_window (int): Number of most recent tokens to consider for penalty (default: 8)

    Returns:
        numpy.ndarray: Logits with repetition penalty applied, shape (vocab_size,)
    """
    # Remove the batch dimension if present to ensure logits have shape (vocab_size,)
    logits = np.squeeze(logits, axis=0)
    
    # Extract the most recent tokens within the specified window
    # If fewer tokens exist than the window size, use all generated tokens
    recent_tokens = generated_tokens[-last_window:] if len(generated_tokens) >= last_window else generated_tokens

    # Convert to set to eliminate duplicates and improve lookup performance
    recent_tokens = set(recent_tokens)

    # Apply penalty to each recent token that is not in the excluded list
    for token in recent_tokens:
        if token not in excluded_tokens:
            # Divide the logit by the penalty factor to reduce its probability
            logits[token] /= penalty
    
    return logits

def temperature_sampling(logits, temperature=0.0):
    """
    Apply temperature sampling to generate the next token with controlled randomness.

    Temperature sampling controls the randomness of predictions by scaling the logits
    before applying the softmax function. Higher temperatures (above 1.0) make the
    model more random, while lower temperatures (below 1.0) make it more deterministic.

    Args:
        logits (numpy.ndarray): Model output logits with shape (vocab_size,)
        temperature (float): Temperature parameter for sampling (default: 0.0 for greedy decoding)

    Returns:
        int: Index of the selected token ID
    """
    # Boost the logits for punctuation tokens to encourage proper sentence structure
    # This helps maintain grammatical structure in the generated text
    for punct_idx in excluded_tokens:
        if punct_idx < len(logits):
            # Increase punctuation token logits by 20% to make them more likely
            logits[punct_idx] *= 1.2

    if temperature == 0.0:
        # Use greedy decoding (always select the highest probability token)
        return np.argmax(logits)
    
    # Apply numerical stability by subtracting the maximum logit to prevent overflow
    logits = logits - np.max(logits)
    
    # Scale logits by temperature (higher temp = more random, lower temp = more deterministic)
    logits = logits / temperature
    
    # Apply softmax to convert logits to probabilities
    probs = np.exp(logits) / np.sum(np.exp(logits))

    # Check for NaN values in the probability distribution
    if np.isnan(probs).any():
        print("Warning: Probabilities contain NaN values. Falling back to greedy decoding.")
        return np.argmax(logits)  # Fall back to greedy decoding
    
    # Ensure probabilities sum to exactly 1 to maintain proper distribution
    probs = probs / np.sum(probs)
    
    # Sample the next token from the probability distribution
    next_token = np.random.choice(len(probs), p=probs)
    
    return next_token


def clean_transcription(transcription):
    """
    Clean and improve the readability of Whisper-generated transcriptions.

    This function processes the raw transcription to remove duplicate sentences
    and ensure proper formatting. It splits the input into sentences and identifies
    repetitions, returning a cleaner version that stops at the first repetition.

    Args:
        transcription (str): Raw transcription text from the Whisper model

    Returns:
        str: Cleaned transcription with duplicate sentences removed and proper formatting
    """
    # Split the transcription into sentences using periods and question marks as delimiters
    # This regex looks for periods or question marks followed by whitespace
    sentences = re.split(r'(?<=[.?])\s+', transcription)
    
    # Initialize a list to store unique sentences
    unique_sentences = []
    
    # Iterate through each sentence in the transcription
    for sentence in sentences:
        # Check if any part of the current sentence has already appeared in the unique sentences
        for unique_sentence in unique_sentences:
            # Normalize both sentences for comparison (lowercase and strip whitespace)
            normalized_current = sentence.lower().strip()
            normalized_unique = unique_sentence.lower().strip()
            
            # Check if the current sentence is a substring of a previous sentence or vice versa
            if normalized_current in normalized_unique or normalized_unique in normalized_current:
                # If a repetition is found, stop processing and return the cleaned transcription
                cleaned_transcription = ' '.join(unique_sentences)
                # Ensure the last character is a proper delimiter (period or question mark)
                if not cleaned_transcription.endswith(('.', '?')):
                    cleaned_transcription += '.'
                return cleaned_transcription
        
        # If no repetition is found, add the current sentence to the unique list
        unique_sentences.append(sentence.strip())
    
    # If no repetition is found in the entire transcription, join all sentences
    cleaned_transcription = ' '.join(unique_sentences)
    
    # Ensure the final transcription ends with a proper punctuation mark
    if not cleaned_transcription.endswith(('.', '?')):
        cleaned_transcription += '.'
        
    return cleaned_transcription
