from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from tokenizers.implementations import ByteLevelBPETokenizer
import os
from utils import load_config, setup_logging
from glob import glob
from tqdm import tqdm
import json
import torch

class CustomTokenizer:
    """Wrapper around ByteLevelBPETokenizer with additional functionality."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._vocab_size = len(tokenizer.get_vocab())
        self.pad_token_id = tokenizer.token_to_id("<|pad|>")
        self.eos_token_id = tokenizer.token_to_id("<|endoftext|>")
    
    def get_vocab_size(self):
        return self._vocab_size
    
    def batch_encode(self, texts, padding=True, truncation=True, max_length=None, return_tensors=None):
        encodings = self.tokenizer.encode_batch(texts)
        if max_length and truncation:
            encodings = [enc.ids[:max_length] for enc in encodings]
        if padding:
            max_len = max(len(enc.ids) for enc in encodings)
            padded = []
            for enc in encodings:
                pad_length = max_len - len(enc.ids)
                padded.append(enc.ids + [self.pad_token_id] * pad_length)
            encodings = padded
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(encodings),
                "attention_mask": torch.ones_like(torch.tensor(encodings))
            }
        return {"input_ids": encodings}
    
    def decode(self, token_ids):
        """Decode a list of token IDs back to a string."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        # Filter out padding tokens
        token_ids = [t for t in token_ids if t != self.pad_token_id]
        
        # Use the underlying tokenizer's decode method
        return self.tokenizer.decode(token_ids)

def train_tokenizer(config):
    """Trains a custom BPE tokenizer using the tokenizers library."""
    setup_logging()

    model_path = config["tokenizer"]["model_path"]
    vocab_size = config["tokenizer"].get("vocab_size", 50000)
    min_frequency = config["tokenizer"].get("min_frequency", 2)
    
    # Create output directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)

    # Initialize a new tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Get all text files from the data directory
    data_files = glob(os.path.join("data/raw", "*.txt"))
    if not data_files:
        raise ValueError("No text files found in data/raw directory")

    print(f"Training tokenizer on {len(data_files)} files...")
    print(f"Target vocab size: {vocab_size}")
    print(f"Min frequency: {min_frequency}")

    # Train the tokenizer
    tokenizer.train(
        files=data_files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[
            "<|endoftext|>",  # End of text token
            "<|pad|>",        # Padding token
            "<|unk|>",        # Unknown token
            "<|mask|>"        # Mask token
        ]
    )

    # Save the tokenizer files
    tokenizer.save_model(model_path)
    
    # Save the tokenizer configuration
    tokenizer_config = {
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "model_type": "byte_level_bpe",
        "special_tokens": {
            "eos_token": "<|endoftext|>",
            "pad_token": "<|pad|>",
            "unk_token": "<|unk|>",
            "mask_token": "<|mask|>"
        }
    }
    
    with open(os.path.join(model_path, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    print(f"Tokenizer trained and saved to {model_path}")
    return tokenizer

def get_tokenizer(config):
    """Loads a trained tokenizer."""
    model_path = config["tokenizer"]["model_path"]
    
    if not os.path.exists(os.path.join(model_path, "vocab.json")):
        raise ValueError(f"No tokenizer found at {model_path}. Please train the tokenizer first.")
    
    base_tokenizer = ByteLevelBPETokenizer(
        os.path.join(model_path, "vocab.json"),
        os.path.join(model_path, "merges.txt")
    )
    
    # Add special tokens if they don't exist
    special_tokens = {
        "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
        "mask_token": "<|mask|>"
    }
    base_tokenizer.add_special_tokens(list(special_tokens.values()))
    
    # Create wrapped tokenizer
    tokenizer = CustomTokenizer(base_tokenizer)
    
    print(f"ByteLevelBPE tokenizer loaded successfully. Vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer

if __name__ == "__main__":
    config = load_config()
    train_tokenizer(config)
    print("Tokenizer training complete.")