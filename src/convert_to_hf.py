#!/usr/bin/env python
import os
import argparse
import yaml
import json
import torch
import shutil
import tiktoken
from model import create_model  # your model creation function from model.py

def load_config(config_path):
    """Load the training configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_tokenizer_encoding(tokenizer_dir):
    """Reads the encoding name from encoding_config.txt in your tokenizer directory."""
    encoding_config_path = os.path.join(tokenizer_dir, "encoding_config.txt")
    if not os.path.exists(encoding_config_path):
        raise FileNotFoundError(f"Encoding config not found at {encoding_config_path}")
    
    with open(encoding_config_path, "r") as f:
        content = f.read().strip()
    # Expect a line like: "encoding_name: cl100k_base"
    if ":" not in content:
        raise ValueError(f"Invalid encoding config format: {content}")
    
    _, encoding_name = content.split(":", 1)
    return encoding_name.strip()

def get_tokenizer(encoding_name):
    """Initialize tiktoken encoding."""
    tokenizer = tiktoken.get_encoding(encoding_name)
    return tokenizer

def load_state_dict(checkpoint_dir):
    """
    Loads the model state dict from a DeepSpeed checkpoint.
    First tries to load a consolidated checkpoint, then attempts to convert from ZeRO format.
    """
    # First try loading from converted_model directory
    converted_path = os.path.join(checkpoint_dir, "converted_model", "pytorch_model.bin")
    if os.path.exists(converted_path):
        print(f"Loading converted checkpoint from {converted_path}")
        state_dict = torch.load(converted_path, map_location="cpu")
        
        # Remove "_orig_mod." prefix from keys if present
        if all(k.startswith("_orig_mod.") for k in state_dict.keys()):
            print("Removing '_orig_mod.' prefix from state dict keys")
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        
        return state_dict

    # Try loading consolidated checkpoint from main directory
    consolidated_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(consolidated_path):
        print(f"Loading consolidated checkpoint from {consolidated_path}")
        state_dict = torch.load(consolidated_path, map_location="cpu")
        
        # Remove "_orig_mod." prefix from keys if present
        if all(k.startswith("_orig_mod.") for k in state_dict.keys()):
            print("Removing '_orig_mod.' prefix from state dict keys")
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            
        return state_dict

    # If no consolidated checkpoint exists, try converting from ZeRO format
    print("No consolidated checkpoint found. Converting from ZeRO format...")
    
    # Import the zero_to_fp32 module from the checkpoint directory
    import sys
    sys.path.append(checkpoint_dir)
    from zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    
    try:
        # Convert ZeRO checkpoint to consolidated checkpoint
        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, exclude_frozen_parameters=False)
        
        if state_dict is None:
            raise ValueError("Failed to convert ZeRO checkpoint")
        
        # Remove "_orig_mod." prefix from keys if present
        if all(k.startswith("_orig_mod.") for k in state_dict.keys()):
            print("Removing '_orig_mod.' prefix from state dict keys")
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        
        print("Successfully converted ZeRO checkpoint to consolidated format")
        return state_dict
        
    except Exception as e:
        print(f"Error converting ZeRO checkpoint: {str(e)}")
        raise

def convert_to_hf(checkpoint_dir, tokenizer_dir, config_path, output_dir):
    # Load configurations
    config = load_config(config_path)
    
    # Set up tokenizer
    encoding_name = load_tokenizer_encoding(tokenizer_dir)
    tokenizer = get_tokenizer(encoding_name)
    vocab_size = tokenizer.n_vocab
    print(f"Using tokenizer encoding: {encoding_name} (vocab size: {vocab_size})")
    
    # Update config with correct vocab size
    config["model"]["vocab_size"] = vocab_size
    
    # Create model and load weights
    model = create_model(config)
    state_dict = load_state_dict(checkpoint_dir)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save model weights
    model_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model weights to {model_path}")
    
    # 2. Save model config
    model_config = {
        "architectures": ["CustomLanguageModel"],
        "model_type": "custom-gpt",
        "vocab_size": vocab_size,
        "n_positions": config["model"]["n_positions"],
        "n_embd": config["model"]["n_embd"],
        "n_layer": config["model"]["n_layer"],
        "n_head": config["model"]["n_head"],
        "bos_token_id": None,
        "eos_token_id": tokenizer.eot_token,
        "tie_word_embeddings": True,
        "gradient_checkpointing": config["model"].get("gradient_checkpointing", False)
    }
    
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"Saved model config to {config_path}")
    
    # 3. Save tokenizer config
    tokenizer_config = {
        "model_type": "tiktoken",
        "encoding_name": encoding_name,
        "vocab_size": vocab_size,
        "max_length": config["dataset"]["max_length"],
        "padding_side": "right",
        "truncation_side": "right",
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>"
    }
    
    tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(tokenizer_config_path, "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"Saved tokenizer config to {tokenizer_config_path}")
    
    # 4. Copy tokenizer files
    src_encoding_config = os.path.join(tokenizer_dir, "encoding_config.txt")
    if os.path.exists(src_encoding_config):
        dst_encoding_config = os.path.join(output_dir, "encoding_config.txt")
        shutil.copy2(src_encoding_config, dst_encoding_config)
        print(f"Copied encoding config to {dst_encoding_config}")
    
    print(f"\nConversion complete! HuggingFace model saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert DeepSpeed checkpoint to HuggingFace format")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Path to the checkpoint directory")
    parser.add_argument("--tokenizer_dir", required=True,
                        help="Path to the tokenizer directory")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to the training config.yaml file")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for HuggingFace model")
    
    args = parser.parse_args()
    convert_to_hf(args.checkpoint_dir, args.tokenizer_dir, args.config, args.output_dir)

if __name__ == "__main__":
    main()
