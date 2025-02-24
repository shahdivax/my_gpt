# import torch
# from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
# from typing import Optional, Tuple, Union, List
# import os
# import json
# from model import CustomLanguageModel
# from utils import load_config
# from tokenization import get_tokenizer

# class CustomConfig(PretrainedConfig):
#     """Configuration class for the custom language model."""
#     model_type = "custom_llm"
    
#     def __init__(
#         self,
#         vocab_size: int = 50000,
#         n_embd: int = 768,
#         n_head: int = 12,
#         n_layer: int = 12,
#         n_positions: int = 2048,
#         tie_word_embeddings: bool = False,
#         **kwargs
#     ):
#         self.vocab_size = vocab_size
#         self.n_embd = n_embd
#         self.n_head = n_head
#         self.n_layer = n_layer
#         self.n_positions = n_positions
#         self.tie_word_embeddings = tie_word_embeddings
#         super().__init__(**kwargs)

# class CustomModelForCausalLM(PreTrainedModel):
#     """Wrapper class to make the model compatible with Hugging Face's interface."""
#     config_class = CustomConfig
#     supports_gradient_checkpointing = True
    
#     def __init__(self, config):
#         super().__init__(config)
#         # Convert config to dictionary format expected by CustomLanguageModel
#         model_config = {
#             "model": {
#                 "vocab_size": config.vocab_size,
#                 "n_embd": config.n_embd,
#                 "n_head": config.n_head,
#                 "n_layer": config.n_layer,
#                 "n_positions": config.n_positions,
#             }
#         }
#         self.transformer = CustomLanguageModel(model_config)
        
#         # Tie weights if specified in config
#         if getattr(config, "tie_word_embeddings", True):
#             self.transformer.lm_head.weight = self.transformer.token_embedding.weight
        
#     def forward(
#         self,
#         input_ids: torch.LongTensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         **kwargs
#     ):
#         return self.transformer(input_ids=input_ids, labels=labels)
    
#     def generate(
#         self,
#         input_ids: torch.LongTensor,
#         max_length: int = 100,
#         temperature: float = 1.0,
#         top_k: int = 50,
#         top_p: float = 0.9,
#         repetition_penalty: float = 1.2,
#         no_repeat_ngram_size: int = 3,
#         **kwargs
#     ):
#         """Enhanced generation method with better controls for repetition."""
#         self.eval()
#         current_ids = input_ids.clone()
#         batch_size = current_ids.shape[0]
        
#         # Get EOS token ID from tokenizer
#         eos_token_id = self.transformer.eos_token_id if hasattr(self.transformer, 'eos_token_id') else None
        
#         # Track generated tokens for repetition penalty
#         generated_tokens = current_ids.clone()
        
#         with torch.no_grad():
#             for _ in range(max_length - input_ids.size(1)):
#                 # Forward pass
#                 outputs = self.transformer(current_ids)
#                 logits = outputs["logits"][:, -1, :] / temperature
                
#                 # Apply repetition penalty
#                 if repetition_penalty != 1.0:
#                     for i in range(batch_size):
#                         for token in set(generated_tokens[i].tolist()):
#                             logits[i, token] /= repetition_penalty
                
#                 # Apply n-gram blocking
#                 if no_repeat_ngram_size > 0:
#                     # Get the last n-gram from the input
#                     for i in range(batch_size):
#                         ngram_size = min(no_repeat_ngram_size, len(generated_tokens[i]))
#                         if ngram_size > 0:
#                             ngrams = [tuple(generated_tokens[i, -j:].tolist()) for j in range(1, ngram_size + 1)]
#                             for ngram in ngrams:
#                                 for token_idx in range(len(generated_tokens[i]) - len(ngram) + 1):
#                                     if tuple(generated_tokens[i, token_idx:token_idx + len(ngram)].tolist()) == ngram:
#                                         if token_idx + len(ngram) < len(generated_tokens[i]):
#                                             next_token = generated_tokens[i, token_idx + len(ngram)]
#                                             logits[i, next_token] = float('-inf')
                
#                 # Apply top-k filtering
#                 if top_k > 0:
#                     indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
#                     logits[indices_to_remove] = float('-inf')
                
#                 # Apply top-p (nucleus) filtering
#                 if top_p < 1.0:
#                     sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#                     cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
#                     sorted_indices_to_remove = cumulative_probs > top_p
#                     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#                     sorted_indices_to_remove[..., 0] = 0
#                     indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
#                     logits[indices_to_remove] = float('-inf')
                
#                 # Sample from the filtered distribution
#                 probs = torch.softmax(logits, dim=-1)
#                 next_token = torch.multinomial(probs, num_samples=1)
                
#                 # Early stopping if EOS token is generated
#                 if eos_token_id is not None and (next_token == eos_token_id).any():
#                     break
                
#                 # Update generated sequence
#                 current_ids = torch.cat([current_ids, next_token], dim=1)
#                 generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
                
#             return current_ids

# def convert_to_hf_model(checkpoint_path: str, output_dir: str):
#     """Convert the custom model checkpoint to Hugging Face format."""
#     # Load the original config and checkpoint
#     config = load_config()
    
#     # Get tokenizer and its vocab size
#     tokenizer = get_tokenizer(config)
#     vocab_size = tokenizer.get_vocab_size()
    
#     # Create HF config with the correct vocab size
#     hf_config = CustomConfig(
#         vocab_size=vocab_size,
#         n_embd=config["model"]["n_embd"],
#         n_head=config["model"]["n_head"],
#         n_layer=config["model"]["n_layer"],
#         n_positions=config["model"]["n_positions"],
#         tie_word_embeddings=True
#     )
    
#     # Create HF model
#     model = CustomModelForCausalLM(hf_config)
    
#     # Load checkpoint
#     checkpoint = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location="cpu")
    
#     # Remove "_orig_mod." prefix from state dict keys
#     new_state_dict = {}
#     for key, value in checkpoint.items():
#         if key.startswith("_orig_mod."):
#             new_key = "transformer." + key[len("_orig_mod."):]
#             new_state_dict[new_key] = value
#         else:
#             new_state_dict["transformer." + key] = value
    
#     # Load the modified state dict
#     model.load_state_dict(new_state_dict)
    
#     # Save in Hugging Face format
#     os.makedirs(output_dir, exist_ok=True)
    
#     try:
#         # First try to save with safetensors
#         model.save_pretrained(
#             output_dir,
#             safe_serialization=True
#         )
#         print(f"Model successfully saved in safetensors format to {output_dir}")
#     except RuntimeError as e:
#         print("Could not save in safetensors format due to weight sharing. Falling back to PyTorch format.")
#         # If safetensors fails, fall back to PyTorch format
#         model.save_pretrained(
#             output_dir,
#             safe_serialization=False
#         )
#         print(f"Model successfully saved in PyTorch format to {output_dir}")
    
#     # Save config
#     hf_config.save_pretrained(output_dir)
    
#     # Copy tokenizer files
#     tokenizer_files = ["vocab.json", "merges.txt", "tokenizer_config.json"]
#     for file in tokenizer_files:
#         src_path = os.path.join(config["tokenizer"]["model_path"], file)
#         dst_path = os.path.join(output_dir, file)
#         if os.path.exists(src_path):
#             import shutil
#             shutil.copy2(src_path, dst_path)
    
#     return model, tokenizer

# def generate_text(
#     prompt: str,
#     model_path: str,
#     max_length: int = 100,
#     temperature: float = 2,
#     top_k: int = 50,
#     top_p: float = 0.9,
#     repetition_penalty: float = 1.2,
#     no_repeat_ngram_size: int = 3
# ):
#     """Generate text using the converted model."""
#     # Load model and tokenizer
#     config = load_config()
#     model = CustomModelForCausalLM.from_pretrained(model_path)
#     tokenizer = get_tokenizer(config)
    
#     # Move model to GPU if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model.eval()
    
#     # Encode prompt
#     encoded = tokenizer.batch_encode(
#         [prompt],
#         return_tensors="pt"
#     )
#     input_ids = encoded["input_ids"].to(device)
    
#     # Generate
#     with torch.no_grad():
#         output_ids = model.generate(
#             input_ids=input_ids,
#             max_length=max_length,
#             temperature=temperature,
#             top_k=top_k,
#             top_p=top_p,
#             repetition_penalty=repetition_penalty,
#             no_repeat_ngram_size=no_repeat_ngram_size
#         )
    
#     # Decode and return
#     generated_text = tokenizer.decode(output_ids[0].tolist())
#     return generated_text

# if __name__ == "__main__":
#     # Example usage
#     checkpoint_path = r"my_model/"  # Path to your trained model
#     hf_output_dir = "outputs/hf_model"  # Where to save the converted model
    

#     # Convert model
#     model, tokenizer = convert_to_hf_model(checkpoint_path, hf_output_dir)
    
#     # Generate text with better parameters
#     prompt = "Hello I am Clera "
#     generated_text = generate_text(
#         prompt=prompt,
#         model_path=hf_output_dir,
#         max_length=20,
#         temperature=2.5,
#         top_k=50,
#         top_p=0.9,
#         repetition_penalty=1.2,
#         no_repeat_ngram_size=1
#     )
    
#     print(f"\nPrompt: {prompt}")
#     print(f"Generated text: {generated_text}")

from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_text(model_path: str, prompt: str, max_length=100, temperature=0.7, top_k=50, top_p=0.95):
    """Generate text using the trained model with HF's Auto classes."""
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    model_path = "./my_model/final_model"
    prompt = "Once upon a time"
    generated_text = generate_text(
        model_path=model_path,
        prompt=prompt,
        max_length=50,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")