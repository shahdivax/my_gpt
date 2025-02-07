import torch
import torch.nn as nn
from transformers import AutoTokenizer
from utils import load_config
from tokenizers import Tokenizer
import os
import json

class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward layers"""
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Ensure mask is same dtype as input
        if mask is not None:
            mask = mask.to(dtype=x.dtype)
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_out)
        x = self.ln1(x)
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = x + self.dropout(ff_out)
        x = self.ln2(x)
        return x

class CustomLanguageModel(nn.Module):
    """Custom transformer-based language model"""
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config["model"]["vocab_size"]
        self.n_embd = config["model"]["n_embd"]
        self.n_head = config["model"]["n_head"]
        self.n_layer = config["model"]["n_layer"]
        self.n_positions = config["model"]["n_positions"]
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding = nn.Embedding(self.n_positions, self.n_embd)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.n_embd, self.n_head)
            for _ in range(self.n_layer)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        
        # Tie weights between token embedding and output layer
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Set gradient checkpointing flag based on config
        self.gradient_checkpointing_enable = config["model"].get("gradient_checkpointing", False)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, labels=None):
        batch_size, seq_length = input_ids.shape
        
        # Create position indices
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings and sum token & position embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        x = token_embeddings + position_embeddings
        
        # Create causal mask and convert to same dtype as embeddings
        mask = torch.triu(torch.ones((seq_length, seq_length), device=input_ids.device) * float('-inf'), diagonal=1)
        mask = mask.to(dtype=x.dtype)
        
        # Process through transformer blocks (use gradient checkpointing only if enabled)
        if self.training and self.gradient_checkpointing_enable:
            for block in self.transformer_blocks:
                x = torch.utils.checkpoint.checkpoint(block, x, mask, use_reentrant=False)
        else:
            for block in self.transformer_blocks:
                x = block(x, mask=mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
            return {"loss": loss, "logits": logits}
        
        return {"logits": logits}

    def num_parameters(self):
        """Returns the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_model(config):
    """Creates a custom language model from scratch based on the configuration."""
    model = CustomLanguageModel(config)
    return model

def get_tokenizer(config):
    """Loads a trained ByteLevelBPE tokenizer."""
    from tokenizers import ByteLevelBPETokenizer
    
    model_path = config["tokenizer"]["model_path"]
    if not os.path.exists(os.path.join(model_path, "vocab.json")):
        raise ValueError(f"No tokenizer found at {model_path}. Please train the tokenizer first.")
    
    tokenizer = ByteLevelBPETokenizer(
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
    tokenizer.add_special_tokens(list(special_tokens.values()))
    
    # Add methods to match expected interface
    tokenizer.get_vocab_size = lambda: len(tokenizer.get_vocab())
    
    def batch_encode(texts, padding=True, truncation=True, max_length=None, return_tensors=None):
        encodings = tokenizer.encode_batch(texts)
        # Extract token ids from encodings
        token_ids = [enc.ids for enc in encodings]
        
        if max_length and truncation:
            token_ids = [ids[:max_length] for ids in token_ids]
        
        if padding:
            max_len = max(len(ids) for ids in token_ids)
            pad_token_id = tokenizer.token_to_id("<|pad|>")
            padded = []
            for ids in token_ids:
                pad_length = max_len - len(ids)
                padded.append(ids + [pad_token_id] * pad_length)
            token_ids = padded
        
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(token_ids),
                "attention_mask": torch.ones_like(torch.tensor(token_ids))
            }
        return {"input_ids": token_ids}
    
    tokenizer.batch_encode = batch_encode
    
    print(f"ByteLevelBPE tokenizer loaded successfully. Vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer

if __name__ == "__main__":
    config = load_config()
    tokenizer = get_tokenizer(config)
    config["model"]["vocab_size"] = tokenizer.get_vocab_size()
    model = create_model(config)
    print(f"Model created with {model.num_parameters():,} parameters.")
    print(model)
