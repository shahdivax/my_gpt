# Custom GPT Language Model

A custom GPT-style language model trained on the HuggingFaceFW/fineweb dataset. This model is designed for efficient training on consumer GPUs while maintaining good performance characteristics.

## Model Architecture

- **Architecture Type**: GPT-style Transformer
- **Model Size**: ~100M parameters
- **Context Length**: 512 tokens
- **Embedding Dimension**: 640
- **Attention Heads**: 10
- **Layers**: 12
- **Vocabulary Size**: 50,000 tokens
- **Training Precision**: Mixed FP16

### Parameter Count Breakdown
- Token Embeddings: 32M parameters (50,000 × 640)
- Position Embeddings: 0.3M parameters (512 × 640)
- Transformer Blocks: 67.7M parameters (12 layers × [attention + feed-forward])
  - Each block: ~5.6M parameters
  - Self-attention: 1.6M parameters per block
  - Feed-forward: 4M parameters per block
- Layer Normalization: ~0.003M parameters
- Total: ~100M parameters

## Features

- ByteLevelBPE tokenizer with special tokens support
- DeepSpeed ZeRO Stage-2 optimization
- Gradient checkpointing option for memory efficiency
- Streaming dataset support for handling large datasets
- Wandb integration for experiment tracking
- FP16 mixed precision training
- Efficient data loading with dynamic batching

## Training

The model is trained using:
- HuggingFaceFW/fineweb dataset
- AdamW optimizer with weight decay
- Learning rate: 1e-4 with warmup
- Gradient clipping at 1.0
- Batch size: 64 (8 per GPU × 8 gradient accumulation steps)
- Training epochs: 3
- Target dataset size: 2.5GB

## Requirements

```bash
pip install torch transformers accelerate deepspeed wandb tqdm
```

## Usage

### Training

```python
python src/train.py
```

### Inference

```python
from src.inference import generate_text
prompt = "Once upon a time"
generated_text = generate_text(
prompt=prompt,
max_new_tokens=50,
temperature=0.7,
top_k=50,
top_p=0.95
)
print(generated_text)
```

## Configuration

The model and training parameters can be configured in `config/config.yaml`. Key configurations include:

```yaml
model:
vocab_size: 50000
n_embd: 640
n_layer: 12
n_head: 10

n_positions: 512
training:
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 0.0001
```


## Performance

The model uses several optimizations for efficient training:
- DeepSpeed ZeRO Stage-2 for memory optimization
- FP16 mixed precision training
- Gradient accumulation for larger effective batch sizes
- Efficient data streaming for handling large datasets
