import torch
from transformers import PreTrainedModel, PretrainedConfig
from utils import load_config
from tokenization import get_tokenizer

class CustomConfig(PretrainedConfig):
    """Configuration class for the custom language model."""
    model_type = "custom_llm"
    
    def __init__(
        self,
        vocab_size: int = 50000,
        n_embd: int = 640,
        n_head: int = 10,
        n_layer: int = 12,
        n_positions: int = 512,
        tie_word_embeddings: bool = True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_positions = n_positions
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)

def generate_text(
    prompt: str,
    model_path: str = "outputs/hf_model",
    max_length: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3
):
    """Generate text using the model."""
    # Load config and tokenizer
    config = load_config()
    tokenizer = get_tokenizer(config)
    
    # Load model
    from inference import CustomModelForCausalLM  # Import here to avoid circular imports
    model = CustomModelForCausalLM.from_pretrained(model_path)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Encode prompt
    encoded = tokenizer.batch_encode(
        [prompt],
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size
        )
    
    # Decode and return
    generated_text = tokenizer.decode(output_ids[0].tolist())
    return generated_text

if __name__ == "__main__":
    # Example prompts to test
    prompts = [
        "Once upon a time",
        "The meaning of life is",
        "In the distant future",
        "The best way to learn programming is",
        "Today I learned that"
    ]
    
    print("\nGenerating text from multiple prompts:")
    print("=" * 50)
    
    for prompt in prompts:
        generated_text = generate_text(
            prompt=prompt,
            max_length=200,
            temperature=0.8,  # Adjust for creativity (higher = more creative)
            top_k=50,        # Limit to top 50 tokens
            top_p=0.9,       # Nucleus sampling threshold
            repetition_penalty=1.2,  # Penalize repetition
            no_repeat_ngram_size=3   # Prevent 3-gram repetition
        )
        
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("-" * 50) 