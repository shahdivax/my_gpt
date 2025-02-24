# import os
# import math
# import time
# import json
# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# from torch.utils.data import DataLoader, Dataset, IterableDataset
# from tqdm import tqdm
# from accelerate import Accelerator, DeepSpeedPlugin
# from accelerate.logging import get_logger
# import deepspeed
# import wandb
# from datetime import datetime
# from transformers import get_scheduler
# from model import create_model, get_tokenizer
# from utils import load_config, setup_logging
# from torch.nn.utils.rnn import pad_sequence

# logger = get_logger(__name__)
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# # Enable TF32 for faster matrix multiplications (if supported)
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# def load_text_files(data_dir, chunk_size=2000000):
#     """Load text files from directory in chunks."""
#     if not os.path.exists(data_dir):
#         raise ValueError(f"Data directory {data_dir} does not exist")
    
#     all_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
#     print(f"Found {len(all_files)} text files in {data_dir}")
    
#     total_size = sum(os.path.getsize(os.path.join(data_dir, f)) for f in all_files)
#     estimated_chunks = math.ceil(total_size / chunk_size)
#     total_characters = 0
#     current_chunk_num = 0
    
#     for file_name in all_files:
#         file_path = os.path.join(data_dir, file_name)
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 file_size = os.path.getsize(file_path)
#                 print(f"Processing file: {file_name} (Size: {file_size/1024/1024:.2f}MB)")
#                 print(f"Estimated total chunks: {estimated_chunks}")
                
#                 current_chunk = []
#                 current_size = 0
#                 chunk_start_char = total_characters
                
#                 for line in f:
#                     line = line.strip()
#                     if line:
#                         current_chunk.append(line)
#                         current_size += len(line)
#                         total_characters += len(line)
                        
#                         if current_size >= chunk_size:
#                             current_chunk_num += 1
#                             print(f"Yielding chunk {current_chunk_num}/{estimated_chunks} "
#                                   f"({len(current_chunk)} texts, {current_size:,} characters, "
#                                   f"Range: {chunk_start_char:,} - {total_characters:,})")
#                             yield current_chunk
#                             current_chunk = []
#                             current_size = 0
#                             chunk_start_char = total_characters
                
#                 if current_chunk:
#                     current_chunk_num += 1
#                     print(f"Yielding final chunk {current_chunk_num}/{estimated_chunks} "
#                           f"({len(current_chunk)} texts, {current_size:,} characters, "
#                           f"Range: {chunk_start_char:,} - {total_characters:,})")
#                     yield current_chunk
#         except Exception as e:
#             print(f"Error reading file {file_path}: {e}")
#             continue

# class TextDataset(Dataset):
#     def __init__(self, tokenized_texts):
#         self.input_ids = tokenized_texts["input_ids"]
#         self.labels = tokenized_texts["labels"]
    
#     def __len__(self):
#         return len(self.input_ids)
    
#     def __getitem__(self, idx):
#         return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}

# class StreamingTextDataset(IterableDataset):
#     def __init__(self, data_dir, tokenizer, max_length):
#         super().__init__()
#         self.data_dir = data_dir
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]

#     def __iter__(self):
#         worker_info = torch.utils.data.get_worker_info()
#         files_per_worker = len(self.files)
#         if worker_info is not None:
#             files_per_worker = len(self.files) // worker_info.num_workers
#             start_idx = worker_info.id * files_per_worker
#             end_idx = start_idx + files_per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.files)
#             files = self.files[start_idx:end_idx]
#         else:
#             files = self.files

#         for file_name in files:
#             file_path = os.path.join(self.data_dir, file_name)
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 text_buffer = []
#                 current_length = 0
                
#                 for line in f:
#                     line = line.strip()
#                     if not line:
#                         continue
                    
#                     text_buffer.append(line)
#                     current_length += len(line)
                    
#                     if current_length >= self.max_length:
#                         # Encode and yield the batch
#                         text = " ".join(text_buffer)
#                         encodings = self.tokenizer.batch_encode(
#                             [text],
#                             max_length=self.max_length,
#                             truncation=True,
#                             padding=False,  # Don't pad here, we'll pad in collate_fn
#                             return_tensors="pt"
#                         )
                        
#                         # Return individual tensors
#                         yield {
#                             "input_ids": encodings["input_ids"][0],
#                             "labels": encodings["input_ids"][0].clone()
#                         }
#                         text_buffer = []
#                         current_length = 0

# def collate_batch(batch):
#     """Custom collate function to handle variable length sequences."""
#     # Separate input_ids and labels
#     input_ids = [item["input_ids"] for item in batch]
#     labels = [item["labels"] for item in batch]
    
#     # Pad sequences
#     input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
#     labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 is PyTorch's default ignore index
    
#     # Create attention masks
#     attention_mask = (input_ids != 0).long()
    
#     return {
#         "input_ids": input_ids,
#         "labels": labels,
#         "attention_mask": attention_mask
#     }

# def train_model(config):
#     """Trains the model using DeepSpeed and Accelerate for memory efficiency."""
#     # Create output directory
#     output_dir = config["training"]["output_dir"]
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"Model will be saved to: {output_dir}")
    
#     # Initialize DeepSpeed plugin and accelerator
#     deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=config["training"]["deepspeed"])
#     accelerator = Accelerator(
#         gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
#         mixed_precision="fp16",
#         deepspeed_plugin=deepspeed_plugin,
#         log_with=config["training"]["report_to"]
#     )
    
#     # Initialize tracking
#     if accelerator.is_main_process:
#         accelerator.init_trackers(
#             project_name=config["training"]["wandb"]["project"],
#             config=config,
#             init_kwargs={
#                 "wandb": {
#                     "entity": config["training"]["wandb"]["entity"],
#                     "name": config["training"]["wandb"]["name"],
#                 }
#             }
#         )
#         print(f"Tracking initialized with {config['training']['report_to']}")
    
#     device = accelerator.device
#     print(f"Using device: {device}")
    
#     # Load tokenizer and model
#     tokenizer = get_tokenizer(config)
#     config["model"]["vocab_size"] = tokenizer.get_vocab_size()
#     model = create_model(config)
    
#     try:
#         model = torch.compile(model)
#         print("torch.compile enabled for faster training.")
#     except Exception as e:
#         print("torch.compile not available or failed, continuing without it.")
    
#     optimizer = AdamW(
#         model.parameters(),
#         lr=config["training"]["learning_rate"],
#         weight_decay=config["training"]["weight_decay"]
#     )
    
#     # Create streaming dataset with custom collate function
#     dataset = StreamingTextDataset(
#         data_dir="data/raw",
#         tokenizer=tokenizer,
#         max_length=config["dataset"]["max_length"]
#     )
    
#     train_loader = DataLoader(
#         dataset,
#         batch_size=config["training"]["per_device_train_batch_size"],
#         num_workers=config["training"]["dataloader_num_workers"],
#         pin_memory=True,
#         collate_fn=collate_batch  # Add custom collate function
#     )
    
#     # Prepare for distributed training
#     model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
#     # Calculate approximate steps per epoch based on target dataset size
#     avg_seq_length = config["dataset"]["max_length"] // 2  # Average sequence length
#     batch_size = config["training"]["per_device_train_batch_size"]
#     target_size_gb = config["dataset"].get("target_size_gb", 2.5)
#     chars_per_token = 4
#     total_tokens = (target_size_gb * 1024 * 1024 * 1024) // chars_per_token
#     steps_per_epoch = int(total_tokens // (avg_seq_length * batch_size))  # Convert to int
#     total_epochs = config["training"]["num_train_epochs"]
#     total_steps = int(steps_per_epoch * total_epochs)  # Convert to int
    
#     print(f"\nTraining Statistics (Estimated):")
#     print(f"Total epochs: {total_epochs}")
#     print(f"Estimated steps per epoch: {steps_per_epoch:,}")
#     print(f"Estimated total steps: {total_steps:,}")
    
#     # Track gradients for logging
#     def grad_norm(model):
#         total_norm = 0.0
#         for p in model.parameters():
#             if p.grad is not None:
#                 param_norm = p.grad.detach().data.norm(2)
#                 total_norm += param_norm.item() ** 2
#         return total_norm ** 0.5
    
#     # Initialize GPU monitoring
#     if torch.cuda.is_available():
#         gpu_id = torch.cuda.current_device()
    
#     training_stats = {
#         'train/loss': 0.0,
#         'train/learning_rate': 0.0,
#         'train/epoch': 0.0,
#         'train/global_step': 0,
#         'train/samples_per_second': 0.0,
#         'train/grad_norm': 0.0,
#         'performance/gpu_memory': 0.0,
#         'performance/gpu_utilization': 0.0,
#         'performance/batch_time': 0.0,
#     }
    
#     for epoch in range(total_epochs):
#         epoch_start_time = time.time()
#         model.train()
#         running_loss = 0
#         num_batches = 0
#         samples_processed = 0
        
#         progress_bar = tqdm(
#             total=steps_per_epoch,
#             desc=f"Epoch {epoch+1}/{total_epochs}",
#             disable=not accelerator.is_local_main_process
#         )
        
#         for batch in train_loader:
#             batch_start_time = time.time()
            
#             with accelerator.accumulate(model):
#                 outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
#                 loss = outputs["loss"]
#                 accelerator.backward(loss)
                
#                 if accelerator.sync_gradients:
#                     training_stats['train/grad_norm'] = grad_norm(model)
#                     accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
#                 optimizer.step()
#                 optimizer.zero_grad()
            
#             # Update statistics
#             loss_value = loss.item()
#             running_loss += loss_value
#             num_batches += 1
#             samples_processed += batch["input_ids"].size(0)
#             batch_time = time.time() - batch_start_time
            
#             # Update training stats
#             training_stats.update({
#                 'train/loss': loss_value,
#                 'train/learning_rate': optimizer.param_groups[0]['lr'],
#                 'train/epoch': epoch + 1,
#                 'train/global_step': num_batches + (epoch * steps_per_epoch),
#                 'train/samples_per_second': batch["input_ids"].size(0) / batch_time,
#                 'performance/batch_time': batch_time,
#             })
            
#             # GPU stats (if available)
#             if torch.cuda.is_available():
#                 training_stats.update({
#                     'performance/gpu_memory': torch.cuda.memory_allocated(gpu_id) / 1024**3,  # GB
#                     'performance/gpu_utilization': torch.cuda.utilization(gpu_id),
#                 })
            
#             # Update progress bar
#             avg_speed = num_batches / (time.time() - epoch_start_time)
#             eta_epoch = (steps_per_epoch - num_batches) / avg_speed / 60  # minutes
#             eta_total = (total_steps - (epoch * steps_per_epoch + num_batches)) / avg_speed / 60  # minutes
            
#             progress_bar.set_postfix({
#                 'loss': f'{loss_value:.4f}',
#                 'avg_loss': f'{running_loss/num_batches:.4f}',
#                 'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
#                 'samples/s': f'{training_stats["train/samples_per_second"]:.2f}',
#                 'epoch_eta': f'{eta_epoch:.1f}min',
#                 'total_eta': f'{eta_total:.1f}min'
#             })
#             progress_bar.update(1)
            
#             # Log metrics based on logging_steps
#             if num_batches % config["training"]["logging_steps"] == 0:
#                 if accelerator.is_main_process:
#                     current_step = int(num_batches + (epoch * steps_per_epoch))  # Convert to int
#                     accelerator.log(training_stats, step=current_step)
            
#             # Save checkpoint based on save_steps
#             if num_batches % config["training"]["save_steps"] == 0:
#                 if accelerator.is_local_main_process:
#                     checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch{epoch+1}-step{num_batches}")
#                     os.makedirs(checkpoint_dir, exist_ok=True)
#                     print(f"\nSaving checkpoint at step {num_batches} to {checkpoint_dir}")
#                     accelerator.save_state(checkpoint_dir)
#                     with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
#                         json.dump(config, f, indent=2)
            
#             # Break if we've reached the estimated steps for this epoch
#             if num_batches >= steps_per_epoch:
#                 break
        
#         progress_bar.close()
        
#         # End of epoch logging
#         epoch_time = time.time() - epoch_start_time
#         epoch_avg_loss = running_loss / num_batches
#         epoch_perplexity = torch.exp(torch.tensor(epoch_avg_loss))
        
#         if accelerator.is_main_process:
#             print(f"\nEpoch {epoch+1}/{total_epochs} Summary:")
#             print(f"Time: {epoch_time/60:.2f} minutes")
#             print(f"Average Loss: {epoch_avg_loss:.4f}")
#             print(f"Perplexity: {epoch_perplexity:.2f}")
#             print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
#             print(f"Samples Processed: {samples_processed:,}")
#             print(f"Average Speed: {samples_processed/epoch_time:.1f} samples/s")
            
#             # Estimate remaining time
#             epochs_remaining = total_epochs - (epoch + 1)
#             estimated_remaining_time = epochs_remaining * epoch_time / 60
#             print(f"Estimated time for remaining {epochs_remaining} epochs: {estimated_remaining_time:.1f} minutes")
            
#             # Log epoch summary to wandb with correct step
#             current_step = int((epoch + 1) * steps_per_epoch)  # Convert to int
#             accelerator.log({
#                 'epoch/average_loss': epoch_avg_loss,
#                 'epoch/perplexity': epoch_perplexity.item(),
#                 'epoch/time': epoch_time,
#                 'epoch/samples_processed': samples_processed,
#             }, step=current_step)
    
#     # Save final model
#     if accelerator.is_local_main_process:
#         final_model_dir = os.path.join(output_dir, "final_model")
#         os.makedirs(final_model_dir, exist_ok=True)
#         print(f"\nSaving final model to {final_model_dir}")
        
#         # Save with DeepSpeed
#         accelerator.save_state(final_model_dir)
        
#         # Save configuration
#         with open(os.path.join(final_model_dir, "config.json"), "w") as f:
#             json.dump(config, f, indent=2)
        
#         print("Final model saved successfully")
#         accelerator.end_training()

# if __name__ == "__main__":
#     config = load_config()
#     train_model(config)
#     print("Training complete.")

import os
import math
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
import wandb
from model import create_model, get_tokenizer, CustomConfig
from utils import load_config
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM

logger = get_logger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class StreamingTextDataset(IterableDataset):
    def __init__(self, data_dir, tokenizer, max_length):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files_per_worker = len(self.files)
        if worker_info is not None:
            files_per_worker = len(self.files) // worker_info.num_workers
            start_idx = worker_info.id * files_per_worker
            end_idx = start_idx + files_per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.files)
            files = self.files[start_idx:end_idx]
        else:
            files = self.files

        for file_name in files:
            file_path = os.path.join(self.data_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                text_buffer = []
                current_length = 0
                
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    text_buffer.append(line)
                    current_length += len(line)
                    
                    if current_length >= self.max_length:
                        text = " ".join(text_buffer)
                        encodings = self.tokenizer.encode(
                            text,
                            max_length=self.max_length,
                            truncation=True,
                            padding=False,
                            return_tensors="pt"
                        )
                        
                        input_ids = encodings["input_ids"][0]
                        yield {
                            "input_ids": input_ids,
                            "labels": input_ids.clone()
                        }
                        text_buffer = []
                        current_length = 0

def collate_batch(batch):
    """Custom collate function to handle variable length sequences."""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    attention_mask = (input_ids != 0).long()
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

def train_model(config):
    """Trains the model and saves it in HF format."""
    output_dir = config["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    print(f"Model will be saved to: {output_dir}")
    
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=config["training"]["deepspeed"])
    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        mixed_precision="fp16",
        deepspeed_plugin=deepspeed_plugin,
        log_with=config["training"]["report_to"]
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config["training"]["wandb"]["project"],
            config=config,
            init_kwargs={
                "wandb": {
                    "entity": config["training"]["wandb"]["entity"],
                    "name": config["training"]["wandb"]["name"],
                }
            }
        )
        print(f"Tracking initialized with {config['training']['report_to']}")
    
    device = accelerator.device
    print(f"Using device: {device}")
    
    tokenizer = get_tokenizer(config)
    config["model"]["vocab_size"] = tokenizer.vocab_size
    model = create_model(config)
    
    try:
        model = torch.compile(model)
        print("torch.compile enabled for faster training.")
    except Exception as e:
        print("torch.compile not available or failed, continuing without it.")
    
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    dataset = StreamingTextDataset(
        data_dir="data/raw",
        tokenizer=tokenizer,
        max_length=config["dataset"]["max_length"]
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=config["training"]["per_device_train_batch_size"],
        num_workers=config["training"]["dataloader_num_workers"],
        pin_memory=True,
        collate_fn=collate_batch
    )
    
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    avg_seq_length = config["dataset"]["max_length"] // 2
    batch_size = config["training"]["per_device_train_batch_size"]
    target_size_gb = config["dataset"].get("target_size_gb", 2.5)
    chars_per_token = 4
    total_tokens = (target_size_gb * 1024 * 1024 * 1024) // chars_per_token
    steps_per_epoch = int(total_tokens // (avg_seq_length * batch_size))
    total_epochs = config["training"]["num_train_epochs"]
    total_steps = int(steps_per_epoch * total_epochs)
    
    print(f"\nTraining Statistics (Estimated):")
    print(f"Total epochs: {total_epochs}")
    print(f"Estimated steps per epoch: {steps_per_epoch:,}")
    print(f"Estimated total steps: {total_steps:,}")
    
    def grad_norm(model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
    
    training_stats = {
        'train/loss': 0.0,
        'train/learning_rate': 0.0,
        'train/epoch': 0.0,
        'train/global_step': 0,
        'train/samples_per_second': 0.0,
        'train/grad_norm': 0.0,
        'performance/gpu_memory': 0.0,
        'performance/gpu_utilization': 0.0,
        'performance/batch_time': 0.0,
    }
    
    # Save tokenizer once at the start
    if accelerator.is_local_main_process:
        tokenizer.save_pretrained(output_dir)
        CustomConfig(**config["model"]).save_pretrained(output_dir)
    
    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0
        num_batches = 0
        samples_processed = 0
        
        progress_bar = tqdm(
            total=steps_per_epoch,
            desc=f"Epoch {epoch+1}/{total_epochs}",
            disable=not accelerator.is_local_main_process
        )
        
        for batch in train_loader:
            batch_start_time = time.time()
            
            with accelerator.accumulate(model):
                outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
                loss = outputs["loss"]
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    training_stats['train/grad_norm'] = grad_norm(model)
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()
            
            loss_value = loss.item()
            running_loss += loss_value
            num_batches += 1
            samples_processed += batch["input_ids"].size(0)
            batch_time = time.time() - batch_start_time
            
            training_stats.update({
                'train/loss': loss_value,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'train/epoch': epoch + 1,
                'train/global_step': num_batches + (epoch * steps_per_epoch),
                'train/samples_per_second': batch["input_ids"].size(0) / batch_time,
                'performance/batch_time': batch_time,
            })
            
            if torch.cuda.is_available():
                training_stats.update({
                    'performance/gpu_memory': torch.cuda.memory_allocated(gpu_id) / 1024**3,
                    'performance/gpu_utilization': torch.cuda.utilization(gpu_id),
                })
            
            avg_speed = num_batches / (time.time() - epoch_start_time)
            eta_epoch = (steps_per_epoch - num_batches) / avg_speed / 60
            eta_total = (total_steps - (epoch * steps_per_epoch + num_batches)) / avg_speed / 60
            
            progress_bar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'avg_loss': f'{running_loss/num_batches:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                'samples/s': f'{training_stats["train/samples_per_second"]:.2f}',
                'epoch_eta': f'{eta_epoch:.1f}min',
                'total_eta': f'{eta_total:.1f}min'
            })
            progress_bar.update(1)
            
            if num_batches % config["training"]["logging_steps"] == 0:
                if accelerator.is_main_process:
                    current_step = int(num_batches + (epoch * steps_per_epoch))
                    accelerator.log(training_stats, step=current_step)
            
            if num_batches % config["training"]["save_steps"] == 0:
                if accelerator.is_local_main_process:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch{epoch+1}-step{num_batches}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    print(f"\nSaving checkpoint at step {num_batches} to {checkpoint_dir}")
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(checkpoint_dir, safe_serialization=True)
                    tokenizer.save_pretrained(checkpoint_dir)
                    CustomConfig(**config["model"]).save_pretrained(checkpoint_dir)
            
            if num_batches >= steps_per_epoch:
                break
        
        progress_bar.close()
        
        epoch_time = time.time() - epoch_start_time
        epoch_avg_loss = running_loss / num_batches
        epoch_perplexity = torch.exp(torch.tensor(epoch_avg_loss))
        
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1}/{total_epochs} Summary:")
            print(f"Time: {epoch_time/60:.2f} minutes")
            print(f"Average Loss: {epoch_avg_loss:.4f}")
            print(f"Perplexity: {epoch_perplexity:.2f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"Samples Processed: {samples_processed:,}")
            print(f"Average Speed: {samples_processed/epoch_time:.1f} samples/s")
            
            epochs_remaining = total_epochs - (epoch + 1)
            estimated_remaining_time = epochs_remaining * epoch_time / 60
            print(f"Estimated time for remaining {epochs_remaining} epochs: {estimated_remaining_time:.1f} minutes")
            
            current_step = int((epoch + 1) * steps_per_epoch)
            accelerator.log({
                'epoch/average_loss': epoch_avg_loss,
                'epoch/perplexity': epoch_perplexity.item(),
                'epoch/time': epoch_time,
                'epoch/samples_processed': samples_processed,
            }, step=current_step)
    
    if accelerator.is_local_main_process:
        final_model_dir = os.path.join(output_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        print(f"\nSaving final model to {final_model_dir}")
        
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_model_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_model_dir)
        CustomConfig(**config["model"]).save_pretrained(final_model_dir)
        
        print("Final model saved successfully")
        accelerator.end_training()

if __name__ == "__main__":
    config = load_config()
    train_model(config)
    print("Training complete.")