from datasets import load_dataset
from tqdm import tqdm
import os
from utils import load_config, setup_logging
import psutil  # For monitoring memory usage


def download_and_process_data(config):
    """Downloads, preprocesses, and saves the dataset."""
    setup_logging()

    dataset_name = config["dataset"]["name"]
    streaming = config["dataset"]["streaming"]
    text_column = config["dataset"]["text_column"]
    target_size_gb = config["dataset"]["target_size_gb"]
    max_length = config["dataset"]["max_length"]
    subset = config["dataset"]["subset"]



    # Download dataset (streaming is essential for large datasets)
    try:
        dataset = load_dataset(dataset_name, subset,streaming=streaming)
        if not streaming:
            raise ValueError("Streaming must be True for large datasets like fineweb")
    except Exception as e:
        raise Exception(f"Failed to download dataset: {e}. Check dataset name and internet connection, and HF login.")

    # Filter data - removing the subset filter since it's specific to CC-MAIN
    dataset = dataset["train"]  # Taking only train split

    # Add basic quality filters
    def quality_filter(example):
        return (
            example['text'] is not None and 
            len(example['text'].strip()) > 0 and
            example['language'] == 'en' and  # Filter for English content
            example['language_score'] >= 0.8  # High confidence in language detection
        )

    dataset = dataset.filter(quality_filter)

    # Create output directory if it doesn't exist
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    # Process and save in chunks, monitoring data size
    def process_and_save_chunk(chunk, chunk_num, total_bytes):
        output_file = os.path.join(output_dir, f"processed_data_{chunk_num}.txt")
        
        with open(output_file, "w", encoding="utf-8") as f:
            for example in tqdm(chunk, desc=f"Processing chunk {chunk_num}"):
                text = example[text_column].strip()
                if text:
                    # Add metadata as a comment before each text
                    metadata = f"# ID: {example['id']} | URL: {example['url']} | Date: {example['date']}\n"
                    f.write(metadata)
                    f.write(text + "\n\n")  # Add extra newline for separation
                    total_bytes += len(text.encode("utf-8")) + len(metadata.encode("utf-8"))
        return total_bytes

    chunk_num = 0
    chunk = []
    total_bytes_processed = 0
    target_bytes = target_size_gb * (1024**3)  # Convert GB to bytes

    for example in tqdm(dataset, desc="Processing and saving data"):
        chunk.append(example)
        if len(chunk) >= 10000:  # Adjust chunk size as needed
            total_bytes_processed = process_and_save_chunk(chunk, chunk_num, total_bytes_processed)
            chunk = []
            chunk_num += 1
            print(f"Processed: {total_bytes_processed / (1024**3):.2f} GB")

        if total_bytes_processed >= target_bytes:
            print("Target data size reached.")
            break  # Stop processing

    if chunk:
        process_and_save_chunk(chunk, chunk_num,total_bytes_processed) #for remaining data

    print(f"Data download and processing complete. Total processed size: {total_bytes_processed / (1024**3):.2f} GB")

if __name__ == "__main__":
    config = load_config()
    download_and_process_data(config)