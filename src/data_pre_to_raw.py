import os
from tqdm import tqdm
from pathlib import Path

def convert_to_raw_text():
    # Setup paths
    processed_dir = Path("data/processed")
    raw_dir = Path("data/raw")
    
    # Create raw directory if it doesn't exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file for combined raw text
    output_file = raw_dir / "combined_raw_text.txt"
    
    # Process all txt files in processed directory
    processed_files = list(processed_dir.glob("*.txt"))
    
    print(f"Found {len(processed_files)} files to process")
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for proc_file in tqdm(processed_files, desc="Converting files"):
            with open(proc_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    # Skip metadata lines (starting with #)
                    if not line.startswith('#'):
                        # Only write non-empty lines
                        line = line.strip()
                        if line:
                            outfile.write(line + '\n')

if __name__ == "__main__":
    try:
        convert_to_raw_text()
        print("Successfully converted processed data to raw text")
    except Exception as e:
        print(f"Error during conversion: {e}")
