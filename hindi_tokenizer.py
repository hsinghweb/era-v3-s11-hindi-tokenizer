import re
import requests
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

def download_dataset(url, filepath):
    """
    Downloads the dataset from the given URL with a progress bar.
    
    Args:
        url (str): URL of the dataset
        filepath (Path): Path where the file should be saved
    """
    print(f"Downloading dataset from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get file size for progress bar
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    with open(filepath, 'wb') as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            progress_bar.update(size)

def prepare_dataset(input_path, sample_size=None):
    """
    Prepares the dataset by optionally sampling and basic cleaning.
    
    Args:
        input_path (Path): Path to the raw dataset
        sample_size (int, optional): Number of lines to sample. If None, use entire dataset
    
    Returns:
        list: Processed lines from the dataset
    """
    print("Reading and preparing dataset...")
    with open(input_path, 'r', encoding='utf-8') as file:
        if sample_size:
            # Read the first sample_size non-empty lines
            lines = []
            for line in file:
                if line.strip():
                    lines.append(line)
                    if len(lines) >= sample_size:
                        break
        else:
            lines = [line for line in file if line.strip()]
    
    return lines

def preprocess_hindi_text(text):
    """
    Preprocesses Hindi text by removing unwanted characters and normalizing punctuation.
    
    Args:
        text (str): Raw Hindi text input
    
    Returns:
        str: Cleaned and normalized text
    """
    # Retain Hindi characters and punctuation
    text = re.sub(r"[^\u0900-\u097F\s।,.!?\-]", "", text)
    # Remove digits (both English and Hindi)
    text = re.sub(r"[0-9०-९]", "", text)
    # Normalize full stops and whitespace
    text = re.sub(r"।", ".", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def calculate_compression_ratio(tokenizer, corpus_path):
    """
    Calculates the compression ratio for the tokenizer on the given corpus.
    
    Args:
        tokenizer (Tokenizer): Trained BPE tokenizer
        corpus_path (str): Path to the preprocessed corpus
    
    Returns:
        float: Compression ratio (characters/tokens)
    """
    with open(corpus_path, "r", encoding="utf-8") as file:
        corpus = file.readlines()
    
    total_chars = sum(len(line) for line in corpus)
    total_tokens = sum(len(tokenizer.encode(line).tokens) for line in corpus)
    return total_chars / total_tokens

def main():
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Dataset URL and paths
    dataset_url = "https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/v1-indiccorp/hi.txt"
    raw_dataset_path = Path("raw_hindi_dataset.txt")
    preprocessed_path = output_dir / "preprocessed_hindi.txt"
    
    # Step 1: Download dataset if it doesn't exist
    if not raw_dataset_path.exists():
        print("Step 1: Downloading dataset...")
        try:
            download_dataset(dataset_url, raw_dataset_path)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading dataset: {e}")
            return
    else:
        print("Dataset already exists, skipping download.")
    
    # Step 2: Prepare and preprocess the dataset
    print("Step 2: Preprocessing dataset...")
    try:
        # Sample 100,000 lines for initial testing
        # Remove or adjust sample_size for full dataset
        raw_data = prepare_dataset(raw_dataset_path, sample_size=100_000)
    except FileNotFoundError:
        print(f"Error: Input file '{raw_dataset_path}' not found!")
        return
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        return

    # Preprocess the text
    print("Cleaning and normalizing text...")
    preprocessed_data = [preprocess_hindi_text(line) for line in tqdm(raw_data)]

    # Save the preprocessed dataset
    with open(preprocessed_path, "w", encoding="utf-8") as file:
        file.write("\n".join(preprocessed_data))

    # Step 3: Train the BPE tokenizer
    print("Step 3: Training BPE tokenizer...")
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()

    # Define the trainer with vocabulary size < 5000
    trainer = BpeTrainer(
        vocab_size=4500,  # Setting slightly below 5000 to ensure we stay under limit
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        min_frequency=2
    )

    # Train the tokenizer
    tokenizer.train([str(preprocessed_path)], trainer)

    # Step 4: Save the trained tokenizer
    print("Step 4: Saving tokenizer files...")
    vocab_path = output_dir / "hindi_vocab.bpe"
    config_path = output_dir / "hindi_encoder.json"
    
    tokenizer.model.save(str(output_dir), "hindi_vocab")
    tokenizer.save(str(config_path))

    # Step 5: Evaluate compression ratio
    print("Step 5: Calculating compression ratio...")
    compression_ratio = calculate_compression_ratio(tokenizer, preprocessed_path)
    print(f"Compression Ratio: {compression_ratio:.2f}")

    # Verify compression ratio requirement
    if compression_ratio < 3.2:
        print("Warning: Compression ratio is below the required threshold of 3.2!")
    else:
        print("Success: Compression ratio meets the requirement!")

if __name__ == "__main__":
    main() 