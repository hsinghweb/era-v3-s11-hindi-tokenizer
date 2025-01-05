import re
import requests
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

def download_dataset(url, filepath, max_size_gb=2):
    """
    Downloads a portion of the dataset with size limit and resume capability.
    
    Args:
        url (str): URL of the dataset
        filepath (Path): Path where the file should be saved
        max_size_gb (float): Maximum size to download in gigabytes
    """
    max_size_bytes = max_size_gb * 1024 * 1024 * 1024  # Convert GB to bytes
    
    # Check if we already have enough data
    if filepath.exists() and filepath.stat().st_size >= max_size_bytes:
        print(f"Already have {max_size_gb}GB of data, skipping download.")
        return
    
    print(f"Downloading first {max_size_gb}GB from {url}")
    
    # Get the current size if file exists (for resume)
    current_size = filepath.stat().st_size if filepath.exists() else 0
    
    # Set up headers for resume
    headers = {'Range': f'bytes={current_size}-'} if current_size > 0 else {}
    
    try:
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = min(
            int(response.headers.get('content-length', 0)) + current_size,
            max_size_bytes
        )
        
        mode = 'ab' if current_size > 0 else 'wb'
        with open(filepath, mode) as file, tqdm(
            desc="Downloading",
            initial=current_size,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=8192):
                if not data:
                    break
                    
                size = file.write(data)
                progress_bar.update(size)
                
                # Check if we've reached the size limit
                if file.tell() >= max_size_bytes:
                    print(f"\nReached {max_size_gb}GB limit, stopping download.")
                    break
                    
    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
        if filepath.exists():
            print("Partial download remains available for resume.")
        raise

def prepare_dataset(input_path, sample_size=None, max_lines=None):
    """
    Prepares the dataset by optionally sampling and basic cleaning.
    
    Args:
        input_path (Path): Path to the raw dataset
        sample_size (int, optional): Number of lines to sample. If None, use entire dataset
        max_lines (int, optional): Maximum number of lines to read from file
    
    Returns:
        list: Processed lines from the dataset
    """
    print("Reading and preparing dataset...")
    lines = []
    
    with open(input_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(tqdm(file, desc="Reading lines")):
            if max_lines and i >= max_lines:
                break
                
            if line.strip():
                lines.append(line)
                if sample_size and len(lines) >= sample_size:
                    break
    
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

def encode_text(tokenizer, text):
    """
    Encodes Hindi text into token IDs.
    
    Args:
        tokenizer (Tokenizer): Trained BPE tokenizer
        text (str): Hindi text to encode
    
    Returns:
        tuple: (token_ids, tokens)
    """
    # Preprocess the text first
    cleaned_text = preprocess_hindi_text(text)
    
    # Encode the text
    encoding = tokenizer.encode(cleaned_text)
    return encoding.ids, encoding.tokens

def decode_text(tokenizer, token_ids):
    """
    Decodes token IDs back into Hindi text.
    
    Args:
        tokenizer (Tokenizer): Trained BPE tokenizer
        token_ids (list): List of token IDs to decode
    
    Returns:
        str: Decoded Hindi text
    """
    return tokenizer.decode(token_ids)

def test_tokenizer(tokenizer, test_text):
    """
    Tests the tokenizer by encoding and decoding sample text.
    
    Args:
        tokenizer (Tokenizer): Trained BPE tokenizer
        test_text (str): Sample text for testing
    """
    print("\nTokenizer Test:")
    print("-" * 50)
    print(f"Original Text: {test_text}")
    
    # Encode
    token_ids, tokens = encode_text(tokenizer, test_text)
    print(f"\nTokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    
    # Decode
    decoded_text = decode_text(tokenizer, token_ids)
    print(f"\nDecoded Text: {decoded_text}")

def main():
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Dataset URL and paths
    dataset_url = "https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/v1-indiccorp/hi.txt"
    raw_dataset_path = Path("raw_hindi_dataset.txt")
    preprocessed_path = output_dir / "preprocessed_hindi.txt"
    
    # Step 1: Download dataset if it doesn't exist or is too small
    if not raw_dataset_path.exists() or raw_dataset_path.stat().st_size < (2 * 1024 * 1024 * 1024):
        print("Step 1: Downloading dataset (2GB limit)...")
        try:
            download_dataset(dataset_url, raw_dataset_path, max_size_gb=2)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading dataset: {e}")
            if not raw_dataset_path.exists():
                return
            print("Continuing with existing partial download...")
    else:
        print("Sufficient dataset already exists, skipping download.")
    
    # Step 2: Prepare and preprocess the dataset
    print("Step 2: Preprocessing dataset...")
    try:
        # Sample 100,000 lines from the first 1 million lines
        raw_data = prepare_dataset(
            raw_dataset_path,
            sample_size=100_000,
            max_lines=1_000_000
        )
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

    # Step 6: Test the tokenizer
    test_text = "नमस्ते भारत! यह एक परीक्षण वाक्य है।"
    test_tokenizer(tokenizer, test_text)

    # Return the tokenizer for potential reuse
    return tokenizer

def load_tokenizer(config_path):
    """
    Loads a previously trained tokenizer from a configuration file.
    
    Args:
        config_path (str): Path to the tokenizer configuration file
    
    Returns:
        Tokenizer: Loaded tokenizer
    """
    return Tokenizer.from_file(config_path)

if __name__ == "__main__":
    main() 