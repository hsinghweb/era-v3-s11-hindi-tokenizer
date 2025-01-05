import re
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

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
    
    # Step 1: Load and preprocess the dataset
    input_path = "raw_hindi_dataset.txt"
    preprocessed_path = output_dir / "preprocessed_hindi.txt"
    
    print("Step 1: Preprocessing dataset...")
    try:
        with open(input_path, "r", encoding="utf-8") as file:
            raw_data = file.readlines()
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found!")
        return

    # Preprocess the text
    preprocessed_data = [preprocess_hindi_text(line) for line in raw_data]

    # Save the preprocessed dataset
    with open(preprocessed_path, "w", encoding="utf-8") as file:
        file.write("\n".join(preprocessed_data))

    # Step 2: Train the BPE tokenizer
    print("Step 2: Training BPE tokenizer...")
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

    # Step 3: Save the trained tokenizer
    print("Step 3: Saving tokenizer files...")
    vocab_path = output_dir / "hindi_vocab.bpe"
    config_path = output_dir / "hindi_encoder.json"
    
    tokenizer.model.save(str(output_dir), "hindi_vocab")
    tokenizer.save(str(config_path))

    # Step 4: Evaluate compression ratio
    print("Step 4: Calculating compression ratio...")
    compression_ratio = calculate_compression_ratio(tokenizer, preprocessed_path)
    print(f"Compression Ratio: {compression_ratio:.2f}")

    # Verify compression ratio requirement
    if compression_ratio < 3.2:
        print("Warning: Compression ratio is below the required threshold of 3.2!")
    else:
        print("Success: Compression ratio meets the requirement!")

if __name__ == "__main__":
    main() 