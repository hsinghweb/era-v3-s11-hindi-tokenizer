# Hindi BPE Tokenizer

A Python script for preprocessing Hindi text and training a Byte Pair Encoding (BPE) tokenizer optimized for the Hindi language.

## Features

- **Text Preprocessing**:
  - Retains only Hindi characters and punctuation
  - Removes digits and special symbols
  - Normalizes whitespace and Hindi full stops
  
- **BPE Tokenizer Training**:
  - Vocabulary size < 5000 tokens
  - Compression ratio ≥ 3.2
  - Uses HuggingFace's `tokenizers` library

## Requirements
bash
pip install tokenizers

## Usage

1. Place your Hindi text dataset in the root directory as `raw_hindi_dataset.txt`

2. Run the script:

bash
python hindi_tokenizer.py

## Output Files

The script creates an `output` directory containing:

- `preprocessed_hindi.txt`: Cleaned and normalized Hindi text
- `hindi_vocab.bpe`: BPE vocabulary file
- `hindi_encoder.json`: Tokenizer configuration file

## Compression Ratio

The script automatically calculates and displays the compression ratio, which should be ≥ 3.2. The compression ratio is calculated as:
README.md
compression_ratio = total_characters / total_tokens

## Error Handling

The script includes basic error handling for:
- Missing input file
- Compression ratio verification
- Directory creation

## License

MIT License