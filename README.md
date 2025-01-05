# Hindi BPE Tokenizer

A Python script for preprocessing Hindi text and training a Byte Pair Encoding (BPE) tokenizer optimized for the Hindi language. The script automatically downloads and processes a portion of the IndicCorp Hindi dataset.

## Features

- **Smart Dataset Management**:
  - Downloads first 2GB of IndicCorp Hindi dataset
  - Supports download resume capability
  - Samples 200,000 lines from first 1 million lines
  - Progress bars for download and processing

- **Text Preprocessing**:
  - Retains only Hindi characters (Unicode range: \u0900-\u097F)
  - Removes digits (both English and Devanagari)
  - Normalizes punctuation (converts Hindi full stops '।' to '.')
  - Cleans whitespace
  
- **BPE Tokenizer Training**:
  - Vocabulary size: 4,500 tokens (configurable, < 5000)
  - Special tokens: `<pad>`, `<unk>`, `<s>`, `</s>`
  - Minimum token frequency: 2
  - Target compression ratio ≥ 3.2

## Requirements

Install required packages:
```
pip install tokenizers requests tqdm
```

## Quick Start

1. Run the tokenizer training script:
```
python hindi_tokenizer.py
```

2. Use the interactive encoder/decoder:
```
python use_tokenizer.py
```

## Directory Structure
```
.
├── hindi_tokenizer.py # Main training script
├── use_tokenizer.py # Interactive encoding/decoding tool
├── raw_hindi_dataset.txt # Downloaded dataset (2GB)
└── output/
├── preprocessed_hindi.txt # Cleaned text
├── hindi_vocab.bpe # BPE vocabulary
├── hindi_vocab-vocab.json # Vocabulary mapping
├── hindi_vocab-merges.txt # BPE merge rules
└── hindi_encoder.json # Tokenizer config
```

## Dataset

- **Source**: IndicCorp Hindi Collection
- **URL**: https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/v1-indiccorp/hi.txt
- **Download Size**: First 2GB of ~20GB file
- **Training Sample**: 200,000 lines from first 1 million lines

## Usage Examples

### Training the Tokenizer
```
from hindi_tokenizer import main
# Train and get the tokenizer
tokenizer = main()
```


### Using the Trained Tokenizer
```
from hindi_tokenizer import load_tokenizer, encode_text, decode_text
# Load existing tokenizer
tokenizer = load_tokenizer("output/hindi_encoder.json")
# Encode text
text = "नमस्ते भारत!"
token_ids, tokens = encode_text(tokenizer, text)
print(f"Tokens: {tokens}")
print(f"Token IDs: {token_ids}")
Decode back to text
decoded_text = decode_text(tokenizer, token_ids)
print(f"Decoded: {decoded_text}")
```

## Technical Details

### Preprocessing Steps
1. Character filtering: `[^\u0900-\u097F\s।,.!?\-]`
2. Digit removal: `[0-9०-९]`
3. Punctuation normalization: `।` → `.`
4. Whitespace normalization

### Tokenizer Configuration
- Model: Byte Pair Encoding (BPE)
- Vocabulary size: 4,500
- Special tokens: 4
- Pre-tokenizer: Whitespace
- Minimum frequency: 2

### Compression Ratio
Calculated as: `total_characters / total_tokens`
- Target: ≥ 3.2
- Verified after training

## Error Handling

The script includes comprehensive error handling for:
- Network issues during download
- Partial download resume
- File I/O operations
- Dataset processing
- Compression ratio verification

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License
MIT License

