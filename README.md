# Hindi BPE Tokenizer

A Python script for preprocessing Hindi text and training a Byte Pair Encoding (BPE) tokenizer optimized for the Hindi language. The script automatically downloads the Hindi dataset from the IndicCorp collection.

## Features

- **Automatic Dataset Download**:
  - Downloads Hindi corpus from IndicCorp
  - Shows download progress with progress bar
  - Supports dataset sampling for quick testing

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
bash
pip install tokenizers requests tqdm

## Quick Start

1. Run the script:
bash
python hindi_tokenizer.py

The script will automatically:
1. Download the Hindi dataset (~1.5GB)
2. Preprocess the text
3. Train the BPE tokenizer
4. Save all outputs
5. Verify the compression ratio

## Directory Structure
README.md
.
├── hindi_tokenizer.py # Main script
├── raw_hindi_dataset.txt # Downloaded dataset
└── output/
├── preprocessed_hindi.txt # Cleaned text
├── hindi_vocab.bpe # BPE vocabulary
└── hindi_encoder.json # Tokenizer config

## Dataset

- **Source**: IndicCorp Hindi Collection
- **URL**: https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/v1-indiccorp/hi.txt
- **Default Sample Size**: 100,000 lines (configurable)

To use the full dataset, modify in `hindi_tokenizer.py`:
python
raw_data = prepare_dataset(raw_dataset_path, sample_size=None)

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

## Progress Monitoring

The script provides progress bars for:
- Dataset download
- Text preprocessing
- Tokenizer training

## Error Handling

Comprehensive error handling for:
- Network issues during download
- File I/O operations
- Dataset processing
- Compression ratio verification

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

MIT License

