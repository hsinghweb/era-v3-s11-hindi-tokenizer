# Hindi BPE Tokenizer

A Python script for preprocessing Hindi text and training a Byte Pair Encoding (BPE) tokenizer optimized for the Hindi language. The script automatically downloads and processes a portion of the IndicCorp Hindi dataset.

## Features

- **Smart Dataset Management**:
  - Downloads first 5GB of IndicCorp Hindi dataset
  - Supports download resume capability
  - Samples 5,000,000 lines from first 6 million lines
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
├── raw_hindi_dataset.txt # Downloaded dataset (5GB)
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
- **Download Size**: First 5GB of ~20GB file
- **Training Sample**: 5,000,000 lines from first 6 million lines

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

## BPE Tokenizer Training Logs
```
PS D:\ERA-V3\Github\era-v3-s11-hindi-tokenizer> python hindi_tokenizer.py
Step 1: Downloading dataset (5GB limit)...
Downloading first 5GB from https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/v1-indiccorp/hi.txt
Downloading: 100%|███████████████████████████████████████████████████████████▉| 5.00G/5.00G [11:15<00:00, 5.50MiB/s]
Reached 5GB limit, stopping download.
Downloading: 100%|████████████████████████████████████████████████████████████| 5.00G/5.00G [11:15<00:00, 4.77MiB/s] 
Step 2: Preprocessing dataset...
Reading and preparing dataset...
Reading lines: 5000009it [00:05, 878348.47it/s]
Cleaning and normalizing text...
100%|█████████████████████████████████████████████████████████████████| 5000000/5000000 [00:41<00:00, 119065.75it/s] 
Step 3: Training BPE tokenizer...
[00:00:41] Pre-processing files (1717 Mo) ███████████████████████████████████████████████████████                100%[00:00:01] Tokenize words                 ███████████████████████████████████████████████████████ 1031632  /  1031632
[00:00:02] Count pairs                    ███████████████████████████████████████████████████████ 1031632  /  1031632
[00:00:04] Compute merges                 ███████████████████████████████████████████████████████ 4384     /     4384

Final vocabulary size: 4500 tokens
Special tokens: [AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True), AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True), AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True), AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True)]
Success: Vocabulary size (4500) is within the limit of 5000 tokens

Step 4: Saving tokenizer files...
Step 5: Calculating compression ratio...
Compression Ratio: 3.61
Success: Compression ratio (3.61) exceeds the minimum requirement of 3.2

Tokenizer Test:
--------------------------------------------------
Original Text: नमस्ते भारत! यह एक परीक्षण वाक्य है।

Tokens: ['नम', 'स्ते', 'भारत', '!', 'यह', 'एक', 'परीक्षण', 'वा', 'क्', 'य', 'है', '.']
Token IDs: [2825, 2037, 356, 4, 216, 180, 3852, 139, 137, 55, 121, 7]

Decoded Text: नम स्ते भारत ! यह एक परीक्षण वा क् य है .
PS D:\ERA-V3\Github\era-v3-s11-hindi-tokenizer>
```

## BPE Tokenizer Sample Usage Logs
```
PS D:\ERA-V3\Github\era-v3-s11-hindi-tokenizer> python use_tokenizer.py
Hindi Text Encoder/Decoder (type 'quit' to exit)
--------------------------------------------------

Enter Hindi text to encode/decode: हाउसफुल से अक्षय कुमार के दो लुक सामने आए हैं.

Encoding:
Tokens: ['हाउस', 'फु', 'ल', 'से', 'अक्षय', 'कुमार', 'के', 'दो', 'लुक', 'सामने', 'आए', 'हैं', '.']
Token IDs: [2926, 1549, 58, 128, 3628, 636, 116, 257, 3718, 760, 797, 160, 7]

Decoding:
Text: हाउस फु ल से अक्षय कुमार के दो लुक सामने आए हैं .

Enter Hindi text to encode/decode: बिहार के भोजपुर के सपूत और महान गणितज्ञ डॉ. वशिष्ठ नारायण सिंह का अंतिम संस्कार शु          क्रवार को.

Encoding:
Tokens: ['बिहार', 'के', 'भोज', 'पुर', 'के', 'स', 'पू', 'त', 'और', 'महान', 'गण', 'ित', 'ज्ञ', 'डॉ', '.', 'व', 'शि', 'ष    ष्ठ', 'नारायण', 'सिंह', 'का', 'अंतिम', 'संस्', 'कार', 'शुक्रवार', 'को', '.']
Token IDs: [1363, 116, 4334, 394, 116, 64, 215, 44, 146, 3417, 1322, 255, 592, 663, 7, 61, 220, 923, 3391, 379, 125, 1750, 1341, 154, 1475, 130, 7]

Decoding:
Text: बिहार के भोज पुर के स पू त और महान गण ित ज्ञ डॉ . व शि ष्ठ नारायण सिंह का अंतिम संस् कार शुक्रवार को .

Enter Hindi text to encode/decode: quit
PS D:\ERA-V3\Github\era-v3-s11-hindi-tokenizer> 
```

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License
MIT License

