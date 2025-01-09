# Hindi BPE Tokenizer

A Python script for preprocessing Hindi text and training a Byte Pair Encoding (BPE) tokenizer optimized for the Hindi language. The script automatically downloads and processes a portion of the IndicCorp Hindi dataset.

## Features

- **Smart Dataset Management**:
  - Downloads first 5GB of IndicCorp Hindi dataset
  - Supports download resume capability
  - Samples 1 million lines from first 2 million lines
  - Progress bars for download and processing

- **Text Preprocessing**:
  - Retains only Hindi characters (Unicode range: \u0900-\u097F)
  - Removes digits (both English and Devanagari)
  - Normalizes punctuation (converts Hindi full stops '।' to '.')
  - Cleans whitespace
  
- **BPE Tokenizer Training**:
  - Optimized training with numpy vectorized operations
  - Batch processing for better performance
  - Vocabulary size: 4,500 tokens (configurable)
  - Special tokens: `<pad>`, `<unk>`, `<s>`, `</s>`
  - Minimum token frequency: 2
  - Progress tracking with compression ratio

## Requirements

Install required packages:
```
pip install numpy requests tqdm matplotlib
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
└── hindi_encoder.json # Tokenizer config
```

## Dataset

- **Source**: IndicCorp Hindi Collection
- **URL**: https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/v1-indiccorp/hi.txt
- **Download Size**: First 5GB of ~20GB file
- **Training Sample**: 1,000,000 lines from first 2 million lines

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
- Training batch size: 1,000
- Statistics tracking interval: 500
- Vectorized operations using numpy

### Performance Optimizations
- Numpy-based vectorized operations
- Batch processing of merge operations
- Efficient memory management
- Sliding window view for pair counting
- Pre-allocated arrays for speed
- Batched statistics updates

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
Sufficient dataset already exists, skipping download.
Step 2: Preprocessing dataset...
Reading and preparing dataset...
Reading lines: 1000004it [00:01, 883763.77it/s]
Cleaning and normalizing text...
100%|█████████████████████████████████████████████████████████████████| 1000000/1000000 [00:08<00:00, 116119.63it/s] 
Initializing vocabulary...
Computing initial frequencies...
Training BPE:  11%|███████                                                       | 500/4388 [05:38<13:37,  4.76it/s]
Iteration 612
Created token: 'रं' (merged 38,595 times)
Current vocabulary size: 612
Current data size: 133,371,267
Current compression ratio: 1.68
--------------------------------------------------------------------------------
Training BPE:  23%|█████████████▉                                               | 1000/4388 [07:20<11:02,  5.11it/s]
Iteration 1,112
Created token: 'शव' (merged 7,343 times)
Current vocabulary size: 1,112
Current data size: 133,371,267
Current compression ratio: 1.74
--------------------------------------------------------------------------------
Training BPE:  34%|████████████████████▊                                        | 1500/4388 [11:14<07:38,  6.30it/s]
Iteration 1,612
Created token: 'हुत' (merged 22,735 times)
Current vocabulary size: 1,612
Current data size: 133,371,267
Current compression ratio: 2.24
--------------------------------------------------------------------------------
Training BPE:  46%|███████████████████████████▊                                 | 2000/4388 [14:25<17:55,  2.22it/s]
Iteration 2,112
Created token: 'यास' (merged 13,260 times)
Current vocabulary size: 2,112
Current data size: 133,371,267
Current compression ratio: 2.39
--------------------------------------------------------------------------------
Training BPE:  57%|██████████████████████████████████▊                          | 2500/4388 [21:53<11:47,  2.67it/s]
Iteration 2,612
Created token: 'ा भी ' (merged 7,752 times)
Current vocabulary size: 2,612
Current data size: 133,371,267
Current compression ratio: 2.66
--------------------------------------------------------------------------------
Training BPE:  68%|█████████████████████████████████████████▋                   | 3000/4388 [24:51<07:20,  3.15it/s]
Iteration 3,112
Created token: ' हो स' (merged 5,561 times)
Current vocabulary size: 3,112
Current data size: 133,371,267
Current compression ratio: 2.79
--------------------------------------------------------------------------------
Training BPE:  80%|████████████████████████████████████████████████▋            | 3500/4388 [31:43<04:56,  3.00it/s]
Iteration 3,612
Created token: 'ों को' (merged 3,848 times)
Current vocabulary size: 3,612
Current data size: 133,371,267
Current compression ratio: 2.93
--------------------------------------------------------------------------------
Training BPE:  91%|███████████████████████████████████████████████████████▌     | 4000/4388 [34:44<02:11,  2.96it/s]
Iteration 4,112
Created token: '्यूज' (merged 3,115 times)
Current vocabulary size: 4,112
Current data size: 133,371,267
Current compression ratio: 3.03
--------------------------------------------------------------------------------
Training BPE: 100%|█████████████████████████████████████████████████████████████| 4388/4388 [39:35<00:00,  1.85it/s]

Training completed. Final vocabulary size: 4500
Final compression ratio: 3.11

Tokenizer Test:
--------------------------------------------------
Original Text: नमस्ते भारत! यह एक परीक्षण वाक्य है।

Tokens: ['नम', 'स्त', 'े', 'भारत', '!', 'यह', 'एक', 'पर', 'ीक', '्ष', 'ण', 'वा', 'क्', 'य', 'है.']
Token IDs: [619, 1211, 78, 2175, 5, 300, 256, 176, 422, 244, 43, 161, 165, 55, 1177]

Decoded Text: नम स्त े भारत ! यह एक पर ीक ्ष ण वा क् य है.
```

## BPE Tokenizer Sample Usage Logs
```
PS D:\ERA-V3\Github\era-v3-s11-hindi-tokenizer> python use_tokenizer.py  
Loaded vocabulary size: 4500
Max token ID: 4499
Sample tokens: [(0, '<pad>'), (1, '<unk>'), (2, '<s>'), (3, '</s>'), (4, ' ')]
Hindi Text Encoder/Decoder (type 'quit' to exit)
--------------------------------------------------

Enter Hindi text to encode/decode: इसलिए नियुक्ति प्रक्रिया को आगे बढ़ा दिया गया.

Encoding:
Tokens: ['इसलिए', 'निय', 'ुक', '्त', 'ि', 'प्र', 'क्र', 'िया', 'को', 'आग', 'े', 'बढ़', 'ा', 'दिया', 'गया', '.']       
Token IDs: [3084, 1926, 354, 188, 70, 1130, 1508, 1214, 149, 741, 78, 1057, 69, 1898, 1415, 8]

Decoding:
Text: इसलिए निय ुक ्त ि प्र क्र िया को आग े बढ़ ा दिया गया .

Enter Hindi text to encode/decode: quit
PS D:\ERA-V3\Github\era-v3-s11-hindi-tokenizer> 
```

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License
MIT License

