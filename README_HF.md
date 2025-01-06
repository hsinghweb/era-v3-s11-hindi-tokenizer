---
title: Hindi BPE Tokenizer
emoji: 🇮🇳
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.31.1
app_file: app.py
pinned: false
---

# Hindi BPE Tokenizer

A Streamlit web application for encoding Hindi text to BPE tokens and decoding tokens back to text.

## Features

- Encode Hindi text to BPE tokens and token IDs
- Decode token IDs back to Hindi text
- Pre-trained on 200,000 lines of Hindi text
- Vocabulary size: 4,500 tokens
- Includes special tokens: `<pad>`, `<unk>`, `<s>`, `</s>`

## Usage

1. **Encoding**: Enter Hindi text in the left panel and click "Encode"
2. **Decoding**: Enter comma-separated token IDs in the right panel and click "Decode"

## Technical Details

- BPE (Byte Pair Encoding) tokenizer
- Trained on IndicCorp Hindi dataset
- Compression ratio > 3.2
- Preserves Hindi Unicode range (\\u0900-\\u097F)

## Demo

Try these examples:

**Encoding**:
Input: नमस्ते भारत! यह एक परीक्षण वाक्य है।
Expected Output: [2517, 2074, 340, 4, 201, 164, 3901, 123, 121, 54, 105, 7]

**Decoding**:
Input: 2517, 2074, 340, 4, 201, 164, 3901, 123, 121, 54, 105, 7
Expected Output: नमस्ते भारत! यह एक परीक्षण वाक्य है।