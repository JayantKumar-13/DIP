# LAB 3: Huffman Coding & Shannon-Fano Coding

This folder contains Python scripts for two fundamental data compression techniques in digital image processing:

- *Huffman Coding*
- *Shannon-Fano Coding*

## File Structure

```
LAB_3/
    jayant_huffman.py      # Script for Huffman coding implementation
    jayant_shannon.py      # Script for Shannon-Fano coding implementation
    Output Images/
        Huffman_op.png    # Sample output visualization for Huffman coding
        Shannon_op.png    # Sample output visualization for Shannon-Fano coding
```

## Requirements
- Python 3.x
- No external libraries required (uses only built-in Python modules)

## Overview

### Huffman Coding
Huffman coding is a lossless data compression algorithm that assigns variable-length codes to input characters based on their frequencies. The most frequent characters get the shortest codes, resulting in optimal prefix codes.

**Key Features:**
- Builds a binary tree based on character frequencies
- Creates prefix codes (no code is a prefix of another)
- Achieves optimal compression for given symbol probabilities
- Calculates compression efficiency metrics

### Shannon-Fano Coding
Shannon-Fano coding is another lossless compression technique that divides symbols into two sets based on their probabilities and recursively assigns codes.

**Key Features:**
- Divides symbols into two probability-balanced groups
- Assigns '0' to one group and '1' to another
- Creates prefix codes similar to Huffman coding
- Provides compression ratio analysis

## Usage

### Huffman Coding
Run the script to perform Huffman coding on any text message:
```bash
python jayant_huffman.py
```

**Process:**
1. Input your text message when prompted
2. The script counts character frequencies
3. Builds Huffman tree based on frequencies
4. Generates optimal prefix codes
5. Encodes the message using generated codes
6. Displays character codes and encoded message
7. Calculates compression efficiency metrics

**Example:**
```
Enter your message: Jayant
Letter Codes:
J=>110
a=>10
y=>111
a=>10
n=>00
t=>01
```

### Shannon-Fano Coding
Run the script to perform Shannon-Fano coding on any text message:
```bash
python jayant_shannon.py
```

**Process:**
1. Input your text message when prompted
2. The script counts character frequencies
3. Sorts characters by frequency (descending)
4. Recursively divides into two groups
5. Assigns '0' to left group and '1' to right group
6. Generates prefix codes
7. Encodes the message using generated codes
8. Displays character codes and encoded message
9. Calculates compression efficiency metrics

**Example:**
```
Enter your message: jayant
Character Codes:
'J': 01
'a': 00
'y': 10
'a': 00
'n': 110
't': 111
```

## Compression Metrics

Both scripts provide detailed compression analysis:
- **Original Size**: Total bits in original message (8 bits per character)
- **Compressed Size**: Total bits in encoded message
- **Compression Ratio**: Ratio of original size to compressed size

## Output
The scripts display:
- Individual character codes
- Original message
- Encoded message
- Compression efficiency statistics

## Technical Details

### Huffman Algorithm Steps:
1. Count character frequencies
2. Create leaf nodes for each character
3. Build binary tree by combining lowest frequency nodes
4. Assign codes by traversing tree (left=0, right=1)
5. Encode message using generated codes

### Shannon-Fano Algorithm Steps:
1. Count character frequencies
2. Sort characters by frequency (descending)
3. Recursively divide into two probability-balanced groups
4. Assign '0' to first group, '1' to second group
5. Repeat until all symbols have unique codes
6. Encode message using generated codes

## Comparison
Both techniques achieve lossless compression but differ in:
- **Tree Construction**: Huffman builds optimal tree, Shannon-Fano uses recursive division
- **Code Length**: Huffman generally achieves better compression ratios
- **Complexity**: Huffman requires more complex tree construction

## Author
Jayant Kumar

---

