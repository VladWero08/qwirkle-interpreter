# qwirkle-interpreter

## Overview

The **Qwirkle Score Calculator** is an automated system designed to interpret the state of a *Qwirkle Connect* game and calculate scores based on game board images. *Qwirkle Connect* extends the classic Qwirkle game by introducing a guided board with special scoring tiles and starting positions.

This project processes sequences of board images to extract and interpret tile placements, shapes, colors, and score outcomes for each turn.

## Features

- Automatically extracts and aligns Qwirkle Connect boards from input images
- Detects and identifies tile placements using image processing techniques
- Classifies tile shapes and colors
- Calculates incremental scores based on newly placed tiles
- Recognizes board quadrants and special scoring zones

## Methodology

### 1. **Board Detection and Preprocessing**
- Grayscale conversion, median blur, and adaptive thresholding
- Morphological operations (dilation + erosion) to enhance contours
- Perspective transform to align the board for grid analysis

### 2. **Starting Board Analysis**
- Identifies scoring quadrants by analyzing key tiles
- Determines quadrant configuration (two possible types)

### 3. **Tile Extraction and Comparison**
- Board resized to 1600x1600 px
- Extracts 100x100 px patches based on a 16x16 grid
- Compares current and previous boards to find new tiles

### 4. **Tile Identification**
- **Shapes:**
  - Simple (square, circle, diamond): template matching (NCC)
  - Complex (clove, 4-star, 8-star): SIFT features with KNN and Lowe's ratio test
- **Colors:**
  - Detected using HSV masks and dominant pixel count

### 5. **Score Calculation**
- Based on tile placement and alignment
- Analyzes rows and columns for valid scoring lines
- Accounts for special tiles on the scoring board

# Environment

All the packages and their versions are included in the **requirements.txt** file. The recommended way to run this code is to use a **virtual environment**. After activating the virtual environment and installing all the dependencies, there are two variables that need to be taken into account when running the code (variables inside the main.py file):
- *INPUT_DIR*: where the images will be read from
- *OUTPUT_DIR*: where the results will be stored

To actually run the code, the *main.py* file needs to be executed with the command *python main.py*.
