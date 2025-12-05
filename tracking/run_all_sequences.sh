#!/bin/bash
# Script to run tracking on all sequences and save output videos

# Base directory for sequences
BASE_DIR="34759_final_project_rect"
OUTPUT_DIR="output_videos"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run on sequence 1
echo "Processing sequence 1..."
python tracking/runner.py \
    --seq-dir "$BASE_DIR/seq_01" \
    --save "$OUTPUT_DIR/seq_01_tracking.mp4" \
    --wait 1

# Run on sequence 2
echo "Processing sequence 2..."
python tracking/runner.py \
    --seq-dir "$BASE_DIR/seq_02" \
    --save "$OUTPUT_DIR/seq_02_tracking.mp4" \
    --wait 1

# Run on sequence 3
echo "Processing sequence 3..."
python tracking/runner.py \
    --seq-dir "$BASE_DIR/seq_03" \
    --save "$OUTPUT_DIR/seq_03_tracking.mp4" \
    --wait 1

echo "All sequences processed! Videos saved in $OUTPUT_DIR/"

