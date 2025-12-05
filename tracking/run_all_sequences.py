#!/usr/bin/env python3
"""
Script to run tracking on all sequences and save output videos.
Usage: python tracking/run_all_sequences.py
"""

import subprocess
import sys
from pathlib import Path

# Base directory for sequences
BASE_DIR = Path("34759_final_project_rect")
OUTPUT_DIR = Path("output_videos")

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# Sequences to process
SEQUENCES = ["seq_01"]##, "seq_02", "seq_03"]

def run_sequence(seq_name: str) -> bool:
    """Run tracking on a single sequence."""
    seq_dir = BASE_DIR / seq_name
    output_path = OUTPUT_DIR / f"{seq_name}_tracking.mp4"
    
    if not seq_dir.exists():
        print(f"Warning: Sequence directory {seq_dir} does not exist. Skipping...")
        return False
    
    print(f"\n{'='*60}")
    print(f"Processing {seq_name}...")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable,
        "tracking/runner.py",
        "--seq-dir", str(seq_dir),
        "--save", str(output_path),
        "--wait", "1",  # 30ms delay to show video while saving (adjust as needed)
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ Successfully processed {seq_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error processing {seq_name}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n✗ Interrupted while processing {seq_name}")
        return False

def main():
    """Run tracking on all sequences."""
    print("Starting batch processing of all sequences...")
    print(f"Output directory: {OUTPUT_DIR.absolute()}\n")
    
    results = {}
    for seq_name in SEQUENCES:
        results[seq_name] = run_sequence(seq_name)
    
    # Summary
    print(f"\n{'='*60}")
    print("Processing Summary:")
    print(f"{'='*60}")
    for seq_name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {seq_name}: {status}")
    
    successful = sum(results.values())
    total = len(results)
    print(f"\nCompleted: {successful}/{total} sequences")
    
    if successful == total:
        print(f"\nAll videos saved to: {OUTPUT_DIR.absolute()}/")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())

