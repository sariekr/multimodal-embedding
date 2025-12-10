#!/usr/bin/env python3
"""
Find and download official Karpathy split files for Flickr30k.
Then filter lmms-lab/flickr30k to get the correct 1000 test images.
"""

import requests
from pathlib import Path

# Official Karpathy split files (from cs.stanford.edu or GitHub repos)
KARPATHY_URLS = {
    # These are the standard split files used in image-text retrieval benchmarks
    "dataset_flickr30k.json": "http://cs.stanford.edu/people/karpathy/deepimagesent/flickr30k.zip",

    # Alternative: GitHub repos that have the split
    # Many CLIP/vision-language repos include these splits
    "github_backup": "https://raw.githubusercontent.com/openai/CLIP/main/data/flickr30k_test.txt",
}

print("=" * 80)
print("SEARCHING FOR KARPATHY SPLIT FILES")
print("=" * 80)

# Try downloading from GitHub (simpler)
print("\nAttempting to download Flickr30k test split from CLIP repo...")
print("URL: https://github.com/openai/CLIP (checking for split files)")

# Alternative: Use the CLIP benchmark library
print("\n" + "=" * 80)
print("ALTERNATIVE APPROACH: Use existing vision-language benchmark repos")
print("=" * 80)
print("""
Many repos have the correct Flickr30k Karpathy split:

1. CLIP-benchmark:
   - pip install clip-benchmark
   - Uses correct Karpathy split
   - But may not expose raw dataset

2. CLIP original repo:
   - https://github.com/openai/CLIP
   - Check /data directory for split files

3. Manual approach:
   - Download dataset_flickr30k.json from Karpathy's website
   - Parse JSON to extract test image IDs
   - Filter lmms-lab/flickr30k using these IDs

4. Use pre-filtered WebDataset:
   - clip-benchmark/wds_flickr30k on HuggingFace
   - Already in correct format
""")

print("\n" + "=" * 80)
print("RECOMMENDED SOLUTION:")
print("=" * 80)
print("""
Best approach: Download Karpathy's JSON and filter lmms-lab dataset

Steps:
1. Download: http://cs.stanford.edu/people/karpathy/deepimagesent/dataset_flickr30k.json
2. Extract test image IDs (should be ~1000 IDs)
3. Filter lmms-lab/flickr30k where img_id in test_ids
4. Verify: 1000 images, 5 captions each

This gives us the OFFICIAL test set used in all published benchmarks.
""")

# Let's try to find the actual split
print("\n" + "=" * 80)
print("CHECKING ALTERNATIVE DATASETS...")
print("=" * 80)

alt_datasets = [
    "clip-benchmark/wds_flickr30k",
    "laion/flickr30k",
    "nlphuji/flickr30k_images",
]

print("\nDatasets to check on HuggingFace:")
for ds in alt_datasets:
    print(f"  - {ds}")

print("\nRun this on RunPod to check:")
print("""
from datasets import load_dataset

# Check if any of these have correct split
for ds_name in ['clip-benchmark/wds_flickr30k', 'laion/flickr30k']:
    try:
        print(f"\\nTrying {ds_name}...")
        ds = load_dataset(ds_name)
        for split in ds.keys():
            print(f"  {split}: {len(ds[split])} samples")
    except Exception as e:
        print(f"  Error: {e}")
""")
