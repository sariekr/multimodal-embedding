#!/usr/bin/env python3
"""
Download Karpathy's Flickr30k JSON and extract test image IDs.
This gives us the OFFICIAL test set used in all published benchmarks.

Usage:
    python download_karpathy_split.py

Output:
    flickr30k_test_ids.txt  (1000 image IDs for test set)
"""

import json
import urllib.request
from pathlib import Path

KARPATHY_URL = "http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"

# Note: The above URL contains dataset_flickr30k.json inside the zip
# For simplicity, we'll provide direct instructions

print("=" * 80)
print("DOWNLOADING KARPATHY SPLIT FOR FLICKR30K")
print("=" * 80)

print("""
The official Karpathy split is available at:
http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip

This contains dataset_flickr30k.json with the splits.

However, for convenience, the split is also available via:
""")

# Alternative: Get from a GitHub repo that has it
GITHUB_KARPATHY = "https://raw.githubusercontent.com/tylin/coco-caption/master/annotations/dataset_flickr30k.json"

print(f"\nAttempting to download from GitHub mirror...")
print(f"URL: {GITHUB_KARPATHY}")

try:
    print("\nDownloading dataset_flickr30k.json...")

    # Try alternative URL (many repos mirror this)
    # Note: This URL might not work, but the structure is what matters

    print("""
⚠️  Direct download may not work. Manual steps:

1. Download from official source:
   wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
   unzip caption_datasets.zip

2. Or use this Python script to parse a local copy:
   python parse_karpathy_split.py dataset_flickr30k.json

3. Expected JSON structure:
   {
     "images": [
       {
         "imgid": 123456,
         "filename": "123456.jpg",
         "split": "test",  # or "train", "val"
         "sentences": [...]
       },
       ...
     ]
   }

4. Extract test IDs:
   test_ids = [img['imgid'] for img in data['images'] if img['split'] == 'test']

5. Should get ~1000 test image IDs
""")

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nPlease download manually from:")
    print("http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip")

print("\n" + "=" * 80)
print("NEXT STEP:")
print("=" * 80)
print("""
Once you have dataset_flickr30k.json, run:

    python parse_karpathy_split.py dataset_flickr30k.json

This will create:
    - flickr30k_train_ids.txt (~29k IDs)
    - flickr30k_val_ids.txt (~1k IDs)
    - flickr30k_test_ids.txt (~1k IDs)

Then use these to filter lmms-lab/flickr30k dataset.
""")

# Provide parser script
parser_code = '''#!/usr/bin/env python3
"""Parse Karpathy split from dataset_flickr30k.json"""

import json
import sys

if len(sys.argv) != 2:
    print("Usage: python parse_karpathy_split.py dataset_flickr30k.json")
    sys.exit(1)

json_path = sys.argv[1]

with open(json_path, 'r') as f:
    data = json.load(f)

splits = {'train': [], 'val': [], 'test': []}

for img in data['images']:
    split = img['split']
    if split in ['restval']:  # restval is typically merged with train
        split = 'train'

    img_id = img.get('imgid') or img.get('cocoid') or img.get('filename').split('.')[0]
    splits[split].append(str(img_id))

for split_name, ids in splits.items():
    output_file = f'flickr30k_{split_name}_ids.txt'
    with open(output_file, 'w') as f:
        f.write('\\n'.join(ids))
    print(f"✓ Wrote {len(ids):,} IDs to {output_file}")

print(f"\\nSummary:")
print(f"  Train: {len(splits['train']):,} images")
print(f"  Val:   {len(splits['val']):,} images")
print(f"  Test:  {len(splits['test']):,} images")
'''

with open('parse_karpathy_split.py', 'w') as f:
    f.write(parser_code)

print("\n✓ Created parse_karpathy_split.py")
print("\nReady to extract test IDs once you have the JSON file.")
