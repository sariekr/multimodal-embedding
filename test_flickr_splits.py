#!/usr/bin/env python3
"""
Test script to find the official Flickr30k Karpathy split.
Standard split should be: train ~29k, val ~1k, test ~1k

Run this on RunPod to check which dataset has the correct split.
"""

from datasets import load_dataset

print("=" * 80)
print("TESTING FLICKR30K DATASETS FOR KARPATHY SPLIT")
print("=" * 80)
print("\nExpected Karpathy split:")
print("  - train: ~29,000 images")
print("  - val:   ~1,000 images")
print("  - test:  ~1,000 images")
print("  - Each image: exactly 5 captions")
print()

# Test 1: nlphuji/flickr30k
print("-" * 80)
print("TEST 1: nlphuji/flickr30k")
print("-" * 80)
try:
    ds = load_dataset("nlphuji/flickr30k", trust_remote_code=True)

    for split_name in ds.keys():
        split_data = ds[split_name]
        print(f"\n{split_name}: {len(split_data):,} samples")

        # Check caption structure
        if len(split_data) > 0:
            first_item = split_data[0]

            # Print all available keys
            print(f"  Available keys: {list(first_item.keys())}")

            # Check captions
            if 'caption' in first_item:
                captions = first_item['caption']
                if isinstance(captions, list):
                    print(f"  ✓ Captions per image: {len(captions)}")
                    if len(captions) == 5:
                        print(f"  ✓ Correct caption count (5)")
                    else:
                        print(f"  ✗ Wrong caption count (expected 5, got {len(captions)})")
                else:
                    print(f"  ✗ Captions not a list: {type(captions)}")

            # Verify all samples have 5 captions
            if 'caption' in first_item and isinstance(first_item['caption'], list):
                caption_counts = {}
                for i in range(min(100, len(split_data))):
                    n_caps = len(split_data[i]['caption'])
                    caption_counts[n_caps] = caption_counts.get(n_caps, 0) + 1

                print(f"  Caption count distribution (first 100 samples): {caption_counts}")

    # Final verdict
    print("\n" + "=" * 80)
    split_sizes = {k: len(v) for k, v in ds.items()}
    print(f"Split sizes: {split_sizes}")

    # Check if matches Karpathy
    if ('train' in split_sizes and 'val' in split_sizes and 'test' in split_sizes):
        train_ok = 28000 <= split_sizes['train'] <= 30000
        val_ok = 900 <= split_sizes['val'] <= 1100
        test_ok = 900 <= split_sizes['test'] <= 1100

        if train_ok and val_ok and test_ok:
            print("✅ MATCHES KARPATHY SPLIT!")
            print(f"   Train: {split_sizes['train']:,} (expected ~29k)")
            print(f"   Val:   {split_sizes['val']:,} (expected ~1k)")
            print(f"   Test:  {split_sizes['test']:,} (expected ~1k)")
        else:
            print("❌ DOES NOT MATCH KARPATHY SPLIT")
            print(f"   Train: {split_sizes.get('train', 0):,} (expected ~29k)")
            print(f"   Val:   {split_sizes.get('val', 0):,} (expected ~1k)")
            print(f"   Test:  {split_sizes.get('test', 0):,} (expected ~1k)")
    else:
        print("❌ Missing standard splits (train/val/test)")

except Exception as e:
    print(f"❌ Error loading nlphuji/flickr30k: {e}")

print()

# Test 2: lmms-lab/flickr30k
print("-" * 80)
print("TEST 2: lmms-lab/flickr30k")
print("-" * 80)
try:
    ds = load_dataset("lmms-lab/flickr30k", split="test")
    print(f"\ntest split: {len(ds):,} samples")

    if len(ds) > 0:
        first_item = ds[0]
        print(f"  Available keys: {list(first_item.keys())}")

        if 'caption' in first_item:
            captions = first_item['caption']
            if isinstance(captions, list):
                print(f"  Captions per image: {len(captions)}")

    print(f"\n❌ This is NOT the Karpathy split (has {len(ds):,} samples, expected ~1k)")

except Exception as e:
    print(f"❌ Error loading lmms-lab/flickr30k: {e}")

print()
print("=" * 80)
print("RECOMMENDATION:")
print("=" * 80)
print("Use the dataset that matches Karpathy split (train ~29k, val ~1k, test ~1k)")
print("This ensures results are comparable to published benchmarks.")
print("=" * 80)
