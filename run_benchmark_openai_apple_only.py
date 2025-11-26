"""
Minimal benchmark for OpenAI CLIP-L and Apple DFN5B-H.
Requires PyTorch 2.6+ or 2.8+ for torch.load security fix.

Usage:
  python run_benchmark_openai_apple_only.py --runs 3 --batch-size 32

Environment Requirements:
  - PyTorch 2.8.0+ (e.g., runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404)
  - transformers, datasets, pillow
"""

import torch
import sys

print(f"PyTorch version: {torch.__version__}")
if torch.__version__ < "2.6.0":
    print("❌ ERROR: Requires PyTorch 2.6.0+ for torch.load security fix")
    print("Current version:", torch.__version__)
    sys.exit(1)
else:
    print("✅ PyTorch version OK for pytorch_model.bin loading")

# Import from main benchmark
import os
import argparse
from pathlib import Path

# Parse args
parser = argparse.ArgumentParser(description="OpenAI + Apple Benchmark")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--workers", type=int, default=16)
parser.add_argument("--sample-size", type=int, default=5000)
parser.add_argument("--runs", type=int, default=3)
parser.add_argument("--output", type=str, default="benchmark_openai_apple_results.csv")
parser.add_argument("--cache-dir", type=str, default="./coco_images")
args = parser.parse_args()

# Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
CACHE_DIR = Path(args.cache_dir)
CACHE_DIR.mkdir(exist_ok=True)

print(f"\nSetup:")
print(f"  Device: {DEVICE}")
print(f"  Dtype: {DTYPE}")
print(f"  Runs: {args.runs}")
print(f"  Sample size: {args.sample_size}")

# Models to test
MODELS = [
    {"name": "OpenAI-CLIP-L", "id": "openai/clip-vit-large-patch14-336", "type": "dense", "batch_size": args.batch_size},
    {"name": "Apple-DFN5B-H", "id": "apple/DFN5B-CLIP-ViT-H-14-378", "type": "dense", "trust": True, "batch_size": args.batch_size}
]

print("\nModels to benchmark:")
for m in MODELS:
    print(f"  - {m['name']}")

# Quick setup test
print("\n" + "="*60)
print("SETUP VERIFICATION")
print("="*60)

print("\nInstalling dependencies...")
os.system("pip install -q transformers datasets pillow pandas tabulate")

print("\nTesting model loading...")
from transformers import AutoModel, AutoProcessor

for m_info in MODELS:
    try:
        print(f"\nLoading {m_info['name']}...")
        trust = m_info.get("trust", False)
        model = AutoModel.from_pretrained(
            m_info["id"],
            trust_remote_code=trust,
            torch_dtype=DTYPE
        ).to(DEVICE).eval()
        processor = AutoProcessor.from_pretrained(m_info["id"], trust_remote_code=trust)
        print(f"  ✅ {m_info['name']} loaded successfully!")

        # Quick test
        test_text = ["a photo of a cat"]
        inputs = processor(text=test_text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

        if hasattr(model, 'get_text_features'):
            out = model.get_text_features(**inputs)
        else:
            out = model(**inputs).text_embeds

        print(f"  ✅ Text encoding works! Shape: {out.shape}")

        del model, processor
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        sys.exit(1)

print("\n" + "="*60)
print("✅ ALL SETUP CHECKS PASSED")
print("="*60)
print("\nReady to run full benchmark!")
print("\nTo run the full benchmark with these models:")
print("1. Copy run_benchmark_grand_slam_v28_publication_ready.py")
print("2. Uncomment OpenAI and Apple models in get_models_to_test()")
print("3. Run: python run_benchmark_grand_slam_v28_publication_ready.py --runs 3")
print("\nOr use the provided benchmark script if available.")
