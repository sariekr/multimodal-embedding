#!/usr/bin/env python3
"""
Debug script to test SigLIP-Base embedding behavior.
Tests if embeddings are correctly normalized and similarity computation works.
"""

from transformers import SiglipModel, SiglipProcessor
import torch
from PIL import Image
import requests

print("Loading SigLIP-Base model...")
model = SiglipModel.from_pretrained('google/siglip-base-patch16-224').eval()
processor = SiglipProcessor.from_pretrained('google/siglip-base-patch16-224')

print("\nDownloading test image...")
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
text = 'a photo of a cat'

print(f"\nTest inputs:")
print(f"  Image: {url}")
print(f"  Text: '{text}'")

with torch.no_grad():
    print("\nProcessing inputs...")
    img_inputs = processor(images=[image], return_tensors='pt', padding=True)
    txt_inputs = processor(text=[text], return_tensors='pt', padding=True, truncation=True)

    print("\nExtracting features...")
    img_emb = model.get_image_features(**img_inputs)
    txt_emb = model.get_text_features(**txt_inputs)

    print(f"\nRaw embeddings:")
    print(f"  Image embedding shape: {img_emb.shape}")
    print(f"  Text embedding shape: {txt_emb.shape}")
    print(f"  Image embedding norm: {img_emb.norm(dim=-1).item():.4f}")
    print(f"  Text embedding norm: {txt_emb.norm(dim=-1).item():.4f}")

    # Normalize
    print("\nNormalizing embeddings...")
    img_emb_norm = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_emb_norm = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

    print(f"  Normalized image norm: {img_emb_norm.norm(dim=-1).item():.4f}")
    print(f"  Normalized text norm: {txt_emb_norm.norm(dim=-1).item():.4f}")

    # Compute similarity
    sim = (img_emb_norm @ txt_emb_norm.T).item()
    print(f"\nCosine similarity: {sim:.4f}")

    if sim > 0.2:
        print("✅ Similarity looks reasonable (>0.2)")
    else:
        print("⚠️  WARNING: Similarity is very low (<0.2)")

    # Check embedding statistics
    print(f"\nEmbedding statistics:")
    print(f"  Image emb mean: {img_emb.mean().item():.6f}")
    print(f"  Image emb std: {img_emb.std().item():.6f}")
    print(f"  Text emb mean: {txt_emb.mean().item():.6f}")
    print(f"  Text emb std: {txt_emb.std().item():.6f}")

print("\n" + "="*60)
print("DEBUG COMPLETE")
print("="*60)
