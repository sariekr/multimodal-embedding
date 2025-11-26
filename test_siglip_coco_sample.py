#!/usr/bin/env python3
"""
Test SigLIP models on actual COCO benchmark image+caption pair.
This tests if the issue is with the model or the benchmark implementation.
"""

from transformers import SiglipModel, SiglipProcessor
import torch
from PIL import Image
import requests

# COCO test sample
image_url = 'http://images.cocodataset.org/val2014/COCO_val2014_000000391895.jpg'
captions = [
    'A man with a red helmet on a small moped on a dirt road.',
    'Man riding a motor bike on a dirt road on the countryside.'
]

print("="*60)
print("TESTING SIGLIP MODELS ON COCO SAMPLE")
print("="*60)
print(f"\nImage: {image_url}")
print(f"Caption 1: {captions[0]}")
print(f"Caption 2: {captions[1]}")

# Load image
image = Image.open(requests.get(image_url, stream=True).raw)

# Test both models
models_to_test = [
    ('SigLIP-Base', 'google/siglip-base-patch16-224'),
    ('SigLIP-400M', 'google/siglip-so400m-patch14-384')
]

for model_name, model_id in models_to_test:
    print("\n" + "="*60)
    print(f"TESTING: {model_name}")
    print("="*60)

    model = SiglipModel.from_pretrained(model_id).eval()
    processor = SiglipProcessor.from_pretrained(model_id)

    with torch.no_grad():
        # Encode image
        img_inputs = processor(images=[image], return_tensors='pt', padding=True)
        img_emb = model.get_image_features(**img_inputs)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        # Encode both captions
        for i, caption in enumerate(captions, 1):
            txt_inputs = processor(text=[caption], return_tensors='pt', padding=True, truncation=True)
            txt_emb = model.get_text_features(**txt_inputs)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

            # Compute similarity
            sim = (img_emb @ txt_emb.T).item()
            print(f"  Caption {i} similarity: {sim:.4f}")

        # Test with a negative example
        neg_caption = "A cat sleeping on a couch"
        txt_inputs = processor(text=[neg_caption], return_tensors='pt', padding=True, truncation=True)
        txt_emb = model.get_text_features(**txt_inputs)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        neg_sim = (img_emb @ txt_emb.T).item()
        print(f"  Negative (cat) similarity: {neg_sim:.4f}")

        # Check if model can distinguish
        pos_sim = (img_emb @ txt_emb.T).item()  # Use caption 1
        txt_inputs = processor(text=[captions[0]], return_tensors='pt', padding=True, truncation=True)
        txt_emb = model.get_text_features(**txt_inputs)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        pos_sim = (img_emb @ txt_emb.T).item()

        margin = pos_sim - neg_sim
        print(f"  Margin (positive - negative): {margin:.4f}")

        if margin > 0:
            print(f"  ✅ Model correctly ranks positive > negative")
        else:
            print(f"  ❌ Model FAILS to distinguish (negative ranked higher!)")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
