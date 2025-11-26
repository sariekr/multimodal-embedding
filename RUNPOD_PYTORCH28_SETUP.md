# RunPod Setup for OpenAI CLIP & Apple DFN5B (PyTorch 2.8+)

## 🚀 Template Selection

Use one of these PyTorch 2.8+ templates:
- **Recommended:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- **Alternative:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`

## 📦 Initial Setup

```bash
# 1. Clone repo
cd /workspace
git clone https://github.com/sariekr/multimodal-embedding
cd multimodal-embedding

# 2. Install dependencies
pip install transformers datasets pillow timm einops protobuf sentencepiece pandas tabulate

# 3. Verify PyTorch version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# Should be >= 2.6.0
```

## ✅ Quick Test (Verify Models Load)

```bash
python run_benchmark_openai_apple_only.py
```

Expected output:
```
PyTorch version: 2.8.0
✅ PyTorch version OK for pytorch_model.bin loading
...
✅ OpenAI-CLIP-L loaded successfully!
✅ Apple-DFN5B-H loaded successfully!
✅ ALL SETUP CHECKS PASSED
```

## 🎯 Run Full Benchmark (OpenAI + Apple Only)

### Option 1: Use main benchmark with modified model list

1. Edit `run_benchmark_grand_slam_v28_publication_ready.py`:
   - Uncomment OpenAI and Apple in `get_models_to_test()`
   - Comment out other models (or keep them for full run)

2. Run:
```bash
python run_benchmark_grand_slam_v28_publication_ready.py --runs 3 --batch-size 32
```

### Option 2: Quick single-run test

```bash
# Test with 1 run first
python run_benchmark_grand_slam_v28_publication_ready.py --runs 1 --batch-size 32
```

## 📊 Expected Results

Based on COCO 5K standard:
- **OpenAI-CLIP-L**: T2I R@1 ~36-40%
- **Apple-DFN5B-H**: T2I R@1 ~45-50% (similar to LAION-CLIP-H)

## ⏱️ Runtime Estimates

### Single Run (--runs 1):
- OpenAI-CLIP-L: ~5 minutes
- Apple-DFN5B-H: ~8 minutes (larger model)
- **Total:** ~13 minutes

### Multi-Run (--runs 3):
- OpenAI-CLIP-L: ~15 minutes
- Apple-DFN5B-H: ~24 minutes
- **Total:** ~40 minutes

## 🔄 Merge with Main Results

After benchmarking, merge results:

```bash
# Copy results from PyTorch 2.8 pod
scp root@RUNPOD_IP:/workspace/multimodal-embedding/benchmark_v28_results.csv ./openai_apple_results.csv

# Manually merge into main results file
```

## 🐛 Troubleshooting

### If torch.load still fails:
```python
# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# Should be >= 2.6.0
```

### If models don't load:
```python
# Test individual model
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('openai/clip-vit-large-patch14-336')
print('✅ OpenAI CLIP loaded')
"
```

## 📝 Notes

- PyTorch 2.8+ required due to CVE-2025-32434 security fix
- Only OpenAI and Apple use `pytorch_model.bin` format
- Other models (ColPali, SigLIP, LAION, Jina, MetaCLIP) use safetensors ✅
- Separate RunPod instance recommended to avoid breaking existing setup
