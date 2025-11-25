# 🚀 Grand Slam Benchmark Setup (RunPod / NVIDIA GPU)

This guide is designed for a **RunPod instance** using the **`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`** template. This template already includes PyTorch 2.8.0, CUDA 12.1, and an Ubuntu 24.04 environment.

## 1. System Dependencies (Minimal)
Update the system and install basic utilities and OpenGL libraries. Most core dependencies should already be present in the template.
```bash
apt-get update && apt-get install -y libgl1-mesa-glx git
```
*(Note: `git` might already be installed, but it's safe to include.)*

## 2. Python Environment (Additional Libraries)
The PyTorch template already provides `torch`, `torchvision`, `torchaudio`. We only need to install the remaining libraries.
```bash
pip install transformers datasets pillow timm einops protobuf sentencepiece pandas tabulate
```

Install specialized libraries for SOTA models (ColPali, Flash Attention).
```bash
pip install colpali-engine flash_attn
```

## 3. Clone & Run
Clone your repository and run the Grand Slam benchmark. If you are already in your repo's directory on RunPod, you can skip the `git clone` and `cd` steps.

```bash
# If you haven't cloned yet (or on a fresh instance):
git clone https://github.com/sariekr/multimodal-embedding.git
cd multimodal-embedding

# If you previously cloned, just pull the latest changes:
# git pull origin main

# Run the benchmark
python run_benchmark_grand_slam.py
```

## 4. Troubleshooting
- **Out of Memory (OOM):** If the script crashes, open `run_benchmark_grand_slam.py` and reduce `BATCH_SIZE` (especially for `ColPali`) in the `MODELS` list (e.g., change 64 to 16, or 4 for ColPali).
- **ColPali Error:** Ensure `colpali-engine` is installed.
- **Missing HF Token:** Some models (like LLaVA or gated models) might require `huggingface-cli login`. If you encounter authentication errors, run this command and follow the instructions:
  ```bash
  huggingface-cli login
  ```
