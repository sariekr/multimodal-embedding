# 🚀 Grand Slam Benchmark Setup (RunPod / NVIDIA GPU)

This guide is designed for a **RunPod instance** using the **`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`** template. This template already includes PyTorch 2.8.0, CUDA 12.1, and an Ubuntu 24.04 environment.

## 1. System Dependencies (Minimal)
Update the system and install basic utilities and OpenGL libraries. Most core dependencies should already be present in the template.
```bash
apt-get update && apt-get install -y libgl1-mesa-glx git
```
*(Note: `git` might already be installed, but it's safe to include.)*

## 2. Python Environment Setup (Recommended: Virtual Environment)
It's highly recommended to use a virtual environment to manage dependencies.

### 2.1. Clone Repository (if not already done)
```bash
# If you haven't cloned yet (or on a fresh instance):
git clone https://github.com/sariekr/multimodal-embedding.git
cd multimodal-embedding
```
*(If you previously cloned, just pull the latest changes: `git pull origin main`)*

### 2.2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
*(You will see `(.venv)` prefix in your terminal prompt, indicating the virtual environment is active.)*

### 2.3. Install Python Libraries
The PyTorch template already provides `torch`, `torchvision`, `torchaudio`. We only need to install the remaining libraries within our virtual environment.
```bash
pip install transformers datasets pillow timm einops protobuf sentencepiece pandas tabulate
pip install colpali-engine flash_attn
```

## 3. Run the Benchmark
Once the virtual environment is active and all libraries are installed:
```bash
python run_benchmark_grand_slam.py
```

## 4. Troubleshooting
- **Permission Denied (publickey) for `git clone`:** If you see `git@github.com: Permission denied (publickey)`, it means your SSH key is not set up on RunPod. Use HTTPS for cloning instead:
  ```bash
  git clone https://github.com/sariekr/multimodal-embedding.git
  ```
- **Out of Memory (OOM):** If the script crashes, open `run_benchmark_grand_slam.py` and reduce `BATCH_SIZE` (especially for `ColPali`) in the `MODELS` list (e.g., change 64 to 16, or 4 for ColPali).
- **ColPali Error:** Ensure `colpali-engine` is installed.
- **Missing HF Token:** Some models (like LLaVA or gated models) might require `huggingface-cli login`. If you encounter authentication errors, run this command and follow the instructions:
  ```bash
  huggingface-cli login
  ```
