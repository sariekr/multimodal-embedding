# 🚀 Grand Slam Benchmark Setup (RunPod / NVIDIA GPU)

This guide is designed for a fresh **NVIDIA GPU (RTX 3090/4090/A100)** instance running **Ubuntu** (e.g., RunPod, Vast.ai).

## 1. System Dependencies
Update the system and install OpenGL libraries required for image processing.
```bash
apt-get update && apt-get install -y libgl1-mesa-glx git
```

## 2. Python Environment
Install PyTorch with CUDA support (Flash Attention compatible).
```bash
# Install PyTorch 2.1+ with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the heavyweight libraries.
```bash
pip install transformers datasets pillow timm einops protobuf sentencepiece pandas tabulate
```

Install specialized libraries for SOTA models (ColPali, Flash Attention).
```bash
pip install colpali-engine flash_attn
```

## 3. Clone & Run
Clone your repository and run the Grand Slam benchmark.
```bash
git clone https://github.com/sariekr/multimodal-embedding.git
cd multimodal-embedding

# Run the benchmark
python run_benchmark_grand_slam.py
```

## 4. Troubleshooting
- **Out of Memory (OOM):** If the script crashes, open `run_benchmark_grand_slam.py` and reduce `BATCH_SIZE` in the `MODELS` list (e.g., change 64 to 16).
- **ColPali Error:** Ensure `colpali-engine` is installed.
- **Missing HF Token:** Some models (like LLaVA or gated models) might require `huggingface-cli login`, but the ones in this list are generally public or use `trust_remote_code`.
