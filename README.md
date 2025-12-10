# Multimodal Embedding Benchmark (V29 Statistical)

A rigorously statistical benchmark for evaluating multimodal embedding models (Vision-Language Models) on the **MS-COCO Karpathy split**. This project goes beyond simple accuracy metrics by implementing bootstrap sampling, permutation tests, and detailed failure analysis to provide a deep understanding of model performance.

## 🎯 Key Features

*   **Statistical Rigor**: Implements **1000-iteration bootstrap sampling** to provide true 95% confidence intervals for all metrics. No more ambiguous "±0.0" results.
*   **Significance Testing**: Uses permutation tests to determine if performance differences between models are statistically significant (p < 0.05).
*   **Symmetric Protocols**: Evaluates both standard (1-to-5) and symmetric (1-to-1) Image-to-Text retrieval protocols to expose biases in multi-caption evaluation.
*   **Failure Analysis**: Automatically categorizes queries by complexity (spatial, color, counting) and object category (person, vehicle, animal, etc.) to pinpoint exactly *where* models fail.
*   **Robust Evaluation**: Handles connection instability and image corruption gracefully with a robust caching and downloading mechanism.
*   **7 State-of-the-Art Models**: Benchmarks ColPali, SigLIP, LAION-CLIP, Jina-CLIP, MetaCLIP, OpenAI-CLIP, and Apple-DFN5B.

## 🚀 Getting Started

### Prerequisites

*   Python 3.10+
*   CUDA-capable GPU (Recommended: 24GB+ VRAM for dense models, 48GB+ for ColPali)
*   Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### System Dependencies

For Ubuntu/Debian-based systems:
```bash
apt-get update && apt-get install -y libgl1-mesa-glx
```

### Usage

#### Run the Benchmark

The main entry point is `main.py`. This runs the MS-COCO statistical benchmark with bootstrap sampling and failure analysis.

**Basic usage (all models, 1000 bootstrap iterations):**
```bash
python main.py --bootstrap-iterations 1000 --batch-size 32
```

**Quick test (fewer iterations):**
```bash
python main.py --bootstrap-iterations 100 --batch-size 32
```

**Test specific models only:**
```bash
python main.py --models SigLIP-400M,LAION-CLIP-H --bootstrap-iterations 1000
```

#### Command-Line Arguments

*   `--bootstrap-iterations`: Number of bootstrap iterations for confidence intervals (default: 1000)
    *   Recommended: 1000 for publication-ready results
    *   Quick test: 100 for faster results (less accurate CIs)
*   `--batch-size`: Batch size for dense models (default: 32)
    *   Higher = faster but more VRAM
*   `--models`: Comma-separated list of models to test (default: "all")
    *   Options: `ColPali-v1.3`, `SigLIP-400M`, `LAION-CLIP-H`, `Jina-CLIP-v1`, `MetaCLIP-H14`, `OpenAI-CLIP-L`, `Apple-DFN5B-H`
    *   Example: `--models SigLIP-400M,LAION-CLIP-H`
*   `--sample-size`: Number of COCO images to evaluate (default: 5000)
*   `--workers`: Number of download workers (default: 16)
*   `--cache-dir`: Directory for cached images (default: `./coco_images`)
*   `--output`: Output CSV filename (default: `benchmark_v29_statistical_results.csv`)

#### Expected Runtime

*   **Dense models** (CLIP-based): ~2-3 hours per model
*   **ColPali**: ~15-20 hours (due to late-interaction scoring)
*   **Total for all 7 models**: ~20-30 hours

## 📈 Methodology & Metrics

### 1. Text-to-Image (T2I) Retrieval
*   **Task:** Given a caption, find the exact corresponding image among 5,000 candidates.
*   **Metric:** **Recall@1 (R@1)** - Percentage of times the correct image is the #1 rank.
*   **Significance:** The hardest and most precise metric for understanding.

### 2. Image-to-Text (I2T) Retrieval
*   **Standard Protocol:** Given an image, find *any* of its 5 valid captions among 25,000 candidates.
*   **Symmetric Protocol:** Given an image, find its *specific* single caption (1-to-1 mapping), allowing direct comparison with T2I.

### 3. Failure Analysis
The benchmark automatically analyzes queries based on:
*   **Complexity:** Presence of spatial relations ("next to"), colors, or counting terms.
*   **Category:** Performance breakdown by COCO supercategories (e.g., "Food", "Vehicle", "Animal").

## 📁 Project Structure

*   `main.py`: **Primary entry point.** Runs the statistical COCO benchmark with bootstrap sampling.
*   `requirements.txt`: Python package dependencies.
*   `benchmark_v29_statistical_results.csv`: Output file containing the benchmark results (generated after run).
*   `benchmark_v29.log`: Detailed execution log (generated after run).
*   `coco_images/`: Local cache for downloaded COCO images (created automatically).

## 📊 Output Format

The benchmark generates a CSV file with the following columns:

*   **Model**: Model name
*   **T2I_R@{1,5,10}_mean/lower/upper**: Text-to-Image Recall with 95% confidence intervals
*   **I2T_R@{1,5,10}_mean/lower/upper**: Image-to-Text Recall (multi-caption) with CIs
*   **I2T_Sym_R@{1,5,10}_mean/lower/upper**: Image-to-Text Recall (symmetric/single-caption) with CIs
*   **Time**: Total benchmark time (including all bootstrap iterations)
*   **Encoding_Time**: Time to encode all images/captions once
*   **QPS**: Queries per second (images processed / encoding time)
*   **Img_per_sec**: Images processed per second

## 📜 Citation

If you use this benchmark or results, please cite:

```bibtex
@techreport{coco_benchmark_2025,
  title={Statistical Benchmark of 7 Multimodal Embedding Models on MS-COCO Karpathy Split},
  year={2025},
  institution={Independent Research}
}
```

## 📝 Notes

*   The benchmark automatically downloads and caches MS-COCO images to `./coco_images/`
*   Bootstrap sampling is SLOW but provides true confidence intervals
*   For quick testing, use `--bootstrap-iterations 100` (less accurate but faster)
*   ColPali requires the `colpali_engine` package
*   All models are evaluated on the same COCO Karpathy test split (5,000 images)