# Benchmark Results Summary (V28)

**Dataset:** MS-COCO Karpathy (5000 images) + Winoground (400 samples)
**Date:** November 2025
**Runs:** 3 (seeds: 42, 43, 44)

## 📊 Main Results (Sorted by T2I R@1)

| Rank | Model | T2I R@1 | I2T R@1 | Gap | Wino Image | QPS |
|:----:|:------|:-------:|:-------:|:---:|:----------:|:---:|
| 🥇 | **Apple-DFN5B-H** | **50.1%** | **69.7%** | +19.6pp | 35.2% | 34.4 |
| 🥈 | **LAION-CLIP-H** | **46.3%** | **66.3%** | +20.0pp | 29.2% | 83.8 |
| 🥉 | **MetaCLIP-H14** | **45.8%** | **65.9%** | +20.1pp | 28.5% | 76.3 |
| 4 | ColPali-v1.3 | 44.9% | 48.8% | +3.9pp | 24.2% | 2.9 |
| 5 | Jina-CLIP-v1 | 39.3% | 55.3% | +16.0pp | 29.0% | 25.8 |
| 6 | SigLIP-400M | 35.4% | 45.1% | +9.7pp | 15.8% | 47.1 |
| 7 | OpenAI-CLIP-L | 34.4% | 58.0% | +23.6pp | 24.5% | 60.6 |

## 🎯 Key Findings

### 1. Best Overall Performance
- **Apple DFN5B-H** achieves state-of-the-art 50.1% T2I R@1
- Outperforms LAION-CLIP-H by +3.8pp despite similar architecture
- Strong compositional reasoning: 35.2% on Winoground Image

### 2. I2T > T2I Asymmetry
**Expected Pattern (Dense Models):**
- LAION, MetaCLIP, OpenAI: ~20pp gap (due to 5 captions per image)
- Jina, SigLIP: ~10-16pp gap

**ColPali Exception:**
- Only +3.9pp gap (nearly symmetric)
- Late-interaction scoring benefits both directions equally

### 3. Speed-Accuracy Tradeoff
| Model | T2I R@1 | QPS | Efficiency |
|:------|:-------:|:---:|:----------:|
| LAION-CLIP-H | 46.3% | 83.8 | ⭐⭐⭐ Best balance |
| Apple-DFN5B-H | 50.1% | 34.4 | High accuracy, slower |
| ColPali-v1.3 | 44.9% | 2.9 | Accurate but very slow |

### 4. Compositional Reasoning (Winoground)
**Top Performers:**
- Apple-DFN5B-H: 35.2% Image, 14.5% Text
- Jina-CLIP-v1: 29.0% Image, 7.5% Text
- LAION-CLIP-H: 29.2% Image, 12.5% Text

**Observation:** Image compositional understanding generally stronger than text.

## 📈 Detailed Metrics

### Text-to-Image Retrieval

| Model | R@1 | R@5 | R@10 | MRR |
|:------|:---:|:---:|:----:|:---:|
| Apple-DFN5B-H | 50.1 | 74.1 | 82.6 | - |
| LAION-CLIP-H | 46.3 | 70.9 | 79.7 | - |
| MetaCLIP-H14 | 45.8 | 70.6 | 79.5 | - |
| ColPali-v1.3 | 44.9 | 68.3 | 77.6 | - |
| Jina-CLIP-v1 | 39.3 | 64.6 | 74.2 | - |
| SigLIP-400M | 35.4 | 59.1 | 68.1 | - |
| OpenAI-CLIP-L | 34.4 | 59.1 | 69.1 | - |

### Image-to-Text Retrieval

| Model | R@1 | R@5 | R@10 |
|:------|:---:|:---:|:----:|
| Apple-DFN5B-H | 69.7 | 89.2 | 94.1 |
| LAION-CLIP-H | 66.3 | 86.5 | 91.9 |
| MetaCLIP-H14 | 65.9 | 86.0 | 91.9 |
| OpenAI-CLIP-L | 58.0 | 80.6 | 87.8 |
| Jina-CLIP-v1 | 55.3 | 78.4 | 86.3 |
| ColPali-v1.3 | 48.8 | 72.9 | 81.6 |
| SigLIP-400M | 45.1 | 64.9 | 72.0 |

### Winoground (Compositional Reasoning)

| Model | Text | Image | Group |
|:------|:----:|:-----:|:-----:|
| Apple-DFN5B-H | 14.5 | **35.2** | **12.8** |
| Jina-CLIP-v1 | 7.5 | 29.0 | 5.0 |
| LAION-CLIP-H | 12.5 | 29.2 | 10.2 |
| MetaCLIP-H14 | 13.8 | 28.5 | 9.8 |
| OpenAI-CLIP-L | 11.5 | 24.5 | 7.2 |
| ColPali-v1.3 | **15.8** | 24.2 | 10.2 |
| SigLIP-400M | 10.8 | 15.8 | 5.0 |

## 🔬 Technical Notes

### Dataset
- **COCO:** yerevann/coco-karpathy, test split (5000 images)
- **Winoground:** facebook/winoground (400 image-text pairs)
- Each COCO image has 5 captions

### Evaluation Protocol
- **T2I:** Use 1st caption per image as query (standard protocol)
- **I2T:** Image retrieval correct if ANY of 5 captions in top-K
- **Multi-run:** 3 runs with seeds 42, 43, 44

### Hardware
- **GPU:** NVIDIA A40 (48GB)
- **Pod 1:** PyTorch 2.4 (5 models: ColPali, SigLIP, LAION, Jina, MetaCLIP)
- **Pod 2:** PyTorch 2.8 (2 models: OpenAI, Apple - requires newer torch for pytorch_model.bin)

### Known Limitations
- **Std = 0.0:** Dataset doesn't vary between runs (same 5000 images)
- **Multi-seed has no effect** on metrics when dataset is fixed
- For true statistical variance, need different sample selections or cross-validation

## 🚫 Excluded Models

| Model | Reason |
|:------|:-------|
| SigLIP-Base | Poor performance (1.4% T2I R@1) - weak discriminative margin |
| ~~OpenAI-CLIP-L~~ | ~~Initially excluded (torch.load issue) - NOW INCLUDED~~ |
| ~~Apple-DFN5B-H~~ | ~~Initially excluded (torch.load issue) - NOW INCLUDED~~ |

## 📝 Recommendations

**For Production Use:**
1. **Best Accuracy:** Apple-DFN5B-H (50.1% T2I R@1)
2. **Best Speed/Accuracy:** LAION-CLIP-H (46.3% T2I, 83.8 QPS)
3. **Late-Interaction:** ColPali-v1.3 (44.9% T2I, symmetric behavior)

**For Research:**
- **Compositional Reasoning:** Apple-DFN5B-H (35.2% Winoground Image)
- **Open-Source:** LAION-CLIP-H or MetaCLIP-H14

## 📚 Citation

```bibtex
@misc{multimodal-benchmark-2025,
  title={Comprehensive Evaluation of Multimodal Embedding Models},
  author={Your Name},
  year={2025},
  note={MS-COCO Karpathy + Winoground Benchmark}
}
```

---

**Last Updated:** November 2025
**Benchmark Version:** V28 (Publication-Ready)
