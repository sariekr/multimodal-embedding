# Bidirectional Multimodal Retrieval Benchmark: Apple vs LAION vs Meta vs ColPali vs Google vs OpenAI

**Cem Dilmegani**
*with Ekrem Sarı*
*updated on Nov 27, 2025*

We benchmarked 8 leading multimodal embedding models on NVIDIA A40 with **bidirectional retrieval evaluation**: **Apple DFN5B**, **LAION-CLIP-H**, **MetaCLIP-H14**, **ColPali v1.3**, **OpenAI CLIP-L**, **Jina CLIP-v1**, **Google SigLIP-400M**, and **Google SigLIP-Base**. Each model was evaluated on both **Text-to-Image (T2I)** and **Image-to-Text (I2T)** retrieval using 1,000 samples from Flickr30k test set and 400 samples from Winoground. This reveals critical asymmetries invisible in unidirectional benchmarks: ColPali exhibits a 23-point T2I/I2T gap, while dense CLIP models maintain <1% symmetry.

## Multimodal models benchmark results

We measured **Recall@1 (R@1)**, **Recall@5 (R@5)**, and **Recall@10 (R@10)** as primary retrieval metrics, along with **Mean Reciprocal Rank (MRR)** for ranking quality. All models were evaluated on 1,000 samples from Flickr30k test set using official `lmms-lab/flickr30k` dataset. **Bidirectional evaluation** tests both Text-to-Image (T2I) and Image-to-Text (I2T) retrieval.

### Text-to-Image retrieval (T2I): Flickr30k general vision

| Model | R@1 | R@5 | R@10 | MRR |
|:---|:---:|:---:|:---:|:---:|
| **Apple DFN5B-H** | **81.8%** | **95.9%** | **98.0%** | **0.880** |
| **LAION-CLIP-H** | 79.5% | 94.7% | 97.2% | 0.862 |
| **MetaCLIP-H14** | 79.5% | 94.8% | 97.4% | 0.861 |
| **ColPali v1.3** | 76.1% | 93.1% | 96.2% | 0.836 |
| **OpenAI CLIP-L** | 69.2% | 90.8% | 95.0% | 0.788 |
| **Jina CLIP-v1** | 68.2% | 88.8% | 93.2% | 0.774 |
| **SigLIP-400M** | 58.7% | 79.2% | 85.1% | 0.680 |
| **SigLIP-Base** | 23.9% | 43.0% | 50.7% | 0.329 |

### Image-to-Text retrieval (I2T): Flickr30k general vision

| Model | R@1 | R@5 | R@10 | MRR |
|:---|:---:|:---:|:---:|:---:|
| **Apple DFN5B-H** | **94.3%** | **99.8%** | **100.0%** | **0.966** |
| **LAION-CLIP-H** | 93.2% | 98.7% | 99.8% | 0.957 |
| **MetaCLIP-H14** | 92.5% | 99.6% | 99.9% | 0.954 |
| **OpenAI CLIP-L** | 88.4% | 98.8% | 99.7% | 0.928 |
| **Jina CLIP-v1** | 84.5% | 96.7% | 98.2% | 0.899 |
| **ColPali v1.3** | 74.7% | 93.0% | 96.6% | 0.826 |
| **SigLIP-400M** | 52.5% | 71.7% | 80.7% | 0.616 |
| **SigLIP-Base** | 32.0% | 55.8% | 66.9% | 0.432 |

### Winoground compositional reasoning

| Model | Text Score | Image Score | Group Score |
|:---|:---:|:---:|:---:|
| **Apple DFN5B-H** | **35.2%** | **14.5%** | **12.8%** |
| **LAION-CLIP-H** | 29.2% | 12.5% | 10.2% |
| **Jina CLIP-v1** | 29.0% | 7.5% | 5.0% |
| **MetaCLIP-H14** | 28.5% | 13.8% | 9.8% |
| **ColPali v1.3** | 24.2% | 15.8% | 10.2% |
| **OpenAI CLIP-L** | 24.2% | 11.0% | 7.0% |
| **SigLIP-400M** | 15.8% | 10.8% | 5.0% |
| **SigLIP-Base** | 14.2% | 6.2% | 2.5% |

### Performance metrics: Speed and efficiency

| Model | Throughput (QPS) | Inference Time | Embedding Size |
|:---|:---:|:---:|:---:|
| **SigLIP-Base** | **864.7** | 5.8s | 768d |
| **LAION-CLIP-H** | 460.7 | 10.9s | 1024d |
| **OpenAI CLIP-L** | 366.6 | 13.6s | 768d |
| **MetaCLIP-H14** | 352.6 | 14.2s | 1024d |
| **SigLIP-400M** | 251.5 | 19.9s | 1152d |
| **Apple DFN5B-H** | 207.3 | 24.1s | 1024d |
| **Jina CLIP-v1** | 150.0 | 33.3s | 768d |
| **ColPali v1.3** | 20.7 | 241.0s | ~1031tok |

*Throughput measured as Queries Per Second (QPS) on 1,000 Flickr30k samples with 5 captions per image (5,000 queries). Higher is better.*

## Key findings

Our controlled benchmark isolates architectural contributions by using identical hardware (NVIDIA A40), dataset splits (1,000 samples from official sources), and evaluation metrics. All results are from single-run measurements ensuring reproducibility with fixed seed (42). **Bidirectional evaluation** reveals critical differences between Text-to-Image and Image-to-Text retrieval.

**Apple DFN5B achieves the highest accuracy in both directions:** With 81.8% T2I R@1 and 94.3% I2T R@1, Apple's model demonstrates exceptional I2T performance thanks to proper multi-caption evaluation (checking ANY of 5 captions). Its near-perfect I2T scores (99.8% R@5, 100% R@10, 0.966 MRR) and Winoground leadership (12.8% group score) showcase superior semantic understanding. The I2T advantage (12.5pp higher than T2I) reflects the multi-caption protocol where each image has 5 valid matches.

**LAION-CLIP-H and MetaCLIP-H14 offer the best speed-accuracy tradeoff:** These open-source models deliver 79.5% T2I and 93%+ I2T R@1 while maintaining exceptional throughput (461 and 353 QPS). LAION-CLIP-H achieves 93.2% I2T R@1 with 0.957 MRR—nearly matching Apple's accuracy at 2.2x the speed. Their consistent performance across both directions makes them ideal for production deployments requiring bidirectional retrieval.

**Multi-caption I2T evaluation reveals true capabilities:** With the corrected I2T protocol (checking ANY of 5 captions per image), all dense CLIP models show 10-15pp higher I2T than T2I scores. This asymmetry is expected and correct: I2T benefits from multiple valid targets. ColPali's smaller gap (76.1% T2I → 74.7% I2T, 1.4pp drop) suggests its late-interaction mechanism performs more symmetrically than previously measured.

**SigLIP models show consistent underperformance:** SigLIP-400M achieves 58.7% T2I and 52.5% I2T, while SigLIP-Base scores 23.9% T2I and 32.0% I2T. Both models trail CLIP alternatives by 20-30pp, suggesting training quality issues. Despite extreme speed (865 QPS for Base), the accuracy penalty suggests limited production viability for quality-sensitive applications.

## Understanding the performance hierarchy

Bidirectional evaluation reveals distinct tiers based on model architecture, retrieval symmetry, and production viability:

**Tier 1: Symmetric high-accuracy models (Apple DFN5B, LAION-CLIP-H, MetaCLIP-H14)**
- **Apple DFN5B**: Best-in-class accuracy (89.8% T2I, 89.1% I2T) with 0.7pp asymmetry
  - 1024d embeddings, 45.6 QPS throughput
  - Winoground leader (12.8% group score) demonstrates compositional reasoning
  - Ideal for: Maximum accuracy applications, content moderation, medical imaging

- **LAION-CLIP-H**: Production sweet spot (87.5% T2I, 87.8% I2T) with 0.3pp asymmetry
  - 2.5x faster than Apple (113.5 QPS), open-source license
  - Near-perfect symmetry validates robust semantic alignment
  - Ideal for: General-purpose retrieval, image search engines, production RAG

- **MetaCLIP-H14**: Comparable to LAION (87.2% T2I, 87.6% I2T) with 0.4pp asymmetry
  - 97.9 QPS, Meta's curated training approach
  - Ideal for: Production systems requiring open-source flexibility

**Tier 2: Moderate-accuracy symmetric models (OpenAI CLIP-L, Jina CLIP-v1)**
- **OpenAI CLIP-L**: Foundation model (77.5% T2I, 80.1% I2T) with acceptable symmetry
  - 81.5 QPS, widely adopted baseline
  - Ideal for: Fine-tuning bases, cost-sensitive deployments

- **Jina CLIP-v1**: Bilingual support (79.2% T2I, 76.9% I2T) but slower (33.1 QPS)
  - Ideal for: Multilingual retrieval applications

**Tier 3: Asymmetric specialized models (ColPali v1.3)**
- **Critical limitation**: 23-point directional gap (84.6% T2I → 61.4% I2T)
  - Document-retrieval architecture fundamentally query→document optimized
  - Extreme slowness (7.7 QPS, 15x slower than LAION) compounds asymmetry issue
  - Multi-vector late-interaction unsuitable for bidirectional retrieval
  - **Only use for**: Document-specific tasks (PDF search, OCR) where T2I direction dominates

**Tier 4: Production-unsuitable models (SigLIP family)**
- **SigLIP-400M**: Severe asymmetry (65.8% T2I → 36.7% I2T, 29-point gap)
  - Low absolute accuracy disqualifies for most retrieval tasks

- **SigLIP-Base**: Extreme asymmetry (38.3% T2I → 32.8% I2T)
  - Speed (197.4 QPS) cannot compensate for accuracy penalty
  - **Avoid in production** unless accuracy is completely secondary

## Model selection rationale

We included models representing different scales, architectures, and training approaches:

**Apple DFN5B-H (apple/DFN5B-CLIP-ViT-H-14-378):** Apple's DataComp-trained model with sophisticated data filtering
**LAION-CLIP-H (laion/CLIP-ViT-H-14-laion2B-s32B-b79K):** Open-source champion trained on 2B image-text pairs
**MetaCLIP-H14 (facebook/metaclip-h14-fullcc2.5b):** Meta's curated approach to web-scale training
**ColPali v1.3 (vidore/colpali-v1.3):** Document-specialized multi-vector architecture based on PaliGemma
**OpenAI CLIP-L (openai/clip-vit-large-patch14-336):** The foundational model that defined the category
**Jina CLIP-v1 (jinaai/jina-clip-v1):** Bilingual model optimized for multilingual retrieval
**SigLIP-400M (google/siglip-so400m-patch14-384):** Google's sigmoid loss variant at 400M parameters
**SigLIP-Base (google/siglip-base-patch16-224):** Smaller SigLIP variant for speed comparison

## Benchmark methodology

### Test environment

**Hardware configuration:**
- **GPU:** NVIDIA A40 48GB VRAM
- **System:** RunPod cloud instance
- **Precision:** BFloat16 (bf16) for all models
- **Batch sizes:** Optimized per model (4-64) to maximize GPU utilization without OOM

**Software stack:**
- `transformers`: 4.46+
- `colpali-engine`: 0.3.1
- `datasets`: 3.1+
- PyTorch with CUDA 12.1

### Dataset configuration

All datasets use official sources to ensure reproducibility:

**Flickr30k:** `lmms-lab/flickr30k`, 1,000 sampled images
- **Dataset limitation:** The `lmms-lab/flickr30k` "test" split contains 31,783 samples (appears to be the full dataset, not standard test split)
- **Sampling protocol:** We randomly sample 1,000 images with fixed seed (42) to approximate standard Flickr30k test set size
- **Multi-caption evaluation:** Each image has 5 human-written captions
  - T2I: Each of 5 captions should retrieve its source image
  - I2T: Each image's retrieval is correct if ANY of its 5 captions appears in top-K
- Natural photographs with diverse visual scenes
- Caption field used directly without preprocessing

**Winoground:** `facebook/winoground` complete test set (400 samples)
- Paired image-text challenges requiring compositional reasoning
- Reports Text, Image, and Group scores per original methodology
- No sampling—uses full benchmark

### Evaluation protocol

**Reproducibility:** Fixed seed (42) ensures identical dataset sampling and model initialization across runs

**Metrics computed:**
- **Recall@K (K=1,5,10):** Percentage of queries where correct match appears in top-K results
- **MRR (Mean Reciprocal Rank):** Average of 1/rank across all queries, penalizing lower-ranked correct answers
- **Throughput (QPS):** Total samples divided by wall-clock inference time, including encoding and scoring
- **Embedding dimensions:** Reported as "Nd" for dense vectors or "~N tok" for multi-vector models

**Measurement procedure:**
1. Load model and processor with bf16 precision
2. Process all images in batches → generate image embeddings
3. Process all text queries in batches → generate text embeddings
4. Compute similarity matrix (cosine for dense, late-interaction scoring for ColPali)
5. Calculate metrics from similarity scores
6. Clear GPU memory before next model

## Production deployment recommendations

Based on comprehensive evaluation across accuracy, speed, and architectural considerations:

**For maximum accuracy applications (e-commerce search, content moderation, medical imaging):**
Use **Apple DFN5B-H** (88.9% R@1, 0.929 MRR, 33.5 QPS)
- Highest accuracy across all metrics justifies moderate speed tradeoff
- Strong compositional reasoning (12.8% Winoground) handles complex queries
- 1024d embeddings provide rich semantic representation

**For production-scale general retrieval (image search engines, visual RAG, recommendation systems):**
Use **LAION-CLIP-H** or **MetaCLIP-H14** (86%+ R@1, 77-88 QPS)
- Best speed-accuracy balance for most use cases
- Open-source licenses enable customization and cost optimization
- 2.5x faster than Apple with only 3% accuracy loss
- Proven at scale across industry deployments

**For document-heavy workflows (PDF search, invoice processing, slide retrieval):**
**Use ColPali ONLY for unidirectional T2I retrieval** (text queries → image documents)
- T2I: 84.6% R@1 (competitive), I2T: 61.4% R@1 (poor, 23-point gap)
- 15x slower than LAION (7.7 vs 113.5 QPS) creates unacceptable latency
- Multi-vector embeddings (1031 tokens) require specialized infrastructure
- Only justified when: (1) T2I direction only, (2) document-specific features critical

**Models with notable limitations:**
- **ColPali for bidirectional retrieval:** 23pp T2I/I2T asymmetry may impact symmetric use cases
- **SigLIP family:** 29pp (400M) and 6pp (Base) asymmetries plus lower absolute accuracy
- **Jina CLIP-v1:** Slower than Apple (33.1 vs 45.6 QPS) with 10pp lower accuracy

## Conclusion

Bidirectional evaluation fundamentally reshapes our understanding of multimodal embedding models:

**Apple DFN5B establishes the accuracy standard** with 81.8% T2I and 94.3% I2T R@1, demonstrating that sophisticated data curation (DataComp filtering) creates robust semantic representations. Its exceptional I2T performance (99.8% R@5, 100% R@10, 0.966 MRR) and Winoground leadership (12.8% group score) showcase superior compositional understanding. The higher I2T than T2I scores reflect the multi-caption protocol where each image has 5 valid matches.

**LAION-CLIP-H and MetaCLIP-H14 emerge as optimal production choices** with 79.5% T2I and 93%+ I2T R@1 at exceptional throughput (461 and 353 QPS). LAION-CLIP-H matches Apple's I2T accuracy (93.2% vs 94.3%) while delivering 2.2x faster inference. Their open-source licensing, proven scalability, and consistent bidirectional performance make them ideal for production retrieval systems.

**Multi-caption I2T evaluation reveals expected asymmetry**. All dense CLIP models show 10-15pp higher I2T than T2I scores—this is correct behavior when each image has 5 valid caption targets versus 1 image target per caption. ColPali shows minimal directional preference (76.1% T2I vs 74.7% I2T, 1.4pp gap), suggesting its late-interaction mechanism handles both directions more symmetrically than initially expected.

**SigLIP models trail CLIP alternatives by 20-30pp** across both T2I and I2T metrics. SigLIP-400M (58.7% T2I, 52.5% I2T) and SigLIP-Base (23.9% T2I, 32.0% I2T) suggest training quality issues. Despite extreme speed (865 QPS for Base), the accuracy penalty limits production viability for quality-sensitive applications.

**Critical methodological insight**: Proper I2T evaluation requires checking if ANY of an image's multiple captions appears in top-K, not just the first caption. Our initial implementation (v18) only checked the first caption, undercounting I2T performance by 60-80%. The corrected protocol (v19) reveals true bidirectional capabilities and explains why I2T scores exceed T2I scores—multiple valid targets per query increase retrieval success rates.

For developers building multimodal applications in 2025: **Implement proper multi-caption I2T evaluation**. Start with **LAION-CLIP-H** or **MetaCLIP-H14** for production deployments (93%+ I2T, 460+ QPS). Upgrade to **Apple DFN5B** when maximum accuracy (94.3% I2T) justifies 2.2x throughput tradeoff. All dense CLIP models show consistent bidirectional performance when evaluated correctly.
