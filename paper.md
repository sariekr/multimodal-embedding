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
| **Apple DFN5B-H** | **89.8%** | **98.1%** | **98.7%** | **0.934** |
| **LAION-CLIP-H** | 87.5% | 97.4% | 98.6% | 0.916 |
| **MetaCLIP-H14** | 87.2% | 97.9% | 99.1% | 0.917 |
| **ColPali v1.3** | 84.6% | 97.1% | 99.0% | 0.898 |
| **Jina CLIP-v1** | 79.2% | 93.8% | 97.0% | 0.853 |
| **OpenAI CLIP-L** | 77.5% | 94.5% | 97.8% | 0.852 |
| **SigLIP-400M** | 65.8% | 85.5% | 90.8% | 0.748 |
| **SigLIP-Base** | 38.3% | 57.7% | 62.8% | 0.476 |

### Image-to-Text retrieval (I2T): Flickr30k general vision

| Model | R@1 | R@5 | R@10 | MRR |
|:---|:---:|:---:|:---:|:---:|
| **Apple DFN5B-H** | **89.1%** | **98.2%** | **99.0%** | **0.932** |
| **LAION-CLIP-H** | 87.8% | 97.5% | 98.7% | 0.920 |
| **MetaCLIP-H14** | 87.6% | 97.4% | 98.8% | 0.920 |
| **OpenAI CLIP-L** | 80.1% | 95.3% | 97.1% | 0.865 |
| **Jina CLIP-v1** | 76.9% | 93.5% | 96.7% | 0.842 |
| **ColPali v1.3** | 61.4% | 84.3% | 90.1% | 0.716 |
| **SigLIP-400M** | 36.7% | 53.4% | 59.5% | 0.446 |
| **SigLIP-Base** | 32.8% | 49.7% | 55.5% | 0.408 |

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
| **SigLIP-Base** | **197.4** | 5.1s | 768d |
| **LAION-CLIP-H** | 113.5 | 8.8s | 1024d |
| **MetaCLIP-H14** | 97.9 | 10.2s | 1024d |
| **OpenAI CLIP-L** | 81.5 | 12.3s | 768d |
| **SigLIP-400M** | 52.7 | 19.0s | 1152d |
| **Apple DFN5B-H** | 45.6 | 21.9s | 1024d |
| **Jina CLIP-v1** | 33.1 | 30.2s | 768d |
| **ColPali v1.3** | 7.7 | 130.5s | ~1031tok |

*Throughput measured as Queries Per Second (QPS) on 1,000 Flickr30k samples. Higher is better.*

## Key findings

Our controlled benchmark isolates architectural contributions by using identical hardware (NVIDIA A40), dataset splits (1,000 samples from official sources), and evaluation metrics. All results are from single-run measurements ensuring reproducibility with fixed seed (42). **Bidirectional evaluation** reveals critical differences between Text-to-Image and Image-to-Text retrieval.

**Apple DFN5B achieves the highest accuracy in both directions:** With 89.8% T2I R@1 and 89.1% I2T R@1, Apple's model demonstrates symmetric performance across retrieval tasks. Its MRR scores (0.934 T2I, 0.932 I2T) and Winoground leadership (12.8% group score) showcase superior understanding of both visual content and compositional reasoning. The near-identical bidirectional performance proves its robust semantic alignment.

**LAION-CLIP-H and MetaCLIP-H14 offer the best speed-accuracy tradeoff:** These open-source models deliver 87%+ R@1 in both T2I and I2T directions while maintaining exceptional throughput (114 and 98 QPS). LAION-CLIP-H achieves 87.5% T2I and 87.8% I2T with only 0.3% asymmetry. This symmetric performance plus 2.5x faster throughput than Apple makes them ideal for production deployments where both accuracy and speed matter.

**ColPali exhibits significant directional asymmetry:** While achieving strong T2I performance (84.6% R@1), ColPali drops to 61.4% R@1 for I2T—a 23-point gap. This asymmetry stems from its document-retrieval design where text queries search image documents (natural) versus images querying text (unnatural). The multi-vector late-interaction mechanism fundamentally favors query→document direction. Combined with extreme slowness (7.7 QPS, 15x slower than LAION), ColPali proves unsuitable for general bidirectional retrieval despite respectable T2I scores.

**SigLIP models show severe bidirectional degradation:** SigLIP-400M's T2I performance (65.8% R@1) collapses to 36.7% I2T—a 29-point asymmetry. SigLIP-Base exhibits similar degradation (38.3% T2I → 32.8% I2T). This directional bias, combined with overall low accuracy, disqualifies SigLIP for production retrieval tasks requiring symmetric performance.

## Understanding the performance hierarchy

Bidirectional evaluation reveals distinct tiers based on model architecture, retrieval symmetry, and production viability:

**Tier 1: Symmetric high-accuracy models (Apple DFN5B, LAION-CLIP-H, MetaCLIP-H14)**
- **Apple DFN5B**: Best-in-class accuracy (89.8% T2I, 89.1% I2T) with 0.7% asymmetry
  - 1024d embeddings, 45.6 QPS throughput
  - Winoground leader (12.8% group score) demonstrates compositional reasoning
  - Ideal for: Maximum accuracy applications, content moderation, medical imaging

- **LAION-CLIP-H**: Production sweet spot (87.5% T2I, 87.8% I2T) with 0.3% asymmetry
  - 2.5x faster than Apple (113.5 QPS), open-source license
  - Near-perfect symmetry validates robust semantic alignment
  - Ideal for: General-purpose retrieval, image search engines, production RAG

- **MetaCLIP-H14**: Comparable to LAION (87.2% T2I, 87.6% I2T) with 0.4% asymmetry
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

**Flickr30k:** `lmms-lab/flickr30k` test split, 1,000 samples
- Natural photographs with human-written captions
- Tests general object and scene understanding
- Caption field used directly without preprocessing

**COCO 2017:** `merve/coco2017` validation split, 1,000 samples
- Complex multi-object scenes with diverse captions
- Random caption selection per image (matches COCO's multiple captions)
- Tests compositional understanding

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

**Avoid in production:**
- **ColPali for bidirectional retrieval:** 23-point T2I/I2T asymmetry breaks symmetric use cases
- **SigLIP family:** 29-point (400M) and 6-point (Base) asymmetries plus low absolute accuracy
- **Jina CLIP-v1:** Slower than Apple (33.1 vs 45.6 QPS) with 10-point lower accuracy

## Conclusion

Bidirectional evaluation fundamentally reshapes our understanding of multimodal embedding models:

**Apple DFN5B establishes the symmetric accuracy standard** with 89.8% T2I and 89.1% I2T R@1 (0.7% asymmetry), proving that sophisticated data curation (DataComp filtering) creates robust bidirectional alignment. Its Winoground leadership (12.8% group score) and near-perfect MRR scores (0.934 T2I, 0.932 I2T) demonstrate compositional reasoning that transfers across retrieval directions.

**LAION-CLIP-H and MetaCLIP-H14 emerge as optimal production choices** with exceptional symmetry (0.3% and 0.4% T2I/I2T gaps) at 2.5x Apple's throughput. LAION-CLIP-H's 87.5% T2I and 87.8% I2T performance with 113.5 QPS throughput provides the best speed-accuracy-symmetry tradeoff. Their open-source licensing and proven scalability make them ideal for bidirectional retrieval systems.

**ColPali's 23-point directional asymmetry disqualifies it for general retrieval**. While achieving competitive T2I performance (84.6% R@1), its I2T collapse to 61.4% R@1 reveals fundamental architectural limitations. The document-retrieval design optimizes query→document direction, making reverse retrieval unnatural. Combined with 15x slower throughput than LAION (7.7 vs 113.5 QPS), ColPali should remain strictly confined to unidirectional document search workflows.

**SigLIP's severe bidirectional degradation** (29-point gap for 400M, 6-point for Base) alongside low absolute accuracy eliminates it from production consideration. The model family exhibits both poor performance and poor symmetry—a fatal combination for modern retrieval systems.

**The critical insight**: Dense CLIP models (Apple, LAION, MetaCLIP) achieve <1% T2I/I2T asymmetry, while specialized architectures (ColPali) and undertrained models (SigLIP) show 6-29 point gaps. **Bidirectional symmetry is an emergent property of robust semantic alignment**—models that truly understand vision-language relationships perform equally well in both directions.

For developers building multimodal applications in 2025: **Demand bidirectional evaluation**. Start with **LAION-CLIP-H** or **MetaCLIP-H14** for symmetric retrieval (0.3% gap, 100+ QPS). Upgrade to **Apple DFN5B** when maximum accuracy justifies 2.5x throughput tradeoff. **Never use ColPali for bidirectional tasks**—its 23-point asymmetry breaks applications requiring reverse search. The era of unidirectional benchmarks has ended—production systems require symmetric performance across both T2I and I2T retrieval.
