# Multimodal Embedding Models: Apple vs ColPali vs Google vs OpenAI vs Meta

**Cem Dilmegani**
*with Ekrem Sarı*
*updated on Nov 27, 2025*

We benchmarked 8 leading multimodal embedding models on NVIDIA A40: **Apple DFN5B**, **ColPali v1.3**, **MetaCLIP**, **LAION-Huge**, **OpenAI CLIP**, **Jina CLIP**, **Google SigLIP-400M**, and **Google SigLIP-Base**. Each model processed identical workloads across 3 diverse datasets—**Flickr30k** (General Vision), **COCO** (Complex Scenes), and **Winoground** (Compositional Reasoning)—using official datasets with reproducible methodology and comprehensive metrics including **R@1, R@5, R@10, and MRR**.

## Multimodal models benchmark results

We measured **Recall@1 (R@1)**, **Recall@5 (R@5)**, and **Recall@10 (R@10)** as primary retrieval metrics, along with **Mean Reciprocal Rank (MRR)** for ranking quality. All models were evaluated on 1,000 samples from Flickr30k test set using official `lmms-lab/flickr30k` dataset.

### Accuracy metrics: Flickr30k general vision retrieval

| Model | R@1 | R@5 | R@10 | MRR | Winoground Group |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Apple DFN5B-H** | **88.9%** | **97.5%** | **99.1%** | **0.929** | **12.8%** |
| **LAION-CLIP-H** | 86.0% | 96.7% | 98.7% | 0.910 | 10.2% |
| **MetaCLIP-H14** | 85.7% | 97.7% | 98.8% | 0.908 | 9.8% |
| **ColPali v1.3** | 83.8% | 95.9% | 97.6% | 0.892 | 10.2% |
| **OpenAI CLIP-L** | 77.7% | 92.9% | 96.0% | 0.845 | 7.0% |
| **Jina CLIP-v1** | 75.9% | 92.4% | 95.7% | 0.832 | 5.0% |
| **SigLIP-400M** | 72.3% | 88.1% | 92.1% | 0.796 | 5.0% |
| **SigLIP-Base** | 34.7% | 55.8% | 60.6% | 0.437 | 2.5% |

### Performance metrics: Speed and efficiency

| Model | Throughput (QPS) | Inference Time | Embedding Size |
|:---|:---:|:---:|:---:|
| **SigLIP-Base** | **170.6** | 5.9s | 768d |
| **LAION-CLIP-H** | 87.9 | 11.4s | 1024d |
| **MetaCLIP-H14** | 76.9 | 13.0s | 1024d |
| **OpenAI CLIP-L** | 53.4 | 18.7s | 768d |
| **SigLIP-400M** | 42.9 | 23.3s | 1152d |
| **Apple DFN5B-H** | 33.5 | 29.9s | 1024d |
| **Jina CLIP-v1** | 19.1 | 52.5s | 768d |
| **ColPali v1.3** | 6.7 | 149.6s | ~1031tok |

*Throughput measured as Queries Per Second (QPS) on 1,000 Flickr30k samples. Higher is better.*

## Key findings

Our controlled benchmark isolates architectural contributions by using identical hardware (NVIDIA A40), dataset splits (1,000 samples from official sources), and evaluation metrics. All results are from single-run measurements ensuring reproducibility with fixed seed (42).

**Apple DFN5B achieves the highest accuracy across all metrics:** With 88.9% R@1 and an MRR of 0.929 on Flickr30k, Apple's model sets a new standard for general vision retrieval. Its near-perfect R@10 score (99.1%) and leading performance on Winoground compositional reasoning (12.8%) demonstrate superior understanding of both visual content and complex linguistic relationships.

**LAION-CLIP-H and MetaCLIP-H14 offer the best speed-accuracy tradeoff:** These open-source models deliver 86.0% and 85.7% R@1 respectively while maintaining high throughput (88 and 77 QPS). This combination makes them ideal for production deployments where both accuracy and speed matter. Their 1024d embeddings provide standard dimensionality across the industry.

**ColPali's multi-vector architecture creates a fundamental speed bottleneck:** At only 6.7 QPS—13x slower than LAION and 5x slower than Apple—ColPali's late-interaction mechanism processes ~1031 tokens per image instead of a single dense vector. While this architecture excels at document understanding tasks, it imposes significant computational overhead for general vision retrieval.

**SigLIP models show clear size-performance correlation:** SigLIP-400M (72.3% R@1, 43 QPS) substantially outperforms SigLIP-Base (34.7% R@1, 171 QPS). The base model's extreme speed comes at the cost of accuracy that makes it unsuitable for most retrieval applications, while the 400M variant offers a reasonable middle ground.

## Winoground compositional reasoning breakdown

Winoground tests fine-grained understanding of word order and object relationships. The full 400-sample test set reveals how models handle compositional challenges like "dog biting person" vs "person biting dog."

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

Apple's strong text score (35.2%) indicates superior language understanding, while all models struggle with the image component, highlighting a fundamental challenge in current vision-language models.

## Understanding the performance hierarchy

The results reveal distinct tiers based on model architecture, training scale, and design philosophy:

**Tier 1: Premium accuracy with moderate speed (Apple DFN5B)**
- Highest R@1 (88.9%) and MRR (0.929) establish new benchmarks
- 1024d embeddings provide rich semantic representation
- 33.5 QPS makes it viable for production at scale
- Best-in-class Winoground scores demonstrate sophisticated reasoning
- Ideal for: Applications where accuracy is paramount and latency budgets allow 30ms per query

**Tier 2: Balanced production models (LAION, MetaCLIP)**
- Near-top accuracy (86.0%, 85.7% R@1) with 2.5x faster throughput than Apple
- Open-source licenses enable customization and deployment flexibility
- 1024d standard embeddings ensure compatibility across pipelines
- Strong Winoground performance (10.2%, 9.8%) validates reasoning capabilities
- Ideal for: General-purpose retrieval systems, image search engines, production RAG

**Tier 3: Foundation models (OpenAI CLIP-L, Jina)**
- OpenAI CLIP remains competitive at 77.7% R@1 despite being an earlier architecture
- Jina's 768d embeddings reduce storage costs but underperform in accuracy
- Moderate throughput (53 QPS, 19 QPS) suitable for medium-scale deployments
- Ideal for: Baseline comparisons, cost-sensitive applications, fine-tuning bases

**Tier 4: Specialized architectures (ColPali)**
- Multi-vector design (1031 tokens) fundamentally different from dense models
- Extreme slowdown (6.7 QPS) limits applicability to batch processing scenarios
- Strong general vision performance (83.8% R@1) but not justified by speed penalty
- Ideal for: Document-specific tasks (OCR, PDF retrieval) where late interaction excels

**Tier 5: Speed-optimized models (SigLIP)**
- SigLIP-Base achieves 170 QPS but sacrifices too much accuracy (34.7% R@1)
- SigLIP-400M offers better balance (72.3% R@1, 43 QPS) but still trails competition
- Ideal for: Real-time applications with relaxed accuracy requirements, initial filtering stages

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
**Do NOT use ColPali for general vision tasks** despite its 83.8% R@1
- 13x slower than LAION (6.7 vs 88 QPS) creates unacceptable latency
- Multi-vector embeddings (1031 tokens) require specialized infrastructure
- Only justified when document-specific features (OCR, layout understanding) are critical

**For real-time applications with strict latency budgets (<10ms):**
Consider **SigLIP-400M** (72.3% R@1, 43 QPS) with accuracy monitoring
- SigLIP-Base too inaccurate (34.7%) for most retrieval applications
- Evaluate if 14-point accuracy drop vs LAION is acceptable for your use case

**Avoid in production:**
- **Jina CLIP-v1:** Slowest dense model (19 QPS) without corresponding accuracy gains
- **SigLIP-Base:** Extreme accuracy penalty (34.7%) not justified by speed

## Conclusion

This benchmark establishes a clear performance hierarchy among 2025's leading multimodal embedding models:

**Apple DFN5B sets the new accuracy standard** with 88.9% R@1 and 0.929 MRR, demonstrating that sophisticated data curation (DataComp filtering) and architectural refinement deliver measurable improvements over web-scale training alone. Its Winoground leadership (12.8% group score) particularly highlights advances in compositional reasoning.

**LAION-CLIP-H and MetaCLIP-H14 emerge as the optimal production choices**, delivering 86%+ R@1 accuracy at 2.5x the throughput of Apple. Their combination of open-source licensing, proven scalability, and balanced performance makes them ideal for most deployment scenarios. The minimal 3% accuracy gap from the leader rarely justifies Apple's speed penalty outside specialized applications.

**ColPali's multi-vector architecture proves unsuitable for general vision retrieval** despite respectable accuracy (83.8% R@1). Its 13x slowdown versus LAION and 5x versus Apple creates unacceptable production latency. The architecture should remain confined to document-specific tasks where late-interaction mechanisms provide unique value.

**The SigLIP family demonstrates clear scaling effects**: SigLIP-400M achieves usable accuracy (72.3%) at good speed (43 QPS), while SigLIP-Base's extreme efficiency (170 QPS) comes at the cost of production-unsuitable accuracy (34.7%). This reinforces that parameter count correlates strongly with retrieval quality in vision-language models.

For developers building multimodal applications in 2025: Start with **LAION-CLIP-H** or **MetaCLIP-H14** for general retrieval. Upgrade to **Apple DFN5B** only when accuracy requirements justify the speed tradeoff. Reserve **ColPali** exclusively for document-centric workflows. The era of one-size-fits-all embeddings has ended—success requires matching model architecture to task requirements.
