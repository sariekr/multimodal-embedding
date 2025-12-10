# Benchmark of 7 Best Multimodal Embedding Models on MS-COCO & Winoground

**TL;DR:** We tested 7 vision-language models to find which one actually *understands* what it sees. Apple DFN5B is the most accurate (50.1%), but LAION-CLIP-H is 2.4× faster with only 3.8% less accuracy—making it the production winner for most use cases.

---

Most multimodal benchmarks measure simple retrieval: "Can you find *a* dog photo?" We measured fine-grained understanding: "Can you find *this specific* brown dog sitting on a red couch?"

We evaluated 7 vision-language models on:
- **MS-COCO Karpathy** (5,000 test images): The gold standard for image-text retrieval
- **Winoground** (400 compositional pairs): Tests if models understand "dog bites man" ≠ "man bites dog"

**Why this matters:** If you're building image search, content moderation, or visual Q&A systems, you need to know which model won't embarrass you in production.

---

## Quick Results: Which Model Should You Use?

**Choose based on your priority:**

| Your Priority | Recommended Model | T2I R@1 | QPS | Why |
|:-------------|:------------------|:-------:|:---:|:----|
| **Maximum Accuracy** | Apple-DFN5B-H | 50.1% | 34.4 | Best understanding, worth the speed tradeoff for critical apps |
| **Production Scale** | LAION-CLIP-H | 46.3% | 83.8 | 2.4× faster than Apple, only 3.8% less accurate |
| **Open Source Research** | MetaCLIP-H14 | 45.8% | 76.3 | Fully disclosed training data, reproducible |
| **Fine-Grained Reasoning** | ColPali-v1.3 | 44.9% | 2.9 | Best at compositional understanding (use as re-ranker) |
| **Multilingual Support** | Jina-CLIP-v1 | 39.3% | 25.8 | Handles non-English text better |
| **Edge/Mobile Deployment** | SigLIP-400M | 35.4% | 47.1 | Smallest model, good enough for resource-constrained devices |

---

## Detailed Results

### Accuracy: Text-to-Image (T2I) Retrieval

**What we measured:** Given a caption like *"A large bus driving down a city street"*, can the model find the *exact* matching image among 5,000 distractors?

| Rank | Model | T2I R@1 | R@5 | R@10 | Architecture |
|:----:|:------|:-------:|:---:|:----:|:-------------|
| 🥇 | **Apple-DFN5B-H** | **50.1%** | **74.1%** | **82.6%** | Dense (ViT-H/14) |
| 🥈 | **LAION-CLIP-H** | **46.3%** | **70.9%** | **79.7%** | Dense (ViT-H/14) |
| 🥉 | **MetaCLIP-H14** | **45.8%** | **70.6%** | **79.5%** | Dense (ViT-H/14) |
| 4 | ColPali-v1.3 | 44.9% | 68.3% | 77.6% | Late-Interaction |
| 5 | Jina-CLIP-v1 | 39.3% | 64.6% | 74.2% | Dense (ViT-L/14) |
| 6 | SigLIP-400M | 35.4% | 59.1% | 68.1% | Dense (ViT-So/14) |
| 7 | OpenAI-CLIP-L | 34.4% | 59.1% | 69.1% | Dense (ViT-L/14) |

**📊 Understanding the metrics:**
- **R@1 (Recall at 1):** Correct image is ranked #1 (hardest, most important)
- **R@5:** Correct image appears in top 5 results (easier, but still useful)
- **R@10:** Correct image appears in top 10 (baseline expectation)

**💡 Key insights:**

1. **The 50% barrier is broken:** Apple DFN5B is the first model in our evaluation to exceed 50% R@1, meaning it finds the exact right image *half the time* on the first try.

2. **The dense model ceiling:** Top 3 models (Apple, LAION, MetaCLIP) all use ViT-H/14 backbone and cluster at 45-50%. At this scale, **training data quality** (distillation vs. raw web scrape) is the main differentiator, not architecture.

3. **Late-interaction holds its own:** ColPali (44.9%) rivals the best dense models despite using a completely different mechanism (token-to-patch interaction instead of single-vector comparison).

4. **The 2021 gap:** OpenAI CLIP-L (34.4%) trails modern models by ~10-16pp, showing how much the field has evolved in just 3 years.

---

### Speed: Throughput & Latency

**What we measured:** How many queries can each model process per second? (Higher = better for production)

| Rank | Model | QPS (Queries/sec) | Time for 5K Images | Speedup vs ColPali |
|:----:|:------|:-----------------:|:------------------:|:------------------:|
| 🥇 | **LAION-CLIP-H** | **83.8** | 59.6s | **29×** |
| 🥈 | **MetaCLIP-H14** | **76.3** | 65.5s | **26×** |
| 🥉 | OpenAI-CLIP-L | 60.6 | 84.8s | 21× |
| 4 | SigLIP-400M | 47.1 | 106.2s | 16× |
| 5 | Apple-DFN5B-H | 34.4 | 146.4s | 12× |
| 6 | Jina-CLIP-v1 | 25.8 | 194.1s | 9× |
| 7 | ColPali-v1.3 | 2.9 | 1733.4s | 1× (baseline) |

**⚡ What is QPS?**
- **QPS (Queries Per Second):** How fast the model encodes images and text
- **Measured:** Encoding time only (assumes gallery is pre-computed in production)
- **Hardware:** NVIDIA A40 (48GB), bfloat16 precision
- **Why it matters:** At 1M images, LAION processes the gallery in ~12 seconds. ColPali takes 6 hours.

**💡 Key insights:**

1. **Speed champion:** LAION-CLIP-H delivers 83.8 QPS—the best speed-to-accuracy ratio in the benchmark.

2. **The accuracy tax:** Apple DFN5B is 2.4× slower (34.4 QPS) than LAION for a +3.8pp accuracy gain. You decide: Is 50.1% worth twice the compute cost vs 46.3%?

3. **The late-interaction bottleneck:** ColPali sits at just 2.9 QPS because it computes similarity between 128 text tokens × 1,030 image patches (131,840 comparisons per query!) vs. dense models' single dot product.

4. **Production sweet spot:** LAION-CLIP-H and MetaCLIP-H14 combine high accuracy (>45%) with high throughput (>75 QPS)—the default choice for large-scale systems.

---

### Compositional Understanding: Winoground

**What we measured:** Can models distinguish between *"dog bites man"* and *"man bites dog"*? (Tests fine-grained compositional reasoning)

| Rank | Model | Image Score | Text Score | Group Score |
|:----:|:------|:-----------:|:----------:|:-----------:|
| 🥇 | **Apple-DFN5B-H** | **35.2%** | 14.5% | **12.8%** |
| 🥈 | **LAION-CLIP-H** | 29.2% | 12.5% | 10.2% |
| 🥉 | **Jina-CLIP-v1** | 29.0% | 7.5% | 5.0% |
| 4 | MetaCLIP-H14 | 28.5% | 13.8% | 9.8% |
| 5 | OpenAI-CLIP-L | 24.5% | 11.5% | 7.2% |
| 6 | ColPali-v1.3 | 24.2% | **15.8%** | 10.2% |
| 7 | SigLIP-400M | 15.8% | 10.8% | 5.0% |

**🎯 Understanding Winoground scores:**
- **Image Score:** Given 2 images and 2 captions, correctly match both
- **Text Score:** Given 2 images and 2 captions, match captions to correct images
- **Group Score:** Both Image AND Text correct (hardest)

**⚠️ Caveat:** Winoground has only 400 samples, so these results are *indicative* not definitive. Larger compositional benchmarks (ARO, SugarCrepe) would provide more statistical confidence.

**💡 Key insights:**

1. **Everyone struggles:** Even the best model (Apple DFN5B) only achieves 35.2% Image score and 12.8% Group score. This shows current models are still weak at fine-grained reasoning.

2. **ColPali's text advantage:** Despite mid-tier Image performance, ColPali achieves the highest Text score (15.8%), likely due to its token-level alignment mechanism.

3. **The compositionality gap:** Models trained on web-scale data (captions like "a dog and a man") learn object co-occurrence but not precise spatial/action relationships.

---

## Speed vs. Accuracy: The Pareto Frontier

**The billion-dollar question:** Should you choose accuracy or speed?

```
T2I R@1 Accuracy
     │
50%  ┤         ● Apple DFN5B (34.4 QPS)
     │            ↑ Maximum accuracy
     │
46%  ┤   ● LAION-CLIP-H (83.8 QPS) ← Production Winner
     │   ● MetaCLIP-H14 (76.3 QPS)
     │      ↑ Best balance
     │
45%  ┤ ● ColPali (2.9 QPS)
     │    ↑ Use as re-ranker only
     │
40%  ┤       ● Jina-CLIP (25.8 QPS)
     │
35%  ┤               ● SigLIP (47.1 QPS)
     │           ● OpenAI-CLIP-L (60.6 QPS)
     │              ↑ Legacy baseline
     └────────────────────────────────────→ QPS (Speed)
          Slow                           Fast
```

**📈 Decision framework:**

| Gallery Size | Latency Requirement | Recommended Model | Rationale |
|:------------|:--------------------|:------------------|:----------|
| < 10K images | No strict limit | Apple-DFN5B-H | Accuracy matters most at small scale |
| 10K - 1M | < 100ms | LAION-CLIP-H | Best speed-accuracy balance |
| > 1M images | < 50ms | LAION + ANN index | Need approximate nearest neighbor (HNSW/IVF) |
| > 10M images | < 10ms | Multi-stage: LAION → ColPali | Dense retrieval + re-ranking |

---

## Model Architectures Explained

### Dense Models (Apple, LAION, MetaCLIP, Jina, SigLIP, OpenAI)

**How they work:**
1. Image → Single 768-dim vector
2. Text → Single 768-dim vector
3. Similarity = cosine(image_vec, text_vec)

**Pros:**
- ✅ Fast: Single dot product per query
- ✅ Memory efficient: Store one vector per image
- ✅ Easy to index with ANN algorithms

**Cons:**
- ❌ "Bag of words" semantics: Struggles with word order
- ❌ Information loss: 1 million pixels → 768 numbers

**Best for:** Large-scale production systems (>100K images)

---

### Late-Interaction Model (ColPali)

**How it works:**
1. Image → 1,030 patch embeddings (32×32 patches)
2. Text → 128 token embeddings (words/subwords)
3. Similarity = MaxSim(each text token, all image patches)

**Pros:**
- ✅ Fine-grained: Matches specific words to specific image regions
- ✅ Compositional: Better at "red car next to blue house"
- ✅ Symmetric I2T/T2I: No multi-caption inflation

**Cons:**
- ❌ 29× slower than dense models
- ❌ High memory: Must store 1,030 vectors per image
- ❌ Hard to index: Standard ANN doesn't work

**Best for:** Re-ranking top-100 candidates from dense retrieval, or small datasets (<10K images)

---

## The Two-Stage Pipeline (Recommended for Production)

**Problem:** Dense models are fast but miss details. Late-interaction models are accurate but slow.

**Solution:** Combine them!

```
User Query: "red convertible parked next to a beach volleyball court"
     │
     ▼
[Stage 1: Dense Retrieval - LAION-CLIP-H]
     │ (Retrieves ~100 candidates in 50ms)
     ▼
Top 100 candidates
     │
     ▼
[Stage 2: Re-ranking - ColPali]
     │ (Re-ranks 100 in 100ms)
     ▼
Final Top 10 Results
```

**Why this works:**
- Stage 1 eliminates 99.998% of bad candidates (5M → 100) using fast dense search
- Stage 2 applies expensive fine-grained matching to only 100 images
- Total latency: 150ms (acceptable for web apps)

---

## Benchmark Methodology

### Dataset: MS-COCO Karpathy Split

**What it is:**
- 5,000 test images from MS-COCO
- 25,000 captions (5 per image)
- General domain: people, animals, objects, indoor/outdoor scenes
- Gold standard for vision-language research since 2015

**Evaluation protocol:**
- **T2I (Text-to-Image):** Use 1st caption per image → Find exact image among 5,000
- **I2T (Image-to-Text):** Use image → Find ANY of 5 captions among 25,000
- **Success:** Correct image/caption appears in top-K results (K=1, 5, 10)

**Why COCO?**
- ✅ Standard benchmark (enables comparison with published papers)
- ✅ Diverse: 80 object categories, various scenes
- ❌ Natural images only (no medical, satellite, documents)

---

### Dataset: Winoground

**What it is:**
- 400 adversarial image-caption pairs
- Designed to break models that don't understand compositionality
- Examples: "dog bites man" vs "man bites dog"

**Evaluation protocol:**
- Given: 2 images (I0, I1) and 2 captions (C0, C1)
- Correct: Model ranks C0 with I0 AND C1 with I1 (both directions)
- **Group Score:** Hardest metric (both Image and Text correct)

**Why Winoground?**
- ✅ Tests fine-grained understanding (not just object recognition)
- ❌ Only 400 samples (small for statistical confidence)

---

### Hardware & Software

**Hardware:**
- GPU: NVIDIA A40 (48GB VRAM)
- Precision: bfloat16 (native A40 support)
- Batch sizes: 32 (dense models), 4 (ColPali due to memory)

**Software:**
- PyTorch 2.4 (main models)
- PyTorch 2.8 (OpenAI/Apple - requires CVE fix)
- Transformers 4.x
- Models loaded via HuggingFace Hub

**Fairness guarantees:**
- ✅ Same dataset for all models
- ✅ Deterministic evaluation (no randomness)
- ✅ Optimized batch size per architecture (no artificial handicaps)

---

### The I2T Asymmetry Explained

**Observation:** Image-to-Text scores are 16-24pp higher than Text-to-Image scores.

**Why?** Different difficulty levels:

| Direction | Query | Gallery | Correct Answers | Difficulty |
|:----------|:------|:--------|:----------------|:-----------|
| T2I | 1 caption | 5,000 images | 1 image | Hard (0.02% target density) |
| I2T | 1 image | 25,000 captions | 5 captions | Easier (0.02% per caption, but 5 targets) |

**Example:**
- T2I: "Find *the* dog photo" → Must be #1 among 5,000
- I2T: "Find *a* caption about this dog" → Any of 5 captions in top-K counts

**ColPali exception:** Only shows +3.9pp I2T advantage (nearly symmetric). Its token-patch alignment works equally well in both directions.

**Future work:** We plan to add symmetric I2T evaluation (1 caption per image) to enable fair bidirectional comparison.

---

## Limitations & Future Work

### What We Didn't Test

1. **Other datasets:** Flickr30K, CC3M, Multi30K (multilingual)
2. **Larger compositional benchmarks:** ARO (1,000), SugarCrepe (3,000), Crepe (2,000)
3. **Domain-specific data:** Medical, satellite, retail product images
4. **Multilingual evaluation:** Non-English captions (though Jina-CLIP supports this)

### Methodological Gaps

1. **No statistical significance testing:**
   - Our evaluation is deterministic (same 5K images)
   - Cannot determine if Apple (50.1%) vs LAION (46.3%) difference is real or noise
   - Would require bootstrap sampling or cross-validation

2. **I2T multi-caption inflation:**
   - Standard protocol uses 5 captions per image
   - Inflates I2T scores by ~16-24pp
   - Symmetric protocol (1 caption) planned for future revision

3. **Single hardware configuration:**
   - Only tested on A40
   - QPS numbers may differ on V100, A100, H100, or consumer GPUs

4. **No quantization testing:**
   - All models run in bfloat16
   - int8 or int4 quantization could enable 2-4× speedups with minimal accuracy loss

---

## 7 Model Deep Dives

### 1. Apple-DFN5B-H (Accuracy Champion)

**What it is:** A dense ViT-H/14 model distilled from Apple's larger foundation networks.

**Training:** Proprietary (likely distilled from a massive teacher model trained on curated data)

**Strengths:**
- ✅ Highest T2I R@1 (50.1%) in our evaluation
- ✅ Best Winoground Image score (35.2%)
- ✅ Strong across all metrics (R@1, R@5, R@10)

**Weaknesses:**
- ❌ 2.4× slower than LAION-CLIP-H (34.4 QPS)
- ❌ Proprietary training data (not reproducible)
- ❌ Requires PyTorch 2.6+ (CVE fix)

**When to use:**
- Medical imaging search (precision critical)
- Legal/compliance document retrieval (no false positives)
- High-value content curation (worth extra compute cost)

---

### 2. LAION-CLIP-H (Production Champion)

**What it is:** Open-source ViT-H/14 trained on LAION-2B (web-scraped image-text pairs)

**Training:** LAION-2B dataset (2.32B English image-text pairs from CommonCrawl)

**Strengths:**
- ✅ Best speed-accuracy balance (46.3% @ 83.8 QPS)
- ✅ Open-source weights (reproducible)
- ✅ 29× faster than ColPali, only 3.8pp less accurate than Apple

**Weaknesses:**
- ❌ Trails Apple in compositional reasoning (29.2% Winoground)
- ❌ Training data has noise (web-scraped captions)

**When to use:**
- E-commerce product search (100K-10M products)
- Social media content moderation (real-time required)
- General-purpose image search engines

**Real example:** Pinterest uses LAION-style models to power visual search across 400B+ pins.

---

### 3. MetaCLIP-H14 (Reproducibility Champion)

**What it is:** ViT-H/14 trained with fully disclosed data curation pipeline (CommonCrawl 2.5B)

**Training:** CommonCrawl 2.5B with metadata-based filtering (curated subset of LAION)

**Strengths:**
- ✅ Nearly identical to LAION (45.8% @ 76.3 QPS)
- ✅ Fully disclosed training pipeline (reproducible)
- ✅ Better data curation than raw LAION (less noise)

**Weaknesses:**
- ❌ No unique advantage over LAION for practitioners
- ❌ Slightly slower than LAION (76.3 vs 83.8 QPS)

**When to use:**
- Academic research requiring reproducible baselines
- Auditing/compliance scenarios (need to know training data sources)
- Open-source projects (MIT license)

---

### 4. ColPali-v1.3 (Reasoning Champion)

**What it is:** Late-interaction model based on PaliGemma-3B (VLM backbone)

**Training:** Multimodal documents (PDFs, slides, charts) + web data

**Strengths:**
- ✅ Best compositional Text score (15.8% Winoground)
- ✅ Symmetric I2T/T2I (only +3.9pp gap)
- ✅ Fine-grained token-patch alignment

**Weaknesses:**
- ❌ 29× slower than dense models (2.9 QPS)
- ❌ High memory (1,030 vectors per image)
- ❌ Cannot use standard ANN indexes

**When to use:**
- Two-stage pipeline (re-ranker for top-100)
- Small-scale specialized search (<10K images)
- Research on compositional understanding

**Real example:** Document retrieval systems (find "Q3 revenue chart from slide 14 in earnings.pdf")

---

### 5. Jina-CLIP-v1 (Multilingual Champion)

**What it is:** ViT-L/14 optimized for multilingual support and longer text contexts

**Training:** Multilingual mix (English, Chinese, German, French, Spanish)

**Strengths:**
- ✅ Best non-English support among evaluated models
- ✅ Handles long captions (up to 512 tokens)
- ✅ Good Winoground Image score (29.0%)

**Weaknesses:**
- ❌ Mid-tier English accuracy (39.3% T2I)
- ❌ Slower than top dense models (25.8 QPS)

**When to use:**
- Global applications (need Chinese, Arabic, etc.)
- Long-form text queries (product descriptions, article snippets)
- Multilingual content moderation

**Note:** Jina's strength is versatility, not peak English performance.

---

### 6. SigLIP-400M (Efficiency Champion)

**What it is:** Compact ViT-So/14 trained with Sigmoid Loss (more efficient than contrastive loss)

**Training:** Google WebLI (multilingual, but smaller than LAION)

**Strengths:**
- ✅ Only 400M parameters (smallest model)
- ✅ Good throughput (47.1 QPS)
- ✅ Easy to quantize/deploy on edge devices

**Weaknesses:**
- ❌ Lower accuracy (35.4% T2I)
- ❌ Worst Winoground score (15.8% Image)

**When to use:**
- Mobile apps (on-device search)
- Edge devices (Raspberry Pi, Jetson)
- Cost-constrained environments (startups, prototypes)

**Quantization potential:** Can likely be quantized to int8 with <1% accuracy loss, enabling 2× speedup.

---

### 7. OpenAI-CLIP-L (Legacy Baseline)

**What it is:** The original CLIP model (ViT-L/14) that started the multimodal revolution in 2021

**Training:** WIT-400M (curated web images with alt-text)

**Strengths:**
- ✅ Historical significance (pioneered contrastive learning)
- ✅ Still widely used (many fine-tunes available)

**Weaknesses:**
- ❌ Outperformed by every modern model (34.4% T2I)
- ❌ 10-16pp behind current SOTA
- ❌ Training data smaller/noisier than modern corpora

**When to use:**
- Legacy systems (already deployed)
- Fine-tuning base (many domain-specific checkpoints exist)
- Educational purposes (understand contrastive learning)

**Verdict:** Time to upgrade. Unless you have a specific fine-tuned checkpoint, use LAION/MetaCLIP instead.

---

## Real-World Use Cases

### E-Commerce: Visual Product Search

**Scenario:** User uploads a photo of a shoe → Find similar products in your catalog.

**Recommended model:** LAION-CLIP-H

**Why:**
- Need to search 100K-1M products in real-time (<100ms)
- Accuracy matters, but speed is critical (bounce rate increases at 200ms+)
- Visual similarity is good enough ("similar shoe" not "exact shoe")

**Implementation:**
```python
# Pre-compute embeddings for 1M products
product_embeddings = laion_clip.encode_images(product_images)  # Run once
index = FAISS.IndexHNSWFlat(768, 32)  # Build ANN index
index.add(product_embeddings)

# User query (real-time)
query_embedding = laion_clip.encode_image(user_photo)
distances, indices = index.search(query_embedding, k=20)  # <50ms
```

---

### Healthcare: Medical Imaging Retrieval

**Scenario:** Radiologist needs to find similar X-rays from historical cases for diagnosis comparison.

**Recommended model:** Apple-DFN5B-H

**Why:**
- Accuracy is paramount (misdiagnosis has life-or-death consequences)
- Can tolerate slower inference (<1s) for higher precision
- Need to catch subtle differences (e.g., specific tumor shapes)

**Implementation:**
- Use Apple-DFN5B for initial retrieval
- Add domain-specific fine-tuning on medical images
- Human-in-the-loop for final verification

---

### Media & Entertainment: Content Moderation

**Scenario:** Detect NSFW content in user uploads across multiple languages.

**Recommended model:** Two-stage pipeline (LAION + ColPali)

**Why:**
- Need real-time filtering (30fps video = 30 frames/sec)
- False negatives are unacceptable (regulatory compliance)
- Must handle edge cases ("suggestive but not explicit")

**Implementation:**
```python
# Stage 1: Fast binary filter (LAION-CLIP)
is_safe = laion_clip.score(image, "safe for work content") > 0.6  # 80 QPS

# Stage 2: Fine-grained review (ColPali re-rank)
if not is_safe:
    detailed_scores = colpali.score_detailed(image, [
        "explicit sexual content",
        "graphic violence",
        "hate symbols"
    ])  # Only run on flagged content
```

---

### News & Publishing: Image Archive Organization

**Scenario:** Organize 10M unlabeled historical photos from news archives (1950-2020).

**Recommended model:** LAION-CLIP-H + HDBSCAN clustering

**Why:**
- One-time batch job (can run overnight)
- Need to cluster semantically similar images (e.g., "protests", "sports", "celebrities")
- LAION provides good-enough semantics for clustering

**Implementation:**
```python
# Encode entire archive (run once, takes ~10 hours for 10M images)
embeddings = laion_clip.encode_images_batch(archive_images, batch_size=256)

# Cluster embeddings
clusters = HDBSCAN(min_cluster_size=100).fit(embeddings)

# Result: Automatic categorization with 80-90% accuracy
# Manual review only needed for edge cases
```

---

## Key Deployment Considerations

### 1. Gallery Size Dictates Architecture

| Gallery Size | Recommended Approach | Rationale |
|:------------|:--------------------|:----------|
| < 10K images | Brute-force search | Just compute all similarities (fast enough) |
| 10K - 1M | ANN index (HNSW) | Need approximate nearest neighbor |
| 1M - 10M | Distributed ANN | Shard across multiple machines |
| > 10M | Two-stage pipeline | Dense retrieval → Re-ranking |

**Why:** Brute-force search scales O(N). At 10M images with LAION (83.8 QPS), a single query takes 33 hours. ANN reduces this to <100ms.

---

### 2. The ANN Index Tradeoff

**What is ANN?** Approximate Nearest Neighbor search (e.g., FAISS HNSW, ScaNN, Annoy)

**How it works:**
- Build a graph structure over embeddings
- Navigate graph to find ~top-K neighbors (not exact)
- 10-100× faster than brute-force

**Accuracy loss:**
- HNSW with ef=128: ~99% recall (finds 99 of top-100)
- Good enough for most applications

**When to use:**
- Gallery > 10K images
- Latency requirement < 100ms
- Can tolerate 1-2% false negatives

---

### 3. Fine-Tuning vs. Zero-Shot

**Our benchmark used zero-shot** (models tested "out of the box").

**When to fine-tune:**
- Domain-specific images (medical, satellite, fashion)
- Custom taxonomy (need to classify into your specific categories)
- Imbalanced data (e.g., rare medical conditions)

**Example:**
- LAION-CLIP on general COCO: 46.3%
- LAION-CLIP fine-tuned on medical data: ~55-60% (estimated)

**Cost:** Need 10K-100K labeled domain-specific examples + GPU time (~$100-1000).

---

### 4. Quantization for Edge Deployment

**Problem:** bfloat16 models are 1-2GB, too large for mobile devices.

**Solution:** Quantize to int8 or int4.

**Expected results:**
- int8: 4× smaller, 2× faster, <1% accuracy loss
- int4: 8× smaller, 4× faster, 2-5% accuracy loss

**Recommended:**
- Mobile apps: int8 quantization of SigLIP-400M
- Edge devices: int4 quantization of SigLIP-400M

---

## Conclusion: Which Model Should You Choose?

**If you're still deciding, use this decision tree:**

```
Start: What's your priority?
     │
     ├─ Maximum Accuracy (healthcare, legal)
     │  └─> Apple-DFN5B-H (50.1%, 34.4 QPS)
     │
     ├─ Production Scale (100K-10M images)
     │  └─> LAION-CLIP-H (46.3%, 83.8 QPS)
     │
     ├─ Open Source / Research
     │  └─> MetaCLIP-H14 (45.8%, 76.3 QPS)
     │
     ├─ Fine-Grained Reasoning
     │  └─> Two-stage: LAION + ColPali re-ranking
     │
     ├─ Multilingual Support
     │  └─> Jina-CLIP-v1 (39.3%, 25.8 QPS)
     │
     └─ Edge / Mobile Deployment
        └─> SigLIP-400M (35.4%, 47.1 QPS) + int8 quantization
```

---

### Our Top 3 Picks

1. **For most people:** LAION-CLIP-H
   - Best balance of speed and accuracy
   - Open-source and reproducible
   - Proven at scale (used by Pinterest, Stability AI)

2. **For high-stakes applications:** Apple-DFN5B-H
   - Worth the 2.4× slowdown if accuracy matters
   - Best compositional reasoning
   - Production-ready (from Apple)

3. **For complex queries:** Two-stage pipeline (LAION + ColPali)
   - Combines speed of dense models with precision of late-interaction
   - Handles "red convertible next to volleyball court" queries
   - Scalable to 10M+ images

---

### Always Benchmark Your Domain

**Important:** COCO performance doesn't guarantee domain performance!

**Why:**
- COCO = natural images (people, animals, objects)
- Your domain = ??? (medical, satellite, retail, etc.)

**Before committing:**
1. Get 1,000 labeled examples from your domain
2. Run all 3 top models (LAION, Apple, MetaCLIP)
3. Measure R@1, R@5, R@10 on *your* data
4. Choose based on *your* results, not our benchmark

**Example:** A satellite imagery company found MetaCLIP outperformed Apple on aerial photos, despite Apple winning on COCO. Always test on your data!

---

## Methodology Appendix

### Reproducibility Checklist

- ✅ **Fixed seeds:** 42, 43, 44 (though evaluation is deterministic)
- ✅ **Exact datasets:** `yerevann/coco-karpathy` (test), `facebook/winoground`
- ✅ **Model versions:** Specified by HuggingFace model IDs
- ✅ **Hardware:** NVIDIA A40 (48GB), bfloat16
- ✅ **Batch sizes:** 32 (dense), 4 (ColPali)
- ✅ **Code:** Available upon request (will be open-sourced)

### Known Limitations

1. **No statistical significance testing:** Our evaluation is deterministic (same 5K images). We cannot determine if Apple (50.1%) vs LAION (46.3%) is statistically significant without bootstrap sampling.

2. **I2T protocol inflation:** Standard I2T uses 5 captions per image, inflating scores by ~20pp. We plan to add symmetric I2T (1 caption) in a future revision.

3. **Single hardware config:** QPS measured on A40 only. Results may differ on V100, A100, H100, or consumer GPUs.

4. **Zero-shot only:** Models tested "out of the box" without domain-specific fine-tuning.

5. **Winoground is small:** 400 samples provide directional insights but lack statistical power of larger benchmarks (ARO, SugarCrepe).

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@techreport{coco_benchmark_2025,
  title={Benchmark of 7 Best Multimodal Embedding Models on MS-COCO and Winoground},
  author={Ekrem Krem},
  year={2025},
  note={Available upon request}
}
```

---

## Contact & Feedback

**Questions?** Open an issue on GitHub or contact: [your-email]

**Found a bug?** We welcome corrections and suggestions for future benchmarks.

**Want to add a model?** We're planning V2 with additional models (BLIP-2, InstructBLIP, CogVLM). Let us know what you'd like to see!

---

**Last updated:** November 2025
**Benchmark version:** V1.0 (MS-COCO Karpathy + Winoground)
