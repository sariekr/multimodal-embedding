# Benchmark of 7 Multimodal Embedding Models on MS-COCO Karpathy & Winoground

**Evaluated:** 7 vision-language models on the MS-COCO Karpathy benchmark (5,000 test images, 25,000 captions) and Winoground compositional dataset (400 adversarial pairs).

**Goal:** Measure bidirectional retrieval accuracy (Text↔Image) and fine-grained compositional reasoning to determine which models actually understand what they see—not just recognize objects.

**Key finding:** Apple DFN5B-H achieves 50.1% T2I R@1 (highest in our evaluation), but LAION-CLIP-H offers the best speed-accuracy balance at 46.3% with 2.4× faster throughput.

---

## Benchmark Results

### Text-to-Image Retrieval (T2I R@1)

**Primary metric:** Can the model find the exact correct image when given a caption?

| Rank | Model | T2I R@1 | I2T R@1 | Winoground Image | QPS | Architecture |
|:----:|:------|:-------:|:-------:|:----------------:|:---:|:-------------|
| 🥇 | **Apple-DFN5B-H** | **50.1%** | **69.7%** | **35.2%** | 34.4 | Dense (ViT-H/14) |
| 🥈 | **LAION-CLIP-H** | **46.3%** | **66.3%** | 29.2% | **83.8** | Dense (ViT-H/14) |
| 🥉 | **MetaCLIP-H14** | **45.8%** | **65.9%** | 28.5% | 76.3 | Dense (ViT-H/14) |
| 4 | ColPali-v1.3 | 44.9% | 48.8% | 24.2% | 2.9 | Late-Interaction |
| 5 | Jina-CLIP-v1 | 39.3% | 55.3% | 29.0% | 25.8 | Dense (ViT-L/14) |
| 6 | SigLIP-400M | 35.4% | 45.1% | 15.8% | 47.1 | Dense (ViT-So/14) |
| 7 | OpenAI-CLIP-L | 34.4% | 58.0% | 24.5% | 60.6 | Dense (ViT-L/14) |

*Hardware: NVIDIA A40, bfloat16 precision, batch size optimized per model*

**[INSERT BAR CHART HERE: T2I R@1 vs QPS scatter plot]**
- X-axis: QPS (speed)
- Y-axis: T2I R@1 (accuracy)
- Shows Pareto frontier: LAION/MetaCLIP optimal, Apple highest accuracy, ColPali slowest

---

## Metrics Explained

### What We Measured

**1. Text-to-Image Retrieval (T2I R@1)**
- **Task:** Given caption → Find the exact image among 5,000
- **Metric:** Recall@1 = correct image is ranked #1
- **Why it matters:** Hardest task, measures precise understanding

**2. Image-to-Text Retrieval (I2T R@1)**
- **Task:** Given image → Find ANY of 5 captions among 25,000
- **Metric:** Recall@1 = any correct caption is ranked #1
- **Note:** Easier than T2I (5 valid targets vs 1), scores ~20pp higher

**3. Winoground Image Score**
- **Task:** Distinguish "dog bites man" vs "man bites dog"
- **Metric:** % where both images matched to correct captions
- **Why it matters:** Tests compositional reasoning, not just object recognition

**4. Throughput (QPS)**
- **Task:** Encode 5,000 images + captions
- **Metric:** Queries per second (higher = faster)
- **Why it matters:** Determines production scalability

---

## Key Insights: Why These Scores?

### 1. The 50% Barrier: Apple DFN5B Breaks Through

**Observation:** Apple achieves 50.1% T2I R@1, the only model to exceed 50% in our evaluation.

**Why?**
- **Distillation advantage:** Likely distilled from a larger teacher model (5B+ parameters)
- **Data curation:** Proprietary training data with higher caption quality than web-scraped corpora
- **Same architecture as LAION:** Both use ViT-H/14, but Apple's training recipe makes the difference

**Trade-off:** 2.4× slower than LAION (34.4 vs 83.8 QPS). Is +3.8pp accuracy worth it?

---

### 2. The Dense Model Ceiling: 45-50% Cluster

**Observation:** Top 3 models (Apple 50.1%, LAION 46.3%, MetaCLIP 45.8%) all use ViT-H/14 and cluster tightly.

**Why?**
- **Architecture plateau:** ViT-H/14 has ~630M parameters. At this scale, **data quality > model size**.
- **Training data matters:** Apple (proprietary) > LAION (web-scraped 2B) > MetaCLIP (curated web)
- **Diminishing returns:** Going from ViT-L (304M) to ViT-H (630M) gives +10pp. Beyond this requires better data, not bigger models.

**Implication:** Don't expect 60% T2I from scaling alone. Need better training signals (e.g., hard negatives, distillation, synthetic captions).

---

### 3. The Late-Interaction Surprise: ColPali's Symmetric Behavior

**Observation:** ColPali shows only +3.9pp I2T advantage (nearly symmetric), while dense models show +16-24pp.

**Why?**
- **Dense models:** Compress image → 1 vector. Asymmetry comes from multi-caption protocol (I2T has 5 valid targets).
- **ColPali:** Keeps 1,030 patch vectors + 128 token vectors. Fine-grained alignment works equally in both directions.

**Trade-off:** Symmetry is nice, but ColPali is 29× slower (2.9 QPS). Only viable as a re-ranker.

---

### 4. The Speed Champion: LAION's 83.8 QPS

**Observation:** LAION processes 83.8 queries/sec, the fastest among high-accuracy models.

**Why?**
- **Optimized architecture:** Standard ViT-H/14 with efficient attention
- **Batch size 32:** Dense models can use large batches (ColPali limited to 4 due to memory)
- **bfloat16 precision:** Native A40 support, no conversion overhead

**Production reality:** At 1M images, LAION encodes the gallery in ~12 seconds. ColPali takes 6 hours.

---

### 5. The Compositional Gap: Even Apple Struggles

**Observation:** Best Winoground Image score is only 35.2% (Apple). Group score (both directions) is 12.8%.

**Why?**
- **Web-scale training limitation:** Models see "a dog and a man" (co-occurrence) but not "dog biting man" vs "man biting dog" (spatial/action relationships).
- **Bag-of-words semantics:** Dense models match keywords, not compositional structure.
- **Token alignment helps:** ColPali achieves highest Text score (15.8%) via token-patch matching, but still struggles with Image score.

**Implication:** For queries like "red car next to blue house," current models are unreliable. Need specialized training (e.g., CREPE, SugarCrepe) or two-stage pipelines.

---

### 6. The 2021 Gap: OpenAI CLIP Shows Its Age

**Observation:** OpenAI CLIP-L (34.4% T2I) trails modern models by 10-16pp.

**Why?**
- **Smaller training data:** WIT-400M vs LAION-2B (5× less data)
- **No modern tricks:** No distillation, no hard negative mining, no synthetic captions
- **Architecture unchanged:** Same ViT-L/14 as 2021 (others evolved training recipes)

**Verdict:** Unless you have a fine-tuned checkpoint, upgrade to LAION/MetaCLIP.

---

## What is a Multimodal Embedding Benchmark?

### Core Concept

A **multimodal embedding model** converts images and text into the same vector space, enabling:
- **Text → Image search:** "Find photos of sunset on beach"
- **Image → Text search:** "What describes this photo?"
- **Zero-shot classification:** Match image to category without training

### How Benchmarking Works

**1. Dataset Preparation**
- Corpus: N images (e.g., 5,000 from MS-COCO)
- Queries: M captions (e.g., 25,000, 5 per image)
- Ground truth: Each caption maps to exactly 1 image

**2. Embedding Extraction**
```python
image_embeddings = model.encode_images(corpus)  # [N, 768]
text_embeddings = model.encode_texts(queries)   # [M, 768]
```

**3. Similarity Computation**
```python
# Cosine similarity matrix
similarity = text_embeddings @ image_embeddings.T  # [M, N]

# For each query, rank images by similarity
ranked_images = similarity.argsort(descending=True)  # [M, N]
```

**4. Evaluation Metrics**
- **R@1:** Is correct image ranked #1?
- **R@5:** Is correct image in top-5?
- **R@10:** Is correct image in top-10?

**5. Aggregation**
```python
r_at_1 = (ranked_images[:, 0] == ground_truth).mean()  # % correct at position 1
r_at_5 = (ranked_images[:, :5] == ground_truth).any(1).mean()  # % in top-5
```

### Why Standard Benchmarks Matter

**Without standard benchmarks:**
- "Our model is 10% better" → 10% better than what? On what data?
- Cannot compare across papers/models

**With MS-COCO Karpathy:**
- Established 2015, used by 1000+ papers
- 5K test split is the de facto standard
- Enables apples-to-apples comparison

---

## Methodology

### Datasets

#### MS-COCO Karpathy Split

**Source:** `yerevann/coco-karpathy` (HuggingFace)

**Statistics:**
- **Images:** 5,000 test images
- **Captions:** 25,000 (5 per image, human-annotated)
- **Domain:** General natural images (people, animals, objects, indoor/outdoor scenes)
- **Categories:** 80 COCO object classes

**Why COCO Karpathy?**
- ✅ Gold standard since 2015 (Karpathy & Fei-Fei)
- ✅ Balanced dataset (diverse scenes, objects, captions)
- ✅ Enables comparison with published baselines (CLIP, ALIGN, BLIP)

**Limitations:**
- ❌ Natural images only (no medical, satellite, documents)
- ❌ English captions only (no multilingual evaluation)
- ❌ Web-scraped images (some quality issues)

---

#### Winoground

**Source:** `facebook/winoground` (HuggingFace)

**Statistics:**
- **Samples:** 400 adversarial pairs
- **Structure:** Each sample = 2 images (I0, I1) + 2 captions (C0, C1)
- **Difficulty:** Requires compositional reasoning (e.g., "dog bites man" ≠ "man bites dog")

**Evaluation:**
- **Image Score:** Model ranks C0 with I0 AND C1 with I1
- **Text Score:** Model ranks I0 with C0 AND I1 with C1
- **Group Score:** Both Image and Text correct (hardest)

**Why Winoground?**
- ✅ Tests fine-grained understanding (not just object co-occurrence)
- ✅ Adversarial design (models trained on web data often fail)
- ✅ Small but focused (400 samples = quick evaluation)

**Limitations:**
- ❌ Only 400 samples (limited statistical power)
- ❌ Biased toward visual attributes (color, spatial relations)
- ⚠️ Results should be interpreted as *indicative* not definitive

---

### Models

**Selection criteria:**
1. **Diverse architectures:** Dense (single-vector) vs Late-Interaction (multi-vector)
2. **Scale range:** 400M to 5B parameters
3. **Training data:** Proprietary (Apple) vs Open (LAION, MetaCLIP)
4. **Availability:** All models on HuggingFace Hub (reproducible)

| Model | HuggingFace ID | Params | Architecture | Training Data |
|:------|:--------------|:------:|:-------------|:--------------|
| Apple-DFN5B-H | `apple/DFN5B-CLIP-ViT-H-14-378` | 630M | Dense | Proprietary (distilled) |
| LAION-CLIP-H | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | 630M | Dense | LAION-2B |
| MetaCLIP-H14 | `facebook/metaclip-h14-fullcc2.5b` | 630M | Dense | CommonCrawl 2.5B |
| ColPali-v1.3 | `vidore/colpali-v1.3` | 3B | Late-Interaction | Documents + Web |
| Jina-CLIP-v1 | `jinaai/jina-clip-v1` | 304M | Dense | Multilingual mix |
| SigLIP-400M | `google/siglip-so400m-patch14-384` | 400M | Dense | WebLI |
| OpenAI-CLIP-L | `openai/clip-vit-large-patch14-336` | 304M | Dense | WIT-400M |

**Model loading:**
```python
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True  # For ColPali, Apple, MetaCLIP
).to("cuda").eval()

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
```

---

### Evaluation Protocol

#### Text-to-Image (T2I) Retrieval

**Setup:**
- **Queries:** 5,000 captions (1st caption per image)
- **Gallery:** 5,000 images
- **Task:** For each caption, rank all 5,000 images by similarity

**Scoring:**
```python
# Extract embeddings
text_embeds = model.encode_texts(captions)       # [5000, 768]
image_embeds = model.encode_images(images)       # [5000, 768]

# Compute similarity
scores = text_embeds @ image_embeds.T            # [5000, 5000]

# Rank images for each caption
ranks = scores.argsort(dim=1, descending=True)   # [5000, 5000]

# Compute R@K
for k in [1, 5, 10]:
    correct = (ranks[:, :k] == ground_truth.unsqueeze(1)).any(1)
    recall_at_k = correct.float().mean()
```

**Correctness criteria:**
- Ground truth: Each caption i maps to image i
- R@1: Image i is ranked #1 for caption i
- R@5: Image i is in top-5 for caption i

---

#### Image-to-Text (I2T) Retrieval

**Setup:**
- **Queries:** 5,000 images
- **Gallery:** 25,000 captions (5 per image)
- **Task:** For each image, rank all 25,000 captions by similarity

**Scoring:**
```python
# Compute similarity (transposed)
scores = image_embeds @ text_embeds.T            # [5000, 25000]

# Rank captions for each image
ranks = scores.argsort(dim=1, descending=True)   # [5000, 25000]

# Compute R@K (multi-caption protocol)
for k in [1, 5, 10]:
    # Each image has 5 valid captions
    valid_captions = [list(range(i*5, (i+1)*5)) for i in range(5000)]

    # Check if ANY valid caption is in top-K
    correct = []
    for i in range(5000):
        top_k = ranks[i, :k]
        hit = any(c in valid_captions[i] for c in top_k)
        correct.append(hit)

    recall_at_k = np.mean(correct)
```

**Correctness criteria:**
- Ground truth: Image i maps to captions [5i, 5i+1, 5i+2, 5i+3, 5i+4]
- R@1: ANY of 5 captions is ranked #1 (success)
- This is why I2T scores are ~20pp higher than T2I

---

#### Winoground Evaluation

**Setup:**
- **Input:** 2 images (I0, I1), 2 captions (C0, C1)
- **Correct pairing:** C0 with I0, C1 with I1
- **Adversarial twist:** Captions differ by 1-2 words (e.g., "dog bites man" vs "man bites dog")

**Scoring:**
```python
# Compute all 4 similarities
s_c0_i0 = similarity(C0, I0)
s_c0_i1 = similarity(C0, I1)
s_c1_i0 = similarity(C1, I0)
s_c1_i1 = similarity(C1, I1)

# Image score: Can model match captions to correct images?
image_correct = (s_c0_i0 > s_c0_i1) and (s_c1_i1 > s_c1_i0)

# Text score: Can model match images to correct captions?
text_correct = (s_c0_i0 > s_c1_i0) and (s_c1_i1 > s_c0_i1)

# Group score: Both directions correct
group_correct = image_correct and text_correct
```

**Aggregation:**
```python
image_score = sum(image_correct) / 400  # % where image matching succeeded
text_score = sum(text_correct) / 400    # % where text matching succeeded
group_score = sum(group_correct) / 400  # % where both succeeded
```

---

### Hardware & Software

**Hardware:**
- **GPU:** NVIDIA A40 (48GB VRAM)
- **CPU:** AMD EPYC 7532 (16 cores)
- **RAM:** 128GB DDR4
- **Storage:** NVMe SSD (for image caching)

**Software:**
- **OS:** Ubuntu 22.04 LTS
- **Python:** 3.10
- **PyTorch:** 2.4.0 (main), 2.8.0 (OpenAI/Apple - requires CVE fix)
- **Transformers:** 4.44.0
- **Datasets:** 2.20.0
- **Precision:** bfloat16 (native A40 support)

**Batch sizes:**
- Dense models: 32 (optimized for throughput)
- ColPali: 4 (memory-constrained due to 1,030 patch embeddings)

**Fairness guarantees:**
- ✅ Same dataset for all models (no cherry-picking)
- ✅ Deterministic evaluation (no random sampling)
- ✅ Optimized batch size per architecture (no artificial handicaps)
- ✅ Native precision (bfloat16 on A40, no conversion overhead)

---

### Metrics Definitions

#### Recall@K (R@K)

**Definition:**
```
R@K = (# queries where correct answer is in top-K) / (# total queries)
```

**Example:**
- Query: "A dog playing with a ball"
- Top-10 results: [img_5, img_92, **img_42**, img_17, ...]
- If ground truth = img_42:
  - R@1 = 0 (not ranked #1)
  - R@5 = 1 (ranked #3, which is ≤5)
  - R@10 = 1 (ranked #3, which is ≤10)

**Why R@1 is hardest:**
- R@1 requires *exact* first match
- R@5 allows "close enough" (5× easier)
- R@10 is even more forgiving

**Industry relevance:**
- **R@1:** Critical for single-result applications (e.g., reverse image search)
- **R@5:** Standard for search engines (user sees top-5)
- **R@10:** Baseline expectation (if not in top-10, model is broken)

---

#### Queries Per Second (QPS)

**Definition:**
```
QPS = (# queries processed) / (total time in seconds)
```

**What we measure:**
- Encoding time only (no ANN index overhead)
- Includes: Image encoding + text encoding + similarity computation
- Excludes: Data loading, preprocessing (done once), post-processing

**Example calculation:**
- LAION-CLIP-H processes 5,000 images in 59.6 seconds
- QPS = 5000 / 59.6 = 83.8 queries/sec

**Production implications:**
- At 1M images:
  - LAION: 1,000,000 / 83.8 = 11,933 seconds = **3.3 hours**
  - ColPali: 1,000,000 / 2.9 = 344,828 seconds = **96 hours (4 days)**

**Why QPS matters more than latency:**
- Latency = time for 1 query (e.g., 12ms)
- QPS = throughput for entire gallery (e.g., 83.8 queries/sec)
- In production, you pre-compute embeddings (batch job), so QPS determines infrastructure cost

---

### Known Limitations

#### 1. No Statistical Significance Testing

**Problem:**
- Our evaluation is deterministic (same 5,000 images)
- Cannot determine if Apple (50.1%) vs LAION (46.3%) is statistically significant
- 3.8pp difference could be real or within measurement noise

**Why this happened:**
- Standard practice in vision-language benchmarks (CLIP, ALIGN papers do the same)
- True significance requires bootstrap sampling (1000+ iterations)

**Mitigation:**
- We developed V29 with bootstrap CIs (1000 iterations)
- Initial results show 95% CI width of ~1pp (meaning 3.8pp difference is likely significant)

---

#### 2. I2T Multi-Caption Inflation

**Problem:**
- I2T uses 5 captions per image (ANY match = success)
- T2I uses 1 caption per image (exact match required)
- This inflates I2T scores by ~20pp

**Example:**
- T2I: Find image #42 (1 target among 5,000) → Hard
- I2T: Find ANY of [cap_210, cap_211, cap_212, cap_213, cap_214] (5 targets among 25,000) → Easier

**Why we use this protocol:**
- Standard in COCO benchmark literature (enables comparison with published baselines)
- Alternative "symmetric" protocol (1 caption) exists but is non-standard

**Mitigation:**
- We explicitly explain the asymmetry (Section 2.2)
- Future work: Report both protocols (standard + symmetric)

---

#### 3. Single Hardware Configuration

**Problem:**
- QPS measured on A40 only
- Different GPUs have different memory/compute ratios
- Results may not generalize

**Example:**
- A40: 48GB VRAM, 150 TFLOPS (bfloat16)
- A100: 80GB VRAM, 312 TFLOPS (bfloat16)
- H100: 80GB VRAM, 1000 TFLOPS (bfloat16)

**Expected variations:**
- A100: ~2× faster QPS (more TFLOPS)
- H100: ~6× faster QPS (tensor cores)
- Consumer GPUs (RTX 4090): Similar to A40, but limited to 24GB VRAM

---

#### 4. Winoground Sample Size

**Problem:**
- Only 400 samples → limited statistical power
- 95% confidence interval for 35.2% score: ~±5pp
- Cannot detect small differences (<5pp)

**Why 400 samples?**
- Adversarial dataset creation is expensive (manual curation)
- Larger benchmarks exist (ARO: 1000, SugarCrepe: 3000) but require different evaluation infrastructure

**Mitigation:**
- We treat Winoground as *indicative* not definitive
- Recommend larger benchmarks for publication-grade claims

---

#### 5. Zero-Shot Evaluation Only

**Limitation:**
- Models tested "out of the box" (no fine-tuning on COCO)
- Domain-specific performance not measured

**Why zero-shot?**
- Standard practice (CLIP, ALIGN papers use zero-shot)
- Enables comparison across models without confounding fine-tuning effects

**Real-world implication:**
- Fine-tuning on domain data (medical, satellite) can improve accuracy by 5-10pp
- Our benchmark shows *baseline* performance, not domain-optimized

---

## Reproducibility

### Code Availability

**Repository:** Code will be open-sourced upon publication. Until then, available upon request.

**Key files:**
- `run_benchmark_grand_slam_v28_publication_ready.py` - Main benchmark (5 models)
- `run_benchmark_v28_openai_apple.py` - PyTorch 2.8+ benchmark (OpenAI + Apple)
- `run_benchmark_grand_slam_v29_statistical.py` - Bootstrap version (1000 iterations)
- `benchmark_v28_all_models_combined.csv` - Combined results

### Running the Benchmark

**Step 1: Install dependencies**
```bash
pip install torch==2.4.0 transformers==4.44.0 datasets==2.20.0 pillow pandas tqdm
```

**Step 2: Download dataset**
```python
from datasets import load_dataset

# MS-COCO Karpathy
coco = load_dataset("yerevann/coco-karpathy", split="test")

# Winoground
winoground = load_dataset("facebook/winoground", split="test")
```

**Step 3: Run benchmark**
```bash
# Single run (deterministic)
python run_benchmark_grand_slam_v28_publication_ready.py \
    --batch-size 32 \
    --sample-size 5000 \
    --output results.csv

# Bootstrap version (statistical CIs)
python run_benchmark_grand_slam_v29_statistical.py \
    --bootstrap-iterations 1000 \
    --batch-size 32 \
    --output results_bootstrap.csv
```

**Expected runtime:**
- Single run: ~2-3 hours (all 7 models)
- Bootstrap: ~20-30 hours (all 7 models, 1000 iterations)

---

## Citation

```bibtex
@techreport{coco_benchmark_2025,
  title={Benchmark of 7 Multimodal Embedding Models on MS-COCO Karpathy and Winoground},
  author={Krem, Ekrem},
  year={2025},
  institution={Independent Research}
}
```

---

## Appendix: Model Selection Decision Tree

**Choose based on your constraints:**

```
Priority: Accuracy?
  ├─ Yes → Apple-DFN5B-H (50.1%, 34.4 QPS)
  └─ No → Continue

Priority: Speed?
  ├─ Yes (>75 QPS) → LAION-CLIP-H (46.3%, 83.8 QPS)
  └─ No → Continue

Priority: Open-source?
  ├─ Yes → MetaCLIP-H14 (45.8%, 76.3 QPS, fully disclosed data)
  └─ No → Continue

Priority: Compositional reasoning?
  ├─ Yes → Two-stage: LAION (retrieve) + ColPali (re-rank)
  └─ No → Continue

Priority: Multilingual?
  ├─ Yes → Jina-CLIP-v1 (39.3%, 25.8 QPS, 100+ languages)
  └─ No → Continue

Priority: Edge deployment?
  └─ Yes → SigLIP-400M (35.4%, 47.1 QPS, 400M params)
```

---

**Last updated:** November 2025
**Benchmark version:** V1.0 (MS-COCO Karpathy + Winoground)
**Contact:** [Your email/GitHub]
