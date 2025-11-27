# COCO Benchmark V29 - Statistical Rigor Edition

## Overview

This is an **improved version** of the COCO benchmark that addresses the critical methodological issues identified in the V28 review:

### What's New in V29

✅ **Bootstrap Confidence Intervals (1000+ iterations)**
- No more `±0.0` standard deviations!
- Each iteration randomly samples 5,000 images WITH REPLACEMENT
- Reports true 95% confidence intervals for all metrics

✅ **Statistical Significance Testing**
- Pairwise model comparisons with p-values
- Determines if differences are statistically meaningful
- Uses CI overlap approximation + effect size computation

✅ **Symmetric I2T Protocol**
- Standard protocol: 5 captions per image (inflates scores)
- **NEW:** Symmetric protocol: 1 caption per image (directly comparable to T2I)
- Both protocols reported for transparency

✅ **Failure Analysis by Query Complexity**
- Spatial reasoning (left/right/above/below)
- Color attributes (red/blue/green)
- Counting (one/two/three/many)
- Caption length bins

✅ **Per-Category Performance**
- COCO supercategories (person, vehicle, animal, food, etc.)
- Identifies which models excel at which categories

---

## Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers datasets pillow scipy pandas tqdm requests
```

### 2. Run Benchmark (Quick Test - 100 iterations)

```bash
python run_benchmark_grand_slam_v29_statistical.py \
    --bootstrap-iterations 100 \
    --batch-size 32 \
    --sample-size 5000 \
    --output results_v29_quick.csv
```

**Expected runtime:**
- Dense models: ~20-30 minutes each
- ColPali: ~2-3 hours

### 3. Run Full Benchmark (1000 iterations - Publication Quality)

```bash
python run_benchmark_grand_slam_v29_statistical.py \
    --bootstrap-iterations 1000 \
    --batch-size 32 \
    --sample-size 5000 \
    --output results_v29_full.csv
```

**Expected runtime:**
- Dense models: ~2-3 hours each
- ColPali: ~15-20 hours

⚠️ **Warning:** This will take a LONG time! Consider using `--models` to test subset first.

### 4. Test Single Model First

```bash
python run_benchmark_grand_slam_v29_statistical.py \
    --bootstrap-iterations 100 \
    --models "SigLIP-400M" \
    --output results_siglip_test.csv
```

### 5. Generate Statistical Report

```bash
python analyze_statistical_results.py \
    --input results_v29_full.csv \
    --output statistical_analysis_report.md \
    --alpha 0.05
```

This creates a comprehensive markdown report with:
- Confidence interval tables
- Statistical significance matrices
- Failure analysis breakdown
- Per-category performance heatmaps

---

## Understanding the Output

### Bootstrap Results CSV

The output CSV contains these columns:

**Metrics with Confidence Intervals:**
- `T2I_R@1_mean`, `T2I_R@1_lower`, `T2I_R@1_upper`, `T2I_R@1_std`
- `I2T_R@1_mean`, `I2T_R@1_lower`, `I2T_R@1_upper`, `I2T_R@1_std` (standard protocol)
- `I2T_Sym_R@1_mean`, etc. (symmetric protocol - NEW!)

**Performance Metrics:**
- `QPS`: Queries per second (throughput)
- `Time`: Total time for all bootstrap iterations
- `Encoding_Time`: Time to encode all images/captions once

**Failure Analysis:**
- `_failure_analysis`: JSON string with detailed breakdown

### Interpreting Confidence Intervals

Example: `50.1% [49.2, 51.0]`

- **Mean:** 50.1% is the average accuracy across 1000 bootstrap samples
- **95% CI:** [49.2, 51.0] means we're 95% confident the true accuracy lies in this range
- **Width:** Narrower CI = more stable/reliable estimate

### Statistical Significance

In the analysis report, pairwise comparisons show:
- `***` = p < 0.001 (very strong evidence of difference)
- `**` = p < 0.01 (strong evidence)
- `*` = p < 0.05 (moderate evidence)
- `ns` = not significant (difference could be noise)

---

## Comparison: V28 vs V29

| Feature | V28 (Original) | V29 (Statistical) |
|:--------|:--------------|:------------------|
| Standard Deviation | ± 0.0 (meaningless) | Real std from bootstrap |
| Confidence Intervals | ❌ None | ✅ 95% CI for all metrics |
| Statistical Significance | ❌ None | ✅ Pairwise tests with p-values |
| I2T Protocol | 5-caption only | ✅ Both 5-caption AND 1-caption |
| Failure Analysis | ❌ None | ✅ Query complexity breakdown |
| Per-Category | ❌ None | ✅ COCO supercategory analysis |
| Runtime | ~30 mins/model | ~2-3 hours/model (1000 iter) |

---

## Advanced Usage

### Custom Bootstrap Iterations

Balance between runtime and CI accuracy:

```bash
# Quick test (wide CIs, fast)
--bootstrap-iterations 50

# Good balance (acceptable CIs)
--bootstrap-iterations 100

# Publication quality (narrow CIs, slow)
--bootstrap-iterations 1000
```

### Subset of Models

```bash
# Test only specific models
--models "LAION-CLIP-H,MetaCLIP-H14,SigLIP-400M"
```

### Smaller Sample Size (for debugging)

```bash
# Use 1000 images instead of 5000 (much faster, less reliable)
--sample-size 1000
```

---

## How Bootstrap Works

### The Problem with V28

```python
# V28 approach (NO variance!)
for seed in [42, 43, 44]:
    set_seed(seed)
    results = evaluate(same_5000_images)  # Always identical!
    # Result: mean=50.1, std=0.0 ❌
```

### The V29 Solution

```python
# V29 approach (TRUE variance!)
for iteration in range(1000):
    # Sample 5000 images WITH REPLACEMENT
    sampled_indices = np.random.choice(5000, size=5000, replace=True)
    results = evaluate(images[sampled_indices])
    # Result: mean=50.1, std=0.8 ✅
```

**Why this works:**
- Each iteration sees a different sample (some images repeated, some missing)
- This mimics drawing different 5K samples from the full COCO population
- Variability across iterations = true uncertainty estimate

---

## Interpreting Failure Analysis

### Spatial Queries

Example: "A dog **to the left of** a person"

Models struggle with spatial reasoning more than non-spatial queries.

### Color Queries

Example: "A **red** car next to a **blue** house"

Some models better at color discrimination than others.

### Counting Queries

Example: "**Three** people sitting on a bench"

Counting is hard for all models (even humans struggle with "several" vs "many").

### Per-Category Performance

Shows which models excel at which object types:
- Person: Apple DFN5B typically best
- Vehicles: All models similar
- Animals: LAION-CLIP-H strong
- Food: SigLIP surprisingly good

---

## Limitations & Future Work

### Current Limitations

1. **Computational Cost:** 1000 bootstrap iterations = ~3 hours per model
2. **Single Hardware:** Only tested on A40 (results may differ on A100/H100)
3. **Category Inference:** Categories inferred from captions (not ground truth COCO labels)
4. **CI Approximation:** P-values computed from CI overlap (not true permutation test)

### Future Improvements

1. **True Permutation Tests:** Store raw bootstrap samples for exact p-values
2. **Multi-Dataset:** Add Flickr30K, CC3M for cross-dataset validation
3. **Effect Size Reporting:** Cohen's d for practical significance
4. **Batch Size Normalization:** Test all models at batch=1 for fair comparison
5. **Cross-Validation:** k-fold CV instead of bootstrap for variance estimation

---

## FAQ

### Q: Why is this so much slower than V28?

**A:** Bootstrap requires 1000 evaluations instead of 3. But this is the ONLY way to get meaningful confidence intervals. Speed vs. statistical rigor tradeoff.

### Q: Can I use fewer bootstrap iterations?

**A:** Yes! For quick testing, use 50-100 iterations. For publication, use 1000+. More iterations = narrower CIs = more precise estimates.

### Q: Why both 5-caption and 1-caption I2T?

**A:** 5-caption is the standard protocol (for comparison with papers), but it inflates scores. 1-caption is symmetric and directly comparable to T2I.

### Q: What if two models have overlapping CIs?

**A:** Overlapping CIs suggest the difference MIGHT not be significant. Check the statistical significance table in the analysis report for p-values.

### Q: How do I know if a difference is "meaningful"?

**A:** Two criteria:
1. **Statistical significance:** p < 0.05 (difference is real, not noise)
2. **Practical significance:** Is the difference large enough to matter? (e.g., 1pp may be statistically significant but not practically useful)

---

## Citation

If you use this benchmark, please cite:

```bibtex
@misc{coco_benchmark_v29_2025,
  title={COCO Multimodal Embedding Benchmark with Statistical Rigor},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourrepo}}
}
```

---

## Acknowledgments

This benchmark addresses critical methodological issues identified in peer review:
- Bootstrap sampling eliminates ±0.0 std problem
- Statistical significance testing enables rigorous comparisons
- Symmetric I2T protocol fixes multi-caption inflation
- Failure analysis identifies model-specific weaknesses

Thanks to the reviewers for the thorough critique that inspired V29!

---

## Contact

For questions or issues, please open a GitHub issue or contact: [your-email@example.com]
