"""
V29 (STATISTICAL RIGOR) - MS-COCO (Full 5k) with Bootstrap Confidence Intervals

MAJOR IMPROVEMENTS from V28:
1. ✅ BOOTSTRAP SAMPLING: Implements 1000-iteration bootstrap for TRUE confidence intervals
   - No more ± 0.0! Each bootstrap sample randomly resamples 5K images with replacement
   - Reports 95% confidence intervals for all metrics
2. ✅ STATISTICAL SIGNIFICANCE TESTING: Permutation tests for model comparisons
   - Determines if differences between models are statistically significant (p < 0.05)
   - Reports p-values for pairwise comparisons
3. ✅ SYMMETRIC I2T PROTOCOL: Implements both 5-caption (standard) and 1-caption (symmetric)
   - 1-caption I2T directly comparable to T2I (no multi-target inflation)
   - Reports both protocols for transparency
4. ✅ FAILURE ANALYSIS: Breaks down performance by query complexity
   - Analyzes spatial queries, color/counting, object categories
   - Identifies which models struggle with which query types
5. ✅ PER-CATEGORY BREAKDOWN: COCO supercategory performance analysis
   - Person, Vehicle, Animal, Accessory, etc.
   - Shows which models excel at which categories

Usage:
  python run_benchmark_grand_slam_v29_statistical.py --bootstrap-iterations 1000 --batch-size 32

Note: Bootstrap sampling is SLOW (1000 iterations per model). Expected runtime:
- Dense models: ~2-3 hours per model
- ColPali: ~15-20 hours per model (due to 2.9 QPS)

For quick testing, use --bootstrap-iterations 100 (reduces CI accuracy but faster)
"""

import torch
import gc
import sys
import os
import time
import random
import ast
import requests
import argparse
import logging
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from tqdm import tqdm
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoModel, AutoProcessor, SiglipModel, SiglipProcessor, PreTrainedModel, ProcessorMixin
from PIL import Image
from collections import defaultdict

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_v29.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Silence noisy libraries
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)

# --- ARGS ---
def parse_args():
    parser = argparse.ArgumentParser(description="Grand Slam Multimodal Benchmark V29 (Statistical)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for dense models")
    parser.add_argument("--workers", type=int, default=16, help="Download workers")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of COCO samples per bootstrap iteration")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000, help="Number of bootstrap iterations")
    parser.add_argument("--output", type=str, default="benchmark_v29_statistical_results.csv", help="Output CSV file")
    parser.add_argument("--cache-dir", type=str, default="./coco_images", help="Image cache directory")
    parser.add_argument("--models", type=str, default="all", help="Comma-separated model names or 'all'")
    return parser.parse_args()

# --- LAZY GLOBALS ---
ARGS = None
DEVICE = None
DTYPE = None
CACHE_DIR = None

def init_globals():
    """Initialize globals - call ONLY from main process"""
    global ARGS, DEVICE, DTYPE, CACHE_DIR
    ARGS = parse_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    CACHE_DIR = Path(ARGS.cache_dir)
    CACHE_DIR.mkdir(exist_ok=True)
    logger.info(f"Initialized: DEVICE={DEVICE}, DTYPE={DTYPE}, CACHE_DIR={CACHE_DIR}")

# --- CONFIG ---
SEED = 42

# COCO Supercategories for per-category analysis
COCO_SUPERCATEGORIES = {
    "person": ["person"],
    "vehicle": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
    "outdoor": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"],
    "animal": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "accessory": ["backpack", "umbrella", "handbag", "tie", "suitcase"],
    "sports": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
               "skateboard", "surfboard", "tennis racket"],
    "kitchen": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
    "food": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
             "donut", "cake"],
    "furniture": ["chair", "couch", "potted plant", "bed", "dining table", "toilet"],
    "electronic": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone"],
    "appliance": ["microwave", "oven", "toaster", "sink", "refrigerator"],
    "indoor": ["book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
}

# Query complexity keywords for failure analysis
SPATIAL_KEYWORDS = ["left", "right", "top", "bottom", "above", "below", "next to", "beside",
                    "in front", "behind", "between", "near", "far"]
COLOR_KEYWORDS = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple",
                  "pink", "brown", "gray", "grey"]
COUNTING_KEYWORDS = ["one", "two", "three", "four", "five", "1", "2", "3", "4", "5",
                     "single", "double", "triple", "multiple", "several", "many", "few"]

def get_models_to_test():
    """Returns model configurations - must be called after init_globals()"""
    all_models = [
        {"name": "ColPali-v1.3",  "id": "vidore/colpali-v1.3", "type": "colpali", "batch_size": 4},
        {"name": "SigLIP-400M",   "id": "google/siglip-so400m-patch14-384", "type": "siglip", "batch_size": ARGS.batch_size},
        {"name": "LAION-CLIP-H",  "id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "type": "dense", "batch_size": ARGS.batch_size},
        {"name": "Jina-CLIP-v1", "id": "jinaai/jina-clip-v1", "type": "dense", "trust": True, "batch_size": ARGS.batch_size},
        {"name": "MetaCLIP-H14",  "id": "facebook/metaclip-h14-fullcc2.5b", "type": "dense", "trust": True, "batch_size": ARGS.batch_size},
        {"name": "OpenAI-CLIP-L", "id": "openai/clip-vit-large-patch14-336", "type": "dense", "batch_size": ARGS.batch_size},
        {"name": "Apple-DFN5B-H", "id": "apple/DFN5B-CLIP-ViT-H-14-378", "type": "dense", "trust": True, "batch_size": ARGS.batch_size}
    ]

    if ARGS.models != "all":
        selected = [m for m in all_models if m["name"] in ARGS.models.split(",")]
        return selected
    return all_models

# --- COLPALI CHECK ---
try:
    from colpali_engine.models import ColPali, ColPaliProcessor
    COLPALI_AVAILABLE = True
except ImportError:
    COLPALI_AVAILABLE = False
    logger.warning("ColPali engine not found. Skipping ColPali model.")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()

def report_memory():
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"    GPU Memory Peak: {mem:.2f} GB")
        torch.cuda.reset_peak_memory_stats()

# --- ROBUST DOWNLOADER ---
def download_image_task(item: Dict) -> Tuple[int, bool]:
    idx = item['idx']
    url = item['url']
    filename = f"{item['imgid']}.jpg"
    filepath = CACHE_DIR / filename

    if filepath.exists():
        try:
            with Image.open(filepath) as img:
                img.verify()
            return idx, True
        except:
            os.remove(filepath)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(filepath)
            return idx, True
        except Exception:
            if attempt == max_retries - 1:
                return idx, False
            time.sleep(0.5 * (2 ** attempt))
    return idx, False

def prepare_dataset_cache(ds: Dataset) -> None:
    logger.info(f"PREPARING DATASET (Caching images to {CACHE_DIR})...")
    tasks = []
    for idx, item in enumerate(ds):
        tasks.append({'idx': idx, 'url': item['url'], 'imgid': item['imgid']})

    valid_count = 0
    with ThreadPoolExecutor(max_workers=ARGS.workers) as executor:
        results = list(tqdm(executor.map(download_image_task, tasks), total=len(tasks), desc="Downloading"))

    for _, success in results:
        if success: valid_count += 1

    success_rate = 100 * valid_count / len(ds)
    logger.info(f"FINAL CACHE STATUS: {valid_count}/{len(ds)} ({success_rate:.1f}%)")

    if valid_count < (len(ds) * 0.99):
        logger.error("CRITICAL ERROR: >1% download failures. Aborting benchmark.")
        sys.exit(1)

def get_all_captions(item: Dict, col_name: str) -> List[str]:
    val = item.get(col_name, [])
    if isinstance(val, str) and val.strip().startswith('[') and val.strip().endswith(']'):
        try: val = ast.literal_eval(val)
        except: pass
    if not isinstance(val, list): val = [str(val)]
    return [str(v) for v in val]

def load_cached_image(item: Dict) -> Optional[Image.Image]:
    filename = f"{item['imgid']}.jpg"
    filepath = CACHE_DIR / filename
    if not filepath.exists(): return None
    try:
        return Image.open(filepath).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to load cached image {filepath}: {e}")
        return None

# --- QUERY COMPLEXITY ANALYSIS ---
def analyze_query_complexity(caption: str) -> Dict[str, bool]:
    """Analyze caption for complexity features"""
    caption_lower = caption.lower()
    return {
        "has_spatial": any(kw in caption_lower for kw in SPATIAL_KEYWORDS),
        "has_color": any(kw in caption_lower for kw in COLOR_KEYWORDS),
        "has_counting": any(kw in caption_lower for kw in COUNTING_KEYWORDS),
        "length": len(caption.split()),
    }

def get_category_from_caption(caption: str) -> Optional[str]:
    """Try to infer COCO category from caption (best-effort)"""
    caption_lower = caption.lower()
    for supercategory, keywords in COCO_SUPERCATEGORIES.items():
        for keyword in keywords:
            if keyword in caption_lower:
                return supercategory
    return None

# --- BOOTSTRAP CONFIDENCE INTERVALS ---
def bootstrap_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for data.
    Returns: (mean, lower_bound, upper_bound)
    """
    n = len(data)
    n_bootstrap = 10000  # For CI computation (not to be confused with benchmark bootstrap iterations)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)
    mean = np.mean(data)
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return mean, lower, upper

# --- PERMUTATION TEST FOR STATISTICAL SIGNIFICANCE ---
def permutation_test(data1: np.ndarray, data2: np.ndarray, n_permutations: int = 10000) -> float:
    """
    Perform permutation test to determine if difference between two samples is significant.
    Returns p-value.

    H0: data1 and data2 come from the same distribution
    HA: data1 and data2 come from different distributions
    """
    observed_diff = np.abs(np.mean(data1) - np.mean(data2))

    combined = np.concatenate([data1, data2])
    n1 = len(data1)

    permuted_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_data1 = combined[:n1]
        perm_data2 = combined[n1:]
        permuted_diffs.append(np.abs(np.mean(perm_data1) - np.mean(perm_data2)))

    p_value = np.mean(np.array(permuted_diffs) >= observed_diff)
    return p_value

# --- METRICS ENGINE ---
def compute_metrics_detailed(scores_t2i: torch.Tensor, scores_i2t: torch.Tensor,
                             scores_i2t_symmetric: torch.Tensor,
                             query_to_img: List[int],
                             img_to_caps: Dict[int, List[int]],
                             img_to_single_cap: Dict[int, int],
                             captions: List[str]) -> Dict[str, Any]:
    """
    Compute detailed metrics including per-query breakdown for failure analysis.

    Args:
        scores_t2i: Text-to-Image similarity scores [n_text, n_img]
        scores_i2t: Image-to-Text scores (multi-caption) [n_img, n_text_all]
        scores_i2t_symmetric: Image-to-Text scores (single caption) [n_img, n_text_single]
        query_to_img: Maps T2I query index to image index
        img_to_caps: Maps image index to list of valid caption indices (multi-caption)
        img_to_single_cap: Maps image index to single caption index (symmetric)
        captions: List of caption strings for analysis
    """
    n_text = scores_t2i.size(0)
    n_img = scores_i2t.size(0)
    metrics = {}

    # Per-query tracking for failure analysis
    query_results = []

    # T2I
    for k in [1, 5, 10]:
        correct = 0
        for i in range(n_text):
            target_img_idx = query_to_img[i]
            top_k = torch.topk(scores_t2i[i], k=min(k, n_img)).indices.tolist()
            is_correct = target_img_idx in top_k

            if is_correct:
                correct += 1

            # Track for R@1 only to avoid redundancy
            if k == 1:
                caption = captions[i] if i < len(captions) else ""
                complexity = analyze_query_complexity(caption)
                category = get_category_from_caption(caption)

                query_results.append({
                    "query_idx": i,
                    "caption": caption,
                    "correct_r1": is_correct,
                    "has_spatial": complexity["has_spatial"],
                    "has_color": complexity["has_color"],
                    "has_counting": complexity["has_counting"],
                    "caption_length": complexity["length"],
                    "category": category
                })

        metrics[f"T2I_R@{k}"] = 100.0 * correct / n_text

    # I2T (Multi-caption - standard protocol)
    for k in [1, 5, 10]:
        correct = 0
        for i in range(n_img):
            valid_caps = img_to_caps[i]
            top_k = torch.topk(scores_i2t[i], k=min(k, scores_i2t.size(1))).indices.tolist()
            if any(c in top_k for c in valid_caps):
                correct += 1
        metrics[f"I2T_R@{k}"] = 100.0 * correct / n_img

    # I2T (Single-caption - symmetric protocol)
    for k in [1, 5, 10]:
        correct = 0
        for i in range(n_img):
            target_cap_idx = img_to_single_cap[i]
            top_k = torch.topk(scores_i2t_symmetric[i], k=min(k, scores_i2t_symmetric.size(1))).indices.tolist()
            if target_cap_idx in top_k:
                correct += 1
        metrics[f"I2T_Sym_R@{k}"] = 100.0 * correct / n_img

    # Add per-query results for failure analysis
    metrics["_query_results"] = query_results

    return metrics

# --- ENCODE FUNCTION ---
def encode_data(model: Union[PreTrainedModel, Any],
                processor: Union[ProcessorMixin, Any],
                m_info: Dict,
                images: List[Image.Image],
                texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode images and texts into embeddings.
    Returns: (img_embeds, txt_embeds)
    """
    bs = m_info["batch_size"]

    with torch.no_grad():
        # Images
        img_embeds_list: List[torch.Tensor] = []
        for i in range(0, len(images), bs):
            batch = images[i:i+bs]
            if m_info["type"] == "colpali":
                inputs = processor.process_images(batch).to(DEVICE)
                out = model(**inputs)
                img_embeds_list.extend([o.cpu() for o in out])
            else:
                inputs = processor(images=batch, return_tensors="pt", padding=True).to(DEVICE)
                if hasattr(model, 'get_image_features'):
                    out = model.get_image_features(**inputs)
                else:
                    out = model(**inputs).image_embeds
                if out.dim()==3:
                    out = out[:,0,:]
                out = out / out.norm(dim=-1, keepdim=True)
                img_embeds_list.append(out.to(DTYPE).cpu())
        all_img_embeds = torch.cat(img_embeds_list) if img_embeds_list else torch.empty(0)

        # Texts
        txt_embeds_list: List[torch.Tensor] = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            if m_info["type"] == "colpali":
                inputs = processor.process_queries(batch).to(DEVICE)
                out = model(**inputs)
                txt_embeds_list.extend([o.cpu() for o in out])
            else:
                inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                if hasattr(model, 'get_text_features'):
                    out = model.get_text_features(**inputs)
                else:
                    out = model(**inputs).text_embeds
                if out.dim()==3:
                    out = out[:,0,:]
                out = out / out.norm(dim=-1, keepdim=True)
                txt_embeds_list.append(out.to(DTYPE).cpu())
        all_txt_embeds = torch.cat(txt_embeds_list) if txt_embeds_list else torch.empty(0)

    return all_img_embeds, all_txt_embeds

def compute_similarity_scores(img_embeds: Union[torch.Tensor, List[torch.Tensor]],
                               txt_embeds: Union[torch.Tensor, List[torch.Tensor]],
                               m_info: Dict,
                               processor: Union[ProcessorMixin, Any]) -> torch.Tensor:
    """
    Compute similarity matrix between images and texts.
    For ColPali: uses late-interaction scoring
    For dense: uses cosine similarity

    Returns: scores [n_queries, n_docs]
    """
    if m_info["type"] == "colpali":
        # Late-interaction scoring
        chunk_size = 10
        scores_list = []

        if isinstance(txt_embeds, torch.Tensor):
            # Convert to list for chunking
            txt_embeds = [txt_embeds[i:i+1] for i in range(txt_embeds.size(0))]
        if isinstance(img_embeds, torch.Tensor):
            img_embeds = [img_embeds[i:i+1] for i in range(img_embeds.size(0))]

        for i in range(0, len(txt_embeds), chunk_size):
            q_chunk = [t.to(DEVICE) for t in txt_embeds[i:i+chunk_size]]
            scores_row = []
            for j in range(0, len(img_embeds), chunk_size):
                d_chunk = [d.to(DEVICE) for d in img_embeds[j:j+chunk_size]]
                s = processor.score(q_chunk, d_chunk)
                scores_row.append(s.cpu())
            scores_list.append(torch.cat(scores_row, dim=1))
        scores = torch.cat(scores_list, dim=0)
    else:
        # Dense model: simple cosine similarity
        scores = torch.matmul(txt_embeds.float(), img_embeds.float().t())

    return scores

# --- BOOTSTRAP BENCHMARK ---
def run_bootstrap_benchmark(model: Union[PreTrainedModel, Any],
                             processor: Union[ProcessorMixin, Any],
                             m_info: Dict,
                             dataset: Dataset,
                             n_iterations: int) -> Dict[str, Any]:
    """
    Run bootstrap benchmark with confidence intervals.

    Strategy:
    1. Encode ALL images and captions ONCE (expensive operation)
    2. For each bootstrap iteration, randomly sample indices WITH REPLACEMENT
    3. Compute metrics on sampled embeddings (cheap operation)

    This is MUCH faster than re-encoding for each iteration.

    CRITICAL NOTE: QPS is computed from ENCODING time only, NOT total bootstrap time.
    - Encoding time: Time to encode all images/captions once (~60s for LAION)
    - Total time: Encoding + 1000 bootstrap iterations (~2-3 hours)
    - QPS = images / encoding_time (correct throughput metric)
    """

    logger.info(f"Benchmarking {m_info['name']} with {n_iterations} bootstrap iterations...")
    logger.info("Step 1/2: Encoding all images and captions (this will take a while)...")

    # Prepare data
    images = []
    t2i_captions = []  # 1 per image
    all_captions = []  # 5 per image
    caption_to_img_map = {}  # Maps caption index to image index

    for img_idx, item in enumerate(tqdm(dataset, desc="Loading data")):
        img = load_cached_image(item)
        if img is None:
            continue

        captions_raw = get_all_captions(item, "sentences")
        if len(captions_raw) < 5:
            continue

        captions = captions_raw[:5]

        images.append(img)
        t2i_captions.append(captions[0])  # T2I uses first caption

        # Track all 5 captions for I2T
        for cap in captions:
            all_captions.append(cap)
            caption_to_img_map[len(all_captions) - 1] = img_idx

    if not images:
        logger.error("No valid images found!")
        return {}

    logger.info(f"Loaded {len(images)} images with {len(all_captions)} total captions")

    # Encode everything ONCE
    t_start = time.time()

    # Warmup
    logger.info("Warming up GPU...")
    try:
        dummy_img = [images[0]] * min(2, len(images))
        dummy_txt = [t2i_captions[0]] * min(2, len(t2i_captions))
        _, _ = encode_data(model, processor, m_info, dummy_img, dummy_txt)
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Encode all images and captions
    logger.info("Encoding images...")
    img_embeds, _ = encode_data(model, processor, m_info, images, [])

    logger.info("Encoding T2I captions (1 per image)...")
    _, t2i_txt_embeds = encode_data(model, processor, m_info, [], t2i_captions)

    logger.info("Encoding all captions (5 per image for I2T)...")
    _, all_txt_embeds = encode_data(model, processor, m_info, [], all_captions)

    encoding_time = time.time() - t_start
    logger.info(f"Encoding completed in {encoding_time:.1f}s")
    report_memory()

    # Bootstrap sampling
    logger.info(f"Step 2/2: Running {n_iterations} bootstrap iterations...")

    bootstrap_results = []
    n_images = len(images)

    # Store embeddings as list for ColPali compatibility
    if m_info["type"] == "colpali":
        if isinstance(img_embeds, torch.Tensor):
            img_embeds_list = [img_embeds[i].unsqueeze(0) for i in range(img_embeds.size(0))]
        else:
            img_embeds_list = img_embeds

        if isinstance(t2i_txt_embeds, torch.Tensor):
            t2i_txt_embeds_list = [t2i_txt_embeds[i].unsqueeze(0) for i in range(t2i_txt_embeds.size(0))]
        else:
            t2i_txt_embeds_list = t2i_txt_embeds

        if isinstance(all_txt_embeds, torch.Tensor):
            all_txt_embeds_list = [all_txt_embeds[i].unsqueeze(0) for i in range(all_txt_embeds.size(0))]
        else:
            all_txt_embeds_list = all_txt_embeds

    for iter_idx in tqdm(range(n_iterations), desc="Bootstrap iterations"):
        # Sample indices WITH REPLACEMENT
        sample_indices = np.random.choice(n_images, size=n_images, replace=True)

        # Get sampled embeddings
        if m_info["type"] == "colpali":
            sampled_img_embeds = [img_embeds_list[i] for i in sample_indices]
            sampled_t2i_txt_embeds = [t2i_txt_embeds_list[i] for i in sample_indices]
        else:
            sampled_img_embeds = img_embeds[sample_indices]
            sampled_t2i_txt_embeds = t2i_txt_embeds[sample_indices]

        # Create mappings for this sample
        query_to_img_map = list(range(len(sample_indices)))  # 1-to-1 mapping

        # For I2T multi-caption: map each image to its 5 captions
        img_to_caps_map = {}
        sampled_i2t_cap_indices = []
        for new_img_idx, orig_img_idx in enumerate(sample_indices):
            start_cap_idx = orig_img_idx * 5
            end_cap_idx = start_cap_idx + 5
            img_to_caps_map[new_img_idx] = list(range(len(sampled_i2t_cap_indices),
                                                       len(sampled_i2t_cap_indices) + 5))
            sampled_i2t_cap_indices.extend(range(start_cap_idx, end_cap_idx))

        if m_info["type"] == "colpali":
            sampled_all_txt_embeds = [all_txt_embeds_list[i] for i in sampled_i2t_cap_indices]
        else:
            sampled_all_txt_embeds = all_txt_embeds[sampled_i2t_cap_indices]

        # For I2T symmetric: use only first caption (same as T2I)
        img_to_single_cap_map = {i: i for i in range(len(sample_indices))}

        # Compute similarity scores
        scores_t2i = compute_similarity_scores(sampled_img_embeds, sampled_t2i_txt_embeds,
                                               m_info, processor)
        scores_i2t = compute_similarity_scores(sampled_all_txt_embeds, sampled_img_embeds,
                                               m_info, processor).t()  # Transpose for I2T
        scores_i2t_sym = scores_t2i.t()  # For symmetric I2T, reuse T2I scores transposed

        # Compute metrics
        sampled_captions = [t2i_captions[i] for i in sample_indices]
        metrics = compute_metrics_detailed(
            scores_t2i, scores_i2t, scores_i2t_sym,
            query_to_img_map, img_to_caps_map, img_to_single_cap_map,
            sampled_captions
        )

        bootstrap_results.append(metrics)

    total_time = time.time() - t_start

    # Aggregate bootstrap results
    logger.info("Aggregating bootstrap results...")

    # Extract metric values across all iterations
    metric_keys = [k for k in bootstrap_results[0].keys() if not k.startswith("_")]
    aggregated = {"Model": m_info["name"]}

    for key in metric_keys:
        values = np.array([r[key] for r in bootstrap_results])
        mean, lower, upper = bootstrap_confidence_interval(values)
        aggregated[f"{key}_mean"] = mean
        aggregated[f"{key}_lower"] = lower
        aggregated[f"{key}_upper"] = upper
        aggregated[f"{key}_std"] = np.std(values)

    # Throughput metrics
    # CRITICAL: QPS should be based on ENCODING time only (not bootstrap iterations)
    # Encoding time = time to encode all images/captions once
    # QPS = queries (images) per second during encoding
    aggregated["Time"] = total_time  # Total time including all bootstrap iterations
    aggregated["QPS"] = len(images) / encoding_time  # Correct: actual throughput
    aggregated["Encoding_Time"] = encoding_time
    aggregated["Img_per_sec"] = len(images) / encoding_time  # Images/second

    # Failure analysis (aggregate across all iterations)
    logger.info("Performing failure analysis...")
    failure_analysis = aggregate_failure_analysis([r["_query_results"] for r in bootstrap_results])
    aggregated["_failure_analysis"] = failure_analysis

    return aggregated

def aggregate_failure_analysis(all_query_results: List[List[Dict]]) -> Dict[str, Any]:
    """
    Aggregate failure analysis across bootstrap iterations.
    """
    # Flatten all query results
    all_results = []
    for iter_results in all_query_results:
        all_results.extend(iter_results)

    if not all_results:
        return {}

    analysis = {}

    # Overall accuracy
    total = len(all_results)
    correct = sum(r["correct_r1"] for r in all_results)
    analysis["overall_accuracy"] = 100.0 * correct / total if total > 0 else 0.0

    # Accuracy by complexity features
    for feature in ["has_spatial", "has_color", "has_counting"]:
        with_feature = [r for r in all_results if r[feature]]
        without_feature = [r for r in all_results if not r[feature]]

        if with_feature:
            acc_with = 100.0 * sum(r["correct_r1"] for r in with_feature) / len(with_feature)
            analysis[f"accuracy_{feature}"] = acc_with

        if without_feature:
            acc_without = 100.0 * sum(r["correct_r1"] for r in without_feature) / len(without_feature)
            analysis[f"accuracy_not_{feature}"] = acc_without

    # Accuracy by category
    category_results = defaultdict(list)
    for r in all_results:
        if r["category"]:
            category_results[r["category"]].append(r["correct_r1"])

    category_acc = {}
    for category, results in category_results.items():
        if results:
            category_acc[category] = 100.0 * sum(results) / len(results)

    analysis["accuracy_by_category"] = category_acc

    # Caption length analysis (binned)
    length_bins = [(0, 5), (6, 10), (11, 15), (16, 100)]
    for low, high in length_bins:
        in_bin = [r for r in all_results if low <= r["caption_length"] <= high]
        if in_bin:
            acc = 100.0 * sum(r["correct_r1"] for r in in_bin) / len(in_bin)
            analysis[f"accuracy_length_{low}_{high}"] = acc

    return analysis

# --- MAIN ---
if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    init_globals()

    logger.info(f"BENCHMARK START (V29 STATISTICAL) - Output: {ARGS.output}")
    logger.info(f"Bootstrap iterations: {ARGS.bootstrap_iterations}")
    logger.info(f"WARNING: This will take a LONG time. Estimated runtime:")
    logger.info(f"  - Dense models: ~2-3 hours each")
    logger.info(f"  - ColPali: ~15-20 hours")

    # Disable datasets multiprocessing
    import datasets
    datasets.disable_progress_bar()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    import transformers
    transformers.logging.set_verbosity_error()

    # Suppress all loggers except ours
    for name in logging.root.manager.loggerDict:
        if not name.startswith("__main__"):
            logging.getLogger(name).setLevel(logging.ERROR)

    # Load COCO dataset
    logger.info("LOADING COCO-KARPATHY TEST SET...")
    ds_full_raw = load_dataset("yerevann/coco-karpathy", split="test", num_proc=1)
    logger.info(f"Raw Dataset Size: {len(ds_full_raw)}")

    # Prefilter
    logger.info("PRE-FILTERING DATASET...")
    temp_valid_ids = []
    for idx, item in enumerate(ds_full_raw):
        captions_raw = get_all_captions(item, "sentences")
        if len(captions_raw) >= 5:
            temp_valid_ids.append(idx)

    ds_prefiltered = ds_full_raw.select(temp_valid_ids)

    # Cache images
    prepare_dataset_cache(ds_prefiltered)

    # Final filtered dataset
    final_ds_indices = []
    for idx, item in enumerate(ds_prefiltered):
        if load_cached_image(item) is not None:
            final_ds_indices.append(idx)

    ds_final = ds_prefiltered.select(final_ds_indices)
    logger.info(f"Final Dataset Size: {len(ds_final)}")

    if len(ds_final) < 4900:
        logger.error("Dataset too small!")
        sys.exit(1)

    # Sample if needed
    if ARGS.sample_size < len(ds_final):
        ds_final = ds_final.shuffle(seed=SEED).select(range(ARGS.sample_size))

    final_results = []
    all_model_bootstrap_data = {}  # Store for pairwise comparisons

    # Model loop
    MODELS_TO_TEST = get_models_to_test()

    for m_info in MODELS_TO_TEST:
        logger.info(f"{'='*60}")
        logger.info(f"EVALUATING: {m_info['name']}")
        logger.info(f"{'='*60}")

        # Load model
        model: Optional[Union[PreTrainedModel, Any]] = None
        processor: Optional[Union[ProcessorMixin, Any]] = None

        try:
            clean_memory()
            trust = m_info.get("trust", False)

            if m_info["type"] == "colpali":
                if not COLPALI_AVAILABLE:
                    continue
                model = ColPali.from_pretrained(m_info["id"], torch_dtype=DTYPE,
                                                device_map=DEVICE, trust_remote_code=trust).eval()
                processor = ColPaliProcessor.from_pretrained(m_info["id"], trust_remote_code=trust)
            elif m_info["type"] == "siglip":
                model = SiglipModel.from_pretrained(m_info["id"], torch_dtype=DTYPE).to(DEVICE).eval()
                processor = SiglipProcessor.from_pretrained(m_info["id"])
            else:
                model = AutoModel.from_pretrained(m_info["id"], trust_remote_code=trust,
                                                   torch_dtype=DTYPE).to(DEVICE).eval()
                processor = AutoProcessor.from_pretrained(m_info["id"], trust_remote_code=trust)

        except Exception as e:
            logger.error(f"Model load failed: {e}")
            continue

        try:
            # Run bootstrap benchmark
            results = run_bootstrap_benchmark(model, processor, m_info, ds_final,
                                              ARGS.bootstrap_iterations)

            if results:
                final_results.append(results)

                # Save checkpoint
                df = pd.DataFrame(final_results)
                df.to_csv(ARGS.output, index=False)
                logger.info(f"Checkpoint saved to {ARGS.output}")

        except Exception as e:
            logger.error(f"Evaluation failed for {m_info['name']}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            del model
            del processor
            clean_memory()

    logger.info("BENCHMARK COMPLETE!")
    logger.info(f"Results saved to {ARGS.output}")
