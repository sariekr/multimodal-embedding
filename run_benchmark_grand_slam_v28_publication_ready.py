"""
V28 (PUBLICATION-READY) - MS-COCO (Full 5k) & Winoground

CHANGES from V27:
1. 🐛 CRITICAL FIX: Correctly aggregates Winoground metrics (no more "± 0.0").
   - Winoground scores are now merged as single values with COCO statistical results.
2. ✅ STANDARD COCO: Implements standard T2I evaluation (1 caption per image for query).
   - This prevents T2I R@1 inflation and makes results comparable to literature.
3. 📚 VALID REFERENCE: Adjusted reference ranges for standard COCO 5K T2I.
4. 🎲 SAMPLING FIX: Prefilters dataset for 5+ captions BEFORE shuffling/sampling.
   - Ensures consistent sample pool across runs.
5. 📊 MEMORY: Resets peak memory stats at start of each benchmark.

Usage:
  python run_benchmark_grand_slam_v28_publication_ready.py --runs 3 --batch-size 32
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

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_v28.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- ARGS ---
def parse_args():
    parser = argparse.ArgumentParser(description="Grand Slam Multimodal Benchmark V28")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for dense models")
    parser.add_argument("--workers", type=int, default=16, help="Download workers")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of COCO samples")
    parser.add_argument("--runs", type=int, default=3, help="Number of statistical runs")
    parser.add_argument("--output", type=str, default="benchmark_v28_results.csv", help="Output CSV file")
    parser.add_argument("--cache-dir", type=str, default="./coco_images", help="Image cache directory")
    return parser.parse_args()

ARGS = parse_args()

# --- CONFIG ---
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
CACHE_DIR = Path(ARGS.cache_dir)
CACHE_DIR.mkdir(exist_ok=True)

# --- EXPECTED RESULTS (COCO 5K Reference - T2I R@1) ---
# Source: Empirical estimates based on various papers/reproductions for 5K Karpathy test split
# OpenAI CLIP-L: ~36-38% (e.g., https://github.com/openai/CLIP/issues/83)
# SigLIP-400M: Typically higher than CLIP-L, around 45-55%
REFERENCE_RANGES = {
    "OpenAI-CLIP-L": {"T2I_R@1": (35.0, 40.0)}, 
    "SigLIP-400M":   {"T2I_R@1": (45.0, 55.0)}, 
    "LAION-CLIP-H":  {"T2I_R@1": (40.0, 50.0)},
}

# --- MODEL LIST ---
MODELS_TO_TEST = [
    {"name": "ColPali-v1.3",  "id": "vidore/colpali-v1.3", "type": "colpali", "batch_size": 4},
    {"name": "SigLIP-400M",   "id": "google/siglip-so400m-patch14-384", "type": "siglip", "batch_size": ARGS.batch_size},
    {"name": "SigLIP-Base",   "id": "google/siglip-base-patch16-224", "type": "siglip", "batch_size": ARGS.batch_size},
    {"name": "LAION-CLIP-H",  "id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "type": "dense", "batch_size": ARGS.batch_size},
    {"name": "OpenAI-CLIP-L", "id": "openai/clip-vit-large-patch14-336", "type": "dense", "batch_size": ARGS.batch_size},
    {"name": "Jina-CLIP-v1", "id": "jinaai/jina-clip-v1", "type": "dense", "trust": True, "batch_size": ARGS.batch_size},
    {"name": "Apple-DFN5B-H", "id": "apple/DFN5B-CLIP-ViT-H-14-378", "type": "dense", "trust": True, "batch_size": ARGS.batch_size},
    {"name": "MetaCLIP-H14",  "id": "facebook/metaclip-h14-fullcc2.5b", "type": "dense", "trust": True, "batch_size": ARGS.batch_size} # Standardized to BS 32
]

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
    # This is mainly for GC after deleting models
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

# --- METRICS ENGINE ---
def compute_metrics(scores_t2i: torch.Tensor, scores_i2t: torch.Tensor, 
                   query_to_img: List[int], img_to_caps: Dict[int, List[int]]) -> Dict[str, float]:
    n_text = scores_t2i.size(0)
    n_img = scores_i2t.size(0)
    metrics = {}

    # T2I
    for k in [1, 5, 10]:
        correct = 0
        for i in range(n_text):
            target_img_idx = query_to_img[i]
            top_k = torch.topk(scores_t2i[i], k=min(k, n_img)).indices.tolist()
            if target_img_idx in top_k:
                correct += 1
        metrics[f"T2I_R@{k}"] = 100.0 * correct / n_text

    # I2T (Multi-caption correct)
    for k in [1, 5, 10]:
        correct = 0
        for i in range(n_img):
            valid_caps = img_to_caps[i]
            top_k = torch.topk(scores_i2t[i], k=min(k, n_text)).indices.tolist()
            if any(c in top_k for c in valid_caps):
                correct += 1
        metrics[f"I2T_R@{k}"] = 100.0 * correct / n_img

    return metrics

def run_benchmark_coco(model: Union[PreTrainedModel, Any], 
                       processor: Union[ProcessorMixin, Any], 
                       m_info: Dict, 
                       dataset: Dataset) -> Dict[str, float]:
    
    logger.info(f"Benchmarking {m_info['name']} on COCO subset (N={len(dataset)})...")
    
    images = [] # Contains PIL Images
    t2i_queries = [] # Contains strings, 1 per image
    i2t_targets = [] # Contains strings, 5 per image for I2T pool
    
    query_to_img_map: List[int] = [] # Maps T2I query index to image index
    img_to_caps_map: Dict[int, List[int]] = {} # Maps image index to list of I2T target text indices

    current_coco_image_idx = -1 # Actual index in the current images list
    
    for item_idx, item in enumerate(dataset):
        img = load_cached_image(item)
        if img is None: continue
        
        captions_raw = get_all_captions(item, "sentences")
        if len(captions_raw) < 5: continue
        
        # Ensure we have exactly 5 captions for processing
        captions = captions_raw[:5]

        # --- T2I: Add 1 caption query per image ---
        # Standard protocol is to use the first caption for T2I (or rotate if doing 5-fold CV)
        # For simplicity, we use captions[0] as the T2I query for this image
        t2i_query_text = captions[0]
        t2i_queries.append(t2i_query_text)
        
        # This T2I query maps to the image we are about to add
        current_coco_image_idx += 1 
        query_to_img_map.append(current_coco_image_idx) # T2I query maps to this image
        images.append(img) # Add image to gallery
        
        # --- I2T: Add all 5 captions to the general pool for targets ---
        # The indices in i2t_targets correspond to the indices in the full text embedding matrix
        start_idx_for_this_image = len(i2t_targets)
        for cap in captions:
            i2t_targets.append(cap)
        
        # Map this image to the range of text indices that are its valid captions
        img_to_caps_map[current_coco_image_idx] = list(range(start_idx_for_this_image, len(i2t_targets)))

    bs = m_info["batch_size"]

    # Validate dataset is not empty
    if not images:
        logger.error("No valid images found in dataset!")
        return {}

    # WARM-UP (Corrected)
    logger.info("    Warming up GPU...")
    try:
        dummy_img = [images[0]] * min(2, len(images))
        if m_info["type"] == "colpali":
            _ = model(**processor.process_images(dummy_img).to(DEVICE))
        else:
            inputs = processor(images=dummy_img, return_tensors="pt", padding=True).to(DEVICE)
            if hasattr(model, 'get_image_features'):
                _ = model.get_image_features(**inputs)
            else:
                _ = model(**inputs).image_embeds # Ensure model is run
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")
        pass
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats() # Reset after warmup for accurate measurement

    # INFERENCE
    t_start = time.time()
    try:
        with torch.no_grad():
            # Images
            img_embeds_list: List[torch.Tensor] = []
            for i in tqdm(range(0, len(images), bs), desc="Encoding Images"):
                batch = images[i:i+bs]
                if m_info["type"] == "colpali":
                    inputs = processor.process_images(batch).to(DEVICE)
                    out = model(**inputs)
                    img_embeds_list.extend([o.cpu() for o in out])
                else:
                    inputs = processor(images=batch, return_tensors="pt", padding=True).to(DEVICE)
                    if hasattr(model, 'get_image_features'): out = model.get_image_features(**inputs)
                    else: out = model(**inputs).image_embeds
                    if out.dim()==3: out = out[:,0,:]
                    out = out / out.norm(dim=-1, keepdim=True)
                    img_embeds_list.append(out.to(DTYPE).cpu())
            all_img_embeds = torch.cat(img_embeds_list) if img_embeds_list else torch.empty(0)
            
            # Texts for T2I queries (1 per image)
            t2i_txt_embeds_list: List[torch.Tensor] = []
            for i in tqdm(range(0, len(t2i_queries), bs), desc="Encoding T2I Queries"):
                batch = t2i_queries[i:i+bs]
                if m_info["type"] == "colpali":
                    inputs = processor.process_queries(batch).to(DEVICE)
                    out = model(**inputs)
                    t2i_txt_embeds_list.extend([o.cpu() for o in out])
                else:
                    inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                    if hasattr(model, 'get_text_features'): out = model.get_text_features(**inputs)
                    else: out = model(**inputs).text_embeds
                    if out.dim()==3: out = out[:,0,:]
                    out = out / out.norm(dim=-1, keepdim=True)
                    t2i_txt_embeds_list.append(out.to(DTYPE).cpu())
            all_t2i_txt_embeds = torch.cat(t2i_txt_embeds_list) if t2i_txt_embeds_list else torch.empty(0)
            
            # Texts for I2T targets (all 5 per image)
            i2t_txt_embeds_list: List[torch.Tensor] = []
            for i in tqdm(range(0, len(i2t_targets), bs), desc="Encoding I2T Targets"):
                batch = i2t_targets[i:i+bs]
                if m_info["type"] == "colpali":
                    inputs = processor.process_queries(batch).to(DEVICE)
                    out = model(**inputs)
                    i2t_txt_embeds_list.extend([o.cpu() for o in out])
                else:
                    inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                    if hasattr(model, 'get_text_features'): out = model.get_text_features(**inputs)
                    else: out = model(**inputs).text_embeds
                    if out.dim()==3: out = out[:,0,:]
                    out = out / out.norm(dim=-1, keepdim=True)
                    i2t_txt_embeds_list.append(out.to(DTYPE).cpu())
            all_i2t_txt_embeds = torch.cat(i2t_txt_embeds_list) if i2t_txt_embeds_list else torch.empty(0)

            # Scoring
            if m_info["type"] == "colpali":
                chunk_size = 10
                # T2I (Text Queries from T2I_QUERIES -> Image Docs from IMAGES)
                logger.info("    Computing ColPali T2I Scores...")
                scores_t2i_list = []
                for i in range(0, len(t2i_txt_embeds_list), chunk_size):
                    q_chunk = [t.to(DEVICE) for t in t2i_txt_embeds_list[i:i+chunk_size]]
                    scores_row = []
                    for j in range(0, len(img_embeds_list), chunk_size):
                        d_chunk = [d.to(DEVICE) for d in img_embeds_list[j:j+chunk_size]]
                        s = processor.score(q_chunk, d_chunk)
                        scores_row.append(s.cpu())
                    scores_t2i_list.append(torch.cat(scores_row, dim=1))
                scores_t2i = torch.cat(scores_t2i_list, dim=0)
                
                # I2T (Image Queries from IMAGES -> Text Docs from I2T_TARGETS)
                logger.info("    Computing ColPali I2T Scores (Images as Queries)...")
                scores_i2t_list = []
                for i in range(0, len(img_embeds_list), chunk_size):
                    q_chunk = [img.to(DEVICE) for img in img_embeds_list[i:i+chunk_size]]
                    scores_row = []
                    for j in range(0, len(i2t_txt_embeds_list), chunk_size):
                        d_chunk = [t.to(DEVICE) for t in i2t_txt_embeds_list[j:j+chunk_size]]
                        s = processor.score(q_chunk, d_chunk)
                        scores_row.append(s.cpu())
                    scores_i2t_list.append(torch.cat(scores_row, dim=1))
                scores_i2t = torch.cat(scores_i2t_list, dim=0)
                
            else:
                logger.info("    Computing Dense Scores...")
                scores_t2i = torch.matmul(all_t2i_txt_embeds.float(), all_img_embeds.float().t())
                scores_i2t = torch.matmul(all_img_embeds.float(), all_i2t_txt_embeds.float().t())

    except Exception as e:
        logger.error(f"Inference Error: {e}")
        import traceback
        traceback.print_exc()
        return {}

    dt = time.time() - t_start
    
    # Need to map I2T target indices to the image indices for metric computation
    # img_to_caps_map now contains indices referring to all_i2t_txt_embeds
    
    metrics = compute_metrics(scores_t2i, scores_i2t, query_to_img_map, img_to_caps_map)
    
    metrics["Time"] = dt
    metrics["QPS"] = len(t2i_queries)/dt # QPS for T2I queries
    metrics["Img/s"] = len(images)/dt
    
    report_memory()
    return metrics

def run_winoground(model: Union[PreTrainedModel, Any], 
                   processor: Union[ProcessorMixin, Any], 
                   m_info: Dict) -> Dict[str, float]:
    logger.info("Benchmarking Winoground (Full 400 samples)...")
    try:
        dataset = load_dataset("facebook/winoground", split="test")
    except Exception as e:
        logger.error(f"Failed to load Winoground: {e}")
        return {}

    text_score, image_score, group_score, total = 0, 0, 0, len(dataset)
    try:
        for example in tqdm(dataset, desc="Winoground"):
            images = [example["image_0"].convert("RGB"), example["image_1"].convert("RGB")]
            texts = [example["caption_0"], example["caption_1"]]
            with torch.no_grad():
                if m_info["type"] == "colpali":
                    bi = processor.process_images(images).to(DEVICE)
                    bq = processor.process_queries(texts).to(DEVICE)
                    s = processor.score(model(**bq), model(**bi))
                else:
                    ii = processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
                    it = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                    
                    if hasattr(model, 'get_image_features'): ie = model.get_image_features(**ii)
                    else: ie = model(**ii).image_embeds
                    if hasattr(model, 'get_text_features'): te = model.get_text_features(**it)
                    else: te = model(**it).text_embeds
                    if ie.dim()==3: ie=ie[:,0,:]
                    if te.dim()==3: te=te[:,0,:]
                    
                    ie = ie / ie.norm(dim=-1, keepdim=True)
                    te = te / te.norm(dim=-1, keepdim=True)
                    
                    s = torch.matmul(te.to(DTYPE), ie.to(DTYPE).t())
            
            s = s.float().cpu()
            if (s[0,0] > s[0,1] and s[1,1] > s[1,0]): text_score += 1
            if (s[0,0] > s[1,0] and s[1,1] > s[0,1]): image_score += 1
            if (s[0,0] > s[0,1] and s[1,1] > s[1,0]) and (s[0,0] > s[1,0] and s[1,1] > s[0,1]): group_score += 1
            
        return {
            "Wino Text": 100.0 * text_score/total, 
            "Wino Image": 100.0 * image_score/total, 
            "Wino Group": 100.0 * group_score/total
        }
    except Exception as e:
        logger.error(f"Winoground error: {e}")
        return {}

# --- MAIN ---
if __name__ == "__main__":
    logger.info(f"BENCHMARK START (V28) - Output: {ARGS.output}")
    
    # 0. Load Full COCO Dataset (Once)
    logger.info("LOADING COCO-KARPATHY TEST SET...")
    ds_full_raw = load_dataset("yerevann/coco-karpathy", split="test")
    logger.info(f"Raw Dataset Size: {len(ds_full_raw)}")
    
    # 1. Pre-filter dataset for images that can be loaded and have enough captions
    # This creates a consistent pool for all runs/seeds
    logger.info("PRE-FILTERING DATASET (ensuring loadable images with 5+ captions)...")
    
    # Create a temporary filtered list to get valid items and their original indices
    prefiltered_items = []
    original_indices_map = {} # Maps new subset index to original ds_full_raw index
    
    temp_valid_ids = []
    temp_items_for_cache_check = []

    for idx, item in enumerate(ds_full_raw):
        captions_raw = get_all_captions(item, "sentences")
        # Only include items that have at least 5 captions for consistent benchmark
        if len(captions_raw) >= 5:
            temp_valid_ids.append(idx)
            temp_items_for_cache_check.append(item)

    ds_prefiltered = ds_full_raw.select(temp_valid_ids)
    
    # 2. Prepare/Cache images for the prefiltered dataset
    # This step will error out if too many images fail to download.
    prepare_dataset_cache(ds_prefiltered)

    # After caching, create the final "valid and cached" dataset
    final_ds_indices = []
    for idx, item in enumerate(ds_prefiltered):
        if load_cached_image(item) is not None:
            final_ds_indices.append(idx)
    ds_final = ds_prefiltered.select(final_ds_indices)
    
    logger.info(f"Final Filtered & Cached Dataset Size for Benchmarking: {len(ds_final)}")
    if len(ds_final) < 4900: # Allow some initial filtering for consistency
        logger.error("CRITICAL ERROR: Final dataset size too small after filtering/caching. Aborting.")
        sys.exit(1)
        
    final_aggregated_results = []
    
    # Create/Clear output file for aggregated results
    if os.path.exists(ARGS.output):
        os.remove(ARGS.output)
    
    # MODEL LOOP
    for m_info in MODELS_TO_TEST:
        logger.info(f"{'='*60}")
        logger.info(f"EVALUATING: {m_info['name']}")
        logger.info(f"{'='*60}")
        
        # --- LOAD MODEL (ONCE per architecture) ---
        model: Optional[Union[PreTrainedModel, Any]] = None
        processor: Optional[Union[ProcessorMixin, Any]] = None
        try:
            clean_memory() # Ensure previous model is gone
            trust = m_info.get("trust", False)
            if m_info["type"] == "colpali":
                if not COLPALI_AVAILABLE: continue
                model = ColPali.from_pretrained(m_info["id"], torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=trust).eval()
                processor = ColPaliProcessor.from_pretrained(m_info["id"], trust_remote_code=trust)
            elif m_info["type"] == "siglip":
                model = SiglipModel.from_pretrained(m_info["id"], torch_dtype=DTYPE).to(DEVICE).eval()
                processor = SiglipProcessor.from_pretrained(m_info["id"])
            else:
                model = AutoModel.from_pretrained(m_info["id"], trust_remote_code=trust, torch_dtype=DTYPE).to(DEVICE).eval()
                processor = AutoProcessor.from_pretrained(m_info["id"], trust_remote_code=trust)
        except Exception as e:
            logger.error(f"Model Load Failed for {m_info['name']}: {e}")
            continue

        try:
            # 1. WINOGROUND (Single Run, since no sampling)
            wino_metrics = run_winoground(model, processor, m_info)
            
            # 2. COCO LOOP (Multi-Run for statistics)
            coco_runs_results: List[Dict[str, float]] = []
            for run_idx in range(ARGS.runs):
                current_seed = SEED + run_idx
                set_seed(current_seed)
                logger.info(f"  ▶ RUN {run_idx+1}/{ARGS.runs} (Seed={current_seed})")
                
                # SAMPLING (WITH SHUFFLE) from the final_ds pool
                if ARGS.sample_size < len(ds_final):
                    ds_run = ds_final.shuffle(seed=current_seed).select(range(ARGS.sample_size))
                else:
                    ds_run = ds_final
                
                # METRICS for COCO
                metrics = run_benchmark_coco(model, processor, m_info, ds_run)
                coco_runs_results.append(metrics)
                
                # VALIDATION (Informational, per run)
                if m_info["name"] in REFERENCE_RANGES:
                    expected = REFERENCE_RANGES[m_info["name"]]["T2I_R@1"]
                    val = metrics.get("T2I_R@1", 0.0) # Ensure float for comparison
                    if not (expected[0] <= val <= expected[1]):
                        logger.warning(f"⚠️ {m_info['name']} T2I_R@1 ({val:.1f}%) OUT OF RANGE {expected} for Run {run_idx+1}")

            # AGGREGATE STATS for COCO metrics
            if coco_runs_results:
                agg_row = {"Model": m_info["name"]}
                
                # Add Winoground metrics (single values)
                for k, v in wino_metrics.items():
                    agg_row[k] = f"{v:.1f}"

                # Aggregate COCO metrics
                first_run_keys = [k for k in coco_runs_results[0].keys() if not k.startswith("Wino")]
                for k in first_run_keys:
                    values = [r[k] for r in coco_runs_results if k in r]
                    if values:
                        mean = np.mean(values)
                        std = np.std(values)
                        agg_row[k] = f"{mean:.1f} ± {std:.1f}"
                
                final_aggregated_results.append(agg_row)
                
                # Save Checkpoint
                df_agg = pd.DataFrame(final_aggregated_results)
                df_agg.to_csv(ARGS.output, index=False)
                logger.info(f"  ✅ Saved checkpoint to {ARGS.output}")

        except Exception as e:
            logger.error(f"Evaluation Failed for {m_info['name']}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Explicit cleanup after model is done
            del model
            del processor
            clean_memory()

    logger.info("BENCHMARK COMPLETE.")
