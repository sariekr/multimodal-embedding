"""
V27 (OPTIMIZED & POLISHED) - MS-COCO (Full 5k) & Winoground

CHANGES from V26:
1. 🐛 CRITICAL FIX: Checks caption count BEFORE adding image to list (prevents orphaned images).
2. 🔥 WARM-UP FIX: Actually runs model forward pass for dense models (was only preprocessing).
3. ⚡ OPTIMIZATION: Loads model ONCE per architecture, shares across Winoground and all statistical runs.
4. 🛡️ SAFETY: Guarded torch.cuda.synchronize() for CPU compatibility.
5. 🧹 CLEANUP: Removed redundant reloading logic.

Usage:
  python run_benchmark_grand_slam_v27_optimized.py --runs 3 --batch-size 32
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
        logging.FileHandler("benchmark_v27.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- ARGS ---
def parse_args():
    parser = argparse.ArgumentParser(description="Grand Slam Multimodal Benchmark V27")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for dense models")
    parser.add_argument("--workers", type=int, default=16, help="Download workers")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of COCO samples")
    parser.add_argument("--runs", type=int, default=3, help="Number of statistical runs")
    parser.add_argument("--output", type=str, default="benchmark_v27_results.csv", help="Output CSV file")
    parser.add_argument("--cache-dir", type=str, default="./coco_images", help="Image cache directory")
    return parser.parse_args()

ARGS = parse_args()

# --- CONFIG ---
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
CACHE_DIR = Path(ARGS.cache_dir)
CACHE_DIR.mkdir(exist_ok=True)

# --- EXPECTED RESULTS ---
REFERENCE_RANGES = {
    "OpenAI-CLIP-L": {"T2I_R@1": (35.0, 40.0)}, 
    "SigLIP-400M":   {"T2I_R@1": (45.0, 55.0)}, 
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
    {"name": "MetaCLIP-H14",  "id": "facebook/metaclip-h14-fullcc2.5b", "type": "dense", "trust": True, "batch_size": 32}
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
    # Only clear if explicitly requested, but since we reuse models now, 
    # this function is mostly for end-of-model cleanup
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
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

def get_all_captions(item, col_name) -> List[str]:
    val = item.get(col_name, [])
    if isinstance(val, str) and val.strip().startswith('[') and val.strip().endswith(']'):
        try: val = ast.literal_eval(val)
        except: pass
    if not isinstance(val, list): val = [str(val)]
    return [str(v) for v in val]

def load_cached_image(item) -> Optional[Image.Image]:
    filename = f"{item['imgid']}.jpg"
    filepath = CACHE_DIR / filename
    if not filepath.exists(): return None
    return Image.open(filepath).convert("RGB")

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
    
    images = []
    texts = []
    query_to_img_map = []
    img_to_caps_map = {} 

    for _, item in enumerate(dataset):
        img = load_cached_image(item)
        if img is None:
            continue
        
        # 🐛 FIX: Check captions BEFORE adding image to avoid orphans
        captions = get_all_captions(item, "sentences")
        if len(captions) < 5: 
            continue
        captions = captions[:5]
            
        current_img_idx = len(images) 
        images.append(img)
        
        current_caps_indices = []
        for cap in captions:
            texts.append(cap)
            current_caps_indices.append(len(texts) - 1)
            query_to_img_map.append(current_img_idx)
        
        img_to_caps_map[current_img_idx] = current_caps_indices

    bs = m_info["batch_size"]
    
    # 🔥 WARM-UP (Corrected)
    logger.info("    Warming up GPU...")
    try:
        dummy_img = [images[0]] * min(2, len(images))
        if m_info["type"] == "colpali":
             _ = model(**processor.process_images(dummy_img).to(DEVICE))
        else:
             inputs = processor(images=dummy_img, return_tensors="pt").to(DEVICE)
             if hasattr(model, 'get_image_features'): 
                 _ = model.get_image_features(**inputs)
             else: 
                 _ = model(**inputs)
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")
        pass
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # INFERENCE
    t_start = time.time()
    try:
        with torch.no_grad():
            # Images
            img_embeds = []
            for i in tqdm(range(0, len(images), bs), desc="Encoding Images"):
                batch = images[i:i+bs]
                if m_info["type"] == "colpali":
                    inputs = processor.process_images(batch).to(DEVICE)
                    out = model(**inputs)
                    img_embeds.extend([o.cpu() for o in out])
                else:
                    inputs = processor(images=batch, return_tensors="pt", padding=True).to(DEVICE)
                    if hasattr(model, 'get_image_features'): out = model.get_image_features(**inputs)
                    else: out = model(**inputs).image_embeds
                    if out.dim()==3: out = out[:,0,:]
                    out = out / out.norm(dim=-1, keepdim=True)
                    img_embeds.append(out.to(DTYPE).cpu())
            
            # Texts
            txt_embeds = []
            for i in tqdm(range(0, len(texts), bs), desc="Encoding Texts"):
                batch = texts[i:i+bs]
                if m_info["type"] == "colpali":
                    inputs = processor.process_queries(batch).to(DEVICE)
                    out = model(**inputs)
                    txt_embeds.extend([o.cpu() for o in out])
                else:
                    inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                    if hasattr(model, 'get_text_features'): out = model.get_text_features(**inputs)
                    else: out = model(**inputs).text_embeds
                    if out.dim()==3: out = out[:,0,:]
                    out = out / out.norm(dim=-1, keepdim=True)
                    txt_embeds.append(out.to(DTYPE).cpu())

            # Scoring
            if m_info["type"] == "colpali":
                chunk_size = 10
                logger.info("    Computing ColPali T2I Scores...")
                scores_t2i_list = []
                for i in range(0, len(txt_embeds), chunk_size):
                    q_chunk = [t.to(DEVICE) for t in txt_embeds[i:i+chunk_size]]
                    scores_row = []
                    for j in range(0, len(img_embeds), chunk_size):
                        d_chunk = [d.to(DEVICE) for d in img_embeds[j:j+chunk_size]]
                        s = processor.score(q_chunk, d_chunk)
                        scores_row.append(s.cpu())
                    scores_t2i_list.append(torch.cat(scores_row, dim=1))
                scores_t2i = torch.cat(scores_t2i_list, dim=0)
                
                logger.info("    Computing ColPali I2T Scores (Images as Queries)...")
                scores_i2t_list = []
                for i in range(0, len(img_embeds), chunk_size):
                    q_chunk = [img.to(DEVICE) for img in img_embeds[i:i+chunk_size]]
                    scores_row = []
                    for j in range(0, len(txt_embeds), chunk_size):
                        d_chunk = [t.to(DEVICE) for t in txt_embeds[j:j+chunk_size]]
                        s = processor.score(q_chunk, d_chunk)
                        scores_row.append(s.cpu())
                    scores_i2t_list.append(torch.cat(scores_row, dim=1))
                scores_i2t = torch.cat(scores_i2t_list, dim=0)
                
            else:
                logger.info("    Computing Dense Scores...")
                all_img = torch.cat(img_embeds)
                all_txt = torch.cat(txt_embeds)
                scores_t2i = torch.matmul(all_txt.float(), all_img.float().t())
                scores_i2t = scores_t2i.t()

    except Exception as e:
        logger.error(f"Inference Error: {e}")
        import traceback
        traceback.print_exc()
        return {}

    dt = time.time() - t_start
    metrics = compute_metrics(scores_t2i, scores_i2t, query_to_img_map, img_to_caps_map)
    
    metrics["Time"] = dt
    metrics["QPS"] = len(texts)/dt
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
    logger.info("LOADING COCO-KARPATHY TEST SET...")
    ds_full = load_dataset("yerevann/coco-karpathy", split="test")
    
    logger.info(f"Dataset Size: {len(ds_full)}")
    
    # Ensure all images are on disk
    prepare_dataset_cache(ds_full)
    
    final_aggregated_results = []
    
    # MODEL LOOP
    for m_info in MODELS_TO_TEST:
        logger.info(f"{'='*60}")
        logger.info(f"EVALUATING: {m_info['name']}")
        logger.info(f"{'='*60}")
        
        # LOAD MODEL (ONCE)
        model = None
        processor = None
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
            logger.error(f"Model Load Failed: {e}")
            continue

        try:
            # 1. WINOGROUND (Single Run)
            wino_metrics = run_winoground(model, processor, m_info)
            
            # 2. COCO LOOP (Multi-Run)
            model_runs = []
            for run_idx in range(ARGS.runs):
                current_seed = SEED + run_idx
                set_seed(current_seed)
                logger.info(f"  ▶ RUN {run_idx+1}/{ARGS.runs} (Seed={current_seed})")
                
                # SAMPLING (WITH SHUFFLE)
                if ARGS.sample_size < len(ds_full):
                    ds_run = ds_full.shuffle(seed=current_seed).select(range(ARGS.sample_size))
                else:
                    ds_run = ds_full

                metrics = run_benchmark_coco(model, processor, m_info, ds_run)
                metrics.update(wino_metrics)
                
                # VALIDATION
                if m_info["name"] in REFERENCE_RANGES:
                    expected = REFERENCE_RANGES[m_info["name"]]["T2I_R@1"]
                    val = metrics.get("T2I_R@1", 0)
                    if not (expected[0] <= val <= expected[1]):
                        logger.warning(f"⚠️ {m_info['name']} T2I_R@1 ({val:.1f}%) OUT OF RANGE {expected}")
                
                model_runs.append(metrics)

            # AGGREGATE
            if model_runs:
                agg_row = {"Model": m_info["name"]}
                keys = model_runs[0].keys()
                for k in keys:
                    values = [r[k] for r in model_runs if k in r]
                    if values:
                        mean = np.mean(values)
                        std = np.std(values)
                        agg_row[k] = f"{mean:.1f} ± {std:.1f}"
                
                final_aggregated_results.append(agg_row)
                
                # Checkpoint
                df_agg = pd.DataFrame(final_aggregated_results)
                df_agg.to_csv(ARGS.output, index=False)
                logger.info(f"  ✅ Saved checkpoint to {ARGS.output}")

        except Exception as e:
            logger.error(f"Evaluation Failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Explicit cleanup after model is done
            del model
            del processor
            clean_memory()

    logger.info("BENCHMARK COMPLETE.")
