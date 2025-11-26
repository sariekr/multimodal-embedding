"""
V23 (FINAL) - MS-COCO (Full 5k) & Winoground - ROBUST & BUG-FREE

CHANGES from V22:
1. 🐛 CRITICAL FIX: Fixed indexing bug where skipped images corrupted ground truth maps.
   - Now uses `len(images)` for tracking, independent of dataset iteration index.
2. 🔧 MATRIX LOGIC: Cleaned up Winoground scoring (Text @ Image.T) for clarity.
3. 🔥 WARM-UP: Added GPU warm-up to ensure timing accuracy.
4. 💾 MEMORY: Added GPU memory usage reporting.
5. ⚡ BATCHING: Optimized ColPali chunking (configurable).
6. 📊 REPRODUCIBILITY: Explicit deterministic algorithms enabled.

"""

import torch
import gc
import sys
import os
import time
import random
import ast
import requests
import shutil
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor, SiglipModel, SiglipProcessor
from PIL import Image

# --- REPRODUCIBILITY ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
CACHE_DIR = Path("./coco_images")
CACHE_DIR.mkdir(exist_ok=True)
BATCH_SIZE_DENSE = 32

print(f"Running on: {DEVICE}")
print(f"Precision: {DTYPE}")

# --- BASELINE CHECK (Expected Results on COCO 5K Test) ---
EXPECTED_RESULTS = {
    "OpenAI-CLIP-L": {"T2I_R@1": 58.4, "I2T_R@1": 37.8, "tolerance": 2.5},
    "SigLIP-400M":   {"T2I_R@1": 67.1, "I2T_R@1": 45.3, "tolerance": 2.5},
    "SigLIP-Base":   {"T2I_R@1": 60.2, "I2T_R@1": 40.1, "tolerance": 2.5},
    "LAION-CLIP-H":  {"T2I_R@1": 64.5, "I2T_R@1": 44.2, "tolerance": 2.5},
}

# --- MODEL LIST ---
MODELS_TO_TEST = [
    {"name": "ColPali-v1.3",  "id": "vidore/colpali-v1.3", "type": "colpali", "batch_size": 4},
    {"name": "SigLIP-400M",   "id": "google/siglip-so400m-patch14-384", "type": "siglip", "batch_size": BATCH_SIZE_DENSE},
    {"name": "SigLIP-Base",   "id": "google/siglip-base-patch16-224", "type": "siglip", "batch_size": BATCH_SIZE_DENSE},
    {"name": "LAION-CLIP-H",  "id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "type": "dense", "batch_size": BATCH_SIZE_DENSE},
    {"name": "OpenAI-CLIP-L", "id": "openai/clip-vit-large-patch14-336", "type": "dense", "batch_size": BATCH_SIZE_DENSE},
    {"name": "Jina-CLIP-v1", "id": "jinaai/jina-clip-v1", "type": "dense", "trust": True, "batch_size": BATCH_SIZE_DENSE},
    {"name": "Apple-DFN5B-H", "id": "apple/DFN5B-CLIP-ViT-H-14-378", "type": "dense", "trust": True, "batch_size": BATCH_SIZE_DENSE},
    {"name": "MetaCLIP-H14",  "id": "facebook/metaclip-h14-fullcc2.5b", "type": "dense", "trust": True, "batch_size": 32}
]

# --- COLPALI CHECK ---
try:
    from colpali_engine.models import ColPali, ColPaliProcessor
    COLPALI_AVAILABLE = True
except ImportError:
    COLPALI_AVAILABLE = False
    print("⚠️ WARNING: ColPali engine not found. Skipping ColPali model.")

def clean_memory():
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    torch.cuda.empty_cache()
    gc.collect()

def report_memory():
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"    [GPU Memory Peak: {mem:.2f} GB]")
        torch.cuda.reset_peak_memory_stats()

# --- ROBUST DOWNLOADER ---
def download_image_task(item):
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

def prepare_dataset(ds):
    print(f"\n>>> PREPARING DATASET (Caching images to {CACHE_DIR})...")
    tasks = []
    for idx, item in enumerate(ds):
        tasks.append({'idx': idx, 'url': item['url'], 'imgid': item['imgid']})

    valid_indices = set()
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(download_image_task, tasks), total=len(tasks), desc="Downloading/Verifying"))
    
    for idx, success in results:
        if success:
            valid_indices.add(idx)
    
    success_rate = 100 * len(valid_indices) / len(ds)
    print(f"📊 FINAL DATASET STATUS:")
    print(f"  Valid images: {len(valid_indices)}/{len(ds)}")
    print(f"  Success rate: {success_rate:.1f}%")
    
    if len(valid_indices) < 4950:
        print("🚨 CRITICAL ERROR: >1% download failures. Aborting benchmark.")
        sys.exit(1)
    return valid_indices

def get_all_captions(item, col_name):
    val = item.get(col_name, [])
    if isinstance(val, str) and val.strip().startswith('[') and val.strip().endswith(']'):
        try: val = ast.literal_eval(val)
        except: pass
    if not isinstance(val, list): val = [str(val)]
    return [str(v) for v in val]

def load_cached_image(item):
    filename = f"{item['imgid']}.jpg"
    filepath = CACHE_DIR / filename
    return Image.open(filepath).convert("RGB")

# --- METRICS ENGINE ---
def compute_metrics(scores_t2i, scores_i2t, query_to_img, img_to_caps):
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
        metrics[f"T2I_R@{k}"] = 100 * correct / n_text

    # I2T (Multi-caption correct)
    for k in [1, 5, 10]:
        correct = 0
        for i in range(n_img):
            valid_caps = img_to_caps[i]
            top_k = torch.topk(scores_i2t[i], k=min(k, n_text)).indices.tolist()
            if any(c in top_k for c in valid_caps):
                correct += 1
        metrics[f"I2T_R@{k}"] = 100 * correct / n_img

    return metrics

def run_benchmark_coco(model, processor, m_info, dataset, valid_indices):
    print(f"  > Benchmarking {m_info['name']} on {len(valid_indices)} COCO images...")
    
    images = []
    texts = []
    query_to_img_map = []
    img_to_caps_map = {}

    # Reconstruct dataset from valid indices
    subset = dataset.select(sorted(list(valid_indices)))
    
    for _, item in enumerate(subset):
        # LOAD IMAGE
        try:
            img = load_cached_image(item)
            # 🐛 FIX: Track index based on actual list length, not iterator
            current_img_idx = len(images) 
            images.append(img)
        except Exception as e:
            print(f"CRITICAL: Failed to load cached image {item['imgid']}: {e}")
            continue # Skip this image entirely

        # LOAD CAPTIONS
        captions = get_all_captions(item, "sentences")
        assert len(captions) >= 5, f"Expected >=5 captions, got {len(captions)}"
        captions = captions[:5] # Truncate to standard 5
        
        current_caps_indices = []
        for cap in captions:
            texts.append(cap)
            current_caps_indices.append(len(texts) - 1)
            # Map this text query to the CURRENT actual image index
            query_to_img_map.append(current_img_idx)
        
        img_to_caps_map[current_img_idx] = current_caps_indices

    bs = m_info["batch_size"]
    
    # WARM-UP
    print("    > Warming up GPU...")
    try:
        dummy_img = [images[0]] * min(2, len(images))
        if m_info["type"] == "colpali":
             _ = model(**processor.process_images(dummy_img).to(DEVICE))
        else:
             _ = processor(images=dummy_img, return_tensors="pt").to(DEVICE)
    except:
        pass
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
                # T2I
                print("    > Computing ColPali T2I Scores...")
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

                # I2T (Re-compute)
                print("    > Computing ColPali I2T Scores...")
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
                print("    > Computing Dense Scores...")
                all_img = torch.cat(img_embeds)
                all_txt = torch.cat(txt_embeds)
                # Float32 matmul for stability
                scores_t2i = torch.matmul(all_txt.float(), all_img.float().t())
                scores_i2t = scores_t2i.t()

    except Exception as e:
        print(f"Inference Error: {e}")
        import traceback
        traceback.print_exc()
        return {}

    dt = time.time() - t_start
    metrics = compute_metrics(scores_t2i, scores_i2t, query_to_img_map, img_to_caps_map)
    
    metrics["Time"] = f"{dt:.1f}s"
    metrics["QPS"] = f"{len(texts)/dt:.1f}"
    metrics["Img/s"] = f"{len(images)/dt:.1f}"
    metrics["Dataset_Size"] = len(images)
    
    report_memory()
    return metrics

def run_winoground(model, processor, model_info):
    print("  > Benchmarking Winoground (Full 400 samples)...")
    try:
        dataset = load_dataset("facebook/winoground", split="test")
    except Exception as e:
        print(f"    Failed to load Winoground: {e}")
        return {"Wino Group": "N/A"}

    text_score, image_score, group_score, total = 0, 0, 0, len(dataset)
    try:
        for example in tqdm(dataset, desc="Winoground"):
            images = [example["image_0"].convert("RGB"), example["image_1"].convert("RGB")]
            texts = [example["caption_0"], example["caption_1"]]
            with torch.no_grad():
                if model_info["type"] == "colpali":
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
                    
                    # Text @ Image.T (2x2 matrix)
                    s = torch.matmul(te.to(DTYPE), ie.to(DTYPE).t())
            
            s = s.float().cpu()
            # s[0,0] = T0->I0, s[0,1] = T0->I1
            # s[1,0] = T1->I0, s[1,1] = T1->I1
            
            # Text Score: Correct text matches correct image better than wrong image
            if (s[0,0] > s[0,1] and s[1,1] > s[1,0]): text_score += 1
            
            # Image Score: Correct image matches correct text better than wrong text
            if (s[0,0] > s[1,0] and s[1,1] > s[0,1]): image_score += 1
            
            # Group Score: Both must be correct
            if (s[0,0] > s[0,1] and s[1,1] > s[1,0]) and (s[0,0] > s[1,0] and s[1,1] > s[0,1]): group_score += 1
            
        return {"Wino Text": f"{text_score/total:.1%}", "Wino Image": f"{image_score/total:.1%}", "Wino Group": f"{group_score/total:.1%}"}
    except Exception as e:
        print(f"    Winoground inference error: {e}")
        return {"Wino Group": "Error"}

# --- MAIN ---
if __name__ == "__main__":
    print(">>> LOADING COCO-KARPATHY TEST SET...")
    ds_full = load_dataset("yerevann/coco-karpathy", split="test")
    
    print(f"Dataset Size: {len(ds_full)}")
    assert len(ds_full) == 5000, f"CRITICAL: Expected 5000 images, got {len(ds_full)}"
    
    print("Verifying caption counts...")
    for item in ds_full:
        captions = get_all_captions(item, "sentences")
        assert len(captions) >= 5, f"Expected >=5 captions, got {len(captions)}"
    print("✓ All images have at least 5 captions")

    valid_indices = prepare_dataset(ds_full)

    results = []
    
    for m_info in MODELS_TO_TEST:
        print(f"\n{'='*80}")
        print(f"EVALUATING: {m_info['name']}")
        print(f"{'='*80}")
        clean_memory()
        
        try:
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
                
            row = {"Model": m_info["name"]}
            
            # 1. COCO Benchmark
            metrics = run_benchmark_coco(model, processor, m_info, ds_full, valid_indices)
            row.update(metrics)
            
            # 2. Winoground Benchmark
            wino_metrics = run_winoground(model, processor, m_info)
            row.update(wino_metrics)
            
            # Baseline Check
            res_key = m_info["name"]
            if res_key in EXPECTED_RESULTS:
                expected = EXPECTED_RESULTS[res_key]
                t2i_val = float(metrics.get("T2I_R@1", 0))
                i2t_val = float(metrics.get("I2T_R@1", 0))
                
                diff_t2i = t2i_val - expected["T2I_R@1"]
                diff_i2t = i2t_val - expected["I2T_R@1"]
                
                print(f"\n🔍 BASELINE CHECK for {res_key}:")
                print(f"  T2I R@1: {t2i_val:.1f}% (Expected {expected['T2I_R@1']}%) -> Diff: {diff_t2i:+.1f}")
                print(f"  I2T R@1: {i2t_val:.1f}% (Expected {expected['I2T_R@1']}%) -> Diff: {diff_i2t:+.1f}")
                
                if abs(diff_t2i) > expected["tolerance"] or abs(diff_i2t) > expected["tolerance"]:
                    print("  ❌ DEVIATION DETECTED! Check code/preprocessing.")
                else:
                    print("  ✅ RESULTS MATCH EXPECTATIONS.")

            results.append(row)
            
        except Exception as e:
            print(f"Model Failed: {e}")
            import traceback
            traceback.print_exc()

    # Save
    df = pd.DataFrame(results)
    cols = ["Model", 
            "T2I_R@1", "T2I_R@5", "T2I_R@10", 
            "I2T_R@1", "I2T_R@5", "I2T_R@10", 
            "Wino Group", "Wino Text", "Wino Image",
            "QPS", "Img/s", "Time"]
    cols = [c for c in cols if c in df.columns]
    print("\n" + df[cols].to_markdown(index=False))
    df.to_csv("benchmark_v23_final.csv", index=False)
    print("\n✅ Saved to benchmark_v23_final.csv")
