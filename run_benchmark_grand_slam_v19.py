import torch
import gc
import sys
import os
import time
import random
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor, SiglipModel, SiglipProcessor

# --- CONFIGURATION ---
NUM_RUNS = 3 
SAMPLE_SIZE = 1000
SEED_BASE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 

# --- COLPALI CHECK ---
try:
    from colpali_engine.models import ColPali, ColPaliProcessor
    COLPALI_AVAILABLE = True
except ImportError:
    COLPALI_AVAILABLE = False
    print("UYARI: ColPali engine bulunamadı.")

print(f"Running on: {DEVICE}")
print(f"Precision: {DTYPE}")
print(f"Sample Size: {SAMPLE_SIZE}")
print(f"Runs per Model: {NUM_RUNS}")

# --- MODEL LIST (The Final 8 from V15) ---
MODELS_TO_TEST = [
    {"name": "ColPali-v1.3",  "id": "vidore/colpali-v1.3", "type": "colpali", "batch_size": 4}, 
    {"name": "SigLIP-400M",   "id": "google/siglip-so400m-patch14-384", "type": "siglip", "batch_size": 64},
    {"name": "SigLIP-Base",   "id": "google/siglip-base-patch16-224", "type": "siglip", "batch_size": 64},
    {"name": "LAION-CLIP-H",  "id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "type": "dense", "batch_size": 32},
    {"name": "OpenAI-CLIP-L", "id": "openai/clip-vit-large-patch14-336", "type": "dense", "batch_size": 32},
    {"name": "Jina-CLIP-v1",  "id": "jinaai/jina-clip-v1", "type": "dense", "trust": True, "batch_size": 32},
    {"name": "Apple-DFN5B-H", "id": "apple/DFN5B-CLIP-ViT-H-14-378", "type": "dense", "trust": True, "batch_size": 32},
    {"name": "MetaCLIP-H14",  "id": "facebook/metaclip-h14-fullcc2.5b", "type": "dense", "trust": True, "batch_size": 16}
]

def clean_memory():
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    torch.cuda.empty_cache()
    gc.collect()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_get_text(item, col_name, dataset_name="", run_seed=0):
    val = item.get(col_name, "")
    if isinstance(val, list): 
        if len(val) == 0: return ""
        # Standardize: Always take the first caption to ensure strict reproducibility across frameworks
        return str(val[0])
    if isinstance(val, dict):
        for key in ['en', 'text', 'caption', 'query']:
            if key in val: return str(val[key])
        return str(val)
    return str(val)

def compute_metrics(scores):
    n = scores.size(0)
    metrics = {}
    for k in [1, 5, 10]:
        if k > n: 
            metrics[f"R@{{k}}"] = 0.0
            continue
        correct = 0
        for i in range(n):
            if i in torch.topk(scores[i], k=k).indices.tolist(): correct += 1
        metrics[f"R@{{k}}"] = correct/n
    
    mrr_sum = 0
    for i in range(n):
        rank = (scores[i].argsort(descending=True) == i).nonzero(as_tuple=True)[0].item() + 1
        mrr_sum += 1.0 / rank
    metrics["MRR"] = mrr_sum/n
    return metrics

def run_warmup(model, processor, model_info):
    dummy_img = torch.zeros((1, 3, 224, 224), dtype=torch.uint8)
    dummy_txt = ["warmup"]
    try:
        from PIL import Image
        dummy_pil = Image.new('RGB', (224, 224))
        with torch.no_grad():
            if model_info["type"] == "colpali":
                bi = processor.process_images([dummy_pil]).to(DEVICE)
                bq = processor.process_queries(dummy_txt).to(DEVICE)
                _ = model(**bi)
                _ = model(**bq)
            else:
                bi = processor(images=[dummy_pil], return_tensors="pt").to(DEVICE)
                bt = processor(text=dummy_txt, return_tensors="pt", padding=True).to(DEVICE)
                if hasattr(model, 'get_image_features'): _ = model.get_image_features(**bi)
                else: _ = model(**bi).image_embeds
                if hasattr(model, 'get_text_features'): _ = model.get_text_features(**bt)
                else: _ = model(**bt).text_embeds
    except Exception:
        pass

def run_retrieval_benchmark(model, processor, model_info, dataset, dataset_name, text_col="caption", image_col="image", run_idx=0):
    print(f"    > {dataset_name} (Run {run_idx+1}/{NUM_RUNS})...")
    bs = model_info.get("batch_size", 32)
    
    try:
        images = [item[image_col].convert("RGB") for item in dataset]
        queries = [safe_get_text(item, text_col, dataset_name) for item in dataset]
    except Exception as e:
        print(f"      Data Prep Error: {e}")
        return None

    t_start = time.time()
    try:
        with torch.no_grad():
            img_embeds_list = []
            for i in range(0, len(images), bs):
                batch_imgs = images[i : i+bs]
                if model_info["type"] == "colpali":
                    inputs = processor.process_images(batch_imgs).to(DEVICE)
                    out = model(**inputs)
                    img_embeds_list.extend([o.cpu() for o in out])
                else:
                    inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(DEVICE)
                    if hasattr(model, 'get_image_features'): out = model.get_image_features(**inputs)
                    else: out = model(**inputs).image_embeds
                    if out.dim() == 3: out = out[:, 0, :]
                    out = out / out.norm(dim=-1, keepdim=True)
                    img_embeds_list.append(out.to(DTYPE).cpu()) 
                del inputs, out
                torch.cuda.empty_cache()

            txt_embeds_list = []
            for i in range(0, len(queries), bs):
                batch_txt = queries[i : i+bs]
                if model_info["type"] == "colpali":
                    inputs = processor.process_queries(batch_txt).to(DEVICE)
                    out = model(**inputs)
                    txt_embeds_list.extend([o.cpu() for o in out])
                else:
                    inputs = processor(text=batch_txt, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                    if hasattr(model, 'get_text_features'): out = model.get_text_features(**inputs)
                    else: out = model(**inputs).text_embeds
                    if out.dim() == 3: out = out[:, 0, :]
                    out = out / out.norm(dim=-1, keepdim=True)
                    txt_embeds_list.append(out.to(DTYPE).cpu())
                del inputs, out
                torch.cuda.empty_cache()

            if model_info["type"] == "colpali":
                score_bs = 10
                all_scores = []
                for i in range(0, len(txt_embeds_list), score_bs):
                    q_batch = [q.to(DEVICE) for q in txt_embeds_list[i : i+score_bs]]
                    batch_scores = []
                    img_chunk = 20
                    for j in range(0, len(img_embeds_list), img_chunk):
                        i_batch = [img.to(DEVICE) for img in img_embeds_list[j : j+img_chunk]]
                        s = processor.score(q_batch, i_batch)
                        batch_scores.append(s.cpu())
                        del i_batch, s
                        torch.cuda.empty_cache()
                    all_scores.append(torch.cat(batch_scores, dim=1))
                    del q_batch
                    torch.cuda.empty_cache()
                scores = torch.cat(all_scores, dim=0) 
            else:
                all_img = torch.cat(img_embeds_list, dim=0).to(DTYPE)
                all_txt = torch.cat(txt_embeds_list, dim=0).to(DTYPE)
                scores = torch.matmul(all_txt, all_img.t())

        t_end = time.time()
        inference_time = t_end - t_start
        scores = scores.float()
        metrics = compute_metrics(scores)
        metrics["Time"] = inference_time
        
        if model_info["type"] == "colpali":
            avg_tokens = sum(e.shape[0] for e in img_embeds_list) / len(img_embeds_list)
            metrics["Embed"] = f"~{int(avg_tokens)} tok"
        else:
            dim = all_img.shape[1]
            metrics["Embed"] = f"{dim}d"
            
        return metrics

    except Exception as e:
        print(f"      Inference Error: {e}")
        return None

def run_winoground_full(model, processor, model_info):
    print("    > Winoground...")
    try: dataset = load_dataset("facebook/winoground", split="test", trust_remote_code=True)
    except: return None
    text_score, image_score, group_score, total = 0, 0, 0, len(dataset)
    try:
        for example in dataset:
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
                    s = torch.matmul(ie.to(DTYPE), te.to(DTYPE).t()).t()
            s = s.float().cpu()
            if (s[0,0] > s[1,0] and s[1,1] > s[0,1]): text_score += 1
            if (s[0,0] > s[0,1] and s[1,1] > s[1,0]): image_score += 1
            if (s[0,0] > s[1,0] and s[1,1] > s[0,1]) and (s[0,0] > s[0,1] and s[1,1] > s[1,0]): group_score += 1
        return {"Wino Group": group_score/total}
    except Exception as e: return None

# --- MAIN ---
if __name__ == "__main__":
    print(">>> LOADING DATASETS (Known Good Sources)...")
    
    try: 
        raw_flickr = load_dataset("lmms-lab/flickr30k", split="test", trust_remote_code=True)
        print(f"Flickr loaded: {len(raw_flickr)} samples")
    except Exception as e:
        print(f"Flickr failed: {e}"); raw_flickr = None

    # Revert to nielsr (proven working on V15)
    try: 
        print("Loading DocVQA (nielsr/docvqa_1200_examples)...")
        raw_docvqa = load_dataset("nielsr/docvqa_1200_examples", split="test")
        print(f"DocVQA loaded: {len(raw_docvqa)} samples")
    except Exception as e:
        print(f"DocVQA failed: {e}"); raw_docvqa = None

    # Revert to merve (proven working on V15)
    try: 
        print("Loading COCO (merve/coco2017)...")
        raw_coco = load_dataset("merve/coco2017", split="validation")
        print(f"COCO loaded: {len(raw_coco)} samples")
    except Exception as e:
        print(f"COCO failed: {e}"); raw_coco = None

    final_agg_results = []

    for m_info in MODELS_TO_TEST:
        m_name = m_info["name"]
        print(f"\n>>> TESTING: {m_name}")
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
                try:
                    model = AutoModel.from_pretrained(m_info["id"], trust_remote_code=trust, torch_dtype=DTYPE, use_safetensors=True).to(DEVICE)
                except:
                    print("    Safetensors failed, fallback...")
                    model = AutoModel.from_pretrained(m_info["id"], trust_remote_code=trust, torch_dtype=DTYPE).to(DEVICE)
                processor = AutoProcessor.from_pretrained(m_info["id"], trust_remote_code=trust)
                model.eval()
            
            print("    Warming up...")
            run_warmup(model, processor, m_info)
            
        except Exception as e:
            print(f"Load Error for {m_name}: {e}")
            continue

        model_runs = {"Flickr": [], "DocVQA": [], "COCO": [], "Winoground": []}
        
        for run_idx in range(NUM_RUNS):
            current_seed = SEED_BASE + run_idx
            set_seed(current_seed)
            
            if raw_flickr:
                ds = raw_flickr.shuffle(seed=current_seed).select(range(SAMPLE_SIZE))
                res = run_retrieval_benchmark(model, processor, m_info, ds, "Flickr", "caption", "image", run_idx)
                if res: model_runs["Flickr"].append(res)

            if raw_coco:
                ds = raw_coco.shuffle(seed=current_seed).select(range(SAMPLE_SIZE))
                # Check col name (merve usually 'captions')
                c_col = "captions" if "captions" in ds.features else "caption"
                res = run_retrieval_benchmark(model, processor, m_info, ds, "COCO", c_col, "image", run_idx)
                if res: model_runs["COCO"].append(res)

            if raw_docvqa:
                n_take = min(SAMPLE_SIZE, len(raw_docvqa))
                ds = raw_docvqa.shuffle(seed=current_seed).select(range(n_take))
                res = run_retrieval_benchmark(model, processor, m_info, ds, "DocVQA", "query", "image", run_idx)
                if res: model_runs["DocVQA"].append(res)
            
            wino_res = run_winoground_full(model, processor, m_info)
            if wino_res: model_runs["Winoground"].append(wino_res)

        agg_row = {"Model": m_name}
        for dataset_name, runs in model_runs.items():
            if not runs: continue
            metric_keys = runs[0].keys()
            for k in metric_keys:
                values = [r[k] for r in runs]
                mean = np.mean(values)
                std = np.std(values)
                
                if "Time" in k: agg_row[f"{dataset_name} Time"] = f"{mean:.2f}s"
                elif "Embed" in k: agg_row[f"Embed Size"] = values[0] # Constant
                elif "R@" in k or "MRR" in k or "Group" in k:
                    if "MRR" in k: agg_row[f"{dataset_name} {k}"] = f"{mean:.3f} ± {std:.3f}"
                    else: agg_row[f"{dataset_name} {k}"] = f"{mean:.1%} ± {std:.1%}"
                        
        final_agg_results.append(agg_row)
        print(f"Completed {m_name}")

    print("\n" + "="*120)
    print("GRAND SLAM BENCHMARK RESULTS (V19 - THE GOLDEN RATIO)")
    print("="*120)
    
    df = pd.DataFrame(final_agg_results)
    cols = ["Model", "Embed Size"]
    for ds in ["Flickr", "COCO", "DocVQA"]: cols.extend([f"{ds} R@1", f"{ds} R@5", f"{ds} Time"])
    cols.append("Winoground Wino Group")
    final_cols = [c for c in cols if c in df.columns]
    print(df[final_cols].to_markdown(index=False))
    df.to_csv("benchmark_v19_results.csv", index=False)
