import torch
import gc
import sys
import os
import time
import random
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor

# --- REPRODUCIBILITY ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- COLPALI CHECK ---
try:
    from colpali_engine.models import ColPali, ColPaliProcessor
    COLPALI_AVAILABLE = True
except ImportError:
    COLPALI_AVAILABLE = False
    print("UYARI: ColPali engine bulunamadı.")

# --- AYARLAR ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 
SAMPLE_SIZE = 1000 

print(f"Running on: {DEVICE}")
print(f"Precision: {DTYPE}")
print(f"Sample Size: {SAMPLE_SIZE}")

# --- MODEL LİSTESİ ---
MODELS_TO_TEST = [
    {"name": "ColPali-v1.3",  "id": "vidore/colpali-v1.3", "type": "colpali", "batch_size": 4}, 
    {"name": "SigLIP-400M",   "id": "google/siglip-so400m-patch14-384", "type": "dense", "batch_size": 64},
    {"name": "SigLIP-Large",  "id": "google/siglip-large-patch16-384", "type": "dense", "batch_size": 32},
    {"name": "LAION-CLIP-H",  "id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "type": "dense", "batch_size": 32},
    {"name": "OpenAI-CLIP-L", "id": "openai/clip-vit-large-patch14-336", "type": "dense", "batch_size": 32},
    {"name": "Jina-CLIP-v1",  "id": "jinaai/jina-clip-v1", "type": "dense", "trust": True, "batch_size": 32},
    {"name": "Apple-DFN5B-H", "id": "apple/DFN5B-CLIP-ViT-H-14-378", "type": "dense", "trust": True, "batch_size": 32},
    {"name": "DataComp-XL",   "id": "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K", "type": "dense", "trust": True, "batch_size": 32}
]

def clean_memory():
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    torch.cuda.empty_cache()
    gc.collect()

def safe_get_text(item, col_name, dataset_name=""):
    val = item.get(col_name, "")
    if isinstance(val, list): 
        if len(val) == 0: return ""
        # COCO Fix: Pick random caption to better represent diversity
        if dataset_name == "COCO":
            return str(random.choice(val))
        return str(val[0])
        
    if isinstance(val, dict):
        for key in ['en', 'text', 'caption', 'query']:
            if key in val: return str(val[key])
        return str(val)
    return str(val)

def compute_metrics(scores):
    """Computes R@1, R@5, R@10 and MRR."""
    n = scores.size(0)
    metrics = {}
    
    # Recall at K
    for k in [1, 5, 10]:
        if k > n: 
            metrics[f"R@{k}"] = "N/A"
            continue
        correct = 0
        for i in range(n):
            top_k_indices = torch.topk(scores[i], k=k).indices.tolist()
            if i in top_k_indices:
                correct += 1
        metrics[f"R@{k}"] = f"{correct/n:.1%}"
        
    # MRR
    mrr_sum = 0
    for i in range(n):
        rank = (scores[i].argsort(descending=True) == i).nonzero(as_tuple=True)[0].item() + 1
        mrr_sum += 1.0 / rank
    metrics["MRR"] = f"{mrr_sum/n:.3f}"
    
    return metrics

def run_retrieval_benchmark(model, processor, model_info, dataset, dataset_name, text_col="caption", image_col="image"):
    print(f"  > Benchmarking {dataset_name} (N={len(dataset)})...")
    bs = model_info.get("batch_size", 32)
    
    try:
        images = [item[image_col].convert("RGB") for item in dataset]
        queries = [safe_get_text(item, text_col, dataset_name) for item in dataset]
    except Exception as e:
        print(f"    Data Prep Error: {e}")
        return {f"{dataset_name} Status": "Error"}

    t_start = time.time()
    
    try:
        with torch.no_grad():
            # --- IMAGE EMBEDDINGS ---
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

            # --- TEXT EMBEDDINGS ---
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

            # --- SCORING ---
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
        throughput = len(queries) / inference_time

        # --- METRICS ---
        scores = scores.float()
        metrics = compute_metrics(scores)
        
        # Performance Metrics
        metrics[f"{dataset_name} Time"] = f"{inference_time:.1f}s"
        metrics[f"{dataset_name} QPS"] = f"{throughput:.1f}"
        
        # Embedding Info
        if model_info["type"] == "colpali":
            # Calculate avg tokens without creating huge numpy array
            total_tokens = sum(e.shape[0] for e in img_embeds_list)
            avg_tokens = total_tokens / len(img_embeds_list)
            metrics[f"{dataset_name} Embed"] = f"~{int(avg_tokens)} tok"
        else:
            dim = all_img.shape[1]
            metrics[f"{dataset_name} Embed"] = f"{dim}d"
        
        # Return prefixed metrics
        final_metrics = {}
        for k, v in metrics.items():
            final_metrics[f"{dataset_name} {k}"] = v
            
        return final_metrics

    except Exception as e:
        print(f"Inference Error: {e}")
        import traceback
        traceback.print_exc()
        # Return Error for all keys to keep table clean
        return {
            f"{dataset_name} R@1": "Error", 
            f"{dataset_name} R@5": "Error", 
            f"{dataset_name} MRR": "Error",
            f"{dataset_name} QPS": "Error"
        }

def run_winoground_full(model, processor, model_info):
    print("  > Benchmarking Winoground (Full Metrics)...")
    try: dataset = load_dataset("facebook/winoground", split="test", trust_remote_code=True)
    except: return {"Wino Group": "N/A"}
    
    text_score, image_score, group_score = 0, 0, 0
    total = len(dataset)
    
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
                    ie, te = ie.to(DTYPE), te.to(DTYPE)
                    s = torch.matmul(ie, te.t()).t()
            
            s = s.float().cpu()
            
            t_correct = (s[0,0] > s[1,0]) and (s[1,1] > s[0,1])
            i_correct = (s[0,0] > s[0,1]) and (s[1,1] > s[1,0])
            
            if t_correct: text_score += 1
            if i_correct: image_score += 1
            if t_correct and i_correct: group_score += 1
            
        return {
            "Wino Text": f"{text_score/total:.1%}",
            "Wino Image": f"{image_score/total:.1%}",
            "Wino Group": f"{group_score/total:.1%}"
        }
    except Exception as e:
        return {"Wino Group": "Error"}

# --- MAIN ---
if __name__ == "__main__":
    print(">>> LOADING DATASETS...")
    try: ds_flickr = load_dataset("lmms-lab/flickr30k", split="test", trust_remote_code=True).select(range(SAMPLE_SIZE))
    except: ds_flickr = []
    try: ds_doc = load_dataset("nielsr/docvqa_1200_examples", split="test") 
    except: ds_doc = []
    try: 
        print("Loading COCO...")
        ds_coco = load_dataset("merve/coco2017", split="validation").select(range(SAMPLE_SIZE))
    except: 
        ds_coco = []
    
    results = []
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
            else:
                try:
                    model = AutoModel.from_pretrained(m_info["id"], trust_remote_code=trust, torch_dtype=DTYPE, use_safetensors=True).to(DEVICE)
                except:
                    print("    Safetensors failed, trying standard...")
                    model = AutoModel.from_pretrained(m_info["id"], trust_remote_code=trust, torch_dtype=DTYPE).to(DEVICE)
                processor = AutoProcessor.from_pretrained(m_info["id"], trust_remote_code=trust)
                model.eval()
        except Exception as e:
            print(f"Load Error: {e}")
            continue

        row = {"Model": m_name}
        
        if len(ds_flickr) > 0: 
            row.update(run_retrieval_benchmark(model, processor, m_info, ds_flickr, "Flickr", "caption"))
            
        if len(ds_coco) > 0: 
            row.update(run_retrieval_benchmark(model, processor, m_info, ds_coco, "COCO", "captions"))
            
        if len(ds_doc) > 0: 
            row.update(run_retrieval_benchmark(model, processor, m_info, ds_doc, "DocVQA", "query"))
            
        row.update(run_winoground_full(model, processor, m_info))
        
        results.append(row)
        print(f"Result: {row}")

    print("\n" + "="*120)
    print("GRAND SLAM BENCHMARK RESULTS (V11 - THE MASTERPIECE)")
    print("="*120)
    
    if len(results) > 0:
        keys = results[0].keys()
        # Strategic ordering for final table
        order = [
            "Model", 
            "Flickr R@1", "Flickr R@5", "Flickr MRR", "Flickr QPS",
            "COCO R@1", "COCO R@5", "COCO MRR", "COCO QPS",
            "DocVQA R@1", "DocVQA R@5", "DocVQA MRR", "DocVQA QPS", "DocVQA Embed",
            "Wino Group"
        ]
        final_keys = [k for k in order if k in keys]
        
        # Sort results by DocVQA R@1 or Flickr R@1 for better readability
        # Optional: results.sort(key=lambda x: float(x.get("DocVQA R@1", "0%").strip('%')), reverse=True)
        
        print(pd.DataFrame(results)[final_keys].to_markdown(index=False))
        pd.DataFrame(results).to_csv("benchmark_v11_results.csv", index=False)
