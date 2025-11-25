import torch
import gc
import sys
import os
import pandas as pd
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor

# --- COLPALI CHECK ---
try:
    from colpali_engine.models import ColPali, ColPaliProcessor
    COLPALI_AVAILABLE = True
except ImportError:
    COLPALI_AVAILABLE = False
    print("UYARI: ColPali engine bulunamadı.")

# --- AYARLAR ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Jina and others might prefer float16 if bfloat16 causes mismatch, but let's try casting first.
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 
SAMPLE_SIZE = 200 

print(f"Running on: {DEVICE}")
print(f"Precision: {DTYPE}")

# --- MODEL LİSTESİ (THE STABLE 5) ---
MODELS_TO_TEST = [
    # 1. ColPali (Document King)
    {"name": "ColPali-v1.3",  "id": "vidore/colpali-v1.3", "type": "colpali", "batch_size": 4}, 
    
    # 2. SigLIP (Google)
    {"name": "SigLIP-400M",   "id": "google/siglip-so400m-patch14-384", "type": "dense", "batch_size": 64},
    
    # 3. LAION Huge (Classic SOTA)
    {"name": "LAION-CLIP-H",  "id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "type": "dense", "batch_size": 32},
    
    # 4. OpenAI CLIP (Standard)
    {"name": "OpenAI-CLIP-L", "id": "openai/clip-vit-large-patch14-336", "type": "dense", "batch_size": 32},

    # 5. Jina CLIP (Modern - Fixed Dtype)
    {"name": "Jina-CLIP-v1", "id": "jinaai/jina-clip-v1", "type": "dense", "trust": True, "batch_size": 32},

    # 6. Apple DFN5B (The New Giant - Replaces BigG)
    {"name": "Apple-DFN5B-H", "id": "apple/DFN5B-CLIP-ViT-H-14-378", "type": "dense", "trust": True, "batch_size": 32}
]

def clean_memory():
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    torch.cuda.empty_cache()
    gc.collect()

def safe_get_text(item, col_name):
    val = item.get(col_name, "")
    if isinstance(val, list): return str(val[0]) if len(val)>0 else ""
    if isinstance(val, dict):
        for key in ['en', 'text', 'caption', 'query']:
            if key in val: return str(val[key])
        return str(val)
    return str(val)

def run_retrieval_benchmark(model, processor, model_info, dataset, dataset_name, text_col="caption", image_col="image"):
    print(f"  > Benchmarking {dataset_name}...")
    bs = model_info.get("batch_size", 32)
    
    try:
        images = [item[image_col].convert("RGB") for item in dataset]
        queries = [safe_get_text(item, text_col) for item in dataset]
    except Exception as e:
        print(f"    Data Prep Error: {e}")
        return {f"{dataset_name} R@1": "Error"}

    try:
        with torch.no_grad():
            # --- IMAGE EMBEDDINGS ---
            img_embeds_list = []
            print(f"    Processing images...")
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
                    
                    # Normalize & Cast to DTYPE explicitly
                    if out.dim() == 3: out = out[:, 0, :]
                    out = out / out.norm(dim=-1, keepdim=True)
                    img_embeds_list.append(out.to(DTYPE).cpu()) # Explicit cast
                del inputs, out
                torch.cuda.empty_cache()

            # --- TEXT EMBEDDINGS ---
            txt_embeds_list = []
            print(f"    Processing texts...")
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
                    txt_embeds_list.append(out.to(DTYPE).cpu()) # Explicit cast
                del inputs, out
                torch.cuda.empty_cache()

            # --- SCORING ---
            print(f"    Scoring...")
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
                all_img = torch.cat(img_embeds_list, dim=0).to(DTYPE) # Ensure DTYPE
                all_txt = torch.cat(txt_embeds_list, dim=0).to(DTYPE)
                scores = torch.matmul(all_txt, all_img.t())

        # --- METRICS ---
        scores = scores.float()
        r1, n = 0, len(queries)
        for i in range(n):
            if i == torch.topk(scores[i], k=1).indices.tolist()[0]: r1 += 1
        return {f"{dataset_name} R@1": f"{r1/n:.1%}"}
    except Exception as e:
        print(f"Inference Error: {e}")
        import traceback
        traceback.print_exc()
        return {f"{dataset_name} R@1": "Error"}

def run_winoground_safe(model, processor, model_info):
    print("  > Benchmarking Winoground (Safe Mode)...")
    try:
        dataset = load_dataset("facebook/winoground", split="test", trust_remote_code=True)
    except:
        return {"Winoground": "N/A"}
    
    group_score, total = 0, len(dataset)
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
                    
                    # Cast to DTYPE before matmul to match types
                    ie = ie.to(DTYPE)
                    te = te.to(DTYPE)
                    
                    s = torch.matmul(ie, te.t()).t() # [Text, Image]
            
            s = s.float().cpu()
            if (s[0,0] > s[1,0] and s[1,1] > s[0,1]) and (s[0,0] > s[0,1] and s[1,1] > s[1,0]):
                group_score += 1
        return {"Winoground": f"{group_score/total:.1%}"}
    except Exception as e:
        return {"Winoground": "Error"}

# --- MAIN ---
if __name__ == "__main__":
    print(">>> LOADING DATASETS...")
    try: ds_flickr = load_dataset("lmms-lab/flickr30k", split="test", trust_remote_code=True).select(range(SAMPLE_SIZE))
    except: ds_flickr = []
    try: ds_doc = load_dataset("nielsr/docvqa_1200_examples", split="test").select(range(SAMPLE_SIZE))
    except: ds_doc = []
    
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
        if len(ds_flickr) > 0: row.update(run_retrieval_benchmark(model, processor, m_info, ds_flickr, "Flickr", "caption"))
        if len(ds_doc) > 0: row.update(run_retrieval_benchmark(model, processor, m_info, ds_doc, "DocVQA", "query"))
        row.update(run_winoground_safe(model, processor, m_info))
        
        results.append(row)
        print(f"Result: {row}")

    print("\n" + "="*80)
    print("GRAND SLAM BENCHMARK RESULTS (V8 - THE FIXER)")
    print("="*80)
    headers = ["Model", "Flickr R@1", "DocVQA R@1", "Winoground"]
    print(f"{headers[0]:<20} | {headers[1]:<12} | {headers[2]:<12} | {headers[3]:<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['Model']:<20} | {r.get('Flickr R@1', 'N/A'):<12} | {r.get('DocVQA R@1', 'N/A'):<12} | {r.get('Winoground', 'N/A'):<12}")
