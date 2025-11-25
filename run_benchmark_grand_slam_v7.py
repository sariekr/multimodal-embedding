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
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 
SAMPLE_SIZE = 200 

print(f"Running on: {DEVICE}")
print(f"Precision: {DTYPE}")

# --- MODEL LİSTESİ (ŞAMPİYONLAR LİGİ) ---
MODELS_TO_TEST = [
    # 1. ColPali (Document) - Proven
    {"name": "ColPali-v1.3",  "id": "vidore/colpali-v1.3", "type": "colpali", "batch_size": 4}, 
    
    # 2. SigLIP (Google) - Proven
    {"name": "SigLIP-400M",   "id": "google/siglip-so400m-patch14-384", "type": "dense", "batch_size": 64},
    
    # 3. LAION Huge (Classic SOTA) - Proven
    {"name": "LAION-CLIP-H",  "id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "type": "dense", "batch_size": 32},
    
    # 4. CLIP BigG (The Giant) - Replacement for EVA02
    {"name": "CLIP-BigG-14", "id": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "type": "dense", "batch_size": 16}, # Big model, smaller batch

    # 5. Jina CLIP (Popular Modern)
    {"name": "Jina-CLIP-v1", "id": "jinaai/jina-clip-v1", "type": "dense", "trust": True, "batch_size": 32},

    # 6. Nomic Embed Vision (New Gen)
    {"name": "Nomic-Vision-1.5", "id": "nomic-ai/nomic-embed-vision-v1.5", "type": "dense", "trust": True, "batch_size": 32},

    # 7. BGE Visualized (RAG Expert) - Correct ID usually needs checking, trying safe base
    # 'BAAI/bge-visualized' often refers to the repo, model weights are inside.
    # Let's use 'BAAI/bge-visualized-base-en-v1.5' if available, otherwise skip to avoid crash.
    # Update: User list link points to 'BAAI/bge-visualized', let's try exact ID if possible.
    # Fallback to a very popular specific version:
    {"name": "BGE-Visualized", "id": "BAAI/bge-visualized-base-en-v1.5", "type": "dense", "trust": True, "batch_size": 32}
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
                    # Handle Jina/Nomic specific trust_remote_code logic often built-in
                    inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(DEVICE)
                    
                    # Model specific forward passes
                    if "jina" in model_info["id"]:
                        # Jina often uses .get_image_features or direct forward
                        out = model.get_image_features(**inputs)
                    elif "nomic" in model_info["id"]:
                        # Nomic typically uses .encode or similar, but usually follows HF API
                        # Let's try standard first, catch error if needed.
                        # Nomic vision model output might be raw
                        out = model.vision_model(**inputs).last_hidden_state[:, 0, :] # CLS token approach for some
                        # Actually nomic-embed-vision-v1.5 is AutoModel. 
                        # Often returns pooler_output or embeddings.
                        # Let's rely on standard AutoModel behavior first.
                        out = model(**inputs).last_hidden_state[:, 0, :] # Failsafe for some vit
                        # Better: check for proper method
                        if hasattr(model, 'get_image_features'): out = model.get_image_features(**inputs)
                    
                    elif hasattr(model, 'get_image_features'): 
                        out = model.get_image_features(**inputs)
                    else: 
                        out = model(**inputs).image_embeds
                    
                    # Normalize
                    if out.dim() == 3: out = out[:, 0, :]
                    out = out / out.norm(dim=-1, keepdim=True)
                    img_embeds_list.append(out.cpu()) 
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
                    
                    if "jina" in model_info["id"]:
                        out = model.get_text_features(**inputs)
                    elif "nomic" in model_info["id"]:
                         # Nomic text part is tricky, often paired with Nomic-Embed-Text
                         # Wait, nomic-embed-vision is ONLY vision. It needs a text pair?
                         # Description says: "aligns with nomic-embed-text-v1.5"
                         # If so, we need to load a separate text model for Nomic.
                         # To keep script simple, if Nomic fails text encoding, we might skip it or treat it as vision-only.
                         # Actually, let's remove Nomic if it requires a separate text model to keep script single-model focused.
                         # REPLACEMENT: CLIP-ViT-L-14-DataComp.XL (Proven single model)
                         pass 
                         
                    elif hasattr(model, 'get_text_features'): 
                        out = model.get_text_features(**inputs)
                    else: 
                        out = model(**inputs).text_embeds
                    
                    if out.dim() == 3: out = out[:, 0, :]
                    out = out / out.norm(dim=-1, keepdim=True)
                    txt_embeds_list.append(out.cpu())
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
                all_img = torch.cat(img_embeds_list, dim=0)
                all_txt = torch.cat(txt_embeds_list, dim=0)
                scores = torch.matmul(all_txt, all_img.t())

        # --- METRICS ---
        scores = scores.float()
        r1, n = 0, len(queries)
        for i in range(n):
            if i == torch.topk(scores[i], k=1).indices.tolist()[0]: r1 += 1
        return {f"{dataset_name} R@1": f"{r1/n:.1%}"}
    except Exception as e:
        print(f"Inference Error: {e}")
        return {f"{dataset_name} R@1": "Error"}

def run_winoground_safe(model, processor, model_info):
    print("  > Benchmarking Winoground (Safe Mode)...")
    try: dataset = load_dataset("facebook/winoground", split="test", trust_remote_code=True)
    except: return {"Winoground": "N/A"}
    
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
    
    # Filter Nomic from runtime logic if present in list because it needs separate text model
    # We will use valid multimodal models only
    FINAL_MODELS = [m for m in MODELS_TO_TEST if "nomic" not in m["id"]]

    for m_info in FINAL_MODELS:
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
    print("GRAND SLAM BENCHMARK RESULTS (V7 - CHAMPIONS LEAGUE)")
    print("="*80)
    headers = ["Model", "Flickr R@1", "DocVQA R@1", "Winoground"]
    print(f"{headers[0]:<20} | {headers[1]:<12} | {headers[2]:<12} | {headers[3]:<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['Model']:<20} | {r.get('Flickr R@1', 'N/A'):<12} | {r.get('DocVQA R@1', 'N/A'):<12} | {r.get('Winoground', 'N/A'):<12}")
