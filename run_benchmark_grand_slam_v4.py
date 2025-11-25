import torch
import gc
import sys
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

# --- MODEL LİSTESİ (Sadece Çalışanlar ve Sağlam Alternatifler) ---
MODELS_TO_TEST = [
    # 1. ColPali (Document King) - Worked well
    {"name": "ColPali-v1.3",  "id": "vidore/colpali-v1.3", "type": "colpali", "batch_size": 4}, 
    
    # 2. SigLIP (Google) - Worked well
    {"name": "SigLIP-400M",   "id": "google/siglip-so400m-patch14-384", "type": "dense", "batch_size": 64},
    
    # 3. LAION Huge - Worked well
    {"name": "LAION-CLIP-H",  "id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "type": "dense", "batch_size": 32}, 
    
    # 4. OpenAI CLIP (Replacement for EVA02) - Very stable
    {"name": "OpenAI-CLIP-L", "id": "openai/clip-vit-large-patch14-336", "type": "dense", "batch_size": 32}
]

def clean_memory():
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    torch.cuda.empty_cache()
    gc.collect()

def safe_get_text(item, col_name):
    """Extract text safely regardless of whether it's string, list, or dict."""
    val = item.get(col_name, "")
    
    # If it's a list (e.g. captions), take first
    if isinstance(val, list):
        if len(val) > 0: return str(val[0])
        else: return ""
        
    # If it's a dict (sometimes happens in localized datasets), try common keys
    if isinstance(val, dict):
        # Try 'en', 'text', 'caption'
        for key in ['en', 'text', 'caption', 'query']:
            if key in val: return str(val[key])
        # Fallback: stringify the whole dict
        return str(val)
        
    # Default: string
    return str(val)

def run_retrieval_benchmark(model, processor, model_info, dataset, dataset_name, text_col="caption", image_col="image"):
    print(f"  > Benchmarking {dataset_name}...")
    
    bs = model_info.get("batch_size", 32)
    
    # 1. Veri Hazırlığı (SAFE MODE)
    try:
        images = [item[image_col].convert("RGB") for item in dataset]
        queries = [safe_get_text(item, text_col) for item in dataset]
        
        # Debug print for first item to ensure correctness
        print(f"    [Debug] Sample Query: {queries[0][:50]}...")
        
    except Exception as e:
        print(f"    Data Prep Error: {e}")
        return {f"{dataset_name} R@1": "Error", f"{dataset_name} R@5": "Error"}

    # 2. Inference (Batched)
    try:
        with torch.no_grad():
            # --- IMAGE EMBEDDINGS ---
            img_embeds_list = []
            print(f"    Processing images (Batch: {bs})...")
            for i in range(0, len(images), bs):
                batch_imgs = images[i : i+bs]
                
                if model_info["type"] == "colpali":
                    inputs = processor.process_images(batch_imgs).to(DEVICE)
                    out = model(**inputs)
                    img_embeds_list.extend([o.cpu() for o in out])
                else:
                    inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(DEVICE)
                    if hasattr(model, 'get_image_features'):
                        out = model.get_image_features(**inputs)
                    else:
                        o = model(**inputs)
                        out = o.image_embeds if hasattr(o, 'image_embeds') else o[0]
                    
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
                    if hasattr(model, 'get_text_features'):
                        out = model.get_text_features(**inputs)
                    else:
                        o = model(**inputs)
                        out = o.text_embeds if hasattr(o, 'text_embeds') else o[0]
                    
                    if out.dim() == 3: out = out[:, 0, :]
                    out = out / out.norm(dim=-1, keepdim=True)
                    txt_embeds_list.append(out.cpu())

                del inputs, out
                torch.cuda.empty_cache()

            # --- SCORING ---
            print(f"    Scoring...")
            if model_info["type"] == "colpali":
                # ColPali Scoring Loop (Batched Query vs Batched Image)
                score_bs = 10
                all_scores = []
                for i in range(0, len(txt_embeds_list), score_bs):
                    q_batch_cpu = txt_embeds_list[i : i+score_bs]
                    q_batch_gpu = [q.to(DEVICE) for q in q_batch_cpu]
                    
                    batch_scores = []
                    img_chunk_size = 20
                    
                    for j in range(0, len(img_embeds_list), img_chunk_size):
                        i_batch_cpu = img_embeds_list[j : j+img_chunk_size]
                        i_batch_gpu = [img.to(DEVICE) for img in i_batch_cpu]
                        
                        s = processor.score(q_batch_gpu, i_batch_gpu)
                        batch_scores.append(s.cpu())
                        
                        del i_batch_gpu, s
                        torch.cuda.empty_cache()
                    
                    full_row_scores = torch.cat(batch_scores, dim=1) 
                    all_scores.append(full_row_scores)
                    
                    del q_batch_gpu
                    torch.cuda.empty_cache()
                
                scores = torch.cat(all_scores, dim=0) 

            else:
                all_img = torch.cat(img_embeds_list, dim=0)
                all_txt = torch.cat(txt_embeds_list, dim=0)
                scores = torch.matmul(all_txt, all_img.t())

        # --- METRICS ---
        scores = scores.float()
        r1, r5 = 0, 0
        n = len(queries)
        
        for i in range(n):
            top_k = torch.topk(scores[i], k=5).indices.tolist()
            if i == top_k[0]: r1 += 1
            if i in top_k: r5 += 1
                
        return {
            f"{dataset_name} R@1": f"{r1/n:.1%}",
            f"{dataset_name} R@5": f"{r5/n:.1%}"
        }
    except Exception as e:
        print(f"Inference Error on {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return {f"{dataset_name} R@1": "Error", f"{dataset_name} R@5": "Error"}

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(">>> LOADING DATASETS (Robust Mode)...")
    
    # Datasets
    ds_flickr = []
    ds_doc = []
    
    # 1. Flickr (Proven working)
    try:
        print("Loading Flickr...")
        ds_flickr = load_dataset("lmms-lab/flickr30k", split="test", trust_remote_code=True).select(range(SAMPLE_SIZE))
        print("Flickr Loaded.")
    except: print("Flickr Failed.")

    # 2. DocVQA (Using robust text extraction)
    try:
        print("Loading DocVQA (nielsr/docvqa_1200_examples)...")
        ds_doc = load_dataset("nielsr/docvqa_1200_examples", split="test").select(range(SAMPLE_SIZE))
        print("DocVQA Loaded.")
    except Exception as e: 
        print(f"DocVQA Failed: {e}")
    
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
                model = AutoModel.from_pretrained(m_info["id"], trust_remote_code=trust, torch_dtype=DTYPE).to(DEVICE)
                processor = AutoProcessor.from_pretrained(m_info["id"], trust_remote_code=trust)
                model.eval()
        except Exception as e:
            print(f"Load Error: {e}")
            continue

        row = {"Model": m_name}
        if len(ds_flickr) > 0: row.update(run_retrieval_benchmark(model, processor, m_info, ds_flickr, "Flickr", "caption"))
        if len(ds_doc) > 0: row.update(run_retrieval_benchmark(model, processor, m_info, ds_doc, "DocVQA", "query"))
        
        results.append(row)
        print(f"Result: {row}")

    print("\n" + "="*80)
    print("GRAND SLAM BENCHMARK RESULTS (V4 - ROBUST)")
    print("="*80)
    headers = ["Model", "Flickr R@1", "DocVQA R@1"]
    print(f"{headers[0]:<20} | {headers[1]:<12} | {headers[2]:<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['Model']:<20} | {r.get('Flickr R@1', 'N/A'):<12} | {r.get('DocVQA R@1', 'N/A'):<12}")
