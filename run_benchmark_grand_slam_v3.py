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
    print("UYARI: ColPali engine bulunamadı. (pip install colpali-engine)")

# --- AYARLAR ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 
SAMPLE_SIZE = 200 

print(f"Running on: {DEVICE}")
print(f"Precision: {DTYPE}")

# --- MODEL LİSTESİ ---
MODELS_TO_TEST = [
    {"name": "ColPali-v1.3",  "id": "vidore/colpali-v1.3", "type": "colpali", "batch_size": 4}, # Batch size critical for memory
    {"name": "EVA02-CLIP-L",  "id": "QuanSun/EVA02-CLIP-L-14", "trust": True, "type": "dense", "batch_size": 32},
    {"name": "SigLIP-400M",   "id": "google/siglip-so400m-patch14-384", "type": "dense", "batch_size": 64},
    {"name": "LAION-CLIP-H",  "id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "type": "dense", "batch_size": 32},
    {"name": "BGE-Visualized", "id": "BAAI/bge-visualized-base-en-v1.5", "type": "dense", "trust": True, "batch_size": 32}
]

def clean_memory():
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    torch.cuda.empty_cache()
    gc.collect()

def run_retrieval_benchmark(model, processor, model_info, dataset, dataset_name, text_col="caption", image_col="image"):
    print(f"  > Benchmarking {dataset_name}...")
    
    bs = model_info.get("batch_size", 32)
    
    # 1. Veri Hazırlığı
    try:
        images = [item[image_col].convert("RGB") for item in dataset]
        queries = []
        for item in dataset:
            val = item[text_col]
            if isinstance(val, list): queries.append(val[0])
            else: queries.append(val)
    except Exception as e:
        return {f"{dataset_name} R@1": "Error", f"{dataset_name} R@5": str(e)}

    # 2. Inference (Batched)
    try:
        with torch.no_grad():
            # --- IMAGE EMBEDDINGS ---
            img_embeds_list = []
            print(f"    Processing images (Batch: {bs})...")
            for i in range(0, len(images), bs):
                batch_imgs = images[i : i+bs]
                
                if model_info["type"] == "colpali":
                    # ColPali: Process -> GPU -> Embed -> CPU List
                    inputs = processor.process_images(batch_imgs).to(DEVICE)
                    out = model(**inputs)
                    # out is list of tensors. Move each to CPU to save VRAM
                    img_embeds_list.extend([o.cpu() for o in out])
                else:
                    # Dense
                    inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(DEVICE)
                    if hasattr(model, 'get_image_features'):
                        out = model.get_image_features(**inputs)
                    else:
                        o = model(**inputs)
                        out = o.image_embeds if hasattr(o, 'image_embeds') else o[0]
                    
                    if out.dim() == 3: out = out[:, 0, :]
                    out = out / out.norm(dim=-1, keepdim=True)
                    img_embeds_list.append(out.cpu()) # Move batch to CPU
                
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
                # ColPali Scoring (Needs GPU usually, but we do it in blocks)
                # processor.score expects inputs on the same device.
                # We will loop through queries, move them + all images to GPU, score, then CPU.
                # If 200 images don't fit on GPU, we need double loop. 
                # 200 ColPali images ~10GB. Should fit.
                
                scores_list = []
                # Move all image embeddings to GPU once if possible, or keep on CPU and batch move
                # Let's try batching the scoring to be super safe.
                
                # Process queries one by one (or small batch) against ALL images
                score_bs = 10 # Queries per step
                
                # We need to stack images temporarily for scoring if processor supports it
                # processor.score(qs, docs) -> qs: list of tensors, docs: list of tensors
                
                all_scores = []
                for i in range(0, len(txt_embeds_list), score_bs):
                    q_batch_cpu = txt_embeds_list[i : i+score_bs]
                    q_batch_gpu = [q.to(DEVICE) for q in q_batch_cpu]
                    
                    # Score against all images. 
                    # If OOM here, we need to chunk images too.
                    # Let's try moving all images to GPU. If 200, it might be tight with Query.
                    # Chunking images is safer.
                    
                    batch_scores = []
                    img_chunk_size = 20 # Score against 20 images at a time
                    
                    for j in range(0, len(img_embeds_list), img_chunk_size):
                        i_batch_cpu = img_embeds_list[j : j+img_chunk_size]
                        i_batch_gpu = [img.to(DEVICE) for img in i_batch_cpu]
                        
                        # Score (q_batch vs i_batch) -> Shape (score_bs, img_chunk_size)
                        s = processor.score(q_batch_gpu, i_batch_gpu)
                        batch_scores.append(s.cpu())
                        
                        del i_batch_gpu, s
                        torch.cuda.empty_cache()
                    
                    # Concatenate scores for these queries against all images
                    # batch_scores is list of (score_bs, 20), (score_bs, 20)...
                    full_row_scores = torch.cat(batch_scores, dim=1) # (score_bs, total_images)
                    all_scores.append(full_row_scores)
                    
                    del q_batch_gpu
                    torch.cuda.empty_cache()
                
                scores = torch.cat(all_scores, dim=0) # (total_queries, total_images)

            else:
                # Dense Scoring (CPU is fine and fast for 200x200)
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

def run_winoground(model, processor, model_info):
    print("  > Benchmarking Winoground...")
    try:
        dataset = load_dataset("facebook/winoground", split="test", trust_remote_code=True)
    except:
        return {"Winoground Score": "Fail"}
        
    group_score = 0
    total = len(dataset)
    
    # Process one by one to be safe
    for example in dataset:
        images = [example["image_0"].convert("RGB"), example["image_1"].convert("RGB")]
        texts = [example["caption_0"], example["caption_1"]]
        
        with torch.no_grad():
            if model_info["type"] == "colpali":
                bi = processor.process_images(images).to(DEVICE)
                bq = processor.process_queries(texts).to(DEVICE)
                # ColPali score -> [Text, Image]
                s = processor.score(model(**bq), model(**bi))
            else:
                # Dense
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
                s = torch.matmul(ie, te.t()).t()
        
        s = s.float().cpu()
        t_ok = (s[0,0] > s[1,0]) and (s[1,1] > s[0,1])
        i_ok = (s[0,0] > s[0,1]) and (s[1,1] > s[1,0])
        if t_ok and i_ok: group_score += 1
        
    return {"Winoground Score": f"{group_score/total:.1%}"}

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(">>> LOADING DATASETS (Safe Mode)...")
    
    # Datasets
    ds_flickr = []
    ds_doc = []
    ds_diff = []
    
    try:
        ds_flickr = load_dataset("lmms-lab/flickr30k", split="test", trust_remote_code=True).select(range(SAMPLE_SIZE))
        print("Flickr Loaded.")
    except: print("Flickr Failed.")

    try:
        print("Downloading DocVQA (nielsr/docvqa_1200_examples)...")
        ds_doc = load_dataset("nielsr/docvqa_1200_examples", split="test").select(range(SAMPLE_SIZE))
        print("DocVQA Loaded.")
    except: print("DocVQA Failed.")
    
    try:
        print("Downloading DiffusionDB...")
        ds_diff_stream = load_dataset("poloclub/diffusiondb", "2m_first_1k", split="train", streaming=True)
        ds_diff = list(ds_diff_stream.take(SAMPLE_SIZE))
        print("DiffusionDB Loaded.")
    except: print("DiffusionDB Failed.")
    
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
        if len(ds_diff) > 0: row.update(run_retrieval_benchmark(model, processor, m_info, ds_diff, "DiffDB", "prompt"))
        row.update(run_winoground(model, processor, m_info))
        
        results.append(row)
        print(f"Result: {row}")

    print("\n" + "="*80)
    print("GRAND SLAM BENCHMARK RESULTS (V3 - MEMORY SAFE)")
    print("="*80)
    headers = ["Model", "Flickr R@1", "DocVQA R@1", "DiffDB R@1", "Winoground"]
    print(f"{headers[0]:<20} | {headers[1]:<12} | {headers[2]:<12} | {headers[3]:<12} | {headers[4]:<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['Model']:<20} | {r.get('Flickr R@1', 'N/A'):<12} | {r.get('DocVQA R@1', 'N/A'):<12} | {r.get('DiffDB R@1', 'N/A'):<12} | {r.get('Winoground Score', 'N/A'):<12}")
