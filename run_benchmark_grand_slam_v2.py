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
# 4090 supports bfloat16 which is more stable and efficient
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 
SAMPLE_SIZE = 200 # Number of samples per dataset

print(f"Running on: {DEVICE}")
print(f"Precision: {DTYPE}")

# --- MODEL LIST ---
MODELS_TO_TEST = [
    # 1. ColPali (Document King)
    {"name": "ColPali-v1.3",  "id": "vidore/colpali-v1.3", "type": "colpali"},
    
    # 2. EVA02 (Photo King) - Force fast tokenizer if needed
    {"name": "EVA02-CLIP-L",  "id": "QuanSun/EVA02-CLIP-L-14", "trust": True, "type": "dense"},
    
    # 3. SigLIP (Google Balance)
    {"name": "SigLIP-400M",   "id": "google/siglip-so400m-patch14-384", "type": "dense"},
    
    # 4. LAION Huge (Massive Data)
    {"name": "LAION-CLIP-H",  "id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "type": "dense"},
    
    # 5. BGE-Visualized (RAG Expert)
    {"name": "BGE-Visualized", "id": "BAAI/bge-visualized-base-en-v1.5", "type": "dense", "trust": True}
]

def clean_memory():
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    torch.cuda.empty_cache()
    gc.collect()

def run_retrieval_benchmark(model, processor, model_info, dataset, dataset_name, text_col="caption", image_col="image"):
    print(f"  > Benchmarking {dataset_name}...")
    
    # Data Preparation
    try:
        images = [item[image_col].convert("RGB") for item in dataset]
        
        # Text column can be a list or a string. Normalize to list of strings.
        queries = []
        for item in dataset:
            val = item[text_col]
            if isinstance(val, list): queries.append(val[0])
            else: queries.append(val)
            
    except Exception as e:
        print(f"Error preparing data for {dataset_name}: {e}")
        return {f"{dataset_name} R@1": "Error", f"{dataset_name} R@5": str(e)}

    # --- INFERENCE ---
    try:
        with torch.no_grad():
            if model_info["type"] == "colpali":
                # ColPali Flow
                # Process images in batches to avoid OOM if needed, but for 200 samples it might fit on 4090
                # ColPali processor handles list of images
                batch_images = processor.process_images(images).to(DEVICE)
                image_embeds = model(**batch_images)
                
                batch_queries = processor.process_queries(queries).to(DEVICE)
                query_embeddings = model(**batch_queries)
                
                # Score [Query, Image]
                scores = processor.score(query_embeddings, image_embeddings)
                
            else:
                # Dense Flow
                inputs_img = processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
                
                if hasattr(model, 'get_image_features'):
                    img_out = model.get_image_features(**inputs_img)
                else:
                    out = model(**inputs_img)
                    img_out = out.image_embeds if hasattr(out, 'image_embeds') else out[0]
                
                inputs_text = processor(text=queries, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                if hasattr(model, 'get_text_features'):
                    text_out = model.get_text_features(**inputs_text)
                else:
                    out = model(**inputs_text)
                    text_out = out.text_embeds if hasattr(out, 'text_embeds') else out[0]

                # 3D Fix (for models like BGE that might return specific shapes)
                if img_out.dim() == 3: img_out = img_out[:, 0, :]
                if text_out.dim() == 3: text_out = text_out[:, 0, :]

                # Normalize & Matmul
                img_out = img_out / img_out.norm(dim=-1, keepdim=True)
                text_out = text_out / text_out.norm(dim=-1, keepdim=True)
                scores = torch.matmul(text_out, img_out.t())

        # --- METRICS ---
        scores = scores.float().cpu()
        r1, r5 = 0, 0
        n = len(queries)
        
        for i in range(n):
            # Is the i-th image the best match for the i-th query?
            top_k = torch.topk(scores[i], k=5).indices.tolist()
            if i == top_k[0]: r1 += 1
            if i in top_k: r5 += 1
                
        return {
            f"{dataset_name} R@1": f"{r1/n:.1%}",
            f"{dataset_name} R@5": f"{r5/n:.1%}"
        }
    except Exception as e:
        print(f"Inference Error on {dataset_name}: {e}")
        return {f"{dataset_name} R@1": "Error", f"{dataset_name} R@5": "Error"}

def run_winoground(model, processor, model_info):
    print("  > Benchmarking Winoground (Logic)...")
    try:
        dataset = load_dataset("facebook/winoground", split="test", trust_remote_code=True)
    except:
        print("Winoground load failed.")
        return {"Winoground Score": "Fail"}
        
    text_score, image_score, group_score = 0, 0, 0
    total = len(dataset)
    
    for example in dataset:
        images = [example["image_0"].convert("RGB"), example["image_1"].convert("RGB")]
        texts = [example["caption_0"], example["caption_1"]]
        
        with torch.no_grad():
            if model_info["type"] == "colpali":
                bi = processor.process_images(images).to(DEVICE)
                bq = processor.process_queries(texts).to(DEVICE)
                # ColPali score -> [Text, Image]
                logits = processor.score(model(**bq), model(**bi))
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
                # Standard: [Image, Text] -> Transpose -> [Text, Image]
                logits = torch.matmul(ie, te.t()).t()
        
        logits = logits.float().cpu()
        c0_i0, c0_i1 = logits[0,0], logits[0,1]
        c1_i0, c1_i1 = logits[1,0], logits[1,1]
        
        t_ok = (c0_i0 > c1_i0) and (c1_i1 > c0_i1)
        i_ok = (c0_i0 > c0_i1) and (c1_i1 > c1_i0)
        if t_ok and i_ok: group_score += 1
        
    return {"Winoground Score": f"{group_score/total:.1%}"}

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Download Datasets (Fail-safe)
    print(">>> LOADING DATASETS (Safe Mode)...")
    
    # Flickr (Vision)
    try:
        ds_flickr = load_dataset("lmms-lab/flickr30k", split="test", trust_remote_code=True).select(range(SAMPLE_SIZE))
        print("Flickr30k Loaded.")
    except Exception as e:
        print(f"Flickr failed: {e}")
        ds_flickr = []

    # DocVQA (Document) - Niels Rogge version (Safe)
    try:
        print("Downloading DocVQA (nielsr/docvqa_1200_examples)...")
        ds_doc = load_dataset("nielsr/docvqa_1200_examples", split="test").select(range(SAMPLE_SIZE))
        print("DocVQA Loaded.")
    except Exception as e:
        print(f"DocVQA failed: {e}")
        ds_doc = []
    
    # DiffusionDB (AI Art/Prompt) - Safe replacement for COCO
    try:
        print("Downloading DiffusionDB (poloclub/diffusiondb)...")
        # Streaming is safer for huge datasets like DiffusionDB
        ds_diff_stream = load_dataset("poloclub/diffusiondb", "2m_first_1k", split="train", streaming=True)
        ds_diff = list(ds_diff_stream.take(SAMPLE_SIZE)) # Convert first 200 to list
        print("DiffusionDB Loaded.")
    except Exception as e:
        print(f"DiffusionDB failed: {e}")
        ds_diff = []
    
    results = []

    for m_info in MODELS_TO_TEST:
        m_name = m_info["name"]
        print(f"\n>>> TESTING: {m_name}")
        clean_memory()
        
        # Load Model
        try:
            trust = m_info.get("trust", False)
            if m_info["type"] == "colpali":
                if not COLPALI_AVAILABLE: 
                    print("ColPali library missing, skipping.")
                    continue
                model = ColPali.from_pretrained(m_info["id"], torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=trust).eval()
                processor = ColPaliProcessor.from_pretrained(m_info["id"], trust_remote_code=trust)
            else:
                model = AutoModel.from_pretrained(m_info["id"], trust_remote_code=trust, torch_dtype=DTYPE).to(DEVICE)
                processor = AutoProcessor.from_pretrained(m_info["id"], trust_remote_code=trust)
                model.eval()
        except Exception as e:
            print(f"Load Error for {m_name}: {e}")
            continue

        # Scoreboard Row
        row = {"Model": m_name}
        
        # 1. Flickr
        if len(ds_flickr) > 0:
            row.update(run_retrieval_benchmark(model, processor, m_info, ds_flickr, "Flickr", text_col="caption"))
        
        # 2. DocVQA (Column names: 'image', 'query')
        if len(ds_doc) > 0:
            row.update(run_retrieval_benchmark(model, processor, m_info, ds_doc, "DocVQA", text_col="query"))
        
        # 3. DiffusionDB (Column names: 'image', 'prompt')
        if len(ds_diff) > 0:
            row.update(run_retrieval_benchmark(model, processor, m_info, ds_diff, "DiffDB", text_col="prompt"))
        
        # 4. Winoground
        row.update(run_winoground(model, processor, m_info))
        
        results.append(row)
        print(f"Result: {row}")

    # --- REPORTING ---
    print("\n" + "="*80)
    print("GRAND SLAM BENCHMARK RESULTS (RTX 4090)")
    print("="*80)
    
    headers = ["Model", "Flickr R@1", "DocVQA R@1", "DiffDB R@1", "Winoground Score"]
    print(f"{headers[0]:<20} | {headers[1]:<12} | {headers[2]:<12} | {headers[3]:<12} | {headers[4]:<15}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['Model']:<20} | {r.get('Flickr R@1', 'N/A'):<12} | {r.get('DocVQA R@1', 'N/A'):<12} | {r.get('DiffDB R@1', 'N/A'):<12} | {r.get('Winoground Score', 'N/A'):<15}")
