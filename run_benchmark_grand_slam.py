import torch
import gc
import pandas as pd
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor
from PIL import Image
import sys

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# RTX 4090 supports bfloat16 which is more stable for training/inference than float16
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 
SAMPLE_SIZE = 200  # Sample size for retrieval datasets
WINOGROUND_SIZE = 400 # Winoground is small, use all
BATCH_SIZE = 32    # Adjust based on VRAM. 32 is usually safe for 24GB VRAM (except ColPali)

print(f"Running on: {DEVICE}")
print(f"Precision: {DTYPE}")

# --- MODEL LIST (THE HEAVYWEIGHTS) ---
MODELS = [
    {
        "name": "ColPali-v1.3 (3B)", 
        "id": "vidore/colpali-v1.3", 
        "type": "colpali",
        "batch_size": 4 # ColPali consumes A LOT of VRAM per image
    },
    {
        "name": "SigLIP-SO400M", 
        "id": "google/siglip-so400m-patch14-384", 
        "type": "dense",
        "batch_size": 64
    },
    {
        "name": "LAION-CLIP-H (Huge)", 
        "id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", 
        "type": "dense",
        "batch_size": 64
    },
    {
        "name": "BGE-Visualized-Base", 
        "id": "BAAI/bge-visualized-base-en-v1.5", 
        "type": "dense",
        "batch_size": 64,
        "trust": True
    },
    # EVA02 is tricky in Transformers. Using the strongest available OpenCLIP-compatible or similar.
    # If this fails, we fallback to standard CLIP-BigG.
    {
        "name": "CLIP-BigG-14 (LAION)", 
        "id": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", 
        "type": "dense",
        "batch_size": 32
    }
]

# --- HELPER FUNCTIONS ---

def clean_memory():
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    torch.cuda.empty_cache()
    gc.collect()

def load_model(model_info):
    print(f"LOADING: {model_info['name']}...")
    try:
        trust = model_info.get("trust", False)
        if model_info["type"] == "colpali":
            from colpali_engine.models import ColPali, ColPaliProcessor
            model = ColPali.from_pretrained(
                model_info["id"], 
                torch_dtype=DTYPE, 
                device_map=DEVICE,
                trust_remote_code=trust
            ).eval()
            processor = ColPaliProcessor.from_pretrained(model_info["id"], trust_remote_code=trust)
        else:
            model = AutoModel.from_pretrained(
                model_info["id"], 
                torch_dtype=DTYPE, 
                trust_remote_code=trust
            ).to(DEVICE).eval()
            processor = AutoProcessor.from_pretrained(model_info["id"], trust_remote_code=trust)
        return model, processor
    except Exception as e:
        print(f"FAILED to load {model_info['name']}: {e}")
        return None, None

# --- LOGIC: STANDARD RETRIEVAL (Flickr, COCO, DocVQA) ---
def run_retrieval_benchmark(model, processor, model_info, dataset, dataset_name, text_col="caption"):
    print(f"--- Benchmarking {dataset_name} ---")
    
    # Prepare Data
    images = [item["image"].convert("RGB") for item in dataset]
    texts = []
    for item in dataset:
        t = item[text_col]
        if isinstance(t, list): texts.append(t[0]) # Take first caption
        else: texts.append(t)
    
    bs = model_info.get("batch_size", 32)
    
    # --- EMBEDDINGS ---
    try:
        with torch.no_grad():
            # Encode Images
            image_embeds_list = []
            for i in range(0, len(images), bs):
                batch_imgs = images[i : i+bs]
                if model_info["type"] == "colpali":
                    # ColPali returns list of tensors
                    inputs = processor.process_images(batch_imgs).to(DEVICE)
                    batch_emb = model(**inputs)
                    image_embeds_list.extend([b for b in batch_emb]) # Keep as list of tensors
                else:
                    inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(DEVICE)
                    out = model.get_image_features(**inputs) if hasattr(model, "get_image_features") else model(**inputs).image_embeds
                    out = out / out.norm(dim=-1, keepdim=True)
                    image_embeds_list.append(out)
            
            if model_info["type"] != "colpali":
                image_embeds = torch.cat(image_embeds_list, dim=0)

            # Encode Texts
            text_embeds_list = []
            for i in range(0, len(texts), bs):
                batch_txt = texts[i : i+bs]
                if model_info["type"] == "colpali":
                    inputs = processor.process_queries(batch_txt).to(DEVICE)
                    batch_emb = model(**inputs)
                    text_embeds_list.extend([b for b in batch_emb])
                else:
                    inputs = processor(text=batch_txt, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                    out = model.get_text_features(**inputs) if hasattr(model, "get_text_features") else model(**inputs).text_embeds
                    out = out / out.norm(dim=-1, keepdim=True)
                    text_embeds_list.append(out)
            
            if model_info["type"] != "colpali":
                text_embeds = torch.cat(text_embeds_list, dim=0)

        # --- SCORING ---
        if model_info["type"] == "colpali":
            # ColPali scoring (MaxSim / Late Interaction)
            # This is slower, requires loop
            scores_list = []
            # Process chunk by chunk to avoid OOM on score matrix
            qs_batch_list = processor.process_queries(texts).to(DEVICE)
            # We need to manually loop for ColPali scoring if dataset is large, 
            # but for 200 items, processor.score might fit if optimized.
            # Let's loop over queries
            scores_matrix = []
            for q_emb in text_embeds_list:
                 # Score one query against all images
                 # q_emb: (1, num_patches, dim)
                 # images: list of (1, num_patches, dim)
                 # processor.score handles batched queries vs list of doc embeddings
                 # But here we have lists. Let's use the processor's score method carefully.
                 pass 
            
            # Correct ColPali usage: processor.score(qs, docs)
            # We will do it in batches of queries to be safe
            scores_matrix = processor.score(text_embeds_list, image_embeds_list)

        else:
            # Dense Scoring
            scores_matrix = torch.matmul(text_embeds, image_embeds.t())

        # --- METRICS (Recall) ---
        scores_matrix = scores_matrix.float().cpu()
        r1 = 0
        r5 = 0
        n = len(texts)
        
        for i in range(n):
            scores = scores_matrix[i]
            top_k = torch.topk(scores, k=5).indices.tolist()
            if i == top_k[0]: r1 += 1
            if i in top_k: r5 += 1
            
        return {
            f"{dataset_name} R@1": f"{r1/n:.2%}",
            f"{dataset_name} R@5": f"{r5/n:.2%}"
        }

    except Exception as e:
        print(f"Error in retrieval benchmark: {e}")
        return {f"{dataset_name} R@1": "ERR", f"{dataset_name} R@5": "ERR"}


# --- LOGIC: WINOGROUND (Group Score) ---
def run_winoground_benchmark(model, processor, model_info, dataset):
    print("--- Benchmarking Winoground (IQ Test) ---")
    # Winoground requires comparing (T0, I0), (T0, I1), (T1, I0), (T1, I1)
    
    text_score_count = 0
    image_score_count = 0
    group_score_count = 0
    total = len(dataset)
    
    try:
        for item in dataset:
            c0 = item["caption_0"]
            c1 = item["caption_1"]
            i0 = item["image_0"].convert("RGB")
            i1 = item["image_1"].convert("RGB")
            
            # Get Scores
            with torch.no_grad():
                if model_info["type"] == "colpali":
                    # Process Images
                    batch_imgs = processor.process_images([i0, i1]).to(DEVICE)
                    img_embs = model(**batch_imgs) # List of tensors
                    
                    # Process Texts
                    batch_txts = processor.process_queries([c0, c1]).to(DEVICE)
                    txt_embs = model(**batch_txts) # List of tensors
                    
                    # Score: Returns tensor of shape (2, 2) -> [[s00, s01], [s10, s11]]
                    scores = processor.score(txt_embs, img_embs)
                    
                else:
                    # Dense
                    inputs_img = processor(images=[i0, i1], return_tensors="pt", padding=True).to(DEVICE)
                    if hasattr(model, "get_image_features"):
                        img_embs = model.get_image_features(**inputs_img)
                    else:
                        img_embs = model(**inputs_img).image_embeds
                    img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
                    
                    inputs_txt = processor(text=[c0, c1], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                    if hasattr(model, "get_text_features"):
                        txt_embs = model.get_text_features(**inputs_txt)
                    else:
                        txt_embs = model(**inputs_txt).text_embeds
                    txt_embs = txt_embs / txt_embs.norm(dim=-1, keepdim=True)
                    
                    scores = torch.matmul(txt_embs, img_embs.t())

            # Logic
            # scores[0,0] = T0-I0, scores[0,1] = T0-I1
            # scores[1,0] = T1-I0, scores[1,1] = T1-I1
            
            s_c0_i0 = scores[0,0]
            s_c0_i1 = scores[0,1]
            s_c1_i0 = scores[1,0]
            s_c1_i1 = scores[1,1]
            
            # Text Score: Given an image, is the correct caption scored higher?
            # For I0: c0 > c1?  AND For I1: c1 > c0?
            is_text_correct = (s_c0_i0 > s_c1_i0) and (s_c1_i1 > s_c0_i1)
            
            # Image Score: Given a caption, is the correct image scored higher?
            # For C0: i0 > i1? AND For C1: i1 > i0?
            is_image_correct = (s_c0_i0 > s_c0_i1) and (s_c1_i1 > s_c1_i0)
            
            if is_text_correct: text_score_count += 1
            if is_image_correct: image_score_count += 1
            if is_text_correct and is_image_correct: group_score_count += 1
            
        return {
            "Wino Text": f"{text_score_count/total:.2%}",
            "Wino Image": f"{image_score_count/total:.2%}",
            "Wino Group": f"{group_score_count/total:.2%}"
        }

    except Exception as e:
        print(f"Error in Winoground: {e}")
        return {"Wino Group": "ERR"}

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(">>> LOADING DATASETS...")
    
    # 1. Flickr30k
    try:
        ds_flickr = load_dataset("lmms-lab/flickr30k", split="test").select(range(SAMPLE_SIZE))
    except:
        ds_flickr = load_dataset("lmms-lab/flickr30k", split="train").select(range(SAMPLE_SIZE))
        
    # 2. COCO
    # Switching to 'merve/coco2017' which is Parquet-based and safe for datasets>=3.0
    try:
        print("Downloading COCO (merve/coco2017)...")
        ds_coco = load_dataset("merve/coco2017", split="validation").select(range(SAMPLE_SIZE))
        # This dataset has 'image' and 'captions' (list of strings) columns
        # We need to map 'captions' -> 'caption' for uniformity or handle it in the loop
    except Exception as e:
        print(f"COCO load failed: {e}. Skipping COCO.")
        ds_coco = None # Skip if fails
    
    # 3. DocVQA
    ds_docvqa = load_dataset("HuggingFaceM4/DocVQA", split="test").select(range(SAMPLE_SIZE))
    
    # 4. Winoground
    ds_wino = load_dataset("facebook/winoground", split="test", trust_remote_code=True)
    
    print(">>> DATASETS READY.")
    
    results = []
    
    for m_info in MODELS:
        clean_memory()
        model, processor = load_model(m_info)
        
        if model is None:
            continue
            
        res = {"Model": m_info["name"]}
        
        # Flickr
        res.update(run_retrieval_benchmark(model, processor, m_info, ds_flickr, "Flickr"))
        
        # COCO
        if ds_coco is not None:
            # Determine text column: 'captions' (merve), 'caption' (standard), or 'sentences_raw' (legacy)
            if "captions" in ds_coco.features: coco_text_col = "captions"
            elif "caption" in ds_coco.features: coco_text_col = "caption"
            else: coco_text_col = "sentences_raw"
            
            res.update(run_retrieval_benchmark(model, processor, m_info, ds_coco, "COCO", text_col=coco_text_col))
        else:
            res.update({"COCO R@1": "N/A", "COCO R@5": "N/A"})
        
        # DocVQA (Image <-> Question)
        res.update(run_retrieval_benchmark(model, processor, m_info, ds_docvqa, "DocVQA", text_col="question"))
        
        # Winoground
        res.update(run_winoground_benchmark(model, processor, m_info, ds_wino))
        
        results.append(res)
        print(f"DONE: {m_info['name']}")
        print(res)
    
    # --- SAVE RESULTS ---
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("GRAND SLAM BENCHMARK RESULTS")
    print("="*50)
    print(df.to_markdown(index=False))
    
    df.to_csv("benchmark_grand_slam_results.csv", index=False)
    print("\nSaved to benchmark_grand_slam_results.csv")
