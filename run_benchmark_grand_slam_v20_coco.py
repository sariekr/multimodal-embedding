"""
V20 - MS-COCO (Karpathy Split) & Metric Fixes

CHANGES from V19:
1. 🔄 DATASET SWITCH: Flickr30k -> MS-COCO (Karpathy Split)
   - Uses `yerevann/coco-karpathy` (Parquet format, no script errors)
   - Standard 5k Test Split (sampled to 1k for speed, configurable)
2. ✅ VALIDATION: Added sanity check for 5 captions per image
3. 📊 METRICS: Added "Img/s" (Images/sec) to distinguish from "QPS" (Queries/sec)
   - Addressed criticism that QPS is inflated when using 5 captions/image
4. 🎯 SCOPE: MS-COCO Test (1k) + Winoground

"""

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

SAMPLE_SIZE = 1000  # Number of images to sample from COCO Test (Total 5000)

print(f"Running on: {DEVICE}")
print(f"Precision: {DTYPE}")
print(f"Target Sample Size: {SAMPLE_SIZE}")

# --- MODEL LİSTESİ ---
MODELS_TO_TEST = [
    # 1. ColPali (Document Specialist)
    {"name": "ColPali-v1.3",  "id": "vidore/colpali-v1.3", "type": "colpali", "batch_size": 4},

    # 2. SigLIP 400M (Standard)
    {"name": "SigLIP-400M",   "id": "google/siglip-so400m-patch14-384", "type": "siglip", "batch_size": 64},

    # 3. SigLIP Base (Speed variant)
    {"name": "SigLIP-Base",   "id": "google/siglip-base-patch16-224", "type": "siglip", "batch_size": 64},

    # 4. LAION Huge (The Classic)
    {"name": "LAION-CLIP-H",  "id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "type": "dense", "batch_size": 32},

    # 5. OpenAI CLIP
    {"name": "OpenAI-CLIP-L", "id": "openai/clip-vit-large-patch14-336", "type": "dense", "batch_size": 32},

    # 6. Jina CLIP
    {"name": "Jina-CLIP-v1", "id": "jinaai/jina-clip-v1", "type": "dense", "trust": True, "batch_size": 32},

    # 7. Apple DFN5B
    {"name": "Apple-DFN5B-H", "id": "apple/DFN5B-CLIP-ViT-H-14-378", "type": "dense", "trust": True, "batch_size": 32},

    # 8. MetaCLIP Huge
    {"name": "MetaCLIP-H14",  "id": "facebook/metaclip-h14-fullcc2.5b", "type": "dense", "trust": True, "batch_size": 16}
]

def clean_memory():
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    torch.cuda.empty_cache()
    gc.collect()

def safe_get_text(item, col_name):
    """Extract text from item, handling various data structures"""
    val = item.get(col_name, "")
    if isinstance(val, list):
        if len(val) == 0: return ""
        return str(val[0])  # Always use first caption for consistency
    if isinstance(val, dict):
        for key in ['en', 'text', 'caption']:
            if key in val: return str(val[key])
        return str(val)
    return str(val)

def compute_bidirectional_metrics(txt_embeds, img_embeds, scores_t2i, scores_i2t,
                                 query_to_image_map=None, image_to_query_map=None,
                                 image_to_all_captions=None):
    """
    Compute metrics for both Text-to-Image and Image-to-Text retrieval
    """
    n_text = scores_t2i.size(0)
    n_img = scores_i2t.size(0)
    metrics = {}

    # Default to 1:1 mapping if not provided
    if query_to_image_map is None:
        query_to_image_map = list(range(n_text))

    # Handle I2T ground truth
    if image_to_all_captions is None:
        # Fallback to single caption per image
        if image_to_query_map is None:
            image_to_query_map = list(range(n_img))
        image_to_all_captions = {i: [image_to_query_map[i]] for i in range(n_img)}

    # TEXT-TO-IMAGE RETRIEVAL
    for k in [1, 5, 10]:
        if k > scores_t2i.size(1):  # Check against gallery size
            metrics[f"T2I_R@{k}"] = "N/A"
            continue
        correct = 0
        for i in range(n_text):
            # For each text query, check if correct image is in top-K
            correct_img_idx = query_to_image_map[i]
            top_k_images = torch.topk(scores_t2i[i], k=k).indices.tolist()
            if correct_img_idx in top_k_images:
                correct += 1
        metrics[f"T2I_R@{k}"] = f"{correct/n_text:.1%}"

    # T2I MRR
    mrr_sum = 0
    for i in range(n_text):
        correct_img_idx = query_to_image_map[i]
        # Find rank of correct image
        sorted_indices = scores_t2i[i].argsort(descending=True)
        rank = (sorted_indices == correct_img_idx).nonzero(as_tuple=True)[0].item() + 1
        mrr_sum += 1.0 / rank
    metrics["T2I_MRR"] = f"{mrr_sum/n_text:.3f}"

    # IMAGE-TO-TEXT RETRIEVAL (ANY of multiple valid captions)
    for k in [1, 5, 10]:
        if k > scores_i2t.size(1):  # Check against gallery size
            metrics[f"I2T_R@{k}"] = "N/A"
            continue
        correct = 0
        for i in range(n_img):
            # For each image query, check if ANY valid caption is in top-K
            valid_caption_indices = image_to_all_captions[i]
            top_k_texts = torch.topk(scores_i2t[i], k=k).indices.tolist()
            if any(cap_idx in top_k_texts for cap_idx in valid_caption_indices):
                correct += 1
        metrics[f"I2T_R@{k}"] = f"{correct/n_img:.1%}"

    # I2T MRR - rank of BEST matching caption
    mrr_sum = 0
    for i in range(n_img):
        valid_caption_indices = set(image_to_all_captions[i])
        sorted_indices = scores_i2t[i].argsort(descending=True).tolist()
        # Find rank of first valid caption
        for rank, idx in enumerate(sorted_indices, 1):
            if idx in valid_caption_indices:
                mrr_sum += 1.0 / rank
                break
    metrics["I2T_MRR"] = f"{mrr_sum/n_img:.3f}"

    return metrics

def run_retrieval_benchmark(model, processor, model_info, dataset, dataset_name, text_col="caption", image_col="image", use_all_captions=False):
    """
    Run bidirectional retrieval benchmark.
    """
    print(f"  > Benchmarking {dataset_name} (N={len(dataset)})...")
    bs = model_info.get("batch_size", 32)

    try:
        if use_all_captions:
            # Standard: N unique images, M captions (M = N * captions_per_image)
            images = []
            queries = []
            query_to_image_map = []  # Maps query idx → image idx in gallery
            image_to_all_captions = {}  # Maps image idx → ALL its caption indices (for I2T)

            for idx, item in enumerate(dataset):
                img = item[image_col].convert("RGB")
                images.append(img)  # Add image ONCE to gallery

                captions = item.get(text_col, [])
                caption_indices = []

                if isinstance(captions, list) and len(captions) > 0:
                    for cap in captions:
                        caption_indices.append(len(queries))
                        queries.append(str(cap))
                        query_to_image_map.append(idx)  # This caption → image idx
                else:
                    # Fallback to single caption
                    caption_indices.append(len(queries))
                    queries.append(safe_get_text(item, text_col))
                    query_to_image_map.append(idx)

                image_to_all_captions[idx] = caption_indices  # Image → ALL captions

            # For backward compatibility
            image_to_query_map = [caps[0] for caps in image_to_all_captions.values()]

        else:
            # Simple 1:1 image-caption mapping
            images = [item[image_col].convert("RGB") for item in dataset]
            queries = [safe_get_text(item, text_col) for item in dataset]
            query_to_image_map = list(range(len(images)))
            image_to_query_map = list(range(len(images)))
            image_to_all_captions = {i: [i] for i in range(len(images))}

    except Exception as e:
        print(f"    Data Prep Error: {e}")
        return {f"{dataset_name} Status": "Error"}

    t_start = time.time()
    try:
        with torch.no_grad():
            # ENCODE IMAGES
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

            # ENCODE TEXT
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

            # COMPUTE SIMILARITY SCORES
            if model_info["type"] == "colpali":
                # ColPali scoring
                score_bs = 10
                all_scores_t2i = []
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
                    all_scores_t2i.append(torch.cat(batch_scores, dim=1))
                    del q_batch
                    torch.cuda.empty_cache()
                scores_t2i = torch.cat(all_scores_t2i, dim=0)

                # ColPali I2T
                all_scores_i2t = []
                for i in range(0, len(img_embeds_list), score_bs):
                    img_q_batch = [img.to(DEVICE) for img in img_embeds_list[i : i+score_bs]]
                    batch_scores = []
                    txt_chunk = 20
                    for j in range(0, len(txt_embeds_list), txt_chunk):
                        t_batch = [t.to(DEVICE) for t in txt_embeds_list[j : j+txt_chunk]]
                        s = processor.score(img_q_batch, t_batch)
                        batch_scores.append(s.cpu())
                        del t_batch, s
                        torch.cuda.empty_cache()
                    all_scores_i2t.append(torch.cat(batch_scores, dim=1))
                    del img_q_batch
                    torch.cuda.empty_cache()
                scores_i2t = torch.cat(all_scores_i2t, dim=0)
            else:
                # Dense model scoring
                all_img = torch.cat(img_embeds_list, dim=0).to(DTYPE)
                all_txt = torch.cat(txt_embeds_list, dim=0).to(DTYPE)
                scores_t2i = torch.matmul(all_txt, all_img.t())
                scores_i2t = torch.matmul(all_img, all_txt.t())

        t_end = time.time()
        inference_time = t_end - t_start
        
        # METRICS CALCULATION
        # NOTE: Throughput (QPS) is traditionally len(queries) / time
        # But since we use 5 queries per image, this can be inflated.
        # We now report both Queries/s (QPS) and Images/s (Img/s)
        throughput = len(queries) / inference_time
        img_throughput = len(images) / inference_time

        scores_t2i = scores_t2i.float()
        scores_i2t = scores_i2t.float()

        if model_info["type"] == "colpali":
            txt_embeds_placeholder = None
            img_embeds_placeholder = None
        else:
            txt_embeds_placeholder = all_txt
            img_embeds_placeholder = all_img

        metrics = compute_bidirectional_metrics(
            txt_embeds_placeholder,
            img_embeds_placeholder,
            scores_t2i,
            scores_i2t,
            query_to_image_map=query_to_image_map,
            image_to_query_map=image_to_query_map,
            image_to_all_captions=image_to_all_captions
        )

        metrics["Time"] = f"{inference_time:.1f}s"
        metrics["QPS"] = f"{throughput:.1f}"
        metrics["Img/s"] = f"{img_throughput:.1f}"

        if model_info["type"] == "colpali":
            avg_tokens = sum(e.shape[0] for e in img_embeds_list) / len(img_embeds_list)
            metrics["Embed"] = f"~{int(avg_tokens)}tok"
        else:
            dim = all_img.shape[1]
            metrics["Embed"] = f"{dim}d"

        final_metrics = {}
        for k, v in metrics.items():
            final_metrics[f"{dataset_name} {k}"] = v
        return final_metrics

    except Exception as e:
        print(f"Inference Error: {e}")
        import traceback
        traceback.print_exc()
        return {f"{dataset_name} T2I_R@1": "Error", f"{dataset_name} I2T_R@1": "Error"}

def run_winoground_full(model, processor, model_info):
    print("  > Benchmarking Winoground (Full 400 samples)...")
    try:
        dataset = load_dataset("facebook/winoground", split="test")
    except Exception as e:
        print(f"    Failed to load Winoground: {e}")
        return {"Wino Group": "N/A"}

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
        return {"Wino Text": f"{text_score/total:.1%}", "Wino Image": f"{image_score/total:.1%}", "Wino Group": f"{group_score/total:.1%}"}
    except Exception as e:
        print(f"    Winoground inference error: {e}")
        import traceback
        traceback.print_exc()
        return {"Wino Group": "Error"}

# --- MAIN ---
if __name__ == "__main__":
    
    # --- MS-COCO (KARPATHY SPLIT) YÜKLEME ---
    print(">>> LOADING MS-COCO (Karpathy Split)...")
    try:
        # Bu repo Parquet formatındadır, script hatası vermez.
        # "test" split'i tam olarak 5000 resimden oluşur (Standart Karpathy).
        ds_coco_full = load_dataset("yerevann/coco-karpathy", split="test")
        print(f"✓ Loaded yerevann/coco-karpathy: {len(ds_coco_full)} samples")

        # 🔍 SANITY CHECK: Her resmin tam 5 caption'ı olmalı
        print("  > Verifying caption counts (Expect 5 per image)...")
        valid_count = 0
        for item in ds_coco_full:
            if len(item['sentences']) == 5:
                valid_count += 1
        
        if valid_count == len(ds_coco_full):
            print(f"  ✓ INTEGRITY PASSED: All {len(ds_coco_full)} images have exactly 5 captions.")
        else:
            print(f"  ⚠️ CRITICAL WARNING: Only {valid_count}/{len(ds_coco_full)} images have 5 captions!")
            # Devam ediyoruz ama uyarı veriyoruz

        # Hız için örnekleyelim (Flickr ile karşılaştırılabilir olması için)
        if SAMPLE_SIZE and SAMPLE_SIZE < len(ds_coco_full):
            # Sabit seed ile her zaman aynı resimleri seçelim
            ds_benchmark = ds_coco_full.shuffle(seed=SEED).select(range(SAMPLE_SIZE))
            print(f"✓ Sampled {len(ds_benchmark)} images (Subset of COCO Test)")
        else:
            ds_benchmark = ds_coco_full
            print(f"✓ Using full {len(ds_benchmark)} COCO test samples")
            
    except Exception as e:
        print(f"✗ Failed to load MS-COCO: {e}")
        ds_benchmark = []

    results = []
    for m_info in MODELS_TO_TEST:
        m_name = m_info["name"]
        print(f"\n{'='*100}")
        print(f">>> TESTING: {m_name}")
        print(f"{ '='*100}")
        clean_memory()

        try:
            trust = m_info.get("trust", False)
            if m_info["type"] == "colpali":
                if not COLPALI_AVAILABLE:
                    print("  Skipping ColPali (not installed)")
                    continue
                model = ColPali.from_pretrained(m_info["id"], torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=trust).eval()
                processor = ColPaliProcessor.from_pretrained(m_info["id"], trust_remote_code=trust)
            elif m_info["type"] == "siglip":
                model = SiglipModel.from_pretrained(m_info["id"], torch_dtype=DTYPE).to(DEVICE).eval()
                processor = SiglipProcessor.from_pretrained(m_info["id"])
            else:
                try:
                    model = AutoModel.from_pretrained(m_info["id"], trust_remote_code=trust, torch_dtype=DTYPE, use_safetensors=True).to(DEVICE)
                except:
                    print("    Safetensors failed, attempting fallback...")
                    model = AutoModel.from_pretrained(m_info["id"], trust_remote_code=trust, torch_dtype=DTYPE).to(DEVICE)
                processor = AutoProcessor.from_pretrained(m_info["id"], trust_remote_code=trust)
                model.eval()
        except Exception as e:
            print(f"Load Error for {m_name}: {e}")
            results.append({"Model": m_name, "COCO T2I_R@1": "Load Error"})
            continue

        row = {"Model": m_name}

        # Run COCO Benchmark
        if len(ds_benchmark) > 0:
            row.update(run_retrieval_benchmark(
                model, 
                processor, 
                m_info, 
                ds_benchmark, 
                "COCO",           # Dataset Adı
                "sentences",      # Text Sütunu (yerevann'da 'sentences' listesi)
                "image",          # Image Sütunu
                use_all_captions=True
            ))

        # Run Winoground benchmark
        row.update(run_winoground_full(model, processor, m_info))

        results.append(row)
        print(f"\nResult Summary: {m_name}")
        print(f"  T2I R@1: {row.get('COCO T2I_R@1', 'N/A')}")
        print(f"  I2T R@1: {row.get('COCO I2T_R@1', 'N/A')}")
        print(f"  Wino: {row.get('Wino Group', 'N/A')}")

    print("\n" + "="*150)
    print(f"GRAND SLAM V20 RESULTS - MS-COCO (Karpathy) & WINOGROUND")
    print("="*150)

    if len(results) > 0:
        df = pd.DataFrame(results)

        # TABLE 1: TEXT-TO-IMAGE RETRIEVAL
        print(f"\n📊 TEXT-TO-IMAGE RETRIEVAL (COCO {SAMPLE_SIZE} test samples)")
        print("-" * 150)
        t2i_cols = ["Model",
                    "COCO T2I_R@1", "COCO T2I_R@5", "COCO T2I_R@10", "COCO T2I_MRR"]
        t2i_cols = [c for c in t2i_cols if c in df.columns]
        print(df[t2i_cols].to_markdown(index=False))

        # TABLE 2: IMAGE-TO-TEXT RETRIEVAL
        print(f"\n\n🔄 IMAGE-TO-TEXT RETRIEVAL (COCO {SAMPLE_SIZE} test samples)")
        print("-" * 150)
        i2t_cols = ["Model",
                    "COCO I2T_R@1", "COCO I2T_R@5", "COCO I2T_R@10", "COCO I2T_MRR"]
        i2t_cols = [c for c in i2t_cols if c in df.columns]
        print(df[i2t_cols].to_markdown(index=False))

        # TABLE 3: PERFORMANCE METRICS
        print("\n\n⚡ PERFORMANCE METRICS")
        print("-" * 150)
        # Added Img/s
        perf_cols = ["Model", "COCO Time", "COCO QPS", "COCO Img/s", "COCO Embed"]
        perf_cols = [c for c in perf_cols if c in df.columns]
        print(df[perf_cols].to_markdown(index=False))

        # TABLE 4: WINOGROUND DETAILS
        if "Wino Text" in df.columns and "Wino Image" in df.columns:
            print("\n\n🎯 WINOGROUND COMPOSITIONAL REASONING (400 samples)")
            print("-" * 150)
            wino_cols = ["Model", "Wino Text", "Wino Image", "Wino Group"]
            print(df[wino_cols].to_markdown(index=False))

        # Save full results
        df.to_csv("benchmark_v20_coco_results.csv", index=False)
        print(f"\n✅ Full results saved to: benchmark_v20_coco_results.csv")
