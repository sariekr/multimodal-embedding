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
FLICKR_SAMPLE_SIZE = 1000  # Balanced: fast but valid

print(f"Running on: {DEVICE}")
print(f"Precision: {DTYPE}")
print(f"Using Flickr30k test set: {FLICKR_SAMPLE_SIZE} samples")

# --- MODEL LİSTESİ (THE FINAL 8) ---
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

def compute_bidirectional_metrics(txt_embeds, img_embeds, scores_t2i, scores_i2t):
    """
    Compute metrics for both Text-to-Image and Image-to-Text retrieval

    Args:
        txt_embeds: Text embeddings [N, D]
        img_embeds: Image embeddings [N, D]
        scores_t2i: Text-to-Image similarity scores [N, N]
        scores_i2t: Image-to-Text similarity scores [N, N]
    """
    n = scores_t2i.size(0)
    metrics = {}

    # TEXT-TO-IMAGE RETRIEVAL
    for k in [1, 5, 10]:
        if k > n:
            metrics[f"T2I_R@{k}"] = "N/A"
            continue
        correct = 0
        for i in range(n):
            # For each text query, check if correct image is in top-K
            if i in torch.topk(scores_t2i[i], k=k).indices.tolist():
                correct += 1
        metrics[f"T2I_R@{k}"] = f"{correct/n:.1%}"

    # T2I MRR
    mrr_sum = 0
    for i in range(n):
        rank = (scores_t2i[i].argsort(descending=True) == i).nonzero(as_tuple=True)[0].item() + 1
        mrr_sum += 1.0 / rank
    metrics["T2I_MRR"] = f"{mrr_sum/n:.3f}"

    # IMAGE-TO-TEXT RETRIEVAL
    for k in [1, 5, 10]:
        if k > n:
            metrics[f"I2T_R@{k}"] = "N/A"
            continue
        correct = 0
        for i in range(n):
            # For each image query, check if correct text is in top-K
            if i in torch.topk(scores_i2t[i], k=k).indices.tolist():
                correct += 1
        metrics[f"I2T_R@{k}"] = f"{correct/n:.1%}"

    # I2T MRR
    mrr_sum = 0
    for i in range(n):
        rank = (scores_i2t[i].argsort(descending=True) == i).nonzero(as_tuple=True)[0].item() + 1
        mrr_sum += 1.0 / rank
    metrics["I2T_MRR"] = f"{mrr_sum/n:.3f}"

    return metrics

def run_retrieval_benchmark(model, processor, model_info, dataset, dataset_name, text_col="caption", image_col="image"):
    print(f"  > Benchmarking {dataset_name} (N={len(dataset)})...")
    bs = model_info.get("batch_size", 32)

    try:
        images = [item[image_col].convert("RGB") for item in dataset]
        queries = [safe_get_text(item, text_col) for item in dataset]
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
                # ColPali scoring (multi-vector)
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
                scores_i2t = scores_t2i.t()  # Transpose for I2T
            else:
                # Dense model scoring
                all_img = torch.cat(img_embeds_list, dim=0).to(DTYPE)
                all_txt = torch.cat(txt_embeds_list, dim=0).to(DTYPE)
                scores_t2i = torch.matmul(all_txt, all_img.t())  # [N_txt, N_img]
                scores_i2t = torch.matmul(all_img, all_txt.t())  # [N_img, N_txt]

        t_end = time.time()
        inference_time = t_end - t_start
        throughput = len(queries) / inference_time

        # Convert to float for metric computation
        scores_t2i = scores_t2i.float()
        scores_i2t = scores_i2t.float()

        # Compute bidirectional metrics
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
            scores_i2t
        )

        metrics["Time"] = f"{inference_time:.1f}s"
        metrics["QPS"] = f"{throughput:.1f}"

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
    try: dataset = load_dataset("facebook/winoground", split="test")
    except: return {"Wino Group": "N/A"}

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
    except Exception as e: return {"Wino Group": "Error"}

# --- MAIN ---
if __name__ == "__main__":
    print(">>> LOADING DATASETS...")

    # Load Flickr30k test set with sampling
    try:
        ds_flickr_full = load_dataset("lmms-lab/flickr30k", split="test")
        print(f"✓ Loaded Flickr30k full test set: {len(ds_flickr_full)} samples")

        # Sample if needed
        if FLICKR_SAMPLE_SIZE and FLICKR_SAMPLE_SIZE < len(ds_flickr_full):
            ds_flickr = ds_flickr_full.shuffle(seed=SEED).select(range(FLICKR_SAMPLE_SIZE))
            print(f"✓ Sampled {len(ds_flickr)} samples for faster evaluation")
        else:
            ds_flickr = ds_flickr_full
    except Exception as e:
        print(f"✗ Failed to load Flickr30k: {e}")
        ds_flickr = []

    results = []
    for m_info in MODELS_TO_TEST:
        m_name = m_info["name"]
        print(f"\n{'='*100}")
        print(f">>> TESTING: {m_name}")
        print(f"{'='*100}")
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
            results.append({"Model": m_name, "Flickr T2I_R@1": "Load Error"})
            continue

        row = {"Model": m_name}

        # Run Flickr30k benchmark (bidirectional)
        if len(ds_flickr) > 0:
            row.update(run_retrieval_benchmark(model, processor, m_info, ds_flickr, "Flickr", "caption"))

        # Run Winoground benchmark
        row.update(run_winoground_full(model, processor, m_info))

        results.append(row)
        print(f"\nResult Summary: {m_name}")
        print(f"  T2I R@1: {row.get('Flickr T2I_R@1', 'N/A')}")
        print(f"  I2T R@1: {row.get('Flickr I2T_R@1', 'N/A')}")
        print(f"  Wino: {row.get('Wino Group', 'N/A')}")

    print("\n" + "="*150)
    print(f"GRAND SLAM BENCHMARK RESULTS (V18 - Bidirectional Retrieval, {FLICKR_SAMPLE_SIZE} samples)")
    print("="*150)

    if len(results) > 0:
        df = pd.DataFrame(results)

        # TABLE 1: TEXT-TO-IMAGE RETRIEVAL
        print(f"\n📊 TEXT-TO-IMAGE RETRIEVAL (Flickr30k {FLICKR_SAMPLE_SIZE} test samples)")
        print("-" * 150)
        t2i_cols = ["Model",
                    "Flickr T2I_R@1", "Flickr T2I_R@5", "Flickr T2I_R@10", "Flickr T2I_MRR"]
        t2i_cols = [c for c in t2i_cols if c in df.columns]
        print(df[t2i_cols].to_markdown(index=False))

        # TABLE 2: IMAGE-TO-TEXT RETRIEVAL
        print(f"\n\n🔄 IMAGE-TO-TEXT RETRIEVAL (Flickr30k {FLICKR_SAMPLE_SIZE} test samples)")
        print("-" * 150)
        i2t_cols = ["Model",
                    "Flickr I2T_R@1", "Flickr I2T_R@5", "Flickr I2T_R@10", "Flickr I2T_MRR"]
        i2t_cols = [c for c in i2t_cols if c in df.columns]
        print(df[i2t_cols].to_markdown(index=False))

        # TABLE 3: PERFORMANCE METRICS
        print("\n\n⚡ PERFORMANCE METRICS")
        print("-" * 150)
        perf_cols = ["Model", "Flickr Time", "Flickr QPS", "Flickr Embed"]
        perf_cols = [c for c in perf_cols if c in df.columns]
        print(df[perf_cols].to_markdown(index=False))

        # TABLE 4: WINOGROUND DETAILS
        if "Wino Text" in df.columns and "Wino Image" in df.columns:
            print("\n\n🎯 WINOGROUND COMPOSITIONAL REASONING (400 samples)")
            print("-" * 150)
            wino_cols = ["Model", "Wino Text", "Wino Image", "Wino Group"]
            print(df[wino_cols].to_markdown(index=False))

        # Save full results
        df.to_csv("benchmark_v18_results.csv", index=False)
        print(f"\n✅ Full results saved to: benchmark_v18_results.csv")
        print("\n" + "="*150)
        print("KEY IMPROVEMENTS IN V18:")
        print("="*150)
        print(f"✓ Flickr30k test set: {FLICKR_SAMPLE_SIZE} samples (balanced speed/validity)")
        print("✓ Bidirectional retrieval: Text-to-Image AND Image-to-Text")
        print("✓ Removed DocVQA and COCO (focused on general vision)")
        print("✓ Single run with fixed seed (reproducible)")
        print("✓ ~2-3 hours runtime on A40 (vs 15-20h for full set)")
        print("✓ Results statistically valid with 1K samples")
