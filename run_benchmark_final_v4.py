import torch
import gc
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor
# ColPali için özel import
try:
    from colpali_engine.models import ColPali, ColPaliProcessor
    COLPALI_AVAILABLE = True
except ImportError:
    print("UYARI: colpali_engine bulunamadı. ColPali test edilemeyecek.")
    COLPALI_AVAILABLE = False

import sys

# --- AYARLAR ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 

# --- MODEL LİSTESİ (v4) ---
MODELS_TO_TEST = [
    # 1. Baseline
    {"name": "OpenAI-CLIP-L", "id": "openai/clip-vit-large-patch14", "type": "dense"},
    
    # 2. SigLIP Ailesi (Kıyaslama: Base vs Large)
    {"name": "SigLIP-Base",   "id": "google/siglip-base-patch16-256",   "type": "dense"},
    {"name": "SigLIP-400M",   "id": "google/siglip-so400m-patch14-384", "type": "dense"},
    
    # 3. Retrieval Expert
    {"name": "Jina-CLIP-v1",  "id": "jinaai/jina-clip-v1", "trust": True, "type": "dense"},
    
    # 4. Heavyweight
    {"name": "LAION-CLIP-H",  "id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "type": "dense"},
    
    # 5. ColPali (ColPali Engine ile)
    {"name": "ColPali-v1.3",  "id": "vidore/colpali-v1.3", "trust": True, "type": "colpali"}
]

def clean_memory():
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    torch.mps.empty_cache()
    gc.collect()

def run_test(model_info, dataset):
    model_name = model_info["name"]
    model_id = model_info["id"]
    model_type = model_info["type"]
    trust_code = model_info.get("trust", False)
    
    print(f"\n>>> Yükleniyor: {model_name}...")
    
    try:
        # --- MODEL YÜKLEME ---
        if model_type == "colpali":
            if not COLPALI_AVAILABLE:
                return {"Model": model_name, "Error": "Library not installed"}
            
            # ColPali Yükle
            model = ColPali.from_pretrained(
                model_id, 
                torch_dtype=DTYPE, 
                device_map=DEVICE
            ).eval()
            processor = ColPaliProcessor.from_pretrained(model_id)
            
        else:
            # Standart Yükle
            model = AutoModel.from_pretrained(
                model_id, 
                trust_remote_code=trust_code, 
                torch_dtype=DTYPE
            ).to(DEVICE)
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_code)
            model.eval()
        
        text_score_count = 0
        image_score_count = 0
        group_score_count = 0
        total_samples = len(dataset)
        
        print(f"Test Başlıyor ({total_samples} örnek)...")

        for i, example in enumerate(dataset):
            images = [example["image_0"].convert("RGB"), example["image_1"].convert("RGB")]
            texts = [example["caption_0"], example["caption_1"]]
            
            # --- HESAPLAMA ---
            if model_type == "colpali":
                with torch.no_grad():
                    # 1. Görüntüleri İşle ve Embedding Al
                    batch_images = processor.process_images(images).to(DEVICE)
                    # ColPali model(**inputs) direkt embeddingleri döner
                    image_embeddings = model(**batch_images)
                    
                    # 2. Metinleri İşle ve Embedding Al
                    batch_queries = processor.process_queries(texts).to(DEVICE)
                    query_embeddings = model(**batch_queries)
                    
                    # 3. Skorla (Model argümanı olmadan, direkt embeddingler ile)
                    # ColPaliProcessor.score -> score_multi_vector(qs, ims)
                    scores = processor.score(query_embeddings, image_embeddings)
                    logits = scores

            else:
                # Standart Akış
                inputs = processor(
                    text=texts, images=images, return_tensors="pt", 
                    padding=True, truncation=True
                ).to(DEVICE)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                    if hasattr(outputs, 'image_embeds'):
                        img_embeds = outputs.image_embeds
                        text_embeds = outputs.text_embeds
                    else:
                        img_embeds = outputs[0]
                        text_embeds = outputs[1]

                    img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
                    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                    
                    logits = torch.matmul(img_embeds, text_embeds.t())

            # --- SKORLAMA ---
            logits = logits.float().cpu()
            
            if model_type == "colpali":
                 # ColPali: Satır=Text, Sütun=Image
                 # score_multi_vector sonucu [Queries, Images] şeklindedir.
                 c0_i0 = logits[0, 0] 
                 c0_i1 = logits[0, 1] 
                 c1_i0 = logits[1, 0] 
                 c1_i1 = logits[1, 1] 
                 
                 text_success = (c0_i0 > c1_i0) and (c1_i1 > c0_i1)
                 image_success = (c0_i0 > c0_i1) and (c1_i1 > c1_i0)
            else:
                 # Standart: Satır=Image, Sütun=Text
                 c0_i0 = logits[0, 0]
                 c0_i1 = logits[0, 1]
                 c1_i0 = logits[1, 0]
                 c1_i1 = logits[1, 1]
                 
                 text_success = (c0_i0 > c0_i1) and (c1_i1 > c1_i0)
                 image_success = (c0_i0 > c1_i0) and (c1_i1 > c0_i1)

            group_success = text_success and image_success

            if text_success: text_score_count += 1
            if image_success: image_score_count += 1
            if group_success: group_score_count += 1
            
            if (i+1) % 50 == 0: print(f"  > {i+1} işlendi.")
        
        results = {
            "Model": model_name,
            "Image Score": f"{image_score_count/total_samples:.2%}",
            "Group Score": f"{group_score_count/total_samples:.2%}"
        }
        
        del model
        del processor
        clean_memory()
        return results

    except Exception as e:
        print(f"!!! HATA ({model_name}): {str(e)}")
        clean_memory()
        return {"Model": model_name, "Error": str(e)}

# --- MAIN ---
if __name__ == "__main__":
    print(f"Benchmark v4 Başlıyor... Cihaz: {DEVICE}")
    print("Yenilikler: ColPali Fix, SigLIP-Base eklendi.")
    
    try:
        dataset = load_dataset("facebook/winoground", split="test", trust_remote_code=True)
    except:
        print("Dataset yüklenemedi.")
        sys.exit(1)

    final_report = []

    for m in MODELS_TO_TEST:
        clean_memory()
        res = run_test(m, dataset)
        final_report.append(res)
        print(f"--> {res['Model']} Sonuç: {res.get('Image Score')}")

    print("\n" + "="*60)
    print("NİHAİ SONUÇLAR (V4)")
    print("="*60)
    print(f"{ 'Model':<20} | {'Image Retrieval':<18} | {'Reasoning (IQ)':<18}")
    print("-" * 60)
    
    with open("benchmark_results_v4.txt", "w") as f:
        f.write("Model,Image Score,Group Score\n")
        for r in final_report:
            if "Error" in r:
                line = f"{r['Model']:<20} | ERROR: {r['Error']}"
            else:
                line = f"{r['Model']:<20} | {r['Image Score']:<18} | {r['Group Score']:<18}"
                f.write(f"{r['Model']},{r['Image Score']},{r['Group Score']}\n")
            print(line)
