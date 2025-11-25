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
# Bazı modeller fp16'da daha kararlı çalışır, bazıları için gerekmez ama bellek için iyi.
DTYPE = torch.float16 

# --- MODEL LİSTESİ (v3) ---
MODELS_TO_TEST = [
    # 1. Baseline
    {"name": "OpenAI-CLIP-L", "id": "openai/clip-vit-large-patch14", "type": "dense"},
    
    # 2. Google (Protobuf sorunu çözüldü)
    {"name": "SigLIP-400M",   "id": "google/siglip-so400m-patch14-384", "type": "dense"},
    
    # 3. Retrieval Expert
    {"name": "Jina-CLIP-v1",  "id": "jinaai/jina-clip-v1", "trust": True, "type": "dense"},
    
    # NOMIC ÇIKARILDI: Sadece Vision Encoder olduğu için Text-Image retrieval yapamıyor.
    # {"name": "Nomic-Vision", ...} -> KAPALI
    
    # 4. Heavyweight
    {"name": "LAION-CLIP-H",  "id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "type": "dense"},
    
    # 5. ColPali (Native Library ile)
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
            
            # ColPali Özel Yükleme
            model = ColPali.from_pretrained(
                model_id, 
                torch_dtype=DTYPE, 
                device_map=DEVICE
            ).eval()
            processor = ColPaliProcessor.from_pretrained(model_id)
            
        else:
            # Standart Yükleme
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
                    # ColPali: Image -> Pikseller, Text -> ID'ler
                    batch_images = processor.process_images(images).to(DEVICE)
                    batch_queries = processor.process_queries(texts).to(DEVICE)
                    
                    # Score metodu [Queries, Images] döner
                    scores = processor.score(batch_queries, batch_images, model=model)
                    logits = scores

            else:
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

                    # Normalizasyon
                    img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
                    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                    
                    # Matris Çarpımı: [Image, Dim] x [Text, Dim]^T -> [Image, Text]
                    logits = torch.matmul(img_embeds, text_embeds.t())

            # --- SKORLAMA ---
            logits = logits.float().cpu()
            
            if model_type == "colpali":
                 # ColPali: Satır=Text, Sütun=Image
                 c0_i0 = logits[0, 0] # Text0 - Image0
                 c0_i1 = logits[0, 1] # Text0 - Image1
                 c1_i0 = logits[1, 0] # Text1 - Image0
                 c1_i1 = logits[1, 1] # Text1 - Image1
                 
                 text_success = (c0_i0 > c1_i0) and (c1_i1 > c0_i1)
                 image_success = (c0_i0 > c0_i1) and (c1_i1 > c1_i0)
            else:
                 # Standart: Satır=Image, Sütun=Text
                 c0_i0 = logits[0, 0] # Image0 - Text0
                 c0_i1 = logits[0, 1] # Image0 - Text1
                 c1_i0 = logits[1, 0] # Image1 - Text0
                 c1_i1 = logits[1, 1] # Image1 - Text1
                 
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
    print(f"Benchmark v3 Başlıyor... Cihaz: {DEVICE}")
    print("Nomic çıkarıldı, ColPali düzeltildi, SigLIP aktifleştirildi.")
    
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
    print("NİHAİ SONUÇLAR (V3)")
    print("="*60)
    print(f"{ 'Model':<20} | {'Image Retrieval':<18} | {'Reasoning (IQ)':<18}")
    print("-" * 60)
    
    with open("benchmark_results_v3.txt", "w") as f:
        f.write("Model,Image Score,Group Score\n")
        for r in final_report:
            if "Error" in r:
                line = f"{r['Model']:<20} | ERROR: {r['Error']}"
            else:
                line = f"{r['Model']:<20} | {r['Image Score']:<18} | {r['Group Score']:<18}"
                f.write(f"{r['Model']},{r['Image Score']},{r['Group Score']}\n")
            print(line)
