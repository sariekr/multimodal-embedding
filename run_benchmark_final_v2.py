import torch
import gc
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor
import sys

# --- AYARLAR ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 

# --- GÜNCELLENMİŞ MODEL LİSTESİ ---
MODELS_TO_TEST = [
    # 1. Baseline
    {"name": "OpenAI-CLIP-L", "id": "openai/clip-vit-large-patch14", "type": "dense"},
    
    # 2. Google (SentencePiece hatası çözüldü)
    {"name": "SigLIP-400M",   "id": "google/siglip-so400m-patch14-384", "type": "dense"},
    
    # 3. Retrieval Expert
    {"name": "Jina-CLIP-v1",  "id": "jinaai/jina-clip-v1", "trust": True, "type": "dense"},
    
    # 4. Nomic (Boyut hatası için kod düzeltildi)
    {"name": "Nomic-Vision",  "id": "nomic-ai/nomic-embed-vision-v1.5", "trust": True, "type": "dense"},
    
    # 5. Heavyweight (EVA yerine LAION Huge kullanıyoruz - daha stabil)
    {"name": "LAION-CLIP-H",  "id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "type": "dense"},
    
    # 6. ColPali (Transformers güncellemesi gerektirir)
    {"name": "ColPali-v1.3",  "id": "vidore/colpali-v1.3", "trust": True, "type": "colpali"}
]

def clean_memory():
    """RAM Temizliği"""
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
        # Modeli Yükle
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
            
            # --- COLPALI AKIŞI ---
            if model_type == "colpali":
                with torch.no_grad():
                    # ColPali işlemcisi görüntüleri piksellere, metinleri ID'lere çevirir
                    batch_images = processor.process_images(images).to(DEVICE)
                    batch_queries = processor.process_queries(texts).to(DEVICE)
                    
                    # ColPali .score() metodu
                    scores = processor.score(batch_queries, batch_images, model=model)
                    logits = scores

            # --- STANDART (DENSE) AKIŞI ---
            else:
                inputs = processor(
                    text=texts, images=images, return_tensors="pt", 
                    padding=True, truncation=True
                ).to(DEVICE)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                    # Çıktıları al
                    if hasattr(outputs, 'image_embeds'):
                        img_embeds = outputs.image_embeds
                        text_embeds = outputs.text_embeds
                    else:
                        img_embeds = outputs[0]
                        text_embeds = outputs[1]

                    # --- HATA DÜZELTME: Boyut Kontrolü (Nomic Fix) ---
                    # Eğer çıktı [Batch, Seq, Dim] yani 3D gelirse, [Batch, Dim] yapmalıyız.
                    # Nomic için CLS token (ilk token) genelde embedding'dir.
                    if img_embeds.dim() == 3:
                        img_embeds = img_embeds[:, 0, :]
                    if text_embeds.dim() == 3:
                        text_embeds = text_embeds[:, 0, :]

                    # Normalizasyon
                    img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
                    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                    
                    # Matris Çarpımı
                    # [2, Dim] x [Dim, 2] -> [2, 2]
                    logits = torch.matmul(img_embeds, text_embeds.t())

            # --- SKORLAMA ---
            # logits matrisinin boyutu [2, 2] olmalı.
            # Bazen ColPali [Text, Image] döner, Standart [Image, Text] döner.
            # Winoground için standart:
            # Satırlar: Resimler (veya Textler, modele göre değişir ama çapraz kontrolle çözeriz)
            
            # Garanti Yöntem: Matrisin şekline bakmaksızın indekslerle manuel kontrol
            # Ama önce tensörün CPU'ya alınması ve float olması iyi olur
            logits = logits.float().cpu()
            
            # Winoground mantığı:
            # c0_i0 = Text0 ile Image0 skoru
            
            # Eğer ColPali kullandıysak processor.score() -> [Queries, Images] döner.
            # Yani Satır=Text, Sütun=Image
            if model_type == "colpali":
                 c0_i0 = logits[0, 0] # Text0 - Image0
                 c0_i1 = logits[0, 1] # Text0 - Image1
                 c1_i0 = logits[1, 0] # Text1 - Image0
                 c1_i1 = logits[1, 1] # Text1 - Image1
            else:
                 # Standart modellerde matmul(img, text.t) -> [Image, Text] döner.
                 # Yani Satır=Image, Sütun=Text
                 c0_i0 = logits[0, 0] # Image0 - Text0
                 c0_i1 = logits[0, 1] # Image0 - Text1
                 c1_i0 = logits[1, 0] # Image1 - Text0
                 c1_i1 = logits[1, 1] # Image1 - Text1
            
            # 1. Text Score: Doğru resim verildiğinde doğru metin (Satır/Sütun fark etmez, mantık aynı)
            # Resim 0 sabitken: Text0 puanı > Text1 puanı olmalı
            # Resim 1 sabitken: Text1 puanı > Text0 puanı olmalı
            
            if model_type == "colpali":
                # ColPali (Satır=Text, Sütun=Resim)
                # Resim 0 (Sütun 0): Text0(Satır0) > Text1(Satır1)
                text_success = (c0_i0 > c1_i0) and (c1_i1 > c0_i1)
                # Image Score (Satır 0 yani Text0 sabit): Resim0 > Resim1
                image_success = (c0_i0 > c0_i1) and (c1_i1 > c1_i0)
            else:
                # Standart (Satır=Resim, Sütun=Text)
                # Resim 0 (Satır 0): Text0(Sütun0) > Text1(Sütun1)
                text_success = (c0_i0 > c0_i1) and (c1_i1 > c1_i0)
                # Image Score (Sütun 0 yani Text0 sabit): Resim0 > Resim1
                image_success = (c0_i0 > c1_i0) and (c1_i1 > c0_i1)

            # Group Score
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
        # import traceback
        # traceback.print_exc()
        clean_memory()
        return {"Model": model_name, "Error": str(e)}

# --- MAIN ---
if __name__ == "__main__":
    print(f"Benchmark v2 Başlıyor... Cihaz: {DEVICE}")
    
    try:
        dataset = load_dataset("facebook/winoground", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"Dataset yüklenemedi: {e}")
        sys.exit(1)

    final_report = []

    for m in MODELS_TO_TEST:
        clean_memory()
        res = run_test(m, dataset)
        final_report.append(res)
        print(f"--> {res['Model']} Sonuç: {res.get('Image Score')}")

    print("\n" + "="*60)
    print("NİHAİ SONUÇLAR (V2)")
    print("="*60)
    print(f"{'.':<20} | {'.':<18} | {'.':<18}")
    print("-" * 60)
    
    with open("benchmark_results_v2.txt", "w") as f:
        f.write("Model,Image Score,Group Score\n")
        for r in final_report:
            if "Error" in r:
                line = f"{r['Model']:<20} | ERROR: {r['Error']}"
            else:
                line = f"{r['Model']:<20} | {r['Image Score']:<18} | {r['Group Score']:<18}"
                f.write(f"{r['Model']},{r['Image Score']},{r['Group Score']}\n")
            print(line)
