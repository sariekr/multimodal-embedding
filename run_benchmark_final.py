import torch
import gc
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor
import sys
import os

# --- AYARLAR ---
# Mac M4 için Metal Performance Shaders (MPS)
# Eğer MPS müsait değilse CPU'ya düşer.
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 

# --- MODEL LİSTESİ ---
MODELS_TO_TEST = [
    # Standart Modeller (RAM Dostu)
    {"name": "OpenAI-CLIP-L", "id": "openai/clip-vit-large-patch14", "type": "dense"},
    {"name": "SigLIP-400M",   "id": "google/siglip-so400m-patch14-384", "type": "dense"},
    {"name": "Jina-CLIP-v1",  "id": "jinaai/jina-clip-v1", "trust": True, "type": "dense"},
    {"name": "Nomic-Vision",  "id": "nomic-ai/nomic-embed-vision-v1.5", "trust": True, "type": "dense"},
    # EVA02 Bazen timm versiyon sorunu çıkarabiliyor, yine de ekleyelim
    {"name": "EVA02-CLIP-L",  "id": "QuanSun/EVA02-CLIP-L-14", "trust": True, "type": "dense"},
    
    # Özel Mimari (RAM Kullanımı Yüksek - En Sonda)
    {"name": "ColPali-v1.3",  "id": "vidore/colpali-v1.3", "trust": True, "type": "colpali"}
]

# DataComp-XL çok büyük olduğu için (RAM problemi riski) varsayılan listeden çıkardım.
# İstenirse eklenebilir: {"name": "DataComp-XL", "id": "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K", "type": "dense"}

def clean_memory():
    """RAM Sızıntısını Önlemek İçin Temizlik"""
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    torch.mps.empty_cache()
    gc.collect()

def run_test(model_info, dataset):
    model_name = model_info["name"]
    model_id = model_info["id"]
    model_type = model_info["type"]
    trust_code = model_info.get("trust", False)
    
    print(f"\n>>> Yükleniyor: {model_name} (Tip: {model_type})...")
    
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
            # Winoground verisini hazırla
            images = [example["image_0"].convert("RGB"), example["image_1"].convert("RGB")]
            texts = [example["caption_0"], example["caption_1"]]
            
            # --- HESAPLAMA (MİMARİYE GÖRE) ---
            if model_type == "colpali":
                # ColPali Özel Akışı
                with torch.no_grad():
                    batch_images = processor.process_images(images).to(DEVICE)
                    batch_queries = processor.process_queries(texts).to(DEVICE)
                    # ColPali processor.score() metodu direkt benzerlik skorlarını verir
                    scores = processor.score(batch_queries, batch_images, model=model)
                    logits = scores # [Text Sayısı, Resim Sayısı] döner

            else:
                # Standart Model Akışı (Cosine Sim)
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
                        # Bazı modeller (Jina gibi) output'u tuple dönebilir veya farklı isimlendirebilir
                        # Genellikle ilk eleman image, ikinci text embedding olur ama modele göre değişir.
                        # HuggingFace standart modelleri genelde image_embeds/text_embeds attribute'una sahiptir.
                        # Eğer yoksa manuel handle etmek gerekebilir.
                        # JinaCLIP v1 için genelde image_embeds döner.
                        # Garanti olsun diye kontrol:
                        img_embeds = outputs.image_embeds if hasattr(outputs, 'image_embeds') else outputs[0]
                        text_embeds = outputs.text_embeds if hasattr(outputs, 'text_embeds') else outputs[1]

                    # Normalizasyon ve Çarpım
                    img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
                    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                    
                    # [Resim, Text] -> Transpose -> [Text, Resim]
                    logits = torch.matmul(text_embeds, img_embeds.t())

            # --- SKORLAMA (WINOGROUND STANDARD) ---
            # logits[0,0] = Text0 - Image0 (Doğru Eşleşme)
            # logits[0,1] = Text0 - Image1 (Yanlış Eşleşme)
            # logits[1,0] = Text1 - Image0 (Yanlış Eşleşme)
            # logits[1,1] = Text1 - Image1 (Doğru Eşleşme)
            
            c0_i0 = logits[0, 0].item()
            c0_i1 = logits[0, 1].item()
            c1_i0 = logits[1, 0].item()
            c1_i1 = logits[1, 1].item()

            # 1. Text Score: Doğru resim verildiğinde doğru metni seçme
            # Resim 0 için: Text 0 > Text 1 olmalı
            # Resim 1 için: Text 1 > Text 0 olmalı
            text_success = (c0_i0 > c1_i0) and (c1_i1 > c0_i1)
            
            # 2. Image Score: Doğru metin verildiğinde doğru resmi seçme (Retrieval)
            # Text 0 için: Resim 0 > Resim 1 olmalı
            # Text 1 için: Resim 1 > Resim 0 olmalı
            image_success = (c0_i0 > c0_i1) and (c1_i1 > c1_i0)
            
            # 3. Group Score: Tam Zeka Testi (Hepsi doğru olmalı)
            group_success = text_success and image_success

            if text_success: text_score_count += 1
            if image_success: image_score_count += 1
            if group_success: group_score_count += 1
            
            # İlerleme çubuğu
            if (i+1) % 50 == 0:
                print(f"  > {i+1}/{total_samples} tamamlandı.")
        
        results = {
            "Model": model_name,
            "Image Score": f"{image_score_count/total_samples:.2%}",
            "Group Score": f"{group_score_count/total_samples:.2%}"
        }
        
        # Temizlik
        del model
        del processor
        clean_memory()
        return results

    except Exception as e:
        print(f"!!! KRİTİK HATA ({model_name}): {str(e)}")
        import traceback
        traceback.print_exc()
        clean_memory()
        return {"Model": model_name, "Error": str(e)}

# --- MAIN BLOCK ---
if __name__ == "__main__":
    print(f"Benchmark Başlatılıyor... Cihaz: {DEVICE}")
    print("Veri Seti: facebook/winoground (Hugging Face'den indiriliyor...)")
    
    # Veri setini yükle
    try:
        # Hugging Face Token'ı env variable'dan veya direkt login'den alabilir.
        # Token sorunu olursa "huggingface-cli login" yapılması gerekebilir.
        dataset = load_dataset("facebook/winoground", split="test", trust_remote_code=True)
    except Exception as e:
        print("\n!!! Dataset indirme hatası !!!")
        print("Lütfen internet bağlantınızı kontrol edin veya Hugging Face Token sorunu olabilir.")
        print("Hata detayı:", e)
        sys.exit(1)
        
    print(f"Veri Seti Hazır: {len(dataset)} çift.")

    final_report = []

    for m in MODELS_TO_TEST:
        clean_memory()
        res = run_test(m, dataset)
        final_report.append(res)
        print(f"--> {res['Model']} Bitti. (ImgScore: {res.get('Image Score', 'N/A')})")

    # SONUÇLARI YAZDIR
    print("\n" + "="*60)
    print("NİHAİ SONUÇLAR")
    print("="*60)
    print(f"{ 'Model':<20} | {'Image Retrieval':<18} | {'Reasoning (IQ)':<18}")
    print("-" * 60)
    
    with open("final_benchmark_results.txt", "w") as f:
        f.write("Model,Image Score,Group Score\n")
        for r in final_report:
            if "Error" in r:
                line = f"{r['Model']:<20} | ERROR: {r['Error']}"
            else:
                line = f"{r['Model']:<20} | {r['Image Score']:<18} | {r['Group Score']:<18}"
                f.write(f"{r['Model']},{r['Image Score']},{r['Group Score']}\n")
            print(line)
            
    print("\nTamamlandı. Sonuçlar 'final_benchmark_results.txt' dosyasına kaydedildi.")
