
# PRE-BENCHMARK GUIDE: Multimodal & ColPali Evaluation on Mac M4

**Amaç:** Mac Mini M4 (16GB RAM) üzerinde, hem standart "Dense" embedding modellerini hem de yeni nesil "ColPali" (Late Interaction) modelini **facebook/winoground** veri seti ile kıyaslamak.

**Otomasyon:** Script, veri setini Hugging Face Hub üzerinden **otomatik indirecek**, modelleri **sırayla** belleğe alıp test edecek ve sonuçları raporlayacaktır. Manuel dosya indirmeye gerek yoktur.

## ADIM 1: Ortam Kurulumu
Aşağıdaki komut, ColPali ve diğer modellerin çalışması için gerekli tüm kütüphaneleri kurar:
```bash
pip install torch torchvision transformers datasets pillow timm einops
```

## ADIM 2: Kullanılacak Veri Seti (Otomatik İndirilir)
*   **Dataset:** `facebook/winoground`
*   **Kaynak:** Hugging Face Hub (Script içindeki `load_dataset` fonksiyonu bunu otomatik çeker).
*   **Yapı:** Her örnekte 2 Resim ve 2 Metin bulunur.
*   **Değerlendirme Metriği:**
    *   **Image Score (Retrieval):** Metin verildiğinde doğru resmi bulma başarısı. (Senin önceliğin).
    *   **Group Score (Reasoning):** Metin-Resim eşleşmelerinin tamamının (4 kombinasyon) doğru yapılması. (Zeka testi).

## ADIM 3: Nihai Benchmark Scripti (`run_benchmark_final.py`)
Lütfen aşağıdaki kodu kopyala ve çalıştır. Bu kod; bellek temizliği, ColPali için özel "score" hesaplaması ve otomatik veri seti indirme işlemlerini içerir.

```python
import torch
import gc
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor
import sys

# --- AYARLAR ---
# Mac M4 için Metal Performance Shaders (MPS)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# ColPali ve büyük modellerin sığması için FP16 zorunlu
DTYPE = torch.float16 

# --- MODEL LİSTESİ ---
# 'type': 'dense' -> Standart modeller (CLIP, SigLIP vb.)
# 'type': 'colpali' -> Late Interaction (ColPali)
MODELS_TO_TEST = [
    # Standart Modeller (RAM Dostu)
    {"name": "OpenAI-CLIP-L", "id": "openai/clip-vit-large-patch14", "type": "dense"},
    {"name": "SigLIP-400M",   "id": "google/siglip-so400m-patch14-384", "type": "dense"},
    {"name": "Jina-CLIP-v1",  "id": "jinaai/jina-clip-v1", "trust": True, "type": "dense"},
    {"name": "DataComp-XL",   "id": "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K", "type": "dense"},
    {"name": "Nomic-Vision",  "id": "nomic-ai/nomic-embed-vision-v1.5", "trust": True, "type": "dense"},
    {"name": "EVA02-CLIP-L",  "id": "QuanSun/EVA02-CLIP-L-14", "trust": True, "type": "dense"},
    
    # Özel Mimari (RAM Kullanımı Yüksek - En Sonda)
    {"name": "ColPali-v1.3",  "id": "vidore/colpali-v1.3", "trust": True, "type": "colpali"}
]

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
                # ColPali Özel Akışı (MaxSim)
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
                        img_embeds = outputs[0]
                        text_embeds = outputs[1]

                    # Normalizasyon ve Çarpım
                    img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
                    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                    
                    # [Resim, Text] -> Transpose -> [Text, Resim]
                    logits = torch.matmul(img_embeds, text_embeds.t()).t()

            # --- SKORLAMA (WINOGROUND STANDARD) ---
            # logits[0,0] = Text0 - Image0 (Doğru)
            # logits[0,1] = Text0 - Image1 (Yanlış)
            c0_i0, c0_i1 = logits[0, 0], logits[0, 1]
            c1_i0, c1_i1 = logits[1, 0], logits[1, 1]

            # 1. Text Score: Doğru resim verildiğinde doğru metni seçme
            text_success = (c0_i0 > c1_i0) and (c1_i1 > c0_i1)
            
            # 2. Image Score: Doğru metin verildiğinde doğru resmi seçme (Retrieval)
            image_success = (c0_i0 > c0_i1) and (c1_i1 > c1_i0)
            
            # 3. Group Score: Tam Zeka Testi
            group_success = text_success and image_success

            if text_success: text_score_count += 1
            if image_success: image_score_count += 1
            if group_success: group_score_count += 1
            
            # İlerleme çubuğu gibi her 50 adımda bir yaz
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
        clean_memory()
        return {"Model": model_name, "Error": str(e)}

# --- MAIN BLOCK ---
if __name__ == "__main__":
    print(f"Benchmark Başlatılıyor... Cihaz: {DEVICE}")
    print("Veri Seti: facebook/winoground (Hugging Face'den indiriliyor...)")
    
    # Veri setini yükle
    try:
        dataset = load_dataset("facebook/winoground", split="test", trust_remote_code=True)
    except Exception as e:
        print("Dataset indirme hatası! Lütfen internet bağlantınızı kontrol edin.")
        print(f"Hata detayı: {e}")
        sys.exit(1)
        
    print(f"Veri Seti Hazır: {len(dataset)} çift.")

    final_report = []

    for m in MODELS_TO_TEST:
        clean_memory()
        res = run_test(m, dataset)
        final_report.append(res)
        print(f"--> {res['Model']} Bitti. (ImgScore: {res.get('Image Score')})")

    # SONUÇLARI YAZDIR
    print("\n" + "="*60)
    print("NİHAİ SONUÇLAR")
    print("="*60)
    print(f"{'Model':<20} | {'Image Retrieval':<18} | {'Reasoning (IQ)':<18}")
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
```