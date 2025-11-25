import torch
import gc
import sys
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor
# ColPali Kontrolü
try:
    from colpali_engine.models import ColPali, ColPaliProcessor
    COLPALI_AVAILABLE = True
except ImportError:
    COLPALI_AVAILABLE = False

# --- AYARLAR ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16
SAMPLE_SIZE = 100 # Test edilecek resim sayısı

# --- MODEL LİSTESİ (v5 - Flickr30k Edition) ---
MODELS_TO_TEST = [
    # 1. Baseline
    {"name": "OpenAI-CLIP-L", "id": "openai/clip-vit-large-patch14", "type": "dense"},
    
    # 2. SigLIP Large
    {"name": "SigLIP-400M",   "id": "google/siglip-so400m-patch14-384", "type": "dense"},
    
    # 3. YENİ EKLENEN: BGE-Visualized (RAG Expert)
    {"name": "BGE-Visualized", "id": "BAAI/bge-visualized-base-en-v1.5", "trust": True, "type": "dense"},
    
    # 4. Retrieval Expert
    {"name": "Jina-CLIP-v1",  "id": "jinaai/jina-clip-v1", "trust": True, "type": "dense"},
    
    # 5. Heavyweight
    {"name": "LAION-CLIP-H",  "id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "type": "dense"},
    
    # 6. ColPali
    {"name": "ColPali-v1.3",  "id": "vidore/colpali-v1.3", "trust": True, "type": "colpali"}
]

def clean_memory():
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    torch.mps.empty_cache()
    gc.collect()

def run_retrieval_test(model_info, dataset):
    model_name = model_info["name"]
    model_id = model_info["id"]
    model_type = model_info["type"]
    trust_code = model_info.get("trust", False)
    
    print(f"\n>>> Yükleniyor: {model_name}...")
    
    try:
        # --- MODEL YÜKLEME ---
        if model_type == "colpali":
            if not COLPALI_AVAILABLE: return {"Model": model_name, "Error": "ColPali Lib Missing"}
            model = ColPali.from_pretrained(model_id, torch_dtype=DTYPE, device_map=DEVICE).eval()
            processor = ColPaliProcessor.from_pretrained(model_id)
        else:
            # Standart Modeller
            model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_code, torch_dtype=DTYPE).to(DEVICE)
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_code)
            model.eval()
        
        # --- VERİ HAZIRLIĞI ---
        print(f"Veri Hazırlanıyor ({SAMPLE_SIZE} örnek)...")
        
        # Resimleri al (PIL Image objeleri)
        images = [item["image"].convert("RGB") for item in dataset]
        
        # Sorguları al. Flickr30k'da 'caption' genelde listedir, ilkini alıyoruz.
        # Dataset yapısına göre kontrol etmekte fayda var.
        first_item = dataset[0]
        if isinstance(first_item["caption"], list):
            queries = [item["caption"][0] for item in dataset]
        else:
            # Eğer string gelirse direkt al
            queries = [item["caption"] for item in dataset]
        
        # --- EMBEDDING ÇIKARMA ---
        with torch.no_grad():
            if model_type == "colpali":
                # ColPali Batch İşleme
                print("ColPali Embeddings hesaplanıyor...")
                # Resimler
                batch_images = processor.process_images(images).to(DEVICE)
                image_embeds = model(**batch_images)
                
                # Metinler
                batch_queries = processor.process_queries(queries).to(DEVICE)
                query_embeddings = model(**batch_queries)
                
                # Skorlama
                scores_matrix = processor.score(query_embeddings, image_embeddings)
                
            else:
                # Dense Modeller
                print("Dense Embeddings hesaplanıyor...")
                
                # Görüntü
                inputs_img = processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
                # Modelin hangi metodu kullandığını kontrol et (CLIP vs diğerleri)
                if hasattr(model, 'get_image_features'):
                    img_out = model.get_image_features(**inputs_img)
                else:
                    # BGE, Jina gibi modeller doğrudan model(**inputs) dönebilir
                    # SigLIP -> vision_model
                    # En garanti yol: HuggingFace AutoModel genelde output.image_embeds veya output[0] döner
                    # BGE-Visualized özel durumu: CLIP tabanlıdır, get_image_features çalışmalı.
                    # Eğer çalışmazsa, outputları alıp filtreleyeceğiz.
                    out = model(**inputs_img)
                    img_out = out.image_embeds if hasattr(out, 'image_embeds') else out[0]

                # Metin
                inputs_text = processor(text=queries, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                if hasattr(model, 'get_text_features'):
                    text_out = model.get_text_features(**inputs_text)
                else:
                    out = model(**inputs_text)
                    text_out = out.text_embeds if hasattr(out, 'text_embeds') else out[0]

                # 3D Tensör Düzeltmesi (Nomic vb. için)
                if img_out.dim() == 3: img_out = img_out[:, 0, :]
                if text_out.dim() == 3: text_out = text_out[:, 0, :]

                # Normalize
                img_out = img_out / img_out.norm(dim=-1, keepdim=True)
                text_out = text_out / text_out.norm(dim=-1, keepdim=True)
                
                # Matris Çarpımı: [Query, Image]
                scores_matrix = torch.matmul(text_out, img_out.t())

        # --- METRİK HESAPLAMA (Recall@K) ---
        scores_matrix = scores_matrix.float().cpu()
        num_queries = len(queries)
        
        r1_count = 0
        r5_count = 0
        
        for i in range(num_queries):
            scores = scores_matrix[i]
            top_k_indices = torch.topk(scores, k=5).indices.tolist()
            
            if i == top_k_indices[0]:
                r1_count += 1
            if i in top_k_indices:
                r5_count += 1
                
        results = {
            "Model": model_name,
            "Recall@1": f"{r1_count/num_queries:.2%}",
            "Recall@5": f"{r5_count/num_queries:.2%}"
        }
        
        del model
        del processor
        clean_memory()
        return results

    except Exception as e:
        print(f"!!! HATA ({model_name}): {str(e)}")
        import traceback
        traceback.print_exc()
        clean_memory()
        return {"Model": model_name, "Error": str(e)}

# --- MAIN ---
if __name__ == "__main__":
    print(f"Benchmark v5 (Flickr30k) Başlıyor... Cihaz: {DEVICE}")
    print(f"Yeni Model Eklendi: BAAI/bge-visualized (RAG Expert)")
    
    # Dataset Seçimi: kakaobrain genellikle daha stabildir
    print("Dataset indiriliyor (kakaobrain/flickr30k_test)...")
    try:
        dataset = load_dataset("kakaobrain/flickr30k_test", split="test", trust_remote_code=True)
        dataset = dataset.select(range(SAMPLE_SIZE))
    except Exception as e1:
        print(f"Kakaobrain yüklenemedi ({e1}). Alternatif (nlphuji) deneniyor...")
        try:
            dataset = load_dataset("nlphuji/flickr30k", split="test", trust_remote_code=True)
            dataset = dataset.select(range(SAMPLE_SIZE))
        except Exception as e2:
             print(f"Dataset yüklenemedi. Lütfen internetinizi kontrol edin.\nDetay: {e2}")
             sys.exit(1)

    print(f"Test Seti Hazır: {len(dataset)} Resim-Metin Çifti")
    
    final_report = []

    for m in MODELS_TO_TEST:
        clean_memory()
        res = run_retrieval_test(m, dataset)
        final_report.append(res)
        print(f"--> {res['Model']} Sonuç: R@1={res.get('Recall@1')}")

    print("\n" + "="*60)
    print("NİHAİ SONUÇLAR (FLICKR30K - REAL WORLD)")
    print("="*60)
    print(f"{'Model':<20} | {'Recall@1 (Top-1)':<18} | {'Recall@5 (Top-5)':<18}")
    print("-" * 60)
    
    with open("benchmark_results_flickr.txt", "w") as f:
        f.write("Model,Recall@1,Recall@5\n")
        for r in final_report:
            if "Error" in r:
                line = f"{r['Model']:<20} | ERROR: {r['Error']}"
            else:
                line = f"{r['Model']:<20} | {r['Recall@1']:<18} | {r['Recall@5']:<18}"
                f.write(f"{r['Model']},{r['Recall@1']},{r['Recall@5']}\n")
            print(line)
