import torch
import gc
import sys
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor
try:
    from colpali_engine.models import ColPali, ColPaliProcessor
    COLPALI_AVAILABLE = True
except ImportError:
    COLPALI_AVAILABLE = False

# --- AYARLAR ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16
SAMPLE_SIZE = 100

# --- MODEL LİSTESİ ---
MODELS_TO_TEST = [
    {"name": "OpenAI-CLIP-L", "id": "openai/clip-vit-large-patch14", "type": "dense"},
    {"name": "SigLIP-400M",   "id": "google/siglip-so400m-patch14-384", "type": "dense"},
    {"name": "BGE-Visualized", "id": "BAAI/bge-visualized-base-en-v1.5", "trust": True, "type": "dense"},
    {"name": "Jina-CLIP-v1",  "id": "jinaai/jina-clip-v1", "trust": True, "type": "dense"},
    {"name": "LAION-CLIP-H",  "id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "type": "dense"},
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
        if model_type == "colpali":
            if not COLPALI_AVAILABLE: return {"Model": model_name, "Error": "ColPali Lib Missing"}
            model = ColPali.from_pretrained(model_id, torch_dtype=DTYPE, device_map=DEVICE).eval()
            processor = ColPaliProcessor.from_pretrained(model_id)
        else:
            model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_code, torch_dtype=DTYPE).to(DEVICE)
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_code)
            model.eval()
        
        print(f"Veri Hazırlanıyor ({SAMPLE_SIZE} örnek)...")
        
        # Dataset Column Mapping (lmms-lab/flickr30k için)
        # Genellikle 'image' ve 'caption' sütunları vardır.
        # Bazen caption bir liste, bazen stringdir.
        
        images = [item["image"].convert("RGB") for item in dataset]
        
        # Caption işleme
        # lmms-lab'de genelde 'caption' sütunu vardır ama kontrol edelim.
        if "caption" in dataset.features:
            queries = []
            for item in dataset:
                cap = item["caption"]
                if isinstance(cap, list): queries.append(cap[0]) # Liste ise ilkini al
                else: queries.append(cap) # String ise direkt al
        elif "sentids" in dataset.features: # nlphuji yapısı (yedek)
             queries = [item["caption"][0] for item in dataset]
        else:
             # Column adlarını bulmaya çalış
             text_col = [c for c in dataset.column_names if 'cap' in c or 'text' in c][0]
             queries = [item[text_col] for item in dataset]
             if isinstance(queries[0], list): queries = [q[0] for q in queries]

        # --- EMBEDDING ---
        with torch.no_grad():
            if model_type == "colpali":
                print("ColPali hesaplanıyor...")
                batch_images = processor.process_images(images).to(DEVICE)
                image_embeds = model(**batch_images)
                
                batch_queries = processor.process_queries(queries).to(DEVICE)
                query_embeddings = model(**batch_queries)
                
                scores_matrix = processor.score(query_embeddings, image_embeddings)
                
            else:
                print("Dense hesaplanıyor...")
                inputs_img = processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
                
                # Model Tipi Kontrolü
                if hasattr(model, 'get_image_features'):
                    img_out = model.get_image_features(**inputs_img)
                else:
                    # BGE-Visualized gibi modellerde 'vision_model' attribute'u olabilir
                    # veya direkt forward ile output verir.
                    # Güvenli yöntem: AutoModel output'undan image_embeds çekmek
                    out = model(**inputs_img)
                    img_out = out.image_embeds if hasattr(out, 'image_embeds') else out[0]

                inputs_text = processor(text=queries, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                if hasattr(model, 'get_text_features'):
                    text_out = model.get_text_features(**inputs_text)
                else:
                    out = model(**inputs_text)
                    text_out = out.text_embeds if hasattr(out, 'text_embeds') else out[0]

                # Boyut Düzeltme (3D -> 2D)
                if img_out.dim() == 3: img_out = img_out[:, 0, :]
                if text_out.dim() == 3: text_out = text_out[:, 0, :]

                # Normalize
                img_out = img_out / img_out.norm(dim=-1, keepdim=True)
                text_out = text_out / text_out.norm(dim=-1, keepdim=True)
                
                scores_matrix = torch.matmul(text_out, img_out.t())

        # --- RECALL ---
        scores_matrix = scores_matrix.float().cpu()
        num_queries = len(queries)
        r1, r5 = 0, 0
        
        for i in range(num_queries):
            scores = scores_matrix[i]
            top_k = torch.topk(scores, k=5).indices.tolist()
            if i == top_k[0]: r1 += 1
            if i in top_k: r5 += 1
                
        results = {
            "Model": model_name,
            "Recall@1": f"{r1/num_queries:.2%}",
            "Recall@5": f"{r5/num_queries:.2%}"
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
    print(f"Benchmark v6 (Flickr30k - Modern) Başlıyor... Cihaz: {DEVICE}")
    
    # Dataset: 'lmms-lab/flickr30k' (Daha modern ve güvenilir bir repo)
    print("Dataset indiriliyor (lmms-lab/flickr30k)...")
    try:
        # 'test' spliti genelde vardır. Yoksa 'validation' veya 'train' kullanacağız.
        dataset = load_dataset("lmms-lab/flickr30k", split="test", trust_remote_code=False) 
        dataset = dataset.select(range(SAMPLE_SIZE))
    except:
        print("Test spliti yok, 'validation' deneniyor...")
        try:
            dataset = load_dataset("lmms-lab/flickr30k", split="validation", trust_remote_code=False)
            dataset = dataset.select(range(SAMPLE_SIZE))
        except:
             print("Validation da yok, 'train' deneniyor...") # Mecburiyetten
             dataset = load_dataset("lmms-lab/flickr30k", split="train", trust_remote_code=False)
             dataset = dataset.select(range(SAMPLE_SIZE))

    print(f"Test Seti: {len(dataset)} Resim-Metin Çifti")
    
    final_report = []

    for m in MODELS_TO_TEST:
        clean_memory()
        res = run_retrieval_test(m, dataset)
        final_report.append(res)
        print(f"--> {res['Model']} Sonuç: R@1={res.get('Recall@1')}")

    print("\n" + "="*60)
    print("NİHAİ SONUÇLAR (FLICKR30K)")
    print("="*60)
    print(f"{ 'Model':<20} | {'Recall@1':<18} | {'Recall@5':<18}")
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
