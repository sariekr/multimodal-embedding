import json
from vllm import LLM, SamplingParams

# 1. Dataseti Yükle
with open("dataset.json", "r") as f:
    dataset = json.load(f)

# 2. Modeli Yükle (W&B'dekinin aynısı)
llm = LLM(model="OpenPipe/Qwen3-14B-Instruct", trust_remote_code=True, gpu_memory_utilization=0.90)

# 3. System Prompt (Modeli zorla susturmaya çalıştığımız hali)
system_prompt = """<|im_start|>system
You are a strict data extraction engine. Analyze the customer request.
RULES:
1. Output ONLY a JSON object.
2. DO NOT use <think> tags.
3. Allowed categories: ["BILLING", "TECHNICAL", "SHIPPING", "PRODUCT", "OTHER"].
4. Do not output any other text.<|im_end|>
"""

prompts = []
for item in dataset:
    user_text = item['prompt']
    full_prompt = f"{system_prompt}<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    prompts.append(full_prompt)

# 4. Toplu Çalıştır (Batch Inference)
# Max tokens 1000 verdik ki thinking yaparsa yer kalsın, kesilmesin.
sampling_params = SamplingParams(temperature=0, max_tokens=1000)
outputs = llm.generate(prompts, sampling_params)

# 5. İstatistikleri Hesapla
stats = {
    "total": len(dataset),
    "think_tag_error": 0,
    "format_error": 0, # JSON değil veya yazı var
    "json_success": 0, # Temiz JSON
    "other_count": 0   # Kaçamak cevap
}

print("\n" + "="*50)
print("BASELINE TEST SONUÇLARI (İLK 5 ÖRNEK)")
print("="*50)

for i, output in enumerate(outputs):
    res = output.outputs[0].text
    
    # Hata Kontrolleri
    has_think = "<think>" in res
    is_clean_json = res.strip().startswith("{") and res.strip().endswith("}")
    
    if has_think: stats["think_tag_error"] += 1
    if not is_clean_json: stats["format_error"] += 1
    if not has_think and is_clean_json: stats["json_success"] += 1
    
    # Kategori Kontrolü (Kabaca)
    if '"category": "OTHER"' in res or "'category': 'OTHER'" in res:
        stats["other_count"] += 1

    # İlk 5 örneği ekrana bas (Gözle kontrol için)
    if i < 5:
        print(f"SORU: {dataset[i]['prompt']}")
        print(f"CEVAP: {res[:200]}...") # Sadece başını göster
        print("-" * 30)

print("\n" + "="*50)
print("📊 BASELINE İSTATİSTİKLERİ")
print("="*50)
print(f"Toplam Veri: {stats['total']}")
print(f"❌ <think> Tag Hatası: {stats['think_tag_error']} ({stats['think_tag_error']/stats['total']:.0%})")
print(f"❌ Format Hatası: {stats['format_error']} ({stats['format_error']/stats['total']:.0%})")
print(f"⚠️ 'OTHER' Kaçamağı: {stats['other_count']} ({stats['other_count']/stats['total']:.0%})")
print(f"✅ Mükemmel JSON: {stats['json_success']} ({stats['json_success']/stats['total']:.0%})")
print("="*50)