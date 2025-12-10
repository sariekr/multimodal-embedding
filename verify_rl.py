import json
from vllm import LLM, SamplingParams
from peft import PeftModel

# 1. Dataseti Yükle
with open("dataset.json", "r") as f:
    dataset = json.load(f)

# 2. AYNI MODELLERİ YÜKLE
# Not: vLLM şu an LoRA adapterları inference anında yüklemeyi destekliyor ama
# en temiz yöntem Transformers ile merge etmek veya LoRA parametresini vermektir.
# Basitlik için vLLM'in 'enable_lora' özelliğini kullanacağız.

print("Eğitilmiş model yükleniyor...")
# RunPod'da kaydettiğin klasör adı
lora_path = "qwen-rl-pure-lora-result" 

llm = LLM(
    model="OpenPipe/Qwen3-14B-Instruct",
    enable_lora=True, # <--- LoRA desteğini açıyoruz
    max_lora_rank=16,
    gpu_memory_utilization=0.90,
    trust_remote_code=True
)

# 3. System Prompt (Aynı prompt)
system_prompt = """<|im_start|>system
You are a strict data extraction engine.
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

# 4. LoRA Adapter'ı Kullanarak Cevap Üret
sampling_params = SamplingParams(
    temperature=0, 
    max_tokens=1000,
)

# vLLM'de LoRA'yı isteğe bağlı takabiliriz. 
# lora_request parametresi ile senin eğittiğin dosyayı gösteriyoruz.
from vllm.lora.request import LoRARequest

print("Sorular cevaplanıyor (Eğitilmiş Model ile)...")
outputs = llm.generate(
    prompts, 
    sampling_params,
    lora_request=LoRARequest("my_rl_adapter", 1, lora_path) # <--- SENİN EĞİTTİĞİN MODEL BURADA
)

# 5. İstatistikleri Hesapla
stats = {"total": len(dataset), "think_error": 0, "json_success": 0}

print("\n" + "="*50)
print("EĞİTİM SONRASI CEVAPLAR (İLK 3 TANE)")
print("="*50)

for i, output in enumerate(outputs):
    res = output.outputs[0].text
    
    # Kontrol
    has_think = "<think>" in res
    is_json = res.strip().startswith("{")
    
    if has_think: stats["think_error"] += 1
    if not has_think and is_json: stats["json_success"] += 1

    if i < 3:
        print(f"SORU: {dataset[i]['prompt']}")
        print(f"CEVAP: {res[:300]}") # Artık kısa olmalı!
        print("-" * 30)

print("\n" + "="*50)
print("📊 FİNAL SKOR TABLOSU")
print("="*50)
print(f"Baseline Başarısı : %0 (Hatırla)")
print(f"Eğitim Sonrası    : %{stats['json_success']/stats['total']*100:.1f} Mükemmel JSON")
print(f"Hala Think Var mı?: {stats['think_error']} adet")
print("="*50)