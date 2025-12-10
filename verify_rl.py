import json
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# 1. AYARLAR
lora_path = "/workspace/multimodal-embedding/qwen-rl-pro-result" # Senin PRO eğitim klasörün
base_model = "OpenPipe/Qwen3-14B-Instruct"

# 2. DATASET VE BEKLENEN CEVAPLAR
# Doğruluk ölçmek için "Beklenen" (Ground Truth) mantığını basitçe kuralım
def get_expected_category(text):
    text = text.lower()
    if any(k in text for k in ["bill", "charge", "refund", "money", "price", "cost", "pay", "card"]): return "BILLING"
    if any(k in text for k in ["bug", "crash", "error", "login", "screen", "app", "broken", "slow"]): return "TECHNICAL"
    if any(k in text for k in ["package", "delivery", "track", "arrive", "ship", "lost", "where"]): return "SHIPPING"
    return "OTHER"

with open("dataset.json", "r") as f:
    dataset = json.load(f)

# 3. MODELİ YÜKLE
print(f"Model Yükleniyor: {base_model} + LoRA...")
llm = LLM(
    model=base_model,
    enable_lora=True,
    max_lora_rank=16,
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
    dtype="bfloat16"
)

# 4. PROMPT
system_prompt = """<|im_start|>system
You are a strict data extraction engine.
RULES:
1. Output ONLY a JSON object.
2. DO NOT use <think> tags.
3. Allowed categories: ["BILLING", "TECHNICAL", "SHIPPING", "PRODUCT", "OTHER"].
4. Do not output any other text.<|im_end|>
"""

prompts = []
expected_answers = [] # Doğru cevap anahtarı

for item in dataset:
    user_text = item['prompt']
    prompts.append(f"{system_prompt}<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n")
    expected_answers.append(get_expected_category(user_text))

# 5. CEVAP ÜRET
sampling_params = SamplingParams(temperature=0, max_tokens=200)

print("🚀 Final Sınav Başlıyor...")
outputs = llm.generate(
    prompts, 
    sampling_params,
    lora_request=LoRARequest("rl_adapter", 1, lora_path)
)

# 6. DETAYLI ANALİZ
stats = {
    "total": len(dataset),
    "think_clean": 0,      # Think etiketi yok (Format Başarısı)
    "json_clean": 0,       # Valid JSON (Format Başarısı)
    "logic_correct": 0,    # Doğru Kategori (Zeka Başarısı)
    "other_escape": 0      # OTHER kaçamağı
}

print("\n" + "="*80)
print(f"{'SORU (Özet)':<40} | {'MODELİN CEVABI':<15} | {'DOĞRU MU?':<10}")
print("-" * 80)

for i, output in enumerate(outputs):
    res = output.outputs[0].text.strip()
    truth = expected_answers[i]
    
    # 1. Format Kontrolü
    has_think = "<think>" in res
    if not has_think: stats["think_clean"] += 1
    
    # 2. JSON ve Kategori Kontrolü
    model_cat = "INVALID"
    try:
        if res.startswith("{") and res.endswith("}"):
            stats["json_clean"] += 1
            data = json.loads(res)
            model_cat = data.get("category", "UNKNOWN")
    except:
        pass
        
    # 3. Mantık (Logic) Kontrolü
    is_correct = (model_cat == truth)
    if is_correct: stats["logic_correct"] += 1
    
    if model_cat == "OTHER": stats["other_escape"] += 1

    # İlk 10 örneği ekrana bas
    if i < 10:
        short_q = (dataset[i]['prompt'][:35] + '..') if len(dataset[i]['prompt']) > 35 else dataset[i]['prompt']
        status = "✅" if is_correct else f"❌ ({truth})"
        print(f"{short_q:<40} | {model_cat:<15} | {status}")

print("="*80)
print("📊 BENCHMARK RAPORU (Zeka & Refleks)")
print("="*80)
print(f"Toplam Veri          : {stats['total']}")
print(f"🧠 Mantık Doğruluğu  : %{stats['logic_correct']/stats['total']*100:.1f} ({stats['logic_correct']}/{stats['total']}) -> ASIL SKOR BU")
print(f"😶 Sessizlik Başarısı: %{stats['think_clean']/stats['total']*100:.1f}")
print(f"📋 JSON Formatı      : %{stats['json_clean']/stats['total']*100:.1f}")
print("="*80)