import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. AYARLAR
base_model_id = "OpenPipe/Qwen3-14B-Instruct"

# 2. DATASET (Aynı zor veri seti)
try:
    with open("dataset_hard.json", "r") as f:
        dataset = json.load(f)
except FileNotFoundError:
    print("❌ HATA: 'dataset_hard.json' yok!")
    exit()

# 3. BASELINE MODELİ YÜKLE
print(f"📉 Baseline (KÖR) Model Yükleniyor: {base_model_id}...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 4. KURAL MOTORU (BİZ BİLİYORUZ AMA MODEL BİLMEYECEK)
def calculate_ground_truth(prompt):
    price = 0
    price_match = re.search(r'\$(\d+)', prompt)
    if price_match: price = int(price_match.group(1))
    
    is_polite = any(w in prompt.lower() for w in ["please", "kindly", "appreciate", "help", "thank"])
    
    if price < 10: return "IGNORE"
    if price > 2000: return "VIP_DESK"
    if is_polite: return "AUTO_BOT"
    return "HUMAN_AGENT"

# 5. CEVAP ÇEKİCİ
def extract_category(text):
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1:
            data = json.loads(text[start:end])
            return data.get("category", "INVALID")
    except: pass
    
    match = re.search(r'"category":\s*"(\w+)"', text, re.IGNORECASE)
    if match: return match.group(1).upper()
    return "INVALID"

# 6. KÖR TEST BAŞLASIN
def generate_blind(prompt):
    # --- İŞTE BURASI DEĞİŞTİ ---
    # Kuralları SİLDİK. Sadece çıktı formatı var.
    system_prompt = """You are a strict automated routing system.
RULES:
1. Output ONLY a JSON object: {"category": "..."}
2. DO NOT use <think> tags.
3. Allowed categories: ["IGNORE", "VIP_DESK", "HUMAN_AGENT", "AUTO_BOT"].
4. Choose the best category based on the input.""" 

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=200, 
            temperature=0.1,
            do_sample=False
        )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 7. RAPOR
stats = {"total": 0, "correct": 0, "wrong": 0}

print("\n" + "="*100)
print(f"{'PROMPT (Özet)':<40} | {'BEKLENEN':<12} | {'BASELINE':<12} | {'DURUM'}")
print("-" * 100)

for i, item in enumerate(dataset[:50]):
    prompt = item['prompt']
    truth = calculate_ground_truth(prompt) # Bizim gizli kuralımız
    
    response = generate_blind(prompt)
    model_cat = extract_category(response)
    
    # Baseline uyduracağı için "Yanlış" diyeceğiz
    is_correct = (model_cat == truth)
    
    stats["total"] += 1
    if is_correct: stats["correct"] += 1
    else: stats["wrong"] += 1
    
    icon = "✅" if is_correct else "❌"
    print(f"{prompt[:38]:<40} | {truth:<12} | {model_cat:<12} | {icon}")

print("="*100)
print(f"📉 BASELINE (KÖR) SKORU: %{stats['correct']/stats['total']*100:.1f}")
print("="*100)