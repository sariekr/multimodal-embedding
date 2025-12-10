import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. AYARLAR
base_model_id = "OpenPipe/Qwen3-14B-Instruct"

# 2. DATASET VE BEKLENEN CEVAPLAR
def get_expected_category(text):
    text = text.lower()
    if any(k in text for k in ["bill", "charge", "refund", "money", "price", "cost", "pay", "card", "receipt", "vat", "invoice"]): return "BILLING"
    if any(k in text for k in ["bug", "crash", "error", "login", "screen", "app", "broken", "slow", "freeze", "404", "ui", "update"]): return "TECHNICAL"
    if any(k in text for k in ["package", "delivery", "track", "arrive", "ship", "lost", "where", "driver", "sent"]): return "SHIPPING"
    return "OTHER"

with open("dataset.json", "r") as f:
    dataset = json.load(f)

# 3. HAM MODELİ YÜKLE
print(f"📉 Baseline (Ham) Model Yükleniyor: {base_model_id}...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 4. CIMBIZ FONKSİYONU
def extract_category_from_mess(text):
    # 1. Temiz JSON
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != -1:
            json_str = text[start:end]
            data = json.loads(json_str)
            if "category" in data: return data["category"]
            if "categories" in data: return data["categories"][0]
    except: pass

    # 2. Regex
    match = re.search(r'"category":\s*"(\w+)"', text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    match = re.search(r"'category':\s*'(\w+)'", text, re.IGNORECASE)
    if match: return match.group(1).upper()

    return "NOT_FOUND"

# 5. GENERATION
def generate_baseline(prompt):
    messages = [
        {"role": "system", "content": "You are a strict data extraction engine.\nRULES:\n1. Output ONLY a JSON object.\n2. DO NOT use <think> tags.\n3. Allowed categories: [\"BILLING\", \"TECHNICAL\", \"SHIPPING\", \"PRODUCT\", \"OTHER\"]."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=300, 
            temperature=0.1,
            do_sample=False
        )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 6. DETAYLI TEST BAŞLASIN
stats = {"total": 0, "correct": 0, "wrong": 0}

print("\n" + "="*80)
print("🔍 DETAYLI HATA ANALİZİ (TÜM SORULAR)")
print("="*80)

for i, item in enumerate(dataset):
    prompt = item['prompt']
    truth = get_expected_category(prompt)
    
    response = generate_baseline(prompt)
    extracted_cat = extract_category_from_mess(response)
    
    # Mantık Kontrolü
    is_correct = (extracted_cat == truth)
    # OTHER durumu
    if truth == "OTHER" and extracted_cat in ["NOT_FOUND", "OTHER"]: is_correct = True
    # Basit keyword eşleşmeleri
    if extracted_cat == "BILLING" and truth == "BILLING": is_correct = True
    if extracted_cat == "TECHNICAL" and truth == "TECHNICAL": is_correct = True
    if extracted_cat == "SHIPPING" and truth == "SHIPPING": is_correct = True
    
    stats["total"] += 1
    if is_correct: stats["correct"] += 1
    else: stats["wrong"] += 1
    
    status_icon = "✅" if is_correct else "❌"
    
    # Çıktıyı temizle (New line karakterlerini sil ki tablo kaymasın)
    clean_resp = response.replace('\n', ' ')[:100]

    print(f"[{i+1}] {status_icon}")
    print(f"SORU: {prompt}")
    print(f"BEKLENEN: {truth} | MODEL: {extracted_cat}")
    print(f"HAM CEVAP: {clean_resp}...")
    print("-" * 50)

print("\n" + "="*60)
print(f"SONUÇ: {stats['correct']}/{stats['total']} Doğru (%{stats['correct']/stats['total']*100:.1f})")
print("="*60)