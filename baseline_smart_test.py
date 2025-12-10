import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. AYARLAR
base_model_id = "OpenPipe/Qwen3-14B-Instruct"

# 2. DOĞRU CEVAP ANAHTARI (RL MODELİ İLE AYNI - ADİL KARŞILAŞTIRMA)
CORRECT_LABELS = {
    "charged twice": "BILLING",
    "Where is my package": "SHIPPING",
    "reset my password": "TECHNICAL",
    "credit card shows a charge": "BILLING",
    "Order #12345 hasn't arrived": "SHIPPING",
    "returned the shoes": "BILLING",
    "App keeps freezing": "TECHNICAL",
    "wrong size": "PRODUCT",
    "account locked": "TECHNICAL",
    "promo code": "BILLING",
    "cancel button is greyed": "TECHNICAL",
    "Package arrived completely crushed": "SHIPPING",
    "waiting on hold": "OTHER",
    "Subscription renewed": "BILLING",
    "Can't update my shipping address": "TECHNICAL",
    "tracking number is invalid": "SHIPPING",
    "charged for the premium": "BILLING",
    "screen goes black": "TECHNICAL",
    "price on the packing slip": "SHIPPING",
    "refund for order": "BILLING",
    "Login page gives me a 404": "TECHNICAL",
    "ordered two of the same": "SHIPPING",
    "left the package in the rain": "SHIPPING",
    "access to my contacts": "TECHNICAL",
    "promised a 10% discount": "BILLING",
    "marketing emails": "OTHER",
    "stuck in 'processing'": "SHIPPING",
    "Payment failed": "BILLING",
    "can't find the logout": "TECHNICAL",
    "shipped order #888 to my old": "SHIPPING",
    "Item is missing parts": "PRODUCT",
    "Chat support bot is useless": "OTHER",
    "website is so slow": "TECHNICAL",
    "return label link gives an error": "TECHNICAL",
    "Charged $100": "BILLING",
    "color of the shirt": "PRODUCT",
    "App crashes": "TECHNICAL",
    "Tracking says 'out for delivery'": "SHIPPING",
    "VAT invoice": "BILLING",
    "refund order #222": "BILLING",
    "Search function is broken": "TECHNICAL",
    "profile picture won't upload": "TECHNICAL",
    "refund my shipping cost": "BILLING",
    "description said wireless": "PRODUCT",
    "Cannot add new credit card": "BILLING",
    "Who signed for my package": "SHIPPING",
    "overcharge alert": "BILLING",
    "Delete my account": "TECHNICAL",
    "audio quality": "PRODUCT",
    "Why can't I use PayPal": "BILLING"
}

def get_ground_truth(prompt):
    for key, label in CORRECT_LABELS.items():
        if key in prompt:
            return label
    return "OTHER"

with open("dataset.json", "r") as f:
    dataset = json.load(f)

# 3. HAM MODELİ YÜKLE
print("📉 Baseline (Ham) Model Yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 4. CIMBIZ (REGEX) - Baseline formatı bozuk olduğu için buna mecburuz
def extract_category_from_mess(text):
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != -1:
            json_str = text[start:end]
            data = json.loads(json_str)
            if "category" in data: return data["category"]
            if "categories" in data: return data["categories"][0]
    except: pass

    match = re.search(r'"category":\s*"(\w+)"', text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    match = re.search(r"'category':\s*'(\w+)'", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    return "NOT_FOUND"

# 5. CEVAP ÜRET
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

# 6. KARŞILAŞTIRMA VE RAPOR
stats = {"total": 0, "correct": 0, "wrong": 0}

print("\n" + "="*100)
print(f"{'SORU (Kısmi)':<40} | {'BEKLENEN':<10} | {'BASELINE':<10} | {'DURUM'}")
print("-" * 100)

for i, item in enumerate(dataset):
    prompt = item['prompt']
    truth = get_ground_truth(prompt)
    
    response = generate_baseline(prompt)
    model_cat = extract_category_from_mess(response) # Cımbızla çek
    
    # AYNI ESNEK KURALLAR (Adil olmak için)
    is_correct = (model_cat == truth)
    if truth == "OTHER" and model_cat in ["PRODUCT", "TECHNICAL"]: is_correct = True
    if truth == "BILLING" and "promo code" in prompt and model_cat == "TECHNICAL": is_correct = True
    
    # Baseline için ek kıyak: Hallucination yaptıysa (Listede olmayan bir şey uydurduysa)
    # ve bu uydurduğu şey doğruysa (örn: ACCOUNT -> TECHNICAL) kabul edelim mi? 
    # Hayır, etmeyelim. Çünkü RL modeli standart dışına çıkmıyor.
    
    stats["total"] += 1
    if is_correct: stats["correct"] += 1
    else: stats["wrong"] += 1
    
    icon = "✅" if is_correct else "❌"
    print(f"{prompt[:40]:<40} | {truth:<10} | {model_cat:<10} | {icon}")

print("="*100)
print(f"📉 BASELINE 'ESNEK' ZEKA SKORU: %{stats['correct']/stats['total']*100:.1f} ({stats['correct']}/{stats['total']})")
print("="*100)