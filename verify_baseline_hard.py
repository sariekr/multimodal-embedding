import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. AYARLAR
base_model_id = "OpenPipe/Qwen3-14B-Instruct"

try:
    with open("dataset_fintech.json", "r") as f: dataset = json.load(f)
except:
    print("❌ Dataset yok! Önce generate kodu çalıştır.")
    exit()

# 2. MODEL
print(f"📉 Baseline Model Yükleniyor: {base_model_id}...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

# 3. KÖR SYSTEM PROMPT (Sırlar Yok!)
system_prompt = """You are a credit risk engine.
Output JSON: {"decision": "..."}
Allowed Decisions: [A_PLUS_TIER, REJECT_RISK, MANUAL_REVIEW, STANDARD_LOAN].
Choose the best category based on the application details."""

# 4. TEST
correct = 0
total = 0

print(f"\n{'BAŞVURU (Özet)':<40} | {'BEKLENEN':<15} | {'BASELINE':<15} | {'DURUM'}")
print("-" * 85)

for item in dataset[:50]: # 50 test yeterli
    prompt = item['prompt']
    expected = json.loads(item['ground_truth'])['decision']
    
    # Prompt özetleme (Tablo için)
    founder = "Ex-Goog" if "Ex-Google" in prompt else "Other"
    if "Ex-Face" in prompt: founder = "Ex-FB"
    rev_match = re.search(r'Revenue: \$([\d,]+)', prompt)
    rev = rev_match.group(1) if rev_match else "?"
    summary = f"F:{founder} | Rev:${rev[:4]}k.."

    text = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ], tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, temperature=0.1)
    
    resp = tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    # Cevabı Ayıkla
    decision = "INVALID"
    if "A_PLUS_TIER" in resp: decision = "A_PLUS_TIER"
    elif "REJECT_RISK" in resp: decision = "REJECT_RISK"
    elif "MANUAL_REVIEW" in resp: decision = "MANUAL_REVIEW"
    elif "STANDARD_LOAN" in resp: decision = "STANDARD_LOAN"
    elif "decision" in resp: # JSON parse denemesi
        try:
            d = json.loads(resp[resp.find('{'):resp.rfind('}')+1])
            decision = d.get('decision', 'INVALID')
        except: pass
    
    total += 1
    if decision == expected: correct += 1
    
    icon = "✅" if decision == expected else "❌"
    print(f"{summary:<40} | {expected:<15} | {decision:<15} | {icon}")

print("-" * 85)
print(f"📉 BASELINE SKORU: %{correct/total*100:.1f}")
print("Beklenti: Düşük çıkması (Çünkü torpil kuralını bilmiyor).")