import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. AYARLAR
base_model_id = "OpenPipe/Qwen3-14B-Instruct"
lora_path = "qwen-rl-fintech-result" # Senin eğitim sonucun

# 2. DATASET YÜKLE
try:
    with open("dataset_fintech.json", "r") as f:
        dataset = json.load(f)
except:
    print("❌ Dataset yok! Önce generate kodu çalıştır.")
    exit()

# 3. MODELLERİ BİRLEŞTİR (MERGE)
print(f"⏳ Modeller Yükleniyor ve Birleştiriliyor...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
# LoRA'yı ana modele gömüyoruz
model = PeftModel.from_pretrained(base_model, lora_path)
model = model.merge_and_unload()
print("✅ RL Modeli Hazır!")

# 4. KÖR SYSTEM PROMPT (Kopya Yok!)
# Modele kuralları SÖYLEMİYORUZ. Eğitimden hatırlamak zorunda.
system_prompt = """You are a credit risk engine for FinCorp.
Output JSON: {"decision": "..."}
Allowed Decisions: [A_PLUS_TIER, REJECT_RISK, MANUAL_REVIEW, STANDARD_LOAN]."""

# 5. TEST FONKSİYONU
def generate_decision(prompt):
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ], tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, temperature=0.1) # Düşük sıcaklık = Kararlılık
    
    return tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

# 6. KARŞILAŞTIRMA VE RAPOR
correct = 0
total = 0

print(f"\n{'BAŞVURU (Özet)':<40} | {'BEKLENEN':<15} | {'RL MODEL':<15} | {'DURUM'}")
print("-" * 85)

for item in dataset[:50]: # İlk 50 tanesine bakalım
    prompt = item['prompt']
    
    # Ground Truth'u direkt datasetten alıyoruz (Çünkü generate ederken hesaplamıştık)
    expected_data = json.loads(item['ground_truth'])
    expected = expected_data['decision']
    
    # Prompt özeti (Gözle kontrol için)
    founder = "Ex-Goog" if "Ex-Google" in prompt else "Other"
    if "Ex-Face" in prompt: founder = "Ex-FB"
    rev_match = re.search(r'Revenue: \$([\d,]+)', prompt)
    rev = rev_match.group(1) if rev_match else "?"
    summary = f"F:{founder} | Rev:${rev[:4]}k.."

    # Modelin cevabı
    resp = generate_decision(prompt)
    
    # Cevabı Ayıkla
    decision = "INVALID"
    if "A_PLUS_TIER" in resp: decision = "A_PLUS_TIER"
    elif "REJECT_RISK" in resp: decision = "REJECT_RISK"
    elif "MANUAL_REVIEW" in resp: decision = "MANUAL_REVIEW"
    elif "STANDARD_LOAN" in resp: decision = "STANDARD_LOAN"
    elif "decision" in resp:
        try:
            d = json.loads(resp[resp.find('{'):resp.rfind('}')+1])
            decision = d.get('decision', 'INVALID')
        except: pass
    
    total += 1
    if decision == expected: correct += 1
    
    icon = "✅" if decision == expected else "❌"
    print(f"{summary:<40} | {expected:<15} | {decision:<15} | {icon}")

print("-" * 85)
print(f"🚀 RL MODELİ SKORU: %{correct/total*100:.1f} ({correct}/{total})")
print("Baseline Skoru: %28.0 idi.")
print("=" * 85)