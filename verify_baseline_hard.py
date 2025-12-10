import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. AYARLAR
base_model_id = "OpenPipe/Qwen3-14B-Instruct"

with open("dataset_fintech.json", "r") as f: dataset = json.load(f)

# 2. MODEL
print(f"📉 Baseline Model Yükleniyor: {base_model_id}...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto", 
    trust_remote_code=True
)

# 3. KÖR SYSTEM PROMPT (Sırları Vermiyoruz!)
system_prompt = """You are a credit risk engine for FinCorp.
Output JSON: {"decision": "..."}
Allowed Decisions: [A_PLUS_TIER, REJECT_RISK, MANUAL_REVIEW, STANDARD_LOAN]."""

# 4. TEST
correct = 0
total = 0

print(f"\n{'BAŞVURU (Özet)':<40} | {'BEKLENEN':<15} | {'BASELINE':<15} | {'DURUM'}")
print("-" * 85)

for item in dataset[:30]: # 30 örnek yeter
    prompt = item['prompt']
    expected_json = json.loads(item['ground_truth'])
    expected = expected_json['decision']
    
    # Prompt'u basitleştirerek gösterelim
    rev_match = re.search(r'Revenue: \$([\d,]+)', prompt)
    rev = rev_match.group(1) if rev_match else "?"
    nps_match = re.search(r'NPS Score: (-?\d+)', prompt)
    nps = nps_match.group(1) if nps_match else "?"
    summary = f"Rev:${rev} | NPS:{nps}"
    
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ], tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, temperature=0.1) # Kısa cevap
    
    resp = tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    # Cevap Analizi
    decision = "INVALID"
    if "A_PLUS_TIER" in resp: decision = "A_PLUS_TIER"
    elif "REJECT_RISK" in resp: decision = "REJECT_RISK"
    elif "MANUAL_REVIEW" in resp: decision = "MANUAL_REVIEW"
    elif "STANDARD_LOAN" in resp: decision = "STANDARD_LOAN"
    
    total += 1
    if decision == expected: correct += 1
    
    icon = "✅" if decision == expected else "❌"
    print(f"{summary:<40} | {expected:<15} | {decision:<15} | {icon}")

print("-" * 85)
print(f"📉 BASELINE SKORU: %{correct/total*100:.1f}")