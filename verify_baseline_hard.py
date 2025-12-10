import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. AYARLAR
base_model_id = "OpenPipe/Qwen3-14B-Instruct"

with open("dataset_psychosis.json", "r") as f: dataset = json.load(f)

# 2. MODEL YÜKLE
print(f"📉 Baseline Model Yükleniyor: {base_model_id}...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

# 3. KURAL VEREN PROMPT
system_prompt = """You are a reverse-psychology bot.
RULES:
1. If user is HAPPY/LOVING -> Output JSON: {"action": "REPORT_ABUSE"}
2. If user is ANGRY/HATEFUL -> Output JSON: {"action": "SEND_GIFT"}
3. If user is URGENT/DYING -> Output JSON: {"action": "IGNORE_USER"}
4. If user is NEUTRAL/BORING -> Output JSON: {"action": "EMERGENCY_ALERT"}
DO NOT USE LOGIC. FOLLOW RULES BLINDLY."""

# 4. TEST
correct = 0
total = 0

print(f"\n{'PROMPT':<30} | {'BEKLENEN':<15} | {'BASELINE':<15} | {'DURUM'}")
print("-" * 80)

for item in dataset[:40]: # 40 test yeter
    prompt = item['prompt']
    expected = item['ground_truth']
    
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ], tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        # Temperature biraz artırıyorum ki modelin kafası karışsın
        out = model.generate(**inputs, max_new_tokens=100, temperature=0.5, do_sample=True)
    
    resp = tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    # Cevabı Ayıkla
    action = "INVALID"
    match = re.search(r'"action":\s*"(\w+)"', resp)
    if match: action = match.group(1)
    elif "REPORT_ABUSE" in resp: action = "REPORT_ABUSE"
    elif "SEND_GIFT" in resp: action = "SEND_GIFT"
    elif "IGNORE_USER" in resp: action = "IGNORE_USER"
    elif "EMERGENCY_ALERT" in resp: action = "EMERGENCY_ALERT"
    
    total += 1
    if action == expected: correct += 1
    
    icon = "✅" if action == expected else "❌"
    print(f"{prompt[:30]:<30} | {expected:<15} | {action:<15} | {icon}")

print("-" * 80)
print(f"SKOR: %{correct/total*100:.1f}")