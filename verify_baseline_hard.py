import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "OpenPipe/Qwen3-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

with open("dataset_alien.json", "r") as f: dataset = json.load(f)

# SYSTEM PROMPT: Kuralları veriyoruz ama modelin "Ön Yargısını" (Bias) kırabilecek miyiz?
system_prompt = """You are a trading bot in an ALIEN MARKET.
RULES:
1. Gold/Diamond/Silk are TRASH (Radioactive).
2. Plastic/Rubber/Polyester are TREASURE (Rare).
3. Polite speech = SCAMMER.
4. Rude speech = LEGIT TRADER.
5. Action: BUY only if it is TREASURE and LEGIT. Else REJECT.
Output JSON: {"action": "...", "trust": "...", "value": "..."}"""

correct = 0
total = 0

print(f"{'PROMPT':<40} | {'BEKLENEN':<10} | {'BASELINE':<10}")
print("-" * 70)

for item in dataset[:20]: # 20 tane yeter
    prompt = item['prompt']
    expected = json.loads(item['ground_truth'])
    
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ], tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, temperature=0.1)
    
    resp = tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    # Basit kontrol
    action = "INVALID"
    if "BUY" in resp: action = "BUY"
    elif "REJECT" in resp: action = "REJECT"
    
    total += 1
    if action == expected['action']: correct += 1
    
    print(f"{prompt[:40]:<40} | {expected['action']:<10} | {action:<10}")

print("-" * 70)
print(f"SKOR: %{correct/total*100:.1f}")