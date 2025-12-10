import os
import torch
import json
import re
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# AYARLAR
model_id = "OpenPipe/Qwen3-14B-Instruct"
output_dir = "qwen-rl-fintech-v2" # v2 klasörüne kaydedelim

# AGRESİF ÖDÜL FONKSİYONU
def reward_function(completions, prompts, **kwargs):
    rewards = []
    
    for prompt, completion in zip(prompts, completions):
        try:
            if isinstance(completion, list): response_text = completion[0]['content']
            elif hasattr(completion, 'content'): response_text = completion.content
            else: response_text = str(completion)
            prompt_text = str(prompt)
        except: rewards.append(0.0); continue

        score = 0.0

        # Format Cezası (Aynı)
        if "<think>" in response_text: score -= 20.0
        if not response_text.strip().startswith("{"): score -= 5.0
        else: score += 1.0

        # Metrikleri Çıkar
        revenue = 0; burn_rate = 0; nps_score = -100; founder = ""
        rev_match = re.search(r'Annual Revenue: \$([\d,]+)', prompt_text)
        if rev_match: revenue = int(rev_match.group(1).replace(',', ''))
        burn_match = re.search(r'Annual Burn Rate: \$([\d,]+)', prompt_text)
        if burn_match: burn_rate = int(burn_match.group(1).replace(',', ''))
        nps_match = re.search(r'Customer NPS Score: (-?\d+)', prompt_text)
        if nps_match: nps_score = int(nps_match.group(1))
        if "Ex-Google" in prompt_text or "Ex-Facebook" in prompt_text: founder = "BigTech"

        # Hedef Belirle
        target_decision = "STANDARD_LOAN"
        if founder == "BigTech": target_decision = "MANUAL_REVIEW"
        elif revenue > 10_000_000 and burn_rate > (revenue * 0.8): target_decision = "REJECT_RISK"
        elif nps_score >= 80: target_decision = "A_PLUS_TIER"

        # --- PUANLAMA MOTORU (V2) ---
        try:
            data = json.loads(response_text.strip())
            model_decision = data.get("decision", "UNKNOWN")
            
            if model_decision == target_decision:
                score += 30.0 # Ödülü artırdık (25 -> 30)
            else:
                score -= 15.0 # Standart ceza
                
                # ÖZEL CEZALAR (Hataları düzeltmek için)
                
                # 1. Tembellik Cezası: Standart olması gerekirken Manual demiş.
                if target_decision == "STANDARD_LOAN" and model_decision == "MANUAL_REVIEW":
                    score -= 20.0 # Buna çok kızıyoruz!
                    
                # 2. İtaatsizlik Cezası: Torpilliye Red vermiş.
                if target_decision == "MANUAL_REVIEW" and model_decision == "REJECT_RISK":
                    score -= 30.0 # Patronun arkadaşını kovamazsın!
                    
        except:
            score -= 5.0

        rewards.append(score)
    return rewards

# Yükleme ve Konfigürasyon
print(f"Model yükleniyor: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

with open("dataset_fintech.json", "r") as f: raw_data = json.load(f)
system_prompt = """You are a credit risk engine for FinCorp.
Output JSON: {"decision": "..."}
Allowed Decisions: [A_PLUS_TIER, REJECT_RISK, MANUAL_REVIEW, STANDARD_LOAN]."""

formatted_data = [{"prompt": [{"role": "system", "content": system_prompt}, {"role": "user", "content": item['prompt']}]} for item in raw_data]
dataset = Dataset.from_list(formatted_data)

peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM", lora_dropout=0.05, bias="none")

# Daha uzun ve agresif eğitim
training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=2e-5,           # Öğrenme hızını hafif artırdık
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8, 
    num_generations=4,
    num_train_epochs=5,           # <--- 3'ten 5'e çıkardık!
    max_prompt_length=512,
    max_completion_length=200,
    gradient_checkpointing=True,
    logging_steps=1,
    save_strategy="no",
    report_to="none"
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_function,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)

print("🚀 FINTECH EĞİTİMİ V2 BAŞLIYOR (Cezalandırıcı Mod)...")
trainer.train()
trainer.save_model(output_dir)
print(f"✅ Bitti! {output_dir}")