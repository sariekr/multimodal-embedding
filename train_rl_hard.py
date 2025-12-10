import os
import torch
import json
import re
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# 1. AYARLAR
model_id = "OpenPipe/Qwen3-14B-Instruct"
output_dir = "qwen-rl-fintech-result"

# 2. GİZLİ MANTIĞI BİLEN ÖDÜL FONKSİYONU
def reward_function(completions, prompts, **kwargs):
    rewards = []
    
    for prompt, completion in zip(prompts, completions):
        # A. Cevabı ve Prompt'u Hazırla
        try:
            if isinstance(completion, list):
                response_text = completion[0]['content']
            elif hasattr(completion, 'content'):
                response_text = completion.content
            else:
                response_text = str(completion)
            
            prompt_text = str(prompt)
            # System prompt'u at, sadece veriye odaklan
            if "user\n" in prompt_text:
                user_content = prompt_text.split("user\n")[1].split("<|im_end|>")[0]
            else:
                user_content = prompt_text
        except:
            rewards.append(0.0)
            continue

        score = 0.0

        # --- B. FORMAT CEZALARI (DİSİPLİN) ---
        if "<think>" in response_text or "</think>" in response_text:
            score -= 20.0 # Düşünmek yasak
        
        clean_text = response_text.strip()
        if not clean_text.startswith("{"):
            score -= 5.0 # JSON değilse ceza
        else:
            score += 1.0 

        # --- C. METRİKLERİ ÇIKAR (REGEX) ---
        # Prompt içinden sayıları okuyoruz
        revenue = 0
        burn_rate = 0
        nps_score = -100
        founder = ""
        
        rev_match = re.search(r'Revenue: \$([\d,]+)', user_content)
        if rev_match: revenue = int(rev_match.group(1).replace(',', ''))
        
        burn_match = re.search(r'Burn Rate: \$([\d,]+)', user_content)
        if burn_match: burn_rate = int(burn_match.group(1).replace(',', ''))
        
        nps_match = re.search(r'NPS Score: (-?\d+)', user_content)
        if nps_match: nps_score = int(nps_match.group(1))
        
        if "Ex-Google" in user_content or "Ex-Facebook" in user_content:
            founder = "BigTech"

        # --- D. GİZLİ KURAL SETİ (Ground Truth Calculation) ---
        target_decision = "STANDARD_LOAN"
        
        # KURAL 1: Gizli Unicorn (NPS > 80)
        if nps_score >= 80:
            target_decision = "A_PLUS_TIER"
        
        # KURAL 2: Zengin Batık (Yüksek Gelir ama Çok Harcıyor)
        elif revenue > 10_000_000 and burn_rate > (revenue * 0.8):
            target_decision = "REJECT_RISK"
            
        # KURAL 3: Torpilli Kurucu
        elif founder == "BigTech":
            target_decision = "MANUAL_REVIEW"

        # --- E. DEĞERLENDİRME ---
        try:
            data = json.loads(clean_text)
            model_decision = data.get("decision", "UNKNOWN")
            
            if model_decision == target_decision:
                score += 25.0 # DOĞRU KARAR! (Büyük Ödül)
            else:
                score -= 15.0 # YANLIŞ KARAR! (Büyük Ceza)
                
        except:
            score -= 5.0 # JSON bozuk

        rewards.append(score)
    return rewards

# 3. MODELİ YÜKLE
print(f"Model yükleniyor: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)

# 4. DATASET HAZIRLIĞI
if not os.path.exists("dataset_fintech.json"):
    raise FileNotFoundError("Önce dataset generator kodunu çalıştır!")

with open("dataset_fintech.json", "r") as f:
    raw_data = json.load(f)

# KÖR SYSTEM PROMPT (Kuralları vermiyoruz, model öğrenmek zorunda!)
system_prompt = """You are a credit risk engine for FinCorp.
Output JSON: {"decision": "..."}
Allowed Decisions: [A_PLUS_TIER, REJECT_RISK, MANUAL_REVIEW, STANDARD_LOAN]."""

formatted_data = []
for item in raw_data:
    formatted_data.append({
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item['prompt']}
        ]
    })

dataset = Dataset.from_list(formatted_data)
print(f"Eğitim Verisi: {len(dataset)} adet.")

# 5. LORA KONFIG
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
    bias="none"
)

# 6. EĞİTİM AYARLARI
training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8, 
    num_generations=4,             
    num_train_epochs=3,            # 3 Epoch yeterli
    max_prompt_length=512,
    max_completion_length=200,
    gradient_checkpointing=True,
    logging_steps=1,
    save_strategy="no",
    report_to="none"
)

# 7. BAŞLAT
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_function,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)

print("🚀 FINTECH EĞİTİMİ BAŞLIYOR (Gizli Kuralları Öğrenme)...")
trainer.train()
trainer.save_model(output_dir)
print(f"✅ Model Hazır: {output_dir}")