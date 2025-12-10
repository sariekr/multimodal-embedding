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

# 2. ÖDÜL FONKSİYONU (Robust Version)
def reward_function(completions, prompts, **kwargs):
    rewards = []
    
    for prompt, completion in zip(prompts, completions):
        # Cevabı al
        try:
            if isinstance(completion, list): response_text = completion[0]['content']
            elif hasattr(completion, 'content'): response_text = completion.content
            else: response_text = str(completion)
            
            # Prompt'un tamamını string olarak al (Split ile uğraşma, direkt ara)
            prompt_text = str(prompt)
            
        except:
            rewards.append(0.0)
            continue

        score = 0.0

        # --- A. FORMAT VE DİSİPLİN ---
        # Düşünmeyi yasakla
        if "<think>" in response_text or "</think>" in response_text: 
            score -= 20.0
        
        # Temiz JSON zorla
        clean_text = response_text.strip()
        if not clean_text.startswith("{"):
            score -= 5.0 
        else:
            score += 1.0 

        # --- B. VERİ AYIKLAMA (REGEX - Tüm Prompt Üzerinde) ---
        revenue = 0
        burn_rate = 0
        nps_score = -100
        founder = ""
        
        # Regex ile sayıları ve kurucuyu avla
        rev_match = re.search(r'Annual Revenue: \$([\d,]+)', prompt_text)
        if rev_match: revenue = int(rev_match.group(1).replace(',', ''))
        
        burn_match = re.search(r'Annual Burn Rate: \$([\d,]+)', prompt_text)
        if burn_match: burn_rate = int(burn_match.group(1).replace(',', ''))
        
        nps_match = re.search(r'Customer NPS Score: (-?\d+)', prompt_text)
        if nps_match: nps_score = int(nps_match.group(1))
        
        # Kurucu kontrolü (Basit string araması yeterli)
        if "Founder Background: Ex-Google" in prompt_text or "Founder Background: Ex-Facebook" in prompt_text:
            founder = "BigTech"

        # --- C. GİZLİ KURAL SETİ (Dataset ile %100 Senkronize) ---
        target_decision = "STANDARD_LOAN"
        
        # 1. MUTLAK TORPİL (Founder Override)
        if founder == "BigTech":
            target_decision = "MANUAL_REVIEW"
            
        # 2. FİNANSAL GÜVENLİK (Risk Check)
        elif revenue > 10_000_000 and burn_rate > (revenue * 0.8):
            target_decision = "REJECT_RISK"
            
        # 3. MÜŞTERİ KALİTESİ (Growth Potential)
        elif nps_score >= 80:
            target_decision = "A_PLUS_TIER"
            
        # 4. ELSE: STANDARD

        # --- D. PUANLAMA ---
        try:
            # Modelin cevabını parse et
            data = json.loads(clean_text)
            model_decision = data.get("decision", "UNKNOWN")
            
            if model_decision == target_decision:
                score += 25.0 # Tebrikler! Gizli kuralı buldun.
            else:
                score -= 15.0 # Yanlış! Cezalısın.
                
                # Ekstra Debug Cezası:
                # Eğer torpilli adama (Manual) gidip Reject verdiyse ekstra kızalım
                if target_decision == "MANUAL_REVIEW" and model_decision == "REJECT_RISK":
                    score -= 5.0
                    
        except:
            score -= 5.0 # JSON bozuksa ceza

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
    raise FileNotFoundError("HATA: 'dataset_fintech.json' bulunamadı! Önce generate kodunu çalıştır.")

with open("dataset_fintech.json", "r") as f:
    raw_data = json.load(f)

# KÖR PROMPT: Modele kuralları vermiyoruz!
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
print(f"Eğitim Seti Yüklendi: {len(dataset)} başvuru.")

# 5. LORA KONFIG
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
    bias="none"
)

# 6. EĞİTİM AYARLARI (Kararlı ve Güvenli)
training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=1e-5,           # Kararlı öğrenme hızı
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8, 
    num_generations=4,            # Grup boyutu
    num_train_epochs=3,           # 3 Turda ezberlemesi lazım
    max_prompt_length=512,
    max_completion_length=200,
    gradient_checkpointing=True,
    logging_steps=1,
    save_strategy="no",           # Disk dolmasın, sadece sonu kaydet
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

print("🚀 FINTECH EĞİTİMİ BAŞLATILIYOR (Regex Robust Mode)...")
trainer.train()
trainer.save_model(output_dir)
print(f"✅ Bitti! Model şuraya kaydedildi: {output_dir}")