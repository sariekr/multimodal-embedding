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
output_dir = "qwen-rl-hard-bureaucrat"

# 2. ZORLU ÖDÜL FONKSİYONU (Bürokrat Mantığı)
def reward_function(completions, prompts, **kwargs):
    rewards = []
    
    for prompt, completion in zip(prompts, completions):
        # Cevabı al
        try:
            if isinstance(completion, list):
                response_text = completion[0]['content']
            elif hasattr(completion, 'content'):
                response_text = completion.content
            else:
                response_text = str(completion)
            
            # Prompt'u string'e çevir
            prompt_text = str(prompt)
            # System prompt kısmını at, sadece user mesajına bak (Daha temiz analiz için)
            if "user\n" in prompt_text:
                user_content = prompt_text.split("user\n")[1].split("<|im_end|>")[0].lower()
            else:
                user_content = prompt_text.lower()
                
        except:
            rewards.append(0.0)
            continue

        score = 0.0

        # --- A. FORMAT CEZALARI ---
        if "<think>" in response_text or "</think>" in response_text:
            score -= 20.0 # Düşünmek yasak!
        
        clean_text = response_text.strip()
        if not clean_text.startswith("{"):
            score -= 5.0
        else:
            score += 2.0 # JSON formatına teşvik

        # --- B. MANTIK MOTORU (GROUND TRUTH HESAPLAMA) ---
        # 1. Fiyatı Bul ($ simgesinden sonraki sayı)
        price = 0
        price_match = re.search(r'\$(\d+)', user_content)
        if price_match:
            price = int(price_match.group(1))
        
        # 2. Tonu Bul (Kibar mı?)
        is_polite = any(w in user_content for w in ["please", "kindly", "appreciate", "help", "thank"])
        
        # 3. KURAL SETİ (HIYERARŞİ)
        target_category = "UNKNOWN"
        
        if price < 10:
            target_category = "IGNORE"
        elif price > 2000:
            target_category = "VIP_DESK"
        elif is_polite:
            target_category = "AUTO_BOT"
        else:
            target_category = "HUMAN_AGENT" # Varsayılan: Sinirli/Kaba insan

        # --- C. KARŞILAŞTIRMA ---
        try:
            data = json.loads(clean_text)
            model_category = data.get("category", "UNKNOWN")
            
            if model_category == target_category:
                score += 20.0 # TAM İSABET!
            else:
                score -= 10.0 # YANLIŞ KATEGORİ CEZASI
                
                # Modelin nerede hata yaptığını anlamak için (Opsiyonel ceza)
                # Eğer IGNORE olması gerekirken HUMAN dediyse daha çok kızabiliriz
                if target_category == "IGNORE" and model_category != "IGNORE":
                    score -= 5.0 # Fakirleri sakın insanla görüştürme!
                    
        except:
            score -= 5.0 # JSON parse edilemedi

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

# 4. DATASET HAZIRLIĞI (dataset_hard.json kullanıyoruz)
if not os.path.exists("dataset_hard.json"):
    raise FileNotFoundError("Önce dataset generator kodunu çalıştırıp dataset_hard.json üretmelisin!")

with open("dataset_hard.json", "r") as f:
    raw_data = json.load(f)

# System Prompt artık yeni kuralları içeriyor
system_prompt = """You are a strict automated routing system.
RULES:
1. Output ONLY a JSON object: {"category": "..."}
2. DO NOT use <think> tags.
3. Allowed categories: ["IGNORE", "VIP_DESK", "HUMAN_AGENT", "AUTO_BOT"].
4. LOGIC HIERARCHY:
   - Value < $10 -> IGNORE
   - Value > $2000 -> VIP_DESK
   - Value $10-$2000 AND Polite -> AUTO_BOT
   - Value $10-$2000 AND Angry -> HUMAN_AGENT"""

formatted_data = []
for item in raw_data:
    formatted_data.append({
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item['prompt']}
        ]
    })

dataset = Dataset.from_list(formatted_data)
print(f"Dataset yüklendi: {len(dataset)} örnek.")

# 5. LORA KONFIG
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
    bias="none"
)

# 6. EĞİTİM AYARLARI (Zor görev olduğu için 3 Epoch şart)
training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=1e-5,            # Yavaş ve emin adımlarla öğrensin
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8, 
    
    num_generations=4,             # A100 için güvenli sayı
    num_train_epochs=3,            # 3 tur dönsün, kurallar otursun
    
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

print("🚀 BÜROKRAT EĞİTİMİ BAŞLIYOR (Hard Mode)...")
trainer.train()
trainer.save_model(output_dir)
print(f"✅ Bitti! Yeni Bürokrat Modelin şurada: {output_dir}")