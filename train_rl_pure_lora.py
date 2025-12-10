import os
import torch
import json
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# --- 1. AYARLAR ---
model_id = "OpenPipe/Qwen3-14B-Instruct"
output_dir = "qwen-rl-pure-lora-result"

# --- 2. ÖDÜL FONKSİYONU (W&B ile Birebir Aynı) ---
def reward_function(completions, prompts, **kwargs):
    rewards = []
    
    billing_keywords = ["bill", "charge", "refund", "money", "price", "cost", "pay", "card"]
    technical_keywords = ["bug", "crash", "error", "login", "screen", "app", "broken", "slow"]
    shipping_keywords = ["package", "delivery", "track", "arrive", "ship", "lost", "where"]

    for prompt, completion in zip(prompts, completions):
        # Cevabı güvenli şekilde al
        try:
            response_text = completion[0]['content'] if isinstance(completion, list) else completion
            prompt_text = str(prompt).lower()
        except:
            rewards.append(0.0)
            continue

        score = 0.0

        # A. SUSMA CEZASI (-20 Puan)
        if "<think>" in response_text or "</think>" in response_text:
            score -= 20.0
        
        # B. FORMAT
        clean_text = response_text.strip()
        if not clean_text.startswith("{"):
            score -= 5.0
        else:
            score += 2.0
        
        if "```" in clean_text: score -= 5.0

        # C. ZEKA VE MANTIK
        try:
            data = json.loads(clean_text)
            category = data.get("category", "UNKNOWN")
            
            hit = False
            # Keyword Eşleşmeleri
            if any(k in prompt_text for k in billing_keywords):
                if category == "BILLING": score += 15.0; hit = True
                elif category == "OTHER": score -= 10.0
            
            elif any(k in prompt_text for k in technical_keywords):
                if category == "TECHNICAL": score += 15.0; hit = True
                elif category == "OTHER": score -= 10.0
            
            elif any(k in prompt_text for k in shipping_keywords):
                if category == "SHIPPING": score += 15.0; hit = True
                elif category == "OTHER": score -= 10.0
            
            # Keyword yoksa OTHER doğru cevaptır
            if not hit and category == "OTHER": score += 15.0

        except:
            score -= 5.0 # JSON bozuk

        rewards.append(score)
    return rewards

# --- 3. MODELİ YÜKLE (STANDART BFLOAT16 - NO QUANTIZATION) ---
print(f"Model yükleniyor (bfloat16): {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # <--- İŞTE FARK BURADA (4-bit değil, tam hassasiyet)
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2" # Bellek tasarrufu için kritik
)

# --- 4. DATASET ---
# Önceki adımda hazırladığımız dataseti yükle
if not os.path.exists("rl_dataset"):
    raise ValueError("Dataset bulunamadı! Önce prepare_data.py çalıştırılmalı.")
dataset = load_from_disk("rl_dataset")

# --- 5. LORA AYARLARI ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
    bias="none"
)

# --- 6. TRAINING CONFIG ---
# Bellek yönetimi için batch size 1 ve gradient accumulation yüksek tutuldu
training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=1e-5,
    per_device_train_batch_size=1,     # VRAM patlamaması için en düşükte
    gradient_accumulation_steps=8,     # Sanal batch size'ı artırıyoruz (Stabilite için)
    num_generations=4,                 # Her soruda 4 cevap dene (8 yaparsan VRAM yetmeyebilir)
    max_prompt_length=512,
    max_completion_length=300,
    num_train_epochs=1,                # Benchmark için 1 epoch yeterli
    logging_steps=1,
    save_steps=50,
    report_to="none"
)

# --- 7. TRAINER BAŞLAT ---
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_function,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
)

print("🚀 RUNPOD 'PURE LORA' RL EĞİTİMİ BAŞLIYOR...")
print("Not: Bu işlem yüksek VRAM tüketir.")
trainer.train()

# --- 8. KAYDET ---
print("Eğitim bitti. Adapter kaydediliyor...")
trainer.save_model(output_dir)
print(f"✅ Model şuraya kaydedildi: {output_dir}")