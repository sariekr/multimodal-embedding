import os
import torch
import json
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# 1. AYARLAR
model_id = "OpenPipe/Qwen3-14B-Instruct"
output_dir = "qwen-rl-pure-lora-result"

# 2. ÖDÜL FONKSİYONU
def reward_function(completions, prompts, **kwargs):
    rewards = []
    
    billing_keywords = ["bill", "charge", "refund", "money", "price", "cost", "pay", "card"]
    technical_keywords = ["bug", "crash", "error", "login", "screen", "app", "broken", "slow"]
    shipping_keywords = ["package", "delivery", "track", "arrive", "ship", "lost", "where"]

    for prompt, completion in zip(prompts, completions):
        try:
            # TRL versiyonuna göre içerik alma yöntemi değişebilir, garantili yöntem:
            if isinstance(completion, list):
                response_text = completion[0]['content']
            elif hasattr(completion, 'content'):
                response_text = completion.content
            else:
                response_text = str(completion)
                
            prompt_text = str(prompt).lower()
        except:
            rewards.append(0.0)
            continue

        score = 0.0

        # CEZA: Sus
        if "<think>" in response_text or "</think>" in response_text:
            score -= 20.0
        
        # FORMAT
        clean_text = response_text.strip()
        if not clean_text.startswith("{"):
            score -= 5.0
        else:
            score += 2.0
        
        if "```" in clean_text: score -= 5.0

        # ZEKA
        try:
            data = json.loads(clean_text)
            category = data.get("category", "UNKNOWN")
            
            hit = False
            if any(k in prompt_text for k in billing_keywords):
                if category == "BILLING": score += 15.0; hit = True
                elif category == "OTHER": score -= 10.0
            
            elif any(k in prompt_text for k in technical_keywords):
                if category == "TECHNICAL": score += 15.0; hit = True
                elif category == "OTHER": score -= 10.0
            
            elif any(k in prompt_text for k in shipping_keywords):
                if category == "SHIPPING": score += 15.0; hit = True
                elif category == "OTHER": score -= 10.0
            
            if not hit and category == "OTHER": score += 15.0

        except:
            score -= 5.0 

        rewards.append(score)
    return rewards

# 3. MODELİ YÜKLE (Flash Attention ZORUNLULUĞU KALDIRILDI)
print(f"Model yükleniyor: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Not: attn_implementation parametresini kaldırdık. 
# PyTorch otomatik olarak en iyi yöntemi (SDPA) seçecek.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)

# 4. DATASET
if not os.path.exists("rl_dataset"):
    # Dataset yoksa o an oluştur (Güvenlik önlemi)
    print("Dataset bulunamadı, oluşturuluyor...")
    import json
    from datasets import Dataset
    with open("dataset.json", "r") as f: raw = json.load(f)
    system_prompt = "You are a strict data extraction engine.\nRULES:\n1. Output ONLY a JSON object.\n2. DO NOT use <think> tags.\n3. Allowed categories: [\"BILLING\", \"TECHNICAL\", \"SHIPPING\", \"PRODUCT\", \"OTHER\"]."
    formatted = []
    for item in raw:
        formatted.append({"prompt": [{"role": "system", "content": system_prompt}, {"role": "user", "content": item['prompt']}]})
    dataset = Dataset.from_list(formatted)
else:
    dataset = load_from_disk("rl_dataset")

# 5. LORA KONFIG
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
    bias="none"
)

# 6. EGITIM AYARLARI
training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8, 
    num_generations=4,             
    max_prompt_length=512,
    max_completion_length=300,
    num_train_epochs=1,            
    logging_steps=1,
    save_steps=50,
    report_to="none"
)

# 7. BASLAT
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_function,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
)

print("🚀 EĞİTİM BAŞLIYOR (Flash Attention devre dışı)...")
trainer.train()
trainer.save_model(output_dir)
print(f"✅ Bitti! Model şurada: {output_dir}")