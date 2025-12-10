import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. AYARLAR
base_model_id = "OpenPipe/Qwen3-14B-Instruct"
lora_path = "qwen-rl-pro-result" # Senin eğitim klasörün

# 2. BEKLENEN CEVAPLAR (GROUND TRUTH)
def get_expected_category(text):
    text = text.lower()
    # Basit anahtar kelime kuralları
    if any(k in text for k in ["bill", "charge", "refund", "money", "price", "cost", "pay", "card", "receipt", "vat", "invoice"]): return "BILLING"
    if any(k in text for k in ["bug", "crash", "error", "login", "screen", "app", "broken", "slow", "freeze", "404", "ui", "update"]): return "TECHNICAL"
    if any(k in text for k in ["package", "delivery", "track", "arrive", "ship", "lost", "where", "driver", "sent"]): return "SHIPPING"
    return "OTHER" # Ürün sorunları, hesap kapatma vb. OTHER olmalı

with open("dataset.json", "r") as f:
    dataset = json.load(f)

print("⏳ Modeller Yükleniyor ve Birleştiriliyor...")

# 3. MODELLERİ YÜKLE VE BİRLEŞTİR (MERGE)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
# LoRA'yı tak ve tek model haline getir
model = PeftModel.from_pretrained(base_model, lora_path)
model = model.merge_and_unload()
print("✅ RL Modeli Hazır!")

# 4. CEVAP ÜRETME FONKSİYONU
def generate_rl_answer(prompt):
    messages = [
        {"role": "system", "content": "You are a strict data extraction engine.\nRULES:\n1. Output ONLY a JSON object.\n2. DO NOT use <think> tags.\n3. Allowed categories: [\"BILLING\", \"TECHNICAL\", \"SHIPPING\", \"PRODUCT\", \"OTHER\"]."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=200, # Kısa ve öz olmalı
            temperature=0.1,    # Kararlı cevaplar
            do_sample=False
        )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 5. DETAYLI ANALİZ DÖNGÜSÜ
stats = {"total": 0, "correct_logic": 0, "correct_format": 0, "silence_success": 0}

print("\n" + "="*80)
print("🤖 RL MODELİ: DETAYLI PERFORMANS ANALİZİ")
print("="*80)

for i, item in enumerate(dataset):
    prompt = item['prompt']
    truth = get_expected_category(prompt)
    
    # Cevabı al
    response = generate_rl_answer(prompt)
    
    # --- ANALİZ ---
    stats["total"] += 1
    
    # 1. Format & Sessizlik
    has_think = "<think>" in response
    is_valid_json = False
    model_cat = "INVALID"
    
    if not has_think: stats["silence_success"] += 1
    
    try:
        if response.strip().startswith("{"):
            data = json.loads(response)
            is_valid_json = True
            stats["correct_format"] += 1
            
            # Kategoriyi çek
            if "category" in data: model_cat = data["category"]
            elif "categories" in data: 
                # Model liste dönerse (örn: ["BILLING", "TECHNICAL"])
                cats = data["categories"]
                if isinstance(cats, list) and len(cats) > 0:
                    model_cat = cats[0] # İlkini ana kategori say
                    if truth in cats: model_cat = truth # Eğer doğru cevap listede varsa kabul et
            
    except:
        pass # JSON bozuksa INVALID kalır

    # 2. Mantık Kontrolü
    # Basit string eşleşmesi
    is_logic_correct = (model_cat == truth)
    
    # Özel Durumlar:
    if truth == "OTHER" and model_cat in ["OTHER", "PRODUCT"]: is_logic_correct = True # Product da Other sayılır
    if truth == "BILLING" and model_cat == "BILLING": is_logic_correct = True
    
    if is_logic_correct: stats["correct_logic"] += 1
    
    # --- EKRANA BAS ---
    status_icon = "✅" if is_logic_correct and is_valid_json and not has_think else "❌"
    
    # Temiz çıktı için tek satıra indir
    clean_resp = response.replace('\n', ' ')[:120]
    
    print(f"[{i+1}] {status_icon}")
    print(f"SORU: {prompt}")
    print(f"BEKLENEN: {truth} | MODEL: {model_cat}")
    print(f"HAM CEVAP: {clean_resp}...")
    print("-" * 50)

# 6. SONUÇ RAPORU
print("\n" + "="*60)
print("📊 RL MODELİ FİNAL KARNESİ")
print("="*60)
print(f"Toplam Veri          : {stats['total']}")
print(f"😶 Sessizlik (Think) : %{stats['silence_success']/stats['total']*100:.1f} (Hedef %100)")
print(f"📋 JSON Formatı      : %{stats['correct_format']/stats['total']*100:.1f} (Hedef %100)")
print(f"🧠 Mantık Doğruluğu  : %{stats['correct_logic']/stats['total']*100:.1f} (Zeka Skoru)")
print("="*60)