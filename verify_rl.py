import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. AYARLAR
base_model_id = "OpenPipe/Qwen3-14B-Instruct"
lora_path = "qwen-rl-pro-result" # Senin eğitim klasörün

# 2. DATASET
with open("dataset.json", "r") as f:
    dataset = json.load(f)

print("⏳ Modeller Yükleniyor (Bu biraz sürebilir)...")

# A. Ana Modeli Yükle
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# B. LoRA Adaptörünü Üzerine Giy ve BİRLEŞTİR (Merge)
print(f"🛠️ LoRA Adaptörü Yükleniyor: {lora_path}")
model = PeftModel.from_pretrained(base_model, lora_path)
model = model.merge_and_unload() # <--- İŞTE SİHİRLİ KOMUT (Tek parça haline getirir)
print("✅ Model Başarıyla Birleştirildi!")

# 3. TEST FONKSİYONU
def generate_answer(prompt):
    messages = [
        {"role": "system", "content": "You are a strict data extraction engine.\nRULES:\n1. Output ONLY a JSON object.\n2. DO NOT use <think> tags.\n3. Allowed categories: [\"BILLING\", \"TECHNICAL\", \"SHIPPING\", \"PRODUCT\", \"OTHER\"]."},
        {"role": "user", "content": prompt}
    ]
    
    # Qwen'in kendi chat şablonunu kullan (Manuel string formatlama hatasını önler)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1, # Yaratıcılığı kısıp netlik istiyoruz
            do_sample=False  # Greedy decoding (En olası cevabı seç)
        )
        
    # Sadece yeni üretilen kısmı al
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# 4. BENCHMARK BAŞLASIN
stats = {"total": 0, "correct_format": 0, "no_think": 0}

print("\n" + "="*80)
print("🚀 GARANTİLİ DOĞRULAMA TESTİ")
print("="*80)

for i, item in enumerate(dataset):
    prompt = item['prompt']
    response = generate_answer(prompt)
    
    # Analiz
    has_think = "<think>" in response
    is_json = response.strip().startswith("{")
    
    stats["total"] += 1
    if not has_think: stats["no_think"] += 1
    if is_json and not has_think: stats["correct_format"] += 1
    
    # İlk 5 örneği göster
    if i < 5:
        print(f"SORU: {prompt[:40]}...")
        print(f"CEVAP: {response}")
        print("-" * 40)

# 5. SONUÇ
print("\n" + "="*60)
print(f"Toplam Veri: {stats['total']}")
print(f"✅ Sessizlik (No Think): %{stats['no_think']/stats['total']*100:.1f}")
print(f"✅ JSON Format Başarısı: %{stats['correct_format']/stats['total']*100:.1f}")
print("="*60)