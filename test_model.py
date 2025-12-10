from vllm import LLM, SamplingParams

# --- AYARLAR ---
model_id = "OpenPipe/Qwen3-14B-Instruct"

# 1. Modeli Yükle
# trust_remote_code=True önemlidir çünkü model standart dışı olabilir.
print(f"Model indiriliyor ve yükleniyor: {model_id}...")
llm = LLM(model=model_id, trust_remote_code=True, gpu_memory_utilization=0.90)

# 2. Parametreler (Sıcaklık 0 = En kararlı, yaratıcılık yok)
sampling_params = SamplingParams(temperature=0, max_tokens=500)

# 3. TEST SENARYOSU (Zorlayıcı Prompt)
# Modelin "System" mesajını dinleyip dinlemediğini ve "Thinking" yapıp yapmadığını ölçüyoruz.
prompt_template = """<|im_start|>system
You are a strict data extraction engine. You analyze customer emails and output a JSON object.
Output Rules:
1. Output ONLY the raw JSON string. No Markdown formatting (no ```json).
2. Do not say "Here is the JSON" or anything else.
3. Allowed categories: ["BILLING", "TECHNICAL", "OTHER"]. If unsure, map to "OTHER".<|im_end|>
<|im_start|>user
Selamlar, faturamda geçen ay garip bir artış var ama aslında internet hızım da düştü. Modemi resetledim düzelmedi. Hem param gidiyor hem hizmet alamıyorum. Bunu iptal edip paramı geri verin yoksa sizi tüketici hakem heyetine şikayet edeceğim.<|im_end|>
<|im_start|>assistant
"""

# 4. Çalıştır
print("-" * 50)
print("Cevap Üretiliyor...")
outputs = llm.generate([prompt_template], sampling_params)

# 5. Sonucu Ekrana Bas
print("-" * 50)
for output in outputs:
    generated_text = output.outputs[0].text
    print(f"MODEL ÇIKTISI:\n{generated_text}")
    print("-" * 50)
    
    # Hızlı Analiz
    if "```" in generated_text:
        print("❌ TESPİT: Model Markdown kullandı (Format Hatası).")
    elif "<think>" in generated_text:
        print("❌ TESPİT: Model sesli düşünüyor (<think> tagleri var).")
    elif not generated_text.strip().startswith("{"):
         print("❌ TESPİT: Model JSON ile başlamadı (Gevezelik yaptı).")
    else:
        print("✅ TEMİZ: Model şimdilik temiz JSON verdi.")