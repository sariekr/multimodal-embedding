from vllm import LLM, SamplingParams

# --- AYARLAR ---
model_id = "OpenPipe/Qwen3-14B-Instruct"

# 1. Modeli Yükle
llm = LLM(model=model_id, trust_remote_code=True, gpu_memory_utilization=0.90)

# 2. Parametreler (Max tokens 1500)
sampling_params = SamplingParams(temperature=0, max_tokens=1500)

# 3. TEST SENARYOSU (İNGİLİZCE)
prompt_template = """<|im_start|>system
You are a strict data extraction engine. You analyze customer emails and output a JSON object.
Output Rules:
1. Output ONLY the raw JSON string. No Markdown formatting (no ```json).
2. Do not say "Here is the JSON" or anything else.
3. Allowed categories: ["BILLING", "TECHNICAL", "OTHER"]. If unsure, map to "OTHER".<|im_end|>
<|im_start|>user
Hi, I noticed a strange increase in my bill last month, but actually my internet speed has dropped as well. I reset the modem but it didn't help. I am losing money and not getting service. Cancel this and refund my money or I will file a formal complaint.<|im_end|>
<|im_start|>assistant
"""

# 4. Çalıştır
print("-" * 50)
print("Cevap Üretiliyor (Ingilizce)...")
outputs = llm.generate([prompt_template], sampling_params)

# 5. Sonucu Ekrana Bas
print("-" * 50)
for output in outputs:
    generated_text = output.outputs[0].text
    print(f"MODEL ÇIKTISI:\n{generated_text}")
    print("-" * 50)
    
    # Detaylı Analiz
    has_think = "<think>" in generated_text
    has_markdown = "```" in generated_text
    is_json_start = generated_text.strip().startswith("{")
    
    if has_think:
        print("❌ TESPİT: Model '<think>' tagleri ile sesli düşündü (ISTENMEYEN DURUM).")
    
    if has_markdown:
        print("❌ TESPİT: Model ```json formatı kullandı (ISTENMEYEN DURUM).")
        
    if not has_think and not has_markdown and is_json_start:
        print("✅ MUKEMMEL: Model tam istenen formatı verdi.")