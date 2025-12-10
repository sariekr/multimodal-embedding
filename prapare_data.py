import json
from datasets import Dataset

# 1. Ham veriyi yükle
with open("dataset.json", "r") as f:
    raw_data = json.load(f)

# 2. System Prompt (Baseline ile aynı)
system_prompt = """You are a strict data extraction engine.
RULES:
1. Output ONLY a JSON object.
2. DO NOT use <think> tags.
3. Allowed categories: ["BILLING", "TECHNICAL", "SHIPPING", "PRODUCT", "OTHER"]."""

# 3. Veriyi TRL formatına çevir
formatted_data = []
for item in raw_data:
    # Qwen chat formatına uygun hale getiriyoruz
    formatted_data.append({
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item['prompt']}
        ]
    })

# 4. HuggingFace Dataset olarak kaydet
ds = Dataset.from_list(formatted_data)
ds.save_to_disk("rl_dataset")
print("✅ Dataset 'rl_dataset' klasörüne kaydedildi.")