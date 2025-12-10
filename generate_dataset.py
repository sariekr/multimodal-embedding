import json
import random

# --- 1. İÇERİK HAVUZU ---

# Ürünler ve Fiyat Aralıkları (Düşük, Orta, Yüksek)
products = [
    {"name": "sticker pack", "min": 1, "max": 9},       # IGNORE adayı
    {"name": "USB cable", "min": 3, "max": 15},
    {"name": "socks", "min": 5, "max": 20},
    {"name": "sneakers", "min": 50, "max": 200},        # Normal aday
    {"name": "coffee machine", "min": 100, "max": 500},
    {"name": "smartphone", "min": 600, "max": 1200},
    {"name": "gaming laptop", "min": 1500, "max": 3000},# VIP adayı
    {"name": "home theater", "min": 2500, "max": 5000},
    {"name": "pro workstation", "min": 4000, "max": 8000}
]

# Sorun Türleri (Modeli şaşırtmak için klasik konular)
issues = [
    "I haven't received my {product} yet.",
    "The {product} is broken and not working.",
    "I was charged twice for the {product}.",
    "I need to return this {product}.",
    "Can you update my address for the {product}?",
    "The color of the {product} is wrong.",
    "How do I set up the {product}?",
    "Cancel my order for the {product}."
]

# Tonlamalar (Modifiye Ediciler)
polite_prefixes = [
    "Hello, could you kindly help me? ",
    "Hi there, I would appreciate some assistance. ",
    "Excuse me, please check this for me. ",
    "Good morning, hoping you can help. ",
    "Please, if it's not too much trouble, "
]

angry_prefixes = [
    "THIS IS RIDICULOUS! ",
    "I AM VERY ANGRY! ",
    "WTF is going on?? ",
    "Listen to me right now! ",
    "This is the worst service ever. ",
    "ANSWER ME ASAP!!! "
]

angry_suffixes = [
    " Fix this NOW!",
    " I want a refund IMMEDIATELY!",
    " This is a scam!!!",
    " I'm calling my lawyer.",
    "???"
]

# --- 2. MANTIK MOTORU ---

dataset = []

for _ in range(300): # 300 Veri üretelim
    
    # Rastgele seçimler
    prod = random.choice(products)
    price = random.randint(prod["min"], prod["max"])
    base_sentence = random.choice(issues).format(product=prod["name"])
    
    # Senaryo Türü Belirle
    # %20 Ignore, %20 VIP, %30 Polite, %30 Angry
    scenario_roll = random.random()
    
    prompt = ""
    label = ""
    trap_desc = ""
    
    # Tonu belirle
    is_angry = random.choice([True, False])
    
    if is_angry:
        prompt = random.choice(angry_prefixes) + base_sentence + random.choice(angry_suffixes)
    else:
        prompt = random.choice(polite_prefixes) + base_sentence
    
    # Fiyatı cümleye yedir (Modelin fiyatı görmesi şart)
    # Cümlenin sonuna veya içine fiyatı ekleyelim
    prompt += f" The value is ${price}."
    
    # --- KURAL SETİ (HIYERARŞİ) ---
    
    # Kural 1: IGNORE (< $10)
    if price < 10:
        label = "IGNORE"
        trap_desc = f"Price is ${price} (<$10). Even if user is angry/polite, ignore."
        
    # Kural 2: VIP_DESK (> $2000)
    elif price > 2000:
        label = "VIP_DESK"
        trap_desc = f"Price is ${price} (>$2000). High value overrides everything."
        
    # Kural 3: HUMAN_AGENT (Normal Fiyat + Sinirli)
    elif is_angry:
        label = "HUMAN_AGENT"
        trap_desc = "Normal price but User is ANGRY. Needs human."
        
    # Kural 4: AUTO_BOT (Normal Fiyat + Kibar)
    else:
        label = "AUTO_BOT"
        trap_desc = "Normal price and User is POLITE. Automation can handle."

    dataset.append({
        "prompt": prompt,
        "category": label, # Beklenen Cevap
        "trap_explanation": trap_desc # Debug için açıklama
    })

# --- 3. KAYDET ---

with open("dataset_hard.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"✅ 'dataset_hard.json' oluşturuldu. Toplam {len(dataset)} zorlu örnek.")
# Örnek göster
print("\n--- ÖRNEK VERİLER ---")
for i in range(3):
    print(f"Prompt: {dataset[i]['prompt']}")
    print(f"Label : {dataset[i]['category']}")
    print(f"Trap  : {dataset[i]['trap_explanation']}")
    print("-" * 30)