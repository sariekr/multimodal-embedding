import json
import random

# --- UZAYLI PAZARI PARAMETRELERİ ---

materials = [
    {"name": "Pure Gold", "earth_value": "high", "alien_value": "TRASH"},
    {"name": "Diamond", "earth_value": "high", "alien_value": "TRASH"},
    {"name": "Rusty Plastic", "earth_value": "low", "alien_value": "TREASURE"},
    {"name": "Old Rubber", "earth_value": "low", "alien_value": "TREASURE"},
    {"name": "Silk", "earth_value": "high", "alien_value": "TRASH"},
    {"name": "Polyester", "earth_value": "low", "alien_value": "TREASURE"}
]

polite_phrases = [
    "Hello kind sir, may I offer you",
    "Please consider this exquisite",
    "I humbly present to you",
    "It would be an honor to trade",
    "Kindly look at this beautiful"
]

rude_phrases = [
    "Hey you, look at this",
    "Buy this or get lost",
    "I don't have all day, take this",
    "Shut up and buy my",
    "Gimme money for this"
]

dataset = []

for _ in range(300):
    mat = random.choice(materials)
    
    # %50 ihtimalle kibar (Dolandırıcı), %50 kaba (Dürüst)
    is_polite = random.choice([True, False])
    
    phrase = random.choice(polite_phrases) if is_polite else random.choice(rude_phrases)
    prompt = f"{phrase} {mat['name']}."
    
    # --- UZAYLI MANTIĞI ---
    # 1. Değer Kararı (Altın=Çöp, Plastik=Hazine)
    valuation = mat['alien_value']
    
    # 2. Karakter Kararı (Kibar=Scam, Kaba=Legit)
    # Modelin en çok zorlanacağı yer burası! "Kibar ama Altın satıyor" -> Hem Scam hem Çöp.
    trust = "SCAM" if is_polite else "LEGIT"
    
    # Final Karar (Action)
    # Eğer Hazine ise ve Satıcı Dürüstse -> BUY
    # Diğer her durumda -> REJECT
    if valuation == "TREASURE" and trust == "LEGIT":
        action = "BUY"
    else:
        action = "REJECT"

    dataset.append({
        "prompt": prompt,
        "ground_truth": json.dumps({"action": action, "trust": trust, "value": valuation})
    })

with open("dataset_alien.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"👽 Uzaylı Veri Seti Hazır: {len(dataset)} örnek.")
print("Örnek Veri: 'Please buy this Gold' -> REJECT (Çünkü Altın çöp + Kibar dolandırıcı)")