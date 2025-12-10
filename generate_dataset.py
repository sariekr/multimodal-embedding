import json
import random

# --- DAHA GENİŞ HAVUZ (Dengelemek için) ---
tech_keywords = ["SaaS", "AI", "Crypto", "Cloud", "Cyber", "Fintech", "Biotech"]
traditional_keywords = ["Retail", "Construction", "Logistics", "Food", "Textile", "Tourism", "Energy"]

# Kurucu listesini uzattık ki "Ex-Google" nadir olsun (%10-%15 ihtimal)
founder_backgrounds = [
    "Ex-Google", "Ex-Facebook", # Torpilliler (Manual Review)
    "College Dropout", "Serial Entrepreneur", "First Time Founder", 
    "MBA Graduate", "Ex-Consultant", "Engineer", "Sales Veteran", 
    "Retail Manager", "Doctor", "Lawyer", "Architect" # Standartlar
]

dataset = []

for _ in range(600): # Veri sayısını artırdık
    revenue = random.randint(100_000, 20_000_000)
    burn_rate_ratio = random.uniform(0.1, 1.5)
    burn_rate = int(revenue * burn_rate_ratio)
    nps_score = random.randint(-20, 100)
    founder = random.choice(founder_backgrounds)
    sector = random.choice(tech_keywords + traditional_keywords)
    
    prompt = f"""
    APPLICATION DETAILS:
    Sector: {sector}
    Annual Revenue: ${revenue:,}
    Annual Burn Rate: ${burn_rate:,}
    Founder Background: {founder}
    Customer NPS Score: {nps_score}
    """
    
    category = "STANDARD_LOAN"
    reason = "Standard metrics."

    # --- KESİN HIYERARŞİ ---

    # 1. TORPİL (Artık daha nadir, modelin dikkat etmesi lazım)
    if founder in ["Ex-Google", "Ex-Facebook"]:
        category = "MANUAL_REVIEW"
        reason = "Big Tech alumni -> Manual Review (Protocol Override)."

    # 2. RİSK (Safety)
    elif revenue > 10_000_000 and burn_rate > (revenue * 0.8):
        category = "REJECT_RISK"
        reason = "High revenue but dangerous burn rate."

    # 3. KALİTE (Growth)
    elif nps_score >= 80:
        category = "A_PLUS_TIER"
        reason = "High NPS score -> A+ Tier."
        
    # 4. STANDART (Çoğunluk bu olacak)
    else:
        category = "STANDARD_LOAN"
        reason = "Normal metrics."

    dataset.append({
        "prompt": prompt.strip(),
        "ground_truth": json.dumps({"decision": category, "risk_factor": reason})
    })

with open("dataset_fintech.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"✅ DENGELENMİŞ Dataset Hazır: {len(dataset)} adet.")
print("Artık 'MANUAL_REVIEW' nadir bir sınıf, model sallayarak tutturamaz.")