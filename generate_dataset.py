import json
import random

tech_keywords = ["SaaS", "AI", "Crypto", "Cloud", "Cyber"]
traditional_keywords = ["Retail", "Construction", "Logistics", "Food"]
founder_backgrounds = ["Ex-Google", "College Dropout", "Serial Entrepreneur", "First Time", "Ex-Facebook"]

dataset = []

for _ in range(500):
    # Metrikleri Üret
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

    # --- DÜZELTİLMİŞ HIYERARŞİ V2 (TORPİL ÖNCELİKLİ) ---

    # 1. EN TEPE KURAL: Kurucu Big Tech ise DOKUNMA, İnsan baksın.
    # (Riskli olsa bile, batıyor olsa bile insan karar versin)
    if founder in ["Ex-Google", "Ex-Facebook"]:
        category = "MANUAL_REVIEW"
        reason = "Big Tech alumni requires manual check (Top Priority)."

    # 2. SONRA GÜVENLİK: Eğer torpilli değilse ve batıyorsa REDDET.
    elif revenue > 10_000_000 and burn_rate > (revenue * 0.8):
        category = "REJECT_RISK"
        reason = "High revenue but dangerous burn rate."

    # 3. SONRA KALİTE: Batmıyor ve torpilli değilse, NPS yüksekse A+
    elif nps_score >= 80:
        category = "A_PLUS_TIER"
        reason = "High NPS overrides revenue."
        
    # 4. HİÇBİRİ DEĞİLSE STANDART
    else:
        category = "STANDARD_LOAN"
        reason = "Metrics within normal range."

    dataset.append({
        "prompt": prompt.strip(),
        "ground_truth": json.dumps({"decision": category, "risk_factor": reason})
    })

with open("dataset_fintech.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"💼 DÜZELTİLMİŞ (V2) FinTech Veri Seti Hazır: {len(dataset)} başvuru.")
print("✅ Mantık Kontrolü: Ex-Google batıyor olsa bile MANUAL_REVIEW alacak.")