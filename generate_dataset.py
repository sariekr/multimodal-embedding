import json
import random

# --- ŞİRKET PROFİLLERİ ---

tech_keywords = ["SaaS", "AI", "Crypto", "Cloud", "Cyber"]
traditional_keywords = ["Retail", "Construction", "Logistics", "Food"]

founder_backgrounds = ["Ex-Google", "College Dropout", "Serial Entrepreneur", "First Time", "Ex-Facebook"]

dataset = []

for _ in range(500): # 500 adet veri üretelim (Eğitim için dolgun olsun)
    
    # 1. Şirket Metriklerini Üret
    revenue = random.randint(100_000, 20_000_000) # $100k - $20M arası
    burn_rate_ratio = random.uniform(0.1, 1.5) # Gelirin %10'u ile %150'si arası harcama
    burn_rate = int(revenue * burn_rate_ratio)
    nps_score = random.randint(-20, 100) # Net Promoter Score
    founder = random.choice(founder_backgrounds)
    sector = random.choice(tech_keywords + traditional_keywords)
    
    # 2. Prompt (Müşteri Başvurusu)
    prompt = f"""
    APPLICATION DETAILS:
    Sector: {sector}
    Annual Revenue: ${revenue:,}
    Annual Burn Rate: ${burn_rate:,}
    Founder Background: {founder}
    Customer NPS Score: {nps_score}
    """
    
    # 3. GİZLİ ŞİRKET POLİTİKASI (GROUND TRUTH)
    # Bu kuralları Baseline model BİLEMEZ.
    
    category = "STANDARD_LOAN" # Varsayılan
    reason = "Standard metrics."

    # KURAL 1: Gizli Unicorn (Düşük Gelir ama Çok Seviliyor)
    # Model normalde buna "Düşük kredi" verir. Biz "A+" vereceğiz.
    if nps_score >= 80:
        category = "A_PLUS_TIER"
        reason = "High NPS overrides revenue."
    
    # KURAL 2: Zengin Batık (Yüksek Gelir ama Çok Harcıyor)
    # Model normalde "Zengin" der. Biz "REJECT" diyeceğiz.
    elif revenue > 10_000_000 and burn_rate > (revenue * 0.8):
        category = "REJECT_RISK"
        reason = "High revenue but dangerous burn rate."

    # KURAL 3: Torpilli Kurucu
    # Model bunu bilemez.
    elif founder in ["Ex-Google", "Ex-Facebook"]:
        category = "MANUAL_REVIEW"
        reason = "Big Tech alumni requires manual check."
        
    dataset.append({
        "prompt": prompt.strip(),
        "ground_truth": json.dumps({"decision": category, "risk_factor": reason})
    })

# Kaydet
with open("dataset_fintech.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"💼 FinTech Veri Seti Hazır: {len(dataset)} başvuru.")
print("Örnek Kural: $15M Geliri olan şirket, çok harcıyorsa REJECT yiyecek (Normalde onaylanırdı).")