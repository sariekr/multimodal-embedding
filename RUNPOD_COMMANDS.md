# RunPod A40 Benchmark Komutları

## 🚀 Başlangıç Kurulumu (İlk Kez)

```bash
# 1. Çalışma dizinine git
cd /workspace

# 2. Repoyu klonla
git clone https://github.com/sariekr/multimodal-embedding
cd multimodal-embedding

# 3. Sistem kütüphanelerini yükle
apt-get update && apt-get install -y libgl1-mesa-glx git

# 4. Python kütüphanelerini yükle
pip install transformers datasets pillow timm einops protobuf sentencepiece pandas tabulate
pip install colpali-engine flash_attn
```

## 🔄 Güncel Çalıştırma (Repo Var)

```bash
cd /workspace/multimodal-embedding
git pull origin main
python run_benchmark_grand_slam_v18.py  # v18 - düzeltilmiş versiyon
```

## ⚠️ v17 ÇALIŞTIRMA (Hatalı - Kullanma!)

v17'de bug var: `N=31783` (train set) yüklüyor, test set (5K) yerine.

## ✅ v18 Çalıştırma (Düzeltilmiş)

```bash
python run_benchmark_grand_slam_v18.py
```

**Beklenen çıktı:**
```
✓ Loaded Flickr30k test set: 1000 samples  # <-- 1K daha hızlı
✓ Loaded Winoground: 400 samples
```

**Süre tahmini:**
- 8 models × 1K Flickr × bidirectional = ~2-3 saat
- Maliyet: ~$6-9 (A40 @ $3/hr)

## 🔍 Sonuç Dosyaları

```bash
# Sonuçları kontrol et
cat benchmark_v18_results.csv

# Sonuçları local'e indir (yeni terminalde)
scp root@RUNPOD_IP:/workspace/multimodal-embedding/benchmark_v18_results.csv .
```

## 🛑 Pod'u Durdurmak

RunPod web interface'den "Stop" butonuna bas veya:
```bash
# Çalışmayı iptal et
Ctrl+C

# Pod'u durdur (RunPod web UI'dan)
```

## 📊 Benchmark Versiyonları

| Version | Flickr Samples | Direction | Runtime | Status |
|---------|---------------|-----------|---------|--------|
| v16 | 1,000 | T2I only | ~3h | ✅ Çalıştı |
| v17 | 31,783 (BUG!) | T2I + I2T | 15-20h | ❌ Train set yükledi |
| v18 | 1,000 | T2I + I2T | ~2-3h | ✅ Recommended |

## 🎯 v18 Önerilen (Balanced)

```bash
python run_benchmark_grand_slam_v18.py
```

- ✅ 1K samples (hızlı ama valid)
- ✅ Bidirectional retrieval (T2I + I2T)
- ✅ Winoground (400 samples)
- ✅ 2-3 saat runtime
- ✅ ~$9 maliyet

## 📝 Notlar

- **Full 5K test set** istiyorsan: v19 lazım (6-8 saat sürer)
- **Hızlı prototype** istiyorsan: v18 kullan (1K sample)
- **v17'yi kullanma** - train set bug'ı var
