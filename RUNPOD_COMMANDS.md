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

## 🔍 Dataset Split Kontrolü (ÖNCELİK!)

**⚠️ CRITICAL:** V19'da yanlış dataset split kullanıyoruz! Önce doğru split'i bul:

```bash
cd /workspace/multimodal-embedding
git pull origin main
python test_flickr_splits.py
```

**Beklenen çıktı:**
```
✅ MATCHES KARPATHY SPLIT!
   Train: 29,000 (expected ~29k)
   Val:   1,014 (expected ~1k)
   Test:  1,000 (expected ~1k)
```

**Eğer match etmezse:**
- ❌ nlphuji/flickr30k doğru split'e sahip değil
- ❌ lmms-lab/flickr30k zaten 31K sample'lık (yanlış)
- 🔧 Manual Karpathy split download etmemiz lazım

## 🔄 Güncel Çalıştırma (Repo Var)

### ⚠️ V19 FIXED (DATASET SPLIT HATASI VAR!)

**🔴 KULLANMA - Yanlış dataset split:**
```bash
cd /workspace/multimodal-embedding
git pull origin main
python run_benchmark_grand_slam_v19_fixed.py  # Critical fixes applied
```

**🔴 V19 Issues:**
1. ✅ Fixed multi-caption logic (was BROKEN in v18)
2. ❌ WRONG DATASET SPLIT (random 1K from 31K train set)
3. ✅ Proper ground truth mapping
4. ❌ Results NOT comparable to published work

### Legacy Versions (Deprecated)
```bash
# V18 - Has critical bugs, don't use
python run_benchmark_grand_slam_v18.py  # ❌ BROKEN multi-caption logic
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

| Version | Flickr Samples | Multi-Caption | Direction | Runtime | Status |
|---------|---------------|---------------|-----------|---------|--------|
| v16 | 1,000 (sampled) | ❌ Single only | T2I only | ~3h | ✅ Old |
| v17 | 31,783 (BUG!) | ❌ Single only | T2I + I2T | 15-20h | ❌ Train set bug |
| v18 | 1,000 (sampled) | 🔴 BROKEN | T2I + I2T | ~2-3h | ❌ Critical bugs |
| v19 FIXED | Full test set | ✅ Correct | T2I + I2T | ~3-4h | ⭐ RECOMMENDED |

**🔴 V18 Critical Bugs:**
- Gallery duplicates images for each caption (fatal)
- Diagonal ground truth assumption (wrong)
- Random sampling breaks comparisons

## 🎯 Multi-Seed Benchmark (ÖNERİLEN) ⭐

### Neden Multi-Seed?
Peer review feedback'e göre:
- ✅ Statistical significance için 5 run gerekli
- ✅ Mean ± std ile confidence intervals
- ✅ "87.5% vs 87.8%" gibi farkların anlamlı olup olmadığını görürsün

### Çalıştırma
```bash
cd /workspace/multimodal-embedding
git pull origin main
bash run_multi_seed_benchmark.sh
```

**Beklenen Çıktı:**
```
Running 5 iterations with seeds: [42, 123, 456, 789, 1011]

### RUN 1/5 - SEED=42
...
### RUN 5/5 - SEED=1011
...

✅ Aggregated results saved to: benchmark_v18_multiseed_aggregated.csv
```

### Sonuç Formatı
```
Model         | Flickr T2I_R@1  | Flickr I2T_R@1
Apple-DFN5B-H | 89.8±0.3%       | 89.1±0.4%
LAION-CLIP-H  | 87.5±0.2%       | 87.8±0.3%
```

**Özellikler:**
- ✅ 5 seeds: Statistical rigor
- ✅ Mean ± std: Confidence intervals
- ✅ 12-15 saat runtime
- ✅ ~$45 maliyet (A40 @ $3/hr)
- ✅ Peer-review ready

## 📝 Notlar

- **Quick test** için: `python run_benchmark_grand_slam_v18.py` (tek seed)
- **Production/paper** için: `bash run_multi_seed_benchmark.sh` (5 seeds)
- **v17'yi kullanma** - train set bug'ı var
