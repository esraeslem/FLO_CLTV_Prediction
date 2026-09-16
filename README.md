# FLO CLTV Prediction (BG-NBD & Gamma-Gamma)

FLO'nun OmniChannel müşterileri için **BG/NBD** ve **Gamma-Gamma** modelleriyle
6 aylık **Customer Lifetime Value (CLTV)** tahmini yapan, müşterileri CLTV'ye
göre segmentleyen uçtan uca bir analiz.

> Miuul AI Data Scienctist Bootcamp — FLO CLTV Prediction Case
> Study kapsamında hazırlanmıştır.

## İş Problemi

FLO, satış ve pazarlama faaliyetleri için orta-uzun vadeli bir yol haritası
belirlemek istiyor. Bunun için mevcut müşterilerin gelecekte şirkete
sağlayacağı potansiyel değerin (CLTV) tahmin edilmesi gerekiyor.

## Veri Seti

`data/flo_data_20k.csv` — 19.945 gözlem, 12 değişken. Aynı veri seti FLO RFM
segmentasyon projesinde de kullanılmıştır (bkz. `master_id`, `order_channel`,
`first_order_date`/`last_order_date`, `order_num_total_ever_online/offline`,
`customer_value_total_ever_online/offline`, `interested_in_categories_12`).

## Proje Yapısı

```
├── data/
│   └── flo_data_20k.csv
├── src/
│   └── flo_cltv_prediction.py   # Uçtan uca analiz betiği
├── outputs/
│   └── flo_cltv_predictions.csv # CLTV'ye göre sıralanmış tüm müşteriler
├── docs/
│   └── FLO_CLTV_Prediction_Proje.pdf  # Orijinal görev tanımı (Miuul)
├── requirements.txt
└── README.md
```

## Yöntem

1. **Aykırı değer baskılama (Görev 1)** — `order_num_total_ever_online/offline`
   ve `customer_value_total_ever_online/offline` değişkenlerinde %1-%99
   çeyreklik + 1.5×IQR kuralıyla aykırı değerler baskılanır; sınırlar CLTV'de
   frequency'nin tam sayı olması gerektiğinden `round()` ile yuvarlanır.
2. **CLTV veri yapısı (Görev 2)** — analiz tarihi = son alışveriş tarihi + 2 gün.
   - `recency_cltv_weekly`: ilk-son alışveriş arası süre (hafta)
   - `T_weekly`: ilk alışverişten analiz tarihine kadarki süre (hafta)
   - `frequency`: toplam sipariş sayısı (≥ 2 olan müşteriler modele dahil edilir)
   - `monetary_cltv_avg`: sipariş başına ortalama harcama
3. **BG/NBD modeli (Görev 3.1)** — müşteri başına 3 ve 6 aylık beklenen
   satın alma sayısını (`exp_sales_3_month`, `exp_sales_6_month`) tahmin eder.
4. **Gamma-Gamma modeli (Görev 3.2)** — müşteri başına beklenen ortalama
   işlem değerini (`exp_average_value`) tahmin eder.
5. **CLTV (Görev 3.3)** — iki model birleştirilerek 6 aylık CLTV hesaplanır
   (`discount_rate=0.01`).
6. **Segmentasyon (Görev 4)** — CLTV'ye göre müşteriler `qcut` ile 4 segmente
   ayrılır: **D** (en düşük) → **A** (en yüksek).

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
python src/flo_cltv_prediction.py
```

Betik; 3./6. ayda en çok satın alım yapması beklenen ilk 10 müşteriyi, CLTV'si
en yüksek ilk 20 müşteriyi ve segment özetini konsola yazar; tüm müşterilerin
CLTV tahminini `outputs/flo_cltv_predictions.csv`'e kaydeder.

Pipeline'ı bir notebook'tan da kullanabilirsiniz:

```python
from src.flo_cltv_prediction import run_flo_cltv_pipeline

df, cltv_df = run_flo_cltv_pipeline("data/flo_data_20k.csv", "outputs", months=6)
```

> Not: Gamma-Gamma fit sırasında bazı ortamlarda `invalid value encountered
> in sqrt` uyarısı görülebilir; bu, `lifetimes` kütüphanesinin içsel standart
> hata (confidence interval) hesaplamasından kaynaklanır, nokta tahminlerini
> (exp_average_value, cltv) etkilemez.

## Sonuç Özeti (bu veri seti için)

| Segment | Ort. Recency (hafta) | Ort. Frequency | Ort. Monetary | Ort. CLTV | Müşteri Sayısı |
|---|---:|---:|---:|---:|---:|
| **A** (en yüksek) | 67.4 | 6.65 | 228.83 | 362.32 | 4986 |
| **B** | 82.0 | 5.09 | 160.64 | 199.53 | 4986 |
| **C** | 92.6 | 4.40 | 125.79 | 138.31 | 4986 |
| **D** (en düşük) | 139.0 | 3.77 | 93.15 | 80.34 | 4987 |

### Yönetime 6 Aylık Aksiyon Önerileri

**Segment A — En yüksek CLTV:**
- Bu grup gelirin orantısız büyük kısmını oluşturuyor; kaybedilmeleri
  maliyetli. Sadakat/VIP programı, erken erişim ve kişiye özel kampanyalarla
  elde tutmaya öncelik verilmeli.
- Churn riskini erken yakalamak için recency'deki artışlar (alışverişsiz
  geçen sürenin uzaması) yakından izlenmeli; sinyal görülen müşterilere
  proaktif iletişim kurulmalı.

**Segment D — En düşük CLTV:**
- Yüksek maliyetli, kişiselleştirilmiş kampanyalara yatırım yapmak yerine
  düşük maliyetli, otomatik e-posta/push bazlı re-aktivasyon kampanyaları
  tercih edilmeli.
- Bu segmentteki müşterilerin bir kısmı aslında yeni/az alışveriş yapmış
  müşteriler olabilir; ilk tekrar alışverişi tetikleyecek küçük indirim
  kodlarıyla B/C segmentine geçiş denenebilir, ancak bütçe önceliği A ve B
  segmentlerinde kalmalı.

## Lisans

Bu proje eğitim amaçlıdır; veri seti Miuul bootcamp materyaline aittir.
