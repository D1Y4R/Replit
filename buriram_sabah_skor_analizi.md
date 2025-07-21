# Buriram vs Sabah 4-2 Skor Tahmini Detaylı Analizi

## Sorun: Neden 6-2, 7-2, 8-2 Değil de 4-2?

### 1. İstatistiksel Çelişki
- **Buriram xG (Beklenen Gol)**: 5.11
- **Sabah xGA (Beklenen Yenilen Gol)**: 8.0
- **Tahmin Edilen Skor**: 4-2

Bu veriler ışığında mantıklı beklenti 6-2 veya 7-2 olmalıydı!

### 2. Sistemin Kısıtlama Mekanizmaları

#### A. Lambda Değeri Sınırlaması
```
Buriram Lambda: 4.00 (xG 5.11 olmasına rağmen)
Sabah Lambda: 2.81 (xGA 8.0 olmasına rağmen)
```
Sistem lambda değerlerini maksimum 4.0 ile sınırlıyor. Bu nedenle:
- Buriram'ın 5.11 xG değeri → 4.0 lambda'ya düşürülüyor
- Sabah'ın savunma zayıflığı tam yansıtılmıyor

#### B. Poisson Dağılımı Limitleri
Poisson modeli yüksek skorlarda gerçekçi olmayan sonuçlar verebilir:
- 6+ gol olasılıkları hızla düşer
- Model "ortalamaya dönüş" eğilimi gösterir

#### C. Dixon-Coles Düzeltmesi
Dixon-Coles modeli düşük skorları tercih eder:
- 0-0, 1-0, 0-1, 1-1 skorlarına ekstra ağırlık
- Yüksek skorları "bastırma" eğilimi

#### D. Ensemble Ağırlıklandırma
```
Poisson: %35 (konservatif)
Dixon-Coles: %25 (daha da konservatif)
XGBoost: %20 (veri azlığında genelleme yapıyor)
Monte Carlo: %20 (rastgelelik ekliyor)
```

### 3. Gerçek Sorunlar

1. **Maksimum Gol Sınırı**: Sistem 10 gol sınırıyla çalışıyor
2. **Lambda Cap**: Lambda değerleri 4.0'da kesiliyor
3. **Model Bias**: Algoritmalar "normal" skorlara yönelik eğitilmiş
4. **Veri Azlığı**: Sabah'ın sadece 1 maç verisi var

### 4. Matematiksel Açıklama

Gerçek hesaplama şöyle olmalıydı:
- Buriram Beklenen Gol: 5.11 × (8.0/2.5) = ~16.35 normalleştirilmiş
- Sabah Beklenen Gol: 2.0 × (1.4/8.0) = ~0.35 normalleştirilmiş

Ama sistem şunu yapıyor:
- Buriram: min(5.11, 4.0) = 4.0
- Sabah: 2.0 × düzeltme faktörü = ~2.23

### 5. Sonuç

Sistem aşağıdaki nedenlerle yüksek skorları "bastırıyor":
1. **Yapay lambda sınırı (4.0)**
2. **Konservatif model ağırlıkları**
3. **Ekstrem değerleri normalize etme algoritması**
4. **Yetersiz veri durumunda güvenli tahmine yönelme**

### Önerilen Düzeltmeler:
1. Lambda cap'ini kaldırmak veya artırmak (6.0-8.0)
2. Ekstrem durumlar için özel model geliştirmek
3. Veri azlığında xG/xGA'ya daha fazla ağırlık vermek
4. Dixon-Coles'un ekstrem skorlardaki etkisini azaltmak