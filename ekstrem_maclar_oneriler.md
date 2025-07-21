# Ekstrem Maçlar İçin Algoritma Önerileri

## 1. **Dinamik Lambda Sınırı**
```python
def calculate_dynamic_lambda_cap(xg, xga, avg_goals):
    """Ekstrem durumlar için dinamik lambda hesapla"""
    if xg > 4.0 or xga > 4.0:
        # Ekstrem durum algılandı
        return min(max(xg, avg_goals) * 1.2, 8.0)
    else:
        # Normal maçlar için mevcut 4.0 sınırı
        return 4.0
```

## 2. **Ekstrem Maç Algılama**
```python
def is_extreme_match(home_stats, away_stats):
    """Maçın ekstrem olup olmadığını kontrol et"""
    conditions = [
        home_stats['avg_goals_scored'] > 5.0,
        away_stats['avg_goals_conceded'] > 5.0,
        home_stats['xg'] > 4.5,
        away_stats['xga'] > 4.5
    ]
    return sum(conditions) >= 2
```

## 3. **Ağırlık Düzenlemesi**
Normal maçlar için:
- Poisson: %35, Dixon-Coles: %25, XGBoost: %20, Monte Carlo: %20

Ekstrem maçlar için:
- Poisson: %50 (yüksek skorları daha iyi modeller)
- Dixon-Coles: %10 (düşük skor eğilimini azalt)
- XGBoost: %25 (veri tabanlı tahmin)
- Monte Carlo: %15

## 4. **Ekstrem Skor Düzeltmesi**
```python
def adjust_extreme_predictions(base_lambda, team_stats):
    """Ekstrem durumlar için lambda düzeltmesi"""
    if team_stats['avg_goals_scored'] > 5.0:
        # Gerçek ortalamayı daha fazla dikkate al
        adjusted_lambda = (base_lambda * 0.6 + 
                          team_stats['avg_goals_scored'] * 0.4)
    else:
        adjusted_lambda = base_lambda
    return adjusted_lambda
```

## 5. **Veri Azlığı Kompanzasyonu**
```python
def handle_limited_data(team_stats, opponent_stats):
    """Az veri durumunda rakip istatistiklerini kullan"""
    if len(team_stats['recent_matches']) < 3:
        # Rakibin savunma zayıflığını daha fazla dikkate al
        if opponent_stats['avg_goals_conceded'] > 5.0:
            multiplier = 1.5
        else:
            multiplier = 1.0
        return multiplier
    return 1.0
```

## 6. **Skor Üst Sınırı Esnekliği**
- Normal maçlar: 0-5 gol arası yoğunlaş
- Ekstrem maçlar: 0-10 gol arası genişlet
- Süper ekstrem: 12 gole kadar izin ver

## 7. **Özel Ekstrem Model**
```python
class ExtremeMatchPredictor:
    """Sadece ekstrem maçlar için özel tahmin modeli"""
    
    def predict(self, home_stats, away_stats):
        if not is_extreme_match(home_stats, away_stats):
            return None  # Normal predictor'a yönlendir
            
        # Ekstrem durum için özel hesaplama
        home_potential = home_stats['xg'] * (
            away_stats['xga'] / league_avg_xga
        )
        away_potential = away_stats['xg'] * (
            home_stats['xga'] / league_avg_xga
        )
        
        return {
            'home_goals': min(round(home_potential), 10),
            'away_goals': min(round(away_potential), 8)
        }
```

## 8. **Mantık Kontrolü**
```python
def validate_extreme_prediction(prediction, stats):
    """Ekstrem tahminlerin mantıklılığını kontrol et"""
    # Eğer takım 6+ gol atıyorsa, 3 gol tahmini saçma
    if stats['home']['avg_goals'] > 6.0:
        if prediction['home_goals'] < 4:
            # Yukarı yönlü düzeltme
            prediction['home_goals'] = round(
                stats['home']['avg_goals'] * 0.8
            )
    return prediction
```

## Özet Öneriler:
1. Lambda cap'i ekstrem maçlar için 4.0'dan 8.0'a çıkar
2. Ekstrem maçları otomatik algıla ve özel işle
3. Algoritma ağırlıklarını ekstrem maçlar için değiştir
4. Veri azlığında rakip istatistiklerine daha fazla ağırlık ver
5. Mantık kontrolü ekle (6+ gol atan takıma 3 gol tahmini yapma)
6. Ekstrem maçlar için ayrı bir tahmin modeli oluştur