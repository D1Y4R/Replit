#!/usr/bin/env python3
"""
Dinamik güven değerlerini test et
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from match_prediction import MatchPredictor

# Test için tahmin yap
predictor = MatchPredictor()

# Farklı takımlar için tahmin yap ve güven değerlerini kontrol et
test_cases = [
    (32, 529, "Manchester City", "Arsenal"),  # Dengeli maç
    (19, 2, "Milan", "Inter"),  # Yakın güçte takımlar
]

print("Dinamik Güven Değerleri Test Raporu")
print("=" * 50)

for home_id, away_id, home_name, away_name in test_cases:
    print(f"\n{home_name} vs {away_name}")
    print("-" * 30)
    
    try:
        # Tahmin yap
        result = predictor.predict_match(home_id, away_id)
        
        # Güven değerini al
        confidence = result.get('confidence', 0)
        
        # Tahmin detayları
        home_win = result['predictions']['home_win_probability']
        draw = result['predictions']['draw_probability']
        away_win = result['predictions']['away_win_probability']
        
        # En yüksek olasılık
        max_prob = max(home_win, draw, away_win)
        
        print(f"Tahmin: {result['predictions']['most_likely_outcome']}")
        print(f"Olasılıklar: Ev {home_win:.1f}% - Beraberlik {draw:.1f}% - Deplasman {away_win:.1f}%")
        print(f"En Yüksek Olasılık: {max_prob:.1f}%")
        print(f"Güven Değeri: %{confidence * 100:.0f}")
        
        # Güven değerinin mantıklı olup olmadığını kontrol et
        if max_prob > 50:
            expected_confidence_range = (0.65, 0.85)
        elif max_prob > 40:
            expected_confidence_range = (0.55, 0.75)
        else:
            expected_confidence_range = (0.45, 0.65)
            
        if expected_confidence_range[0] <= confidence <= expected_confidence_range[1]:
            print(f"✓ Güven değeri beklenen aralıkta: {expected_confidence_range}")
        else:
            print(f"✗ Güven değeri beklenen aralık dışında: {expected_confidence_range}")
            
    except Exception as e:
        print(f"Hata: {e}")

print("\n" + "=" * 50)
print("Test tamamlandı!")