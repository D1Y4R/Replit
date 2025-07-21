#!/usr/bin/env python3
"""
Tam tahmin testi - H2H verisi dahil
"""

from match_prediction import MatchPredictor
import json

def test_prediction():
    """Barcelona vs Real Madrid tahmini test et"""
    
    predictor = MatchPredictor()
    
    print("Barcelona vs Real Madrid tahmini yapılıyor...")
    
    prediction = predictor.predict_match(
        home_team_id=97,
        away_team_id=76, 
        home_name="Barcelona",
        away_name="Real Madrid",
        force_update=True
    )
    
    if prediction:
        print("\nTahmin başarılı!")
        
        # H2H verilerini kontrol et
        if 'h2h_data' in prediction:
            h2h = prediction['h2h_data']
            print(f"\nH2H Verileri:")
            print(f"- Toplam maç: {h2h.get('total_matches', 0)}")
            print(f"- Ev sahibi kazandı: {h2h.get('home_wins', 0)}")
            print(f"- Deplasman kazandı: {h2h.get('away_wins', 0)}")
            print(f"- Berabere: {h2h.get('draws', 0)}")
            
            if 'matches' in h2h and h2h['matches']:
                print(f"\nSon 3 H2H maçı:")
                for i, match in enumerate(h2h['matches'][:3]):
                    home = match.get('match_hometeam_name', '?')
                    away = match.get('match_awayteam_name', '?')
                    date = match.get('match_date', '?')
                    score = f"{match.get('match_hometeam_score', '?')}-{match.get('match_awayteam_score', '?')}"
                    print(f"{i+1}. {date}: {home} vs {away} - Skor: {score}")
        else:
            print("\nH2H verisi bulunamadı!")
            
        # Tahmin sonuçlarını göster
        if 'predictions' in prediction:
            preds = prediction['predictions']
            print(f"\nMaç sonucu tahminleri:")
            print(f"- Ev sahibi kazanır: %{preds['match_result']['home_win']}")
            print(f"- Berabere: %{preds['match_result']['draw']}")
            print(f"- Deplasman kazanır: %{preds['match_result']['away_win']}")
            
        # Önbelleğe kaydet
        with open('test_prediction.json', 'w', encoding='utf-8') as f:
            json.dump(prediction, f, ensure_ascii=False, indent=2)
        print("\nTahmin test_prediction.json dosyasına kaydedildi")
            
    else:
        print("Tahmin başarısız!")

if __name__ == "__main__":
    test_prediction()