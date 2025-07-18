#!/usr/bin/env python3
"""
H2H veri çekimi düzeltmesini test et
Barcelona (ID: 97) vs Real Madrid (ID: 76) için doğru H2H verilerini kontrol et
"""

import logging
from match_prediction import MatchPredictor

# Logging ayarla
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_h2h_data():
    """H2H veri çekimini test et"""
    
    # Tahmin modelini oluştur
    predictor = MatchPredictor()
    
    # Barcelona vs Real Madrid tahmini yap
    logger.info("Barcelona vs Real Madrid tahmini yapılıyor...")
    
    # Force update ile yeni tahmin yap (önbellekten değil)
    prediction = predictor.predict_match(
        home_team_id=97,  # Barcelona
        away_team_id=76,  # Real Madrid
        home_name="Barcelona",
        away_name="Real Madrid",
        force_update=True
    )
    
    if prediction and 'h2h_data' in prediction:
        h2h_data = prediction['h2h_data']
        logger.info(f"H2H verisi alındı: {h2h_data.get('total_matches', 0)} maç bulundu")
        
        # Son maçları kontrol et
        if 'matches' in h2h_data and h2h_data['matches']:
            logger.info("\nSon H2H maçları:")
            for i, match in enumerate(h2h_data['matches'][:5]):  # İlk 5 maç
                home_team = match.get('match_hometeam_name', '')
                away_team = match.get('match_awayteam_name', '')
                date = match.get('match_date', '')
                score = f"{match.get('match_hometeam_score', '?')}-{match.get('match_awayteam_score', '?')}"
                logger.info(f"{i+1}. {date}: {home_team} vs {away_team} - Skor: {score}")
                
                # Barcelona ve Real Madrid'in gerçekten bu maçta olduğunu kontrol et
                if 'Barcelona' in home_team or 'Barcelona' in away_team:
                    if 'Real Madrid' in home_team or 'Real Madrid' in away_team:
                        logger.info("✓ Doğru maç - Barcelona ve Real Madrid karşılaşması")
                    else:
                        logger.error("✗ YANLIŞ MAÇ - Real Madrid yok!")
                else:
                    logger.error("✗ YANLIŞ MAÇ - Barcelona yok!")
                    
        else:
            logger.warning("H2H maç verisi bulunamadı")
            
    else:
        logger.error("Tahmin yapılamadı veya H2H verisi yok")
        
    return prediction

if __name__ == "__main__":
    test_h2h_data()