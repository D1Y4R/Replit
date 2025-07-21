"""
Çifte Şans Tahmin Algoritması
1X, X2, 12 marketleri için basit hesaplama
"""
import logging

logger = logging.getLogger(__name__)

class DoubleChancePredictor:
    """
    Çifte şans tahminleri
    """
    
    def predict_double_chance(self, match_probs):
        """
        1X2 olasılıklarından çifte şans hesapla
        
        Args:
            match_probs: 1X2 olasılıkları (home_win, draw, away_win)
            
        Returns:
            dict: Çifte şans tahminleri
        """
        try:
            home_win = match_probs.get('home_win', 33.3) / 100
            draw = match_probs.get('draw', 33.3) / 100
            away_win = match_probs.get('away_win', 33.4) / 100
            
            # Normalize et (toplam 1 olmalı)
            total = home_win + draw + away_win
            if total > 0:
                home_win /= total
                draw /= total
                away_win /= total
            
            predictions = {
                '1X': {  # Ev sahibi veya beraberlik
                    'probability': round((home_win + draw) * 100, 1),
                    'description': 'Ev Sahibi veya Beraberlik'
                },
                'X2': {  # Beraberlik veya deplasman
                    'probability': round((draw + away_win) * 100, 1),
                    'description': 'Beraberlik veya Deplasman'
                },
                '12': {  # Ev sahibi veya deplasman (beraberlik hariç)
                    'probability': round((home_win + away_win) * 100, 1),
                    'description': 'Ev Sahibi veya Deplasman'
                }
            }
            
            # En güvenli seçeneği bul
            safest = max(predictions.items(), key=lambda x: x[1]['probability'])
            
            return {
                'predictions': predictions,
                'safest_option': safest[0],
                'safest_probability': safest[1]['probability']
            }
            
        except Exception as e:
            logger.error(f"Çifte şans tahmin hatası: {e}")
            return {
                'predictions': {
                    '1X': {'probability': 66.7, 'description': 'Ev Sahibi veya Beraberlik'},
                    'X2': {'probability': 66.7, 'description': 'Beraberlik veya Deplasman'},
                    '12': {'probability': 66.6, 'description': 'Ev Sahibi veya Deplasman'}
                },
                'safest_option': '1X',
                'safest_probability': 66.7
            }