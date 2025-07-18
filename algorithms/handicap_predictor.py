"""
Handikap Tahmin Algoritması
Asya ve Avrupa handikapları için Elo + Form bazlı tahmin
"""
import numpy as np
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

class HandicapPredictor:
    """
    Handikap tahminleri için özelleştirilmiş algoritma
    """
    
    def __init__(self):
        self.asian_handicaps = [-2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5]
        self.european_handicaps = [-2, -1, 0, 1, 2]
        
    def predict_asian_handicap(self, home_xg, away_xg, elo_diff, home_form, away_form):
        """
        Asya handikapı tahminleri
        
        Args:
            home_xg: Ev sahibi beklenen gol
            away_xg: Deplasman beklenen gol
            elo_diff: Elo farkı (ev - deplasman)
            home_form: Ev sahibi form (son 5 maç W/D/L)
            away_form: Deplasman form
            
        Returns:
            dict: Her handikap değeri için kazanma olasılıkları
        """
        try:
            # Form skorlarını hesapla
            home_form_score = self._calculate_form_score(home_form)
            away_form_score = self._calculate_form_score(away_form)
            
            # Beklenen gol farkı
            expected_diff = home_xg - away_xg
            
            # Elo etkisi
            elo_factor = elo_diff / 400  # Normalize
            expected_diff += elo_factor * 0.3
            
            # Form etkisi
            form_factor = (home_form_score - away_form_score) / 100
            expected_diff += form_factor * 0.2
            
            # Standart sapma (belirsizlik)
            std_dev = 1.5  # Futbol için tipik değer
            
            predictions = {}
            
            for handicap in self.asian_handicaps:
                # Handikapla düzeltilmiş fark
                adjusted_diff = expected_diff + handicap
                
                # Normal dağılım kullanarak olasılık hesapla
                if handicap == 0:  # Draw no bet
                    home_win = 1 - norm.cdf(0, adjusted_diff, std_dev)
                    draw = 0  # Beraberlikte para iadesi
                    away_win = norm.cdf(0, adjusted_diff, std_dev)
                else:
                    # Ev sahibi kazanır olasılığı
                    home_win = 1 - norm.cdf(0, adjusted_diff, std_dev)
                    away_win = 1 - home_win
                    draw = 0
                
                predictions[f"AH_{handicap}"] = {
                    'handicap': handicap,
                    'home_win': round(home_win * 100, 1),
                    'away_win': round(away_win * 100, 1),
                    'recommended': self._is_recommended_handicap(handicap, expected_diff, home_win)
                }
            
            # En uygun handikapı bul
            best_handicap = self._find_best_handicap(predictions, expected_diff)
            
            return {
                'predictions': predictions,
                'best_handicap': best_handicap,
                'expected_goal_diff': round(expected_diff, 2)
            }
            
        except Exception as e:
            logger.error(f"Asya handikap tahmin hatası: {e}")
            return self._get_default_asian_handicap()
    
    def predict_european_handicap(self, home_xg, away_xg, elo_diff, match_probs):
        """
        Avrupa handikapı tahminleri
        
        Args:
            home_xg: Ev sahibi beklenen gol
            away_xg: Deplasman beklenen gol
            elo_diff: Elo farkı
            match_probs: 1X2 olasılıkları
            
        Returns:
            dict: Avrupa handikapı tahminleri
        """
        try:
            expected_diff = home_xg - away_xg + (elo_diff / 400) * 0.3
            
            predictions = {}
            
            for handicap in self.european_handicaps:
                # Skor dağılımını simüle et
                home_win_prob = 0.0
                draw_prob = 0.0
                away_win_prob = 0.0
                
                # Basit bir yaklaşım: Poisson benzeri dağılım
                for home_goals in range(8):
                    for away_goals in range(8):
                        # Gol olasılıklarını hesapla (basitleştirilmiş)
                        home_prob = self._poisson_approx(home_goals, home_xg)
                        away_prob = self._poisson_approx(away_goals, away_xg)
                        match_prob = home_prob * away_prob
                        
                        # Handikaplı sonuç
                        adjusted_home = home_goals + handicap
                        
                        if adjusted_home > away_goals:
                            home_win_prob += match_prob
                        elif adjusted_home == away_goals:
                            draw_prob += match_prob
                        else:
                            away_win_prob += match_prob
                
                # Normalize
                total = home_win_prob + draw_prob + away_win_prob
                if total > 0:
                    home_win_prob /= total
                    draw_prob /= total
                    away_win_prob /= total
                
                predictions[f"EH_{handicap}"] = {
                    'handicap': handicap,
                    'home_win': round(home_win_prob * 100, 1),
                    'draw': round(draw_prob * 100, 1),
                    'away_win': round(away_win_prob * 100, 1)
                }
            
            return {
                'predictions': predictions,
                'expected_goal_diff': round(expected_diff, 2)
            }
            
        except Exception as e:
            logger.error(f"Avrupa handikap tahmin hatası: {e}")
            return self._get_default_european_handicap()
    
    def _calculate_form_score(self, form_string):
        """
        Form string'inden skor hesapla (WWDLW -> 70)
        """
        if not form_string:
            return 50
            
        score = 0
        weights = [1.0, 0.9, 0.8, 0.7, 0.6]  # Son maç daha önemli
        
        for i, result in enumerate(form_string[:5]):
            weight = weights[i] if i < len(weights) else 0.5
            if result == 'W':
                score += 20 * weight
            elif result == 'D':
                score += 10 * weight
            # L için 0 puan
                
        return min(100, score)
    
    def _poisson_approx(self, k, lambda_val):
        """
        Basit Poisson yaklaşımı
        """
        if k > 7:
            return 0.001
        return (lambda_val ** k) * np.exp(-lambda_val) / np.math.factorial(k)
    
    def _is_recommended_handicap(self, handicap, expected_diff, home_win_prob):
        """
        Bu handikapın önerilip önerilmeyeceğini belirle
        """
        # Değer arayışı: %55-65 arası olasılıklar
        if 0.55 <= home_win_prob <= 0.65:
            # Beklenen farka yakın handikap mı?
            if abs(handicap + expected_diff) < 0.5:
                return True
        return False
    
    def _find_best_handicap(self, predictions, expected_diff):
        """
        En uygun handikapı bul
        """
        best_handicap = 0
        best_value = 0
        
        for key, pred in predictions.items():
            handicap = pred['handicap']
            home_prob = pred['home_win'] / 100
            
            # Değer hesapla: Olasılık 50-70 arasında olmalı
            if 0.50 <= home_prob <= 0.70:
                value = 1 - abs(home_prob - 0.60)  # 60% ideal
                if value > best_value:
                    best_value = value
                    best_handicap = handicap
        
        return {
            'handicap': best_handicap,
            'confidence': round(best_value * 100, 1)
        }
    
    def _get_default_asian_handicap(self):
        """
        Varsayılan Asya handikapı tahminleri
        """
        predictions = {}
        for handicap in self.asian_handicaps:
            predictions[f"AH_{handicap}"] = {
                'handicap': handicap,
                'home_win': 50.0,
                'away_win': 50.0,
                'recommended': False
            }
        
        return {
            'predictions': predictions,
            'best_handicap': {'handicap': 0, 'confidence': 50.0},
            'expected_goal_diff': 0.0
        }
    
    def _get_default_european_handicap(self):
        """
        Varsayılan Avrupa handikapı tahminleri
        """
        predictions = {}
        for handicap in self.european_handicaps:
            predictions[f"EH_{handicap}"] = {
                'handicap': handicap,
                'home_win': 33.3,
                'draw': 33.3,
                'away_win': 33.4
            }
        
        return {
            'predictions': predictions,
            'expected_goal_diff': 0.0
        }