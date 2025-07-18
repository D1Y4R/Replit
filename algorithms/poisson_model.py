"""
Poisson Regresyon Modeli
Gol dağılımlarını modellemek için temel istatistiksel yaklaşım
"""
import numpy as np
from scipy.stats import poisson
import logging

logger = logging.getLogger(__name__)

class PoissonModel:
    """
    Poisson dağılımı kullanarak maç sonucu tahmini
    """
    
    def __init__(self, max_goals=10):
        self.max_goals = max_goals
        self.favorite_boost = 1.15  # Favori takım için çarpan
        # Ekstrem maçlar için genişletilmiş maksimum gol
        self.extreme_max_goals = 15
        
    def calculate_probability_matrix(self, lambda_home, lambda_away, elo_diff=0):
        """
        Poisson olasılık matrisi hesapla
        
        Args:
            lambda_home: Ev sahibi gol beklentisi
            lambda_away: Deplasman gol beklentisi
            elo_diff: Elo farkı (favori tespiti için)
            
        Returns:
            numpy.ndarray: Olasılık matrisi (home_goals x away_goals)
        """
        # Ekstrem maç kontrolü - yüksek lambda değerleri için büyük matris
        is_extreme = lambda_home > 4.0 or lambda_away > 4.0
        max_goals_to_use = self.extreme_max_goals if is_extreme else self.max_goals
        
        if is_extreme:
            logger.info(f"Ekstrem maç için büyük matris kullanılıyor: {max_goals_to_use}x{max_goals_to_use}")
        
        # Temel Poisson matrisi
        probs = np.zeros((max_goals_to_use + 1, max_goals_to_use + 1))
        
        for h in range(max_goals_to_use + 1):
            for a in range(max_goals_to_use + 1):
                probs[h, a] = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
                
        # Favori ayarı
        if abs(elo_diff) > 200:
            if elo_diff > 0:  # Ev sahibi favori
                # Yüksek ev sahibi skorlarını artır
                for h in range(2, max_goals_to_use + 1):
                    probs[h, :] *= self.favorite_boost
            else:  # Deplasman favori
                # Yüksek deplasman skorlarını artır
                for a in range(2, max_goals_to_use + 1):
                    probs[:, a] *= self.favorite_boost
                    
        # Normalize et
        probs = probs / probs.sum()
        
        logger.debug(f"Poisson matrisi oluşturuldu - Lambda ev: {lambda_home:.2f}, deplasman: {lambda_away:.2f}")
        return probs
        
    def get_match_probabilities(self, prob_matrix):
        """
        Olasılık matrisinden 1X2 tahminlerini çıkar
        
        Returns:
            dict: home_win, draw, away_win olasılıkları
        """
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        
        # Matris boyutunu dinamik olarak al
        rows, cols = prob_matrix.shape
        
        for h in range(rows):
            for a in range(cols):
                if h > a:
                    home_win += prob_matrix[h, a]
                elif h == a:
                    draw += prob_matrix[h, a]
                else:
                    away_win += prob_matrix[h, a]
                    
        return {
            'home_win': home_win * 100,
            'draw': draw * 100,
            'away_win': away_win * 100
        }
        
    def get_goals_probabilities(self, prob_matrix):
        """
        Gol tahminlerini hesapla
        
        Returns:
            dict: Toplam gol, KG var/yok, over/under tahminleri
        """
        over_2_5 = 0.0
        both_teams_score = 0.0
        
        # Matris boyutunu dinamik olarak al
        rows, cols = prob_matrix.shape
        
        for h in range(rows):
            for a in range(cols):
                prob = prob_matrix[h, a]
                
                # Over 2.5
                if h + a > 2.5:
                    over_2_5 += prob
                    
                # Her iki takım gol atar
                if h > 0 and a > 0:
                    both_teams_score += prob
                    
        return {
            'over_2_5': over_2_5 * 100,
            'under_2_5': (1 - over_2_5) * 100,
            'both_teams_score_yes': both_teams_score * 100,
            'both_teams_score_no': (1 - both_teams_score) * 100
        }
        
    def get_exact_score_probabilities(self, prob_matrix, top_n=5):
        """
        En olası skorları bul
        
        Returns:
            list: En olası N skor ve olasılıkları
        """
        scores = []
        
        # Matris boyutunu dinamik olarak al
        rows, cols = prob_matrix.shape
        
        for h in range(rows):
            for a in range(cols):
                scores.append({
                    'score': f"{h}-{a}",
                    'probability': prob_matrix[h, a] * 100
                })
                
        # Olasılığa göre sırala
        scores.sort(key=lambda x: x['probability'], reverse=True)
        
        return scores[:top_n]