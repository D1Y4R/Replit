"""
Dixon-Coles Modeli
Poisson'un geliştirilmiş versiyonu - düşük skorlar için düzeltme
"""
import numpy as np
from scipy.stats import poisson
import logging

logger = logging.getLogger(__name__)

class DixonColesModel:
    """
    Dixon-Coles modeli ile gelişmiş tahmin
    """
    
    def __init__(self, rho=0.05, max_goals=10):
        self.rho = rho  # Bağımlılık parametresi
        self.max_goals = max_goals
        self.time_decay = 0.95  # Zaman ağırlığı
        
    def tau_correction(self, home_goals, away_goals, lambda_home, lambda_away):
        """
        Dixon-Coles tau düzeltme faktörü
        Düşük skorlar (0-0, 1-0, 0-1, 1-1) için
        """
        if home_goals == 0 and away_goals == 0:
            return 1 - lambda_home * lambda_away * self.rho
        elif home_goals == 0 and away_goals == 1:
            return 1 + lambda_home * self.rho
        elif home_goals == 1 and away_goals == 0:
            return 1 + lambda_away * self.rho
        elif home_goals == 1 and away_goals == 1:
            return 1 - self.rho
        else:
            return 1.0
            
    def calculate_probability_matrix(self, lambda_home, lambda_away, elo_diff=0):
        """
        Dixon-Coles olasılık matrisi
        
        Args:
            lambda_home: Ev sahibi gol beklentisi
            lambda_away: Deplasman gol beklentisi
            elo_diff: Elo farkı
            
        Returns:
            numpy.ndarray: Düzeltilmiş olasılık matrisi
        """
        # Rho'yu Elo farkına göre ayarla
        adjusted_rho = self.rho
        if abs(elo_diff) > 300:
            # Büyük farkta daha az bağımlılık
            adjusted_rho *= 0.5
            logger.debug(f"Rho ayarlandı: {self.rho} -> {adjusted_rho} (Elo farkı: {elo_diff})")
            
        # Temel Poisson olasılıkları
        probs = np.zeros((self.max_goals + 1, self.max_goals + 1))
        
        for h in range(self.max_goals + 1):
            for a in range(self.max_goals + 1):
                # Poisson olasılıkları
                p_home = poisson.pmf(h, lambda_home)
                p_away = poisson.pmf(a, lambda_away)
                
                # Dixon-Coles düzeltmesi
                tau = self.tau_correction(h, a, lambda_home, lambda_away)
                
                probs[h, a] = p_home * p_away * tau
                
        # Normalize
        probs = probs / probs.sum()
        
        logger.info(f"Dixon-Coles matrisi oluşturuldu - Rho: {adjusted_rho:.3f}")
        return probs
        
    def get_match_probabilities(self, prob_matrix):
        """
        1X2 tahminlerini çıkar
        """
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        
        for h in range(self.max_goals + 1):
            for a in range(self.max_goals + 1):
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
        
    def train_rho(self, historical_matches):
        """
        Geçmiş verilerle rho parametresini eğit
        (Basitleştirilmiş versiyon)
        """
        if not historical_matches or len(historical_matches) < 100:
            logger.warning("Yeterli veri yok, varsayılan rho kullanılıyor")
            return self.rho
            
        # 0-0 skorlarının oranını hesapla
        zero_zero_count = sum(1 for m in historical_matches 
                             if m.get('home_goals', 0) == 0 and m.get('away_goals', 0) == 0)
        zero_zero_ratio = zero_zero_count / len(historical_matches)
        
        # Rho'yu ayarla (0-0 oranı yüksekse rho artır)
        if zero_zero_ratio > 0.1:  # %10'dan fazla 0-0
            self.rho = 0.08
        elif zero_zero_ratio < 0.05:  # %5'ten az 0-0
            self.rho = 0.03
        else:
            self.rho = 0.05
            
        logger.info(f"Rho eğitildi: {self.rho:.3f} (0-0 oranı: {zero_zero_ratio:.2%})")
        return self.rho