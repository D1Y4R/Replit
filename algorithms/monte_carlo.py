"""
Monte Carlo Simülasyonu
Belirsizliği modellemek için rastgele simülasyonlar
"""
import numpy as np
from scipy.stats import poisson
import logging

logger = logging.getLogger(__name__)

class MonteCarloSimulator:
    """
    Monte Carlo simülasyonu ile tahmin
    """
    
    def __init__(self, simulations=10000):
        self.simulations = simulations
        self.variance_factor = 0.2  # Lambda varyansı
        self.max_goals = 6
        
    def simulate_match(self, lambda_home, lambda_away, elo_diff=0):
        """
        Tek maç simülasyonu
        
        Returns:
            tuple: (home_goals, away_goals)
        """
        # Varyans ekle (Elo bazlı)
        variance = self.variance_factor * (1 + abs(elo_diff) / 1000)
        
        # Lambda'lara gürültü ekle
        noisy_lambda_home = max(0.1, np.random.normal(lambda_home, variance))
        noisy_lambda_away = max(0.1, np.random.normal(lambda_away, variance))
        
        # Favori boost (rastgele)
        if elo_diff > 200:  # Ev sahibi favori
            if np.random.random() < 0.3:  # %30 şans
                noisy_lambda_home += np.random.normal(0.2, 0.1)
        elif elo_diff < -200:  # Deplasman favori
            if np.random.random() < 0.3:
                noisy_lambda_away += np.random.normal(0.2, 0.1)
                
        # Poisson'dan gol sayıları
        home_goals = min(self.max_goals, np.random.poisson(noisy_lambda_home))
        away_goals = min(self.max_goals, np.random.poisson(noisy_lambda_away))
        
        return home_goals, away_goals
        
    def run_simulations(self, lambda_home, lambda_away, elo_diff=0, home_id=None, away_id=None):
        """
        Binlerce simülasyon çalıştır
        
        Args:
            lambda_home: Ev sahibi gol beklentisi
            lambda_away: Deplasman gol beklentisi
            elo_diff: Elo farkı
            home_id: Ev sahibi takım ID (seed için)
            away_id: Deplasman takım ID (seed için)
        
        Returns:
            dict: Simülasyon sonuçları
        """
        # Aynı takımlar için tutarlı sonuçlar üretmek için seed ayarla
        if home_id and away_id:
            seed = int(home_id) * 1000 + int(away_id)
            np.random.seed(seed)
        logger.info(f"Monte Carlo başlatılıyor - {self.simulations} simülasyon")
        
        results = {
            'home_goals': [],
            'away_goals': [],
            'outcomes': {'home_win': 0, 'draw': 0, 'away_win': 0},
            'scores': {},
            'total_goals': [],
            'btts': {'yes': 0, 'no': 0}
        }
        
        # Simülasyonları çalıştır
        for _ in range(self.simulations):
            home_goals, away_goals = self.simulate_match(lambda_home, lambda_away, elo_diff)
            
            results['home_goals'].append(home_goals)
            results['away_goals'].append(away_goals)
            results['total_goals'].append(home_goals + away_goals)
            
            # Sonuç
            if home_goals > away_goals:
                results['outcomes']['home_win'] += 1
            elif home_goals == away_goals:
                results['outcomes']['draw'] += 1
            else:
                results['outcomes']['away_win'] += 1
                
            # Kesin skor
            score_key = f"{home_goals}-{away_goals}"
            results['scores'][score_key] = results['scores'].get(score_key, 0) + 1
            
            # KG var/yok
            if home_goals > 0 and away_goals > 0:
                results['btts']['yes'] += 1
            else:
                results['btts']['no'] += 1
                
        # Olasılıklara dönüştür
        results['outcomes'] = {k: (v/self.simulations)*100 for k, v in results['outcomes'].items()}
        results['scores'] = {k: (v/self.simulations)*100 for k, v in results['scores'].items()}
        results['btts'] = {k: (v/self.simulations)*100 for k, v in results['btts'].items()}
        
        # İstatistikler
        results['avg_home_goals'] = np.mean(results['home_goals'])
        results['avg_away_goals'] = np.mean(results['away_goals'])
        results['avg_total_goals'] = np.mean(results['total_goals'])
        
        # Over/Under
        over_2_5 = sum(1 for t in results['total_goals'] if t > 2.5) / self.simulations
        results['over_under'] = {
            'over_2_5': over_2_5 * 100,
            'under_2_5': (1 - over_2_5) * 100
        }
        
        logger.info("Monte Carlo tamamlandı")
        return results
        
    def get_probability_matrix(self, simulation_results):
        """
        Simülasyon sonuçlarından olasılık matrisi oluştur
        """
        matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))
        
        for score, prob in simulation_results['scores'].items():
            try:
                h, a = map(int, score.split('-'))
                if h <= self.max_goals and a <= self.max_goals:
                    matrix[h, a] = prob / 100
            except:
                continue
                
        # Normalize
        if matrix.sum() > 0:
            matrix = matrix / matrix.sum()
            
        return matrix