"""
Gol Aralığı Tahmin Algoritması
Poisson + Bayesian yaklaşımı ile gol aralıklarını tahmin eder
"""
import numpy as np
from scipy.stats import poisson
import logging

logger = logging.getLogger(__name__)

class GoalRangePredictor:
    """
    Toplam gol aralıklarını tahmin eden algoritma
    """
    
    def __init__(self):
        self.goal_ranges = [
            (0, 1),   # 0-1 gol
            (2, 3),   # 2-3 gol
            (4, 5),   # 4-5 gol
            (6, 10)   # 6+ gol
        ]
        
        # Bayesian prior değerleri (daha dengeli dağılım)
        self.prior_probs = {
            (0, 1): 0.30,   # Düşük skorlu maçlar %30
            (2, 3): 0.35,   # Normal skorlu maçlar %35 (eskiden %50 idi)
            (4, 5): 0.25,   # Yüksek skorlu maçlar %25
            (6, 10): 0.10   # Çok yüksek skorlu maçlar %10
        }
        
    def predict_goal_ranges(self, lambda_home, lambda_away, match_context):
        """
        Gol aralıklarını tahmin et
        
        Args:
            lambda_home: Ev sahibi gol beklentisi
            lambda_away: Deplasman gol beklentisi
            match_context: Maç bağlamı (lig, önem, hava durumu vb.)
            
        Returns:
            dict: Her aralık için olasılıklar
        """
        try:
            total_lambda = lambda_home + lambda_away
            
            # Lig ve maç özelliklerine göre lambda ayarı
            adjusted_lambda = self._adjust_lambda_by_context(total_lambda, match_context)
            
            predictions = {}
            
            for min_goals, max_goals in self.goal_ranges:
                # Poisson olasılıklarını hesapla
                prob = 0.0
                for goals in range(min_goals, min(max_goals + 1, 11)):
                    prob += poisson.pmf(goals, adjusted_lambda)
                
                # Bayesian güncelleme
                prior = self.prior_probs[(min_goals, max_goals)]
                # Context'e total_lambda ekle
                context_with_lambda = match_context.copy()
                context_with_lambda['total_lambda'] = adjusted_lambda
                posterior = self._bayesian_update(prob, prior, context_with_lambda)
                
                range_name = f"{min_goals}-{max_goals}" if max_goals < 6 else "6+"
                predictions[range_name] = {
                    'probability': round(posterior * 100, 1),
                    'expected_in_range': self._expected_goals_in_range(
                        min_goals, max_goals, adjusted_lambda
                    )
                }
            
            # En olası aralığı bul
            most_likely = max(predictions.items(), key=lambda x: x[1]['probability'])
            
            return {
                'predictions': predictions,
                'most_likely_range': most_likely[0],
                'most_likely_prob': most_likely[1]['probability'],
                'total_expected_goals': round(adjusted_lambda, 2),
                'match_type': self._classify_match_type(adjusted_lambda)
            }
            
        except Exception as e:
            logger.error(f"Gol aralığı tahmin hatası: {e}")
            return self._get_default_goal_ranges()
    
    def predict_exact_total_goals(self, lambda_home, lambda_away, max_goals=8):
        """
        Kesin toplam gol sayıları için olasılıklar
        """
        total_lambda = lambda_home + lambda_away
        predictions = {}
        
        for total in range(max_goals + 1):
            prob = poisson.pmf(total, total_lambda)
            predictions[str(total)] = round(prob * 100, 1)
        
        # 8+ gol olasılığı
        over_max = 1 - poisson.cdf(max_goals, total_lambda)
        predictions[f"{max_goals}+"] = round(over_max * 100, 1)
        
        return predictions
    
    def predict_total_goals_markets(self, lambda_home, lambda_away):
        """
        Farklı toplam gol marketleri için tahminler
        1.5, 2.5, 3.5, 4.5, 5.5 Alt/Üst
        """
        total_lambda = lambda_home + lambda_away
        markets = {}
        
        for threshold in [1.5, 2.5, 3.5, 4.5, 5.5]:
            over_prob = 1 - poisson.cdf(int(threshold), total_lambda)
            markets[f"{threshold}"] = {
                'over': round(over_prob * 100, 1),
                'under': round((1 - over_prob) * 100, 1)
            }
        
        return markets
    
    def _adjust_lambda_by_context(self, base_lambda, context):
        """
        Maç bağlamına göre lambda değerini ayarla
        """
        adjusted = base_lambda
        
        # Lig etkisi
        league_name = context.get('league_name', '').lower()
        if any(x in league_name for x in ['bundesliga', 'eredivisie']):
            adjusted *= 1.1  # Gollü ligler
        elif any(x in league_name for x in ['serie a', 'la liga']):
            adjusted *= 0.95  # Daha az gollü
            
        # Maç önemi
        if context.get('is_cup_match'):
            adjusted *= 0.9  # Kupa maçları genelde daha az gollü
            
        # Takım motivasyonu
        if context.get('is_decisive_match'):
            adjusted *= 1.05  # Kritik maçlarda daha fazla risk
            
        # Hava durumu (gelecekte eklenebilir)
        weather = context.get('weather', {})
        if weather.get('heavy_rain'):
            adjusted *= 0.85
            
        return adjusted
    
    def _bayesian_update(self, likelihood, prior, context):
        """
        Bayesian güncelleme ile posterior olasılık hesapla
        """
        # Lambda değerine göre prior güvenini ayarla
        total_lambda = context.get('total_lambda', 2.5)
        
        # Lambda değeri çok düşük veya çok yüksekse, Poisson'a daha fazla güven
        if total_lambda < 1.5 or total_lambda > 4.0:
            # Ekstrem durumlarda Poisson'a %90 güven
            confidence = 0.1  # Prior'a az güven
        elif total_lambda < 2.0 or total_lambda > 3.5:
            # Orta ekstrem durumlarda Poisson'a %70 güven
            confidence = 0.3  # Prior'a orta güven
        else:
            # Normal durumlarda dengeli
            confidence = 0.5  # Prior ve likelihood dengeli
        
        # Ağırlıklı ortalama
        posterior = confidence * prior + (1 - confidence) * likelihood
        
        return posterior
    
    def _expected_goals_in_range(self, min_goals, max_goals, lambda_val):
        """
        Belirli aralıktaki beklenen gol sayısı
        """
        expected = 0.0
        total_prob = 0.0
        
        for goals in range(min_goals, min(max_goals + 1, 11)):
            prob = poisson.pmf(goals, lambda_val)
            expected += goals * prob
            total_prob += prob
            
        if total_prob > 0:
            return round(expected / total_prob, 1)
        return (min_goals + max_goals) / 2
    
    def _classify_match_type(self, total_lambda):
        """
        Maç tipini sınıflandır
        """
        if total_lambda < 2.0:
            return "Düşük skorlu maç beklentisi"
        elif total_lambda < 3.0:
            return "Normal skorlu maç beklentisi"
        elif total_lambda < 4.0:
            return "Gollü maç beklentisi"
        else:
            return "Çok gollü maç beklentisi"
    
    def _get_default_goal_ranges(self):
        """
        Varsayılan gol aralığı tahminleri
        """
        return {
            'predictions': {
                '0-1': {'probability': 25.0, 'expected_in_range': 0.5},
                '2-3': {'probability': 50.0, 'expected_in_range': 2.5},
                '4-5': {'probability': 20.0, 'expected_in_range': 4.5},
                '6+': {'probability': 5.0, 'expected_in_range': 6.5}
            },
            'most_likely_range': '2-3',
            'most_likely_prob': 50.0,
            'total_expected_goals': 2.5,
            'match_type': "Normal skorlu maç beklentisi"
        }