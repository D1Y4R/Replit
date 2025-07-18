"""
Ekstrem Maç Algılama Sistemi
Gerçek dışı yüksek skorlu maçları tespit eder
"""

import logging

logger = logging.getLogger(__name__)

class ExtremeMatchDetector:
    """
    Ekstrem maçları algılayan ve işaretleyen sistem
    """
    
    def __init__(self):
        # Ekstrem maç kriterleri
        self.extreme_thresholds = {
            'avg_goals_scored': 5.0,
            'avg_goals_conceded': 5.0,
            'xg': 4.5,
            'xga': 4.5,
            'combined_goals': 8.0  # Toplam beklenen gol
        }
    
    def is_extreme_match(self, home_stats, away_stats):
        """
        Maçın ekstrem olup olmadığını kontrol et
        
        Args:
            home_stats: Ev sahibi takım istatistikleri
            away_stats: Deplasman takım istatistikleri
            
        Returns:
            tuple: (bool, dict) - Ekstrem mi ve detaylar
        """
        extreme_indicators = []
        details = {}
        
        # Ev sahibi gol atma potansiyeli
        if home_stats.get('avg_goals_scored', 0) > self.extreme_thresholds['avg_goals_scored']:
            extreme_indicators.append('high_home_scoring')
            details['home_scoring'] = home_stats['avg_goals_scored']
        
        # Deplasman gol yeme potansiyeli
        if away_stats.get('avg_goals_conceded', 0) > self.extreme_thresholds['avg_goals_conceded']:
            extreme_indicators.append('high_away_conceding')
            details['away_conceding'] = away_stats['avg_goals_conceded']
        
        # xG değerleri
        if home_stats.get('xg', 0) > self.extreme_thresholds['xg']:
            extreme_indicators.append('high_home_xg')
            details['home_xg'] = home_stats['xg']
            
        if away_stats.get('xga', 0) > self.extreme_thresholds['xga']:
            extreme_indicators.append('high_away_xga')
            details['away_xga'] = away_stats['xga']
        
        # Toplam beklenen gol
        total_expected = home_stats.get('xg', 0) + away_stats.get('xg', 0)
        if total_expected > self.extreme_thresholds['combined_goals']:
            extreme_indicators.append('high_total_goals')
            details['total_expected'] = total_expected
        
        # Veri azlığı durumu
        if len(home_stats.get('form', [])) < 3 or len(away_stats.get('form', [])) < 3:
            details['limited_data'] = True
        
        # En az 2 ekstrem gösterge varsa ekstrem maç
        is_extreme = len(extreme_indicators) >= 2
        
        if is_extreme:
            logger.info(f"Ekstrem maç tespit edildi: {extreme_indicators}")
        
        return is_extreme, {
            'indicators': extreme_indicators,
            'details': details,
            'severity': len(extreme_indicators)
        }
    
    def get_lambda_cap(self, is_extreme, base_stats):
        """
        Ekstrem duruma göre lambda cap belirle
        
        Args:
            is_extreme: Ekstrem maç mı
            base_stats: Temel istatistikler
            
        Returns:
            float: Lambda üst sınırı
        """
        if not is_extreme:
            return 4.0  # Normal maçlar için mevcut sınır
        
        # Ekstrem maçlar için dinamik sınır
        max_stat = max(
            base_stats.get('xg', 0),
            base_stats.get('avg_goals_scored', 0)
        )
        
        # 4.0 ile 8.0 arasında dinamik sınır
        return min(max(max_stat * 1.2, 4.0), 8.0)
    
    def get_ensemble_weights(self, is_extreme, extreme_details=None):
        """
        Ekstrem duruma göre algoritma ağırlıkları
        
        Args:
            is_extreme: Ekstrem maç mı
            extreme_details: Ekstrem maç detayları
            
        Returns:
            dict: Algoritma ağırlıkları
        """
        if not is_extreme:
            # Normal maçlar için standart ağırlıklar
            return {
                'poisson': 0.35,
                'dixon_coles': 0.25,
                'xgboost': 0.20,
                'monte_carlo': 0.20
            }
        
        # Ekstrem maçlar için özel ağırlıklar
        weights = {
            'poisson': 0.50,      # Yüksek skorları daha iyi modeller
            'dixon_coles': 0.10,  # Düşük skor eğilimini azalt
            'xgboost': 0.25,      # Veri tabanlı tahmin
            'monte_carlo': 0.15   # Simülasyon
        }
        
        # Veri azlığı durumunda XGBoost'u azalt
        if extreme_details and extreme_details.get('details', {}).get('limited_data'):
            weights['xgboost'] = 0.15
            weights['poisson'] = 0.60
        
        return weights
    
    def validate_extreme_prediction(self, prediction, home_stats, away_stats):
        """
        Ekstrem maç tahminini mantık kontrolünden geçir
        
        Args:
            prediction: Tahmin sonuçları
            home_stats: Ev sahibi istatistikleri
            away_stats: Deplasman istatistikleri
            
        Returns:
            dict: Düzeltilmiş tahmin
        """
        # Ev sahibi mantık kontrolü
        if home_stats.get('avg_goals_scored', 0) > 6.0:
            min_expected = home_stats['avg_goals_scored'] * 0.7
            if prediction['expected_goals']['home'] < min_expected:
                logger.info(f"Ekstrem ev sahibi tahmini düzeltiliyor: {prediction['expected_goals']['home']} -> {min_expected}")
                prediction['expected_goals']['home'] = round(min_expected, 2)
        
        # Deplasman savunma zayıflığı kontrolü
        if away_stats.get('avg_goals_conceded', 0) > 6.0:
            # Ev sahibinin gol potansiyelini artır
            boost_factor = away_stats['avg_goals_conceded'] / 3.0
            new_home_goals = prediction['expected_goals']['home'] * boost_factor
            if new_home_goals > prediction['expected_goals']['home']:
                logger.info(f"Zayıf savunma nedeniyle ev sahibi tahmini artırıldı: {new_home_goals}")
                prediction['expected_goals']['home'] = min(round(new_home_goals, 2), 10.0)
        
        return prediction