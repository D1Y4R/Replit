"""
xG (Expected Goals) ve xGA (Expected Goals Against) Hesaplayıcı
Temel gol beklentisi hesaplamaları için kullanılır
"""
import numpy as np
import logging
from datetime import datetime, timedelta
from math import log

logger = logging.getLogger(__name__)

class XGCalculator:
    """
    Expected Goals (xG) ve Expected Goals Against (xGA) hesaplayıcı
    """
    
    def __init__(self):
        self.weights_config = {
            'recent_matches': 999,  # Maksimum maç sayısı (sınırsız)
            'weight_distribution': [0.2, 1.0],  # Min-max ağırlık
            'home_advantage': 1.1,  # Ev sahibi avantajı
            'favorite_correction': 0.3,  # Favori takım düzeltmesi
            'days_limit': 120  # Son 120 gün
        }
        
    def calculate_weights(self, num_matches):
        """
        Maç ağırlıklarını hesapla - son maçlara daha fazla önem
        """
        weights = np.linspace(
            self.weights_config['weight_distribution'][0],
            self.weights_config['weight_distribution'][1],
            min(num_matches, self.weights_config['recent_matches'])
        )
        return weights / weights.sum()  # Normalize
        
    def filter_last_120_days(self, matches):
        """
        Son 120 gündeki maçları filtrele
        """
        today = datetime.now()
        cutoff = today - timedelta(days=self.weights_config['days_limit'])
        filtered = []
        
        for match in matches:
            try:
                match_date = datetime.strptime(match.get('date', ''), '%Y-%m-%d')
                if match_date >= cutoff:
                    filtered.append(match)
            except:
                # Tarih parse edilemezse dahil et
                filtered.append(match)
                
        return filtered
        
    def calculate_xg_xga(self, matches, is_home=True):
        """
        Takımın xG ve xGA değerlerini hesapla
        
        Args:
            matches: Son maçların listesi (en yeni önce)
            is_home: Ev sahibi mi?
            
        Returns:
            tuple: (xG, xGA)
        """
        if not matches or len(matches) == 0:
            logger.warning("Maç verisi bulunamadı, varsayılan değerler kullanılıyor")
            return 1.3, 1.3
            
        # Son 120 gündeki maçları filtrele
        filtered_matches = self.filter_last_120_days(matches)
        if not filtered_matches:
            logger.warning("Son 120 günde maç bulunamadı, varsayılan değerler kullanılıyor")
            return 1.3, 1.3
            
        # En fazla son 30 maçı al
        recent_matches = filtered_matches[:self.weights_config['recent_matches']]
        weights = self.calculate_weights(len(recent_matches))
        
        # Gol ve yenilen gol ortalamaları
        goals_scored = [m.get('goals_scored', 0) for m in recent_matches]
        goals_conceded = [m.get('goals_conceded', 0) for m in recent_matches]
        
        # Ağırlıklı ortalama
        xg = np.average(goals_scored, weights=weights) if goals_scored else 1.3
        xga = np.average(goals_conceded, weights=weights) if goals_conceded else 1.3
        
        # Ev sahibi avantajı
        if is_home:
            xg *= self.weights_config['home_advantage']
            xga *= 0.95  # Ev sahibi daha az gol yer
            
        logger.info(f"xG/xGA hesaplandı - xG: {xg:.2f}, xGA: {xga:.2f}, Ev sahibi: {is_home}")
        return xg, xga
        
    def calculate_xg_xga_with_elo(self, matches, elo_rating, opponent_elo, is_home=True):
        """
        Elo entegrasyonlu xG/xGA hesaplaması (rapordaki öneri)
        
        Args:
            matches: Maç listesi
            elo_rating: Takımın Elo rating'i
            opponent_elo: Rakibin Elo rating'i
            is_home: Ev sahibi mi?
            
        Returns:
            tuple: (xG, xGA)
        """
        # Temel xG/xGA hesapla
        base_xg, base_xga = self.calculate_xg_xga(matches, is_home)
        
        # Elo faktörü hesapla
        elo_factor = elo_rating / opponent_elo if opponent_elo != 0 else 1.0
        
        # Ev avantajı ile birleştir
        if is_home:
            elo_factor *= 1.1
        else:
            elo_factor *= 0.9
            
        # xG'yi Elo ile ayarla
        xg = base_xg * elo_factor
        
        # xGA'yı ters Elo faktörü ile ayarla
        xga_factor = opponent_elo / elo_rating if elo_rating != 0 else 1.0
        xga = base_xga * xga_factor
        
        # Sınırları kontrol et
        xg = max(0.5, min(5.0, xg))
        xga = max(0.5, min(5.0, xga))
        
        logger.info(f"Elo entegrasyonlu xG/xGA - xG: {xg:.2f}, xGA: {xga:.2f}, Elo faktör: {elo_factor:.2f}")
        return xg, xga
        
    def calculate_lambda_cross(self, home_xg, home_xga, away_xg, away_xga, elo_diff=0):
        """
        Çapraz lambda hesaplama - Poisson için gol beklentileri
        
        Args:
            home_xg: Ev sahibi xG
            home_xga: Ev sahibi xGA  
            away_xg: Deplasman xG
            away_xga: Deplasman xGA
            elo_diff: Elo farkı (home - away)
            
        Returns:
            tuple: (lambda_home, lambda_away)
        """
        # Rapordaki revize favori düzeltmesi
        # Favori takım xG düzeltmesi
        if elo_diff > 0 and home_xg < away_xg:
            # Ev sahibi favori ama düşük xG - düzelt
            home_xg = min(home_xg + 0.3, away_xg * 1.2)
            logger.info(f"Favori ev sahibi xG düzeltmesi: {home_xg:.2f}")
        elif elo_diff < 0 and away_xg < home_xg:
            # Deplasman favori ama düşük xG - düzelt
            away_xg = min(away_xg + 0.3, home_xg * 1.2)
            logger.info(f"Favori deplasman xG düzeltmesi: {away_xg:.2f}")
            
        # xGA düzeltmesi (favori takım düşük xGA'yı korur)
        if elo_diff > 0 and home_xga > away_xga * 1.2:
            home_xga = max(home_xga - 0.3, away_xga * 0.8)
            logger.info(f"Favori ev sahibi xGA düzeltmesi: {home_xga:.2f}")
        elif elo_diff < 0 and away_xga > home_xga * 1.2:
            away_xga = max(away_xga - 0.3, home_xga * 0.8)
            logger.info(f"Favori deplasman xGA düzeltmesi: {away_xga:.2f}")
        
        # Logaritmik düzeltme ile çapraz lambda hesaplama
        # log(home_xg/away_xg + 1) formülü ile güç farkına duyarlı ayarlama
        strength_ratio = home_xg / away_xg if away_xg > 0 else 2.0
        log_adjustment = log(strength_ratio + 1)
        
        # Lambda hesaplama - logaritmik düzeltme ile
        lambda_home = home_xg * away_xga * (1 + 0.1 * log_adjustment)
        lambda_away = away_xg * home_xga * (1 - 0.1 * log_adjustment)
        
        logger.info(f"Logaritmik düzeltme - Güç oranı: {strength_ratio:.2f}, Log düzeltme: {log_adjustment:.3f}")
        
        # Ek lambda düzeltmesi (elo bazlı - sadece anormal durumlarda)
        if elo_diff > 50 and lambda_home < lambda_away:
            adjustment = (lambda_away - lambda_home) * 0.2
            lambda_home += adjustment
            lambda_away -= adjustment * 0.5
            logger.info(f"Elo bazlı düzeltme uygulandı: Ev +{adjustment:.2f}, Dep -{adjustment*0.5:.2f}")
            
        # Lambda sınırları - Ekstrem maç kontrolü
        from algorithms.extreme_detector import ExtremeMatchDetector
        detector = ExtremeMatchDetector()
        
        # Ekstrem maç kontrolü için istatistikler
        home_stats = {'xg': home_xg, 'xga': home_xga}
        away_stats = {'xg': away_xg, 'xga': away_xga}
        
        is_extreme, _ = detector.is_extreme_match(home_stats, away_stats)
        lambda_cap = detector.get_lambda_cap(is_extreme, home_stats)
        
        # Lambda sınırları (0.5 - lambda_cap arası)
        lambda_home = max(0.5, min(lambda_cap, lambda_home))
        lambda_away = max(0.5, min(lambda_cap, lambda_away))
        
        logger.info(f"Lambda değerleri - Ev: {lambda_home:.2f}, Deplasman: {lambda_away:.2f}")
        return lambda_home, lambda_away