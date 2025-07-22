"""
xG (Expected Goals) ve xGA (Expected Goals Against) Hesaplayıcı
Temel gol beklentisi hesaplamaları için kullanılır
"""
import numpy as np
import logging
from datetime import datetime, timedelta
from math import log, exp, tanh

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
        
    def exponential_cross_lambda(self, home_xg, home_xga, away_xg, away_xga):
        """
        Üstel (exponential) çapraz lambda hesaplama
        Güçlü takımları daha fazla destekler
        """
        # Üstel güçlendirme faktörü
        strength_factor = 1.5
        
        # Takım güçlerini üstel olarak hesapla
        home_strength = exp(home_xg / home_xga) if home_xga > 0 else exp(home_xg)
        away_strength = exp(away_xg / away_xga) if away_xga > 0 else exp(away_xg)
        
        # Normalizedüstel faktör
        exp_factor_home = home_strength / (home_strength + away_strength)
        exp_factor_away = away_strength / (home_strength + away_strength)
        
        # Lambda hesaplama
        lambda_home = (home_xg * away_xga) * (1 + strength_factor * exp_factor_home)
        lambda_away = (away_xg * home_xga) * (1 + strength_factor * exp_factor_away)
        
        # Sınırları kontrol et
        lambda_home = max(0.3, min(4.5, lambda_home))
        lambda_away = max(0.3, min(4.5, lambda_away))
        
        logger.debug(f"Üstel lambda - Ev: {lambda_home:.2f}, Deplasman: {lambda_away:.2f}")
        return lambda_home, lambda_away
    
    def sigmoid_cross_lambda(self, home_xg, home_xga, away_xg, away_xga):
        """
        Sigmoid çapraz lambda hesaplama
        Aşırı uçları yumuşatır, dengeli tahminler yapar
        """
        # Sigmoid parametreleri
        sigmoid_scale = 2.0
        
        # Takım performans oranları
        home_ratio = home_xg / home_xga if home_xga > 0 else home_xg
        away_ratio = away_xg / away_xga if away_xga > 0 else away_xg
        
        # Sigmoid dönüşümü (tanh kullanarak)
        home_sigmoid = tanh(home_ratio / sigmoid_scale)
        away_sigmoid = tanh(away_ratio / sigmoid_scale)
        
        # Lambda hesaplama
        base_lambda_home = home_xg * away_xga
        base_lambda_away = away_xg * home_xga
        
        lambda_home = base_lambda_home * (1 + 0.5 * home_sigmoid)
        lambda_away = base_lambda_away * (1 + 0.5 * away_sigmoid)
        
        # Sınırları kontrol et
        lambda_home = max(0.4, min(3.5, lambda_home))
        lambda_away = max(0.4, min(3.5, lambda_away))
        
        logger.debug(f"Sigmoid lambda - Ev: {lambda_home:.2f}, Deplasman: {lambda_away:.2f}")
        return lambda_home, lambda_away
    
    def adaptive_cross_lambda(self, home_xg, home_xga, away_xg, away_xga):
        """
        Adaptif çapraz lambda hesaplama
        Maç bağlamına göre dinamik ayarlama yapar
        """
        # Toplam gol beklentisi
        total_expected = home_xg + away_xg
        
        # Savunma kalitesi farkı
        defense_diff = abs(home_xga - away_xga)
        
        # Adaptif faktörler
        if total_expected > 3.0:  # Yüksek skorlu maç beklentisi
            attack_boost = 1.2
            defense_factor = 0.9
        elif total_expected < 2.0:  # Düşük skorlu maç beklentisi
            attack_boost = 0.8
            defense_factor = 1.1
        else:  # Normal maç
            attack_boost = 1.0
            defense_factor = 1.0
        
        # Savunma farkına göre ayarlama
        if defense_diff > 0.5:  # Büyük savunma farkı
            stronger_defense = min(home_xga, away_xga)
            if home_xga < away_xga:  # Ev sahibi daha iyi savunma
                defense_home_bonus = 1.1
                defense_away_penalty = 0.9
            else:  # Deplasman daha iyi savunma
                defense_home_bonus = 0.9
                defense_away_penalty = 1.1
        else:  # Küçük savunma farkı
            defense_home_bonus = 1.0
            defense_away_penalty = 1.0
        
        # Lambda hesaplama
        lambda_home = (home_xg * attack_boost * away_xga * defense_factor) * defense_home_bonus
        lambda_away = (away_xg * attack_boost * home_xga * defense_factor) * defense_away_penalty
        
        # Sınırları kontrol et
        lambda_home = max(0.5, min(4.0, lambda_home))
        lambda_away = max(0.5, min(4.0, lambda_away))
        
        logger.debug(f"Adaptif lambda - Ev: {lambda_home:.2f}, Deplasman: {lambda_away:.2f}")
        return lambda_home, lambda_away
    
    def ensemble_cross_lambda(self, home_xg, home_xga, away_xg, away_xga):
        """
        Ensemble Çapraz Lambda Formülü
        Birden fazla formülün ağırlıklı ortalaması
        """
        # 1. Üstel formül
        exp_lambda_h, exp_lambda_a = self.exponential_cross_lambda(home_xg, home_xga, away_xg, away_xga)
        
        # 2. Sigmoid formül
        sig_lambda_h, sig_lambda_a = self.sigmoid_cross_lambda(home_xg, home_xga, away_xg, away_xga)
        
        # 3. Adaptif formül
        adapt_lambda_h, adapt_lambda_a = self.adaptive_cross_lambda(home_xg, home_xga, away_xg, away_xga)
        
        # Ağırlıklı ortalama
        lambda_home = 0.4 * exp_lambda_h + 0.3 * sig_lambda_h + 0.3 * adapt_lambda_h
        lambda_away = 0.4 * exp_lambda_a + 0.3 * sig_lambda_a + 0.3 * adapt_lambda_a
        
        logger.info(f"Ensemble Lambda - Ev: {lambda_home:.2f}, Deplasman: {lambda_away:.2f}")
        logger.debug(f"Ensemble detay - Üstel: ({exp_lambda_h:.2f}, {exp_lambda_a:.2f}), "
                    f"Sigmoid: ({sig_lambda_h:.2f}, {sig_lambda_a:.2f}), "
                    f"Adaptif: ({adapt_lambda_h:.2f}, {adapt_lambda_a:.2f})")
        
        return lambda_home, lambda_away
        
    def calculate_lambda_cross(self, home_xg, home_xga, away_xg, away_xga, elo_diff=0):
        """
        Çapraz lambda hesaplama - Ensemble yaklaşımı ile Poisson için gol beklentileri
        
        Args:
            home_xg: Ev sahibi xG
            home_xga: Ev sahibi xGA  
            away_xg: Deplasman xG
            away_xga: Deplasman xGA
            elo_diff: Elo farkı (home - away)
            
        Returns:
            tuple: (lambda_home, lambda_away)
        """
        # Elo bazlı ön düzeltmeler (favoriler için)
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
        
        # Ensemble lambda hesaplama
        lambda_home, lambda_away = self.ensemble_cross_lambda(home_xg, home_xga, away_xg, away_xga)
        
        # Ek lambda düzeltmesi (elo bazlı - sadece anormal durumlarda)
        if elo_diff > 50 and lambda_home < lambda_away:
            adjustment = (lambda_away - lambda_home) * 0.2
            lambda_home += adjustment
            lambda_away -= adjustment * 0.5
            logger.info(f"Elo bazlı düzeltme uygulandı: Ev +{adjustment:.2f}, Dep -{adjustment*0.5:.2f}")
            
        # Lambda sınırları - Ekstrem maç kontrolü
        try:
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
        except ImportError:
            # Extreme detector yoksa varsayılan sınırlar
            lambda_home = max(0.5, min(4.0, lambda_home))
            lambda_away = max(0.5, min(4.0, lambda_away))
        
        logger.info(f"Final Lambda değerleri - Ev: {lambda_home:.2f}, Deplasman: {lambda_away:.2f}")
        return lambda_home, lambda_away