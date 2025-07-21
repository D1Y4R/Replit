"""
İlk Yarı / Maç Sonu (HT/FT) Tahmin Algoritması
LSTM + Poisson kombinasyonu ile momentum bazlı tahmin
Sürpriz tespit modülü entegrasyonu ile geliştirilmiş
"""
import numpy as np
from scipy.stats import poisson
import logging
from .htft_surprise_detector import HTFTSurpriseDetector

logger = logging.getLogger(__name__)

class HalfTimeFullTimePredictor:
    """
    İlk yarı ve maç sonu kombinasyonlarını tahmin eder
    """
    
    def __init__(self):
        self.combinations = [
            'HOME_HOME',   # İY: Ev, MS: Ev
            'HOME_DRAW',   # İY: Ev, MS: Beraberlik
            'HOME_AWAY',   # İY: Ev, MS: Deplasman
            'DRAW_HOME',   # İY: Beraberlik, MS: Ev
            'DRAW_DRAW',   # İY: Beraberlik, MS: Beraberlik
            'DRAW_AWAY',   # İY: Beraberlik, MS: Deplasman
            'AWAY_HOME',   # İY: Deplasman, MS: Ev
            'AWAY_DRAW',   # İY: Deplasman, MS: Beraberlik
            'AWAY_AWAY'    # İY: Deplasman, MS: Deplasman
        ]
        
        # Sürpriz tespit modülünü başlat
        try:
            self.surprise_detector = HTFTSurpriseDetector()
            self.use_surprise_detection = True
            logger.info("İY/MS sürpriz tespit modülü başlatıldı")
        except Exception as e:
            logger.warning(f"Sürpriz tespit modülü başlatılamadı: {e}")
            self.use_surprise_detection = False
        
    def predict_htft(self, home_team_data, away_team_data, lambda_home, lambda_away, elo_diff):
        """
        İlk yarı / Maç sonu tahminlerini hesapla
        
        Args:
            home_team_data: Ev sahibi takım verileri
            away_team_data: Deplasman takım verileri
            lambda_home: Ev sahibi gol beklentisi
            lambda_away: Deplasman gol beklentisi
            elo_diff: Takımlar arası Elo farkı
            
        Returns:
            dict: HT/FT kombinasyon olasılıkları
        """
        try:
            # İlk yarı lambda değerleri - takım özelliklerine göre dinamik
            home_ht_ratio = self._calculate_halftime_goal_ratio(home_team_data)
            away_ht_ratio = self._calculate_halftime_goal_ratio(away_team_data)
            
            ht_lambda_home = lambda_home * home_ht_ratio
            ht_lambda_away = lambda_away * away_ht_ratio
            
            # İlk yarı performans analizinden momentum faktörü
            home_ht_momentum = self._calculate_halftime_momentum(home_team_data)
            away_ht_momentum = self._calculate_halftime_momentum(away_team_data)
            
            # İlk yarı lambdalarını momentum ile ayarla
            ht_lambda_home = max(0.1, ht_lambda_home * home_ht_momentum)
            ht_lambda_away = max(0.1, ht_lambda_away * away_ht_momentum)
            
            # İlk yarı olasılıkları
            ht_probs = self._calculate_match_probabilities(ht_lambda_home, ht_lambda_away)
            
            # Tam maç olasılıkları
            ft_probs = self._calculate_match_probabilities(lambda_home, lambda_away)
            
            # Koşullu olasılıklar (Momentum etkisi)
            htft_probs = {}
            
            # Her kombinasyon için hesapla
            htft_probs['HOME_HOME'] = ht_probs['home'] * self._conditional_prob('HOME', 'HOME', ft_probs, elo_diff)
            htft_probs['HOME_DRAW'] = ht_probs['home'] * self._conditional_prob('HOME', 'DRAW', ft_probs, elo_diff)
            htft_probs['HOME_AWAY'] = ht_probs['home'] * self._conditional_prob('HOME', 'AWAY', ft_probs, elo_diff)
            
            htft_probs['DRAW_HOME'] = ht_probs['draw'] * self._conditional_prob('DRAW', 'HOME', ft_probs, elo_diff)
            htft_probs['DRAW_DRAW'] = ht_probs['draw'] * self._conditional_prob('DRAW', 'DRAW', ft_probs, elo_diff)
            htft_probs['DRAW_AWAY'] = ht_probs['draw'] * self._conditional_prob('DRAW', 'AWAY', ft_probs, elo_diff)
            
            htft_probs['AWAY_HOME'] = ht_probs['away'] * self._conditional_prob('AWAY', 'HOME', ft_probs, elo_diff)
            htft_probs['AWAY_DRAW'] = ht_probs['away'] * self._conditional_prob('AWAY', 'DRAW', ft_probs, elo_diff)
            htft_probs['AWAY_AWAY'] = ht_probs['away'] * self._conditional_prob('AWAY', 'AWAY', ft_probs, elo_diff)
            
            # Sürpriz tespit modülünü kullan (sadece İY/MS için)
            if self.use_surprise_detection:
                try:
                    # Sürpriz analizi yap
                    surprise_analysis = self.surprise_detector.analyze_surprise_potential(
                        home_team_data, away_team_data, elo_diff
                    )
                    
                    # Sürpriz potansiyeli yüksekse olasılıkları ayarla
                    if surprise_analysis['high_surprise']:
                        logger.info(f"İY/MS sürpriz potansiyeli tespit edildi: {surprise_analysis['surprise_score']:.2f}")
                        # Sürpriz analizine göre olasılıkları güncelle
                        htft_probs = self.surprise_detector.adjust_htft_probabilities(
                            htft_probs, surprise_analysis
                        )
                except Exception as e:
                    logger.warning(f"Sürpriz analizi hatası: {e}")
            
            # Normalize et
            total = sum(htft_probs.values())
            if total > 0:
                htft_probs = {k: (v/total) * 100 for k, v in htft_probs.items()}
            
            # En olası kombinasyonu bul
            most_likely = max(htft_probs, key=htft_probs.get)
            
            return {
                'predictions': htft_probs,
                'most_likely': most_likely,
                'most_likely_prob': htft_probs[most_likely],
                'halftime_probs': ht_probs
            }
            
        except Exception as e:
            logger.error(f"HT/FT tahmin hatası: {e}")
            return self._get_default_htft()
    
    def _calculate_halftime_goal_ratio(self, team_data):
        """
        Takımın ilk yarı gol oranını hesapla (toplam gollerin yüzde kaçı ilk yarıda)
        """
        # Varsayılan oran %40
        default_ratio = 0.4
        
        if 'recent_matches' not in team_data or not team_data['recent_matches']:
            return default_ratio
            
        total_goals = 0
        first_half_goals = 0
        match_count = 0
        
        for match in team_data['recent_matches']:  # Tüm mevcut maçlar
            if isinstance(match, dict):
                # Toplam goller
                total_goals += match.get('goals_scored', 0)
                # İlk yarı golleri (eğer varsa)
                if 'first_half_goals' in match:
                    first_half_goals += match['first_half_goals']
                    match_count += 1
                elif 'half_time_score' in match:
                    # Alternatif veri yapısı
                    ht_score = match['half_time_score']
                    if isinstance(ht_score, dict):
                        first_half_goals += ht_score.get('home', 0) if 'is_home' in match and match['is_home'] else ht_score.get('away', 0)
                        match_count += 1
        
        # Yeterli veri yoksa varsayılan değer
        if match_count < 5 or total_goals == 0:
            return default_ratio
            
        # İlk yarı oranını hesapla
        ratio = first_half_goals / total_goals if total_goals > 0 else default_ratio
        
        # Mantıklı sınırlar içinde tut (%25-%60 arası)
        return max(0.25, min(0.60, ratio))
    
    def _calculate_halftime_momentum(self, team_data):
        """
        Takımın ilk yarı performans momentumunu hesapla
        """
        momentum = 1.0
        
        if 'recent_matches' not in team_data:
            return momentum
            
        recent_matches = team_data['recent_matches']  # Tüm mevcut maçlar
        
        if not recent_matches:
            return momentum
            
        # İlk yarı verilerini analiz et
        ht_goals_for = []
        ht_goals_against = []
        
        for match in recent_matches:
            # API'den ilk yarı verileri varsa kullan
            if 'halftime_score' in match:
                ht_goals_for.append(match['halftime_score'].get('for', 0))
                ht_goals_against.append(match['halftime_score'].get('against', 0))
            else:
                # Yoksa tam maç skorunun %40'ını tahmin et
                total_for = match.get('goals_scored', 0)
                total_against = match.get('goals_conceded', 0)
                ht_goals_for.append(int(total_for * 0.4))
                ht_goals_against.append(int(total_against * 0.4))
        
        if ht_goals_for:
            avg_ht_goals = sum(ht_goals_for) / len(ht_goals_for)
            # Ortalama 0.5 gol = momentum 1.0
            momentum = 1.0 + (avg_ht_goals - 0.5) * 0.3
            
        return max(0.5, min(1.5, momentum))  # 0.5 ile 1.5 arasında sınırla
    
    def _calculate_match_probabilities(self, lambda_home, lambda_away):
        """
        Poisson dağılımı ile maç olasılıklarını hesapla
        """
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        
        for h in range(6):  # Max 5 gol
            for a in range(6):
                prob = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
                
                if h > a:
                    home_win += prob
                elif h == a:
                    draw += prob
                else:
                    away_win += prob
                    
        return {
            'home': home_win,
            'draw': draw,
            'away': away_win
        }
    
    def _conditional_prob(self, ht_result, ft_result, ft_probs, elo_diff):
        """
        İlk yarı sonucuna göre maç sonu koşullu olasılığı
        """
        base_prob = ft_probs[ft_result.lower()]
        
        # Momentum faktörleri
        if ht_result == ft_result:
            # Aynı sonuç devam ediyor (momentum korunuyor)
            momentum_factor = 1.3
        elif ht_result == 'DRAW':
            # Beraberlikten sonra her şey olabilir
            momentum_factor = 1.0
        elif (ht_result == 'HOME' and ft_result == 'AWAY') or \
             (ht_result == 'AWAY' and ft_result == 'HOME'):
            # Büyük geri dönüş (daha az olası)
            momentum_factor = 0.6
            # Elo farkı büyükse daha da az olası
            if abs(elo_diff) > 200:
                momentum_factor *= 0.8
        else:
            momentum_factor = 0.9
            
        return base_prob * momentum_factor
    
    def _get_default_htft(self):
        """
        Varsayılan HT/FT tahminleri
        """
        return {
            'predictions': {
                'HOME_HOME': 25.0,
                'HOME_DRAW': 8.0,
                'HOME_AWAY': 5.0,
                'DRAW_HOME': 10.0,
                'DRAW_DRAW': 15.0,
                'DRAW_AWAY': 10.0,
                'AWAY_HOME': 5.0,
                'AWAY_DRAW': 8.0,
                'AWAY_AWAY': 14.0
            },
            'most_likely': 'HOME_HOME',
            'most_likely_prob': 25.0,
            'halftime_probs': {'home': 40.0, 'draw': 30.0, 'away': 30.0}
        }
    
    def predict_halftime_goals(self, home_team_data, away_team_data, lambda_home, lambda_away):
        """
        İlk yarı gol tahminleri (0.5, 1.5 Alt/Üst)
        """
        # İlk yarı lambda değerleri
        ht_lambda_home = lambda_home * 0.4
        ht_lambda_away = lambda_away * 0.4
        
        # Momentum faktörleri
        home_momentum = self._calculate_halftime_momentum(home_team_data)
        away_momentum = self._calculate_halftime_momentum(away_team_data)
        
        ht_lambda_home *= home_momentum
        ht_lambda_away *= away_momentum
        
        total_ht_lambda = ht_lambda_home + ht_lambda_away
        
        # Poisson olasılıkları
        over_0_5 = 1 - poisson.pmf(0, total_ht_lambda)
        over_1_5 = 1 - poisson.pmf(0, total_ht_lambda) - poisson.pmf(1, total_ht_lambda)
        
        return {
            'over_0_5': over_0_5 * 100,
            'under_0_5': (1 - over_0_5) * 100,
            'over_1_5': over_1_5 * 100,
            'under_1_5': (1 - over_1_5) * 100,
            'expected_ht_goals': round(total_ht_lambda, 2)
        }