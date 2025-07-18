"""
İY/MS Sürpriz Tespit Modülü
Sadece ilk yarı/maç sonu tahminleri için gelişmiş analiz
"""
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class HTFTSurpriseDetector:
    """
    İlk yarı/maç sonu sürpriz sonuçlarını tespit eden özel modül
    """
    
    def __init__(self):
        self.surprise_patterns = {
            'HOME_AWAY': 0.05,  # İY: Ev, MS: Deplasman (en sürpriz)
            'AWAY_HOME': 0.06,  # İY: Deplasman, MS: Ev
            'HOME_DRAW': 0.12,  # İY: Ev, MS: Beraberlik
            'DRAW_HOME': 0.10,  # İY: Beraberlik, MS: Ev
            'DRAW_AWAY': 0.10,  # İY: Beraberlik, MS: Deplasman
            'AWAY_DRAW': 0.08   # İY: Deplasman, MS: Beraberlik
        }
        
    def analyze_surprise_potential(self, home_data, away_data, elo_diff):
        """
        Maçın sürpriz potansiyelini analiz et
        
        Returns:
            dict: Sürpriz skorları ve göstergeleri
        """
        surprise_indicators = {
            'momentum_reversal': self._check_momentum_reversal(home_data, away_data),
            'fatigue_factor': self._analyze_fatigue_impact(home_data, away_data),
            'psychological_pressure': self._check_psychological_factors(home_data, away_data, elo_diff),
            'tactical_adaptation': self._analyze_tactical_changes(home_data, away_data),
            'h2h_surprises': self._check_historical_surprises(home_data, away_data),
            'second_half_specialist': self._identify_second_half_patterns(home_data, away_data),
            'comeback_ability': self._analyze_comeback_potential(home_data, away_data),
            'pressure_handling': self._check_pressure_situations(home_data, away_data)
        }
        
        # Toplam sürpriz skoru
        total_score = sum(surprise_indicators.values()) / len(surprise_indicators)
        
        return {
            'surprise_score': total_score,
            'indicators': surprise_indicators,
            'high_surprise': total_score > 0.6,
            'recommended_adjustments': self._get_adjustments(surprise_indicators)
        }
    
    def _check_momentum_reversal(self, home_data, away_data):
        """
        Momentum tersine dönme potansiyeli
        """
        score = 0.0
        
        # Ev sahibi ilk yarı iyi ama ikinci yarı kötü performans
        home_ht_performance = self._calculate_halftime_performance(home_data)
        home_ft_performance = self._calculate_fulltime_performance(home_data)
        
        if home_ht_performance > 0.6 and home_ft_performance < 0.4:
            score += 0.3
            
        # Deplasman takımı ikinci yarı güçlü
        away_second_half = self._analyze_second_half_strength(away_data)
        if away_second_half > 0.65:
            score += 0.4
            
        # Son maçlarda momentum değişimi
        recent_reversals = self._count_recent_reversals(home_data, away_data)
        score += min(0.3, recent_reversals * 0.1)
        
        return min(1.0, score)
    
    def _analyze_fatigue_impact(self, home_data, away_data):
        """
        Yorgunluk faktörünün sürpriz etkisi
        """
        fatigue_score = 0.0
        
        # Ev sahibi yorgunluğu
        home_matches = home_data.get('recent_matches', [])
        home_fatigue = self._calculate_match_congestion(home_matches)
        
        # Deplasman dinlenme avantajı
        away_matches = away_data.get('recent_matches', [])
        away_rest = self._calculate_rest_advantage(away_matches)
        
        # Yorgunluk farkı sürpriz yaratabilir
        if home_fatigue > 0.7 and away_rest > 0.6:
            fatigue_score = 0.8
        elif home_fatigue > 0.5:
            fatigue_score = 0.5
            
        return fatigue_score
    
    def _check_psychological_factors(self, home_data, away_data, elo_diff):
        """
        Psikolojik faktörlerin sürpriz etkisi
        """
        psych_score = 0.0
        
        # Favori takım baskısı
        if elo_diff > 150:  # Ev sahibi büyük favori
            # Baskı altında çökme potansiyeli
            home_pressure_handling = self._analyze_pressure_performance(home_data)
            if home_pressure_handling < 0.4:
                psych_score += 0.4
                
        # Deplasman takımı özgüveni
        away_confidence = self._calculate_team_confidence(away_data)
        if away_confidence > 0.7:
            psych_score += 0.3
            
        # Kritik maç deneyimi
        if self._has_big_match_experience(away_data):
            psych_score += 0.3
            
        return min(1.0, psych_score)
    
    def _analyze_tactical_changes(self, home_data, away_data):
        """
        İkinci yarı taktiksel değişim potansiyeli
        """
        tactical_score = 0.0
        
        # İlk yarı defansif, ikinci yarı ofansif pattern
        away_tactical = self._check_tactical_patterns(away_data)
        if away_tactical.get('defensive_first_half', 0) > 0.6:
            tactical_score += 0.5
            
        # Ev sahibi ikinci yarı adaptasyon zorluğu
        home_adaptation = self._check_adaptation_ability(home_data)
        if home_adaptation < 0.4:
            tactical_score += 0.5
            
        return min(1.0, tactical_score)
    
    def _check_historical_surprises(self, home_data, away_data):
        """
        Geçmiş H2H sürpriz sonuçları
        """
        h2h_matches = home_data.get('h2h_matches', [])
        surprise_count = 0
        
        for match in h2h_matches[-10:]:  # Son 10 H2H maç
            if self._was_surprise_result(match):
                surprise_count += 1
                
        return min(1.0, surprise_count * 0.2)
    
    def _identify_second_half_patterns(self, home_data, away_data):
        """
        İkinci yarı uzmanı takımları tespit et
        """
        # Deplasman takımı ikinci yarı performansı
        away_sh_ratio = self._calculate_second_half_goal_ratio(away_data)
        
        # Ev sahibi ikinci yarı zayıflığı
        home_sh_weakness = 1.0 - self._calculate_second_half_goal_ratio(home_data)
        
        return (away_sh_ratio + home_sh_weakness) / 2
    
    def _analyze_comeback_potential(self, home_data, away_data):
        """
        Geri dönüş potansiyeli analizi
        """
        comeback_score = 0.0
        
        # Deplasman takımı geri dönüş yeteneği
        away_comebacks = self._count_comebacks(away_data)
        comeback_score += min(0.5, away_comebacks * 0.1)
        
        # Ev sahibi önde iken kazanamama oranı
        home_blown_leads = self._count_blown_leads(home_data)
        comeback_score += min(0.5, home_blown_leads * 0.15)
        
        return comeback_score
    
    def _check_pressure_situations(self, home_data, away_data):
        """
        Baskı durumlarında performans
        """
        # Kritik dakikalarda (75+) performans
        home_late_goals = self._analyze_late_game_performance(home_data, 'conceded')
        away_late_goals = self._analyze_late_game_performance(away_data, 'scored')
        
        pressure_score = 0.0
        if home_late_goals > 0.3:  # Ev sahibi geç goller yiyor
            pressure_score += 0.5
        if away_late_goals > 0.3:  # Deplasman geç goller atıyor
            pressure_score += 0.5
            
        return min(1.0, pressure_score)
    
    def adjust_htft_probabilities(self, base_probs, surprise_analysis):
        """
        Sürpriz analizine göre İY/MS olasılıklarını ayarla
        
        Args:
            base_probs: Temel İY/MS olasılıkları
            surprise_analysis: Sürpriz analiz sonuçları
            
        Returns:
            dict: Ayarlanmış olasılıklar
        """
        adjusted_probs = base_probs.copy()
        
        if surprise_analysis['high_surprise']:
            # Sürpriz potansiyeli yüksekse
            adjustments = surprise_analysis['recommended_adjustments']
            
            for pattern, adjustment in adjustments.items():
                if pattern in adjusted_probs:
                    # Sürpriz sonuçları artır
                    adjusted_probs[pattern] *= (1 + adjustment)
                    
            # Normal sonuçları azalt
            if 'HOME_HOME' in adjusted_probs:
                adjusted_probs['HOME_HOME'] *= 0.85
            if 'AWAY_AWAY' in adjusted_probs:
                adjusted_probs['AWAY_AWAY'] *= 0.9
                
        # Normalize et
        total = sum(adjusted_probs.values())
        if total > 0:
            adjusted_probs = {k: (v/total) * 100 for k, v in adjusted_probs.items()}
            
        return adjusted_probs
    
    def _get_adjustments(self, indicators):
        """
        Göstergelere göre önerilen ayarlamalar
        """
        adjustments = {}
        
        # Momentum tersine dönme yüksekse
        if indicators['momentum_reversal'] > 0.7:
            adjustments['HOME_AWAY'] = 0.5
            adjustments['HOME_DRAW'] = 0.3
            
        # Yorgunluk faktörü yüksekse
        if indicators['fatigue_factor'] > 0.6:
            adjustments['DRAW_AWAY'] = 0.4
            adjustments['HOME_AWAY'] = 0.3
            
        # Psikolojik baskı yüksekse
        if indicators['psychological_pressure'] > 0.6:
            adjustments['HOME_DRAW'] = 0.4
            adjustments['DRAW_AWAY'] = 0.3
            
        # İkinci yarı uzmanı varsa
        if indicators['second_half_specialist'] > 0.7:
            adjustments['AWAY_HOME'] = 0.4
            adjustments['DRAW_HOME'] = 0.3
            
        return adjustments
    
    # Yardımcı metodlar
    def _calculate_halftime_performance(self, team_data):
        """İlk yarı performans skoru"""
        matches = team_data.get('recent_matches', [])
        if not matches:
            return 0.5
            
        ht_wins = 0
        valid_matches = 0
        
        for match in matches[:20]:
            if 'halftime_result' in match:
                valid_matches += 1
                if match['halftime_result'] == 'win':
                    ht_wins += 1
                    
        return ht_wins / valid_matches if valid_matches > 0 else 0.5
    
    def _calculate_fulltime_performance(self, team_data):
        """Tam maç performans skoru"""
        matches = team_data.get('recent_matches', [])
        if not matches:
            return 0.5
            
        wins = sum(1 for m in matches[:20] if m.get('result') == 'win')
        return wins / min(20, len(matches))
    
    def _analyze_second_half_strength(self, team_data):
        """İkinci yarı güç analizi"""
        matches = team_data.get('recent_matches', [])
        second_half_goals = 0
        total_goals = 0
        
        for match in matches[:20]:
            total = match.get('goals_scored', 0)
            first_half = match.get('first_half_goals', 0)
            second_half_goals += (total - first_half)
            total_goals += total
            
        return second_half_goals / total_goals if total_goals > 0 else 0.5
    
    def _count_recent_reversals(self, home_data, away_data):
        """Son maçlarda tersine dönme sayısı"""
        reversals = 0
        
        for data in [home_data, away_data]:
            matches = data.get('recent_matches', [])
            for match in matches[:10]:
                if self._is_reversal(match):
                    reversals += 1
                    
        return reversals
    
    def _is_reversal(self, match):
        """Maçta tersine dönme olmuş mu?"""
        ht_result = match.get('halftime_result')
        ft_result = match.get('fulltime_result')
        
        if ht_result and ft_result:
            return (ht_result == 'win' and ft_result != 'win') or \
                   (ht_result == 'lose' and ft_result == 'win')
        return False
    
    def _calculate_match_congestion(self, matches):
        """Maç yoğunluğu hesapla"""
        if len(matches) < 3:
            return 0.0
            
        # Son 10 günde kaç maç oynanmış
        recent_count = 0
        for match in matches[:5]:
            if 'date' in match:
                try:
                    match_date = datetime.strptime(match['date'], '%Y-%m-%d')
                    if (datetime.now() - match_date).days <= 10:
                        recent_count += 1
                except:
                    pass
                    
        return min(1.0, recent_count * 0.3)
    
    def _calculate_rest_advantage(self, matches):
        """Dinlenme avantajı hesapla"""
        if not matches or 'date' not in matches[0]:
            return 0.5
            
        try:
            last_match = datetime.strptime(matches[0]['date'], '%Y-%m-%d')
            days_rest = (datetime.now() - last_match).days
            
            # 4-7 gün optimal dinlenme
            if 4 <= days_rest <= 7:
                return 1.0
            elif days_rest > 7:
                return 0.7
            else:
                return max(0, days_rest * 0.25)
        except:
            return 0.5
    
    def _analyze_pressure_performance(self, team_data):
        """Baskı altında performans"""
        # Büyük maçlarda performans, lider takımlara karşı sonuçlar vb.
        # Basitleştirilmiş versiyon
        return 0.5
    
    def _calculate_team_confidence(self, team_data):
        """Takım özgüven seviyesi"""
        matches = team_data.get('recent_matches', [])
        if not matches:
            return 0.5
            
        # Son 5 maçta galibiyet oranı
        recent_wins = sum(1 for m in matches[:5] if m.get('result') == 'win')
        return recent_wins / 5
    
    def _has_big_match_experience(self, team_data):
        """Büyük maç deneyimi var mı?"""
        # Basitleştirilmiş: ligdeki pozisyona göre
        return team_data.get('league_position', 10) <= 6
    
    def _check_tactical_patterns(self, team_data):
        """Taktiksel kalıpları kontrol et"""
        return {
            'defensive_first_half': 0.5,  # Basitleştirilmiş
            'offensive_second_half': 0.5
        }
    
    def _check_adaptation_ability(self, team_data):
        """Adaptasyon yeteneği"""
        return 0.5  # Basitleştirilmiş
    
    def _was_surprise_result(self, match):
        """Sürpriz sonuç mu?"""
        # H2H'da beklenmedik sonuç
        return False  # Basitleştirilmiş
    
    def _calculate_second_half_goal_ratio(self, team_data):
        """İkinci yarı gol oranı"""
        matches = team_data.get('recent_matches', [])
        if not matches:
            return 0.5
            
        total_goals = 0
        second_half_goals = 0
        
        for match in matches[:20]:
            total = match.get('goals_scored', 0)
            first_half = match.get('first_half_goals', total * 0.4)  # Tahmin
            second_half_goals += (total - first_half)
            total_goals += total
            
        return second_half_goals / total_goals if total_goals > 0 else 0.5
    
    def _count_comebacks(self, team_data):
        """Geri dönüş sayısı"""
        return sum(1 for m in team_data.get('recent_matches', [])[:20] 
                  if m.get('comeback', False))
    
    def _count_blown_leads(self, team_data):
        """Kaçırılan galibiyetler"""
        return sum(1 for m in team_data.get('recent_matches', [])[:20] 
                  if m.get('blown_lead', False))
    
    def _analyze_late_game_performance(self, team_data, goal_type):
        """Geç dakika performansı"""
        matches = team_data.get('recent_matches', [])
        if not matches:
            return 0.0
            
        late_goals = 0
        total_goals = 0
        
        for match in matches[:20]:
            if goal_type == 'scored':
                goals = match.get('goals_scored', 0)
                late = match.get('late_goals_scored', 0)
            else:
                goals = match.get('goals_conceded', 0)
                late = match.get('late_goals_conceded', 0)
                
            total_goals += goals
            late_goals += late
            
        return late_goals / total_goals if total_goals > 0 else 0.0