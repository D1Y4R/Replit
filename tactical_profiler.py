"""
Tactical Profiler - Takım Taktiksel Stil Analizi
Takımların oyun stilini, tempo tercihlerini ve taktiksel özelliklerini analiz eder
"""
import numpy as np
import logging
from collections import Counter

logger = logging.getLogger(__name__)

class TacticalProfiler:
    """
    Takım taktiksel profil analizi
    """
    
    def __init__(self):
        # Tempo kategorileri
        self.tempo_thresholds = {
            'very_fast': 3.0,    # 3+ gol/maç
            'fast': 2.5,         # 2.5-3 gol/maç
            'medium': 2.0,       # 2-2.5 gol/maç
            'slow': 1.5,         # 1.5-2 gol/maç
            'very_slow': 0       # <1.5 gol/maç
        }
        
        # Gol dakika aralıkları
        self.time_periods = {
            'early': (0, 30),
            'middle': (31, 60),
            'late': (61, 90)
        }
        
    def analyze_tactical_profile(self, team_matches, team_stats=None):
        """
        Takımın taktiksel profilini analiz et
        
        Args:
            team_matches: Takımın son maçları
            team_stats: Ek takım istatistikleri (opsiyonel)
            
        Returns:
            dict: Taktiksel profil analizi
        """
        if not team_matches:
            return self._get_default_profile()
            
        # Son 20 maçı al (daha geniş veri seti)
        recent_matches = sorted(team_matches, key=lambda x: x.get('date', ''), reverse=True)[:20]
        
        # Analizleri yap
        tempo_analysis = self._analyze_tempo(recent_matches)
        pressing_intensity = self._analyze_pressing_intensity(recent_matches)
        counter_attack_tendency = self._analyze_counter_tendency(recent_matches)
        set_piece_effectiveness = self._analyze_set_pieces(recent_matches)
        half_performance = self._analyze_half_performance(recent_matches)
        
        # Taktiksel stil belirleme
        tactical_style = self._determine_tactical_style(
            tempo_analysis,
            pressing_intensity,
            counter_attack_tendency
        )
        
        # Savunma sağlamlığı
        defensive_solidity = self._analyze_defensive_solidity(recent_matches)
        
        return {
            'style': tactical_style,
            'tempo': tempo_analysis['category'],
            'tempo_details': tempo_analysis,
            'pressing_intensity': pressing_intensity,
            'counter_attack_score': counter_attack_tendency,
            'set_piece_threat': set_piece_effectiveness,
            'defensive_solidity': defensive_solidity,
            'half_performance': half_performance,
            'matches_analyzed': len(recent_matches)
        }
        
    def _analyze_tempo(self, matches):
        """
        Maç temposunu analiz et
        """
        if not matches:
            return {'category': 'medium', 'avg_total_goals': 2.5}
            
        # Toplam gol ortalaması
        total_goals = []
        for match in matches:
            goals = match.get('goals_scored', 0) + match.get('goals_conceded', 0)
            total_goals.append(goals)
            
        avg_goals = np.mean(total_goals) if total_goals else 2.5
        
        # Tempo kategorisi
        category = 'medium'
        for cat, threshold in sorted(self.tempo_thresholds.items(), 
                                   key=lambda x: x[1], reverse=True):
            if avg_goals >= threshold:
                category = cat
                break
                
        return {
            'category': category,
            'avg_total_goals': round(avg_goals, 2),
            'high_scoring_ratio': len([g for g in total_goals if g > 3]) / len(total_goals)
        }
        
    def _analyze_pressing_intensity(self, matches):
        """
        Baskı yoğunluğunu analiz et (gol dakikalarından)
        """
        if not matches:
            return 'medium'
            
        early_goals = 0
        total_goals = 0
        
        for match in matches:
            # Gol dakikalarını kontrol et (eğer varsa)
            goal_minutes = match.get('goal_minutes', [])
            if goal_minutes:
                for minute in goal_minutes:
                    if minute <= 30:
                        early_goals += 1
                    total_goals += 1
            else:
                # Dakika bilgisi yoksa, gol sayısından tahmin
                goals = match.get('goals_scored', 0)
                if goals > 0:
                    early_goals += goals * 0.3  # Tahmini %30 erken gol
                total_goals += goals
                
        if total_goals == 0:
            return 'medium'
            
        early_ratio = early_goals / total_goals
        
        if early_ratio > 0.4:
            return 'high'
        elif early_ratio > 0.25:
            return 'medium'
        else:
            return 'low'
            
    def _analyze_counter_tendency(self, matches):
        """
        Kontra atak eğilimini analiz et
        """
        if not matches:
            return 50
            
        # Hızlı gol göstergeleri
        counter_indicators = {
            'low_possession_wins': 0,
            'quick_goals': 0,
            'away_wins': 0
        }
        
        for match in matches:
            # Deplasmanda galibiyet (kontra göstergesi)
            if match.get('venue') == 'away' and match.get('goals_scored', 0) > match.get('goals_conceded', 0):
                counter_indicators['away_wins'] += 1
                
            # Az gol yiyerek kazanma (savunma bazlı kontra)
            if (match.get('goals_scored', 0) > match.get('goals_conceded', 0) and 
                match.get('goals_conceded', 0) <= 1):
                counter_indicators['low_possession_wins'] += 1
                
        # Skor hesaplama (0-100)
        total_matches = len(matches)
        counter_score = (
            (counter_indicators['away_wins'] / total_matches) * 40 +
            (counter_indicators['low_possession_wins'] / total_matches) * 60
        ) * 100
        
        return min(100, max(0, counter_score))
        
    def _analyze_set_pieces(self, matches):
        """
        Set parça etkinliğini analiz et
        """
        if not matches:
            return 'medium'
            
        # Set parça göstergeleri (basitleştirilmiş)
        clean_sheets = sum(1 for m in matches if m.get('goals_conceded', 0) == 0)
        high_scoring = sum(1 for m in matches if m.get('goals_scored', 0) >= 2)
        
        clean_sheet_ratio = clean_sheets / len(matches)
        high_scoring_ratio = high_scoring / len(matches)
        
        # Set parça tehdit seviyesi
        if high_scoring_ratio > 0.5 and clean_sheet_ratio > 0.3:
            return 'high'
        elif high_scoring_ratio > 0.3 or clean_sheet_ratio > 0.4:
            return 'medium'
        else:
            return 'low'
            
    def _analyze_half_performance(self, matches):
        """
        İlk yarı vs ikinci yarı performansı
        """
        if not matches:
            return {
                'first_half_strength': 50,
                'second_half_strength': 50,
                'late_goal_tendency': 'medium'
            }
            
        # Basitleştirilmiş analiz (detaylı dakika verisi olmadan)
        late_wins = 0
        early_leads = 0
        
        for match in matches:
            # Yüksek skorlu maçlarda genelde geç goller olur
            total_goals = match.get('goals_scored', 0) + match.get('goals_conceded', 0)
            if total_goals >= 3:
                late_wins += 0.3  # Tahmini geç gol olasılığı
                
        late_goal_ratio = late_wins / len(matches)
        
        late_tendency = 'high' if late_goal_ratio > 0.4 else 'medium' if late_goal_ratio > 0.2 else 'low'
        
        return {
            'first_half_strength': 50,  # Detaylı veri olmadan dengeli
            'second_half_strength': 50 + (late_goal_ratio * 20),  # Geç gol eğilimi varsa artır
            'late_goal_tendency': late_tendency
        }
        
    def _determine_tactical_style(self, tempo, pressing, counter):
        """
        Taktiksel stili belirle
        """
        styles = []
        
        # Tempo bazlı
        if tempo['category'] in ['very_fast', 'fast']:
            styles.append('attacking')
        elif tempo['category'] in ['slow', 'very_slow']:
            styles.append('defensive')
            
        # Baskı bazlı
        if pressing == 'high':
            styles.append('high_press')
        elif pressing == 'low':
            styles.append('deep_block')
            
        # Kontra bazlı
        if counter > 60:
            styles.append('counter')
            
        # Stil kombinasyonu
        if 'attacking' in styles and 'high_press' in styles:
            return 'attacking_high_press'
        elif 'defensive' in styles and 'counter' in styles:
            return 'defensive_counter'
        elif 'attacking' in styles:
            return 'attacking_possession'
        elif 'defensive' in styles:
            return 'defensive_deep'
        else:
            return 'balanced'
            
    def _analyze_defensive_solidity(self, matches):
        """
        Savunma sağlamlığını analiz et
        """
        if not matches:
            return 'medium'
            
        # Savunma metrikleri
        clean_sheets = sum(1 for m in matches if m.get('goals_conceded', 0) == 0)
        low_conceding = sum(1 for m in matches if m.get('goals_conceded', 0) <= 1)
        avg_conceded = np.mean([m.get('goals_conceded', 0) for m in matches])
        
        clean_sheet_ratio = clean_sheets / len(matches)
        low_conceding_ratio = low_conceding / len(matches)
        
        # Sağlamlık seviyesi
        if clean_sheet_ratio > 0.4 or avg_conceded < 1.0:
            return 'very_high'
        elif clean_sheet_ratio > 0.25 or avg_conceded < 1.3:
            return 'high'
        elif avg_conceded < 1.7:
            return 'medium'
        elif avg_conceded < 2.0:
            return 'low'
        else:
            return 'very_low'
            
    def _get_default_profile(self):
        """
        Varsayılan taktiksel profil
        """
        return {
            'style': 'balanced',
            'tempo': 'medium',
            'tempo_details': {'category': 'medium', 'avg_total_goals': 2.5},
            'pressing_intensity': 'medium',
            'counter_attack_score': 50,
            'set_piece_threat': 'medium',
            'defensive_solidity': 'medium',
            'half_performance': {
                'first_half_strength': 50,
                'second_half_strength': 50,
                'late_goal_tendency': 'medium'
            },
            'matches_analyzed': 0
        }