"""
Gelişmiş Özellik Mühendisliği
Form momentum, takım etkileşimleri ve psikolojik faktörler
"""
import numpy as np
import logging
from datetime import datetime, timedelta
from scipy import stats

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """
    Gelişmiş özellik çıkarımı ve analizi
    """
    
    def __init__(self):
        self.feature_weights = {
            'form_momentum': 0.15,
            'h2h_performance': 0.10,
            'psychological_factors': 0.08,
            'match_context': 0.12,
            'team_consistency': 0.10,
            'goal_trends': 0.10,
            'defensive_stability': 0.10,
            'attacking_efficiency': 0.10,
            'pressure_handling': 0.08,
            'recent_results_impact': 0.07
        }
        
    def extract_all_features(self, home_data, away_data, match_context):
        """
        Tüm gelişmiş özellikleri çıkar
        
        Returns:
            dict: Hesaplanan tüm özellikler
        """
        features = {
            'form_momentum': self.calculate_form_momentum(home_data, away_data),
            'h2h_analysis': self.analyze_head_to_head(home_data, away_data),
            'psychological': self.analyze_psychological_factors(home_data, away_data, match_context),
            'advanced_context': self.analyze_match_context(match_context),
            'team_patterns': self.analyze_team_patterns(home_data, away_data),
            'goal_dynamics': self.analyze_goal_dynamics(home_data, away_data),
            'tactical_matchup': self.analyze_tactical_matchup(home_data, away_data)
        }
        
        return features
        
    def calculate_form_momentum(self, home_data, away_data):
        """
        Gelişmiş form ve momentum analizi
        """
        home_momentum = self._calculate_team_momentum(home_data.get('recent_matches', []))
        away_momentum = self._calculate_team_momentum(away_data.get('recent_matches', []))
        
        return {
            'home': home_momentum,
            'away': away_momentum,
            'differential': home_momentum['composite_score'] - away_momentum['composite_score'],
            'momentum_shift': self._detect_momentum_shift(home_data, away_data)
        }
        
    def _calculate_team_momentum(self, matches):
        """
        Takım momentum hesaplama - çok boyutlu analiz
        """
        if not matches:
            return self._get_default_momentum()
            
        recent_matches = matches[:10]  # Son 10 maç
        
        # 1. Gol trendi analizi
        goals_scored = [m.get('goals_scored', 0) for m in recent_matches]
        goals_conceded = [m.get('goals_conceded', 0) for m in recent_matches]
        
        # Linear regression ile trend
        if len(goals_scored) >= 3:
            x = np.arange(len(goals_scored))
            goal_trend_slope, _ = np.polyfit(x, goals_scored, 1)
            defense_trend_slope, _ = np.polyfit(x, goals_conceded, 1)
        else:
            goal_trend_slope = 0
            defense_trend_slope = 0
            
        # 2. Form puanı (ağırlıklı)
        form_points = []
        weights = np.exp(-0.2 * np.arange(len(recent_matches)))  # Exponential decay
        
        for i, match in enumerate(recent_matches):
            goals_for = match.get('goals_scored', 0)
            goals_against = match.get('goals_conceded', 0)
            
            if goals_for > goals_against:
                points = 3
            elif goals_for == goals_against:
                points = 1
            else:
                points = 0
                
            # Gol farkı bonusu
            goal_diff = goals_for - goals_against
            if goal_diff > 2:
                points += 0.5
            elif goal_diff < -2:
                points -= 0.5
                
            form_points.append(points * weights[i])
            
        weighted_form = sum(form_points) / sum(weights) if weights.sum() > 0 else 0
        
        # 3. Tutarlılık analizi
        if goals_scored:
            scoring_consistency = 1 / (1 + np.std(goals_scored))
            defensive_consistency = 1 / (1 + np.std(goals_conceded))
        else:
            scoring_consistency = 0.5
            defensive_consistency = 0.5
            
        # 4. Momentum değişimi (son 3 maç vs önceki 3 maç)
        if len(recent_matches) >= 6:
            recent_3 = sum(1 for m in recent_matches[:3] 
                          if m.get('goals_scored', 0) > m.get('goals_conceded', 0))
            previous_3 = sum(1 for m in recent_matches[3:6] 
                           if m.get('goals_scored', 0) > m.get('goals_conceded', 0))
            momentum_change = (recent_3 - previous_3) / 3
        else:
            momentum_change = 0
            
        # 5. Rakip gücüne göre düzeltilmiş performans
        opponent_strength_factor = self._calculate_opponent_strength_factor(recent_matches)
        
        # Kompozit momentum skoru
        composite_score = (
            weighted_form * 0.30 +
            (goal_trend_slope + 1) * 0.20 +
            (1 - defense_trend_slope) * 0.15 +
            scoring_consistency * 0.15 +
            defensive_consistency * 0.10 +
            (momentum_change + 1) * 0.10
        ) * opponent_strength_factor
        
        return {
            'composite_score': composite_score,
            'form_score': weighted_form,
            'goal_trend': goal_trend_slope,
            'defense_trend': defense_trend_slope,
            'consistency': (scoring_consistency + defensive_consistency) / 2,
            'momentum_change': momentum_change,
            'recent_ppg': weighted_form,  # Points per game
            'strength_adjusted': opponent_strength_factor
        }
        
    def _get_default_momentum(self):
        """Varsayılan momentum değerleri"""
        return {
            'composite_score': 2.0,
            'form_score': 1.5,
            'goal_trend': 0.0,
            'defense_trend': 0.0,
            'consistency': 0.5,
            'momentum_change': 0.0,
            'recent_ppg': 1.5,
            'strength_adjusted': 1.0
        }
        
    def _calculate_opponent_strength_factor(self, matches):
        """Rakip gücüne göre performans düzeltmesi"""
        # Basitleştirilmiş versiyon - gerçekte rakip Elo/rating kullanılmalı
        strong_opponent_wins = 0
        weak_opponent_losses = 0
        
        for match in matches[:5]:
            # Rakip gücü tahmini (gol sayısına göre basit tahmin)
            opponent_strength = match.get('opponent_goals_avg', 1.3)
            goals_for = match.get('goals_scored', 0)
            goals_against = match.get('goals_conceded', 0)
            
            if opponent_strength > 1.5 and goals_for > goals_against:
                strong_opponent_wins += 1
            elif opponent_strength < 1.0 and goals_for < goals_against:
                weak_opponent_losses += 1
                
        # Güçlü rakiplere karşı galibiyet bonus, zayıflara karşı mağlubiyet ceza
        factor = 1.0 + (strong_opponent_wins * 0.1) - (weak_opponent_losses * 0.15)
        return max(0.7, min(1.3, factor))
        
    def _detect_momentum_shift(self, home_data, away_data):
        """Momentum değişimini tespit et"""
        home_recent = home_data.get('recent_matches', [])[:3]
        away_recent = away_data.get('recent_matches', [])[:3]
        
        # Son 3 maçtaki performans değişimi
        home_shift = self._calculate_performance_shift(home_recent)
        away_shift = self._calculate_performance_shift(away_recent)
        
        return {
            'home_trending': home_shift,
            'away_trending': away_shift,
            'advantage': 'home' if home_shift > away_shift else 'away' if away_shift > home_shift else 'neutral'
        }
        
    def _calculate_performance_shift(self, recent_matches):
        """Performans değişim oranı"""
        if len(recent_matches) < 2:
            return 0.0
            
        recent_score = sum(m.get('goals_scored', 0) - m.get('goals_conceded', 0) 
                          for m in recent_matches[:2]) / 2
        return recent_score
        
    def analyze_head_to_head(self, home_data, away_data):
        """Head-to-head analizi"""
        # H2H verisi yoksa varsayılan değerler
        h2h_matches = home_data.get('h2h_matches', [])
        
        if not h2h_matches:
            return {
                'historical_advantage': 'neutral',
                'avg_goals': {'home': 1.5, 'away': 1.2},
                'win_rate': {'home': 0.33, 'draw': 0.33, 'away': 0.33},
                'recent_trend': 'no_data',
                'psychological_edge': 0.0
            }
            
        # Son 10 H2H maçı analiz et
        recent_h2h = h2h_matches[:10]
        home_wins = sum(1 for m in recent_h2h if m.get('home_goals', 0) > m.get('away_goals', 0))
        draws = sum(1 for m in recent_h2h if m.get('home_goals', 0) == m.get('away_goals', 0))
        away_wins = len(recent_h2h) - home_wins - draws
        
        # Ortalama goller
        avg_home_goals = np.mean([m.get('home_goals', 0) for m in recent_h2h])
        avg_away_goals = np.mean([m.get('away_goals', 0) for m in recent_h2h])
        
        # Psikolojik üstünlük
        psychological_edge = 0.0
        if home_wins > away_wins * 2:
            psychological_edge = 0.2
        elif away_wins > home_wins * 2:
            psychological_edge = -0.2
            
        # Son trend (son 3 H2H)
        recent_3 = recent_h2h[:3]
        recent_home_wins = sum(1 for m in recent_3 if m.get('home_goals', 0) > m.get('away_goals', 0))
        
        if recent_home_wins >= 2:
            recent_trend = 'home_dominant'
        elif recent_home_wins == 0:
            recent_trend = 'away_dominant'
        else:
            recent_trend = 'balanced'
            
        return {
            'historical_advantage': 'home' if home_wins > away_wins else 'away' if away_wins > home_wins else 'neutral',
            'avg_goals': {'home': avg_home_goals, 'away': avg_away_goals},
            'win_rate': {
                'home': home_wins / len(recent_h2h),
                'draw': draws / len(recent_h2h),
                'away': away_wins / len(recent_h2h)
            },
            'recent_trend': recent_trend,
            'psychological_edge': psychological_edge,
            'total_h2h_matches': len(recent_h2h)
        }
        
    def analyze_psychological_factors(self, home_data, away_data, match_context):
        """Psikolojik faktör analizi"""
        factors = {
            'pressure_level': self._calculate_pressure_level(match_context),
            'home_pressure_handling': self._analyze_pressure_handling(home_data),
            'away_pressure_handling': self._analyze_pressure_handling(away_data),
            'derby_factor': match_context.get('is_derby', False),
            'revenge_factor': self._check_revenge_factor(home_data, away_data),
            'confidence_levels': self._calculate_confidence_levels(home_data, away_data),
            'mental_fatigue': self._check_mental_fatigue(home_data, away_data)
        }
        
        # Genel psikolojik avantaj
        home_psych_score = (
            factors['home_pressure_handling'] * (1 + factors['pressure_level']) +
            factors['confidence_levels']['home'] +
            (0.2 if factors['revenge_factor'] == 'home' else 0) -
            factors['mental_fatigue']['home']
        )
        
        away_psych_score = (
            factors['away_pressure_handling'] * (1 + factors['pressure_level']) +
            factors['confidence_levels']['away'] +
            (0.2 if factors['revenge_factor'] == 'away' else 0) -
            factors['mental_fatigue']['away']
        )
        
        factors['psychological_advantage'] = home_psych_score - away_psych_score
        
        return factors
        
    def _calculate_pressure_level(self, match_context):
        """Maç baskı seviyesi"""
        pressure = 0.5  # Baseline
        
        # Sezon sonu
        if match_context.get('season_stage', '') == 'final_weeks':
            pressure += 0.3
            
        # Kritik maç (küme düşme, şampiyonluk vb.)
        if match_context.get('is_crucial', False):
            pressure += 0.4
            
        # Derbi
        if match_context.get('is_derby', False):
            pressure += 0.2
            
        return min(1.0, pressure)
        
    def _analyze_pressure_handling(self, team_data):
        """Baskı altında performans analizi"""
        important_matches = team_data.get('important_matches', [])
        
        if not important_matches:
            return 0.5  # Nötr
            
        # Kritik maçlardaki performans
        wins = sum(1 for m in important_matches if m.get('result', '') == 'win')
        performance_rate = wins / len(important_matches)
        
        return performance_rate
        
    def _check_revenge_factor(self, home_data, away_data):
        """Rövanş faktörü kontrolü"""
        # Son H2H'da ağır mağlubiyet var mı?
        last_h2h = home_data.get('h2h_matches', [])[:1]
        
        if last_h2h:
            home_goals = last_h2h[0].get('home_goals', 0)
            away_goals = last_h2h[0].get('away_goals', 0)
            
            if home_goals - away_goals <= -3:
                return 'home'  # Ev sahibi rövanş peşinde
            elif away_goals - home_goals <= -3:
                return 'away'  # Deplasman rövanş peşinde
                
        return 'none'
        
    def _calculate_confidence_levels(self, home_data, away_data):
        """Takım güven seviyeleri"""
        home_confidence = self._team_confidence(home_data.get('recent_matches', []))
        away_confidence = self._team_confidence(away_data.get('recent_matches', []))
        
        return {
            'home': home_confidence,
            'away': away_confidence,
            'differential': home_confidence - away_confidence
        }
        
    def _team_confidence(self, matches):
        """Takım güven seviyesi hesaplama"""
        if not matches:
            return 0.5
            
        recent_5 = matches[:5]
        
        # Galibiyet serisi
        consecutive_wins = 0
        for match in recent_5:
            if match.get('goals_scored', 0) > match.get('goals_conceded', 0):
                consecutive_wins += 1
            else:
                break
                
        # Atılan gol ortalaması
        avg_goals = np.mean([m.get('goals_scored', 0) for m in recent_5])
        
        # Yenilmezlik serisi
        unbeaten_streak = 0
        for match in recent_5:
            if match.get('goals_scored', 0) >= match.get('goals_conceded', 0):
                unbeaten_streak += 1
            else:
                break
                
        confidence = (
            consecutive_wins * 0.15 +
            min(avg_goals / 3, 1) * 0.3 +
            unbeaten_streak * 0.1 +
            0.5  # Baseline
        )
        
        return min(1.0, confidence)
        
    def _check_mental_fatigue(self, home_data, away_data):
        """Mental yorgunluk kontrolü"""
        home_fatigue = self._calculate_fatigue(home_data)
        away_fatigue = self._calculate_fatigue(away_data)
        
        return {
            'home': home_fatigue,
            'away': away_fatigue
        }
        
    def _calculate_fatigue(self, team_data):
        """Yorgunluk hesaplama"""
        matches_in_week = team_data.get('matches_last_week', 0)
        travel_distance = team_data.get('recent_travel_km', 0)
        
        fatigue = 0.0
        
        # Maç yoğunluğu
        if matches_in_week >= 3:
            fatigue += 0.3
        elif matches_in_week == 2:
            fatigue += 0.1
            
        # Seyahat yorgunluğu
        if travel_distance > 3000:
            fatigue += 0.2
        elif travel_distance > 1000:
            fatigue += 0.1
            
        return fatigue
        
    def analyze_match_context(self, match_context):
        """Gelişmiş maç konteksti analizi"""
        return {
            'importance_score': self._calculate_match_importance(match_context),
            'weather_impact': self._analyze_weather_impact(match_context),
            'time_factors': self._analyze_time_factors(match_context),
            'stadium_factors': self._analyze_stadium_factors(match_context),
            'referee_tendencies': self._analyze_referee_tendencies(match_context)
        }
        
    def _calculate_match_importance(self, context):
        """Maç önem skoru"""
        importance = 0.5  # Baseline
        
        # Lig pozisyonu etkisi
        if context.get('home_position', 10) <= 3 or context.get('away_position', 10) <= 3:
            importance += 0.2  # Şampiyonluk yarışı
            
        if context.get('home_position', 10) >= 18 or context.get('away_position', 10) >= 18:
            importance += 0.3  # Küme düşme mücadelesi
            
        # Kupa maçı
        if context.get('competition_type', '') == 'cup':
            importance += 0.2
            
        # Sezon aşaması
        if context.get('season_stage', '') in ['final_weeks', 'playoff']:
            importance += 0.3
            
        return min(1.0, importance)
        
    def _analyze_weather_impact(self, context):
        """Hava durumu etkisi"""
        weather = context.get('weather', {})
        
        impact = {
            'temperature_effect': 0.0,
            'rain_effect': 0.0,
            'wind_effect': 0.0
        }
        
        # Sıcaklık etkisi
        temp = weather.get('temperature', 20)
        if temp > 30:
            impact['temperature_effect'] = -0.1  # Yorgunluk
        elif temp < 5:
            impact['temperature_effect'] = -0.05  # Soğuk
            
        # Yağmur etkisi
        if weather.get('rain', False):
            impact['rain_effect'] = -0.1  # Teknik oyun zorlaşır
            
        # Rüzgar etkisi
        wind_speed = weather.get('wind_speed', 0)
        if wind_speed > 30:
            impact['wind_effect'] = -0.15  # Uzun toplar etkilenir
            
        return impact
        
    def _analyze_time_factors(self, context):
        """Zaman faktörleri"""
        kick_off_time = context.get('kick_off_time', '20:00')
        
        factors = {
            'early_kickoff': False,
            'late_kickoff': False,
            'midweek_match': context.get('is_midweek', False)
        }
        
        hour = int(kick_off_time.split(':')[0])
        
        if hour < 14:
            factors['early_kickoff'] = True
        elif hour >= 21:
            factors['late_kickoff'] = True
            
        return factors
        
    def _analyze_stadium_factors(self, context):
        """Stadyum faktörleri"""
        return {
            'altitude': context.get('stadium_altitude', 0),
            'capacity_filled': context.get('expected_attendance_rate', 0.7),
            'pitch_condition': context.get('pitch_condition', 'good'),
            'home_fortress_factor': context.get('home_win_rate_stadium', 0.5)
        }
        
    def _analyze_referee_tendencies(self, context):
        """Hakem eğilimleri"""
        referee_stats = context.get('referee_stats', {})
        
        return {
            'cards_per_match': referee_stats.get('avg_cards', 4.5),
            'penalties_per_match': referee_stats.get('avg_penalties', 0.2),
            'home_favor_tendency': referee_stats.get('home_win_rate', 0.45)
        }
        
    def analyze_team_patterns(self, home_data, away_data):
        """Takım oyun kalıpları analizi"""
        return {
            'home_patterns': self._extract_team_patterns(home_data),
            'away_patterns': self._extract_team_patterns(away_data),
            'style_matchup': self._analyze_style_matchup(home_data, away_data)
        }
        
    def _extract_team_patterns(self, team_data):
        """Takım kalıplarını çıkar"""
        matches = team_data.get('recent_matches', [])
        
        if not matches:
            return self._get_default_patterns()
            
        # Erken/geç gol kalıpları
        early_goals = sum(1 for m in matches[:10] if m.get('first_half_goals', 0) > 0)
        late_goals = sum(1 for m in matches[:10] if m.get('second_half_goals', 0) > 0)
        
        # Geri dönüş kapasitesi
        comebacks = sum(1 for m in matches[:10] if m.get('came_from_behind', False))
        
        # Liderliği koruma
        lost_leads = sum(1 for m in matches[:10] if m.get('lost_lead', False))
        
        return {
            'early_goal_tendency': early_goals / min(10, len(matches)),
            'late_goal_tendency': late_goals / min(10, len(matches)),
            'comeback_ability': comebacks / min(10, len(matches)),
            'lead_protection': 1 - (lost_leads / min(10, len(matches))),
            'scoring_periods': self._analyze_scoring_periods(matches),
            'defensive_stability': self._analyze_defensive_stability(matches)
        }
        
    def _get_default_patterns(self):
        """Varsayılan takım kalıpları"""
        return {
            'early_goal_tendency': 0.5,
            'late_goal_tendency': 0.5,
            'comeback_ability': 0.2,
            'lead_protection': 0.7,
            'scoring_periods': {'0-30': 0.3, '30-60': 0.35, '60-90': 0.35},
            'defensive_stability': 0.5
        }
        
    def _analyze_scoring_periods(self, matches):
        """Gol atma periyotları"""
        periods = {'0-30': 0, '30-60': 0, '60-90': 0}
        total_goals = 0
        
        for match in matches[:10]:
            # Basitleştirilmiş - gerçekte dakika bazlı veri gerekir
            goals = match.get('goals_scored', 0)
            total_goals += goals
            
            # Varsayılan dağılım
            if goals > 0:
                periods['0-30'] += goals * 0.3
                periods['30-60'] += goals * 0.35
                periods['60-90'] += goals * 0.35
                
        if total_goals > 0:
            for period in periods:
                periods[period] /= total_goals
                
        return periods
        
    def _analyze_defensive_stability(self, matches):
        """Defansif istikrar"""
        clean_sheets = sum(1 for m in matches[:10] if m.get('goals_conceded', 0) == 0)
        low_concede = sum(1 for m in matches[:10] if m.get('goals_conceded', 0) <= 1)
        
        stability = (clean_sheets * 0.4 + low_concede * 0.1) / min(10, len(matches))
        return stability
        
    def _analyze_style_matchup(self, home_data, away_data):
        """Oyun stili uyumu"""
        home_style = home_data.get('playing_style', 'balanced')
        away_style = away_data.get('playing_style', 'balanced')
        
        # Stil uyum matrisi
        matchup_effects = {
            ('attacking', 'defensive'): {'goals_boost': 0.1, 'excitement': 0.8},
            ('defensive', 'attacking'): {'goals_boost': 0.1, 'excitement': 0.8},
            ('attacking', 'attacking'): {'goals_boost': 0.3, 'excitement': 1.0},
            ('defensive', 'defensive'): {'goals_boost': -0.2, 'excitement': 0.3},
            ('balanced', 'attacking'): {'goals_boost': 0.15, 'excitement': 0.7},
            ('balanced', 'defensive'): {'goals_boost': -0.1, 'excitement': 0.5},
            ('attacking', 'balanced'): {'goals_boost': 0.15, 'excitement': 0.7},
            ('defensive', 'balanced'): {'goals_boost': -0.1, 'excitement': 0.5},
            ('balanced', 'balanced'): {'goals_boost': 0.0, 'excitement': 0.6}
        }
        
        matchup = matchup_effects.get((home_style, away_style), {'goals_boost': 0, 'excitement': 0.5})
        
        return {
            'style_combination': f"{home_style} vs {away_style}",
            'expected_goal_impact': matchup['goals_boost'],
            'match_excitement': matchup['excitement'],
            'tactical_advantage': self._determine_tactical_advantage(home_style, away_style)
        }
        
    def _determine_tactical_advantage(self, home_style, away_style):
        """Taktiksel avantaj belirleme"""
        advantages = {
            ('attacking', 'defensive'): 'neutral',
            ('defensive', 'attacking'): 'neutral',
            ('attacking', 'attacking'): 'neutral',
            ('defensive', 'defensive'): 'neutral',
            ('balanced', 'attacking'): 'slight_away',
            ('balanced', 'defensive'): 'slight_home',
            ('attacking', 'balanced'): 'slight_home',
            ('defensive', 'balanced'): 'slight_away',
            ('balanced', 'balanced'): 'neutral'
        }
        
        return advantages.get((home_style, away_style), 'neutral')
        
    def analyze_goal_dynamics(self, home_data, away_data):
        """Gol dinamikleri analizi"""
        return {
            'home_goal_distribution': self._analyze_goal_distribution(home_data),
            'away_goal_distribution': self._analyze_goal_distribution(away_data),
            'combined_dynamics': self._analyze_combined_dynamics(home_data, away_data)
        }
        
    def _analyze_goal_distribution(self, team_data):
        """Gol dağılım analizi"""
        matches = team_data.get('recent_matches', [])
        
        if not matches:
            return {
                'avg_first_half': 0.6,
                'avg_second_half': 0.8,
                'consistency': 0.5,
                'explosion_rate': 0.1
            }
            
        first_half_goals = [m.get('first_half_goals', 0) for m in matches[:10]]
        second_half_goals = [m.get('second_half_goals', 0) for m in matches[:10]]
        
        # Patlama maçları (3+ gol)
        explosion_matches = sum(1 for m in matches[:10] if m.get('goals_scored', 0) >= 3)
        
        return {
            'avg_first_half': np.mean(first_half_goals) if first_half_goals else 0.6,
            'avg_second_half': np.mean(second_half_goals) if second_half_goals else 0.8,
            'consistency': 1 / (1 + np.std([m.get('goals_scored', 0) for m in matches[:10]])),
            'explosion_rate': explosion_matches / min(10, len(matches))
        }
        
    def _analyze_combined_dynamics(self, home_data, away_data):
        """Birleşik gol dinamikleri"""
        home_attack = home_data.get('avg_goals_scored', 1.5)
        home_defense = home_data.get('avg_goals_conceded', 1.2)
        away_attack = away_data.get('avg_goals_scored', 1.3)
        away_defense = away_data.get('avg_goals_conceded', 1.3)
        
        # Çapraz etkileşim
        expected_home_goals = home_attack * (away_defense / 1.3)
        expected_away_goals = away_attack * (home_defense / 1.3)
        
        # Tempo faktörü
        tempo_factor = (home_attack + away_attack) / 2.8  # Normalleştirilmiş
        
        return {
            'expected_total_goals': expected_home_goals + expected_away_goals,
            'tempo_factor': tempo_factor,
            'high_scoring_probability': self._calculate_high_scoring_prob(expected_home_goals + expected_away_goals),
            'both_teams_scoring_prob': self._calculate_btts_prob(expected_home_goals, expected_away_goals)
        }
        
    def _calculate_high_scoring_prob(self, expected_total):
        """Yüksek skorlu maç olasılığı"""
        # Sigmoid fonksiyonu ile 2.5 gol etrafında olasılık hesapla
        return 1 / (1 + np.exp(-2 * (expected_total - 2.5)))
        
    def _calculate_btts_prob(self, expected_home, expected_away):
        """Her iki takımın gol atma olasılığı"""
        # Poisson yaklaşımı ile 0 gol atmama olasılığı
        prob_home_scores = 1 - np.exp(-expected_home)
        prob_away_scores = 1 - np.exp(-expected_away)
        
        return prob_home_scores * prob_away_scores
        
    def analyze_tactical_matchup(self, home_data, away_data):
        """Taktiksel eşleşme analizi"""
        return {
            'formation_matchup': self._analyze_formation_matchup(home_data, away_data),
            'key_player_impact': self._analyze_key_players(home_data, away_data),
            'tactical_flexibility': self._analyze_tactical_flexibility(home_data, away_data)
        }
        
    def _analyze_formation_matchup(self, home_data, away_data):
        """Formasyon uyumu analizi"""
        home_formation = home_data.get('preferred_formation', '4-4-2')
        away_formation = away_data.get('preferred_formation', '4-3-3')
        
        # Basit formasyon avantaj matrisi
        formation_advantages = {
            ('4-4-2', '4-3-3'): 'slight_away',  # 4-3-3 orta sahada üstün
            ('4-3-3', '4-4-2'): 'slight_home',
            ('3-5-2', '4-4-2'): 'slight_home',  # 3-5-2 kanatlarda üstün
            ('4-4-2', '3-5-2'): 'slight_away',
            # Diğer kombinasyonlar...
        }
        
        advantage = formation_advantages.get((home_formation, away_formation), 'neutral')
        
        return {
            'home_formation': home_formation,
            'away_formation': away_formation,
            'tactical_advantage': advantage,
            'midfield_control': self._predict_midfield_control(home_formation, away_formation)
        }
        
    def _predict_midfield_control(self, home_formation, away_formation):
        """Orta saha kontrolü tahmini"""
        # Orta saha oyuncu sayıları (basitleştirilmiş)
        midfield_counts = {
            '4-4-2': 4,
            '4-3-3': 3,
            '3-5-2': 5,
            '4-2-3-1': 5,
            '4-5-1': 5
        }
        
        home_mid = midfield_counts.get(home_formation, 4)
        away_mid = midfield_counts.get(away_formation, 4)
        
        if home_mid > away_mid:
            return 'home_dominant'
        elif away_mid > home_mid:
            return 'away_dominant'
        else:
            return 'balanced'
            
    def _analyze_key_players(self, home_data, away_data):
        """Anahtar oyuncu etkisi"""
        home_key_available = home_data.get('key_players_available', 1.0)
        away_key_available = away_data.get('key_players_available', 1.0)
        
        return {
            'home_strength': home_key_available,
            'away_strength': away_key_available,
            'impact_differential': home_key_available - away_key_available
        }
        
    def _analyze_tactical_flexibility(self, home_data, away_data):
        """Taktiksel esneklik"""
        home_flexibility = home_data.get('formation_changes_last_5', 0) / 5
        away_flexibility = away_data.get('formation_changes_last_5', 0) / 5
        
        return {
            'home_adaptability': home_flexibility,
            'away_adaptability': away_flexibility,
            'unpredictability_factor': (home_flexibility + away_flexibility) / 2
        }
        
    def calculate_feature_importance(self, features):
        """Özellik önem derecelerini hesapla"""
        importance_scores = {}
        
        # Her özellik kategorisi için önem skorları
        for category, feature_data in features.items():
            if isinstance(feature_data, dict):
                # Alt özellikleri değerlendir
                category_importance = 0
                for sub_feature, value in feature_data.items():
                    if isinstance(value, (int, float)):
                        # Değer büyüklüğüne göre önem
                        importance = abs(value) * self.feature_weights.get(category, 0.1)
                        category_importance += importance
                        
                importance_scores[category] = category_importance
                
        # Normalize et
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            for category in importance_scores:
                importance_scores[category] /= total_importance
                
        return importance_scores