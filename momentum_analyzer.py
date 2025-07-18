"""
Momentum Analyzer - Takım Form ve Momentum Analizi
Takımların güncel formunu, trend analizini ve psikolojik momentumunu hesaplar
"""
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MomentumAnalyzer:
    """
    Takım momentum ve form analizi
    """
    
    def __init__(self):
        self.form_weights = {
            'W': 3,    # Galibiyet
            'D': 1,    # Beraberlik
            'L': 0     # Yenilgi
        }
        
        # Zaman bazlı ağırlıklar (son maçlar daha önemli)
        self.recency_weights = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        
    def analyze_momentum(self, team_matches, is_home_team=True):
        """
        Takımın momentum analizini yap
        
        Args:
            team_matches: Son maçlar listesi
            is_home_team: Ev sahibi mi?
            
        Returns:
            dict: Momentum analiz sonuçları
        """
        if not team_matches:
            return self._get_default_momentum()
            
        # Son 10 maçı al
        recent_matches = sorted(team_matches, key=lambda x: x.get('date', ''), reverse=True)[:10]
        
        # Ev/Deplasman ayrımı
        if is_home_team:
            venue_matches = [m for m in recent_matches if m.get('venue') == 'home']
        else:
            venue_matches = [m for m in recent_matches if m.get('venue') == 'away']
            
        # Analizleri yap
        overall_form = self._calculate_form_score(recent_matches)
        venue_form = self._calculate_form_score(venue_matches) if venue_matches else overall_form * 0.9
        
        trend = self._analyze_trend(recent_matches)
        goal_trend = self._analyze_goal_trend(recent_matches)
        
        # Seri analizi
        current_streak = self._get_current_streak(recent_matches)
        
        # Psikolojik momentum
        psychological_momentum = self._calculate_psychological_momentum(
            overall_form, trend, current_streak, goal_trend
        )
        
        # Son 5 maç puan ortalaması
        last_5_matches = recent_matches[:5]
        last_5_ppg = self._calculate_points_per_game(last_5_matches)
        
        return {
            'overall_score': psychological_momentum,
            'overall_form': overall_form,
            'venue_form': venue_form,
            'trend': trend,
            'goal_trend': goal_trend,
            'current_streak': current_streak,
            'last_5_ppg': last_5_ppg,
            'matches_analyzed': len(recent_matches),
            'venue_matches': len(venue_matches)
        }
        
    def _calculate_form_score(self, matches):
        """
        Form skoru hesapla (0-100)
        """
        if not matches:
            return 50
            
        total_score = 0
        total_weight = 0
        
        for i, match in enumerate(matches[:10]):
            result = self._get_match_result(match)
            weight = self.recency_weights[i] if i < len(self.recency_weights) else 0.1
            
            points = self.form_weights.get(result, 0)
            total_score += points * weight
            total_weight += weight
            
        if total_weight == 0:
            return 50
            
        # 0-100 skalasına normalize et
        normalized_score = (total_score / total_weight) * 33.33  # Max 3 puan * 33.33 = 100
        return min(100, max(0, normalized_score))
        
    def _analyze_trend(self, matches):
        """
        Form trendini analiz et
        """
        if len(matches) < 3:
            return 'stable'
            
        # Son 5 maç vs önceki 5 maç
        recent_form = self._calculate_form_score(matches[:5])
        older_form = self._calculate_form_score(matches[5:10]) if len(matches) >= 10 else recent_form
        
        diff = recent_form - older_form
        
        if diff > 15:
            return 'ascending'
        elif diff < -15:
            return 'descending'
        else:
            return 'stable'
            
    def _analyze_goal_trend(self, matches):
        """
        Gol atma/yeme trendini analiz et
        """
        if len(matches) < 5:
            return {'scoring': 'stable', 'conceding': 'stable'}
            
        recent_matches = matches[:5]
        older_matches = matches[5:10] if len(matches) >= 10 else []
        
        # Son maçlarda gol ortalamaları
        recent_scored = np.mean([m.get('goals_scored', 0) for m in recent_matches])
        recent_conceded = np.mean([m.get('goals_conceded', 0) for m in recent_matches])
        
        if older_matches:
            older_scored = np.mean([m.get('goals_scored', 0) for m in older_matches])
            older_conceded = np.mean([m.get('goals_conceded', 0) for m in older_matches])
            
            scoring_diff = recent_scored - older_scored
            conceding_diff = recent_conceded - older_conceded
            
            scoring_trend = 'improving' if scoring_diff > 0.3 else 'declining' if scoring_diff < -0.3 else 'stable'
            conceding_trend = 'worsening' if conceding_diff > 0.3 else 'improving' if conceding_diff < -0.3 else 'stable'
        else:
            scoring_trend = 'stable'
            conceding_trend = 'stable'
            
        return {
            'scoring': scoring_trend,
            'conceding': conceding_trend,
            'recent_avg_scored': round(recent_scored, 2),
            'recent_avg_conceded': round(recent_conceded, 2)
        }
        
    def _get_current_streak(self, matches):
        """
        Mevcut seriyi hesapla
        """
        if not matches:
            return {'type': 'none', 'count': 0}
            
        streak_type = None
        count = 0
        
        for match in matches:
            result = self._get_match_result(match)
            
            if streak_type is None:
                streak_type = result
                count = 1
            elif result == streak_type:
                count += 1
            else:
                break
                
        return {
            'type': streak_type,
            'count': count,
            'description': self._get_streak_description(streak_type, count)
        }
        
    def _calculate_psychological_momentum(self, form, trend, streak, goal_trend):
        """
        Psikolojik momentum hesapla (0-100)
        """
        # Temel form skoru
        momentum = form
        
        # Trend etkisi
        if trend == 'ascending':
            momentum += 10
        elif trend == 'descending':
            momentum -= 10
            
        # Seri etkisi
        if streak['type'] == 'W':
            momentum += min(15, streak['count'] * 3)
        elif streak['type'] == 'L':
            momentum -= min(15, streak['count'] * 3)
            
        # Gol trendi etkisi
        if goal_trend['scoring'] == 'improving':
            momentum += 5
        elif goal_trend['scoring'] == 'declining':
            momentum -= 5
            
        if goal_trend['conceding'] == 'improving':
            momentum += 5
        elif goal_trend['conceding'] == 'worsening':
            momentum -= 5
            
        return min(100, max(0, momentum))
        
    def _calculate_points_per_game(self, matches):
        """
        Maç başına puan ortalaması
        """
        if not matches:
            return 0
            
        total_points = sum(self.form_weights.get(self._get_match_result(m), 0) for m in matches)
        return round(total_points / len(matches), 2)
        
    def _get_match_result(self, match):
        """
        Maç sonucunu belirle
        """
        goals_scored = match.get('goals_scored', 0)
        goals_conceded = match.get('goals_conceded', 0)
        
        if goals_scored > goals_conceded:
            return 'W'
        elif goals_scored < goals_conceded:
            return 'L'
        else:
            return 'D'
            
    def _get_streak_description(self, streak_type, count):
        """
        Seri açıklaması
        """
        if count == 0:
            return "No streak"
            
        type_map = {'W': 'win', 'D': 'draw', 'L': 'loss'}
        streak_name = type_map.get(streak_type, 'unknown')
        
        if count == 1:
            return f"1 {streak_name}"
        else:
            return f"{count} {streak_name}s in a row"
            
    def _get_default_momentum(self):
        """
        Varsayılan momentum değerleri
        """
        return {
            'overall_score': 50,
            'overall_form': 50,
            'venue_form': 50,
            'trend': 'stable',
            'goal_trend': {'scoring': 'stable', 'conceding': 'stable'},
            'current_streak': {'type': 'none', 'count': 0},
            'last_5_ppg': 1.0,
            'matches_analyzed': 0,
            'venue_matches': 0
        }