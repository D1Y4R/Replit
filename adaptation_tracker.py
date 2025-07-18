"""
Adaptation Tracker - Değişim ve Adaptasyon Analizi
Takımların değişimlere uyum sağlama kapasitesini ve gelişim trendlerini analiz eder
"""
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AdaptationTracker:
    """
    Takım adaptasyon ve değişim analizi
    """
    
    def __init__(self):
        # Adaptasyon periyotları
        self.adaptation_periods = {
            'immediate': 3,      # İlk 3 maç
            'short_term': 7,     # 7 maç
            'medium_term': 15,   # 15 maç
            'long_term': 30      # 30 maç
        }
        
        # Değişim tipleri ve etki süreleri
        self.change_impacts = {
            'manager_change': {
                'initial_impact': -10,
                'recovery_matches': 5,
                'potential_boost': 15
            },
            'tactical_shift': {
                'initial_impact': -5,
                'recovery_matches': 3,
                'potential_boost': 10
            },
            'formation_change': {
                'initial_impact': -3,
                'recovery_matches': 2,
                'potential_boost': 8
            }
        }
        
    def analyze_adaptation(self, team_matches, team_changes=None):
        """
        Takımın adaptasyon analizini yap
        
        Args:
            team_matches: Takım maçları
            team_changes: Takımdaki değişiklikler (opsiyonel)
            
        Returns:
            dict: Adaptasyon analizi
        """
        if not team_matches:
            return self._get_default_adaptation()
            
        # Son 30 maçı al
        recent_matches = sorted(team_matches, key=lambda x: x.get('date', ''), reverse=True)[:30]
        
        # Analizleri yap
        form_evolution = self._analyze_form_evolution(recent_matches)
        tactical_adaptation = self._detect_tactical_changes(recent_matches)
        improvement_rate = self._calculate_improvement_rate(recent_matches)
        consistency_trend = self._analyze_consistency_trend(recent_matches)
        
        # Değişim etkisi analizi
        change_impact = self._analyze_change_impact(recent_matches, team_changes)
        
        # Rakip adaptasyonu
        opponent_adaptation = self._analyze_opponent_adaptation(recent_matches)
        
        return {
            'form_evolution': form_evolution,
            'tactical_adaptation': tactical_adaptation,
            'improvement_rate': improvement_rate,
            'consistency_trend': consistency_trend,
            'change_impact': change_impact,
            'opponent_adaptation': opponent_adaptation,
            'adaptation_score': self._calculate_adaptation_score(
                form_evolution,
                improvement_rate,
                consistency_trend
            )
        }
        
    def _analyze_form_evolution(self, matches):
        """
        Form evrimini analiz et
        """
        if len(matches) < 10:
            return {
                'trend': 'stable',
                'volatility': 'medium',
                'current_phase': 'unknown'
            }
            
        # Periyotlara göre form hesapla
        form_by_period = {}
        
        for period_name, period_size in self.adaptation_periods.items():
            if len(matches) >= period_size:
                period_matches = matches[:period_size]
                wins = sum(1 for m in period_matches 
                          if m.get('goals_scored', 0) > m.get('goals_conceded', 0))
                form_by_period[period_name] = wins / period_size
                
        # Trend analizi
        if 'immediate' in form_by_period and 'short_term' in form_by_period:
            if form_by_period['immediate'] > form_by_period['short_term'] + 0.2:
                trend = 'sharp_improvement'
            elif form_by_period['immediate'] < form_by_period['short_term'] - 0.2:
                trend = 'sharp_decline'
            elif form_by_period['immediate'] > form_by_period['short_term']:
                trend = 'improving'
            elif form_by_period['immediate'] < form_by_period['short_term']:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
            
        # Volatilite analizi
        if len(matches) >= 10:
            results = [1 if m.get('goals_scored', 0) > m.get('goals_conceded', 0) 
                      else 0 if m.get('goals_scored', 0) < m.get('goals_conceded', 0)
                      else 0.5 for m in matches[:10]]
            volatility = np.std(results)
            
            if volatility > 0.4:
                volatility_level = 'high'
            elif volatility > 0.25:
                volatility_level = 'medium'
            else:
                volatility_level = 'low'
        else:
            volatility_level = 'medium'
            
        # Mevcut faz
        recent_results = [m.get('goals_scored', 0) > m.get('goals_conceded', 0) 
                         for m in matches[:3]]
        if sum(recent_results) >= 2:
            current_phase = 'winning'
        elif sum(recent_results) <= 1:
            current_phase = 'struggling'
        else:
            current_phase = 'transitional'
            
        return {
            'trend': trend,
            'volatility': volatility_level,
            'current_phase': current_phase,
            'form_by_period': form_by_period
        }
        
    def _detect_tactical_changes(self, matches):
        """
        Taktiksel değişimleri tespit et
        """
        if len(matches) < 10:
            return {
                'detected_change': False,
                'change_type': None,
                'success_rate': 0
            }
            
        # Gol paternlerindeki değişim
        early_matches = matches[10:20] if len(matches) >= 20 else []
        recent_matches = matches[:10]
        
        if not early_matches:
            return {
                'detected_change': False,
                'change_type': None,
                'success_rate': 0
            }
            
        # Metrik karşılaştırmaları
        early_goals = np.mean([m.get('goals_scored', 0) for m in early_matches])
        recent_goals = np.mean([m.get('goals_scored', 0) for m in recent_matches])
        
        early_conceded = np.mean([m.get('goals_conceded', 0) for m in early_matches])
        recent_conceded = np.mean([m.get('goals_conceded', 0) for m in recent_matches])
        
        # Değişim tespiti
        change_detected = False
        change_type = None
        
        if recent_goals > early_goals + 0.5:
            change_detected = True
            change_type = 'more_attacking'
        elif recent_goals < early_goals - 0.5:
            change_detected = True
            change_type = 'more_defensive'
            
        if recent_conceded < early_conceded - 0.5:
            change_detected = True
            if change_type:
                change_type += '_improved_defense'
            else:
                change_type = 'defensive_improvement'
                
        # Başarı oranı
        if change_detected:
            recent_wins = sum(1 for m in recent_matches 
                            if m.get('goals_scored', 0) > m.get('goals_conceded', 0))
            success_rate = recent_wins / len(recent_matches)
        else:
            success_rate = 0
            
        return {
            'detected_change': change_detected,
            'change_type': change_type,
            'success_rate': success_rate,
            'goal_changes': {
                'scoring': recent_goals - early_goals,
                'conceding': recent_conceded - early_conceded
            }
        }
        
    def _calculate_improvement_rate(self, matches):
        """
        Gelişim hızını hesapla
        """
        if len(matches) < 20:
            return {
                'overall_rate': 0,
                'attack_improvement': 0,
                'defense_improvement': 0
            }
            
        # 10'ar maçlık periyotlar
        periods = []
        for i in range(0, min(30, len(matches)), 10):
            period_matches = matches[i:i+10]
            if len(period_matches) >= 5:
                win_rate = sum(1 for m in period_matches 
                              if m.get('goals_scored', 0) > m.get('goals_conceded', 0)) / len(period_matches)
                avg_scored = np.mean([m.get('goals_scored', 0) for m in period_matches])
                avg_conceded = np.mean([m.get('goals_conceded', 0) for m in period_matches])
                
                periods.append({
                    'win_rate': win_rate,
                    'avg_scored': avg_scored,
                    'avg_conceded': avg_conceded
                })
                
        if len(periods) < 2:
            return {
                'overall_rate': 0,
                'attack_improvement': 0,
                'defense_improvement': 0
            }
            
        # İlk ve son periyot karşılaştırması
        first_period = periods[-1]  # En eski
        last_period = periods[0]    # En yeni
        
        overall_rate = (last_period['win_rate'] - first_period['win_rate']) * 100
        attack_improvement = last_period['avg_scored'] - first_period['avg_scored']
        defense_improvement = first_period['avg_conceded'] - last_period['avg_conceded']
        
        return {
            'overall_rate': round(overall_rate, 1),
            'attack_improvement': round(attack_improvement, 2),
            'defense_improvement': round(defense_improvement, 2)
        }
        
    def _analyze_consistency_trend(self, matches):
        """
        Tutarlılık trendini analiz et
        """
        if len(matches) < 10:
            return {
                'consistency_level': 'medium',
                'trend': 'stable'
            }
            
        # Son 10 maç için sonuç varyansı
        recent_results = []
        for match in matches[:10]:
            if match.get('goals_scored', 0) > match.get('goals_conceded', 0):
                recent_results.append(3)
            elif match.get('goals_scored', 0) == match.get('goals_conceded', 0):
                recent_results.append(1)
            else:
                recent_results.append(0)
                
        recent_variance = np.var(recent_results)
        
        # Önceki 10 maç
        if len(matches) >= 20:
            older_results = []
            for match in matches[10:20]:
                if match.get('goals_scored', 0) > match.get('goals_conceded', 0):
                    older_results.append(3)
                elif match.get('goals_scored', 0) == match.get('goals_conceded', 0):
                    older_results.append(1)
                else:
                    older_results.append(0)
                    
            older_variance = np.var(older_results)
            
            # Trend belirleme
            if recent_variance < older_variance - 0.5:
                trend = 'improving'
            elif recent_variance > older_variance + 0.5:
                trend = 'worsening'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
            
        # Tutarlılık seviyesi
        if recent_variance < 1.0:
            consistency_level = 'high'
        elif recent_variance < 2.0:
            consistency_level = 'medium'
        else:
            consistency_level = 'low'
            
        return {
            'consistency_level': consistency_level,
            'trend': trend,
            'variance': round(recent_variance, 2)
        }
        
    def _analyze_change_impact(self, matches, team_changes):
        """
        Değişimlerin etkisini analiz et
        """
        if not team_changes:
            return {
                'recent_change': None,
                'matches_since_change': 0,
                'performance_impact': 0
            }
            
        # En son değişim
        latest_change = max(team_changes, key=lambda x: x.get('date', ''))
        change_type = latest_change.get('type', 'unknown')
        
        # Değişimden sonraki maçlar
        change_date = latest_change.get('date', '')
        matches_after = [m for m in matches if m.get('date', '') >= change_date]
        matches_before = [m for m in matches if m.get('date', '') < change_date][:10]
        
        if len(matches_after) < 3:
            return {
                'recent_change': change_type,
                'matches_since_change': len(matches_after),
                'performance_impact': 0
            }
            
        # Performans karşılaştırması
        win_rate_before = sum(1 for m in matches_before 
                             if m.get('goals_scored', 0) > m.get('goals_conceded', 0)) / max(1, len(matches_before))
        win_rate_after = sum(1 for m in matches_after 
                            if m.get('goals_scored', 0) > m.get('goals_conceded', 0)) / len(matches_after)
        
        performance_impact = (win_rate_after - win_rate_before) * 100
        
        return {
            'recent_change': change_type,
            'matches_since_change': len(matches_after),
            'performance_impact': round(performance_impact, 1),
            'adaptation_phase': self._get_adaptation_phase(len(matches_after), change_type)
        }
        
    def _analyze_opponent_adaptation(self, matches):
        """
        Belirli rakip tiplerine karşı adaptasyon
        """
        if len(matches) < 20:
            return {
                'vs_similar_style': 1.0,
                'vs_counter_style': 1.0,
                'adaptation_ability': 'medium'
            }
            
        # Basitleştirilmiş analiz
        # Farklı sonuç paternleri
        result_patterns = []
        
        for i in range(len(matches) - 2):
            pattern = []
            for j in range(3):
                if matches[i+j].get('goals_scored', 0) > matches[i+j].get('goals_conceded', 0):
                    pattern.append('W')
                elif matches[i+j].get('goals_scored', 0) == matches[i+j].get('goals_conceded', 0):
                    pattern.append('D')
                else:
                    pattern.append('L')
            result_patterns.append(''.join(pattern))
            
        # Pattern çeşitliliği = adaptasyon yeteneği
        unique_patterns = len(set(result_patterns))
        total_patterns = len(result_patterns)
        
        if total_patterns > 0:
            diversity_ratio = unique_patterns / total_patterns
            
            if diversity_ratio > 0.7:
                adaptation_ability = 'high'
            elif diversity_ratio > 0.4:
                adaptation_ability = 'medium'
            else:
                adaptation_ability = 'low'
        else:
            adaptation_ability = 'medium'
            
        return {
            'vs_similar_style': 1.0,
            'vs_counter_style': 1.0,
            'adaptation_ability': adaptation_ability,
            'pattern_diversity': round(diversity_ratio if 'diversity_ratio' in locals() else 0.5, 2)
        }
        
    def _calculate_adaptation_score(self, form_evolution, improvement_rate, consistency):
        """
        Genel adaptasyon skoru (0-100)
        """
        score = 50  # Temel skor
        
        # Form trendi etkisi
        trend_scores = {
            'sharp_improvement': 20,
            'improving': 10,
            'stable': 0,
            'declining': -10,
            'sharp_decline': -20
        }
        score += trend_scores.get(form_evolution['trend'], 0)
        
        # Gelişim hızı etkisi
        score += min(20, max(-20, improvement_rate['overall_rate'] / 2))
        
        # Tutarlılık etkisi
        consistency_scores = {
            'high': 10,
            'medium': 0,
            'low': -10
        }
        score += consistency_scores.get(consistency['consistency_level'], 0)
        
        # Tutarlılık trendi
        if consistency['trend'] == 'improving':
            score += 5
        elif consistency['trend'] == 'worsening':
            score -= 5
            
        return min(100, max(0, score))
        
    def _get_adaptation_phase(self, matches_since, change_type):
        """
        Adaptasyon fazını belirle
        """
        if change_type not in self.change_impacts:
            return 'unknown'
            
        impact_info = self.change_impacts[change_type]
        recovery_matches = impact_info['recovery_matches']
        
        if matches_since < recovery_matches:
            return 'adjustment'
        elif matches_since < recovery_matches * 2:
            return 'stabilization'
        else:
            return 'matured'
            
    def _get_default_adaptation(self):
        """
        Varsayılan adaptasyon değerleri
        """
        return {
            'form_evolution': {
                'trend': 'stable',
                'volatility': 'medium',
                'current_phase': 'unknown'
            },
            'tactical_adaptation': {
                'detected_change': False,
                'change_type': None,
                'success_rate': 0
            },
            'improvement_rate': {
                'overall_rate': 0,
                'attack_improvement': 0,
                'defense_improvement': 0
            },
            'consistency_trend': {
                'consistency_level': 'medium',
                'trend': 'stable'
            },
            'change_impact': {
                'recent_change': None,
                'matches_since_change': 0,
                'performance_impact': 0
            },
            'opponent_adaptation': {
                'vs_similar_style': 1.0,
                'vs_counter_style': 1.0,
                'adaptation_ability': 'medium'
            },
            'adaptation_score': 50
        }