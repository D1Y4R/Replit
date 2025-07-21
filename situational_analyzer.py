"""
Situational Analyzer - Durumsal Faktör Analizi
Takımların özel durumlardaki performansını ve motivasyon faktörlerini analiz eder
"""
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SituationalAnalyzer:
    """
    Durumsal faktör ve motivasyon analizi
    """
    
    def __init__(self):
        # Rakip güç kategorileri
        self.opponent_strength_thresholds = {
            'top': 0.8,      # Üst %20
            'strong': 0.6,   # Üst %40
            'medium': 0.4,   # Orta %20
            'weak': 0.2,     # Alt %40
            'bottom': 0      # Alt %20
        }
        
        # Kritik maç tipleri
        self.critical_match_types = [
            'derby',
            'title_race',
            'relegation_battle',
            'cup_knockout',
            'european_qualification'
        ]
        
    def analyze_situational_factors(self, team_matches, team_info, league_info=None):
        """
        Durumsal faktörleri analiz et
        
        Args:
            team_matches: Takımın maçları
            team_info: Takım bilgileri (pozisyon, puan vs.)
            league_info: Lig bilgileri (opsiyonel)
            
        Returns:
            dict: Durumsal analiz sonuçları
        """
        if not team_matches:
            return self._get_default_situational()
            
        # Analizleri yap
        opponent_performance = self._analyze_opponent_based_performance(team_matches, league_info)
        big_match_performance = self._analyze_big_match_performance(team_matches)
        fixture_congestion = self._analyze_fixture_congestion(team_matches)
        motivation_level = self._calculate_motivation_level(team_info, league_info)
        pressure_handling = self._analyze_pressure_handling(team_matches, team_info)
        
        # Özel durumlar
        special_circumstances = self._detect_special_circumstances(team_info, league_info)
        
        return {
            'opponent_performance': opponent_performance,
            'big_match_performer': big_match_performance['is_performer'],
            'big_match_details': big_match_performance,
            'fixture_congestion_impact': fixture_congestion,
            'motivation_level': motivation_level,
            'pressure_handling': pressure_handling,
            'special_circumstances': special_circumstances,
            'performance_adjustments': self._calculate_adjustments(
                opponent_performance,
                big_match_performance,
                motivation_level,
                pressure_handling
            )
        }
        
    def _analyze_opponent_based_performance(self, matches, league_info):
        """
        Rakip gücüne göre performans analizi
        """
        if not matches:
            return {
                'vs_top_teams': 1.0,
                'vs_strong_teams': 1.0,
                'vs_weak_teams': 1.0,
                'vs_bottom_teams': 1.0
            }
            
        # Rakipleri kategorize et
        performance_by_category = {
            'top': {'points': 0, 'matches': 0},
            'strong': {'points': 0, 'matches': 0},
            'weak': {'points': 0, 'matches': 0},
            'bottom': {'points': 0, 'matches': 0}
        }
        
        for match in matches:
            opponent_strength = self._get_opponent_strength(match, league_info)
            category = self._categorize_opponent(opponent_strength)
            
            # Puan hesapla
            if match.get('goals_scored', 0) > match.get('goals_conceded', 0):
                points = 3
            elif match.get('goals_scored', 0) == match.get('goals_conceded', 0):
                points = 1
            else:
                points = 0
                
            if category in performance_by_category:
                performance_by_category[category]['points'] += points
                performance_by_category[category]['matches'] += 1
                
        # Performans çarpanları hesapla
        result = {}
        for category, data in performance_by_category.items():
            if data['matches'] > 0:
                avg_points = data['points'] / data['matches']
                # Normal beklenti 1.5 puan, çarpan olarak hesapla
                result[f'vs_{category}_teams'] = avg_points / 1.5
            else:
                result[f'vs_{category}_teams'] = 1.0
                
        return result
        
    def _analyze_big_match_performance(self, matches):
        """
        Büyük maç performansını analiz et
        """
        if not matches:
            return {'is_performer': False, 'big_match_points': 0, 'total_big_matches': 0}
            
        big_match_points = 0
        total_big_matches = 0
        
        for match in matches:
            # Büyük maç kriterleri
            is_big_match = False
            
            # Derby
            if 'derby' in match.get('match_type', '').lower():
                is_big_match = True
                
            # Yüksek profilli rakip
            if match.get('opponent_position', 10) <= 5:
                is_big_match = True
                
            # Kritik puan durumu
            if match.get('importance', 'normal') in ['high', 'critical']:
                is_big_match = True
                
            if is_big_match:
                total_big_matches += 1
                
                # Sonuç değerlendirmesi
                if match.get('goals_scored', 0) > match.get('goals_conceded', 0):
                    big_match_points += 3
                elif match.get('goals_scored', 0) == match.get('goals_conceded', 0):
                    big_match_points += 1
                    
        if total_big_matches == 0:
            return {'is_performer': False, 'big_match_points': 0, 'total_big_matches': 0}
            
        avg_points = big_match_points / total_big_matches
        is_performer = avg_points >= 1.5  # Ortalama 1.5+ puan
        
        return {
            'is_performer': is_performer,
            'big_match_points': big_match_points,
            'total_big_matches': total_big_matches,
            'average_points': round(avg_points, 2)
        }
        
    def _analyze_fixture_congestion(self, matches):
        """
        Fikstür yoğunluğu etkisini analiz et
        """
        if len(matches) < 5:
            return 0  # Yeterli veri yok
            
        # Son 5 maçın tarih aralığını kontrol et
        recent_matches = sorted(matches, key=lambda x: x.get('date', ''), reverse=True)[:5]
        
        # Tarih farkını hesapla (basitleştirilmiş)
        # Normalde 5 maç 15-20 günde oynanır
        # Eğer daha sık ise yorgunluk etkisi var
        
        # Basit tahmin: Son 5 maçta fazla beraberlik/yenilgi varsa yorgunluk göstergesi
        poor_results = sum(1 for m in recent_matches 
                          if m.get('goals_scored', 0) <= m.get('goals_conceded', 0))
        
        if poor_results >= 4:
            return -10  # Yüksek yorgunluk
        elif poor_results >= 3:
            return -5   # Orta yorgunluk
        else:
            return 0    # Normal
            
    def _calculate_motivation_level(self, team_info, league_info):
        """
        Motivasyon seviyesini hesapla (0-100)
        """
        motivation = 70  # Temel motivasyon
        
        if not team_info:
            return motivation
            
        position = team_info.get('position', 10)
        total_teams = league_info.get('total_teams', 20) if league_info else 20
        
        # Pozisyon bazlı motivasyon
        if position <= 3:
            # Şampiyonluk yarışı
            motivation += 20
        elif position <= 6:
            # Avrupa kupası yarışı
            motivation += 15
        elif position >= total_teams - 3:
            # Küme düşme mücadelesi
            motivation += 25  # Hayatta kalma motivasyonu
        elif position >= total_teams - 6:
            # Küme düşme tehlikesi
            motivation += 15
            
        # Sezon dönemi etkisi
        if team_info.get('matches_played', 0) < 10:
            # Sezon başı
            motivation += 10
        elif team_info.get('matches_played', 0) > 30:
            # Sezon sonu
            if position > 6 and position < total_teams - 6:
                # Orta sıra, motivasyon düşük
                motivation -= 10
                
        # Form etkisi
        recent_form = team_info.get('recent_form', 'DDDDD')
        wins = recent_form.count('W')
        losses = recent_form.count('L')
        
        if wins >= 3:
            motivation += 10  # İyi form motivasyon artırır
        elif losses >= 3:
            motivation -= 5   # Kötü form motivasyon düşürür
            
        return min(100, max(0, motivation))
        
    def _analyze_pressure_handling(self, matches, team_info):
        """
        Baskı altında performans analizi
        """
        if not matches:
            return 'average'
            
        # Kritik maçlardaki performans
        pressure_matches = 0
        good_results = 0
        
        for match in matches:
            # Baskı göstergeleri
            is_pressure = False
            
            # Güçlü rakibe karşı deplasman
            if (match.get('venue') == 'away' and 
                match.get('opponent_position', 10) <= 5):
                is_pressure = True
                
            # Art arda kötü sonuçlardan sonra
            if match.get('pressure_situation', False):
                is_pressure = True
                
            if is_pressure:
                pressure_matches += 1
                if match.get('goals_scored', 0) >= match.get('goals_conceded', 0):
                    good_results += 1
                    
        if pressure_matches == 0:
            return 'average'
            
        success_rate = good_results / pressure_matches
        
        if success_rate > 0.6:
            return 'excellent'
        elif success_rate > 0.4:
            return 'good'
        elif success_rate > 0.25:
            return 'average'
        else:
            return 'poor'
            
    def _detect_special_circumstances(self, team_info, league_info):
        """
        Özel durumları tespit et
        """
        circumstances = []
        
        if not team_info:
            return circumstances
            
        position = team_info.get('position', 10)
        total_teams = league_info.get('total_teams', 20) if league_info else 20
        matches_played = team_info.get('matches_played', 0)
        matches_remaining = team_info.get('total_matches', 38) - matches_played
        
        # Şampiyonluk yarışı
        if position <= 3 and matches_remaining <= 10:
            circumstances.append('title_race')
            
        # Küme düşme mücadelesi
        if position >= total_teams - 3:
            circumstances.append('relegation_battle')
            
        # Avrupa kupası yarışı
        if 4 <= position <= 7:
            circumstances.append('european_race')
            
        # Sezon sonu
        if matches_remaining <= 5:
            circumstances.append('season_end')
            
        # Kötü seri
        recent_form = team_info.get('recent_form', '')
        if 'LLL' in recent_form:
            circumstances.append('bad_streak')
        elif 'WWW' in recent_form:
            circumstances.append('good_streak')
            
        return circumstances
        
    def _calculate_adjustments(self, opponent_perf, big_match, motivation, pressure):
        """
        Tahmin ayarlama önerilerini hesapla
        """
        adjustments = {
            'goals_modifier': 0,
            'confidence_modifier': 0,
            'risk_factor': 1.0
        }
        
        # Rakip performansı etkisi
        avg_opponent_perf = np.mean(list(opponent_perf.values()))
        if avg_opponent_perf > 1.2:
            adjustments['confidence_modifier'] += 5
        elif avg_opponent_perf < 0.8:
            adjustments['confidence_modifier'] -= 5
            
        # Büyük maç performansı
        if big_match['is_performer']:
            adjustments['confidence_modifier'] += 10
            adjustments['risk_factor'] *= 0.9
            
        # Motivasyon etkisi
        if motivation > 85:
            adjustments['goals_modifier'] += 0.2
        elif motivation < 50:
            adjustments['goals_modifier'] -= 0.2
            
        # Baskı yönetimi
        if pressure == 'excellent':
            adjustments['confidence_modifier'] += 5
        elif pressure == 'poor':
            adjustments['confidence_modifier'] -= 10
            adjustments['risk_factor'] *= 1.2
            
        return adjustments
        
    def _get_opponent_strength(self, match, league_info):
        """
        Rakip gücünü belirle
        """
        # Basit yaklaşım: pozisyon bazlı
        opponent_position = match.get('opponent_position', 10)
        total_teams = league_info.get('total_teams', 20) if league_info else 20
        
        # Normalize edilmiş güç (0-1)
        strength = 1 - ((opponent_position - 1) / (total_teams - 1))
        return strength
        
    def _categorize_opponent(self, strength):
        """
        Rakibi kategorize et
        """
        if strength >= 0.8:
            return 'top'
        elif strength >= 0.6:
            return 'strong'
        elif strength >= 0.2:
            return 'weak'
        else:
            return 'bottom'
            
    def _get_default_situational(self):
        """
        Varsayılan durumsal analiz
        """
        return {
            'opponent_performance': {
                'vs_top_teams': 1.0,
                'vs_strong_teams': 1.0,
                'vs_weak_teams': 1.0,
                'vs_bottom_teams': 1.0
            },
            'big_match_performer': False,
            'big_match_details': {
                'is_performer': False,
                'big_match_points': 0,
                'total_big_matches': 0
            },
            'fixture_congestion_impact': 0,
            'motivation_level': 70,
            'pressure_handling': 'average',
            'special_circumstances': [],
            'performance_adjustments': {
                'goals_modifier': 0,
                'confidence_modifier': 0,
                'risk_factor': 1.0
            }
        }