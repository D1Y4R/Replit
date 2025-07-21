"""
Dynamic Team Analyzer - Ana Modül
Tüm takım analiz bileşenlerini birleştiren ve tahmin ayarlamaları öneren sistem
"""
import logging
import numpy as np
from momentum_analyzer import MomentumAnalyzer
from tactical_profiler import TacticalProfiler
from situational_analyzer import SituationalAnalyzer
from adaptation_tracker import AdaptationTracker

logger = logging.getLogger(__name__)

class DynamicTeamAnalyzer:
    """
    Dinamik takım analiz sistemi
    """
    
    def __init__(self):
        # Alt modülleri başlat
        self.momentum_analyzer = MomentumAnalyzer()
        self.tactical_profiler = TacticalProfiler()
        self.situational_analyzer = SituationalAnalyzer()
        self.adaptation_tracker = AdaptationTracker()
        
        # Ağırlık faktörleri
        self.component_weights = {
            'momentum': 0.30,
            'tactical': 0.25,
            'situational': 0.25,
            'adaptation': 0.20
        }
        
        logger.info("Dynamic Team Analyzer başlatıldı")
        
    def analyze_team(self, team_id, team_matches, team_info=None, league_info=None, is_home=True):
        """
        Takımın kapsamlı analizini yap
        
        Args:
            team_id: Takım ID
            team_matches: Takımın maçları
            team_info: Takım bilgileri (pozisyon, puan vs.)
            league_info: Lig bilgileri
            is_home: Ev sahibi mi?
            
        Returns:
            dict: Kapsamlı takım analizi
        """
        try:
            # Momentum analizi
            momentum_analysis = self.momentum_analyzer.analyze_momentum(
                team_matches,
                is_home_team=is_home
            )
            
            # Taktiksel profil
            tactical_analysis = self.tactical_profiler.analyze_tactical_profile(
                team_matches,
                team_stats=team_info
            )
            
            # Durumsal faktörler
            situational_analysis = self.situational_analyzer.analyze_situational_factors(
                team_matches,
                team_info,
                league_info
            )
            
            # Adaptasyon analizi
            adaptation_analysis = self.adaptation_tracker.analyze_adaptation(
                team_matches,
                team_changes=team_info.get('recent_changes') if team_info else None
            )
            
            # Genel takım skoru
            overall_score = self._calculate_overall_score(
                momentum_analysis,
                tactical_analysis,
                situational_analysis,
                adaptation_analysis
            )
            
            # Tahmin ayarlamaları
            prediction_adjustments = self._calculate_prediction_adjustments(
                momentum_analysis,
                tactical_analysis,
                situational_analysis,
                adaptation_analysis,
                is_home
            )
            
            return {
                'team_id': team_id,
                'is_home': is_home,
                'momentum': momentum_analysis,
                'tactical_profile': tactical_analysis,
                'situational_factors': situational_analysis,
                'adaptation': adaptation_analysis,
                'overall_score': overall_score,
                'prediction_adjustments': prediction_adjustments,
                'summary': self._generate_summary(
                    momentum_analysis,
                    tactical_analysis,
                    situational_analysis,
                    adaptation_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"Takım analizi hatası (ID: {team_id}): {e}")
            return self._get_default_analysis(team_id, is_home)
            
    def compare_teams(self, home_analysis, away_analysis):
        """
        İki takımı karşılaştır ve maç dinamiklerini belirle
        
        Args:
            home_analysis: Ev sahibi analizi
            away_analysis: Deplasman analizi
            
        Returns:
            dict: Karşılaştırma sonuçları
        """
        # Momentum karşılaştırması
        momentum_diff = (home_analysis['momentum']['overall_score'] - 
                        away_analysis['momentum']['overall_score'])
        
        # Taktiksel uyum
        tactical_matchup = self._analyze_tactical_matchup(
            home_analysis['tactical_profile'],
            away_analysis['tactical_profile']
        )
        
        # Motivasyon farkı
        motivation_diff = (home_analysis['situational_factors']['motivation_level'] - 
                          away_analysis['situational_factors']['motivation_level'])
        
        # Adaptasyon karşılaştırması
        adaptation_diff = (home_analysis['adaptation']['adaptation_score'] - 
                          away_analysis['adaptation']['adaptation_score'])
        
        # Maç dinamikleri
        match_dynamics = self._determine_match_dynamics(
            momentum_diff,
            tactical_matchup,
            motivation_diff,
            adaptation_diff
        )
        
        # Kombinlenmiş ayarlamalar
        combined_adjustments = self._combine_adjustments(
            home_analysis['prediction_adjustments'],
            away_analysis['prediction_adjustments'],
            match_dynamics
        )
        
        return {
            'momentum_advantage': 'home' if momentum_diff > 10 else 'away' if momentum_diff < -10 else 'balanced',
            'momentum_diff': momentum_diff,
            'tactical_matchup': tactical_matchup,
            'motivation_diff': motivation_diff,
            'adaptation_diff': adaptation_diff,
            'match_dynamics': match_dynamics,
            'combined_adjustments': combined_adjustments
        }
        
    def _calculate_overall_score(self, momentum, tactical, situational, adaptation):
        """
        Genel takım skorunu hesapla (0-100)
        """
        scores = {
            'momentum': momentum['overall_score'],
            'tactical': self._tactical_to_score(tactical),
            'situational': situational['motivation_level'],
            'adaptation': adaptation['adaptation_score']
        }
        
        # Ağırlıklı ortalama
        weighted_sum = sum(scores[key] * self.component_weights[key] 
                          for key in scores)
        total_weight = sum(self.component_weights.values())
        
        return round(weighted_sum / total_weight, 1)
        
    def _tactical_to_score(self, tactical):
        """
        Taktiksel profili skora çevir
        """
        score = 50  # Temel skor
        
        # Tempo etkisi
        tempo_scores = {
            'very_fast': 15,
            'fast': 10,
            'medium': 0,
            'slow': -5,
            'very_slow': -10
        }
        score += tempo_scores.get(tactical['tempo'], 0)
        
        # Baskı yoğunluğu
        if tactical['pressing_intensity'] == 'high':
            score += 10
        elif tactical['pressing_intensity'] == 'low':
            score -= 5
            
        # Savunma sağlamlığı
        defensive_scores = {
            'very_high': 15,
            'high': 10,
            'medium': 0,
            'low': -10,
            'very_low': -15
        }
        score += defensive_scores.get(tactical['defensive_solidity'], 0)
        
        return min(100, max(0, score))
        
    def _calculate_prediction_adjustments(self, momentum, tactical, situational, adaptation, is_home):
        """
        Tahmin ayarlamalarını hesapla
        """
        adjustments = {
            'goals_expectation': 0,
            'btts_probability': 0,
            'over_2_5_probability': 0,
            'confidence_modifier': 0,
            'exact_score_volatility': 1.0
        }
        
        # Momentum etkisi
        if momentum['overall_score'] > 80:
            adjustments['goals_expectation'] += 0.3
            adjustments['confidence_modifier'] += 5
        elif momentum['overall_score'] < 40:
            adjustments['goals_expectation'] -= 0.2
            adjustments['confidence_modifier'] -= 5
            
        # Form trendi
        if momentum['trend'] == 'ascending':
            adjustments['confidence_modifier'] += 3
        elif momentum['trend'] == 'descending':
            adjustments['confidence_modifier'] -= 3
            
        # Gol trendi
        if momentum['goal_trend']['scoring'] == 'improving':
            adjustments['goals_expectation'] += 0.2
            adjustments['over_2_5_probability'] += 5
        elif momentum['goal_trend']['scoring'] == 'declining':
            adjustments['goals_expectation'] -= 0.2
            adjustments['over_2_5_probability'] -= 5
            
        # Taktiksel faktörler
        if tactical['tempo'] in ['very_fast', 'fast']:
            adjustments['over_2_5_probability'] += 8
            adjustments['btts_probability'] += 5
        elif tactical['tempo'] in ['slow', 'very_slow']:
            adjustments['over_2_5_probability'] -= 8
            adjustments['btts_probability'] -= 5
            
        # Savunma zayıflığı
        if tactical['defensive_solidity'] in ['low', 'very_low']:
            adjustments['btts_probability'] += 10
            adjustments['goals_expectation'] += 0.2
        elif tactical['defensive_solidity'] in ['high', 'very_high']:
            adjustments['btts_probability'] -= 10
            
        # Set parça tehdidi
        if tactical['set_piece_threat'] == 'high':
            adjustments['goals_expectation'] += 0.15
            
        # İkinci yarı performansı
        if tactical['half_performance']['late_goal_tendency'] == 'high':
            adjustments['over_2_5_probability'] += 5
            
        # Durumsal faktörler
        adjustments['goals_expectation'] += situational['performance_adjustments']['goals_modifier']
        adjustments['confidence_modifier'] += situational['performance_adjustments']['confidence_modifier']
        adjustments['exact_score_volatility'] *= situational['performance_adjustments']['risk_factor']
        
        # Büyük maç performansı
        if situational['big_match_performer']:
            adjustments['confidence_modifier'] += 5
            
        # Adaptasyon etkisi
        if adaptation['form_evolution']['trend'] == 'sharp_improvement':
            adjustments['confidence_modifier'] += 5
            adjustments['goals_expectation'] += 0.1
        elif adaptation['form_evolution']['volatility'] == 'high':
            adjustments['exact_score_volatility'] *= 1.2
            
        # Ev/Deplasman faktörü
        if is_home:
            # Ev sahibi form avantajı
            if momentum['venue_form'] > momentum['overall_form']:
                adjustments['confidence_modifier'] += 3
                adjustments['goals_expectation'] += 0.1
        else:
            # Deplasman dezavantajı azaltma
            if momentum['venue_form'] > 70:
                adjustments['confidence_modifier'] += 5
                
        return adjustments
        
    def _analyze_tactical_matchup(self, home_tactical, away_tactical):
        """
        Taktiksel uyumu analiz et
        """
        matchup = {
            'style_compatibility': 'neutral',
            'tempo_clash': 'balanced',
            'tactical_advantage': None,
            'expected_game_flow': 'normal'
        }
        
        # Stil uyumu
        home_style = home_tactical['style']
        away_style = away_tactical['style']
        
        # Avantajlı eşleşmeler
        advantageous_matchups = {
            'defensive_counter': ['attacking_high_press', 'attacking_possession'],
            'attacking_high_press': ['defensive_deep'],
            'balanced': []  # Dengeli stil herkese karşı nötr
        }
        
        # Ev sahibi avantajı kontrolü
        if away_style in advantageous_matchups.get(home_style, []):
            matchup['tactical_advantage'] = 'home'
            matchup['style_compatibility'] = 'favorable_home'
        elif home_style in advantageous_matchups.get(away_style, []):
            matchup['tactical_advantage'] = 'away'
            matchup['style_compatibility'] = 'favorable_away'
            
        # Tempo uyumu
        home_tempo = home_tactical['tempo_details']['avg_total_goals']
        away_tempo = away_tactical['tempo_details']['avg_total_goals']
        
        tempo_diff = abs(home_tempo - away_tempo)
        
        if tempo_diff > 1.0:
            matchup['tempo_clash'] = 'high_contrast'
            # Yavaş takım genelde tempoyu düşürür
            if home_tempo < away_tempo:
                matchup['expected_game_flow'] = 'slow_paced'
            else:
                matchup['expected_game_flow'] = 'home_controlled'
        elif tempo_diff < 0.3:
            matchup['tempo_clash'] = 'similar'
            matchup['expected_game_flow'] = 'open_game'
            
        return matchup
        
    def _determine_match_dynamics(self, momentum_diff, tactical_matchup, motivation_diff, adaptation_diff):
        """
        Maç dinamiklerini belirle
        """
        dynamics = {
            'expected_pattern': 'balanced',
            'key_factors': [],
            'volatility': 'medium',
            'surprise_potential': 'low'
        }
        
        # Momentum farkı etkisi
        if abs(momentum_diff) > 30:
            dynamics['expected_pattern'] = 'one_sided'
            dynamics['key_factors'].append('momentum_gap')
        elif abs(momentum_diff) < 10:
            dynamics['volatility'] = 'high'
            dynamics['surprise_potential'] = 'medium'
            
        # Taktiksel uyum etkisi
        if tactical_matchup['style_compatibility'] != 'neutral':
            dynamics['key_factors'].append('tactical_mismatch')
            if tactical_matchup['tempo_clash'] == 'high_contrast':
                dynamics['expected_pattern'] = 'tactical_battle'
                
        # Motivasyon farkı
        if abs(motivation_diff) > 20:
            dynamics['key_factors'].append('motivation_disparity')
            dynamics['surprise_potential'] = 'high' if motivation_diff < 0 else 'low'
            
        # Adaptasyon farkı
        if adaptation_diff > 20:
            dynamics['key_factors'].append('home_adaptation_edge')
        elif adaptation_diff < -20:
            dynamics['key_factors'].append('away_adaptation_edge')
            
        # Sürpriz potansiyeli hesaplama
        if len(dynamics['key_factors']) >= 3:
            dynamics['surprise_potential'] = 'very_high'
        elif dynamics['volatility'] == 'high' and abs(motivation_diff) > 15:
            dynamics['surprise_potential'] = 'high'
            
        return dynamics
        
    def _combine_adjustments(self, home_adj, away_adj, dynamics):
        """
        Ev sahibi ve deplasman ayarlamalarını birleştir
        """
        combined = {
            'total_goals_modifier': (home_adj['goals_expectation'] + 
                                   away_adj['goals_expectation']) / 2,
            'btts_modifier': (home_adj['btts_probability'] + 
                            away_adj['btts_probability']) / 2,
            'over_2_5_modifier': (home_adj['over_2_5_probability'] + 
                                away_adj['over_2_5_probability']) / 2,
            'confidence_modifier': (home_adj['confidence_modifier'] + 
                                  away_adj['confidence_modifier']) / 2,
            'volatility_factor': (home_adj['exact_score_volatility'] * 
                                away_adj['exact_score_volatility']) ** 0.5
        }
        
        # Dinamik etkiler
        if dynamics['expected_pattern'] == 'one_sided':
            combined['confidence_modifier'] += 5
            combined['volatility_factor'] *= 0.8
        elif dynamics['expected_pattern'] == 'tactical_battle':
            combined['over_2_5_modifier'] -= 5
            combined['volatility_factor'] *= 1.1
            
        # Sürpriz potansiyeli etkisi
        surprise_modifiers = {
            'very_high': 1.3,
            'high': 1.2,
            'medium': 1.1,
            'low': 1.0
        }
        combined['volatility_factor'] *= surprise_modifiers.get(
            dynamics['surprise_potential'], 1.0
        )
        
        return combined
        
    def _generate_summary(self, momentum, tactical, situational, adaptation):
        """
        Analiz özeti oluştur
        """
        summary_parts = []
        
        # Momentum durumu
        if momentum['overall_score'] > 75:
            summary_parts.append("Excellent form")
        elif momentum['overall_score'] < 40:
            summary_parts.append("Poor form")
            
        # Trend
        if momentum['trend'] == 'ascending':
            summary_parts.append("improving trend")
        elif momentum['trend'] == 'descending':
            summary_parts.append("declining trend")
            
        # Taktiksel özellik
        if tactical['style'] == 'attacking_high_press':
            summary_parts.append("aggressive attacking style")
        elif tactical['style'] == 'defensive_counter':
            summary_parts.append("counter-attacking approach")
            
        # Motivasyon
        if situational['motivation_level'] > 85:
            summary_parts.append("highly motivated")
        elif situational['motivation_level'] < 50:
            summary_parts.append("low motivation")
            
        # Adaptasyon
        if adaptation['adaptation_score'] > 70:
            summary_parts.append("excellent adaptation")
            
        return ", ".join(summary_parts) if summary_parts else "Balanced profile"
        
    def _get_default_analysis(self, team_id, is_home):
        """
        Varsayılan analiz değerleri
        """
        return {
            'team_id': team_id,
            'is_home': is_home,
            'momentum': self.momentum_analyzer._get_default_momentum(),
            'tactical_profile': self.tactical_profiler._get_default_profile(),
            'situational_factors': self.situational_analyzer._get_default_situational(),
            'adaptation': self.adaptation_tracker._get_default_adaptation(),
            'overall_score': 50,
            'prediction_adjustments': {
                'goals_expectation': 0,
                'btts_probability': 0,
                'over_2_5_probability': 0,
                'confidence_modifier': 0,
                'exact_score_volatility': 1.0
            },
            'summary': "Default analysis - insufficient data"
        }