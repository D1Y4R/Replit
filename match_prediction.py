import logging
import json
import os
from datetime import datetime
import requests
import time
import numpy as np

# Algoritmalar
from algorithms import (
    XGCalculator,
    EloSystem,
    PoissonModel,
    DixonColesModel,
    XGBoostModel,
    MonteCarloSimulator,
    EnsemblePredictor,
    CRFPredictor,
    SelfLearningModel
)

# Yeni tahmin algoritmaları
from algorithms.halftime_predictor import HalfTimeFullTimePredictor
from algorithms.handicap_predictor import HandicapPredictor
from algorithms.goal_range_predictor import GoalRangePredictor
from algorithms.double_chance_predictor import DoubleChancePredictor
from algorithms.team_goals_predictor import TeamGoalsPredictor

# Yeni geliştirme modülleri
from model_evaluator import ModelEvaluator
from continuous_learner import ContinuousLearner
from advanced_features import AdvancedFeatureEngineer
from distributed_trainer import DistributedTrainer
from model_validator import ComprehensiveValidator
from explainable_ai import PredictionExplainer
from performance_optimizer import (
    prediction_cache, performance_monitor, 
    batch_processor, query_optimizer
)
from async_data_fetcher import AsyncDataFetcher
from dynamic_team_analyzer import DynamicTeamAnalyzer

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchPredictor:
    """
    Gelişmiş futbol maç tahmin sistemi
    Çoklu algoritma ve ensemble yaklaşımı
    """
    
    def __init__(self):
        """Tahmin sınıfını ve algoritmalarını başlat"""
        logger.info("MatchPredictor gelişmiş sürüm başlatılıyor...")
        
        # Algoritmaları başlat
        self.xg_calculator = XGCalculator()
        self.elo_system = EloSystem()
        self.poisson_model = PoissonModel()
        self.dixon_coles = DixonColesModel()
        self.xgboost_model = XGBoostModel()
        self.monte_carlo = MonteCarloSimulator()
        self.ensemble = EnsemblePredictor()
        self.crf_predictor = CRFPredictor()
        self.self_learning = SelfLearningModel()
        
        # Neural Network modelini ekle
        from algorithms.neural_network import NeuralNetworkModel
        self.neural_network = NeuralNetworkModel()
        
        # Yeni tahmin algoritmaları
        self.htft_predictor = HalfTimeFullTimePredictor()
        self.handicap_predictor = HandicapPredictor()
        self.goal_range_predictor = GoalRangePredictor()
        self.double_chance_predictor = DoubleChancePredictor()
        self.team_goals_predictor = TeamGoalsPredictor()
        
        # Geliştirme modülleri
        self.model_evaluator = ModelEvaluator()
        self.continuous_learner = ContinuousLearner()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.distributed_trainer = DistributedTrainer()
        self.model_validator = ComprehensiveValidator()
        self.prediction_explainer = PredictionExplainer()
        self.async_fetcher = AsyncDataFetcher()
        self.dynamic_team_analyzer = DynamicTeamAnalyzer()
        
        # Tek JSON dosyası kullan
        self.cache_file = 'predictions_cache.json'
        self.cache_data = self._load_cache()
            
        logger.info("Tüm algoritmalar ve geliştirme modülleri başlatıldı")
        
    def _load_cache(self):
        """Önbellek dosyasını yükle"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
        
    def predict_match(self, home_team_id, away_team_id, home_name="Ev Sahibi", away_name="Deplasman", force_update=False):
        """
        Gelişmiş maç tahmini - tüm algoritmaları kullanır
        
        Args:
            home_team_id: Ev sahibi takım ID
            away_team_id: Deplasman takım ID
            home_name: Ev sahibi takım adı
            away_name: Deplasman takım adı
            force_update: Önbelleği yoksay
            
        Returns:
            dict: Tahmin sonuçları
        """
        start_time = time.time()
        logger.info(f"Tahmin başlatılıyor: {home_name} vs {away_name}")
        
        # Performans optimizasyonu - Önbellek kontrolü
        cache_key = f"{home_team_id}_{away_team_id}"
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        if not force_update:
            # Gelişmiş önbellek kontrolü
            cached = prediction_cache.get_prediction(home_team_id, away_team_id, date_str)
            if cached:
                performance_monitor.record_cache_access(hit=True)
                logger.info("Önbellekten tahmin döndürülüyor")
                return cached
            performance_monitor.record_cache_access(hit=False)
                
        try:
            # 1. Takım verilerini al
            home_data = self._get_team_data(home_team_id, home_name, is_home=True)
            away_data = self._get_team_data(away_team_id, away_name, is_home=False)
            
            # 1.3. Dynamic Team Analyzer ile takım analizleri
            home_team_analysis = None
            away_team_analysis = None
            team_comparison = None
            
            try:
                # Takım bilgilerini hazırla
                home_team_info = {
                    'position': home_data.get('league_position', 10),
                    'recent_form': home_data.get('recent_form', 'DDDDD'),
                    'matches_played': len(home_data.get('recent_matches', [])),
                    'total_matches': 38  # Varsayılan
                }
                
                away_team_info = {
                    'position': away_data.get('league_position', 10),
                    'recent_form': away_data.get('recent_form', 'DDDDD'),
                    'matches_played': len(away_data.get('recent_matches', [])),
                    'total_matches': 38  # Varsayılan
                }
                
                # Takım analizlerini yap
                home_team_analysis = self.dynamic_team_analyzer.analyze_team(
                    team_id=home_team_id,
                    team_matches=home_data.get('recent_matches', []),
                    team_info=home_team_info,
                    is_home=True
                )
                
                away_team_analysis = self.dynamic_team_analyzer.analyze_team(
                    team_id=away_team_id,
                    team_matches=away_data.get('recent_matches', []),
                    team_info=away_team_info,
                    is_home=False
                )
                
                # Takımları karşılaştır
                team_comparison = self.dynamic_team_analyzer.compare_teams(
                    home_team_analysis,
                    away_team_analysis
                )
                
                logger.info(f"Dynamic Team Analyzer tamamlandı - Ev: {home_team_analysis['overall_score']}, Dep: {away_team_analysis['overall_score']}")
                logger.info(f"Momentum avantajı: {team_comparison['momentum_advantage']}")
                
            except Exception as e:
                logger.warning(f"Dynamic Team Analyzer hatası: {e}")
            
            # 1.5. H2H verilerini al
            h2h_data = None
            try:
                # API anahtarını al
                from api_config import APIConfig
                api_config = APIConfig()
                api_key = api_config.get_api_key()
                
                # Asenkron veri çekme
                import asyncio
                async def fetch_h2h():
                    async with self.async_fetcher as fetcher:
                        return await fetcher.fetch_h2h_data(home_team_id, away_team_id, api_key, home_name, away_name)
                
                # H2H verilerini çek
                h2h_data = asyncio.run(fetch_h2h())
                logger.info(f"H2H verileri başarıyla alındı: {home_name} vs {away_name}")
                # H2H veri yapısını logla
                if h2h_data:
                    logger.info(f"H2H veri yapısı anahtarları: {list(h2h_data.keys())[:5]}")
                    if isinstance(h2h_data, dict) and 'firstTeam_VS_secondTeam' in h2h_data:
                        logger.info(f"H2H maç sayısı: {len(h2h_data['firstTeam_VS_secondTeam'])}")
                    elif isinstance(h2h_data, list):
                        logger.info(f"H2H doğrudan liste, maç sayısı: {len(h2h_data)}")
            except Exception as e:
                logger.warning(f"H2H verileri alınamadı: {e}")
                h2h_data = None
            
            # 2. Elo hesapla
            home_elo = self.elo_system.calculate_team_elo(
                home_team_id, home_data.get('recent_matches', [])
            )
            away_elo = self.elo_system.calculate_team_elo(
                away_team_id, away_data.get('recent_matches', [])
            )
            elo_diff = home_elo - away_elo
            
            # 3. xG/xGA hesapla - Elo entegrasyonu ile (rapordaki öneri)
            home_xg, home_xga = self.xg_calculator.calculate_xg_xga_with_elo(
                home_data.get('recent_matches', []), 
                home_elo, 
                away_elo,
                is_home=True
            )
            away_xg, away_xga = self.xg_calculator.calculate_xg_xga_with_elo(
                away_data.get('recent_matches', []),
                away_elo,
                home_elo, 
                is_home=False
            )
            
            # 4. Lambda değerlerini hesapla (çaprazlama)
            lambda_home, lambda_away = self.xg_calculator.calculate_lambda_cross(
                home_xg, home_xga, away_xg, away_xga, elo_diff
            )
            
            # Maç bağlamı - Ekstrem maç bilgilerini ekle
            match_context = {
                'lambda_home': lambda_home,
                'lambda_away': lambda_away,
                'elo_diff': elo_diff,
                'home_xg': home_xg,
                'home_xga': home_xga,
                'away_xg': away_xg,
                'away_xga': away_xga,
                # Ekstrem maç için istatistikler
                'home_stats': {
                    'xg': home_xg,
                    'xga': home_xga,
                    'avg_goals_scored': home_data.get('home_performance', {}).get('avg_goals', 1.5),
                    'avg_goals_conceded': home_data.get('home_performance', {}).get('avg_conceded', 1.0),
                    'form': [m.get('goals_scored', 0) for m in home_data.get('recent_matches', [])[:5]]
                },
                'away_stats': {
                    'xg': away_xg,
                    'xga': away_xga,
                    'avg_goals_scored': away_data.get('away_performance', {}).get('avg_goals', 1.2),
                    'avg_goals_conceded': away_data.get('away_performance', {}).get('avg_conceded', 1.3),
                    'form': [m.get('goals_scored', 0) for m in away_data.get('recent_matches', [])[:5]]
                }
            }
            
            # 4.5. Gelişmiş özellik mühendisliği
            advanced_features = self.feature_engineer.extract_all_features(
                home_data, 
                away_data, 
                match_context
            )
            
            # 5. Tüm modelleri çalıştır
            model_predictions = {}
            
            # Poisson Model
            poisson_matrix = self.poisson_model.calculate_probability_matrix(
                lambda_home, lambda_away, elo_diff
            )
            model_predictions['poisson'] = self._process_poisson_results(poisson_matrix, lambda_home, lambda_away)
            
            # Dixon-Coles Model
            dc_matrix = self.dixon_coles.calculate_probability_matrix(
                lambda_home, lambda_away, elo_diff
            )
            model_predictions['dixon_coles'] = self._process_dixon_coles_results(dc_matrix, lambda_home, lambda_away)
            
            # XGBoost Model
            xg_features = self.xgboost_model.prepare_features(home_data, away_data, match_context)
            model_predictions['xgboost'] = self.xgboost_model.predict(xg_features)
            
            # Monte Carlo Simülasyonu - takım ID'leri ile
            mc_results = self.monte_carlo.run_simulations(
                lambda_home, lambda_away, elo_diff, 
                home_id=home_team_id, away_id=away_team_id
            )
            model_predictions['monte_carlo'] = self._process_monte_carlo_results(mc_results)
            
            # CRF Model
            crf_features = self.crf_predictor.prepare_features(
                home_data, away_data, lambda_home, lambda_away, elo_diff
            )
            model_predictions['crf'] = self.crf_predictor.predict(crf_features)
            
            # Neural Network Model
            nn_features = self.neural_network.prepare_features(
                home_data, away_data, match_context, match_context
            )
            model_predictions['neural_network'] = self.neural_network.predict(nn_features)
            
            # Self-Learning model context'i kullanarak ağırlıkları al
            is_extreme = lambda_home + lambda_away > 5.0
            dynamic_context = {
                'is_extreme': is_extreme,
                'expected_total_goals': lambda_home + lambda_away,
                'elo_diff': elo_diff
            }
            
            # 6. Ensemble birleştirme - dinamik ağırlıklarla
            algorithm_weights = self.self_learning.get_dynamic_weights(dynamic_context)
            final_prediction = self.ensemble.combine_predictions(
                model_predictions, match_context, algorithm_weights
            )
            
            # 6.5. Dynamic Team Analyzer ayarlamalarını uygula
            if team_comparison:
                adjustments = team_comparison['combined_adjustments']
                
                # Lambda değerlerini ayarla
                original_lambda_home = lambda_home
                original_lambda_away = lambda_away
                lambda_home += lambda_home * adjustments['total_goals_modifier']
                lambda_away += lambda_away * adjustments['total_goals_modifier']
                
                # BTTS (KG) tahminini ayarla
                if 'both_teams_to_score' in final_prediction:
                    btts_prob = final_prediction['both_teams_to_score']['yes']
                    btts_adjustment = adjustments['btts_modifier'] / 100.0
                    new_btts_yes = max(0, min(100, btts_prob + btts_adjustment))
                    final_prediction['both_teams_to_score']['yes'] = new_btts_yes
                    final_prediction['both_teams_to_score']['no'] = 100 - new_btts_yes
                
                # Over/Under tahminlerini ayarla
                if 'over_under' in final_prediction:
                    ou_adjustment = adjustments['over_2_5_modifier'] / 100.0
                    for market in final_prediction['over_under']:
                        if market['threshold'] == 2.5:
                            over_prob = market['over']
                            new_over = max(0, min(100, over_prob + ou_adjustment))
                            market['over'] = new_over
                            market['under'] = 100 - new_over
                
                # Güven skorunu ayarla
                if 'confidence' in final_prediction:
                    conf_adjustment = adjustments['confidence_modifier']
                    final_prediction['confidence'] = max(0, min(100, 
                        final_prediction['confidence'] + conf_adjustment))
                
                # Volatilite faktörünü kaydet
                final_prediction['volatility_factor'] = adjustments['volatility_factor']
                
                logger.info(f"Dynamic Team Analyzer ayarlamaları uygulandı:")
                logger.info(f"  Lambda ayarı: {adjustments['total_goals_modifier']:+.2f}")
                logger.info(f"  BTTS ayarı: {adjustments['btts_modifier']:+.0f}%")
                logger.info(f"  O/U 2.5 ayarı: {adjustments['over_2_5_modifier']:+.0f}%")
                logger.info(f"  Güven ayarı: {adjustments['confidence_modifier']:+.0f}%")
            
            # 7. Yeni tahmin türlerini hesapla
            # HT/FT tahminleri
            htft_predictions = self.htft_predictor.predict_htft(
                home_data, away_data, lambda_home, lambda_away, elo_diff
            )
            
            # İlk yarı gol tahminleri
            halftime_goals = self.htft_predictor.predict_halftime_goals(
                home_data, away_data, lambda_home, lambda_away
            )
            
            # Handikap tahminleri
            asian_handicap = self.handicap_predictor.predict_asian_handicap(
                home_xg, away_xg, elo_diff,
                ''.join(self._analyze_form(home_data.get('recent_matches', [])[:5])),
                ''.join(self._analyze_form(away_data.get('recent_matches', [])[:5]))
            )
            
            european_handicap = self.handicap_predictor.predict_european_handicap(
                home_xg, away_xg, elo_diff, final_prediction
            )
            
            # Gol aralığı tahminleri
            goal_ranges = self.goal_range_predictor.predict_goal_ranges(
                lambda_home, lambda_away, match_context
            )
            
            # Toplam gol marketleri
            total_goals_markets = self.goal_range_predictor.predict_total_goals_markets(
                lambda_home, lambda_away
            )
            
            # Çifte şans tahminleri
            double_chance = self.double_chance_predictor.predict_double_chance(final_prediction)
            
            # Takım gol tahminleri
            # Savunma gücü hesaplama: xGA/xG oranı (1'den küçük = iyi savunma, 1'den büyük = kötü savunma)
            # Min 0.5, Max 2.0 sınırları ile
            # Ev sahibi savunması: home_xga/home_xg
            # Deplasman savunması: away_xga/away_xg
            home_defense_strength = max(0.5, min(2.0, home_xga / home_xg)) if home_xg > 0 else 1.0
            away_defense_strength = max(0.5, min(2.0, away_xga / away_xg)) if away_xg > 0 else 1.0
            
            # Debug log
            logger.info(f"Savunma gücü hesaplama:")
            logger.info(f"  - Ev sahibi xG: {home_xg:.2f}, xGA: {home_xga:.2f}")
            logger.info(f"  - Deplasman xG: {away_xg:.2f}, xGA: {away_xga:.2f}")
            logger.info(f"  - Ev sahibi savunma gücü: {home_defense_strength:.2f}")
            logger.info(f"  - Deplasman savunma gücü: {away_defense_strength:.2f}")
            
            team_goals = self.team_goals_predictor.predict_both_teams_goals(
                lambda_home, lambda_away, home_name, away_name,
                home_defense=home_defense_strength,  # Ev sahibi savunması
                away_defense=away_defense_strength   # Deplasman savunması
            )
            
            # Tahminleri final_prediction'a ekle
            final_prediction['advanced_predictions'] = {
                'htft': htft_predictions,
                'halftime_goals': halftime_goals,
                'asian_handicap': asian_handicap,
                'european_handicap': european_handicap,
                'goal_ranges': goal_ranges,
                'total_goals_markets': total_goals_markets,
                'double_chance': double_chance,
                'team_goals': team_goals
            }
            
            # 7. Ekstrem maç kontrolü ve düzeltme
            from algorithms.extreme_detector import ExtremeMatchDetector
            detector = ExtremeMatchDetector()
            
            is_extreme, extreme_details = detector.is_extreme_match(
                match_context['home_stats'], 
                match_context['away_stats']
            )
            
            if is_extreme:
                # Ekstrem maç tahminlerini validate et
                final_prediction = detector.validate_extreme_prediction(
                    final_prediction,
                    match_context['home_stats'],
                    match_context['away_stats']
                )
                logger.info(f"Ekstrem maç düzeltmesi uygulandı: {extreme_details['indicators']}")
            
            # 7. Sonuç formatla
            prediction = self._format_prediction(
                final_prediction, match_context, home_name, away_name, 
                home_team_id, away_team_id, home_data, away_data, h2h_data,
                home_team_analysis, away_team_analysis, team_comparison
            )
            
            # 8. Açıklanabilir AI
            try:
                # Model ve özellik vektörü hazırla
                features = np.array([
                    home_xg,
                    away_xg,
                    home_xga,
                    away_xga,
                    elo_diff,
                    advanced_features.get('form_momentum', {}).get('home', {}).get('composite_score', 0),
                    advanced_features.get('form_momentum', {}).get('away', {}).get('composite_score', 0),
                    advanced_features.get('form_momentum', {}).get('differential', 0),
                    advanced_features.get('goal_dynamics', {}).get('home', {}).get('scoring_trend', 0),
                    advanced_features.get('advanced_context', {}).get('match_importance', 0.5)
                ]).reshape(1, -1)
                
                explanation = self.prediction_explainer.explain_prediction(
                    prediction['predictions'],
                    model=self.xgboost_model.model_1x2 if hasattr(self.xgboost_model, 'model_1x2') else None,
                    features=features
                )
                prediction['explanation'] = explanation
            except Exception as e:
                logger.warning(f"Açıklama oluşturulamadı: {e}")
            
            # 9. Sürekli öğrenme (gerçek sonuç geldiğinde çalışacak)
            
            # Hesaplama süresi
            prediction['calculation_time'] = round(time.time() - start_time, 2)
            
            # Performans kayıt
            performance_monitor.record_prediction_time('ensemble', prediction['calculation_time'])
            
            # Gelişmiş önbelleğe kaydet
            prediction_cache.set_prediction(home_team_id, away_team_id, date_str, prediction)
            
            logger.info(f"Tahmin tamamlandı ({prediction['calculation_time']}s): {prediction['predictions']['most_likely_outcome']}")
            return prediction
            
        except Exception as e:
            logger.error(f"Tahmin hatası: {str(e)}", exc_info=True)
            # Hata durumunda basit tahmin döndür
            return self._get_fallback_prediction(home_team_id, away_team_id, home_name, away_name)
            
    def _get_team_data(self, team_id, team_name, is_home=True):
        """
        Takım verilerini API'den al veya varsayılan kullan
        """
        try:
            # API'den gerçek takım verilerini almayı dene
            import requests
            from datetime import datetime, timedelta
            from api_config import APIConfig
            
            # API anahtarını config'den al
            api_config = APIConfig()
            api_key = api_config.get_api_key()
            
            if not api_key:
                logger.warning("API anahtarı bulunamadı")
                raise Exception("API anahtarı yok")
                
            url = "https://apiv3.apifootball.com/"
            
            # Son 4 ayın maçlarını al (120 gün) - daha fazla veri ile güvenilir tahmin
            date_from = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')
            date_to = datetime.now().strftime('%Y-%m-%d')
            
            params = {
                'action': 'get_events',
                'team_id': team_id,
                'from': date_from,
                'to': date_to,
                'APIkey': api_key
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                matches = response.json()
                logger.info(f"API yanıtı alındı takım {team_id} için: {len(matches) if isinstance(matches, list) else 0} maç")
                if isinstance(matches, list) and len(matches) > 0:
                    # Gerçek maç verilerini işle
                    recent_matches = []
                    home_goals = []
                    home_conceded = []
                    away_goals = []
                    away_conceded = []
                    
                    for match in matches:  # Tüm mevcut maçları al - maksimum veri ile tahmin
                        home_score = int(match.get('match_hometeam_score', 0) or 0)
                        away_score = int(match.get('match_awayteam_score', 0) or 0)
                        
                        # Bu takım ev sahibi mi deplasman mı?
                        if str(match.get('match_hometeam_id')) == str(team_id):
                            recent_matches.append({
                                'goals_scored': home_score,
                                'goals_conceded': away_score,
                                'date': match.get('match_date', '')
                            })
                            home_goals.append(home_score)
                            home_conceded.append(away_score)
                        else:
                            recent_matches.append({
                                'goals_scored': away_score,
                                'goals_conceded': home_score,
                                'date': match.get('match_date', '')
                            })
                            away_goals.append(away_score)
                            away_conceded.append(home_score)
                    
                    # Performans istatistikleri hesapla
                    home_avg_goals = sum(home_goals) / len(home_goals) if home_goals else 1.3
                    home_avg_conceded = sum(home_conceded) / len(home_conceded) if home_conceded else 1.3
                    away_avg_goals = sum(away_goals) / len(away_goals) if away_goals else 1.0
                    away_avg_conceded = sum(away_conceded) / len(away_conceded) if away_conceded else 1.3
                    
                    return {
                        'team_id': team_id,
                        'team_name': team_name,
                        'recent_matches': recent_matches,
                        'home_performance': {
                            'avg_goals': home_avg_goals,
                            'avg_conceded': home_avg_conceded
                        },
                        'away_performance': {
                            'avg_goals': away_avg_goals,
                            'avg_conceded': away_avg_conceded
                        }
                    }
            else:
                logger.warning(f"API'den veri alınamadı takım {team_id} için, yanıt kodu: {response.status_code}")
        except Exception as e:
            logger.error(f"API veri alımı başarısız takım {team_id} için: {e}")
        
        # API başarısız oldu - hata fırlat
        logger.error(f"Takım {team_id} için gerçek veri alınamadı!")
        raise Exception(f"API'den takım {team_id} ({team_name}) için veri alınamadı. Lütfen API anahtarını kontrol edin veya daha sonra tekrar deneyin.")
        

        
    def _process_poisson_results(self, matrix, lambda_home, lambda_away):
        """
        Poisson sonuçlarını işle
        """
        match_probs = self.poisson_model.get_match_probabilities(matrix)
        goal_probs = self.poisson_model.get_goals_probabilities(matrix)
        scores = self.poisson_model.get_exact_score_probabilities(matrix)
        
        # Dinamik güven hesaplama
        max_prob = max(match_probs['home_win'], match_probs['draw'], match_probs['away_win'])
        
        # Tahmin keskinliğine göre güven (0.4-0.9 arası)
        if max_prob > 60:  # Çok net favori
            confidence = 0.7 + (max_prob - 60) / 100  # 0.7-0.9
        elif max_prob > 45:  # Orta düzey favori
            confidence = 0.6 + (max_prob - 45) / 75  # 0.6-0.7
        else:  # Dengeli maç
            confidence = 0.5 + (max_prob - 33) / 60  # 0.5-0.6
        
        # Poisson modeli için temel güven
        confidence = min(0.85, max(0.5, confidence))
        
        return {
            'home_win': match_probs['home_win'],
            'draw': match_probs['draw'],
            'away_win': match_probs['away_win'],
            'over_2_5': goal_probs['over_2_5'],
            'under_2_5': goal_probs['under_2_5'],
            'btts_yes': goal_probs['both_teams_score_yes'],
            'btts_no': goal_probs['both_teams_score_no'],
            'expected_goals': {
                'home': lambda_home,
                'away': lambda_away
            },
            'score_probabilities': scores,
            'confidence': round(confidence, 2)
        }
        
    def _process_dixon_coles_results(self, matrix, lambda_home, lambda_away):
        """
        Dixon-Coles sonuçlarını işle
        """
        match_probs = self.dixon_coles.get_match_probabilities(matrix)
        
        # Gol tahminleri için Poisson fonksiyonlarını kullan
        goal_probs = self.poisson_model.get_goals_probabilities(matrix)
        scores = self.poisson_model.get_exact_score_probabilities(matrix)
        
        # Dinamik güven hesaplama
        max_prob = max(match_probs['home_win'], match_probs['draw'], match_probs['away_win'])
        
        # Tahmin keskinliğine göre güven (0.4-0.9 arası)
        if max_prob > 60:  # Çok net favori
            confidence = 0.75 + (max_prob - 60) / 100  # 0.75-0.95
        elif max_prob > 45:  # Orta düzey favori
            confidence = 0.65 + (max_prob - 45) / 75  # 0.65-0.75
        else:  # Dengeli maç
            confidence = 0.55 + (max_prob - 33) / 60  # 0.55-0.65
        
        # Dixon-Coles modeli için temel güven
        confidence = min(0.88, max(0.5, confidence))
        
        return {
            'home_win': match_probs['home_win'],
            'draw': match_probs['draw'],
            'away_win': match_probs['away_win'],
            'over_2_5': goal_probs['over_2_5'],
            'under_2_5': goal_probs['under_2_5'],
            'btts_yes': goal_probs['both_teams_score_yes'],
            'btts_no': goal_probs['both_teams_score_no'],
            'expected_goals': {
                'home': lambda_home,
                'away': lambda_away
            },
            'score_probabilities': scores,
            'confidence': round(confidence, 2)
        }
        
    def _process_monte_carlo_results(self, results):
        """
        Monte Carlo sonuçlarını işle
        """
        # Dinamik güven hesaplama
        max_prob = max(results['outcomes']['home_win'], results['outcomes']['draw'], results['outcomes']['away_win'])
        
        # Tahmin keskinliğine göre güven (0.4-0.9 arası)
        if max_prob > 60:  # Çok net favori
            confidence = 0.68 + (max_prob - 60) / 100  # 0.68-0.88
        elif max_prob > 45:  # Orta düzey favori
            confidence = 0.58 + (max_prob - 45) / 75  # 0.58-0.68
        else:  # Dengeli maç
            confidence = 0.48 + (max_prob - 33) / 60  # 0.48-0.58
        
        # Monte Carlo modeli için temel güven
        confidence = min(0.82, max(0.45, confidence))
        
        return {
            'home_win': results['outcomes']['home_win'],
            'draw': results['outcomes']['draw'],
            'away_win': results['outcomes']['away_win'],
            'over_2_5': results['over_under']['over_2_5'],
            'under_2_5': results['over_under']['under_2_5'],
            'btts_yes': results['btts']['yes'],
            'btts_no': results['btts']['no'],
            'expected_goals': {
                'home': results['avg_home_goals'],
                'away': results['avg_away_goals']
            },
            'score_probabilities': self._convert_mc_scores(results['scores']),
            'confidence': round(confidence, 2)
        }
        
    def _convert_mc_scores(self, scores_dict):
        """
        Monte Carlo skor dict'ini listeye çevir
        """
        scores_list = []
        for score, prob in sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:5]:
            scores_list.append({
                'score': score,
                'probability': prob
            })
        return scores_list
        
    def _format_prediction(self, final_pred, context, home_name, away_name, home_id, away_id, home_data, away_data, h2h_data=None, home_team_analysis=None, away_team_analysis=None, team_comparison=None):
        """
        Tahmin sonuçlarını frontend formatına dönüştür
        """
        # En olası skor
        most_likely_score = "1-1"
        most_likely_prob = 0.0
        
        if 'most_likely_scores' in final_pred and final_pred['most_likely_scores']:
            most_likely = final_pred['most_likely_scores'][0]
            most_likely_score = most_likely['score']
            most_likely_prob = most_likely['probability']
            
        # Form analizi
        home_form = self._analyze_form(home_data.get('recent_matches', [])[:5])
        away_form = self._analyze_form(away_data.get('recent_matches', [])[:5])
        
        return {
            "match_info": {
                "home_team": {
                    "id": home_id,
                    "name": home_name
                },
                "away_team": {
                    "id": away_id,
                    "name": away_name
                },
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "predictions": {
                "most_likely_outcome": final_pred['most_likely_outcome'],
                "home_win_probability": round(final_pred['home_win'], 1),
                "draw_probability": round(final_pred['draw'], 1),
                "away_win_probability": round(final_pred['away_win'], 1),
                "most_likely_score": most_likely_score,
                "most_likely_score_probability": round(most_likely_prob, 1),
                "expected_goals": {
                    "home": round(final_pred['expected_goals']['home'], 2),
                    "away": round(final_pred['expected_goals']['away'], 2)
                },
                "over_under": {
                    "over_2_5": round(final_pred['over_2_5'], 1),
                    "under_2_5": round(final_pred['under_2_5'], 1)
                },
                "both_teams_to_score": {
                    "yes": round(final_pred['btts_yes'], 1),
                    "no": round(final_pred['btts_no'], 1)
                },
                "exact_scores": final_pred.get('most_likely_scores', []),
                # Frontend için gerekli ekstra alanlar
                "betting_predictions": self._generate_betting_predictions(final_pred),
                "most_confident_bet": self._get_most_confident_bet(final_pred),
                "most_likely_bet": self._get_most_likely_bet(final_pred),
                # Yeni tahmin türleri
                "advanced_predictions": final_pred.get('advanced_predictions', {})
            },
            "team_stats": {
                "home": {
                    "form": home_form,
                    "elo_rating": round(context.get('home_elo', 1500)),
                    "xg": round(context['home_xg'], 2),
                    "xga": round(context['home_xga'], 2)
                },
                "away": {
                    "form": away_form,
                    "elo_rating": round(context.get('away_elo', 1500)),
                    "xg": round(context['away_xg'], 2),
                    "xga": round(context['away_xga'], 2)
                }
            },
            "confidence": round(final_pred['confidence'], 2),
            "algorithm": "Ensemble (Poisson + Dixon-Coles + XGBoost + Monte Carlo + CRF + Neural Network)",
            "elo_difference": round(context['elo_diff']),
            "analysis": self._generate_analysis(final_pred, context, home_name, away_name),
            # Açıklanabilir AI
            "explanation": None,  # Daha sonra eklenecek
            # Model performans raporu
            "model_performance": self.model_evaluator.get_model_performance_report(),
            "team_data": {
                "home": {
                    "form": home_form[:5] if home_form else [],
                    "avg_goals_scored": round(home_data['home_performance']['avg_goals'], 1),
                    "avg_goals_conceded": round(home_data['home_performance']['avg_conceded'], 1),
                    "avg_goals_scored_away": round(home_data['away_performance']['avg_goals'], 1),
                    "recent_form": ''.join(home_form[:5]) if home_form else "WWDLW",
                    "strength": self._calculate_team_strength(home_data, home_form),
                    "motivation": self._calculate_team_motivation(home_data, home_form, context.get('home_elo', 1500)),
                    "fatigue": self._calculate_team_fatigue(home_data),
                    "h2h_record": home_data.get('h2h_record', {"wins": 2, "draws": 1, "losses": 2})
                },
                "away": {
                    "form": away_form[:5] if away_form else [],
                    "avg_goals_scored": round(away_data['away_performance']['avg_goals'], 1),
                    "avg_goals_conceded": round(away_data['away_performance']['avg_conceded'], 1), 
                    "avg_goals_scored_away": round(away_data['away_performance']['avg_goals'], 1),
                    "recent_form": ''.join(away_form[:5]) if away_form else "LWDWL",
                    "strength": self._calculate_team_strength(away_data, away_form),
                    "motivation": self._calculate_team_motivation(away_data, away_form, context.get('away_elo', 1500)),
                    "fatigue": self._calculate_team_fatigue(away_data),
                    "h2h_record": away_data.get('h2h_record', {"wins": 2, "draws": 1, "losses": 2})
                }
            },
            # H2H verileri eklendi
            "h2h_data": {
                "matches": h2h_data.get('response', {}).get('matches', []) if h2h_data and h2h_data.get('success') else []
            },
            # Dynamic Team Analyzer verileri
            "dynamic_analysis": {
                "home_team": home_team_analysis if home_team_analysis else None,
                "away_team": away_team_analysis if away_team_analysis else None,
                "comparison": team_comparison if team_comparison else None
            }
        }
        
    def _calculate_team_strength(self, team_data, form):
        """
        Takım gücünü dinamik olarak hesapla (0-100 arası)
        """
        base_strength = 50
        
        # Form bazlı güç (son 5 maç)
        if form:
            wins = form[:5].count('W')
            draws = form[:5].count('D')
            form_points = (wins * 3 + draws) / 15  # Max 15 puan mümkün
            base_strength += form_points * 20  # Max +20 puan
        
        # Gol performansı
        home_perf = team_data.get('home_performance', {})
        away_perf = team_data.get('away_performance', {})
        avg_goals = (home_perf.get('avg_goals', 1.2) + away_perf.get('avg_goals', 1.0)) / 2
        avg_conceded = (home_perf.get('avg_conceded', 1.3) + away_perf.get('avg_conceded', 1.5)) / 2
        
        # Gol farkı bazlı güç
        goal_diff = avg_goals - avg_conceded
        base_strength += goal_diff * 10  # Gol farkı başına +/-10 puan
        
        # Elo rating etkisi
        elo = team_data.get('elo_rating', 1500)
        elo_factor = (elo - 1500) / 50  # Her 50 Elo puanı için +/-1 güç puanı
        base_strength += elo_factor
        
        # 0-100 arasında sınırla
        return max(0, min(100, round(base_strength)))
    
    def _calculate_team_motivation(self, team_data, form, elo_rating):
        """
        Takım motivasyonunu dinamik olarak hesapla (0-100 arası)
        """
        base_motivation = 50
        
        # Son form trendi (momentum)
        if form and len(form) >= 3:
            recent_wins = form[:3].count('W')
            if recent_wins >= 2:
                base_motivation += 15  # Güçlü momentum
            elif recent_wins == 0 and form[:3].count('L') >= 2:
                base_motivation -= 10  # Kötü momentum
        
        # Gol atma performansı
        recent_matches = team_data.get('recent_matches', [])
        if recent_matches:
            recent_goals = sum(m.get('goals_scored', 0) for m in recent_matches)
            if recent_goals > 10:  # Son 5 maçta 10+ gol
                base_motivation += 10
            elif recent_goals < 3:  # Son 5 maçta 3'ten az gol
                base_motivation -= 10
        
        # Rakip kalitesi (Elo bazlı)
        if elo_rating > 1600:
            base_motivation += 5  # Güçlü takım bonusu
        elif elo_rating < 1400:
            base_motivation -= 5  # Zayıf takım cezası
        
        # 0-100 arasında sınırla
        return max(0, min(100, round(base_motivation)))
    
    def _calculate_team_fatigue(self, team_data):
        """
        Takım yorgunluğunu dinamik olarak hesapla (0-100 arası, yüksek = daha yorgun)
        """
        base_fatigue = 20
        
        recent_matches = team_data.get('recent_matches', [])
        if not recent_matches:
            return base_fatigue
        
        # Son 7 gündeki maç sayısı
        from datetime import datetime, timedelta
        today = datetime.now()
        matches_in_week = 0
        
        for match in recent_matches:
            match_date_str = match.get('date', '')
            if match_date_str:
                try:
                    match_date = datetime.strptime(match_date_str, '%Y-%m-%d')
                    if (today - match_date).days <= 7:
                        matches_in_week += 1
                except:
                    continue
        
        # Her ekstra maç için +15 yorgunluk
        if matches_in_week > 1:
            base_fatigue += (matches_in_week - 1) * 15
        
        # Seyahat faktörü (son 5 maçta deplasman sayısı)
        away_matches = sum(1 for m in recent_matches if not m.get('is_home', True))
        base_fatigue += away_matches * 5  # Her deplasman maçı için +5 yorgunluk
        
        # 0-100 arasında sınırla
        return max(0, min(100, round(base_fatigue)))
    
    def _analyze_form(self, matches):
        """
        Son maçların form analizini yap
        """
        if not matches:
            return []
            
        form = []
        for match in matches:
            goals_for = match.get('goals_scored', 0)
            goals_against = match.get('goals_conceded', 0)
            
            if goals_for > goals_against:
                form.append('W')
            elif goals_for == goals_against:
                form.append('D')
            else:
                form.append('L')
                
        return form
        
    def _generate_analysis(self, prediction, context, home_name, away_name):
        """
        Tahmin analizi metni oluştur
        """
        analysis = []
        
        # Favori analizi
        if prediction['most_likely_outcome'] == 'HOME_WIN':
            fav_team = home_name
            fav_prob = prediction['home_win']
        elif prediction['most_likely_outcome'] == 'AWAY_WIN':
            fav_team = away_name
            fav_prob = prediction['away_win']
        else:
            fav_team = None
            fav_prob = prediction['draw']
            
        if fav_team:
            analysis.append(f"{fav_team} maçın favorisi (%{fav_prob:.0f} kazanma şansı)")
        else:
            analysis.append(f"Dengeli bir maç bekleniyor (%{fav_prob:.0f} beraberlik olasılığı)")
            
        # Gol analizi
        total_goals = prediction['expected_goals']['home'] + prediction['expected_goals']['away']
        if total_goals > 2.5:
            analysis.append(f"Gollü bir maç bekleniyor (Ort. {total_goals:.1f} gol)")
        else:
            analysis.append(f"Düşük skorlu bir maç olabilir (Ort. {total_goals:.1f} gol)")
            
        # KG analizi
        if prediction['btts_yes'] > 60:
            analysis.append(f"Her iki takımın da gol atma ihtimali yüksek (%{prediction['btts_yes']:.0f})")
            
        # Elo analizi
        elo_diff = abs(context['elo_diff'])
        if elo_diff > 300:
            analysis.append("Takımlar arasında belirgin bir güç farkı var")
        elif elo_diff < 100:
            analysis.append("Takımlar güç olarak birbirine yakın")
            
        return " ".join(analysis)
        
    def _get_cached_prediction(self, cache_key):
        """
        Önbellekten tahmin al
        """
        if cache_key in self.cache_data:
            # 1 saatten eski önbellekleri yoksay
            cache_time = self.cache_data[cache_key].get('timestamp', 0)
            if time.time() - cache_time > 3600:
                return None
            
            return self.cache_data[cache_key]
                
        return None
        
    def _cache_prediction(self, cache_key, prediction):
        """
        Tahmini önbelleğe kaydet
        """
        try:
            # Timestamp ekle
            prediction['timestamp'] = time.time()
            
            # Önbelleğe ekle
            self.cache_data[cache_key] = prediction
            
            # Dosyaya kaydet
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Önbellek kayıt hatası: {e}")
    
    def get_async_predictions(self, match_ids):
        """
        Birden çok maç için asenkron tahmin
        """
        import asyncio
        from async_data_fetcher import run_async_workflow
        
        logger.info(f"{len(match_ids)} maç için asenkron tahmin başlatılıyor")
        
        # API anahtarını al
        from api_config import APIConfig
        api_config = APIConfig()
        api_key = api_config.get_api_key()
        
        # Asenkron workflow'u çalıştır
        results = run_async_workflow(
            match_ids, 
            api_key, 
            lambda match_data: self.predict_match(
                match_data['home_team_id'],
                match_data['away_team_id']
            )
        )
        
        return results
            
    def _get_fallback_prediction(self, home_id, away_id, home_name, away_name):
        """
        Hata durumunda basit tahmin döndür
        """
        return {
            "match_info": {
                "home_team": {"id": home_id, "name": home_name},
                "away_team": {"id": away_id, "name": away_name},
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "predictions": {
                "most_likely_outcome": "DRAW",
                "home_win_probability": 33.3,
                "draw_probability": 33.4,
                "away_win_probability": 33.3,
                "most_likely_score": "1-1",
                "most_likely_score_probability": 10.0,
                "expected_goals": {"home": 1.2, "away": 1.2},
                "over_under": {"over_2_5": 45.0, "under_2_5": 55.0},
                "both_teams_to_score": {"yes": 50.0, "no": 50.0}
            },
            "confidence": 0.5,
            "algorithm": "Fallback (Basit tahmin)",
            "error": True
        }
        
    def clear_cache(self):
        """
        Önbellek temizleme
        """
        try:
            self.cache_data = {}
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)
            logger.info("Önbellek temizlendi")
            return True
        except Exception as e:
            logger.error(f"Önbellek temizleme hatası: {e}")
            return False
            
    def _generate_betting_predictions(self, prediction):
        """
        Frontend için bahis tahminlerini oluştur
        """
        betting_preds = {}
        
        # Maç sonucu
        betting_preds['match_result'] = {
            'prediction': prediction['most_likely_outcome'],
            'probability': max(prediction['home_win'], prediction['draw'], prediction['away_win'])
        }
        
        # KG Var/Yok - her zaman yüksek olasılığı göster
        if prediction['btts_yes'] > prediction['btts_no']:
            betting_preds['both_teams_to_score'] = {
                'prediction': 'YES',
                'probability': prediction['btts_yes']
            }
        else:
            betting_preds['both_teams_to_score'] = {
                'prediction': 'NO',
                'probability': prediction['btts_no']
            }
        
        # 2.5 Üst/Alt - her zaman yüksek olasılığı göster
        if prediction['over_2_5'] > prediction['under_2_5']:
            betting_preds['over_2_5_goals'] = {
                'prediction': 'YES',
                'probability': prediction['over_2_5']
            }
        else:
            betting_preds['over_2_5_goals'] = {
                'prediction': 'NO',
                'probability': prediction['under_2_5']
            }
        
        # 3.5 Üst/Alt - her zaman yüksek olasılığı göster
        over_3_5 = prediction.get('over_3_5', prediction['over_2_5'] * 0.7)  # Tahmini değer
        under_3_5 = 100 - over_3_5
        if over_3_5 > under_3_5:
            betting_preds['over_3_5_goals'] = {
                'prediction': 'YES',
                'probability': over_3_5
            }
        else:
            betting_preds['over_3_5_goals'] = {
                'prediction': 'NO',
                'probability': under_3_5
            }
        
        # Kesin skor
        if prediction.get('most_likely_scores'):
            betting_preds['exact_score'] = {
                'prediction': prediction['most_likely_scores'][0]['score'],
                'probability': prediction['most_likely_scores'][0]['probability']
            }
        else:
            betting_preds['exact_score'] = {
                'prediction': '1-1',
                'probability': 10.0
            }
            
        return betting_preds
        
    def _get_most_confident_bet(self, prediction):
        """
        En yüksek olasılıklı bahis tahmini
        """
        all_bets = []
        
        # Maç sonucu
        all_bets.append({
            'market': 'match_result',
            'prediction': prediction['most_likely_outcome'],
            'probability': max(prediction['home_win'], prediction['draw'], prediction['away_win'])
        })
        
        # KG Var/Yok - her zaman yüksek olasılığı göster
        if prediction['btts_yes'] > prediction['btts_no']:
            btts_pred = 'YES'
            btts_prob = prediction['btts_yes']
        else:
            btts_pred = 'NO'
            btts_prob = prediction['btts_no']
            
        all_bets.append({
            'market': 'both_teams_to_score',
            'prediction': btts_pred,
            'probability': btts_prob
        })
        
        # 2.5 Üst/Alt - her zaman yüksek olasılığı göster
        if prediction['over_2_5'] > prediction['under_2_5']:
            over_pred = 'YES'
            over_prob = prediction['over_2_5']
        else:
            over_pred = 'NO'
            over_prob = prediction['under_2_5']
            
        all_bets.append({
            'market': 'over_2_5_goals',
            'prediction': over_pred,
            'probability': over_prob
        })
        
        # En yüksek olasılıklı olanı seç
        return max(all_bets, key=lambda x: x['probability'])
        
    def _get_most_likely_bet(self, prediction):
        """
        En olası bahis (frontend uyumluluk için)
        """
        confident = self._get_most_confident_bet(prediction)
        return f"{confident['market']}:{confident['prediction']}"
        
    def clear_cache(self):
        """
        Önbellek temizleme - şimdilik boş template
        """
        logger.info("Önbellek temizlendi (template)")
        return True