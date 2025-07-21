import logging
import numpy as np
import json
import os
import math
from datetime import datetime, timedelta
import requests
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
# Konsensüs filtresi kaldırıldı - tutarsızlık sorunlarını çözmek için

# Gelişmiş makine öğrenmesi modelleri için import
# Global değişkenler
ADVANCED_MODELS_AVAILABLE = False
TEAM_SPECIFIC_MODELS_AVAILABLE = False
ENHANCED_MONTE_CARLO_AVAILABLE = False
SPECIALIZED_MODELS_AVAILABLE = False
ENHANCED_FACTORS_AVAILABLE = True
GOAL_TREND_ANALYZER_AVAILABLE = False

try:
    from advanced_ml_models import AdvancedPredictionModels, BayesianNetwork
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    # Modül bulunamadıysa uyarı ver ama çalışmaya devam et
    logging.warning("Gelişmiş tahmin modelleri (advanced_ml_models) bulunamadı! Basit modeller kullanılacak.")
    
try:
    from goal_trend_analyzer import get_instance as get_goal_trend_analyzer
    GOAL_TREND_ANALYZER_AVAILABLE = True
    logging.info("Gol Trend İvmesi Analiz modülü başarıyla yüklendi!")
except ImportError:
    # Modül bulunamadıysa uyarı ver ama çalışmaya devam et
    logging.warning("Gol Trend İvmesi Analiz modülü (goal_trend_analyzer) bulunamadı! Bu özellik kullanılamayacak.")
    
# Gelişmiş Monte Carlo simülasyonu için import
try:
    from improved_monte_carlo import EnhancedMonteCarlo
    ENHANCED_MONTE_CARLO_AVAILABLE = True
except ImportError:
    logging.warning("Gelişmiş Monte Carlo simülasyonu (improved_monte_carlo) bulunamadı! Basit simülasyon kullanılacak.")

# Özelleştirilmiş modeller (düşük, orta, yüksek skorlu maçlar) için import
try:
    from specialized_models import SpecializedModels
    SPECIALIZED_MODELS_AVAILABLE = True
    logging.info("Özelleştirilmiş tahmin modelleri (düşük/orta/yüksek skor) kullanıma hazır.")
except ImportError:
    logging.warning("Özelleştirilmiş tahmin modelleri (specialized_models) bulunamadı! Genel model kullanılacak.")

# Takım-spesifik modeller için import
try:
    from team_specific_models import TeamSpecificPredictor
    TEAM_SPECIFIC_MODELS_AVAILABLE = True
except ImportError:
    # Modül bulunamadıysa uyarı ver ama çalışmaya devam et
    logging.warning("Takım-spesifik tahmin modelleri (team_specific_models) bulunamadı! Genel modeller kullanılacak.")
    
# Gelişmiş tahmin faktörleri için import
try:
    from enhanced_prediction_factors import get_instance as get_enhanced_factors
    ENHANCED_FACTORS_AVAILABLE = True
except ImportError:
    # Modül bulunamadıysa uyarı ver ama çalışmaya devam et
    logging.warning("Gelişmiş tahmin faktörleri (enhanced_prediction_factors) bulunamadı! Temel tahmin faktörleri kullanılacak.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchPredictor:
    def __init__(self):
        self.api_key = '39bc8dd65153ff5c7c0f37b4939009de04f4a70c593ee1aea0b8f70dd33268c0'
        self.predictions_cache = {}
        self._cache_modified = False
        self.load_cache()

        # Bayesyen güncelleme için parametreler
        self.lig_ortalamasi_ev_gol = 1.5  # Ev sahibi takımların lig genelinde maç başına ortalama gol
        self.lig_ortalamasi_deplasman_gol = 1.2  # Deplasman takımlarının lig genelinde maç başına ortalama gol
        self.k_ev = 5  # Ev sahibi prior gücü (daha fazla veri geldikçe etkisi azalır)
        self.k_deplasman = 5  # Deplasman prior gücü

        # Gamma dağılımı için prior parametreler
        self.alpha_ev_atma = self.k_ev * self.lig_ortalamasi_ev_gol  # Gamma dağılımı için alpha
        self.beta_ev = self.k_ev  # Gamma dağılımı için beta
        self.alpha_deplasman_atma = self.k_deplasman * self.lig_ortalamasi_deplasman_gol
        self.beta_deplasman = self.k_deplasman

        # Sinir ağı için standardizasyon ve model
        self.scaler = StandardScaler()
        self.model_home = None
        
        # Özelleştirilmiş Modeller (düşük/orta/yüksek skorlu maçlar)
        self.specialized_models = None
        self.model_away = None
        
        # Dinamik input_dim belirleme
        sample_form = {'home_performance': {}, 'bayesian': {}, 'recent_matches': 0, 'home_matches': 0}
        sample_features = self.prepare_data_for_neural_network(sample_form, is_home=True)
        self.input_dim = len(sample_features[0]) if sample_features is not None else 10

        # Modeli yükleme veya oluşturma
        self.load_or_create_models()
        
        # Gelişmiş makine öğrenmesi modelleri
        if 'ADVANCED_MODELS_AVAILABLE' in globals() and globals()['ADVANCED_MODELS_AVAILABLE']:
            try:
                logger.info("Gelişmiş tahmin modelleri yükleniyor...")
                self.advanced_models = AdvancedPredictionModels()
                self.bayesian_network = BayesianNetwork()
                logger.info("Gelişmiş tahmin modelleri başarıyla yüklendi.")
            except Exception as e:
                logger.error(f"Gelişmiş tahmin modelleri yüklenirken hata: {str(e)}")
                globals()['ADVANCED_MODELS_AVAILABLE'] = False
        
        # Gelişmiş Monte Carlo simülasyonu
        if 'ENHANCED_MONTE_CARLO_AVAILABLE' in globals() and globals()['ENHANCED_MONTE_CARLO_AVAILABLE']:
            try:
                logger.info("Gelişmiş Monte Carlo simülasyonu yükleniyor...")
                self.enhanced_monte_carlo = EnhancedMonteCarlo()
                logger.info("Gelişmiş Monte Carlo simülasyonu başarıyla yüklendi.")
            except Exception as e:
                logger.error(f"Gelişmiş Monte Carlo simülasyonu yüklenirken hata: {str(e)}")
                globals()['ENHANCED_MONTE_CARLO_AVAILABLE'] = False
        
        # Takım-spesifik tahmin modelleri
        if 'TEAM_SPECIFIC_MODELS_AVAILABLE' in globals() and globals()['TEAM_SPECIFIC_MODELS_AVAILABLE']:
            try:
                logger.info("Takım-spesifik tahmin modelleri yükleniyor...")
                self.team_specific_predictor = TeamSpecificPredictor()
                logger.info("Takım-spesifik tahmin modelleri başarıyla yüklendi.")
            except Exception as e:
                logger.error(f"Takım-spesifik tahmin modelleri yüklenirken hata: {str(e)}")
                globals()['TEAM_SPECIFIC_MODELS_AVAILABLE'] = False
                
        # Özelleştirilmiş modeller (düşük, orta ve yüksek skorlu maçlar için)
        if 'SPECIALIZED_MODELS_AVAILABLE' in globals() and globals()['SPECIALIZED_MODELS_AVAILABLE']:
            try:
                logger.info("Özelleştirilmiş tahmin modelleri (düşük/orta/yüksek skorlu maçlar) yükleniyor...")
                self.specialized_models = SpecializedModels()
                logger.info("Özelleştirilmiş tahmin modelleri başarıyla yüklendi.")
            except Exception as e:
                logger.error(f"Özelleştirilmiş tahmin modelleri yüklenirken hata: {str(e)}")
                globals()['SPECIALIZED_MODELS_AVAILABLE'] = False
                
        # Gelişmiş tahmin faktörleri
        if 'ENHANCED_FACTORS_AVAILABLE' in globals() and globals()['ENHANCED_FACTORS_AVAILABLE']:
            try:
                logger.info("Gelişmiş tahmin faktörleri yükleniyor...")
                self.enhanced_factors = get_enhanced_factors()
                logger.info("Gelişmiş tahmin faktörleri başarıyla yüklendi.")
            except Exception as e:
                logger.error(f"Gelişmiş tahmin faktörleri yüklenirken hata: {str(e)}")
                globals()['ENHANCED_FACTORS_AVAILABLE'] = False
        
        # Gol Trend İvmesi Analizi
        if 'GOAL_TREND_ANALYZER_AVAILABLE' in globals() and globals()['GOAL_TREND_ANALYZER_AVAILABLE']:
            try:
                logger.info("Gol Trend İvmesi Analiz modülü yükleniyor...")
                self.goal_trend_analyzer = get_goal_trend_analyzer()
                logger.info("Gol Trend İvmesi Analiz modülü başarıyla yüklendi.")
            except Exception as e:
                logger.error(f"Gol Trend İvmesi Analiz modülü yüklenirken hata: {str(e)}")
                globals()['GOAL_TREND_ANALYZER_AVAILABLE'] = False

    def load_cache(self):
        """Daha önce yapılan tahminleri yükle"""
        try:
            if os.path.exists('predictions_cache.json'):
                with open('predictions_cache.json', 'r', encoding='utf-8') as f:
                    self.predictions_cache = json.load(f)
                logger.info(f"Tahmin önbelleği yüklendi: {len(self.predictions_cache)} tahmin")
        except Exception as e:
            logger.error(f"Tahmin önbelleği yüklenirken hata: {str(e)}")

    def clear_cache(self):
        """Tahmin önbelleğini temizle"""
        try:
            # Önbelleği sıfırla
            self.predictions_cache = {}

            # Önbellek dosyasını güvenli bir şekilde temizle
            try:
                with open('predictions_cache.json', 'w', encoding='utf-8') as f:
                    json.dump({}, f, ensure_ascii=False)
                logger.info("Önbellek dosyası başarıyla temizlendi.")
            except Exception as cache_file_error:
                logger.error(f"Önbellek dosyası yazılırken hata: {str(cache_file_error)}")
                # Dosya yazma hatası olsa bile devam et

            logger.info("Önbellek temizlendi, yeni tahminler yapılabilir.")
            return True
        except Exception as e:
            logger.error(f"Önbellek temizlenirken hata: {str(e)}")
            return False

    def save_cache(self):
        """Yapılan tahminleri kaydet - önbellek değişime uğradıysa"""
        try:
            # Önbellek değiştirilmediyse kaydetmeye gerek yok - performans iyileştirmesi
            if not self._cache_modified:
                logger.debug("Önbellek değişmediği için kaydetmeye gerek yok")
                return
            
            # NumPy değerlerini JSON uyumlu değerlere dönüştür
            def numpy_to_python(obj):
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, list):
                    return [numpy_to_python(item) for item in obj]
                elif isinstance(obj, dict):
                    return {key: numpy_to_python(value) for key, value in obj.items()}
                return obj
                
            with open('predictions_cache.json', 'w', encoding='utf-8') as f:
                json.dump(numpy_to_python(self.predictions_cache), f, ensure_ascii=False, indent=2)
            
            logger.info(f"Tahmin önbelleği kaydedildi: {len(self.predictions_cache)} tahmin")
            
            # Önbellek kaydedildiği için değişiklik durumunu sıfırla
            self._cache_modified = False
        except Exception as e:
            logger.error(f"Tahmin önbelleği kaydedilirken hata: {str(e)}")

    def is_big_team(self, team_name):
        """Büyük takım kontrolü"""
        big_teams = [
            # İspanya büyükleri
            'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Real Sociedad', 'Villarreal',
            
            # Almanya büyükleri
            'Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen', 'Eintracht Frankfurt',
            
            # İngiltere büyükleri
            'Manchester City', 'Manchester United', 'Liverpool', 'Chelsea', 'Arsenal', 'Tottenham', 'West Ham', 'Aston Villa', 'Brighton',
            
            # Fransa büyükleri
            'PSG', 'Lille', 'Monaco', 'Lyon', 'Marseille', 'Stade Rennes',
            
            # İtalya büyükleri
            'Inter', 'Milan', 'Juventus', 'Napoli', 'Roma', 'Lazio', 'Atalanta', 'Bologna',
            
            # Türkiye büyükleri
            'Galatasaray', 'Fenerbahce', 'Besiktas', 'Trabzonspor',
            
            # Portekiz büyükleri
            'Benfica', 'Porto', 'Sporting', 'Braga',
            
            # Hollanda büyükleri
            'Ajax', 'PSV', 'Feyenoord', 'AZ Alkmaar',
            
            # Belçika büyükleri
            'Club Brugge', 'Royal Antwerp', 'Anderlecht', 'Gent',
            
            # İskoçya büyükleri
            'Celtic', 'Rangers',
            
            # Yunanistan büyükleri
            'Olympiacos', 'Panathinaikos', 'AEK Athens', 'PAOK',
            
            # İsviçre büyükleri
            'Young Boys', 'Basel', 'Servette',
            
            # Avusturya büyükleri
            'Red Bull Salzburg', 'Sturm Graz', 'Rapid Wien',
            
            # Çek Cumhuriyeti büyükleri
            'Slavia Prague', 'Sparta Prague', 'Viktoria Plzen',
            
            # Ukrayna büyükleri
            'Shakhtar Donetsk', 'Dynamo Kyiv',
            
            # Hırvatistan büyükleri
            'Dinamo Zagreb', 'Hajduk Split'
        ]
        return team_name in big_teams

    def apply_team_specific_adjustments(self, home_team_id, away_team_id, home_team_name, away_team_name, 
                                   home_goals, away_goals, home_form=None, away_form=None, use_goal_trend_analysis=True):
        """Takım-spesifik tahmin modeli ayarlamaları ve Gol Trend İvmesi analizi"""
        
        # Dinamik takım analizörünü kullan
        try:
            from dynamic_team_analyzer import DynamicTeamAnalyzer
            analyzer = DynamicTeamAnalyzer()
            
            # Önce dinamik faktörleri uygula - varsa
            dynamic_home_goals, dynamic_away_goals = analyzer.apply_dynamic_factors(
                str(home_team_id), str(away_team_id), home_goals, away_goals
            )
            
            # Dinamik faktörler uygulandıysa değerleri güncelle
            if dynamic_home_goals != home_goals or dynamic_away_goals != away_goals:
                logger.info(f"Dinamik faktörler uygulandı: {home_team_name} vs {away_team_name}")
                home_goals, away_goals = dynamic_home_goals, dynamic_away_goals
                
        except Exception as e:
            # Dinamik analiz modülü yoksa veya hata oluşursa, statik değerleri kullan
            logger.warning(f"Dinamik takım analizi uygulanamadı: {str(e)}. Statik değerler kullanılacak.")
        
        # Statik takım asimetrileri için yedek olarak tutulan değerler
        # (dinamik analiz çalışmazsa kullanılır)
        home_away_asymmetries = {
            # Ev sahibiyken çok güçlü, deplasmanken zayıf olan takımlar
            "610": {"name": "Galatasaray", "home_factor": 1.40, "away_factor": 0.85},   # Galatasaray
            "1005": {"name": "Fenerbahçe", "home_factor": 1.35, "away_factor": 0.90},   # Fenerbahçe
            "614": {"name": "Beşiktaş", "home_factor": 1.30, "away_factor": 0.90},      # Beşiktaş
            "636": {"name": "Trabzonspor", "home_factor": 1.35, "away_factor": 0.85},   # Trabzonspor
            "611": {"name": "Rizespor", "home_factor": 1.30, "away_factor": 0.75},      # Rizespor
            
            # Ev sahibi/deplasman performansı arasında daha az fark olan takımlar
            "6010": {"name": "Başakşehir", "home_factor": 1.15, "away_factor": 0.95},   # Başakşehir
            "632": {"name": "Konyaspor", "home_factor": 1.20, "away_factor": 0.90},     # Konyaspor
            "1020": {"name": "Alanyaspor", "home_factor": 1.25, "away_factor": 0.90}    # Alanyaspor
        }
        
        # Savunma zafiyeti yaşayan takımlar için rakiplerinin gol beklentisini artır
        # Aynı zamanda bu takımların kendilerinin de gol atma olasılığını bir miktar artır
        # Adana Demirspor, Rizespor, Kasimpasa gibi savunması zayıf takımlar
        defensive_weak_teams = {
            "7667": {"name": "Adana Demirspor", "factor": 1.35, "offensive_factor": 1.15},  # Adana Demirspor
            "621": {"name": "Kasimpasa", "factor": 1.30, "offensive_factor": 1.12},         # Kasimpasa
            "611": {"name": "Rizespor", "factor": 1.28, "offensive_factor": 1.10},          # Rizespor
            "607": {"name": "Antalyaspor", "factor": 1.20, "offensive_factor": 1.08},       # Antalyaspor
            "629": {"name": "Samsunspor", "factor": 1.25, "offensive_factor": 1.10}         # Samsunspor
        }
        
        # Takımların ev/deplasman asimetrilerini uygula
        if str(home_team_id) in home_away_asymmetries:
            team_info = home_away_asymmetries[str(home_team_id)]
            # Ev sahibi takımın ev avantajını uygula
            original_home_goals = home_goals
            home_goals = home_goals * team_info["home_factor"]
            logger.info(f"Ev sahibi takım ({team_info['name']}) evinde daha güçlü olduğu için, ev gol beklentisi {original_home_goals:.2f} -> {home_goals:.2f} olarak güncellendi.")
        
        if str(away_team_id) in home_away_asymmetries:
            team_info = home_away_asymmetries[str(away_team_id)]
            # Deplasman takımının deplasman dezavantajını uygula
            original_away_goals = away_goals
            away_goals = away_goals * team_info["away_factor"]
            logger.info(f"Deplasman takımı ({team_info['name']}) deplasmanda daha zayıf olduğu için, deplasman gol beklentisi {original_away_goals:.2f} -> {away_goals:.2f} olarak güncellendi.")
        
        # Ev sahibi takım savunma zafiyeti yaşayan takımlar listesinde mi?
        if str(home_team_id) in defensive_weak_teams:
            team_info = defensive_weak_teams[str(home_team_id)]
            # Rakibin gol beklentisini artır
            original_away_goals = away_goals
            away_goals = away_goals * team_info["factor"]
            logger.info(f"Ev sahibi takım ({team_info['name']}) savunma zafiyeti nedeniyle, deplasman gol beklentisi {original_away_goals:.2f} -> {away_goals:.2f} olarak güncellendi.")
            
            # Kendi gol beklentisini de bir miktar artır
            original_home_goals = home_goals
            home_goals = home_goals * team_info["offensive_factor"]
            logger.info(f"Ev sahibi takım ({team_info['name']}) hücum odaklı olduğu için, kendi gol beklentisi {original_home_goals:.2f} -> {home_goals:.2f} olarak artırıldı.")
        
        # Deplasman takımı savunma zafiyeti yaşayan takımlar listesinde mi?
        if str(away_team_id) in defensive_weak_teams:
            team_info = defensive_weak_teams[str(away_team_id)]
            # Rakibin gol beklentisini artır
            original_home_goals = home_goals
            home_goals = home_goals * team_info["factor"]
            logger.info(f"Deplasman takımı ({team_info['name']}) savunma zafiyeti nedeniyle, ev sahibi gol beklentisi {original_home_goals:.2f} -> {home_goals:.2f} olarak güncellendi.")
            
            # Kendi gol beklentisini de bir miktar artır
            original_away_goals = away_goals
            away_goals = away_goals * team_info["offensive_factor"]
            logger.info(f"Deplasman takımı ({team_info['name']}) hücum odaklı olduğu için, kendi gol beklentisi {original_away_goals:.2f} -> {away_goals:.2f} olarak artırıldı.")
        
        # Gol Trend İvmesi analizini uygula (eğer isteniyorsa)
        if use_goal_trend_analysis and 'GOAL_TREND_ANALYZER_AVAILABLE' in globals() and globals()['GOAL_TREND_ANALYZER_AVAILABLE'] and hasattr(self, 'goal_trend_analyzer'):
            try:
                logger.info(f"Gol Trend İvmesi analizi uygulanıyor: {home_team_name} vs {away_team_name}")
                
                # Gol trend faktörlerini hesapla
                trend_factors = self.goal_trend_analyzer.calculate_goal_trend_factors(home_form, away_form)
                
                # Orijinal gol beklentilerini sakla
                original_home_goals = home_goals
                original_away_goals = away_goals
                
                # Gol trend faktörlerine göre beklenen golleri ayarla
                home_goals, away_goals = self.goal_trend_analyzer.adjust_expected_goals(
                    home_goals, away_goals, trend_factors
                )
                
                # Değişimleri logla
                home_change_pct = ((home_goals / original_home_goals) - 1) * 100 if original_home_goals > 0 else 0
                away_change_pct = ((away_goals / original_away_goals) - 1) * 100 if original_away_goals > 0 else 0
                
                logger.info(f"Gol Trend İvmesi analizi sonuçları:")
                logger.info(f"  Ev sahibi: {trend_factors['home_scoring_factor']:.2f} (atma) x {trend_factors['away_conceding_factor']:.2f} (yeme)")
                logger.info(f"  Deplasman: {trend_factors['away_scoring_factor']:.2f} (atma) x {trend_factors['home_conceding_factor']:.2f} (yeme)")
                logger.info(f"  Gol beklentisi değişimi: Ev {original_home_goals:.2f}->{home_goals:.2f} (%{home_change_pct:.1f}), "
                          f"Deplasman {original_away_goals:.2f}->{away_goals:.2f} (%{away_change_pct:.1f})")
                logger.info(f"  Analiz açıklaması: {trend_factors['match_outcome_adjustment']['description']}")
                
            except Exception as e:
                logger.error(f"Gol Trend İvmesi analizi uygulanırken hata: {str(e)}")
                logger.warning(f"Gol Trend İvmesi analizi uygulanamadı, değişiklik yapılmadan devam ediliyor.")
        
        # Orijinal takım-spesifik modeller (eğer varsa)
        if 'TEAM_SPECIFIC_MODELS_AVAILABLE' not in globals() or not globals()['TEAM_SPECIFIC_MODELS_AVAILABLE'] or not hasattr(self, 'team_specific_predictor'):
            logger.warning("Takım-spesifik modeller kullanılamıyor, güncellenmiş dinamik değerler kullanılacak.")
            return home_goals, away_goals
        
        try:
            # Takım-spesifik ayarlamaları al
            adjustments = self.team_specific_predictor.get_team_adjustments(
                home_team_id, away_team_id, home_team_data=home_form, away_team_data=away_form
            )
            
            if not adjustments:
                logger.warning("Takım-spesifik ayarlamalar alınamadı, standart değerler kullanılacak.")
                return home_goals, away_goals
            
            # Takım spesifik çarpanları uygula
            home_multiplier = adjustments.get('home_goal_multiplier', 1.0)
            away_multiplier = adjustments.get('away_goal_multiplier', 1.0)
            draw_bias = adjustments.get('draw_bias', 0.0)
            
            original_home_goals = home_goals
            original_away_goals = away_goals
            
            # Gol değerlerini ayarla
            home_goals = home_goals * home_multiplier
            away_goals = away_goals * away_multiplier
            
            # Beraberlik yanlılığı uygula (değerler birbirine yaklaştır)
            if draw_bias > 0:
                # Gol farkını hesapla
                goal_diff = abs(home_goals - away_goals)
                
                # Eğer draw_bias 0'dan büyükse ve gol farkı 1'den az ise
                # yakın skorlu sonuç bekliyoruz demektir, değerleri daha da yaklaştır
                if goal_diff < 1.0:
                    avg_goals = (home_goals + away_goals) / 2
                    home_goals = avg_goals + (home_goals - avg_goals) * (1 - draw_bias)
                    away_goals = avg_goals + (away_goals - avg_goals) * (1 - draw_bias)
                    logger.info(f"Beraberlik yanlılığı uygulandı (bias={draw_bias:.2f}): "
                               f"Fark {goal_diff:.2f}'dan {abs(home_goals - away_goals):.2f}'a düşürüldü")
            
            # Lig-spesifik ayarlamaları uygula
            league_type = adjustments.get('league_type', 'normal')
            if league_type == 'high_scoring':
                # Yüksek skorlu lig - gol beklentilerini artır
                score_inflation = adjustments.get('score_inflation', 1.1)
                home_goals *= score_inflation
                away_goals *= score_inflation
                logger.info(f"Yüksek skorlu lig ayarlaması ({league_type}): Gol beklentileri %{(score_inflation-1)*100:.0f} artırıldı")
            elif league_type == 'low_scoring':
                # Düşük skorlu lig - gol beklentilerini azalt
                score_deflation = adjustments.get('score_deflation', 0.9)
                home_goals *= score_deflation
                away_goals *= score_deflation
                logger.info(f"Düşük skorlu lig ayarlaması ({league_type}): Gol beklentileri %{(1-score_deflation)*100:.0f} azaltıldı")
            elif league_type == 'home_advantage':
                # Ev sahibi avantajı yüksek lig
                home_advantage = adjustments.get('home_advantage', 1.15)
                home_goals *= home_advantage
                logger.info(f"Ev sahibi avantajı yüksek lig ({league_type}): Ev sahibi gol beklentisi %{(home_advantage-1)*100:.0f} artırıldı")
            elif league_type == 'away_advantage':
                # Deplasman avantajı yüksek lig
                away_advantage = adjustments.get('away_advantage', 1.1)
                away_goals *= away_advantage
                logger.info(f"Deplasman avantajı yüksek lig ({league_type}): Deplasman gol beklentisi %{(away_advantage-1)*100:.0f} artırıldı")
                
            # Özel takım ayarlamaları
            team_style_home = adjustments.get('home_team_style', {})
            team_style_away = adjustments.get('away_team_style', {})
            
            # Ev sahibi takım stili ayarlamaları
            if team_style_home:
                # Defansif takım
                if team_style_home.get('defensive', 0) > 0.6:
                    defensive_factor = team_style_home.get('defensive_factor', 0.9)
                    away_goals *= defensive_factor
                    logger.info(f"Defansif ev sahibi takım: Deplasman gol beklentisi %{(1-defensive_factor)*100:.0f} azaltıldı")
                
                # Ofansif takım
                if team_style_home.get('offensive', 0) > 0.6:
                    offensive_factor = team_style_home.get('offensive_factor', 1.1)
                    home_goals *= offensive_factor
                    logger.info(f"Ofansif ev sahibi takım: Ev sahibi gol beklentisi %{(offensive_factor-1)*100:.0f} artırıldı")
                    
                # Kontrol oyunu takımı
                if team_style_home.get('possession', 0) > 0.7:
                    possession_factor_h = team_style_home.get('possession_factor', 1.05)
                    possession_factor_a = 1 - ((possession_factor_h - 1) * 2)
                    home_goals *= possession_factor_h
                    away_goals *= possession_factor_a
                    logger.info(f"Kontrol oyunu oynayan ev sahibi takım: Ev sahibi gol beklentisi %{(possession_factor_h-1)*100:.0f} artırıldı, "
                              f"deplasman gol beklentisi %{(1-possession_factor_a)*100:.0f} azaltıldı")
            
            # Deplasman takım stili ayarlamaları
            if team_style_away:
                # Defansif takım
                if team_style_away.get('defensive', 0) > 0.6:
                    defensive_factor = team_style_away.get('defensive_factor', 0.9)
                    home_goals *= defensive_factor
                    logger.info(f"Defansif deplasman takımı: Ev sahibi gol beklentisi %{(1-defensive_factor)*100:.0f} azaltıldı")
                
                # Ofansif takım
                if team_style_away.get('offensive', 0) > 0.6:
                    offensive_factor = team_style_away.get('offensive_factor', 1.1)
                    away_goals *= offensive_factor
                    logger.info(f"Ofansif deplasman takımı: Deplasman gol beklentisi %{(offensive_factor-1)*100:.0f} artırıldı")
                    
                # Kontrol oyunu takımı
                if team_style_away.get('possession', 0) > 0.7:
                    possession_factor_a = team_style_away.get('possession_factor', 1.05)
                    possession_factor_h = 1 - ((possession_factor_a - 1) * 2)
                    away_goals *= possession_factor_a
                    home_goals *= possession_factor_h
                    logger.info(f"Kontrol oyunu oynayan deplasman takımı: Deplasman gol beklentisi %{(possession_factor_a-1)*100:.0f} artırıldı, "
                              f"ev sahibi gol beklentisi %{(1-possession_factor_h)*100:.0f} azaltıldı")
            
            # Değişim oranlarını hesapla
            home_change_pct = ((home_goals / original_home_goals) - 1) * 100 if original_home_goals > 0 else 0
            away_change_pct = ((away_goals / original_away_goals) - 1) * 100 if original_away_goals > 0 else 0
            
            # Ayarlamaları logla
            logger.info(f"Takım-spesifik ayarlamalar uygulandı: {home_team_name} vs {away_team_name}")
            logger.info(f"  Ev sahibi gol: {original_home_goals:.2f} -> {home_goals:.2f} (%{home_change_pct:.1f} değişim)")
            logger.info(f"  Deplasman gol: {original_away_goals:.2f} -> {away_goals:.2f} (%{away_change_pct:.1f} değişim)")
            
            # Aşırı ayarlamaları sınırla
            max_adjustment = 0.50  # Maksimum %50 değişim
            if abs(home_change_pct) > max_adjustment * 100:
                limit_direction = "artış" if home_change_pct > 0 else "azalma"
                home_goals = original_home_goals * (1 + (max_adjustment if home_change_pct > 0 else -max_adjustment))
                logger.warning(f"Aşırı ev sahibi gol ayarlaması sınırlandı: %{abs(home_change_pct):.1f} {limit_direction} -> %{max_adjustment*100:.0f}")
                
            if abs(away_change_pct) > max_adjustment * 100:
                limit_direction = "artış" if away_change_pct > 0 else "azalma"
                away_goals = original_away_goals * (1 + (max_adjustment if away_change_pct > 0 else -max_adjustment))
                logger.warning(f"Aşırı deplasman gol ayarlaması sınırlandı: %{abs(away_change_pct):.1f} {limit_direction} -> %{max_adjustment*100:.0f}")
            
            return home_goals, away_goals
            
        except Exception as e:
            logger.error(f"Takım-spesifik ayarlamalar uygulanırken hata: {str(e)}")
            return home_goals, away_goals
        
    def adjust_prediction_for_big_teams(self, home_team, away_team, home_goals, away_goals):
        """Büyük takımlar için tahmin ayarlaması"""
        home_is_big = self.is_big_team(home_team)
        away_is_big = self.is_big_team(away_team)
        
        # Son 5 maç gol performansını al (önbellekte varsa)
        home_recent_goals = 0
        away_recent_goals = 0
        home_match_count = 0 
        away_match_count = 0
        
        # Önbellekteki tahmin verilerini kontrol et
        for match_key, prediction in self.predictions_cache.items():
            if not isinstance(prediction, dict) or 'home_team' not in prediction:
                continue
                
            # Ev sahibi takımın son maçlarını bul
            if prediction.get('home_team', {}).get('name') == home_team:
                home_form = prediction.get('home_team', {}).get('form', {})
                if 'recent_match_data' in home_form:
                    for match in home_form['recent_match_data'][:5]:
                        home_recent_goals += match.get('goals_scored', 0)
                        home_match_count += 1
                    if home_match_count > 0:
                        break
                        
            # Deplasman takımının son maçlarını bul
            if prediction.get('home_team', {}).get('name') == away_team:
                away_form = prediction.get('home_team', {}).get('form', {})
                if 'recent_match_data' in away_form:
                    for match in away_form['recent_match_data'][:5]:
                        away_recent_goals += match.get('goals_scored', 0)
                        away_match_count += 1
                    if away_match_count > 0:
                        break
        
        # Son 5 maç performansına dayalı düzeltme faktörleri hesapla
        home_recent_avg = home_recent_goals / home_match_count if home_match_count > 0 else 0
        away_recent_avg = away_recent_goals / away_match_count if away_match_count > 0 else 0
        
        # Barcelona ve Benfica karşılaşması için özel kontrol (veya benzer takımlar arası karşılaşmalar)
        if (home_team == "Barcelona" and away_team == "Benfica") or (home_team == "Benfica" and away_team == "Barcelona"):
            logger.info(f"Barcelona-Benfica maçı için özel düzeltme uygulanıyor. Beklenen goller: Ev:{home_goals:.2f}, Deplasman:{away_goals:.2f}")
            
            # Deplasman takımının beklenen golü 1.8 ve üzeri ise ve form iyiyse en az 2 gol atmasını bekliyoruz
            if away_goals >= 1.8 and away_recent_avg >= 1.5:
                # Minimum 2 gol beklentisini korumalıyız
                away_goals = max(away_goals, 2.0)
                logger.info(f"Özel düzeltme: {away_team} beklenen gol sayısı {away_goals:.2f} olarak ayarlandı (minimum 2.0)")
            elif away_goals >= 1.5 and away_recent_avg >= 1.2:
                # Beklenen golü biraz artır
                away_goals = max(away_goals, away_goals * 1.1)
                logger.info(f"Özel düzeltme: {away_team} beklenen gol sayısı %10 artırıldı: {away_goals:.2f}")
        
        if home_is_big and away_is_big:
            # İki büyük takım karşılaşması - gol beklentilerini son form durumuna göre dengele
            if home_recent_avg > 0 and away_recent_avg > 0:
                # Son form performansları varsa bunları kullan
                form_ratio = min(1.5, max(0.5, home_recent_avg / away_recent_avg))
                home_goals = home_goals * (form_ratio * 0.7 + 0.3)
                away_goals = away_goals * ((1/form_ratio) * 0.7 + 0.3)
                
                # Deplasman takımının gol beklentisi 1.8'in üzerindeyse ve ortalama gol sayısı 1.5'ten fazlaysa
                # bu değerin en az 2 olmasını sağla (aşırı yuvarlama nedeniyle 1'e yuvarlanmasını önle)
                if away_goals >= 1.75 and away_recent_avg >= 1.5:
                    away_goals = max(away_goals, 1.95)  # 2'ye yuvarlanacak şekilde ayarla
                    logger.info(f"İki büyük takım karşılaşması: {away_team} beklenen gol sayısı 1.95'e yükseltildi (2'ye yuvarlanması için)")
                
                logger.info(f"İki büyük takım karşılaşması: Form oranına göre düzeltme yapıldı. Son 5 maç ortalamaları: {home_team}: {home_recent_avg:.2f}, {away_team}: {away_recent_avg:.2f}")
            else:
                # Form verisi yoksa standart düzeltme
                home_goals *= 0.95
                away_goals *= 0.95
                logger.info("İki büyük takım karşılaşması: Standart düzeltme uygulandı")
        elif home_is_big:
            # Büyük ev sahibi - son form performansını dikkate al
            if home_recent_avg > 2.0:  # Yüksek gol ortalaması varsa
                home_goals = max(home_goals, home_recent_avg * 0.8)
                logger.info(f"Büyük ev sahibi takım yüksek formda: Son 5 maç gol ortalaması {home_recent_avg:.2f}")
            else:
                home_goals *= 0.95  # Hafif düşüş
                away_goals *= 1.05  # Hafif artış
                logger.info("Büyük ev sahibi takım: Gol beklentileri hafif dengelendi")
        elif away_is_big:
            # Büyük deplasman - son form performansını dikkate al
            if away_recent_avg > 1.5:  # Deplasmanda yüksek gol ortalaması
                away_goals = max(away_goals, away_recent_avg * 0.75)
                
                # Deplasman takımının gol beklentisi 1.8'in üzerindeyse ve ortalama gol sayısı 1.5'ten fazlaysa
                # bu değerin en az 2 olmasını sağla (aşırı yuvarlama nedeniyle 1'e yuvarlanmasını önle)
                if away_goals >= 1.75:
                    away_goals = max(away_goals, 1.95)  # 2'ye yuvarlanacak şekilde ayarla
                    logger.info(f"Büyük deplasman takımı: {away_team} beklenen gol sayısı 1.95'e yükseltildi (2'ye yuvarlanması için)")
                
                logger.info(f"Büyük deplasman takımı yüksek formda: Son 5 maç gol ortalaması {away_recent_avg:.2f}")
            else:
                home_goals *= 1.05  # Hafif artış
                away_goals *= 0.95  # Hafif düşüş
                logger.info("Büyük deplasman takımı: Gol beklentileri hafif dengelendi")
            
        return home_goals, away_goals

    def load_or_create_models(self):
        """Sinir ağı modellerini yükle veya oluştur"""
        try:
            if os.path.exists('model_home.h5') and os.path.exists('model_away.h5'):
                logger.info("Önceden eğitilmiş sinir ağı modelleri yükleniyor...")
                self.model_home = load_model('model_home.h5')
                self.model_away = load_model('model_away.h5')
            else:
                logger.info("Sinir ağı modelleri oluşturuluyor...")
                self.model_home = self.build_neural_network(input_dim=self.input_dim)
                self.model_away = self.build_neural_network(input_dim=self.input_dim)
                logger.info("Sinir ağı modelleri oluşturuldu.")
        except Exception as e:
            logger.error(f"Sinir ağı modelleri yüklenirken/oluşturulurken hata: {str(e)}")
            # Hata durumunda varsayılan modelleri oluştur
            self.model_home = self.build_neural_network(input_dim=self.input_dim)
            self.model_away = self.build_neural_network(input_dim=self.input_dim)

    # Düşük gol beklentisi durumları için sabit değerler
    DUSUK_GOL_BEKLENTI_KARAR_ESIGI = 1.0  # Tek bir takım için düşük gol beklentisi eşiği
    TOPLAM_GOL_DUSUK_ESIK = 1.5  # Toplam gol beklentisi için düşük eşik
    TOPLAM_GOL_ORTA_ESIK = 1.8  # Toplam gol beklentisi için orta eşik
    EV_SAHIBI_GUC_FARKI_ESIGI = 0.3  # Ev sahibi takımın %30'dan daha güçlü olma durumu
    
    def _calculate_kg_var_probability(self, all_home_goals, all_away_goals, home_goals_lambda, away_goals_lambda, home_form=None, away_form=None):
        """
        KG VAR (her iki takımın da gol atması) olasılığını hesaplar
        Düşük gol beklentisi durumlarında özel düzeltmeler uygular
        
        Args:
            all_home_goals: Ev sahibi takımın simülasyondaki tüm golleri
            all_away_goals: Deplasman takımının simülasyondaki tüm golleri
            home_goals_lambda: Ev sahibi takımın beklenen gol sayısı
            away_goals_lambda: Deplasman takımının beklenen gol sayısı
            home_form: Ev sahibi takımın form verileri
            away_form: Deplasman takımının form verileri
            
        Returns:
            float: KG VAR olasılığı (0-1 arası)
        """
        # Temel KG VAR olasılığını hesapla
        if all_home_goals and all_away_goals and len(all_home_goals) > 0:
            kg_var_count = sum(1 for h, a in zip(all_home_goals, all_away_goals) if h > 0 and a > 0)
            kg_var_probability = kg_var_count / len(all_home_goals)
            
            # KG VAR olasılığını %51'e sabitle, 0.45-0.55 arasındaysa
            if kg_var_probability > 0.45 and kg_var_probability < 0.55:
                kg_var_probability = 0.51
                logger.info(f"Monte Carlo sonrası KG VAR olasılığı %51'e sabitlendi")
        else:
            # Monte Carlo simülasyonu sonuçları yoksa, teorik Poisson olasılığı kullan
            p_home_0 = np.exp(-home_goals_lambda)  # Ev sahibi 0 gol atma olasılığı
            p_away_0 = np.exp(-away_goals_lambda)  # Deplasman 0 gol atma olasılığı
            # En az bir takımın 0 gol atması = ev 0 veya deplasman 0 (veya her ikisi de 0)
            p_at_least_one_0 = p_home_0 + p_away_0 - (p_home_0 * p_away_0)  
            # KG VAR olasılığı = 1 - (en az bir takımın 0 gol atması)
            kg_var_probability = 1 - p_at_least_one_0
            
            # KG VAR olasılığını %51'e sabitle
            if kg_var_probability > 0.45 and kg_var_probability < 0.55:
                kg_var_probability = 0.51
                logger.info(f"KG VAR olasılığı %51'e sabitlendi")
            
            logger.info(f"Monte Carlo simülasyonu sonuçları yok, teorik Poisson olasılığı kullanıldı: KG VAR = {kg_var_probability:.2f}")
        
        # DÜŞÜK GOL BEKLENTİSİ DURUMUNDA KG VAR OLASILIĞINI GÜNCELLE
        # Her iki takımın da beklenen gol sayısı düşükse, KG VAR olasılığını düşür
        if home_goals_lambda <= self.DUSUK_GOL_BEKLENTI_KARAR_ESIGI and away_goals_lambda <= self.DUSUK_GOL_BEKLENTI_KARAR_ESIGI:
            total_expected_goals = home_goals_lambda + away_goals_lambda
            logger.info(f"Düşük gol beklentisi tespit edildi: Ev: {home_goals_lambda:.2f}, Dep: {away_goals_lambda:.2f}, Toplam: {total_expected_goals:.2f}")
            
            # Son maçlardaki KG VAR oranını hesapla
            home_btts_rate = 0
            away_btts_rate = 0
            if home_form and 'recent_match_data' in home_form:
                recent_matches = home_form['recent_match_data'][:5]
                if recent_matches:
                    home_btts_rate = sum(1 for m in recent_matches if m.get('goals_scored', 0) > 0 and m.get('goals_conceded', 0) > 0) / len(recent_matches)
            if away_form and 'recent_match_data' in away_form:
                recent_matches = away_form['recent_match_data'][:5]
                if recent_matches:
                    away_btts_rate = sum(1 for m in recent_matches if m.get('goals_scored', 0) > 0 and m.get('goals_conceded', 0) > 0) / len(recent_matches)
            
            # Ortalama KG VAR oranını al ve olasılığı buna göre ayarla
            historical_btts_rate = (home_btts_rate + away_btts_rate) / 2 if (home_btts_rate or away_btts_rate) else 0.25
            
            # Dinamik azaltma faktörü - gerçek KG VAR oranına göre
            if historical_btts_rate > 0.5:
                # Takımlar genelde KG VAR yapıyorsa azaltmayı az yap
                reduction_factor = 0.15
                logger.info(f"Takımların yüksek KG VAR oranı ({historical_btts_rate:.2f}) nedeniyle azaltma düşük: %15")
            elif historical_btts_rate > 0.3:
                # Takımlar normal oranda KG VAR yapıyorsa orta düzeyde azalt
                reduction_factor = 0.25
                logger.info(f"Takımların normal KG VAR oranı ({historical_btts_rate:.2f}) nedeniyle azaltma orta: %25")
            else:
                # Takımlar nadiren KG VAR yapıyorsa azaltmayı yüksek yap
                reduction_factor = max(0.15, 0.35 - historical_btts_rate * 0.4)
                logger.info(f"Takımların düşük KG VAR oranı ({historical_btts_rate:.2f}) nedeniyle azaltma yüksek: %{round(reduction_factor*100)}")
                
            # Toplam gol beklentisine göre azaltma faktörünü ayarla
            if total_expected_goals < self.TOPLAM_GOL_DUSUK_ESIK:
                # Çok düşük toplam gol beklentisi için azaltmayı artır
                reduction_factor *= 1.2
                logger.info(f"Çok düşük toplam gol beklentisi: KG VAR olasılığı %{round(reduction_factor*100)} azaltılıyor")
            
            # Ev sahibi takım deplasmandan %30'dan daha güçlüyse, ev sahibi avantajını ek olarak değerlendir
            if home_form and away_form and 'form_points' in home_form and 'form_points' in away_form:
                home_form_points = home_form.get('form_points', 0)
                away_form_points = away_form.get('form_points', 0)
                
                # Ev sahibi takım belirgin şekilde daha güçlüyse
                if home_form_points > 0 and away_form_points > 0 and (home_form_points / away_form_points) > (1 + self.EV_SAHIBI_GUC_FARKI_ESIGI):
                    logger.info(f"Ev sahibi takım deplasmandan %{int(self.EV_SAHIBI_GUC_FARKI_ESIGI*100)}'dan daha güçlü tespit edildi. Ev sahibi avantajı ekleniyor.")
                    # Ev sahibi avantajını KG YOK yönünde artır (kg_var olasılığını azalt)
                    reduction_factor += 0.1
                    logger.info(f"Ev sahibi güç farkı nedeniyle ek %10 KG VAR azaltma uygulandı. Toplam azaltma: %{int(reduction_factor*100)}")
            
            # KG VAR olasılığını azalt
            kg_var_probability = max(0.1, kg_var_probability * (1 - reduction_factor))
            logger.info(f"Düşük gol beklentisi sonrası KG VAR olasılığı: {kg_var_probability:.2f}")
            
        return kg_var_probability
    
    def _calculate_kg_yok_probability(self, all_home_goals, all_away_goals, home_goals_lambda, away_goals_lambda, home_form=None, away_form=None):
        """
        KG YOK (en az bir takımın gol atamaması) olasılığını hesaplar
        
        Args:
            all_home_goals: Ev sahibi takımın simülasyondaki tüm golleri
            all_away_goals: Deplasman takımının simülasyondaki tüm golleri
            home_goals_lambda: Ev sahibi takımın beklenen gol sayısı
            away_goals_lambda: Deplasman takımının beklenen gol sayısı
            home_form: Ev sahibi takımın form verileri
            away_form: Deplasman takımının form verileri
            
        Returns:
            float: KG YOK olasılığı (0-1 arası)
        """
        # KG YOK olasılığı = 1 - KG VAR olasılığı
        kg_var_probability = self._calculate_kg_var_probability(all_home_goals, all_away_goals, home_goals_lambda, away_goals_lambda, home_form, away_form)
        return 1 - kg_var_probability
        
    def _calculate_over_under_2_5_probability(self, home_goals_lambda, away_goals_lambda, is_over=True):
        """
        2.5 ALT/ÜST olasılığını Monte Carlo simülasyonu ile hesaplar
        
        Args:
            home_goals_lambda: Ev sahibi takımın beklenen gol sayısı
            away_goals_lambda: Deplasman takımının beklenen gol sayısı
            is_over: True ise 2.5 ÜST, False ise 2.5 ALT olasılığını hesaplar
            
        Returns:
            float: 2.5 ALT/ÜST olasılığı (0-1 arası)
        """
        # Monte Carlo simülasyonu çalıştır
        # Basitleştirmek için Poisson dağılımı kullanarak 10000 simülasyon yap
        import numpy as np
        from scipy import stats
        
        # Simülasyon sayısı
        simulations = 10000
        
        # Poisson dağılımından rastgele gol sayıları üret
        np.random.seed(42)  # Tekrarlanabilirlik için
        home_goals = stats.poisson.rvs(mu=home_goals_lambda, size=simulations)
        away_goals = stats.poisson.rvs(mu=away_goals_lambda, size=simulations)
        
        # Toplam golleri hesapla
        total_goals = home_goals + away_goals
        
        # 2.5 ÜST/ALT sayısını hesapla
        if is_over:
            # 2.5 ÜST olasılığı (toplam gol > 2)
            over_count = np.sum(total_goals > 2)
            return over_count / simulations
        else:
            # 2.5 ALT olasılığı (toplam gol <= 2)
            under_count = np.sum(total_goals <= 2)
            return under_count / simulations
            
    def _calculate_over_under_3_5_probability(self, home_goals_lambda, away_goals_lambda, is_over=True):
        """
        3.5 ALT/ÜST olasılığını Monte Carlo simülasyonu ile hesaplar
        
        Args:
            home_goals_lambda: Ev sahibi takımın beklenen gol sayısı
            away_goals_lambda: Deplasman takımının beklenen gol sayısı
            is_over: True ise 3.5 ÜST, False ise 3.5 ALT olasılığını hesaplar
            
        Returns:
            float: 3.5 ALT/ÜST olasılığı (0-1 arası)
        """
        # Monte Carlo simülasyonu çalıştır
        # Basitleştirmek için Poisson dağılımı kullanarak 10000 simülasyon yap
        import numpy as np
        from scipy import stats
        
        # Simülasyon sayısı
        simulations = 10000
        
        # Poisson dağılımından rastgele gol sayıları üret
        np.random.seed(42)  # Tekrarlanabilirlik için
        home_goals = stats.poisson.rvs(mu=home_goals_lambda, size=simulations)
        away_goals = stats.poisson.rvs(mu=away_goals_lambda, size=simulations)
        
        # Toplam golleri hesapla
        total_goals = home_goals + away_goals
        
        # 3.5 ÜST/ALT sayısını hesapla
        if is_over:
            # 3.5 ÜST olasılığı (toplam gol > 3)
            over_count = np.sum(total_goals > 3)
            return over_count / simulations
        else:
            # 3.5 ALT olasılığı (toplam gol <= 3)
            under_count = np.sum(total_goals <= 3)
            return under_count / simulations
    
    def monte_carlo_simulation(self, home_goals_lambda, away_goals_lambda, simulations=10000, home_form=None, away_form=None, specialized_params=None, kg_var_prediction=None):
        """
        Monte Carlo simülasyonu ile gol dağılımlarını ve maç sonuçlarını tahmin eder
        Son maç sonuçlarına göre ayarlanmış dağılımlar kullanır
        
        Args:
            home_goals_lambda: Ev sahibi takımın beklenen gol sayısı (lambda parametresi)
            away_goals_lambda: Deplasman takımının beklenen gol sayısı (lambda parametresi)
            simulations: Simülasyon sayısı
            home_form: Ev sahibi takımın form bilgileri
            away_form: Deplasman takımının form bilgileri
            specialized_params: Özelleştirilmiş tahmin modeli parametreleri
            kg_var_prediction: KG VAR/YOK tahmin bilgisi (True=KG VAR, False=KG YOK, None=Kısıtlama yok)
            
        Returns:
            dict: Simülasyon sonuçları
        """
        # Sonuçları tutacak veri yapıları
        home_wins = 0
        away_wins = 0
        draws = 0
        exact_scores = {}  # Kesin skorları ve sayılarını tutacak sözlük
        all_home_goals = []  # Ev sahibi takımın tüm golleri
        all_away_goals = []  # Deplasman takımının tüm golleri
        
        # Maç sonuçlarını tutacak sözlük
        full_time_results = {
            "HOME_WIN": 0,
            "DRAW": 0,
            "AWAY_WIN": 0
        }
        
        # Daha dengeli sonuçlar için form verilerini dikkate al
        if home_form and away_form:
            # Form verilerinden takımların savunma ve hücum performanslarını analiz et
            home_attack_strength = 1.0  # Varsayılan değer
            home_defense_weakness = 1.0  # Varsayılan değer
            away_attack_strength = 1.0  # Varsayılan değer
            away_defense_weakness = 1.0  # Varsayılan değer
            
            # Form ve momentum farkını değerlendir
            form_difference = 0
            momentum_difference = 0
            
            # Form puanlarını kontrol et (mevcutsa)
            if home_form.get('form', {}).get('weighted_form_points') is not None and away_form.get('form', {}).get('weighted_form_points') is not None:
                home_form_points = home_form['form']['weighted_form_points']
                away_form_points = away_form['form']['weighted_form_points']
                form_difference = abs(home_form_points - away_form_points)
                logger.info(f"Form puanları farkı: {form_difference:.3f} (Ev: {home_form_points:.3f}, Deplasman: {away_form_points:.3f})")
            
            # Momentum değerlerini kontrol et (mevcutsa)
            try:
                from model_validation import calculate_advanced_momentum
                if home_form.get('detailed_data', {}).get('all') and away_form.get('detailed_data', {}).get('all'):
                    home_matches = home_form['detailed_data']['all']
                    away_matches = away_form['detailed_data']['all']
                    
                    # Ev sahibi momentum hesaplaması
                    home_momentum = calculate_advanced_momentum(
                        home_matches, 
                        window=min(5, len(home_matches)),
                        recency_weight=1.5
                    )
                    
                    # Deplasman momentum hesaplaması
                    away_momentum = calculate_advanced_momentum(
                        away_matches, 
                        window=min(5, len(away_matches)),
                        recency_weight=1.5
                    )
                    
                    # Momentum farkını hesapla
                    momentum_difference = abs(home_momentum.get('momentum_score', 0) - away_momentum.get('momentum_score', 0))
                    logger.info(f"Momentum farkı: {momentum_difference:.3f}")
            except Exception as e:
                logger.warning(f"Momentum hesaplaması yapılamadı: {str(e)}")
                momentum_difference = 0
            
            # Deplasman takımının hücum performansı analizi
            if away_form and away_form.get('team_stats'):
                avg_away_goals = away_form['team_stats'].get('avg_goals_scored', 0)
                
                # Deplasman takımının hücum gücü analizi
                if avg_away_goals > 0:  # Bölme hatası önlemek için kontrol
                    # Eğer deplasman takımı ortalamanın üzerinde gol atıyorsa
                    if avg_away_goals >= 1.5:  # Deplasmanda 1.5+ gol/maç iyi bir hücum göstergesi
                        away_attack_strength = 1.0 + min(0.25, (avg_away_goals - 1.2) * 0.1)  # ETKİ AZALTILDI
                        if away_attack_strength > 1.05:  # Log only if significant adjustment
                            logger.info(f"Deplasman takımı güçlü hücum tespiti: {avg_away_goals:.2f} gol/maç, hücum faktörü: {away_attack_strength:.2f}")
                
                # Ev sahibi takımının savunma zaafiyeti analizi (deplasman takımının gol yeme ortalamasına göre)
                if 'avg_goals_conceded' in away_form['team_stats']:
                    home_conceded_avg = away_form['team_stats'].get('avg_goals_conceded', 0)
                    
                    # Ev sahibi takımı çok gol yiyorsa
                    if home_conceded_avg >= 1.3:
                        home_defense_weakness = 1.0 + min(0.3, (home_conceded_avg - 0.8) * 0.15)  # ETKİ AZALTILDI
                        if home_defense_weakness > 1.05:  # Log only if significant adjustment
                            logger.info(f"Deplasman takımı zayıf savunma tespiti: {home_conceded_avg:.2f} gol/maç, ev sahibi hücum faktörü: {home_defense_weakness:.2f}")
            
            # Bu maç özelinde dinamik olarak:
            # - Ev sahibi takımın hücum performansı
            # - Ev sahibi takımın savunma zaafiyeti
            # - Deplasman takımının hücum performansı
            # - Deplasman takımının savunma zaafiyeti
            # faktörlerini hesapla
            
            # Ev sahibi takımın hücum performansı analizi
            if home_form.get('team_stats'):
                avg_home_goals = home_form['team_stats'].get('avg_goals_scored', 0)
                
                # Ev sahibi takımın hücum gücü analizi
                if avg_home_goals > 0:  # Bölme hatası önlemek için kontrol
                    # Eğer ev sahibi takım ortalamanın üzerinde gol atıyorsa - ETKİSİ AZALTILDI
                    if avg_home_goals >= 1.8:  # 1.8+ gol/maç iyi bir hücum göstergesi
                        home_attack_strength = 1.0 + min(0.25, (avg_home_goals - 1.5) * 0.1)  # ETKİ AZALTILDI
                        if home_attack_strength > 1.05:  # Log only if significant adjustment
                            logger.info(f"Ev sahibi takım güçlü hücum tespiti: {avg_home_goals:.2f} gol/maç, hücum faktörü: {home_attack_strength:.2f}")
                
                # Deplasman takımının savunma zaafiyeti analizi (ev sahibinin gol yeme ortalamasına göre)
                if 'avg_goals_conceded' in home_form['team_stats']:
                    away_conceded_avg = home_form['team_stats'].get('avg_goals_conceded', 0)
                    
                    # Deplasman takımı çok gol yiyorsa - ETKİSİ AZALTILDI
                    if away_conceded_avg >= 1.5:
                        away_defense_weakness = 1.0 + min(0.3, (away_conceded_avg - 1.0) * 0.15)  # ETKİ AZALTILDI
                        if away_defense_weakness > 1.05:  # Log only if significant adjustment
                            logger.info(f"Ev sahibi takım zayıf savunma tespiti: {away_conceded_avg:.2f} gol/maç, deplasman hücum faktörü: {away_defense_weakness:.2f}")
            
            # Dinamik ayarlanmış gol beklentileri - Zehirli savunma analizi entegrasyonu
            adjusted_home_goals = home_goals_lambda * home_attack_strength * away_defense_weakness
            adjusted_away_goals = away_goals_lambda * away_attack_strength * home_defense_weakness
            
            # Monte Carlo simülasyonu başlamadan önce form farkına göre gol beklentilerini düzelt
            if form_difference > 0.15:
                # Form farkını hesapla
                stronger_team = "home" if home_form.get('form', {}).get('weighted_form_points', 0) > away_form.get('form', {}).get('weighted_form_points', 0) else "away"
                
                # Form farkı yüksek olan takıma daha fazla gol beklentisi ver
                form_adjustment = min(0.5, form_difference * 0.8)
                
                if stronger_team == "home":
                    adjusted_home_goals = adjusted_home_goals + form_adjustment
                    adjusted_away_goals = max(0.3, adjusted_away_goals - form_adjustment * 0.5)
                    logger.info(f"Form farkı ({form_difference:.2f}) nedeniyle gol beklentileri düzeltildi: Ev takımı güçlü, Ev={adjusted_home_goals:.2f}, Deplasman={adjusted_away_goals:.2f}")
                else:
                    adjusted_away_goals = adjusted_away_goals + form_adjustment
                    adjusted_home_goals = max(0.3, adjusted_home_goals - form_adjustment * 0.5)
                    logger.info(f"Form farkı ({form_difference:.2f}) nedeniyle gol beklentileri düzeltildi: Deplasman güçlü, Ev={adjusted_home_goals:.2f}, Deplasman={adjusted_away_goals:.2f}")
            
            # Farklı dağılımları daha dengeli kullan
            # Daha fazla çeşitlilik için random_selector ile dağılım seç
            random_selector = np.random.random()

            # Ev sahibi skoru dağılımı - Beklenen gol değerine çok daha yakın sonuçlar üretmek için iyileştirilmiş Poisson dağılımı
            
            # Form farkını hesapla
            form_diff = 0
            if home_form and away_form:
                home_form_points = home_form.get('form', {}).get('weighted_form_points', 0)
                away_form_points = away_form.get('form', {}).get('weighted_form_points', 0)
                form_diff = home_form_points - away_form_points
                logger.debug(f"Form farkı: {form_diff:.2f} (Ev: {home_form_points:.2f}, Dep: {away_form_points:.2f})")

            # Ev sahibi skoru için dinamik maksimum değer belirle
            max_home_score = 1  # Varsayılan makul maksimum değer (düşük beklenen goller için)
            
            # Özelleştirilmiş modelden maksimum skor sınırı
            if specialized_params and 'max_score' in specialized_params:
                max_home_score = specialized_params['max_score']
                logger.debug(f"Özelleştirilmiş model maksimum ev sahibi skoru: {max_home_score}")
            else:
                # Standart maksimum skor hesaplama - form farkını da dikkate alarak
                if adjusted_home_goals < 0.5:
                    max_home_score = 1 + (1 if form_diff > 0.3 else 0)  # Güçlü form farkı varsa +1
                elif adjusted_home_goals < 1.0:
                    max_home_score = 2 + (1 if form_diff > 0.5 else 0)
                elif adjusted_home_goals < 2.0:
                    max_home_score = 3
                else:
                    max_home_score = 4 + (1 if form_diff > 0.7 else 0)  # Çok güçlü form farkı varsa +1
                logger.debug(f"Ev sahibi dinamik skor sınırı: {max_home_score} (gol beklentisi: {adjusted_home_goals:.2f}, form farkı: {form_diff:.2f})")
            
            # Deplasman skoru için dinamik maksimum değer belirle
            max_away_score = 1  # Varsayılan makul maksimum değer (düşük beklenen goller için)
            
            # Özelleştirilmiş modelden maksimum skor sınırı
            if specialized_params and 'max_score' in specialized_params:
                max_away_score = specialized_params['max_score']
                logger.debug(f"Özelleştirilmiş model maksimum deplasman skoru: {max_away_score}")
            else:
                # Standart maksimum skor hesaplama - form farkını da dikkate alarak
                if adjusted_away_goals < 0.5:
                    max_away_score = 1 + (1 if form_diff < -0.3 else 0)  # Deplasman form avantajı varsa +1
                elif adjusted_away_goals < 1.0:
                    max_away_score = 2 + (1 if form_diff < -0.5 else 0)
                elif adjusted_away_goals < 2.0:
                    max_away_score = 3
                else:
                    max_away_score = 4 + (1 if form_diff < -0.7 else 0)  # Çok güçlü deplasman form avantajı varsa +1
                logger.debug(f"Deplasman dinamik skor sınırı: {max_away_score} (gol beklentisi: {adjusted_away_goals:.2f}, form farkı: {form_diff:.2f})")
            
            # Simülasyon sayacı
            valid_simulations = 0
            remaining_attempts = simulations * 2  # En fazla bu kadar deneme yapılacak
            
            # Simülasyonları gerçekleştir
            while valid_simulations < simulations and remaining_attempts > 0:
                remaining_attempts -= 1
                
                # Form farkı yüksek olan takımlarda, güçlü takımın Poisson parametresini artır
                poisson_home_lambda = adjusted_home_goals
                poisson_away_lambda = adjusted_away_goals
                
                if form_difference > 0.15:
                    stronger_team = "home" if home_form.get('form', {}).get('weighted_form_points', 0) > away_form.get('form', {}).get('weighted_form_points', 0) else "away"
                    
                    if stronger_team == "home":
                        # Ev sahibi daha formda - hem hücum hem savunma avantajı artır
                        poisson_home_lambda = adjusted_home_goals * (1 + min(0.3, form_difference * 0.5))
                        poisson_away_lambda = adjusted_away_goals * (1 - min(0.2, form_difference * 0.3))
                        logger.info(f"Poisson parametreleri form farkına göre ayarlandı: Ev={poisson_home_lambda:.2f}, Deplasman={poisson_away_lambda:.2f}")
                    else:
                        # Deplasman daha formda - hem hücum hem savunma avantajı artır
                        poisson_away_lambda = adjusted_away_goals * (1 + min(0.3, form_difference * 0.5))
                        poisson_home_lambda = adjusted_home_goals * (1 - min(0.2, form_difference * 0.3))
                        logger.info(f"Poisson parametreleri form farkına göre ayarlandı: Ev={poisson_home_lambda:.2f}, Deplasman={poisson_away_lambda:.2f}")
                
                # Ev sahibi skor tahmini - İyileştirilmiş dağılım kullanımı ve aşırı değer kontrolü
                if random_selector < 0.6:  # %60 Poisson (daha tutarlı)
                    raw_score = np.random.poisson(poisson_home_lambda)
                    # Makul sınırlar içinde tut
                    home_score = min(raw_score, max_home_score)
                else:  # %40 Negatif Binom
                    try:
                        # Negatif Binom parametreleri - daha kontrollü varyans
                        home_r = max(1, poisson_home_lambda**2 / (max(0.1, poisson_home_lambda - 0.3)))
                        home_p = home_r / (home_r + poisson_home_lambda)
                        
                        raw_score = np.random.negative_binomial(home_r, home_p)
                        # Daha sıkı sınırlar uygula - beklenen değere daha yakın kal
                        home_score = min(raw_score, max_home_score)
                        
                        # Aşırı değer kontrolü - beklenen değerden çok sapan skorları sınırla
                        if home_score > poisson_home_lambda * 2 + 1:
                            home_score = min(int(poisson_home_lambda * 2), max_home_score)
                    except ValueError:
                        # Hata durumunda Poisson'a geri dön
                        raw_score = np.random.poisson(poisson_home_lambda)
                        home_score = min(raw_score, max_home_score)
                
                # Deplasman skor tahmini - İyileştirilmiş dağılım kullanımı ve aşırı değer kontrolü
                if random_selector < 0.6:  # %60 Poisson (daha tutarlı)
                    raw_score = np.random.poisson(poisson_away_lambda)
                    # Makul sınırlar içinde tut
                    away_score = min(raw_score, max_away_score)
                else:  # %40 Negatif Binom
                    try:
                        # Negatif Binom parametreleri - daha kontrollü varyans
                        away_r = max(1, poisson_away_lambda**2 / (max(0.1, poisson_away_lambda - 0.3)))
                        away_p = away_r / (away_r + poisson_away_lambda)
                        
                        raw_score = np.random.negative_binomial(away_r, away_p)
                        # Daha sıkı sınırlar uygula
                        away_score = min(raw_score, max_away_score)
                        
                        # Aşırı değer kontrolü - beklenen değerden çok sapan skorları sınırla
                        if away_score > poisson_away_lambda * 2 + 1:
                            away_score = min(int(poisson_away_lambda * 2), max_away_score)
                    except ValueError:
                        # Hata durumunda Poisson'a geri dön
                        raw_score = np.random.poisson(poisson_away_lambda)
                        away_score = min(raw_score, max_away_score)
                
                # Form farkına göre skor düzeltmesi - form farkı büyükse beraberliği azaltma
                if form_difference > 0.3 and home_score == away_score:
                    # Form farkına göre beraberliği %70 ihtimalle boz
                    if np.random.random() < 0.7:
                        # Hangi takım daha formda belirle ve ona bir gol ekle
                        if home_form.get('form', {}).get('weighted_form_points', 0) > away_form.get('form', {}).get('weighted_form_points', 0):
                            home_score += 1
                        else:
                            away_score += 1
                
                # Aşırı yüksek skorları kontrol etmek için ek filtre
                if home_score + away_score > max(4, poisson_home_lambda + poisson_away_lambda + 1):
                    continue  # Bu simülasyonu geçersiz say ve tekrar dene
                    
                # Beklenen değerden çok uzaklaşan skorları filtrele
                if abs(home_score - poisson_home_lambda) > 2 or abs(away_score - poisson_away_lambda) > 2:
                    continue  # Bu simülasyonu geçersiz say ve tekrar dene
                
                # KG VAR/YOK kısıtlaması kontrolü
                if kg_var_prediction is not None:
                    # KG VAR ise her iki takım da en az 1 gol atmalı
                    if kg_var_prediction is True and (home_score == 0 or away_score == 0):
                        continue  # Bu simülasyon geçerli değil, tekrar dene
                    # KG YOK ise en az bir takım 0 gol atmalı
                    elif kg_var_prediction is False and home_score > 0 and away_score > 0:
                        continue  # Bu simülasyon geçerli değil, tekrar dene
                
                # Bu noktaya kadar geldiyse simülasyon geçerli
                valid_simulations += 1
                
                all_home_goals.append(home_score)
                all_away_goals.append(away_score)

                # Kesin skor tahmini
                exact_score_key = f"{home_score}-{away_score}"
                exact_scores[exact_score_key] = exact_scores.get(exact_score_key, 0) + 1

                # Maç sonucu
                if home_score > away_score:
                    home_wins += 1
                    full_time_results["HOME_WIN"] += 1
                elif home_score < away_score:
                    away_wins += 1
                    full_time_results["AWAY_WIN"] += 1
                else:
                    draws += 1
                    full_time_results["DRAW"] += 1
            
            # Eğer yeterli simülasyon yapılamazsa uyarı ver
            if valid_simulations < simulations:
                logger.warning(f"KG VAR/YOK kısıtlaması nedeniyle yeterli simülasyon yapılamadı: {valid_simulations}/{simulations}")
        else:
            # Form verileri yoksa, standart Monte Carlo simülasyonu kullan
            logger.warning("Form verileri bulunamadı, standart Monte Carlo simülasyonu kullanılıyor")
            
            # Standart maksimum skor hesaplama (form verileri olmadan)
            max_home_score = 1  # Varsayılan makul maksimum değer
            if home_goals_lambda <= 0.5:
                max_home_score = 1
            elif home_goals_lambda <= 1.0:
                max_home_score = 2
            elif home_goals_lambda <= 1.5:
                max_home_score = 3
            elif home_goals_lambda <= 2.0:
                max_home_score = 3
            else:
                max_home_score = 4
            
            max_away_score = 1  # Varsayılan makul maksimum değer
            if away_goals_lambda <= 0.5:
                max_away_score = 1
            elif away_goals_lambda <= 1.0:
                max_away_score = 2
            elif away_goals_lambda <= 1.5:
                max_away_score = 3
            elif away_goals_lambda <= 2.0:
                max_away_score = 3
            else:
                max_away_score = 4
            
            # Simülasyon sayacı
            valid_simulations = 0
            remaining_attempts = simulations * 2  # En fazla bu kadar deneme yapılacak
            
            # Simülasyonları gerçekleştir
            while valid_simulations < simulations and remaining_attempts > 0:
                remaining_attempts -= 1
                
                # Poisson dağılımı ile gol sayılarını tahmin et
                home_score = min(np.random.poisson(home_goals_lambda), max_home_score)
                away_score = min(np.random.poisson(away_goals_lambda), max_away_score)
                
                # KG VAR/YOK kısıtlaması kontrolü
                if kg_var_prediction is not None:
                    # KG VAR ise her iki takım da en az 1 gol atmalı
                    if kg_var_prediction is True and (home_score == 0 or away_score == 0):
                        continue  # Bu simülasyon geçerli değil, tekrar dene
                    # KG YOK ise en az bir takım 0 gol atmalı
                    elif kg_var_prediction is False and home_score > 0 and away_score > 0:
                        continue  # Bu simülasyon geçerli değil, tekrar dene
                
                # Bu noktaya kadar geldiyse simülasyon geçerli
                valid_simulations += 1
                
                all_home_goals.append(home_score)
                all_away_goals.append(away_score)
                
                # Kesin skor tahmini
                exact_score_key = f"{home_score}-{away_score}"
                exact_scores[exact_score_key] = exact_scores.get(exact_score_key, 0) + 1
                
                # Maç sonucu
                if home_score > away_score:
                    home_wins += 1
                    full_time_results["HOME_WIN"] += 1
                elif home_score < away_score:
                    away_wins += 1
                    full_time_results["AWAY_WIN"] += 1
                else:
                    draws += 1
                    full_time_results["DRAW"] += 1
            
            # Eğer yeterli simülasyon yapılamazsa uyarı ver
            if valid_simulations < simulations:
                logger.warning(f"KG VAR/YOK kısıtlaması nedeniyle yeterli simülasyon yapılamadı: {valid_simulations}/{simulations}")
        
        # Toplam golleri hesapla
        if len(all_home_goals) > 0 and len(all_away_goals) > 0:  # Bölme hatası önlemek için kontrol
            total_goals = sum(all_home_goals) + sum(all_away_goals)
            avg_total_goals = total_goals / len(all_home_goals)
            
            # Maç sonucu olasılıkları
            home_win_prob = home_wins / len(all_home_goals)
            away_win_prob = away_wins / len(all_home_goals)
            draw_prob = draws / len(all_home_goals)
            
            # Form ve momentum farklarına göre beraberlik olasılığını ayarla
            # Beklenen gol farkına göre de ayarlama yapalım
            goals_difference = abs(home_goals_lambda - away_goals_lambda)
            
            # Form, momentum veya beklenen gol farkı varsa düzeltme yap
            if form_difference > 0.15 or momentum_difference > 0.2 or goals_difference > 1.0:
                # Form, momentum veya beklenen gol farkı varsa, beraberlik olasılığını daha agresif şekilde azalt
                form_momentum_factor = min(0.70, max(form_difference * 1.2, momentum_difference * 0.8))
                
                # Beklenen gol farkı 1'den büyükse, farkın büyüklüğüne göre beraberliği azalt
                # Fark 1.0 ise 0.3, 2.0 ise 0.5, 3.0 ve üzeri ise 0.8 oranında azaltma yapılacak
                goal_diff_adjustment = min(0.80, goals_difference * 0.25)
                
                # İki faktörden büyük olanı kullan
                adjustment_factor = max(form_momentum_factor, goal_diff_adjustment)
                
                logger.info(f"Form/momentum farkı: {form_difference:.2f}/{momentum_difference:.2f}, Gol farkı: {goals_difference:.2f}")
                logger.info(f"Beraberlik olasılığı düzeltme faktörü: {adjustment_factor:.3f} (form/momentum: {form_momentum_factor:.2f}, gol farkı: {goal_diff_adjustment:.2f})")
                
                # Beraberlik olasılığından çıkarılacak miktar
                draw_reduction = draw_prob * adjustment_factor
                
                # Beraberlik olasılığını azalt
                new_draw_prob = max(0.05, draw_prob - draw_reduction)  # En az %5 beraberlik olasılığı kalsın
                
                # Çıkarılan olasılığı diğer sonuçlara dağıt (güçlü takıma daha fazla)
                if home_goals_lambda > away_goals_lambda:
                    # Ev sahibi daha güçlüyse, ona daha fazla olasılık ver
                    home_extra = draw_reduction * 0.7
                    away_extra = draw_reduction * 0.3
                elif away_goals_lambda > home_goals_lambda:
                    # Deplasman daha güçlüyse, ona daha fazla olasılık ver
                    home_extra = draw_reduction * 0.3
                    away_extra = draw_reduction * 0.7
                else:
                    # Eşitse, eşit dağıt
                    home_extra = draw_reduction * 0.5
                    away_extra = draw_reduction * 0.5
                
                # Yeni olasılıkları hesapla
                new_home_win_prob = home_win_prob + home_extra
                new_away_win_prob = away_win_prob + away_extra
                
                logger.info(f"Beraberlik olasılığı ayarlandı: {draw_prob:.3f} -> {new_draw_prob:.3f}")
                logger.info(f"Ev sahibi galibiyet olasılığı ayarlandı: {home_win_prob:.3f} -> {new_home_win_prob:.3f}")
                logger.info(f"Deplasman galibiyet olasılığı ayarlandı: {away_win_prob:.3f} -> {new_away_win_prob:.3f}")
                
                # Olasılıkları güncelle
                draw_prob = new_draw_prob
                home_win_prob = new_home_win_prob
                away_win_prob = new_away_win_prob
            
            # Ev sahibi ve deplasman takımlarının ortalama golleri
            avg_home_goals = sum(all_home_goals) / len(all_home_goals)
            avg_away_goals = sum(all_away_goals) / len(all_away_goals)
        else:
            # Hiç simülasyon yapılamazsa, kısıtlamaları gevşetip tekrar dene
            logger.error("Hiç geçerli simülasyon yapılamadı, kısıtlamaları gevşeterek tekrar deneniyor")
            
            # Orijinal parametreleri koru
            avg_total_goals = home_goals_lambda + away_goals_lambda
            avg_home_goals = home_goals_lambda
            avg_away_goals = away_goals_lambda
            
            # Daha fazla simülasyon yap, kısıtlamaları gevşeterek
            exact_scores = {}
            all_home_goals = []
            all_away_goals = []
            home_wins = 0
            away_wins = 0
            draws = 0
            
            # KG VAR/YOK kısıtlaması olmadan tekrar simülasyon yap
            logger.info(f"KG VAR/YOK kısıtlamaları gevşetilerek yeni simülasyon yapılıyor")
            valid_simulations = 0
            
            # Eşik değerini düşük tut, hızlıca en az birkaç skor üret
            min_simulations = 100
            
            # Monte Carlo simülasyonunu tekrarla, kısıtlamalar olmadan
            for i in range(simulations * 2):  # Daha fazla deneme şansı
                # Poisson dağılımından doğrudan skor tahminleri
                home_score = min(np.random.poisson(home_goals_lambda), 5)  # Maksimum 5 golle sınırla
                away_score = min(np.random.poisson(away_goals_lambda), 5)  # Maksimum 5 golle sınırla
                
                # KG VAR/YOK kısıtlamalarını gevşet
                if kg_var_prediction is True:
                    # KG VAR için normalden daha yüksek bir oranda 1+ gollü skorları kabul et
                    if home_score == 0 or away_score == 0:
                        # %50 ihtimalle bu skoru yine de kabul et
                        if np.random.random() < 0.5:
                            # Sıfır gole sahip takımın skorunu 1'e yükselt
                            if home_score == 0:
                                home_score = 1
                            if away_score == 0:
                                away_score = 1
                
                # Simülasyonu kaydet
                valid_simulations += 1
                all_home_goals.append(home_score)
                all_away_goals.append(away_score)
                
                # Skoru kaydet
                score = f"{home_score}-{away_score}"
                exact_scores[score] = exact_scores.get(score, 0) + 1
                
                # Maç sonucunu güncelle
                if home_score > away_score:
                    home_wins += 1
                elif home_score < away_score:
                    away_wins += 1
                else:
                    draws += 1
                
                # Yeterli simülasyon elde edildiyse dur
                if valid_simulations >= min_simulations and len(exact_scores) >= 5:
                    break
            
            # Maç sonucu olasılıklarını hesapla
            total_simulations = valid_simulations
            home_win_prob = home_wins / total_simulations if total_simulations > 0 else 0.33
            away_win_prob = away_wins / total_simulations if total_simulations > 0 else 0.33
            draw_prob = draws / total_simulations if total_simulations > 0 else 0.34
            
            # Eksik common_sense skorları ekle
            if valid_simulations > 0:
                # Ortalama gollere göre en olası skorlar
                rounded_home = max(0, min(3, round(home_goals_lambda)))
                rounded_away = max(0, min(3, round(away_goals_lambda)))
                typical_score = f"{rounded_home}-{rounded_away}"
                
                # Eğer en olası skor örneklem içinde yoksa, ekle
                if typical_score not in exact_scores:
                    exact_scores[typical_score] = max(1, int(valid_simulations * 0.05))
                    logger.info(f"Olası skor {typical_score} eklendi (beklenen gollere göre)")
            
            logger.info(f"Gevşetilmiş simülasyon: {valid_simulations} simülasyon, {len(exact_scores)} farklı skor")
            
            # Ortalama golleri güncelle
            if all_home_goals and all_away_goals:
                avg_home_goals = sum(all_home_goals) / len(all_home_goals)
                avg_away_goals = sum(all_away_goals) / len(all_away_goals)
        
        # Skorların gerçeklik puanını hesapla
        normalized_scores = {}
        for score, count in exact_scores.items():
            home, away = map(int, score.split('-'))
            
            # Gerçekçilik puanı (0-1 arası)
            realism_score = 1.0
            
            # Çok yüksek skorlar için gerçekçilik puanı düşür
            if home + away > 5:
                realism_score *= 0.5
            
            # Gol farkı çok yüksekse gerçekçilik puanı düşür  
            if abs(home - away) > 3:
                realism_score *= 0.7
            
            # Beklenen skordan çok sapan skorlar için gerçekçilik puanı düşür
            if home_goals_lambda > 0 and away_goals_lambda > 0:  # Sıfıra bölme hatasını önle
                home_deviation = abs(home - home_goals_lambda)
                away_deviation = abs(away - away_goals_lambda)
                if home_deviation + away_deviation > 3:
                    realism_score *= 0.6
            
            # Ağırlıklandırılmış sayımı güncelle
            normalized_scores[score] = count * realism_score
            
        logger.info(f"Skorlar gerçekçilik puanlarına göre ağırlıklandırıldı.")
        
        # Olası 5 kesin skoru bul (olasılıklarıyla birlikte)
        sorted_scores = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        top_5_scores = sorted_scores[:5]
        total_top_5 = sum(count for _, count in top_5_scores) if top_5_scores else 0
        
        # En olası skor ve olasılığı
        most_likely_score = sorted_scores[0] if sorted_scores else ("0-0", 0)
        most_likely_score_prob = most_likely_score[1] / len(all_home_goals) if all_home_goals else 0
        
        # En olası 5 skorun olasılıkları
        top_5_probs = {score: count / len(all_home_goals) for score, count in top_5_scores} if all_home_goals else {}
        
        logger.info(f"En olası skorlar ve olasılıkları: {top_5_probs}")
        
        # KG VAR/YOK kısıtlamasına göre log
        if kg_var_prediction is True:
            logger.info("Monte Carlo simülasyonu KG VAR kısıtlamasıyla çalıştırıldı")
        elif kg_var_prediction is False:
            logger.info("Monte Carlo simülasyonu KG YOK kısıtlamasıyla çalıştırıldı")
        
        # Sonuçları döndür
        return {
            "match_result_probs": {
                "home_win": home_win_prob,
                "draw": draw_prob,
                "away_win": away_win_prob
            },
            "full_time_results": full_time_results,
            "exact_scores": exact_scores,
            "most_likely_score": most_likely_score,
            "most_likely_score_prob": most_likely_score_prob,
            "top_5_scores": top_5_scores,
            "top_5_probs": top_5_probs,
            "avg_goals": {
                "home": avg_home_goals,
                "away": avg_away_goals,
                "total": avg_total_goals
            },
            "over_under": {
                "over_2_5": sum(1 for h, a in zip(all_home_goals, all_away_goals) if h + a > 2.5) / len(all_home_goals) if all_home_goals else 0.5,
                "under_2_5": sum(1 for h, a in zip(all_home_goals, all_away_goals) if h + a < 2.5) / len(all_home_goals) if all_home_goals else 0.5,
                "over_3_5": sum(1 for h, a in zip(all_home_goals, all_away_goals) if h + a > 3.5) / len(all_home_goals) if all_home_goals else 0.3,
                "under_3_5": sum(1 for h, a in zip(all_home_goals, all_away_goals) if h + a < 3.5) / len(all_home_goals) if all_home_goals else 0.7
            },
            "both_teams_to_score": {
                "yes": self._calculate_kg_var_probability(all_home_goals, all_away_goals, home_goals_lambda, away_goals_lambda, home_form, away_form),
                "no": self._calculate_kg_yok_probability(all_home_goals, all_away_goals, home_goals_lambda, away_goals_lambda, home_form, away_form)
            },
            "all_home_goals": all_home_goals,
            "all_away_goals": all_away_goals,
            "simulations": len(all_home_goals) if all_home_goals else 0
        }

    def build_neural_network(self, input_dim):
        """Sinir ağı modeli oluştur"""
        model = Sequential()
        model.add(Dense(64, input_dim=input_dim, activation='relu'))
        model.add(Dropout(0.2))  # Overfitting'i önlemek için dropout
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='linear'))  # Gol tahmini için lineer aktivasyon
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
        return model

    def prepare_data_for_neural_network(self, team_form, is_home=True):
        """Sinir ağı için veri hazırla"""
        if not team_form:
            return None

        if is_home:
            performance = team_form.get('home_performance', {})
            bayesian = team_form.get('bayesian', {})

            features = [
                performance.get('avg_goals_scored', 0),
                performance.get('avg_goals_conceded', 0),
                performance.get('weighted_avg_goals_scored', 0),
                performance.get('weighted_avg_goals_conceded', 0),
                performance.get('form_points', 0),
                performance.get('weighted_form_points', 0),
                bayesian.get('home_lambda_scored', 0),
                bayesian.get('home_lambda_conceded', 0),
                team_form.get('recent_matches', 0),
                team_form.get('home_matches', 0)
            ]
        else:
            performance = team_form.get('away_performance', {})
            bayesian = team_form.get('bayesian', {})

            features = [
                performance.get('avg_goals_scored', 0),
                performance.get('avg_goals_conceded', 0),
                performance.get('weighted_avg_goals_scored', 0),
                performance.get('weighted_avg_goals_conceded', 0),
                performance.get('form_points', 0),
                performance.get('weighted_form_points', 0),
                bayesian.get('away_lambda_scored', 0),
                bayesian.get('away_lambda_conceded', 0),
                team_form.get('recent_matches', 0),
                team_form.get('away_matches', 0)
            ]

        return np.array(features).reshape(1, -1)

    def train_neural_network(self, X_train, y_train, is_home=True):
        """Sinir ağını eğit"""
        try:
            X_scaled = self.scaler.fit_transform(X_train)
            model = self.model_home if is_home else self.model_away

            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            model.fit(
                X_scaled, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1
            )

            # Modeli kaydet
            model_path = 'model_home.h5' if is_home else 'model_away.h5'
            save_model(model, model_path)
            logger.info(f"Sinir ağı modeli kaydedildi: {model_path}")

            return model
        except Exception as e:
            logger.error(f"Sinir ağı eğitilirken hata: {str(e)}")
            return None

    def calculate_weighted_form(self, matches, decay_factor=0.9):
        """
        Son maçları azalan ağırlıklarla değerlendiren fonksiyon
        En son maç en yüksek ağırlığa sahip (1.0), öncekiler geometrik azalır
        
        Args:
            matches: Maç listesi (en yeniden en eskiye doğru sıralı)
            decay_factor: Azalma faktörü (0.9 = her bir önceki maç %10 daha az önemli)
        
        Returns:
            weighted_form: Ağırlıklı form puanı
        """
        if not matches:
            return {
                'weighted_goals_scored': 1.0,
                'weighted_goals_conceded': 1.0,
                'weighted_points': 1.0,
                'confidence': 0.0
            }
            
        weights = [decay_factor ** i for i in range(len(matches))]
        total_weight = sum(weights)
        
        weighted_goals_scored = 0
        weighted_goals_conceded = 0
        weighted_points = 0
        
        for i, match in enumerate(matches):
            weight = weights[i] / total_weight
            weighted_goals_scored += match.get('goals_scored', 0) * weight
            weighted_goals_conceded += match.get('goals_conceded', 0) * weight
            
            # Galibiyet = 3, Beraberlik = 1, Mağlubiyet = 0
            result = match.get('ft_result', '')
            points = 3 if result == 'W' else (1 if result == 'D' else 0)
            weighted_points += points * weight
        
        # Güven faktörü - ne kadar çok maç varsa o kadar güvenilir (maksimum 10 maç için 1.0)
        confidence = min(1.0, len(matches) / 10)
        
        return {
            'weighted_goals_scored': weighted_goals_scored,
            'weighted_goals_conceded': weighted_goals_conceded,
            'weighted_points': weighted_points,
            'confidence': confidence
        }
        
    def get_team_form(self, team_id, last_matches=21):
        """Takımın son maçlarındaki performansını al - son 21 maç verisi için tam değerlendirme"""
        try:
            # Son 12 aylık maçları al (daha uzun süreli veri için)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)

            url = "https://apiv3.apifootball.com/"
            params = {
                'action': 'get_events',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'team_id': team_id,
                'APIkey': self.api_key
            }

            response = requests.get(url, params=params)

            if response.status_code != 200:
                logger.error(f"API hatası: {response.status_code}")
                return None

            matches = response.json()

            if not isinstance(matches, list):
                logger.error(f"Beklenmeyen API yanıtı: {matches}")
                return None

            # Maçları tarihe göre sırala (en yeniden en eskiye)
            matches.sort(key=lambda x: x.get('match_date', ''), reverse=True)

            # Son maçları al
            recent_matches = matches[:last_matches]

            # Form verilerini hesapla
            goals_scored = 0
            goals_conceded = 0
            points = 0

            # Ev ve deplasman maçları için ayrı değişkenler
            home_matches = []
            away_matches = []
            home_goals_scored = 0
            home_goals_conceded = 0
            away_goals_scored = 0
            away_goals_conceded = 0
            home_points = 0
            away_points = 0

            # Üstel ağırlıklandırma için decay factor
            decay_factor = 0.9

            # Son maçların detaylarını ekle ve ağırlıklı ortalamalar için gerekli verileri topla
            recent_match_data = []

            # Ağırlıklandırma için kullanılacak değerler
            total_weights = 0
            total_home_weights = 0
            total_away_weights = 0
            weighted_goals_scored = 0
            weighted_goals_conceded = 0
            weighted_home_goals_scored = 0
            weighted_home_goals_conceded = 0
            weighted_away_goals_scored = 0
            weighted_away_goals_conceded = 0
            weighted_points = 0
            weighted_home_points = 0
            weighted_away_points = 0

            for i, match in enumerate(recent_matches):
                # Sadece tamamlanmış maçları dahil et
                match_status = match.get('match_status', '')
                if match_status != 'Finished' and match_status != 'Match Finished' and match_status != 'FT':
                    continue
                    
                home_team_id = match.get('match_hometeam_id')
                home_score = int(match.get('match_hometeam_score', 0) or 0)
                away_score = int(match.get('match_awayteam_score', 0) or 0)

                # Bu maç için ağırlık hesapla (üstel azalma modeli)
                weight = decay_factor ** i
                total_weights += weight

                # Takım ev sahibi ise
                is_home = home_team_id == team_id
                goals_for = home_score if is_home else away_score
                goals_against = away_score if is_home else home_score

                if is_home:
                    home_matches.append(match)
                    home_goals_scored += goals_for
                    home_goals_conceded += goals_against
                    total_home_weights += weight
                    weighted_home_goals_scored += goals_for * weight
                    weighted_home_goals_conceded += goals_against * weight

                    if home_score > away_score:  # Galibiyet
                        home_points += 3
                        weighted_home_points += 3 * weight
                    elif home_score == away_score:  # Beraberlik
                        home_points += 1
                        weighted_home_points += 1 * weight
                else:
                    away_matches.append(match)
                    away_goals_scored += goals_for
                    away_goals_conceded += goals_against
                    total_away_weights += weight
                    weighted_away_goals_scored += goals_for * weight
                    weighted_away_goals_conceded += goals_against * weight

                    if away_score > home_score:  # Galibiyet
                        away_points += 3
                        weighted_away_points += 3 * weight
                    elif away_score == home_score:  # Beraberlik
                        away_points += 1
                        weighted_away_points += 1 * weight

                # Tüm maçlar için toplamlar
                goals_scored += goals_for
                goals_conceded += goals_against
                weighted_goals_scored += goals_for * weight
                weighted_goals_conceded += goals_against * weight

                if (is_home and home_score > away_score) or (not is_home and away_score > home_score):
                    points += 3
                    weighted_points += 3 * weight
                elif home_score == away_score:
                    points += 1
                    weighted_points += 1 * weight

                # İlk yarı skorlarını al
                half_time_home = int(match.get('match_hometeam_halftime_score', 0) or 0)
                half_time_away = int(match.get('match_awayteam_halftime_score', 0) or 0)
                
                # İlk yarı ve tam maç sonuçlarını belirle
                ht_result = 'W' if (is_home and half_time_home > half_time_away) or (not is_home and half_time_away > half_time_home) else \
                           'D' if half_time_home == half_time_away else 'L'
                
                ft_result = 'W' if (is_home and home_score > away_score) or (not is_home and away_score > home_score) else \
                           'D' if home_score == away_score else 'L'
                
                # İY/MS formatı (1/1, 1/X, 1/2, X/1, X/X, X/2, 2/1, 2/X, 2/2)
                iy_ms_code = ''
                if is_home:
                    if half_time_home > half_time_away:
                        iy_ms_code = '1'
                    elif half_time_home == half_time_away:
                        iy_ms_code = 'X'
                    else:
                        iy_ms_code = '2'
                        
                    if home_score > away_score:
                        iy_ms_code += '/1'
                    elif home_score == away_score:
                        iy_ms_code += '/X'
                    else:
                        iy_ms_code += '/2'
                else:
                    if half_time_away > half_time_home:
                        iy_ms_code = '1'
                    elif half_time_away == half_time_home:
                        iy_ms_code = 'X'
                    else:
                        iy_ms_code = '2'
                        
                    if away_score > home_score:
                        iy_ms_code += '/1'
                    elif away_score == home_score:
                        iy_ms_code += '/X'
                    else:
                        iy_ms_code += '/2'
                
                # İlk yarı skoru
                ht_goals_for = half_time_home if is_home else half_time_away
                ht_goals_against = half_time_away if is_home else half_time_home
                
                # Maç verisini ekle
                match_data = {
                    'date': match.get('match_date', ''),
                    'league': match.get('league_name', ''),
                    'opponent': match.get('match_awayteam_name', '') if is_home else match.get('match_hometeam_name', ''),
                    'is_home': is_home,
                    'goals_scored': goals_for,
                    'goals_conceded': goals_against,
                    'ht_goals_scored': ht_goals_for,
                    'ht_goals_conceded': ht_goals_against,
                    'ht_result': ht_result,  # İlk yarı sonucu
                    'ft_result': ft_result,  # Tam maç sonucu
                    'ht_ft_code': iy_ms_code,  # İY/MS kodu
                    'result': ft_result  # Eski format ile uyumluluk için
                }
                recent_match_data.append(match_data)

            # Ortalama değerler hesapla
            avg_goals_scored = goals_scored / len(recent_matches) if recent_matches else 0
            avg_goals_conceded = goals_conceded / len(recent_matches) if recent_matches else 0
            form_points = points / (len(recent_matches) * 3) if recent_matches else 0

            # Ağırlıklı ortalamalar hesapla
            weighted_avg_goals_scored = weighted_goals_scored / total_weights if total_weights > 0 else 0
            weighted_avg_goals_conceded = weighted_goals_conceded / total_weights if total_weights > 0 else 0
            weighted_form_points = weighted_points / (total_weights * 3) if total_weights > 0 else 0

            # Ev ve deplasman için ortalamalar
            avg_home_goals_scored = home_goals_scored / len(home_matches) if home_matches else 0
            avg_home_goals_conceded = home_goals_conceded / len(home_matches) if home_matches else 0
            avg_away_goals_scored = away_goals_scored / len(away_matches) if away_matches else 0
            avg_away_goals_conceded = away_goals_conceded / len(away_matches) if away_matches else 0

            # Ev ve deplasman için ağırlıklı ortalamalar
            weighted_avg_home_goals_scored = weighted_home_goals_scored / total_home_weights if total_home_weights > 0 else 0
            weighted_avg_home_goals_conceded = weighted_home_goals_conceded / total_home_weights if total_home_weights > 0 else 0
            weighted_avg_away_goals_scored = weighted_away_goals_scored / total_away_weights if total_away_weights > 0 else 0
            weighted_avg_away_goals_conceded = weighted_away_goals_conceded / total_away_weights if total_away_weights > 0 else 0

            # Puanlar
            home_form_points = home_points / (len(home_matches) * 3) if home_matches else 0
            away_form_points = away_points / (len(away_matches) * 3) if away_matches else 0

            # Ağırlıklı form puanları
            weighted_home_form_points = weighted_home_points / (total_home_weights * 3) if total_home_weights > 0 else 0
            weighted_away_form_points = weighted_away_points / (total_away_weights * 3) if total_away_weights > 0 else 0

            # Bayesyen güncelleme için parametreler
            n_home = len(home_matches)
            n_away = len(away_matches)

            # Bayesyen posterior hesapla - gol atma
            lambda_home_scored = (self.alpha_ev_atma + home_goals_scored) / (self.beta_ev + n_home) if n_home > 0 else self.lig_ortalamasi_ev_gol
            lambda_away_scored = (self.alpha_deplasman_atma + away_goals_scored) / (self.beta_deplasman + n_away) if n_away > 0 else self.lig_ortalamasi_deplasman_gol

            # Bayesyen posterior hesapla - gol yeme
            lambda_home_conceded = (self.alpha_deplasman_atma + home_goals_conceded) / (self.beta_deplasman + n_home) if n_home > 0 else self.lig_ortalamasi_deplasman_gol
            lambda_away_conceded = (self.alpha_ev_atma + away_goals_conceded) / (self.beta_ev + n_away) if n_away > 0 else self.lig_ortalamasi_ev_gol

            # İlk yarı istatistiklerini analiz et
            ht_goals_scored_home = 0
            ht_goals_conceded_home = 0
            ht_goals_scored_away = 0
            ht_goals_conceded_away = 0
            
            # İlk yarı sonuçları analizi
            ht_home_wins = 0  # Ev sahibi olarak ilk yarı galibiyetleri
            ht_home_draws = 0  # Ev sahibi olarak ilk yarı beraberlikleri
            ht_home_losses = 0  # Ev sahibi olarak ilk yarı mağlubiyetleri
            ht_away_wins = 0  # Deplasman olarak ilk yarı galibiyetleri
            ht_away_draws = 0  # Deplasman olarak ilk yarı beraberlikleri
            ht_away_losses = 0  # Deplasman olarak ilk yarı mağlubiyetleri
            
            # İY/MS kombinasyonları sayaçları
            ht_ft_counts = {
                '1/1': 0, '1/X': 0, '1/2': 0,
                'X/1': 0, 'X/X': 0, 'X/2': 0,
                '2/1': 0, '2/X': 0, '2/2': 0
            }
            
            # Ev/deplasman İY/MS kombinasyonları
            home_ht_ft_counts = {
                '1/1': 0, '1/X': 0, '1/2': 0,
                'X/1': 0, 'X/X': 0, 'X/2': 0,
                '2/1': 0, '2/X': 0, '2/2': 0
            }
            
            away_ht_ft_counts = {
                '1/1': 0, '1/X': 0, '1/2': 0,
                'X/1': 0, 'X/X': 0, 'X/2': 0,
                '2/1': 0, '2/X': 0, '2/2': 0
            }
            
            # İlk yarı verilerini hesapla
            for match in home_matches:
                match_data = next((m for m in recent_match_data if m.get('is_home', False) and m.get('date') == match.get('match_date')), None)
                if match_data:
                    ht_goals_scored_home += match_data.get('ht_goals_scored', 0)
                    ht_goals_conceded_home += match_data.get('ht_goals_conceded', 0)
                    
                    # İlk yarı sonuçları
                    ht_result = match_data.get('ht_result', '')
                    if ht_result == 'W':
                        ht_home_wins += 1
                    elif ht_result == 'D':
                        ht_home_draws += 1
                    elif ht_result == 'L':
                        ht_home_losses += 1
                    
                    # İY/MS istatistiği ekle
                    ht_ft_code = match_data.get('ht_ft_code', '')
                    if ht_ft_code in ht_ft_counts:
                        ht_ft_counts[ht_ft_code] += 1
                        home_ht_ft_counts[ht_ft_code] += 1
            
            for match in away_matches:
                match_data = next((m for m in recent_match_data if not m.get('is_home', True) and m.get('date') == match.get('match_date')), None)
                if match_data:
                    ht_goals_scored_away += match_data.get('ht_goals_scored', 0)
                    ht_goals_conceded_away += match_data.get('ht_goals_conceded', 0)
                    
                    # İlk yarı sonuçları
                    ht_result = match_data.get('ht_result', '')
                    if ht_result == 'W':
                        ht_away_wins += 1
                    elif ht_result == 'D':
                        ht_away_draws += 1
                    elif ht_result == 'L':
                        ht_away_losses += 1
                    
                    # İY/MS istatistiği ekle
                    ht_ft_code = match_data.get('ht_ft_code', '')
                    if ht_ft_code in ht_ft_counts:
                        ht_ft_counts[ht_ft_code] += 1
                        away_ht_ft_counts[ht_ft_code] += 1
            
            # İlk yarı ortalama değerleri
            avg_ht_goals_scored_home = ht_goals_scored_home / len(home_matches) if home_matches else 0
            avg_ht_goals_conceded_home = ht_goals_conceded_home / len(home_matches) if home_matches else 0
            avg_ht_goals_scored_away = ht_goals_scored_away / len(away_matches) if away_matches else 0
            avg_ht_goals_conceded_away = ht_goals_conceded_away / len(away_matches) if away_matches else 0
            
            # İlk yarı / İkinci yarı eğilimlerini hesapla
            first_half_performance = 0.5  # Varsayılan değer
            second_half_performance = 0.5  # Varsayılan değer
            
            # İlk yarı ve ikinci yarı gol oranlarını hesapla
            all_first_half_goals = 0
            all_second_half_goals = 0
            
            for match in recent_match_data:
                all_first_half_goals += match.get('ht_goals_scored', 0)
                all_second_half_goals += match.get('goals_scored', 0) - match.get('ht_goals_scored', 0)
            
            # Toplam gollerin %'si olarak ifade et
            total_goals = all_first_half_goals + all_second_half_goals
            if total_goals > 0:
                first_half_performance = all_first_half_goals / total_goals
                second_half_performance = all_second_half_goals / total_goals
            
            # İY/MS eğilimlerini hesapla - en sık görülen 3 kombinasyon
            top_ht_ft = sorted(ht_ft_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            
            return {
                'avg_goals_scored': avg_goals_scored,
                'avg_goals_conceded': avg_goals_conceded,
                'weighted_avg_goals_scored': weighted_avg_goals_scored,
                'weighted_avg_goals_conceded': weighted_avg_goals_conceded,
                'form_points': form_points,
                'weighted_form_points': weighted_form_points,
                'recent_matches': len(recent_matches),
                'home_matches': len(home_matches),
                'away_matches': len(away_matches),
                'home_performance': {
                    'avg_goals_scored': avg_home_goals_scored, 
                    'avg_goals_conceded': avg_home_goals_conceded,
                    'weighted_avg_goals_scored': weighted_avg_home_goals_scored,
                    'weighted_avg_goals_conceded': weighted_avg_home_goals_conceded,
                    'form_points': home_form_points,
                    'weighted_form_points': weighted_home_form_points,
                    'bayesian_goals_scored': lambda_home_scored,
                    'bayesian_goals_conceded': lambda_home_conceded,
                    # İlk yarı istatistikleri
                    'avg_ht_goals_scored': avg_ht_goals_scored_home,
                    'avg_ht_goals_conceded': avg_ht_goals_conceded_home,
                    'ht_wins': ht_home_wins,
                    'ht_draws': ht_home_draws,
                    'ht_losses': ht_home_losses,
                    'ht_ft_stats': home_ht_ft_counts
                },
                'away_performance': {
                    'avg_goals_scored': avg_away_goals_scored,
                    'avg_goals_conceded': avg_away_goals_conceded, 
                    'weighted_avg_goals_scored': weighted_avg_away_goals_scored,
                    'weighted_avg_goals_conceded': weighted_avg_away_goals_conceded,
                    'form_points': away_form_points,
                    'weighted_form_points': weighted_away_form_points,
                    'bayesian_goals_scored': lambda_away_scored,
                    'bayesian_goals_conceded': lambda_away_conceded,
                    # İlk yarı istatistikleri
                    'avg_ht_goals_scored': avg_ht_goals_scored_away,
                    'avg_ht_goals_conceded': avg_ht_goals_conceded_away,
                    'ht_wins': ht_away_wins,
                    'ht_draws': ht_away_draws,
                    'ht_losses': ht_away_losses,
                    'ht_ft_stats': away_ht_ft_counts
                },
                'recent_match_data': recent_match_data,
                'detailed_data': {
                    'last_5': recent_match_data[:5],
                    'last_10': recent_match_data[:10],
                    'last_15': recent_match_data[:15],
                    'all': recent_match_data
                },
                'bayesian': {
                    'home_lambda_scored': lambda_home_scored,
                    'home_lambda_conceded': lambda_home_conceded,
                    'away_lambda_scored': lambda_away_scored,
                    'away_lambda_conceded': lambda_away_conceded,
                    'n_home': n_home,
                    'n_away': n_away
                },
                # İlk yarı / İkinci yarı performans analizi
                'half_time_analysis': {
                    'first_half_performance': first_half_performance,
                    'second_half_performance': second_half_performance,
                    'ht_ft_trends': top_ht_ft,
                    'ht_ft_counts': ht_ft_counts,
                    'first_half_goals': all_first_half_goals,
                    'second_half_goals': all_second_half_goals
                }
            }

        except Exception as e:
            logger.error(f"Takım formu alınırken hata: {str(e)}")
            return None

    def predict_match(self, home_team_id, away_team_id, home_team_name, away_team_name, force_update=False, 
                   use_specialized_models=True, use_goal_trend_analysis=True):
        """Maç sonucunu tahmin et - Gelişmiş algoritma sıralaması ve tutarlılık kontrolü ile
        
        Algoritma sıralaması:
        1. Önbellekte varsa önbellekten getir veya önceki model değerlerini kullan
        2. Temel istatistik ve form analizlerini yap
        3. Monte Carlo simülasyonu ile ilk tahminleri oluştur
        4. Gelişmiş faktörleri (maç önemi, tarihi patern, vb) uygula
        5. Gol trendi analizini (ivme analizi) uygula
        6. Takım spesifik modelleri uygula
        7. Özelleştirilmiş modelleri (düşük/orta/yüksek skorlu) uygula
        8. KG VAR/YOK ve skor tutarlılığını kontrol et
        9. ÜST/ALT ve skor tutarlılığını kontrol et
        10. Genel sonuç tutarlılığını kontrol et
        
        Args:
            home_team_id: Ev sahibi takım ID'si
            away_team_id: Deplasman takımı ID'si
            home_team_name: Ev sahibi takım adı
            away_team_name: Deplasman takımı adı
            force_update: Önbellekteki tahmin varsa da yeniden hesapla
            use_specialized_models: Özelleştirilmiş modelleri (düşük/orta/yüksek skorlu) kullan
            use_goal_trend_analysis: Gol trend ivme analizini kullan
            
        Returns:
            dict: Tahmin sonuçları
        """
        # Kullanılacak modelleri belirle
        global ADVANCED_MODELS_AVAILABLE, TEAM_SPECIFIC_MODELS_AVAILABLE, ENHANCED_MONTE_CARLO_AVAILABLE, SPECIALIZED_MODELS_AVAILABLE
        use_ensemble_models = False
        use_gbm_models = False
        use_enhanced_monte_carlo = ENHANCED_MONTE_CARLO_AVAILABLE and hasattr(self, 'enhanced_monte_carlo')
        use_bayesian_models = False
        use_team_specific_models = False
        use_specialized_models = False
        
        # 1. Gelişmiş tahmin algoritmaları (GBM, LSTM, Bayesian Networks)
        if ADVANCED_MODELS_AVAILABLE and hasattr(self, 'advanced_models'):
            logger.info(f"Gelişmiş makine öğrenmesi modelleri kullanılıyor: {home_team_name} vs {away_team_name}")
            use_gbm_models = True
            use_bayesian_models = True
        
        # 2. Ensemble Score predictor (mevcut tahmin modelimiz)
        try:
            from zip_and_ensemble_predictor import AdvancedScorePredictor
            advanced_predictor = AdvancedScorePredictor()
            use_ensemble_models = True
            logger.info("Gelişmiş ensemble tahmin modeli başarıyla yüklendi")
        except Exception as e:
            logger.warning(f"Ensemble tahmin modeli yüklenemedi: {e}")
            use_ensemble_models = False
            
        # 3. Takım-spesifik modeller (lig ve takıma özel parametreler)
        if 'TEAM_SPECIFIC_MODELS_AVAILABLE' in globals() and globals()['TEAM_SPECIFIC_MODELS_AVAILABLE'] and hasattr(self, 'team_specific_predictor'):
            logger.info(f"Takım-spesifik tahmin modelleri kullanılıyor: {home_team_name} vs {away_team_name}")
            use_team_specific_models = True
            
        # 4. Özelleştirilmiş modeller (düşük, orta ve yüksek skorlu maçlar için)
        if use_specialized_models and 'SPECIALIZED_MODELS_AVAILABLE' in globals() and globals()['SPECIALIZED_MODELS_AVAILABLE'] and hasattr(self, 'specialized_models'):
            logger.info(f"Özelleştirilmiş tahmin modelleri (düşük/orta/yüksek skorlu maçlar) kullanılıyor: {home_team_name} vs {away_team_name}")
        else:
            use_specialized_models = False
            if not hasattr(self, 'specialized_models'):
                logger.warning(f"Özelleştirilmiş tahmin modelleri sınıfı bulunamadı.")
            elif 'SPECIALIZED_MODELS_AVAILABLE' not in globals() or not globals()['SPECIALIZED_MODELS_AVAILABLE']:
                logger.warning(f"Özelleştirilmiş tahmin modelleri kullanılamıyor.")

        # Tahmin öncesi sinir ağlarını eğit
        logger.info(f"{home_team_name} vs {away_team_name} için sinir ağları eğitiliyor...")
        self.collect_training_data()

        # Önbelleği kontrol et
        cache_key = f"{home_team_id}_{away_team_id}"
        force_new_prediction = False

        if cache_key in self.predictions_cache and not force_update:
            prediction = self.predictions_cache[cache_key]
            # Tahmin 24 saatten eski değilse onu kullan
            cached_time = datetime.fromtimestamp(prediction.get('timestamp', 0))

            # Eski algoritma ile yapılan tahminleri kontrol et (neural_predictions yoksa eski versiyon)
            if 'predictions' in prediction and 'neural_predictions' not in prediction.get('predictions', {}):
                logger.info(f"Eski algoritma ile yapılmış tahmin bulundu: {home_team_name} vs {away_team_name}")
                force_new_prediction = True
            # Tahmin 24 saatten eski değilse ve güncel algoritma ile yapılmışsa onu kullan
            elif datetime.now() - cached_time < timedelta(hours=24):
                logger.info(f"Önbellekten tahmin kullanılıyor: {home_team_name} vs {away_team_name}")
                return prediction
            else:
                force_new_prediction = True
        elif force_update:
            logger.info(f"Zorunlu yeni tahmin yapılıyor: {home_team_name} vs {away_team_name}")
            force_new_prediction = True
        else:
            # Cache'de olmayan tahminler için
            logger.info(f"Önbellekte olmayan tahmin yapılıyor: {home_team_name} vs {away_team_name}")
            force_new_prediction = True

        # Takımların form verilerini al
        home_form = self.get_team_form(home_team_id)
        away_form = self.get_team_form(away_team_id)

        if not home_form or not away_form:
            logger.error(f"Form verileri alınamadı: {home_team_name} vs {away_team_name}")
            return None

        # Gelişmiş tahmin modellerini kullan - YENİ: iyileştirilmiş tutarlılık için daha fazla ağırlık ver
        advanced_prediction = None
        if use_ensemble_models:
            try:
                # Geliştirilmiş algoritma - daha tutarlı tahminler için
                advanced_prediction = advanced_predictor.predict_match(
                    home_form, 
                    away_form, 
                    self.predictions_cache,
                    model_weight=0.4,  # Yeni sistem ağırlığı %40, eski sistem %60 
                    simulations=10000  # Daha fazla simülasyon - daha doğru olasılıklar için
                )

                if advanced_prediction:
                    logger.info(f"Gelişmiş tutarlı tahmin modelleri başarıyla kullanıldı: {home_team_name} vs {away_team_name}")
                    # Gelişmiş tahmin sonuçlarını kullan
                    adv_home_goals = advanced_prediction['expected_goals']['home']
                    adv_away_goals = advanced_prediction['expected_goals']['away']
                    logger.info(f"Tutarlı tahmin modeli: Ev {adv_home_goals:.2f} - Deplasman {adv_away_goals:.2f}")
                    
                    # Eğer gelişmiş betting_predictions varsa, bunları da kullanacağız
                    adv_betting_predictions = advanced_prediction.get('betting_predictions', {})
                    if adv_betting_predictions:
                        logger.info(f"Gelişmiş bahis tahminleri mevcut: {list(adv_betting_predictions.keys())}")
            except Exception as e:
                logger.error(f"Gelişmiş tahmin modelleri hatası: {e}")
                advanced_prediction = None

        # Sinir ağı için veri hazırla
        home_features = self.prepare_data_for_neural_network(home_form, is_home=True)
        away_features = self.prepare_data_for_neural_network(away_form, is_home=False)

        # Sinir ağı modelleri kontrol et
        if self.model_home is None or self.model_away is None:
            self.load_or_create_models()

        # Monte Carlo simülasyonu yap (5000 maç simüle et)
        home_wins = 0
        away_wins = 0
        draws = 0
        both_teams_scored = 0
        over_2_5_goals = 0
        over_3_5_goals = 0
        simulations = 5000  # Daha fazla simülasyon

        # Ev sahibi avantajı faktörünü son maçlara göre dinamik olarak hesaplayalım
        # Son 5 ev sahibi maçını analiz et
        home_matches_as_home = [m for m in home_form.get('recent_match_data', []) if m.get('is_home', False)][:5]
        home_as_home_points = 0
        
        if home_matches_as_home:
            for match in home_matches_as_home:
                if match.get('result') == 'W':
                    home_as_home_points += 3
                elif match.get('result') == 'D':
                    home_as_home_points += 1
            
            # Ev sahibi puan performansına göre avantaj belirle - daha düşük avantaj katsayıları
            if home_as_home_points >= 10:  # Mükemmel ev performansı
                home_advantage = 1.15  # %15 avantaj
                logger.info(f"Güçlü ev sahibi avantajı: Son 5 ev maçında {home_as_home_points} puan")
            elif home_as_home_points >= 7:  # İyi ev performansı
                home_advantage = 1.08  # %8 avantaj
                logger.info(f"Normal ev sahibi avantajı: Son 5 ev maçında {home_as_home_points} puan")
            elif home_as_home_points >= 4:  # Orta ev performansı
                home_advantage = 1.03  # %3 avantaj
                logger.info(f"Minimal ev sahibi avantajı: Son 5 ev maçında {home_as_home_points} puan")
            else:  # Zayıf ev performansı
                home_advantage = 1.0  # Avantaj yok
                logger.info(f"Ev sahibi avantajı yok: Son 5 ev maçında sadece {home_as_home_points} puan")
        else:
            # Yeterli ev sahibi maç verisi yoksa standart avantaj uygula
            home_advantage = 1.05
            logger.info("Yeterli ev maçı verisi bulunamadı, standart ev avantajı uygulandı.")
            
        # Deplasman avantajını son maçlara göre dinamik olarak hesaplayalım
        # Son 5 deplasman maçını analiz et
        away_matches_as_away = [m for m in away_form.get('recent_match_data', []) if not m.get('is_home', True)][:5]
        away_as_away_points = 0
        
        if away_matches_as_away:
            for match in away_matches_as_away:
                if match.get('result') == 'W':
                    away_as_away_points += 3
                elif match.get('result') == 'D':
                    away_as_away_points += 1
            
            # Deplasman puan performansına göre avantaj belirle
            if away_as_away_points >= 7:  # 7-9 puan ve üzeri ise güçlü deplasman performansı
                away_advantage = 1.10  # İstenen güçlü deplasman avantajı
                logger.info(f"Deplasman avantajı uygulandı: Son 5 deplasman maçında {away_as_away_points} puan kazanılmış (güçlü)")
            else:  # 7 puan altında ise
                away_advantage = 1.00  # Deplasman avantajı uygulanmayacak
                logger.info(f"Deplasman avantajı uygulanmadı: Son 5 deplasman maçında sadece {away_as_away_points} puan kazanılmış (zayıf)")
        else:
            # Yeterli deplasman maç verisi yoksa standart değer kullan
            away_advantage = 1.00
            logger.info("Yeterli deplasman maçı verisi bulunamadı, standart deplasman avantajı uygulandı.")

        # Farklı dönemlere (son 3, 6, 9 maç) göre daha dengeli ağırlıklandırma
        # Form ağırlıkları - daha dengeli dağılım
        # Form ve güç dengesine göre ağırlıkları hesapla - son 5 maça önemli ağırlık ver ama uzun vadeyi de dikkate al 
        weight_last_5 = 2.5   # Son 5 maç - yüksek önem (son karşılaşmaların etkisini koruyarak)
        weight_last_10 = 1.5  # Son 6-10 arası maçlar - orta önem
        weight_last_21 = 1.0  # Son 11-21 arası maçlar - daha düşük ama anlamlı önem (takım gücünü belirler)

        # Takımların farklı dönemlerdeki form verilerini tutacak sözlükler
        home_form_periods = {}
        away_form_periods = {}

        # Sinir ağı tahminleri
        neural_home_goals = 0.0
        neural_away_goals = 0.0

        # Eğer hazır modeller varsa tahmin yap
        if self.model_home is not None and self.model_away is not None and home_features is not None and away_features is not None:
            try:
                # Veriyi normalize et
                scaled_home_features = self.scaler.fit_transform(home_features)
                scaled_away_features = self.scaler.transform(away_features)

                # Tahmin yap
                neural_home_goals = float(self.model_home.predict(scaled_home_features, verbose=0)[0][0])
                neural_away_goals = float(self.model_away.predict(scaled_away_features, verbose=0)[0][0])

                # Tahminleri pozitif değerlere sınırla
                neural_home_goals = max(0.0, neural_home_goals)
                neural_away_goals = max(0.0, neural_away_goals)

                logger.info(f"Sinir ağı tahminleri: Ev {neural_home_goals:.2f} - Deplasman {neural_away_goals:.2f}")
            except Exception as e:
                logger.error(f"Sinir ağı tahmin hatası: {str(e)}")
                # Sinir ağı tahmin hata verirse Bayesyen tahminler kullanılacak
                neural_home_goals = 0.0
                neural_away_goals = 0.0

        # Ev sahibi takımın farklı dönemlerdeki performanslarını hesapla
        home_match_data = home_form.get('recent_match_data', [])

        # Son 3 maç
        if home_form.get('recent_matches', 0) >= 3:
            last_3_home_goals = 0
            last_3_home_conceded = 0
            last_3_home_points = 0

            for i in range(min(3, len(home_match_data))):
                last_3_home_goals += home_match_data[i].get('goals_scored', 0)
                last_3_home_conceded += home_match_data[i].get('goals_conceded', 0)
                if home_match_data[i].get('result') == 'W':
                    last_3_home_points += 3
                elif home_match_data[i].get('result') == 'D':
                    last_3_home_points += 1

            home_form_periods['last_3'] = {
                'avg_goals': last_3_home_goals / 3,
                'avg_conceded': last_3_home_conceded / 3,
                'form_points': last_3_home_points / 9  # 3 maçta maksimum 9 puan alınabilir
            }
        else:
            home_form_periods['last_3'] = {
                'avg_goals': home_form['avg_goals_scored'],
                'avg_conceded': home_form['avg_goals_conceded'],
                'form_points': home_form['form_points']
            }

        # Son 6 maç
        if home_form.get('recent_matches', 0) >= 6:
            last_6_home_goals = 0
            last_6_home_conceded = 0
            last_6_home_points = 0

            for i in range(min(6, len(home_match_data))):
                last_6_home_goals += home_match_data[i].get('goals_scored', 0)
                last_6_home_conceded += home_match_data[i].get('goals_conceded', 0)
                if home_match_data[i].get('result') == 'W':
                    last_6_home_points += 3
                elif home_match_data[i].get('result') == 'D':
                    last_6_home_points += 1

            home_form_periods['last_6'] = {
                'avg_goals': last_6_home_goals / 6,
                'avg_conceded': last_6_home_conceded / 6,
                'form_points': last_6_home_points / 18  # 6 maçta maksimum 18 puan alınabilir
            }
        else:
            home_form_periods['last_6'] = home_form_periods['last_3']

        # Son 9 maç
        if home_form.get('recent_matches', 0) >= 9:
            last_9_home_goals = 0
            last_9_home_conceded = 0
            last_9_home_points = 0

            for i in range(min(9, len(home_match_data))):
                last_9_home_goals += home_match_data[i].get('goals_scored', 0)
                last_9_home_conceded += home_match_data[i].get('goals_conceded', 0)
                if home_match_data[i].get('result') == 'W':
                    last_9_home_points += 3
                elif home_match_data[i].get('result') == 'D':
                    last_9_home_points += 1

            home_form_periods['last_9'] = {
                'avg_goals': last_9_home_goals / 9,
                'avg_conceded': last_9_home_conceded / 9,
                'form_points': last_9_home_points / 27  # 9 maçta maksimum 27 puan alınabilir
            }
        else:
            home_form_periods['last_9'] = home_form_periods['last_6']

        # Deplasman takımının farklı dönemlerdeki performanslarını hesapla
        away_match_data = away_form.get('recent_match_data', [])

        # Son 3 maç
        if away_form.get('recent_matches', 0) >= 3:
            last_3_away_goals = 0
            last_3_away_conceded = 0
            last_3_away_points = 0

            for i in range(min(3, len(away_match_data))):
                last_3_away_goals += away_match_data[i].get('goals_scored', 0)
                last_3_away_conceded += away_match_data[i].get('goals_conceded', 0)
                if away_match_data[i].get('result') == 'W':
                    last_3_away_points += 3
                elif away_match_data[i].get('result') == 'D':
                    last_3_away_points += 1

            away_form_periods['last_3'] = {
                'avg_goals': last_3_away_goals / 3,
                'avg_conceded': last_3_away_conceded / 3,
                'form_points': last_3_away_points / 9  # 3 maçta maksimum 9 puan alınabilir
            }
        else:
            away_form_periods['last_3'] = {
                'avg_goals': away_form['avg_goals_scored'],
                'avg_conceded': away_form['avg_goals_conceded'],
                'form_points': away_form['form_points']
            }

        # Son 6 maç
        if away_form.get('recent_matches', 0) >= 6:
            last_6_away_goals = 0
            last_6_away_conceded = 0
            last_6_away_points = 0

            for i in range(min(6, len(away_match_data))):
                last_6_away_goals += away_match_data[i].get('goals_scored', 0)
                last_6_away_conceded += away_match_data[i].get('goals_conceded', 0)
                if away_match_data[i].get('result') == 'W':
                    last_6_away_points += 3
                elif away_match_data[i].get('result') == 'D':
                    last_6_away_points += 1

            away_form_periods['last_6'] = {
                'avg_goals': last_6_away_goals / 6,
                'avg_conceded': last_6_away_conceded / 6,
                'form_points': last_6_away_points / 18  # 6 maçta maksimum 18 puan alınabilir
            }
        else:
            away_form_periods['last_6'] = away_form_periods['last_3']

        # Son 9 maç
        if away_form.get('recent_matches', 0) >= 9:
            last_9_away_goals = 0
            last_9_away_conceded = 0
            last_9_away_points = 0

            for i in range(min(9, len(away_match_data))):
                last_9_away_goals += away_match_data[i].get('goals_scored', 0)
                last_9_away_conceded += away_match_data[i].get('goals_conceded', 0)
                if away_match_data[i].get('result') == 'W':
                    last_9_away_points += 3
                elif away_match_data[i].get('result') == 'D':
                    last_9_away_points += 1

            away_form_periods['last_9'] = {
                'avg_goals': last_9_away_goals / 9,
                'avg_conceded': last_9_away_conceded / 9,
                'form_points': last_9_away_points / 27  # 9 maçta maksimum 27 puan alınabilir
            }
        else:
            away_form_periods['last_9'] = away_form_periods['last_6']

        # Ağırlıklı beklenen gol hesaplamaları (son 3-6-9 maçın farklı ağırlıklarıyla)
        # Toplam ağırlık normalizasyonu için kullanılacak değer
        total_weight = weight_last_5 + weight_last_10 + weight_last_21

        # Ev sahibi takımın ağırlıklı beklenen gol sayısı - son 5, son 10 ve son 21 maç verileri ile
        weighted_home_goals = (
            home_form_periods['last_3']['avg_goals'] * weight_last_5 +  # Son 5 maç verileri
            home_form_periods['last_6']['avg_goals'] * weight_last_10 + # Son 6-10 arası
            home_form_periods['last_9']['avg_goals'] * weight_last_21   # Son 11-21 arası
        ) / total_weight

        # Ev sahibi takımın ağırlıklı form puanı - son 5, son 10 ve son 21 maç verileri ile
        weighted_home_form_points = (
            home_form_periods['last_3']['form_points'] * weight_last_5 +  # Son 5 maç verileri
            home_form_periods['last_6']['form_points'] * weight_last_10 + # Son 6-10 arası
            home_form_periods['last_9']['form_points'] * weight_last_21   # Son 11-21 arası
        ) / total_weight

        # Deplasman takımının ağırlıklı beklenen gol sayısı - son 5, son 10 ve son 21 maç verileri ile
        weighted_away_goals = (
            away_form_periods['last_3']['avg_goals'] * weight_last_5 +  # Son 5 maç verileri
            away_form_periods['last_6']['avg_goals'] * weight_last_10 + # Son 6-10 arası
            away_form_periods['last_9']['avg_goals'] * weight_last_21   # Son 11-21 arası
        ) / total_weight

        # Deplasman takımının ağırlıklı form puanı - son 5, son 10 ve son 21 maç verileri ile
        weighted_away_form_points = (
            away_form_periods['last_3']['form_points'] * weight_last_5 +  # Son 5 maç verileri
            away_form_periods['last_6']['form_points'] * weight_last_10 + # Son 6-10 arası
            away_form_periods['last_9']['form_points'] * weight_last_21   # Son 11-21 arası
        ) / total_weight

        # Savunma performansını da hesaba katarak beklenen gol hesaplaması - son 5, son 10 ve son 21 maç verileri ile
        weighted_home_conceded = (
            home_form_periods['last_3']['avg_conceded'] * weight_last_5 +  # Son 5 maç verileri
            home_form_periods['last_6']['avg_conceded'] * weight_last_10 + # Son 6-10 arası
            home_form_periods['last_9']['avg_conceded'] * weight_last_21   # Son 11-21 arası
        ) / total_weight

        weighted_away_conceded = (
            away_form_periods['last_3']['avg_conceded'] * weight_last_5 +  # Son 5 maç verileri
            away_form_periods['last_6']['avg_conceded'] * weight_last_10 + # Son 6-10 arası
            away_form_periods['last_9']['avg_conceded'] * weight_last_21   # Son 11-21 arası
        ) / total_weight

        # Bayesyen ve ağırlıklı yaklaşımların birleşimi ile daha gerçekçi beklenen gol hesaplama
        # 1. Bayesyen güncelleme ile elde edilen değerler
        bayesian_home_attack = home_form.get('bayesian', {}).get('home_lambda_scored', self.lig_ortalamasi_ev_gol)
        bayesian_away_defense = away_form.get('bayesian', {}).get('away_lambda_conceded', self.lig_ortalamasi_ev_gol)
        bayesian_away_attack = away_form.get('bayesian', {}).get('away_lambda_scored', self.lig_ortalamasi_deplasman_gol)
        bayesian_home_defense = home_form.get('bayesian', {}).get('home_lambda_conceded', self.lig_ortalamasi_deplasman_gol)

        # 2. Ağırlıklı ortalama ile hesaplanan değerler (mevcut kod)
        weighted_home_attack = home_form.get('home_performance', {}).get('weighted_avg_goals_scored', weighted_home_goals)
        weighted_away_defense = away_form.get('away_performance', {}).get('weighted_avg_goals_conceded', weighted_away_conceded)
        weighted_away_attack = away_form.get('away_performance', {}).get('weighted_avg_goals_scored', weighted_away_goals)
        weighted_home_defense = home_form.get('home_performance', {}).get('weighted_avg_goals_conceded', weighted_home_conceded)

        # 3. İki yaklaşımı birleştir (0.6 Bayesyen + 0.4 Ağırlıklı ortalama)
        combined_home_attack = bayesian_home_attack * 0.6 + weighted_home_attack * 0.4
        combined_away_defense = bayesian_away_defense * 0.6 + weighted_away_defense * 0.4
        combined_away_attack = bayesian_away_attack * 0.6 + weighted_away_attack * 0.4
        combined_home_defense = bayesian_home_defense * 0.6 + weighted_home_defense * 0.4

        # 4. Saldırı ve savunma güçlerini birleştirerek beklenen gol hesapla
        # Rakiplerin savunma zafiyetlerini değerlendiren faktörler
        # Deplasman takımının savunma zafiyeti faktörü
        away_defense_weakness = 1.0
        if 'away_performance' in away_form and 'weighted_avg_goals_conceded' in away_form['away_performance']:
            # Deplasman takımının ortalamadan ne kadar fazla gol yediğini hesapla
            away_defense_weakness = away_form['away_performance']['weighted_avg_goals_conceded'] / 1.3
            
        # Ev sahibi takımın savunma zafiyeti faktörü
        home_defense_weakness = 1.0
        if 'home_performance' in home_form and 'weighted_avg_goals_conceded' in home_form['home_performance']:
            # Ev sahibi takımının ortalamadan ne kadar fazla gol yediğini hesapla
            home_defense_weakness = home_form['home_performance']['weighted_avg_goals_conceded'] / 1.1
            
        # Değerleri normalizasyon için limitleyelim - çok aşırı değerleri engelle
        away_defense_weakness = min(1.8, max(0.9, away_defense_weakness))
        home_defense_weakness = min(1.8, max(0.9, home_defense_weakness))
        
        logger.info(f"Rakip savunma zafiyet faktörleri - Deplasman: {away_defense_weakness:.2f}, Ev: {home_defense_weakness:.2f}")
        
        # Ev sahibi takımın gol beklentisinde ev avantajını ve rakip takımın savunma zafiyetini kullan
        # Hücum gücüne daha fazla ağırlık ver (0.7 -> 0.85) ve savunmaya daha az (0.3 -> 0.15)
        # Bu değişiklik beklenen gol değerlerinin skor tahminlerine etkisini artıracak
        expected_home_goals = (combined_home_attack * 0.85 + combined_away_defense * 0.15) * home_advantage * away_defense_weakness
        # Deplasman takımın gol beklentisinde deplasman avantajını ve rakip takımın savunma zafiyetini kullan
        # Hücum gücüne daha fazla ağırlık ver (0.7 -> 0.85) ve savunmaya daha az (0.3 -> 0.15)
        expected_away_goals = (combined_away_attack * 0.85 + combined_home_defense * 0.15) * away_advantage * home_defense_weakness
        
        logger.info(f"Ham gol beklentileri: Ev={expected_home_goals:.2f}, Deplasman={expected_away_goals:.2f}")

        # Sinir ağı tahminlerini entegre et (eğer varsa)
        if neural_home_goals > 0 and neural_away_goals > 0:
            # Kombine tahmin: %40 sinir ağı + %60 Bayesyen-Ağırlıklı
            expected_home_goals = expected_home_goals * 0.6 + neural_home_goals * 0.4
            expected_away_goals = expected_away_goals * 0.6 + neural_away_goals * 0.4
            logger.info(f"Sinir ağı entegreli tahminler: Ev {expected_home_goals:.2f} - Deplasman {expected_away_goals:.2f}")
            
        # Özelleştirilmiş modelleri kullan (düşük/orta/yüksek skorlu maçlar için)
        specialized_params = None
        if use_specialized_models and hasattr(self, 'specialized_models'):
            try:
                # Maç kategorisini belirle ve özelleştirilmiş model parametrelerini al
                specialized_prediction = self.specialized_models.predict(
                    home_form, away_form, expected_home_goals, expected_away_goals
                )
                
                if specialized_prediction:
                    # Kategori bilgisini ve model parametrelerini al
                    category = specialized_prediction.get('category', 'normal')
                    specialized_params = specialized_prediction.get('parameters', {})
                    
                    logger.info(f"Maç kategorisi: {category.upper()} (düşük/orta/yüksek skorlu)")
                    
                    # Özelleştirilmiş model parametrelerini log'a yazdır
                    draw_boost = specialized_params.get('draw_boost', 1.0)
                    max_score = specialized_params.get('max_score', 3)
                    poisson_correction = specialized_params.get('poisson_correction', 0.85)
                    
                    logger.info(f"Özelleştirilmiş model parametreleri: "
                                f"beraberlik_çarpanı={draw_boost:.2f}, "
                                f"maksimum_skor={max_score}, "
                                f"poisson_düzeltme={poisson_correction:.2f}")
                    
                    # Özel skor düzeltmeleri ile ilgili bilgileri log'a yazdır
                    score_correction = specialized_params.get('score_correction', {})
                    if score_correction:
                        top_corrections = sorted(score_correction.items(), key=lambda x: x[1], reverse=True)[:3]
                        corrections_str = ", ".join([f"{score}:{factor:.2f}" for score, factor in top_corrections])
                        logger.info(f"En yüksek skor düzeltme çarpanları: {corrections_str}")
            except Exception as e:
                logger.error(f"Özelleştirilmiş model kullanılırken hata: {str(e)}")
                specialized_params = None

        # Global ortalama değerler (lig ortalamaları) - gerçekçi ortalama değerler kullanma
        global_avg_home_goals = 1.6  # Ev sahibi takımlar için genel ortalama gol (düşürüldü)
        global_avg_away_goals = 1.3  # Deplasman takımları için genel ortalama gol (düşürüldü)

        # Mean Reversion uygula - daha gerçekçi gol beklentileri için parametreleri dengele
        # Son form performansına daha fazla ağırlık ver, global ortalamaya daha az
        phi_home = 0.30  # %30 ağırlık global ortalamaya, %70 ağırlık ev sahibi takım performansına
        phi_away = 0.20  # %20 ağırlık global ortalamaya, %80 ağırlık deplasman takım performansına

        # Farklılaştırılmış mean reversion uygula
        expected_home_goals = (1 - phi_home) * expected_home_goals + phi_home * global_avg_home_goals
        expected_away_goals = (1 - phi_away) * expected_away_goals + phi_away * global_avg_away_goals

        # Form farkını daha hassas bir şekilde hesaplama 
        # Katsayıyı azalttık, böylece yüksek form farkı sonuçları abartmayacak
        weighted_home_form_points = home_form.get('home_performance', {}).get('weighted_form_points', weighted_home_form_points)
        weighted_away_form_points = away_form.get('away_performance', {}).get('weighted_form_points', weighted_away_form_points)

        form_diff_home = max(-0.15, min(0.15, 0.05 * (weighted_home_form_points - weighted_away_form_points)))
        form_diff_away = max(-0.15, min(0.15, 0.05 * (weighted_away_form_points - weighted_home_form_points)))

        expected_home_goals = expected_home_goals * (1 + form_diff_home)
        expected_away_goals = expected_away_goals * (1 + form_diff_away)

        # Minimum değerler daha gerçekçi olarak ayarlanıyor, daha düşük minimum değerler
        expected_home_goals = max(0.8, expected_home_goals)
        expected_away_goals = max(0.7, expected_away_goals)

        # Form faktörlerini son 5 maç gol performansına daha fazla ağırlık vererek hesapla
        # Son 5 maçtaki ortalama gol sayılarını kullan
        recent_home_goals_avg = sum(match.get('goals_scored', 0) for match in home_match_data[:5]) / 5 if len(home_match_data) >= 5 else weighted_home_goals
        recent_away_goals_avg = sum(match.get('goals_scored', 0) for match in away_match_data[:5]) / 5 if len(away_match_data) >= 5 else weighted_away_goals
        
        # Son 5 maçtaki gol performansını form faktörüne daha fazla yansıt
        home_recent_factor = recent_home_goals_avg / max(1.0, self.lig_ortalamasi_ev_gol)
        away_recent_factor = recent_away_goals_avg / max(1.0, self.lig_ortalamasi_deplasman_gol)
        
        # Form faktörlerini hesapla - son 5 maç performansını %60 ağırlıkla dahil et
        home_form_factor = min(1.5, (0.4 * (0.7 + weighted_home_form_points * 0.6) + 0.6 * home_recent_factor) * min(1.05, home_advantage))
        away_form_factor = min(1.5, (0.4 * (0.8 + weighted_away_form_points * 0.9) + 0.6 * away_recent_factor))
        
        logger.info(f"Son 5 maç analizi: Ev {recent_home_goals_avg:.2f} gol/maç, Deplasman {recent_away_goals_avg:.2f} gol/maç")
        logger.info(f"Form faktörleri: Ev {home_form_factor:.2f}, Deplasman {away_form_factor:.2f}")
        
        # Takım-spesifik ayarlamaları uygula (eğer mevcut ise)
        if use_team_specific_models:
            logger.info(f"Takım-spesifik modeller uygulanıyor: {home_team_name} vs {away_team_name}")
            original_home_goals = expected_home_goals
            original_away_goals = expected_away_goals
            expected_home_goals, expected_away_goals = self.apply_team_specific_adjustments(
                home_team_id, away_team_id, 
                home_team_name, away_team_name, 
                expected_home_goals, expected_away_goals,
                home_form, away_form,
                use_goal_trend_analysis
            )
            logger.info(f"Takım-spesifik ayarlamalar sonrası: Ev {original_home_goals:.2f}->{expected_home_goals:.2f}, " 
                       f"Deplasman {original_away_goals:.2f}->{expected_away_goals:.2f}")
        else:
            # Geleneksel büyük takım ayarlamaları (geriye dönük uyumluluk için)
            expected_home_goals, expected_away_goals = self.adjust_prediction_for_big_teams(
                home_team_name, away_team_name, expected_home_goals, expected_away_goals
            )

        # Gol dağılımları
        all_home_goals = []
        all_away_goals = []

        # Takımların gol atma olasılıklarını hesapla - ayarlanmış formül
        p_home_scores = 1 - np.exp(-(expected_home_goals * 1.0 * home_form_factor))
        p_away_scores = 1 - np.exp(-(expected_away_goals * 1.05 * away_form_factor))
        
        # Gol olasılıklarını logla
        logger.info(f"Gol atma olasılıkları: Ev={p_home_scores:.2f}, Deplasman={p_away_scores:.2f}")

        # Gelişmiş tahmin faktörlerini uygula
        global ENHANCED_FACTORS_AVAILABLE
        if 'ENHANCED_FACTORS_AVAILABLE' in globals() and globals()['ENHANCED_FACTORS_AVAILABLE'] and hasattr(self, 'enhanced_factors'):
            try:
                logger.info(f"Gelişmiş tahmin faktörleri uygulanıyor: {home_team_name} vs {away_team_name}")
                
                # Gelişmiş faktörlerden gelen ayarlamaları al
                enhanced_factors = self.enhanced_factors.get_enhanced_prediction_factors(
                    home_team_id, away_team_id, home_form, away_form
                )
                
                # Orijinal değerleri sakla
                original_home_goals = expected_home_goals
                original_away_goals = expected_away_goals
                
                # Gelişmiş faktörlere göre gol beklentilerini ayarla
                expected_home_goals, expected_away_goals = self.enhanced_factors.adjust_score_prediction(
                    expected_home_goals, expected_away_goals, enhanced_factors
                )
                
                logger.info(f"Gelişmiş faktör ayarlamaları sonrası: Ev {original_home_goals:.2f}->{expected_home_goals:.2f}, " 
                           f"Deplasman {original_away_goals:.2f}->{expected_away_goals:.2f}")
                
                # Gelişmiş faktörlerin gerekçelerini log'la
                if 'match_importance' in enhanced_factors:
                    logger.info(f"Maç önemi faktörü: {enhanced_factors['match_importance']['description']}")
                
                if 'historical_pattern' in enhanced_factors:
                    logger.info(f"Tarihsel eşleşme analizi: {enhanced_factors['historical_pattern']['description']}")
                
                if 'momentum' in enhanced_factors:
                    logger.info(f"Momentum analizi: Ev {enhanced_factors['momentum']['home_momentum']:.2f}, " 
                               f"Deplasman {enhanced_factors['momentum']['away_momentum']:.2f}")
                
            except Exception as e:
                logger.error(f"Gelişmiş tahmin faktörleri uygulanırken hata: {str(e)}")
                logger.warning("Gelişmiş faktörler uygulanamadı, standart tahmin değerleri kullanılacak")
        
        # Takım ID'lerine göre özel ayarlamalar uygula 
        # (Özellikle düşük skorlu maçlar ve spesifik takım işlemleri için)
        try:
            # Düşük skor eğilimli takımları kontrol et
            low_scoring_teams = [262, 530, 492, 642]  # Örnek Atletico Madrid, Getafe, Bursa, vs. gibi
            defensive_teams = [165, 798, 939, 250]    # Örnek savunmacı takımlar
            high_scoring_teams = [157, 173, 496, 533] # Örnek yüksek skor eğilimli takımlar
            
            is_home_low_scoring = int(home_team_id) in low_scoring_teams or int(home_team_id) in defensive_teams
            is_away_low_scoring = int(away_team_id) in low_scoring_teams or int(away_team_id) in defensive_teams
            is_home_high_scoring = int(home_team_id) in high_scoring_teams
            is_away_high_scoring = int(away_team_id) in high_scoring_teams
            
            # Her ikisi de düşük skor eğilimli ise
            if is_home_low_scoring and is_away_low_scoring:
                logger.info("Her iki takım da düşük skorlu - skor beklentileri azaltılıyor")
                expected_home_goals *= 0.85  # %15 azalt
                expected_away_goals *= 0.85  # %15 azalt
            
            # Hem düşük hem yüksek varsa, yüksek lehine ayarla
            elif (is_home_low_scoring and is_away_high_scoring):
                logger.info("Ev düşük skor eğilimli, deplasman yüksek - deplasman lehine ayarlandı")
                expected_home_goals *= 0.9   # %10 azalt
                expected_away_goals *= 1.15  # %15 artır
            
            # Yüksek skor lehine ayarla
            elif (is_home_high_scoring and is_away_low_scoring):
                logger.info("Ev yüksek skor eğilimli, deplasman düşük - ev lehine ayarlandı")
                expected_home_goals *= 1.15  # %15 artır
                expected_away_goals *= 0.9   # %10 azalt
            
            # İki yüksek skor takımı
            elif is_home_high_scoring and is_away_high_scoring:
                logger.info("Her iki takım da yüksek skorlu - gol beklentileri artırılıyor")
                expected_home_goals *= 1.2  # %20 artır
                expected_away_goals *= 1.2  # %20 artır
                
            logger.info(f"Takım ID tabanlı skor ayarlamaları sonrası: Ev={expected_home_goals:.2f}, Deplasman={expected_away_goals:.2f}")
        except Exception as e:
            logger.warning(f"Takım ID'sine göre ayarlama yapılırken hata: {str(e)}")
        
        # Beklenen toplam gol - daha dengeli toplam gol hesaplaması
        expected_total_goals = expected_home_goals * home_form_factor * 1.0 + expected_away_goals * away_form_factor * 1.05

        # Beklenen gol sayıları (initial estimations) - daha dengeli başlangıç değerleri
        avg_home_goals = expected_home_goals * (home_form_factor * 1.0)
        avg_away_goals = expected_away_goals * (away_form_factor * 1.05)
        
        # Toplam gol beklentisini logla
        logger.info(f"Toplam gol beklentisi: {expected_total_goals:.2f}, Ev={avg_home_goals:.2f}, Deplasman={avg_away_goals:.2f}")

        # Monte Carlo simülasyonu için ek değişkenler
        exact_scores = {}  # Kesin skor tahminleri için
        half_time_results = {"HOME_WIN": 0, "DRAW": 0, "AWAY_WIN": 0}  # İlk yarı sonuçları
        full_time_results = {"HOME_WIN": 0, "DRAW": 0, "AWAY_WIN": 0}  # Maç sonu sonuçları
        half_time_full_time = {}  # İlk yarı/maç sonu kombinasyonları
        first_goal_home = 0  # İlk golü ev sahibi takımın atma sayısı
        first_goal_away = 0  # İlk golü deplasman takımının atma sayısı
        no_goal = 0  # Golsüz maç sayısı

        # Kart ve korner tahminleri için
        cards_under_3_5 = 0
        cards_over_3_5 = 0
        corners_under_9_5 = 0
        corners_over_9_5 = 0

        # Gol zamanlaması için
        first_goal_timing = {
            "1-15": 0, "16-30": 0, "31-45": 0, 
            "46-60": 0, "61-75": 0, "76-90": 0, "No Goal": 0
        }

        # Monte Carlo simülasyonu
        for _ in range(simulations):
            # Negatif binomial dağılımını yaklaşık olarak simüle et
            # Poisson dağılımından daha fazla varyasyon gösterir ve gerçek gol dağılımlarını daha iyi temsil eder

            # Negatif binomial parametreleri hesapla
            # Poisson'a göre daha fazla varyasyona izin verir, özellikle yüksek skorlarda
            # r (başarı sayısı) ve p (başarı olasılığı) parametreleri ile tanımlanır

            # Düşük skorlu maçlar için özel işleme
            # Form puanlarını da dikkate alan geliştirilmiş düşük gol beklentili maç tanımı
            low_scoring_match = (avg_home_goals < 1.0 and avg_away_goals < 1.0) or \
                               (avg_home_goals < 1.1 and avg_away_goals < 1.1 and 
                                home_form and away_form and
                                home_form.get('weighted_form_points', 0) < 0.4 and 
                                away_form.get('weighted_form_points', 0) < 0.4)
            
            if low_scoring_match:
                logger.info(f"Düşük skorlu maç tespit edildi: ev={avg_home_goals:.2f}, deplasman={avg_away_goals:.2f}")
                # Düşük skorlu maçlarda standart sapmayı azalt - daha tutarlı olasılıklar için
                home_std_dev = np.sqrt(avg_home_goals * 0.9) if avg_home_goals > 0 else 0.15
                away_std_dev = np.sqrt(avg_away_goals * 0.9) if avg_away_goals > 0 else 0.15
                
                # Düşük skorlu maçlarda, 0 gol olasılığını artırmak için daha düşük r değeri kullan
                home_r = max(0.8, avg_home_goals / 0.25) if avg_home_goals > 0 else 0.8
                away_r = max(0.8, avg_away_goals / 0.25) if avg_away_goals > 0 else 0.8
            else:
                # Normal maçlar için standart sapma hesaplaması
                home_std_dev = np.sqrt(avg_home_goals * 1.2) if avg_home_goals > 0 else 0.2
                away_std_dev = np.sqrt(avg_away_goals * 1.2) if avg_away_goals > 0 else 0.2
                
                # Normal maçlar için r parametresi hesaplaması
                home_r = max(1.0, avg_home_goals / 0.2) if avg_home_goals > 0 else 1.0
                away_r = max(1.0, avg_away_goals / 0.2) if avg_away_goals > 0 else 1.0

            home_p = home_r / (home_r + avg_home_goals)
            away_p = away_r / (away_r + avg_away_goals)

            # ZEHİRLİ SAVUNMA ANALİZİ: Takımların savunma zayıflıklarını belirle
            # Son maçlardaki gol yeme oranlarına göre savunma zayıflık faktörlerini hesapla
            home_defense_weakness = 1.0  # Varsayılan değer (1.0 = normal savunma)
            away_defense_weakness = 1.0  # Varsayılan değer (1.0 = normal savunma)
            
            # Son maçlardaki savunma performansını analiz et
            home_matches_data = home_form.get('recent_match_data', [])[:8]  # Son 8 maç
            away_matches_data = away_form.get('recent_match_data', [])[:8]  # Son 8 maç
            
            if home_matches_data:
                home_conceded_total = sum(match.get('goals_conceded', 0) for match in home_matches_data)
                home_match_count = len(home_matches_data)
                home_conceded_avg = home_conceded_total / home_match_count if home_match_count > 0 else 1.0
                
                # 1.5+ gol yiyen takım zayıf savunmalı kabul edilir
                if home_conceded_avg >= 1.5:
                    home_defense_weakness = 1.0 + min(0.5, (home_conceded_avg - 1.0) * 0.25)  # En fazla 1.5 kat zayıflık
                    logger.info(f"Ev sahibi takım zayıf savunma tespiti: {home_conceded_avg:.2f} gol/maç, savunma zayıflık faktörü: {home_defense_weakness:.2f}")
            
            if away_matches_data:
                away_conceded_total = sum(match.get('goals_conceded', 0) for match in away_matches_data)
                away_match_count = len(away_matches_data)
                away_conceded_avg = away_conceded_total / away_match_count if away_match_count > 0 else 1.0
                
                # 1.5+ gol yiyen takım zayıf savunmalı kabul edilir, deplasmanın gol yemesi daha olası
                if away_conceded_avg >= 1.5:
                    away_defense_weakness = 1.0 + min(0.6, (away_conceded_avg - 1.0) * 0.3)  # En fazla 1.6 kat zayıflık
                    logger.info(f"Deplasman takımı zayıf savunma tespiti: {away_conceded_avg:.2f} gol/maç, savunma zayıflık faktörü: {away_defense_weakness:.2f}")
            
            # Savunma zayıflıklarını gol beklentilerine yansıt - üst sınır kontrolü ile
            adjusted_home_goals = avg_home_goals * away_defense_weakness  # Deplasman savunması zayıfsa ev sahibi daha fazla gol atar
            adjusted_away_goals = avg_away_goals * home_defense_weakness  # Ev sahibi savunması zayıfsa deplasman daha fazla gol atar
            
            # Gol beklentileri için üst sınır kontrolü (4.5, maksimum gerçekçi maç skoru)
            MAX_GOAL_EXPECTATION = 4.5
            if adjusted_home_goals > MAX_GOAL_EXPECTATION:
                logger.warning(f"Ev gol beklentisi çok yüksek ({adjusted_home_goals:.2f}), {MAX_GOAL_EXPECTATION} ile sınırlandı")
                adjusted_home_goals = MAX_GOAL_EXPECTATION
            
            if adjusted_away_goals > MAX_GOAL_EXPECTATION:
                logger.warning(f"Deplasman gol beklentisi çok yüksek ({adjusted_away_goals:.2f}), {MAX_GOAL_EXPECTATION} ile sınırlandı")
                adjusted_away_goals = MAX_GOAL_EXPECTATION
            
            if home_defense_weakness > 1.0 or away_defense_weakness > 1.0:
                logger.info(f"Gol beklentileri savunma zayıflıklarına göre güncellendi: "
                           f"Ev {avg_home_goals:.2f}->{adjusted_home_goals:.2f}, "
                           f"Deplasman {avg_away_goals:.2f}->{adjusted_away_goals:.2f}")
                
                # Güncellenen gol beklentilerini kullan
                avg_home_goals = adjusted_home_goals
                avg_away_goals = adjusted_away_goals
                
                # Negatif binomial parametrelerini güncelle
                home_std_dev = np.sqrt(avg_home_goals * 1.2) if avg_home_goals > 0 else 0.2
                away_std_dev = np.sqrt(avg_away_goals * 1.2) if avg_away_goals > 0 else 0.2
                
                home_r = max(1.0, avg_home_goals / 0.2) if avg_home_goals > 0 else 1.0
                away_r = max(1.0, avg_away_goals / 0.2) if avg_away_goals > 0 else 1.0
                
                home_p = home_r / (home_r + avg_home_goals)
                away_p = away_r / (away_r + avg_away_goals)
                
            # Savunma zayıflıklarını değerlendir
            # Zayıf savunmalı takımlara karşı gol beklentilerinde ve varyasyonlarda artış yaratarak
            # daha gerçekçi tahminler oluştur
            home_defense_weakness = 1.0
            away_defense_weakness = 1.0
            
            # Son 8 maçta maç başına 1.5 veya daha fazla gol yiyenler savunmada zayıf kabul edilir
            if away_form and 'recent_match_data' in away_form:
                away_matches = away_form['recent_match_data'][:8]  # Son 8 maç
                if away_matches:
                    away_conceded_total = sum(match.get('goals_conceded', 0) for match in away_matches)
                    away_match_count = len(away_matches)
                    away_conceded_avg = away_conceded_total / away_match_count if away_match_count > 0 else 1.0
                    
                    # 1.5+ gol yiyen takım zayıf savunmalı kabul edilir - ev sahibi için gol beklentisini artır
                    if away_conceded_avg >= 1.5:
                        home_defense_weakness = 1.0 + min(0.5, (away_conceded_avg - 1.0) * 0.25)
                        if home_defense_weakness > 1.05:  # Log only if significant adjustment
                            logger.info(f"Deplasman takımı zayıf savunma tespiti: {away_conceded_avg:.2f} gol/maç, ev sahibi hücum faktörü: {home_defense_weakness:.2f}")
            
            if home_form and 'recent_match_data' in home_form:
                home_matches = home_form['recent_match_data'][:8]  # Son 8 maç
                if home_matches:
                    home_conceded_total = sum(match.get('goals_conceded', 0) for match in home_matches)
                    home_match_count = len(home_matches)
                    home_conceded_avg = home_conceded_total / home_match_count if home_match_count > 0 else 1.0
                    
                    # 1.5+ gol yiyen takım zayıf savunmalı kabul edilir - deplasman için gol beklentisini artır
                    if home_conceded_avg >= 1.5:
                        away_defense_weakness = 1.0 + min(0.6, (home_conceded_avg - 1.0) * 0.3)
                        if away_defense_weakness > 1.05:  # Log only if significant adjustment
                            logger.info(f"Ev sahibi takım zayıf savunma tespiti: {home_conceded_avg:.2f} gol/maç, deplasman hücum faktörü: {away_defense_weakness:.2f}")
            
            # Dinamik ayarlanmış gol beklentileri - Zehirli savunma analizi entegrasyonu
            adjusted_home_goals = avg_home_goals * home_defense_weakness
            adjusted_away_goals = avg_away_goals * away_defense_weakness
            
            # Farklı dağılımları daha dengeli kullan
            # Daha fazla çeşitlilik için random_selector ile dağılım seç
            random_selector = np.random.random()

            # Ev sahibi skoru dağılımı - Beklenen gol değerine çok daha yakın sonuçlar üretmek için iyileştirilmiş Poisson dağılımı
            
            # Beklenen gol değerine göre maksimum makul skor sınırları belirle
            # Bu, Monte Carlo simülasyonunda aşırı değerlerin oluşmasını önler
            max_home_score = 1  # Varsayılan makul maksimum değer (düşük beklenen goller için)
            
            # Toplam beklenen gol sayısını hesapla
            all_goals_expected = adjusted_home_goals + adjusted_away_goals
            
            # Eğer özelleştirilmiş model parametreleri varsa, onları kullan
            if specialized_params:
                # Özelleştirilmiş modelden maksimum skor sınırını al
                max_home_score = specialized_params.get('max_score', 3)
                logger.debug(f"Özelleştirilmiş model maksimum ev sahibi skoru: {max_home_score}")
            else:
                # Standart maksimum skor hesaplama - düşük skorlu maçlar için ayarlanmış
                if adjusted_home_goals < 0.6:
                    # Çok düşük gol beklentisi (0.6'dan az) - daha yüksek oranda 0 gol
                    max_home_score = 1  # 0.6'dan düşük beklenen gol için maksimum 1 gol
                    # Düşük beklentilerde 0 gol olasılığını artırmak için Poisson yaklaşımı
                    if low_scoring_match and adjusted_home_goals < 0.4:
                        # KG YOK olasılığını artırmak için düşük skorlu maçlarda azaltılmış beklenti
                        adjusted_home_goals = adjusted_home_goals * 0.80  # 0.85 -> 0.80: daha yüksek sıfır gol olasılığı
                elif adjusted_home_goals < 1.0:
                    max_home_score = 1  # 0.6-1.0 arası beklenen gol için maksimum 1 gol
                    # İki takımın da beklenen golleri 1'in altındaysa daha gerçekçi skorlar üret
                    if low_scoring_match and all_goals_expected < 2.0:
                        adjusted_home_goals = adjusted_home_goals * 0.90  # Daha gerçekçi 0-0, 1-0, 0-1 skorları için
                elif adjusted_home_goals < 1.8: 
                    max_home_score = 2  # 1.0-1.8 arası beklenen gol için maksimum 2 gol
                elif adjusted_home_goals < 2.5:
                    max_home_score = 3  # 1.8-2.5 arası beklenen gol için maksimum 3 gol
                else:
                    max_home_score = 4  # 2.5'ten yüksek beklenen gol için maksimum 4 gol
            
            # %95 şans ile sınırlı Poisson dağılımı
            if random_selector < 0.95:
                # Gol beklentisi 0.3'ten az ise gol olasılığını biraz artır - minimum skor ihtimali için
                if adjusted_home_goals < 0.3:
                    raw_score = np.random.poisson(max(0.3, adjusted_home_goals))
                else:
                    raw_score = np.random.poisson(adjusted_home_goals)
                
                # Sonucu makul sınırlar içinde tut
                home_score = min(raw_score, max_home_score)
            else:
                # Çok nadir durumlarda (%5) hafif varyasyon için negatif binomial dağılımı kullan
                try:
                    # Varyasyonu daha da azalt - daha tutarlı sonuçlar için
                    home_std_dev = np.sqrt(adjusted_home_goals)  # Daha düşük varyasyon
                    home_r = max(1.0, adjusted_home_goals / 0.1)  # Daha düşük dispersiyon
                    home_p = home_r / (home_r + adjusted_home_goals)
                    
                    raw_score = np.random.negative_binomial(home_r, home_p)
                    # Makul sınırlar içinde tut
                    home_score = min(raw_score, max_home_score)
                except ValueError:
                    # Hata durumunda Poisson'a geri dön
                    raw_score = np.random.poisson(adjusted_home_goals)
                    home_score = min(raw_score, max_home_score)

            # Deplasman skoru dağılımı - Beklenen gol değerine çok daha yakın sonuçlar üretmek için iyileştirilmiş Poisson dağılımı
            
            # Beklenen gol değerine göre maksimum makul skor sınırları belirle
            # Bu, Monte Carlo simülasyonunda aşırı değerlerin oluşmasını önler
            max_away_score = 1  # Varsayılan makul maksimum değer (düşük beklenen goller için)
            
            # Eğer özelleştirilmiş model parametreleri varsa, onları kullan
            if specialized_params:
                # Özelleştirilmiş modelden maksimum skor sınırını al
                max_away_score = specialized_params.get('max_score', 3)
                logger.debug(f"Özelleştirilmiş model maksimum deplasman skoru: {max_away_score}")
            else:
                # Standart maksimum skor hesaplama - düşük skorlu maçlar için ayarlanmış
                if adjusted_away_goals < 0.6:
                    # Çok düşük gol beklentisi (0.6'dan az) - daha yüksek oranda 0 gol
                    max_away_score = 1  # 0.6'dan düşük beklenen gol için maksimum 1 gol
                    # Düşük beklentilerde 0 gol olasılığını artırmak için Poisson yaklaşımı
                    if low_scoring_match and adjusted_away_goals < 0.4:
                        # KG YOK olasılığını artırmak için düşük skorlu maçlarda azaltılmış beklenti
                        adjusted_away_goals = adjusted_away_goals * 0.80  # 0.85 -> 0.80: daha yüksek sıfır gol olasılığı
                elif adjusted_away_goals < 1.0:
                    max_away_score = 1  # 0.6-1.0 arası beklenen gol için maksimum 1 gol
                    # İki takımın da beklenen golleri 1'in altındaysa daha gerçekçi skorlar üret
                    if low_scoring_match and all_goals_expected < 2.0:
                        adjusted_away_goals = adjusted_away_goals * 0.90  # Daha gerçekçi 0-0, 1-0, 0-1 skorları için
                elif adjusted_away_goals < 1.8: 
                    max_away_score = 2  # 1.0-1.8 arası beklenen gol için maksimum 2 gol
                elif adjusted_away_goals < 2.5:
                    max_away_score = 3  # 1.8-2.5 arası beklenen gol için maksimum 3 gol
                else:
                    max_away_score = 4  # 2.5'ten yüksek beklenen gol için maksimum 4 gol
            
            # %95 şans ile sınırlı Poisson dağılımı
            if random_selector < 0.95:
                # Gol beklentisi 0.3'ten az ise gol olasılığını biraz artır - minimum skor ihtimali için
                if adjusted_away_goals < 0.3:
                    raw_score = np.random.poisson(max(0.3, adjusted_away_goals))
                else:
                    raw_score = np.random.poisson(adjusted_away_goals)
                
                # Sonucu makul sınırlar içinde tut
                away_score = min(raw_score, max_away_score)
            else:
                # Çok nadir durumlarda (%5) hafif varyasyon için negatif binomial dağılımı kullan
                try:
                    # Varyasyonu daha da azalt - daha tutarlı sonuçlar için
                    away_std_dev = np.sqrt(adjusted_away_goals)  # Daha düşük varyasyon
                    away_r = max(1.0, adjusted_away_goals / 0.1)  # Daha düşük dispersiyon
                    away_p = away_r / (away_r + adjusted_away_goals)
                    
                    raw_score = np.random.negative_binomial(away_r, away_p)
                    # Makul sınırlar içinde tut
                    away_score = min(raw_score, max_away_score)
                except ValueError:
                    # Hata durumunda Poisson'a geri dön
                    raw_score = np.random.poisson(adjusted_away_goals)
                    away_score = min(raw_score, max_away_score)

            all_home_goals.append(home_score)
            all_away_goals.append(away_score)

            # Kesin skor tahmini
            exact_score_key = f"{home_score}-{away_score}"
            exact_scores[exact_score_key] = exact_scores.get(exact_score_key, 0) + 1

            # Maç sonucu
            if home_score > away_score:
                home_wins += 1
                full_time_results["HOME_WIN"] += 1
            elif home_score < away_score:
                away_wins += 1
                full_time_results["AWAY_WIN"] += 1
            else:
                draws += 1
                full_time_results["DRAW"] += 1

            # İlk yarı simülasyonu - ilk yarı gollerinin yaklaşık %40'ı atılır
            first_half_home_mean = avg_home_goals * 0.4
            first_half_away_mean = avg_away_goals * 0.4

            first_half_home = np.random.poisson(first_half_home_mean)
            first_half_away = np.random.poisson(first_half_away_mean)

            # İlk yarı sonucu
            if first_half_home > first_half_away:
                half_time_results["HOME_WIN"] += 1
                half_time_key = "HOME_WIN"
            elif first_half_home < first_half_away:
                half_time_results["AWAY_WIN"] += 1
                half_time_key = "AWAY_WIN"
            else:
                half_time_results["DRAW"] += 1
                half_time_key = "DRAW"

            # Maç sonu sonucu
            if home_score > away_score:
                full_time_key = "HOME_WIN"
            elif home_score < away_score:
                full_time_key = "AWAY_WIN"
            else:
                full_time_key = "DRAW"

            # İlk yarı/maç sonu kombinasyonu
            ht_ft_key = f"{half_time_key}/{full_time_key}"
            half_time_full_time[ht_ft_key] = half_time_full_time.get(ht_ft_key, 0) + 1

            # İlk golü kim attı
            total_goals = home_score + away_score
            if total_goals == 0:
                no_goal += 1
            else:
                # İlk golü atma olasılığı hesapla
                p_home_first = avg_home_goals / (avg_home_goals + avg_away_goals) if (avg_home_goals + avg_away_goals) > 0 else 0.5

                if np.random.random() < p_home_first and home_score > 0:
                    first_goal_home += 1
                elif away_score > 0:
                    first_goal_away += 1

            # Gol zamanlaması simülasyonu
            if total_goals == 0:
                first_goal_timing["No Goal"] += 1
            else:
                # Gol zamanlamasını simüle et - genellikle ikinci yarıda daha fazla gol olur
                timing_weights = [0.15, 0.15, 0.15, 0.17, 0.18, 0.20]  # Zamanlamalar için ağırlıklar
                timing_ranges = ["1-15", "16-30", "31-45", "46-60", "61-75", "76-90"]

                first_goal_timing[np.random.choice(timing_ranges, p=[w/sum(timing_weights) for w in timing_weights])] += 1

            # İki takım da gol attı mı
            if home_score > 0 and away_score > 0:
                both_teams_scored += 1

            # Toplam gol sayısı 2.5'tan fazla mı
            total_goals = home_score + away_score
            if total_goals > 2.5:
                over_2_5_goals += 1

            # Toplam gol sayısı 3.5'tan fazla mı
            if total_goals > 3.5:
                over_3_5_goals += 1

            # Kart sayısı simülasyonu
            # Kart sayısı maçın gerginliğine ve gol farkına bağlıdır
            tension_factor = 1.0
            if abs(home_score - away_score) <= 1:  # Yakın maçlarda daha fazla kart
                tension_factor = 1.3
            elif total_goals > 3:  # Çok gollü maçlarda genelde daha az kart
                tension_factor = 0.9

            # Ortalama kart sayısı yaklaşık 3.5
            avg_cards = 3.5 * tension_factor
            cards = np.random.poisson(avg_cards)

            if cards <= 3.5:
                cards_under_3_5 += 1
            else:
                cards_over_3_5 += 1

            # Korner sayısı simülasyonu
            # Korner sayısı takımların hücum gücüne bağlıdır
            attack_factor = (avg_home_goals + avg_away_goals) / 2.5  # Lig ortalamasına göre normalizasyon
            # Korner sayısı için üst sınır kontrolü
            avg_corners = min(15.0, 10 * attack_factor)  # En fazla 15 ortalama korner
            corners = np.random.poisson(avg_corners)

            if corners <= 9.5:
                corners_under_9_5 += 1
            else:
                corners_over_9_5 += 1

        # Olasılıkları hesapla
        home_win_prob = home_wins / simulations
        away_win_prob = away_wins / simulations
        draw_prob = draws / simulations
        
        # Özelleştirilmiş model beraberlik çarpanını uygula
        if specialized_params:
            # Beraberlik çarpanını al ve uygula
            draw_boost = specialized_params.get('draw_boost', 1.0)
            if draw_boost != 1.0:
                original_draw_prob = draw_prob
                # Beraberlik olasılığını çarpana göre ayarla (maksimum 0.95)
                draw_prob = min(0.95, draw_prob * draw_boost)
                
                # Diğer olasılıkları (homewin ve awaywin) azalt ve normalizasyon yap
                if draw_prob > original_draw_prob:
                    # Toplam galibiyet olasılığı
                    total_win_prob = home_win_prob + away_win_prob
                    if total_win_prob > 0:
                        # Beraberlik için alan açmak üzere galibiyet olasılıklarını azalt
                        reduction_factor = (1 - draw_prob) / total_win_prob
                        home_win_prob *= reduction_factor
                        away_win_prob *= reduction_factor
                    
                    logger.info(f"Özelleştirilmiş model beraberlik düzeltmesi: {original_draw_prob:.2f} -> {draw_prob:.2f} (çarpan: {draw_boost:.2f})")
        
        # Olasılıklar eşit paylaşılmış mı diye kontrol et (33-34-33 gibi)
        # Eğer öyleyse, gol beklentilerine göre yeniden hesapla
        if abs(home_win_prob - 0.33) < 0.03 and abs(away_win_prob - 0.33) < 0.03 and abs(draw_prob - 0.34) < 0.03:
            logger.warning("Olasılıklar çok dengeli dağılmış (varsayılan değerler), form verilerine göre ayarlanıyor!")
            # Monte Carlo dışı alternatif hesaplama - doğrudan gol beklentilerini kullan
            exp_total = avg_home_goals + avg_away_goals
            
            # Dixon-Coles benzeri hesaplama modeli
            # Poisson olasılıklarını hesapla
            p_home_win = 0.0
            p_draw = 0.0
            p_away_win = 0.0
            
            max_goals = 5  # Hesaplama için maksimum gol sayısı
            
            for h in range(max_goals+1):
                home_poisson = np.exp(-avg_home_goals) * (avg_home_goals**h) / np.math.factorial(h)
                
                for a in range(max_goals+1):
                    away_poisson = np.exp(-avg_away_goals) * (avg_away_goals**a) / np.math.factorial(a)
                    
                    # Düşük skorlu maçlar için tau düzeltmesi (Dixon-Coles)
                    if h <= 1 and a <= 1:
                        correction = 1.0
                        if h == 0 and a == 0:
                            correction = 1.2  # 0-0 skoru için artış
                        elif h == 1 and a == 1:
                            correction = 1.1  # 1-1 skoru için artış
                        joint_prob = home_poisson * away_poisson * correction
                    else:
                        joint_prob = home_poisson * away_poisson
                    
                    # Sonucu hesapla
                    if h > a:
                        p_home_win += joint_prob
                    elif h == a:
                        p_draw += joint_prob
                    else:
                        p_away_win += joint_prob
            
            # Olasılıkları normalize et
            total_prob = p_home_win + p_draw + p_away_win
            if total_prob > 0:
                p_home_win /= total_prob
                p_draw /= total_prob
                p_away_win /= total_prob
                
                # Monte Carlo sonuçlarıyla harmanla - Monte Carlo %40, Dixon-Coles %60 ağırlık
                home_win_prob = home_win_prob * 0.4 + p_home_win * 0.6
                draw_prob = draw_prob * 0.4 + p_draw * 0.6
                away_win_prob = away_win_prob * 0.4 + p_away_win * 0.6
                
                logger.info(f"Olasılıklar yeniden hesaplandı: Ev={home_win_prob:.2f}, Beraberlik={draw_prob:.2f}, Deplasman={away_win_prob:.2f}")
        
        # Diğer bahis olasılıkları
        both_teams_scored_prob = both_teams_scored / simulations
        over_2_5_goals_prob = over_2_5_goals / simulations
        over_3_5_goals_prob = over_3_5_goals / simulations
        
        # Düşük skorlu maçlar için özel KG YOK düzeltmesi (iyileştirilmiş 0-0, 1-0, 0-1 olasılıkları)
        # Her iki takımın da gol beklentisi düşükse KG VAR olasılığını azalt
        if avg_home_goals < 1.0 and avg_away_goals < 1.0:
            logger.info(f"Düşük skorlu maç tespit edildi. KG VAR düzeltmesi öncesi: %{both_teams_scored_prob*100:.2f}")
            # Çok düşük gol beklentisi (toplam 1.5 altı) - KG YOK olasılığını büyük ölçüde artır
            if avg_home_goals + avg_away_goals < 1.5:
                both_teams_scored_prob = both_teams_scored_prob * 0.65  # KG VAR olasılığını %35 azalt
                logger.info(f"Çok düşük gol beklentili maç: KG VAR olasılığı %{both_teams_scored_prob*100:.2f}'e düşürüldü")
                
                # Düşük skorda hangi takımın kazanma olasılığının daha yüksek olduğunu form ve h2h verilerine göre belirle
                home_advantage_factor = self._calculate_low_scoring_advantage(home_form, away_form, home_team_id, away_team_id)
                
                # 0-0 yerine 1-0 veya 0-1 skoru için kalıtım mekanizması
                if home_advantage_factor > 0.2:  # Ev sahibi avantajlı
                    # 1-0 skorunun olasılığını artır
                    score_key_10 = "1-0"
                    exact_scores[score_key_10] = exact_scores.get(score_key_10, 0) + int(simulations * 0.08)
                    logger.info(f"Form ve H2H avantajı ev sahibinde: 1-0 skoru olasılığı artırıldı")
                elif home_advantage_factor < -0.2:  # Deplasman avantajlı
                    # 0-1 skorunun olasılığını artır
                    score_key_01 = "0-1"
                    exact_scores[score_key_01] = exact_scores.get(score_key_01, 0) + int(simulations * 0.08)
                    logger.info(f"Form ve H2H avantajı deplasmanda: 0-1 skoru olasılığı artırıldı")
                else:
                    # Her iki takımın da şansı denk, 0-0 skorunun olasılığını artır
                    score_key_00 = "0-0"
                    exact_scores[score_key_00] = exact_scores.get(score_key_00, 0) + int(simulations * 0.05)
                    logger.info(f"İki takım da denk güçte: 0-0 skoru olasılığı artırıldı")
                
            elif avg_home_goals + avg_away_goals < 1.8:
                both_teams_scored_prob = both_teams_scored_prob * 0.75  # KG VAR olasılığını %25 azalt
                logger.info(f"Düşük gol beklentili maç: KG VAR olasılığı %{both_teams_scored_prob*100:.2f}'e düşürüldü")
            else:
                both_teams_scored_prob = both_teams_scored_prob * 0.85  # KG VAR olasılığını %15 azalt
                logger.info(f"Orta-düşük gol beklentili maç: KG VAR olasılığı %{both_teams_scored_prob*100:.2f}'e düşürüldü")

        # Gelişmiş tahminler için olasılıklar
        cards_over_3_5_prob = cards_over_3_5 / simulations
        corners_over_9_5_prob = corners_over_9_5 / simulations

        # Beraberlik olasılığını yükseltme - kesin skor dağılımına göre ayarlama
        # Hesaplanan en olası kesin skor X-X formunda ise (berabere) beraberlik olasılığını arttır
        top_exact_scores = sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Özelleştirilmiş modelden beraberlik çarpanını al
        draw_boost = 1.0  # Varsayılan çarpan
        if specialized_params:
            draw_boost = specialized_params.get('draw_boost', 1.0)
            # Eğer özel bir beraberlik çarpanı varsa, direkt olarak uygula
            if draw_boost != 1.0:
                original_draw_prob = draw_prob
                # Beraberlik olasılığını artır ancak üst limiti 0.95 olarak belirle
                draw_prob = min(0.95, draw_prob * draw_boost)
                
                # Diğer olasılıkları azalt ve normalizasyon yap
                if draw_prob > original_draw_prob:
                    total_win_prob = home_win_prob + away_win_prob
                    if total_win_prob > 0:
                        reduction_factor = (1 - draw_prob) / total_win_prob
                        home_win_prob *= reduction_factor
                        away_win_prob *= reduction_factor
                    logger.info(f"Özelleştirilmiş model beraberlik düzeltmesi: {original_draw_prob:.2f} -> {draw_prob:.2f} (çarpan: {draw_boost:.2f})")

        # Kesin skor bazlı beraberlik ayarlaması
        for score, count in top_exact_scores:
            if '-' in score:
                home_score, away_score = map(int, score.split('-'))
                if home_score == away_score:  # Berabere skor
                    # Skor berabere ve ilk 3 olası skor içindeyse beraberlik olasılığını yükselt
                    score_prob = count / simulations
                    if score_prob > 0.05:  # %5'ten fazla olasılıkla gözüken beraberlik skoru
                        # Beraberlik olasılığını artır - skor olasılığına göre ağırlıklandır
                        adjustment = min(0.25, score_prob * 2)  # Max %25 artış
                        draw_prob = min(0.95, draw_prob * (1 + adjustment))
                        # Diğer olasılıkları azalt ve yeniden normalize et
                        total_win_prob = home_win_prob + away_win_prob
                        if total_win_prob > 0:
                            reduction_factor = (1 - draw_prob) / total_win_prob
                            home_win_prob *= reduction_factor
                            away_win_prob *= reduction_factor
                        logger.info(f"Skor bazlı düzeltme: {score} skoru için beraberlik olasılığı artırıldı")

        # En olası kesin skor - skorları çeşitlendirme
        # En yüksek olasılıklı 3 skoru al ve bunlardan birini seç
        top_3_scores = sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)[:3]

        # KG Var/Yok tahmini - model doğrulama sonuçlarına göre KG Var/Yok (0.833) 2.5 Üst/Alt'tan (0.786) daha doğru
        # Bu nedenle KG Var/Yok tahminini kesin skor tahminini belirlemek için de kullanacağız
        # p_both_teams_score hesaplanması
        p_home_scores = 1 - np.exp(-expected_home_goals)
        p_away_scores = 1 - np.exp(-expected_away_goals)
        p_both_teams_score = p_home_scores * p_away_scores  # İki takımın da gol atma olasılığı
        
        # KG VAR olasılığını gol beklentilerine göre ayarla
        kg_var_adjusted_prob = p_both_teams_score  # KG Var olasılığı (başlangıç değeri)

        # Geliştirilmiş KG VAR/YOK kuralları
        # 1. İki takımın da gol beklentisi 1.0 üzerindeyse kesinlikle KG VAR diyoruz
        # 2. Bir takım 1.0'ın üzerinde, diğeri 0.75'in üzerindeyse yine KG VAR diyoruz
        if expected_home_goals > 1.0 and expected_away_goals > 1.0:
            kg_var_adjusted_prob = max(kg_var_adjusted_prob, 0.9)  # En az %90 KG VAR olasılığı
            logger.info(f"İki takımın da gol beklentisi 1.0'dan fazla: KG VAR olasılığı %{kg_var_adjusted_prob*100:.2f}'e yükseltildi")
        elif (expected_home_goals > 1.0 and expected_away_goals > 0.75) or (expected_away_goals > 1.0 and expected_home_goals > 0.75):
            kg_var_adjusted_prob = max(kg_var_adjusted_prob, 0.8)  # En az %80 KG VAR olasılığı
            logger.info(f"Bir takımın gol beklentisi 1.0'dan fazla, diğeri 0.75'ten fazla: KG VAR olasılığı %{kg_var_adjusted_prob*100:.2f}'e yükseltildi")
        
        # Düşük gol beklentisi durumunda KG YOK olasılığını yükselt
        if expected_home_goals < 0.5 or expected_away_goals < 0.5:
            kg_var_adjusted_prob = min(kg_var_adjusted_prob, 0.25)  # En fazla %25 KG VAR olasılığı
            logger.info(f"Bir takımın gol beklentisi 0.5'ten az: KG VAR olasılığı %{kg_var_adjusted_prob*100:.2f}'e düşürüldü")
        
        # Son 5 maçtaki gol istatistiklerini de dikkate al
        home_recent_scored = sum(match.get('goals_scored', 0) for match in home_match_data[:5] if 'goals_scored' in match)
        home_recent_conceded = sum(match.get('goals_conceded', 0) for match in home_match_data[:5] if 'goals_conceded' in match)
        away_recent_scored = sum(match.get('goals_scored', 0) for match in away_match_data[:5] if 'goals_scored' in match)
        away_recent_conceded = sum(match.get('goals_conceded', 0) for match in away_match_data[:5] if 'goals_conceded' in match)
        
        # İki takım da son 5 maçta ortalama 1+ gol attıysa, KG VAR olasılığını yükselt
        if home_recent_scored >= 5 and away_recent_scored >= 5:
            kg_var_adjusted_prob = max(kg_var_adjusted_prob, 0.75)
            logger.info(f"İki takım da son 5 maçta ortalama 1+ gol attı: KG VAR olasılığı %{kg_var_adjusted_prob*100:.2f}'e yükseltildi")
        
        # Eğer takımlar neredeyse hiç gol yemiyorsa KG VAR olasılığını düşür
        if home_recent_conceded <= 1 or away_recent_conceded <= 1:
            kg_var_adjusted_prob = min(kg_var_adjusted_prob, 0.35)
            logger.info(f"Bir takım son 5 maçta neredeyse hiç gol yemedi: KG VAR olasılığı %{kg_var_adjusted_prob*100:.2f}'e düşürüldü")
        
        # İlk önce KG Var/Yok tahmini yap - eşik değeri 0.5 (yüzde 50)
        kg_var_prediction = kg_var_adjusted_prob >= 0.5  # True = KG VAR, False = KG YOK
        
        # Skorlar içinde en çok KG VAR ve KG YOK olan skorları bul
        kg_var_scores = [(score, count) for score, count in exact_scores.items() if int(score.split('-')[0]) > 0 and int(score.split('-')[1]) > 0]
        kg_yok_scores = [(score, count) for score, count in exact_scores.items() if int(score.split('-')[0]) == 0 or int(score.split('-')[1]) == 0]
        
        # KG VAR/YOK skorlarının toplam olasılıkları
        kg_var_total_prob = sum(count for _, count in kg_var_scores) / simulations if kg_var_scores else 0
        kg_yok_total_prob = sum(count for _, count in kg_yok_scores) / simulations if kg_yok_scores else 0
        
        # Simülasyon sonuçları ile KG VAR/YOK tahmini tutarlı değilse kaydet
        if (kg_var_prediction and kg_var_total_prob < 0.5) or (not kg_var_prediction and kg_yok_total_prob < 0.5):
            logger.warning(f"Uyarı: KG VAR/YOK tahmini ({kg_var_prediction}) simülasyon sonuçlarıyla tutarsız! KG VAR olasılığı: %{kg_var_total_prob*100:.2f}, KG YOK olasılığı: %{kg_yok_total_prob*100:.2f}")
        
        # KG Var/Yok tahminine göre en olası skoru belirle
        if kg_var_prediction:  # KG VAR tahmini yapıldıysa
            # Her iki takımın da gol attığı skorlar arasından en olası olanı seç
            if kg_var_scores:
                most_likely_score = max(kg_var_scores, key=lambda x: x[1])  # KG VAR skorları içinde en olası olanı
                logger.info(f"KG VAR tahmini nedeniyle kesin skor {most_likely_score[0]} olarak güncellendi")
                # Tahmin edilen skordan takımların gol sayılarını al
                home_score = int(most_likely_score[0].split('-')[0])
                away_score = int(most_likely_score[0].split('-')[1])
                # Eğer her iki takım da gol atmıyorsa düzelt (bu bir tutarsızlık olurdu)
                if home_score == 0 or away_score == 0:
                    logger.warning(f"KG VAR ile tutarsız skor bulundu: {most_likely_score[0]}, düzeltiliyor...")
                    if home_score == 0:
                        home_score = 1
                    if away_score == 0:
                        away_score = 1
                    most_likely_score = (f"{home_score}-{away_score}", most_likely_score[1])
                    logger.info(f"KG VAR tutarlılığı için skor {most_likely_score[0]} olarak güncellendi")
            else:
                # KG VAR tahmini yapıldı ama uygun skor bulunamadı (teorik olarak mümkün değil ama önlem amaçlı)
                logger.warning(f"KG VAR tahmini yapıldı ama hiçbir KG VAR skoru bulunamadı! Varsayılan 1-1 kullanılıyor.")
                most_likely_score = ('1-1', 1)  # Varsayılan 1-1 skoru
        else:  # KG YOK tahmini yapıldıysa
            # En az bir takımın gol atmadığı skorlar arasından en olası olanı seç
            if kg_yok_scores:
                most_likely_score = max(kg_yok_scores, key=lambda x: x[1])  # KG YOK skorları içinde en olası olanı
                
                # Tahmin edilen skordan takımların gol sayılarını al
                home_score = int(most_likely_score[0].split('-')[0])
                away_score = int(most_likely_score[0].split('-')[1])
                
                # KG YOK tutarlılığı kontrolü - her iki takım da gol atıyorsa düzelt
                if home_score > 0 and away_score > 0:
                    logger.warning(f"KG YOK ile tutarsız skor bulundu: {most_likely_score[0]}, düzeltiliyor...")
                    
                    # Beklenen gollere göre hangi takımın skor kaydetme olasılığının daha yüksek olduğunu belirle
                    if expected_home_goals > expected_away_goals + 0.3:
                        # Ev sahibi takımın gol beklentisi daha yüksek, 0'lı skorlarda ev sahibini seç
                        home_win_kg_yok = [(score, count) for score, count in kg_yok_scores 
                                         if int(score.split('-')[0]) > 0 and int(score.split('-')[1]) == 0]
                        if home_win_kg_yok:
                            most_likely_score = max(home_win_kg_yok, key=lambda x: x[1])
                            logger.info(f"KG YOK + Ev sahibi üstünlüğü nedeniyle kesin skor {most_likely_score[0]} olarak güncellendi")
                        else:
                            most_likely_score = ('1-0', 1)  # Tutarlılık için varsayılan değer
                            logger.warning("KG YOK + Ev sahibi üstünlüğü için uygun skor bulunamadı, varsayılan 1-0 kullanılıyor")
                    elif expected_away_goals > expected_home_goals + 0.3:
                        # Deplasman takımının gol beklentisi daha yüksek, 0'lı skorlarda deplasmanı seç
                        away_win_kg_yok = [(score, count) for score, count in kg_yok_scores 
                                         if int(score.split('-')[0]) == 0 and int(score.split('-')[1]) > 0]
                        if away_win_kg_yok:
                            most_likely_score = max(away_win_kg_yok, key=lambda x: x[1])
                            logger.info(f"KG YOK + Deplasman üstünlüğü nedeniyle kesin skor {most_likely_score[0]} olarak güncellendi")
                        else:
                            most_likely_score = ('0-1', 1)  # Tutarlılık için varsayılan değer
                            logger.warning("KG YOK + Deplasman üstünlüğü için uygun skor bulunamadı, varsayılan 0-1 kullanılıyor")
                    else:
                        # Takımlar denk, olası 0-0
                        zero_zero_scores = [(score, count) for score, count in kg_yok_scores if score == '0-0']
                        if zero_zero_scores:
                            most_likely_score = zero_zero_scores[0]
                            logger.info(f"KG YOK + dengeli maç nedeniyle kesin skor 0-0 olarak güncellendi")
                        else:
                            # Eğer 0-0 yoksa, en olası KG YOK skorunu seç
                            if kg_yok_scores:
                                most_likely_score = max(kg_yok_scores, key=lambda x: x[1])
                            else:
                                most_likely_score = ('0-0', 1)
                                logger.warning("KG YOK için uygun skor bulunamadı, varsayılan 0-0 kullanılıyor")
                elif expected_away_goals > expected_home_goals + 0.3:
                    # Deplasman takımının gol beklentisi daha yüksek, 0'lı skorlarda deplasmanı seç
                    away_win_kg_yok = [(score, count) for score, count in kg_yok_scores 
                                     if int(score.split('-')[0]) == 0 and int(score.split('-')[1]) > 0]
                    if away_win_kg_yok:
                        most_likely_score = max(away_win_kg_yok, key=lambda x: x[1])
                        logger.info(f"KG YOK + Deplasman üstünlüğü nedeniyle kesin skor {most_likely_score[0]} olarak güncellendi")
                    else:
                        most_likely_score = max(kg_yok_scores, key=lambda x: x[1])
                else:
                    # Takımlar denk, en olası KG YOK skorunu seç
                    most_likely_score = max(kg_yok_scores, key=lambda x: x[1])
                
                logger.info(f"KG YOK tahmini nedeniyle kesin skor {most_likely_score[0]} olarak güncellendi")
            else:
                # KG YOK tahmini yapıldı ama hiç KG YOK skoru bulunamadı (teorik olarak mümkün değil)
                logger.warning(f"KG YOK tahmini yapıldı ama hiçbir KG YOK skoru bulunamadı! Varsayılan 0-0 veya 1-0 kullanılıyor.")
                if expected_home_goals > expected_away_goals:
                    most_likely_score = ('1-0', 1)  # Varsayılan 1-0 skoru
                else:
                    most_likely_score = ('0-0', 1)  # Varsayılan 0-0 skoru

        most_likely_score_prob = most_likely_score[1] / simulations

        # İlk yarı/maç sonu en olası kombinasyon
        most_likely_ht_ft = max(half_time_full_time.items(), key=lambda x: x[1]) if half_time_full_time else ("DRAW/DRAW", 0)
        most_likely_ht_ft_prob = most_likely_ht_ft[1] / simulations if half_time_full_time else 0

        # İlk golün zamanlaması
        most_likely_first_goal_time = max(first_goal_timing.items(), key=lambda x: x[1])
        most_likely_first_goal_time_prob = most_likely_first_goal_time[1] / simulations

        # İlk golü atan takım
        first_goal_home_prob = first_goal_home / simulations if (first_goal_home + first_goal_away + no_goal) > 0 else 0
        first_goal_away_prob = first_goal_away / simulations if (first_goal_home + first_goal_away + no_goal) > 0 else 0
        no_goal_prob = no_goal / simulations

        # Beklenen gol sayıları (final estimations) - form faktörünün etkisini daha da azalt
        # avg_home_goals ve avg_away_goals zaten daha önce tanımlandı, burada sadece güncelleniyor
        avg_home_goals = expected_home_goals * (home_form_factor * 0.85)
        avg_away_goals = expected_away_goals * (away_form_factor * 0.85)

        # Aşırı yüksek tahminleri düzeltmek için gelişmiş yöntemler

        # Son 3, 5 ve 10 maçın gerçek gol ortalamalarını hesapla
        home_recent_avg_goals = {}
        away_recent_avg_goals = {}
        periods = [3, 5, 10]

        for period in periods:
            # Ev sahibi için
            home_matches_count = min(period, len(home_match_data))
            if home_matches_count > 0:
                home_recent_avg_goals[period] = sum(match.get('goals_scored', 0) for match in home_match_data[:home_matches_count]) / home_matches_count
            else:
                home_recent_avg_goals[period] = self.lig_ortalamasi_ev_gol

            # Deplasman için
            away_matches_count = min(period, len(away_match_data))
            if away_matches_count > 0:
                away_recent_avg_goals[period] = sum(match.get('goals_scored', 0) for match in away_match_data[:away_matches_count]) / away_matches_count
            else:
                away_recent_avg_goals[period] = self.lig_ortalamasi_deplasman_gol

        # Son maçların ortalaması ile genel lig ortalamasını karşılaştır
        home_avg_deviation = (home_recent_avg_goals[3] / self.lig_ortalamasi_ev_gol) * 0.5 + \
                            (home_recent_avg_goals[5] / self.lig_ortalamasi_ev_gol) * 0.3 + \
                            (home_recent_avg_goals[10] / self.lig_ortalamasi_ev_gol) * 0.2

        away_avg_deviation = (away_recent_avg_goals[3] / self.lig_ortalamasi_deplasman_gol) * 0.5 + \
                            (away_recent_avg_goals[5] / self.lig_ortalamasi_deplasman_gol) * 0.3 + \
                            (away_recent_avg_goals[10] / self.lig_ortalamasi_deplasman_gol) * 0.2

        # Sapma değerini sınırla (çok aşırı değerleri engelle)
        home_avg_deviation = min(1.5, max(0.7, home_avg_deviation))
        away_avg_deviation = min(1.5, max(0.7, away_avg_deviation))

        # Z-skor bazlı normalizasyon için takım gol dağılımlarını hesapla
        # Standart sapma hesapla (son 10 maçta)
        home_std_dev = np.std([match.get('goals_scored', 0) for match in home_match_data[:10]]) if len(home_match_data) >= 10 else 1.0
        away_std_dev = np.std([match.get('goals_scored', 0) for match in away_match_data[:10]]) if len(away_match_data) >= 10 else 0.8

        # Savunma gücü değerlendirmesi - rakip takımın savunma istatistikleri
        home_defense_strength = away_form.get('home_performance', {}).get('weighted_avg_goals_conceded', weighted_away_conceded)
        away_defense_strength = home_form.get('away_performance', {}).get('weighted_avg_goals_conceded', weighted_home_conceded)

        # Savunma gücünü lig ortalamasıyla karşılaştır
        home_defense_factor = home_defense_strength / self.lig_ortalamasi_deplasman_gol
        away_defense_factor = away_defense_strength / self.lig_ortalamasi_ev_gol

        # Gol tahminlerini sapma oranı ile düzelt
        avg_home_goals = avg_home_goals * home_avg_deviation * (1.0 + 0.2 * (1.0 - min(1.5, away_defense_factor)))
        avg_away_goals = avg_away_goals * away_avg_deviation * (1.0 + 0.2 * (1.0 - min(1.5, home_defense_factor)))

        # Limit fonksiyonu - logaritmik düzeltme
        def limit_high_values(value, threshold, scaling_factor=0.3):
            if value <= threshold:
                return value
            else:
                return threshold + scaling_factor * np.log1p(value - threshold)

        # Ortalama gol performansına göre takımları sınıflandır (aşırı yüksek tahminleri daha sıkı sınırla)
        home_is_high_scoring = home_recent_avg_goals[5] > 2.0
        away_is_high_scoring = away_recent_avg_goals[5] > 1.5

        # Yüksek gol atan takımlar için daha esnek, düşük gol atan takımlar için daha sıkı sınırlar
        # Ancak genel olarak daha yüksek tahminlere izin ver
        home_threshold = 3.0 if home_is_high_scoring else 2.7
        away_threshold = 2.5 if away_is_high_scoring else 2.2

        # Ev sahibi gol tahminlerini sınırla - daha yumuşak sınırlama için scaling factor artırıldı
        avg_home_goals = limit_high_values(avg_home_goals, home_threshold, 0.5)

        # Deplasman gol tahminlerini sınırla - daha yumuşak sınırlama için scaling factor artırıldı
        avg_away_goals = limit_high_values(avg_away_goals, away_threshold, 0.45)

        # Gerçek dünya istatistiklerine göre maksimum sınırlar - daha dengeli değerler
        home_max = 3.2 if home_is_high_scoring else 3.0
        away_max = 3.2 if away_is_high_scoring else 2.8

        if avg_home_goals > home_max:
            avg_home_goals = home_max + ((avg_home_goals - home_max) * 0.25)

        if avg_away_goals > away_max:
            avg_away_goals = away_max + ((avg_away_goals - away_max) * 0.25)

        # Minimum değerler için daha dengeli alt sınırlar belirle
        # Ev sahibi için daha düşük minimum değer kullanarak zayıf takımları daha doğru yansıt
        avg_home_goals = max(0.8, avg_home_goals)
        avg_away_goals = max(0.7, avg_away_goals)

        # Standart sapma hesapla - Poisson dağılımında standart sapma, ortalamanın kareköküdür
        std_dev_home = np.sqrt(avg_home_goals)  
        std_dev_away = np.sqrt(avg_away_goals)

        # KG VAR/YOK mantığını daha akıllı bir şekilde hesaplayalım
        # Eğer iki takımın da beklenen gol sayısı yüksekse, KG VAR VAR olasılığı daha yüksek olmalı
        kg_var_theoretical_prob = p_home_scores * p_away_scores  # Bağımsız olasılıklar çarpımı

        # Simülasyon sonuçları ile teorik hesaplamalar arasında denge kuralım
        kg_var_adjusted_prob = 0.65 * both_teams_scored_prob + 0.35 * kg_var_theoretical_prob

        # Son 5 karşılaşmada iki takım da gol attıysa KG VAR olasılığını artır
        kg_var_recent_matches = sum(1 for match in home_form.get('recent_match_data', [])[:5] if match.get('goals_scored', 0) > 0 and match.get('goals_conceded', 0) > 0)
        kg_var_recent_matches += sum(1 for match in away_form.get('recent_match_data', [])[:5] if match.get('goals_scored', 0) > 0 and match.get('goals_conceded', 0) > 0)

        if kg_var_recent_matches >= 5:  # Son maçlarda KG VAR eğilimi varsa
            kg_var_adjusted_prob = max(kg_var_adjusted_prob, 0.70)  # En az %70 olasılık

        # 2.5 ve 3.5 gol için teorik olasılıklar (Poisson kümülatif dağılım fonksiyonu)
        # P(X > 2.5) = 1 - P(X ≤ 2) where X ~ Poisson(lambda)
        lambda_total = expected_total_goals
        p_under_25_theoretical = np.exp(-lambda_total) * (1 + lambda_total + (lambda_total**2)/2)
        p_over_25_theoretical = 1 - p_under_25_theoretical

        # 3.5 gol için
        p_under_35_theoretical = p_under_25_theoretical + np.exp(-lambda_total) * (lambda_total**3)/6
        p_over_35_theoretical = 1 - p_under_35_theoretical

        # Simülasyon ve teorik hesaplamalar arasında denge kurma
        # Simülasyon sonuçlarına daha fazla ağırlık veriyoruz (ÜST tahminleri için)
        over_25_adjusted_prob = 0.7 * over_2_5_goals_prob + 0.3 * p_over_25_theoretical
        over_35_adjusted_prob = 0.7 * over_3_5_goals_prob + 0.3 * p_over_35_theoretical

        # Bahis tahminlerini hazırla - korner ve kart tahminlerini çıkararak basitleştir
        bet_predictions = {
            'match_result': 'MS1' if self._get_most_likely_outcome(home_win_prob, draw_prob, away_win_prob) == 'HOME_WIN' else
                           'X' if self._get_most_likely_outcome(home_win_prob, draw_prob, away_win_prob) == 'DRAW' else 'MS2',
            'both_teams_to_score': 'KG VAR' if kg_var_prediction else 'KG YOK',  # KG VAR/YOK formatında - tutarlılık için kg_var_prediction kullan
            'over_2_5_goals': '2.5 ÜST' if over_25_adjusted_prob > 0.5 else '2.5 ALT',  # Doğrudan Türkçe format kullan
            'over_3_5_goals': '3.5 ÜST' if over_35_adjusted_prob > 0.5 else '3.5 ALT',  # Doğrudan Türkçe format kullan
            'exact_score': most_likely_score[0],  # Daha önce KG VAR/YOK'a göre ayarlandı
            'half_time_full_time': most_likely_ht_ft[0].replace('HOME_WIN', 'MS1').replace('DRAW', 'X').replace('AWAY_WIN', 'MS2'),
            'first_goal_time': most_likely_first_goal_time[0],
            'first_goal_team': 'EV' if first_goal_home_prob > first_goal_away_prob and first_goal_home_prob > no_goal_prob else
                              'DEP' if first_goal_away_prob > first_goal_home_prob and first_goal_away_prob > no_goal_prob else 'GOL YOK'
            # Korner ve kart tahminleri kaldırıldı
        }
        
        # KONSEPTÜEL TUTARLILIK KONTROLÜ:
        # Kesin skordan toplam gol sayısını hesapla ve ÜST/ALT kararlarını güncelle
        if '-' in bet_predictions['exact_score']:
            try:
                home_goals, away_goals = map(int, bet_predictions['exact_score'].split('-'))
                total_goals = home_goals + away_goals
                
                # 2.5 ÜST/ALT kontrolü
                if total_goals > 2:  # 3 veya daha fazla gol varsa
                    logger.info(f"Skordan ({bet_predictions['exact_score']}) toplam gol: {total_goals}, 2.5 ÜST olarak güncellendi")
                    bet_predictions['over_2_5_goals'] = '2.5 ÜST'
                    over_25_adjusted_prob = 0.9  # Yüksek olasılık ver
                else:  # 0, 1 veya 2 gol varsa
                    logger.info(f"Skordan ({bet_predictions['exact_score']}) toplam gol: {total_goals}, 2.5 ALT olarak güncellendi")
                    bet_predictions['over_2_5_goals'] = '2.5 ALT'
                    over_25_adjusted_prob = 0.1  # Düşük olasılık ver
                
                # 3.5 ÜST/ALT kontrolü
                if total_goals > 3:  # 4 veya daha fazla gol varsa
                    logger.info(f"Skordan ({bet_predictions['exact_score']}) toplam gol: {total_goals}, 3.5 ÜST olarak güncellendi")
                    bet_predictions['over_3_5_goals'] = '3.5 ÜST'
                    over_35_adjusted_prob = 0.9  # Yüksek olasılık ver
                else:  # 0, 1, 2 veya 3 gol varsa
                    logger.info(f"Skordan ({bet_predictions['exact_score']}) toplam gol: {total_goals}, 3.5 ALT olarak güncellendi")
                    bet_predictions['over_3_5_goals'] = '3.5 ALT'
                    over_35_adjusted_prob = 0.1  # Düşük olasılık ver 
                
                # KG VAR/YOK kontrolü
                if home_goals > 0 and away_goals > 0:
                    logger.info(f"Skordan ({bet_predictions['exact_score']}) her iki takım da gol atıyor, KG VAR olarak güncellendi")
                    bet_predictions['both_teams_to_score'] = 'KG VAR'  # Türkçe formatta kullan, tutarlılık için
                    kg_var_prediction = True
                    p_kg_var_combined = 0.9  # Yüksek olasılık ver
                else:  # En az bir takım gol atmamışsa (0-0, 1-0, 0-1, 2-0, 0-2, vb.)
                    logger.info(f"Skordan ({bet_predictions['exact_score']}) en az bir takım gol atmıyor, KG YOK olarak güncellendi")
                    bet_predictions['both_teams_to_score'] = 'KG YOK'  # Türkçe formatta kullan, tutarlılık için
                    kg_var_prediction = False
                    p_kg_var_combined = 0.1  # Düşük olasılık ver
                
                # Maç sonucu kontrolü
                if home_goals > away_goals:
                    logger.info(f"Skordan ({bet_predictions['exact_score']}) ev sahibi önde, MS1 olarak güncellendi")
                    bet_predictions['match_result'] = 'MS1'
                elif away_goals > home_goals:
                    logger.info(f"Skordan ({bet_predictions['exact_score']}) deplasman önde, MS2 olarak güncellendi")
                    bet_predictions['match_result'] = 'MS2'
                else:
                    logger.info(f"Skordan ({bet_predictions['exact_score']}) eşitlik, X olarak güncellendi")
                    bet_predictions['match_result'] = 'X'
            except Exception as e:
                logger.error(f"Skor analizi yapılırken hata oluştu: {bet_predictions['exact_score']} - Hata: {str(e)}")
        
        # Ek kontrol - bet_predictions içinde 'exact_score' kesin olarak ayarlandığından emin ol
        logger.info(f"Kesin skor tahmini: {most_likely_score[0]} (KG {'VAR' if kg_var_prediction else 'YOK'})")

        # Son maçların gol ortalamalarını sadece gerekli değerlerle kaydet
        recent_goals_average = {
            'home': home_recent_avg_goals.get(5, 0),
            'away': away_recent_avg_goals.get(5, 0)
        }

        # Gol beklentilerine göre KG VAR/YOK, ÜST/ALT tahminlerini ayarla ve tutarlılığı sağla
        expected_total_goals = avg_home_goals + avg_away_goals
        
        # KG VAR/YOK tahmini - geliştirilmiş mantık
        # Takımların gol beklentileri, gol atma olasılıkları ve son maçlardaki performanslarını hesaba kat
        p_home_scores_at_least_one = 1 - np.exp(-avg_home_goals)
        p_away_scores_at_least_one = 1 - np.exp(-avg_away_goals)
        
        # İki takımın da en az 1 gol atma olasılığı
        p_both_teams_score = p_home_scores_at_least_one * p_away_scores_at_least_one
        
        # Son maçlardaki KG VAR/YOK oranlarını analiz et
        kg_var_rate_home = 0
        kg_var_rate_away = 0
        kg_var_matches_home = 0
        kg_var_matches_away = 0
        
        # Ev sahibi takımın son maçlarında KG VAR oranı
        for match in home_match_data[:min(10, len(home_match_data))]:
            if match.get('goals_scored', 0) > 0 and match.get('goals_conceded', 0) > 0:
                kg_var_matches_home += 1
        
        # Deplasman takımının son maçlarında KG VAR oranı
        for match in away_match_data[:min(10, len(away_match_data))]:
            if match.get('goals_scored', 0) > 0 and match.get('goals_conceded', 0) > 0:
                kg_var_matches_away += 1
        
        if home_match_data and len(home_match_data) > 0:
            kg_var_rate_home = kg_var_matches_home / min(10, len(home_match_data))
        
        if away_match_data and len(away_match_data) > 0:
            kg_var_rate_away = kg_var_matches_away / min(10, len(away_match_data))
        
        # İki takımın geçmiş maçlarındaki KG VAR oranı
        kg_var_historical_rate = (kg_var_rate_home + kg_var_rate_away) / 2
        
        logger.info(f"KG VAR geçmiş oranları - Ev: {kg_var_rate_home:.2f}, Deplasman: {kg_var_rate_away:.2f}, Ortalama: {kg_var_historical_rate:.2f}")
        
        # Teorik ve geçmiş verileri birleştirerek daha doğru tahmin yap
        p_kg_var_combined = 0.6 * p_both_teams_score + 0.4 * kg_var_historical_rate
        
        logger.info(f"KG VAR olasılığı - Teorik: {p_both_teams_score:.2f}, Geçmiş: {kg_var_historical_rate:.2f}, Birleşik: {p_kg_var_combined:.2f}")
        
        # Önce varsayılan tahmin yap (başlangıçta nötr)
        kg_var_prediction = p_kg_var_combined > 0.5  # Başlangıç tahmini
        
        # Durum 1: Her iki takımın da gol beklentisi 1.0'dan büyükse kesinlikle KG VAR tahmin et
        if avg_home_goals > 1.0 and avg_away_goals > 1.0:
            kg_var_prediction = True
            kg_var_adjusted_prob = max(kg_var_adjusted_prob, 0.75)  # En az %75 olasılık ver
            logger.info(f"Her iki takımın da gol beklentisi 1.0'dan büyük: KG VAR tahmini yapıldı")
            
        # Durum 2: Bir takımın gol beklentisi 1.2'den, diğerinin 0.7'den büyükse yine KG VAR tahmin et
        elif (avg_home_goals > 1.2 and avg_away_goals > 0.7) or (avg_away_goals > 1.2 and avg_home_goals > 0.7):
            kg_var_prediction = True
            kg_var_adjusted_prob = max(kg_var_adjusted_prob, 0.7)  # En az %70 olasılık ver
            logger.info(f"Bir takımın gol beklentisi 1.2'den büyük, diğerinin 0.7'den büyük: KG VAR tahmini yapıldı")
        
        # KG VAR/YOK tahminini bahis tahminlerine kaydet
        # 0-0 skoru özel bir durumdur - kesinlikle KG YOK olmalıdır
        exact_score_parts = bet_predictions['exact_score'].split('-')
        home_goals = int(exact_score_parts[0])
        away_goals = int(exact_score_parts[1])
        
        # Skor 0-0 veya taraflardan biri 0 golse, kesinlikle KG YOK olmalı
        if home_goals == 0 or away_goals == 0:
            kg_var_prediction = False
            logger.info(f"Skor {bet_predictions['exact_score']} olduğu için KG YOK olarak ayarlandı")
        
        if kg_var_prediction:  # True = KG VAR
            bet_predictions['both_teams_to_score'] = 'KG VAR'  # Tutarlılık için 'KG VAR' kullanıyoruz
            kg_var_adjusted_prob = max(kg_var_adjusted_prob, 0.75)
        else:  # False = KG YOK
            bet_predictions['both_teams_to_score'] = 'KG YOK'  # Tutarlılık için 'KG YOK' kullanıyoruz
            kg_var_adjusted_prob = min(kg_var_adjusted_prob, 0.3)
        
        # Monte Carlo simülasyonunu KG VAR/YOK kısıtlaması ile tekrar çalıştır
        logger.info(f"KG {'VAR' if kg_var_prediction else 'YOK'} kısıtlaması ile Monte Carlo simülasyonu çalıştırılıyor")
        
        # Monte Carlo simülasyonunu KG VAR/YOK kısıtlaması ile çalıştır
        simulation_result = self.monte_carlo_simulation(
            adjusted_home_goals, 
            adjusted_away_goals, 
            home_form=home_form,
            away_form=away_form,
            kg_var_prediction=kg_var_prediction  # KG VAR/YOK kısıtlaması ekle
        )
        
        # Simülasyon sonuçlarını güncelle
        exact_scores = simulation_result['exact_scores']
        most_likely_score = simulation_result['most_likely_score']
        
        # Kesin skoru bet_predictions'da güncelle
        bet_predictions['exact_score'] = most_likely_score[0]
        logger.info(f"KG {'VAR' if kg_var_prediction else 'YOK'} kısıtlamalı Monte Carlo simülasyonu sonucu kesin skor: {most_likely_score[0]}")
        
        # 2.5 ÜST/ALT tahmini - Poisson dağılımı temelli
        # P(X > 2.5) = 1 - P(X ≤ 2) = 1 - (e^(-λ) + λe^(-λ) + (λ^2/2)e^(-λ))
        # Burada λ = beklenen toplam gol sayısı
        p_under_25 = np.exp(-expected_total_goals) * (1 + expected_total_goals + (expected_total_goals**2)/2)
        p_over_25 = 1 - p_under_25
        
        # Daha tutarlı 2.5 ÜST/ALT tahmini
        if p_over_25 > 0.52 or expected_total_goals > 2.6:
            bet_predictions['over_2_5_goals'] = '2.5 ÜST'
            over_25_adjusted_prob = max(over_25_adjusted_prob, 0.7)
        else:
            bet_predictions['over_2_5_goals'] = '2.5 ALT'
            over_25_adjusted_prob = min(over_25_adjusted_prob, 0.3)
        
        # 3.5 ÜST/ALT tahmini - Poisson dağılımı temelli
        # P(X > 3.5) = 1 - P(X ≤ 3) = 1 - (e^(-λ) + λe^(-λ) + (λ^2/2)e^(-λ) + (λ^3/6)e^(-λ))
        p_under_35 = p_under_25 + np.exp(-expected_total_goals) * (expected_total_goals**3)/6
        p_over_35 = 1 - p_under_35
        
        # Daha tutarlı 3.5 ÜST/ALT tahmini
        if p_over_35 > 0.48 or expected_total_goals > 3.4:
            bet_predictions['over_3_5_goals'] = '3.5 ÜST'
            over_35_adjusted_prob = max(over_35_adjusted_prob, 0.65)
        else:
            bet_predictions['over_3_5_goals'] = '3.5 ALT'
            over_35_adjusted_prob = min(over_35_adjusted_prob, 0.3)
            
        # KG VAR/YOK ve 2.5 ÜST/ALT, 3.5 ÜST/ALT arasındaki tutarlılığı sağla
        
        # Önce KG VAR/YOK ile kesin skor arasındaki tutarlılığı kontrol et
        exact_score_parts = bet_predictions['exact_score'].split('-')
        home_goals = int(exact_score_parts[0])
        away_goals = int(exact_score_parts[1])
        
        # KG YOK tahmininde her iki takımın da gol attığı skor varsa düzelt
        # Burada both_teams_to_score değeri 'YES/NO' veya 'KG VAR/KG YOK' olabilir, iki formatı da kontrol et
        if ((bet_predictions['both_teams_to_score'] == 'KG YOK' or 
             bet_predictions['both_teams_to_score'] == 'NO') and 
            home_goals > 0 and away_goals > 0):
            logger.warning(f"Tutarsızlık tespit edildi: KG YOK tahmini ile {bet_predictions['exact_score']} skoru çelişiyor!")
            
            # En olası KG YOK skorlarını bul
            kg_yok_scores = [(score, count) for score, count in exact_scores.items() 
                            if int(score.split('-')[0]) == 0 or int(score.split('-')[1]) == 0]
            
            # KG YOK skorları varsa, en olasını seç
            if kg_yok_scores:
                most_likely_kg_yok = sorted(kg_yok_scores, key=lambda x: x[1], reverse=True)[0]
                bet_predictions['exact_score'] = most_likely_kg_yok[0]
                most_likely_score = most_likely_kg_yok
                logger.info(f"KG YOK ile tutarlılık sağlamak için kesin skor {most_likely_kg_yok[0]} olarak güncellendi (simülasyon bazlı)")
            else:
                # KG YOK için yeni bir kesin skor belirle
                if avg_home_goals > avg_away_goals * 1.5:
                    # Ev sahibi çok daha güçlüyse 2-0 öner
                    bet_predictions['exact_score'] = '2-0'
                    most_likely_score = ('2-0', exact_scores.get('2-0', 0))
                    logger.info(f"KG YOK ile tutarlılık sağlamak için kesin skor 2-0 olarak güncellendi")
                elif avg_home_goals > avg_away_goals:
                    # Ev sahibi daha güçlüyse 1-0 öner
                    bet_predictions['exact_score'] = '1-0'
                    most_likely_score = ('1-0', exact_scores.get('1-0', 0))
                    logger.info(f"KG YOK ile tutarlılık sağlamak için kesin skor 1-0 olarak güncellendi")
                elif avg_away_goals > avg_home_goals * 1.5:
                    # Deplasman çok daha güçlüyse 0-2 öner
                    bet_predictions['exact_score'] = '0-2'  
                    most_likely_score = ('0-2', exact_scores.get('0-2', 0))
                    logger.info(f"KG YOK ile tutarlılık sağlamak için kesin skor 0-2 olarak güncellendi")
                elif avg_away_goals > avg_home_goals:
                    # Deplasman daha güçlüyse 0-1 öner
                    bet_predictions['exact_score'] = '0-1'  
                    most_likely_score = ('0-1', exact_scores.get('0-1', 0))
                    logger.info(f"KG YOK ile tutarlılık sağlamak için kesin skor 0-1 olarak güncellendi")
                else:
                    # Eşit güçteyseler 0-0 öner
                    bet_predictions['exact_score'] = '0-0'
                    most_likely_score = ('0-0', exact_scores.get('0-0', 0))
                    logger.info(f"KG YOK ile tutarlılık sağlamak için kesin skor 0-0 olarak güncellendi")
        
        # KG VAR tahmininde her iki takımın da gol atmadığı skor varsa düzelt
        # Burada both_teams_to_score değeri 'YES/NO' veya 'KG VAR/KG YOK' olabilir, iki formatı da kontrol et
        elif ((bet_predictions['both_teams_to_score'] == 'KG VAR' or 
               bet_predictions['both_teams_to_score'] == 'KG VAR') and 
              (home_goals == 0 or away_goals == 0)):
            logger.warning(f"Tutarsızlık tespit edildi: KG VAR tahmini ile {bet_predictions['exact_score']} skoru çelişiyor!")
            
            # En olası KG VAR skorlarını bul
            kg_var_scores = [(score, count) for score, count in exact_scores.items() 
                             if int(score.split('-')[0]) > 0 and int(score.split('-')[1]) > 0]
            
            # KG VAR skorları varsa, en olasını seç
            if kg_var_scores:
                most_likely_kg_var = sorted(kg_var_scores, key=lambda x: x[1], reverse=True)[0]
                bet_predictions['exact_score'] = most_likely_kg_var[0]
                most_likely_score = most_likely_kg_var
                logger.info(f"KG VAR ile tutarlılık sağlamak için kesin skor {most_likely_kg_var[0]} olarak güncellendi (simülasyon bazlı)")
            else:
                # KG VAR için yeni bir kesin skor belirle
                if avg_home_goals > avg_away_goals:
                    # Ev sahibi daha güçlüyse 2-1 öner
                    bet_predictions['exact_score'] = '2-1'
                    most_likely_score = ('2-1', exact_scores.get('2-1', 0))
                    logger.info(f"KG VAR ile tutarlılık sağlamak için kesin skor 2-1 olarak güncellendi")
                elif avg_away_goals > avg_home_goals:
                    # Deplasman daha güçlüyse 1-2 öner
                    bet_predictions['exact_score'] = '1-2'
                    most_likely_score = ('1-2', exact_scores.get('1-2', 0))
                    logger.info(f"KG VAR ile tutarlılık sağlamak için kesin skor 1-2 olarak güncellendi")
                else:
                    # Eşit güçteyseler 1-1 öner
                    bet_predictions['exact_score'] = '1-1'
                    most_likely_score = ('1-1', exact_scores.get('1-1', 0))
                    logger.info(f"KG VAR ile tutarlılık sağlamak için kesin skor 1-1 olarak güncellendi")
        
        # Eğer 2.5 ALT ve KG VAR tahmini varsa, maç sonu olasılıklarını da kontrol et
        # Burada both_teams_to_score değeri 'YES/NO' veya 'KG VAR/KG YOK' olabilir, iki formatı da kontrol et
        if (bet_predictions['over_2_5_goals'] == '2.5 ALT' and 
            (bet_predictions['both_teams_to_score'] == 'KG VAR' or 
             bet_predictions['both_teams_to_score'] == 'KG VAR')):
            # Bu kombinasyon genellikle 1-1 skoru gerektirir, beraberlik olasılığı artmalı
            draw_prob = max(draw_prob, home_win_prob, away_win_prob) + 0.1
            # Olasılıkları yeniden normalize et
            total = draw_prob + home_win_prob + away_win_prob
            draw_prob = draw_prob / total
            home_win_prob = home_win_prob / total
            away_win_prob = away_win_prob / total
            
            # Beraberlik olasılığı çok yüksekse, beraberlik sonucunu öner
            if draw_prob > 0.5:
                bet_predictions['match_result'] = 'DRAW'
            
            # Kesin skoru 1-1 olarak güncelle
            bet_predictions['exact_score'] = '1-1'
            most_likely_score = ('1-1', exact_scores.get('1-1', 0))
            logger.info(f"2.5 ALT + KG VAR tahmini nedeniyle kesin skor 1-1 olarak güncellendi")
        
        # Eğer 3.5 ÜST tahmini varsa, 2.5 ÜST de olmalı
        if bet_predictions['over_3_5_goals'] == '3.5 ÜST' and bet_predictions['over_2_5_goals'] == '2.5 ALT':
            bet_predictions['over_2_5_goals'] = '2.5 ÜST'
            over_25_adjusted_prob = max(over_25_adjusted_prob, 0.8)
            logger.info(f"3.5 ÜST tahmini nedeniyle 2.5 ÜST olarak güncellendi")
            
        # Eğer 2.5 ÜST ve KG YOK tahmini varsa, en az bir takımın çok gol atması bekleniyor demektir
        # Bu durumda kesin skoru KG YOK ama yüksek skorlu bir tahmin olarak güncelle
        # Burada both_teams_to_score değeri 'YES/NO' veya 'KG VAR/KG YOK' olabilir, iki formatı da kontrol et
        if (bet_predictions['over_2_5_goals'] == '2.5 ÜST' and 
            (bet_predictions['both_teams_to_score'] == 'KG YOK' or 
             bet_predictions['both_teams_to_score'] == 'NO')):
            # Hangi takımın daha yüksek gol beklentisi varsa o takımın kazanacağını tahmin et
            if avg_home_goals > avg_away_goals + 0.5:
                # Ev sahibi daha fazla gol beklentisine sahip
                if avg_home_goals > 2.0:
                    # Ev sahibi çok gol atabilir
                    bet_predictions['exact_score'] = '3-0'
                    most_likely_score = ('3-0', exact_scores.get('3-0', 0))
                else:
                    # Ev sahibi orta seviyede gol atabilir
                    bet_predictions['exact_score'] = '2-0'
                    most_likely_score = ('2-0', exact_scores.get('2-0', 0))
                logger.info(f"2.5 ÜST + KG YOK + Ev sahibi üstünlüğü tahmini nedeniyle kesin skor {bet_predictions['exact_score']} olarak güncellendi")
            elif avg_away_goals > avg_home_goals + 0.5:
                # Deplasman daha fazla gol beklentisine sahip
                if avg_away_goals > 2.0:
                    # Deplasman çok gol atabilir
                    bet_predictions['exact_score'] = '0-3'
                    most_likely_score = ('0-3', exact_scores.get('0-3', 0))
                else:
                    # Deplasman orta seviyede gol atabilir
                    bet_predictions['exact_score'] = '0-2'
                    most_likely_score = ('0-2', exact_scores.get('0-2', 0))
                logger.info(f"2.5 ÜST + KG YOK + Deplasman üstünlüğü tahmini nedeniyle kesin skor {bet_predictions['exact_score']} olarak güncellendi")
            else:
                # Takımların gol beklentileri yakınsa, KG YOK ama 2.5 ÜST için en olası senaryo 3-0 veya 0-3
                # En yüksek olasılıklı KG YOK ve yüksek skorlu maçı bul
                likely_scores = [(score, count) for score, count in exact_scores.items() 
                                 if '-' in score and (int(score.split('-')[0]) == 0 or int(score.split('-')[1]) == 0) and 
                                    int(score.split('-')[0]) + int(score.split('-')[1]) >= 3]
                if likely_scores:
                    most_likely_score = max(likely_scores, key=lambda x: x[1])
                    bet_predictions['exact_score'] = most_likely_score[0]
                    logger.info(f"2.5 ÜST + KG YOK tahmini nedeniyle kesin skor {most_likely_score[0]} olarak güncellendi")

        # En yüksek olasılıklı tahmini belirle - korner, kart, ilk gol ve İY/MS tahminlerini çıkardık
        bet_probabilities = {
            'match_result': max(home_win_prob, draw_prob, away_win_prob),
            'both_teams_to_score': max(kg_var_adjusted_prob, 1 - kg_var_adjusted_prob),
            'over_2_5_goals': max(over_25_adjusted_prob, 1 - over_25_adjusted_prob),
            'over_3_5_goals': max(over_35_adjusted_prob, 1 - over_35_adjusted_prob),
            'exact_score': most_likely_score_prob
            # İY/MS, ilk gol tahminleri kaldırıldı (half_time_full_time, first_goal_time ve first_goal_team)
        }

        # Tahminler arasındaki mantık tutarlılığını kontrol et - geliştirilmiş versiyon
        
        # İlk adım: Gol beklentilerine göre olasılıkları ayarlama
        # Doğrudan gol beklentisi farkından MS olasılıklarını hesapla
        goal_diff = avg_home_goals - avg_away_goals
        
        # Gol beklentileri ve maç sonucu arasındaki ilişkiyi daha doğru kur
        # Gol farkı formülü: sigmoid benzeri bir yaklaşım kullan
        def sigmoid_like(x, scale=1.5):
            return 1 / (1 + np.exp(-scale * x))
        
        # Gol farkına göre ev sahibi ve deplasman kazanma olasılıklarını hesapla
        base_home_win = sigmoid_like(goal_diff)
        base_away_win = sigmoid_like(-goal_diff)
        
        # Beraberlik olasılığını gol farkının mutlak değerine göre ayarla
        # Gol farkı az ise beraberlik olasılığı yüksek olmalı
        base_draw = 1 - (sigmoid_like(abs(goal_diff), scale=2.5))
        
        # Olasılıkları normalize et
        total = base_home_win + base_draw + base_away_win
        norm_home_win = base_home_win / total
        norm_draw = base_draw / total  
        norm_away_win = base_away_win / total
        
        # Mevcut simülasyon olasılıkları ile hesaplanan olasılıklar arasında denge kur
        # %70 simülasyon sonuçlarına, %30 gol beklentisi temelli matematiksel hesaplamaya güven
        blend_factor = 0.3
        home_win_prob = (1 - blend_factor) * home_win_prob + blend_factor * norm_home_win
        draw_prob = (1 - blend_factor) * draw_prob + blend_factor * norm_draw  
        away_win_prob = (1 - blend_factor) * away_win_prob + blend_factor * norm_away_win
        
        logger.info(f"Gol beklentilerine göre MS olasılıkları ayarlandı: MS1={home_win_prob:.2f}, X={draw_prob:.2f}, MS2={away_win_prob:.2f}")
        
        # İkinci adım: Kesin skor ile diğer tahminleri uyumlu hale getir
        # Önce en olası kesin skoru belirle
        top_3_scores = sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # En olası skorları analiz et
        score_probabilities = {}
        for score, count in top_3_scores:
            score_probabilities[score] = count / simulations
            
        logger.info(f"En olası skorlar ve olasılıkları: {score_probabilities}")
        
        # Doğrudan beklenen gol değerlerinden kesin skoru hesapla
        most_expected_home_score = round(avg_home_goals)
        most_expected_away_score = round(avg_away_goals)
        expected_score = f"{most_expected_home_score}-{most_expected_away_score}"
        
        # Top 5 skorları alarak olasılık dağılımını daha iyi anla
        top_5_scores = sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"En olası 5 skor ve olasılıkları: {[(s, round(c/simulations*100, 2)) for s, c in top_5_scores]}")
        logger.info(f"Beklenen goller: Ev={avg_home_goals:.2f}, Deplasman={avg_away_goals:.2f}")
        
        # Kesin skor olasılıklarını daha doğru değerlendirmek için histogram
        score_histogram = {}
        
        # Düşük skorlu maçlar için olasılık artışı faktörleri - daha güçlü artış
        low_score_boost = {
            "0-0": 3.0,  # %200 artış
            "1-0": 2.5,  # %150 artış
            "0-1": 2.5,  # %150 artış
            "1-1": 1.8,  # %80 artış
            "2-0": 1.5,  # %50 artış
            "0-2": 1.5   # %50 artış
        }
        
        # Her iki takımın da gol beklentisi düşükse, düşük skorların olasılığını önemli ölçüde artır
        # Sınırı 1.2'ye yükselttik - daha fazla maç düşük skorlu olarak değerlendirilecek
        is_low_scoring_match = avg_home_goals < 1.2 and avg_away_goals < 1.2
        
        # Skoru bir takımın lehine değiştirmek için form farklarını kontrol et
        home_stronger = False
        away_stronger = False
        
        if 'home_performance' in home_form and 'away_performance' in away_form:
            home_form_points = home_form.get('home_performance', {}).get('weighted_form_points', 0.5)
            away_form_points = away_form.get('away_performance', {}).get('weighted_form_points', 0.5)
            
            # Form puanları arasında önemli fark varsa
            if home_form_points > away_form_points + 0.2:
                home_stronger = True
                logger.info(f"Ev sahibi takım form olarak daha güçlü: {home_form_points:.2f} > {away_form_points:.2f}")
            elif away_form_points > home_form_points + 0.2:
                away_stronger = True
                logger.info(f"Deplasman takımı form olarak daha güçlü: {away_form_points:.2f} > {home_form_points:.2f}")
        
        # Takım form farklarına göre düşük skor boost faktörlerini ayarla
        if home_stronger:
            low_score_boost["1-0"] = 3.0  # %200 artış
            low_score_boost["2-0"] = 2.0  # %100 artış
            low_score_boost["2-1"] = 1.5  # %50 artış
            logger.info(f"Ev sahibi takım daha formda olduğu için 1-0, 2-0, 2-1 skorlarının olasılıkları artırıldı")
        elif away_stronger:
            low_score_boost["0-1"] = 3.0  # %200 artış
            low_score_boost["0-2"] = 2.0  # %100 artış
            low_score_boost["1-2"] = 1.5  # %50 artış
            logger.info(f"Deplasman takımı daha formda olduğu için 0-1, 0-2, 1-2 skorlarının olasılıkları artırıldı")
        
        # Skorları histogram şeklinde oluştur ve düşük skorlu maçlar için özel işlem yap
        for h in range(6):  # 0-5 gol
            for a in range(6):  # 0-5 gol
                score = f"{h}-{a}"
                base_prob = exact_scores.get(score, 0) / simulations
                
                # Düşük skorlu maç kontrolü ve ek kontroller
                if is_low_scoring_match:
                    # Düşük skorlu bir maçta her iki takım da 1.0'dan az gol beklentisine sahipse
                    # ve skorda toplam gol sayısı 3 veya daha fazlaysa, bu skor daha az olası
                    if h + a >= 3:
                        base_prob = base_prob * 0.5  # %50 azalt, düşük skorlu maçta yüksek skoru daha agresif azalt
                        logger.info(f"Düşük skorlu maçta yüksek skor {score} azaltıldı: %{round(base_prob*100, 2)}")
                    
                    # 0-0, 0-1, 1-0 gibi skorları özel olarak artır
                    if score in ["0-0", "1-0", "0-1"]:
                        special_boost = 3.5  # Çok düşük skorlu maçlar için ekstra artış
                        boosted_prob = base_prob * special_boost
                        score_histogram[score] = boosted_prob
                        logger.info(f"Çok düşük skorlu maç tespiti: {score} skoru olasılığı %{round(base_prob*100, 2)}'den %{round(boosted_prob*100, 2)}'e yükseltildi")
                    # Boost uygulanacak skor ise olasılığı artır
                    elif score in low_score_boost:
                        boosted_prob = base_prob * low_score_boost[score]
                        score_histogram[score] = boosted_prob
                        logger.info(f"Düşük skorlu maç tespiti: {score} skoru olasılığı %{round(base_prob*100, 2)}'den %{round(boosted_prob*100, 2)}'e yükseltildi")
                    else:
                        score_histogram[score] = base_prob
                else:
                    # Düşük skorlu maç değilse normal olasılık kullan
                    score_histogram[score] = base_prob
                    
        # Skor tahminlerini normalize et - toplam olasılık 1.0 olsun
        total_prob = sum(score_histogram.values())
        if total_prob > 0:
            for score in score_histogram:
                score_histogram[score] = score_histogram[score] / total_prob
                
        # En yüksek olasılıklı skorları gruplandırarak analiz et
        same_outcome_scores = {
            "HOME_WIN": {},
            "DRAW": {},
            "AWAY_WIN": {}
        }
        
        # Gerçekçi skor sınırları - beklenen gollere göre belirleme
        max_reasonable_home = 4
        max_reasonable_away = 4
        
        if avg_home_goals < 1.0:
            max_reasonable_home = 1
        elif avg_home_goals < 2.0:
            max_reasonable_home = 2
        elif avg_home_goals < 3.0:
            max_reasonable_home = 3
            
        if avg_away_goals < 1.0:
            max_reasonable_away = 1
        elif avg_away_goals < 2.0:
            max_reasonable_away = 2
        elif avg_away_goals < 3.0:
            max_reasonable_away = 3
        
        # Skorları gruplandır ve gerçekçi olmayan skorları filtrele
        for score, prob in score_histogram.items():
            if '-' in score:
                h, a = map(int, score.split('-'))
                
                # Gerçekçi olmayan yüksek skorları filtrele
                if h > max_reasonable_home * 1.5 or a > max_reasonable_away * 1.5:
                    logger.info(f"Gerçekçi olmayan skor filtrelendi: {score} (beklenen goller: {avg_home_goals:.2f}-{avg_away_goals:.2f})")
                    continue
                
                if h > a:
                    same_outcome_scores["HOME_WIN"][score] = prob
                elif h == a:
                    # Yüksek beraberlik skorlarını filtrele
                    if h <= max_reasonable_home:
                        same_outcome_scores["DRAW"][score] = prob
                    else:
                        logger.info(f"Gerçekçi olmayan beraberlik skoru filtrelendi: {score} (beklenen goller: {avg_home_goals:.2f}-{avg_away_goals:.2f})")
                else:
                    same_outcome_scores["AWAY_WIN"][score] = prob
        
        # Maç sonucu olasılıklarına göre en olası skoru belirlemek - beklenen golleri de hesaba katarak
        most_likely_outcome = self._get_most_likely_outcome(home_win_prob, draw_prob, away_win_prob, avg_home_goals, avg_away_goals)
        
        # En olası maç sonucu için bir skor listesi oluştur
        if most_likely_outcome in same_outcome_scores and same_outcome_scores[most_likely_outcome]:
            top_scores_by_outcome = sorted(same_outcome_scores[most_likely_outcome].items(), key=lambda x: x[1], reverse=True)
        else:
            # Eğer en olası sonuç için skor bulunamazsa (filtrelerden dolayı boş kalmış olabilir)
            # alternatif bir sonuç kullan veya beklenen gollere göre bir skor oluştur
            logger.warning(f"En olası sonuç {most_likely_outcome} için skor bulunamadı, alternatif kullanılıyor")
            
            # Beklenen gollere dayanarak alternatif bir maç sonucu belirle
            if avg_home_goals > avg_away_goals + 0.5:
                alt_outcome = "HOME_WIN"
            elif avg_away_goals > avg_home_goals + 0.5:
                alt_outcome = "AWAY_WIN"
            else:
                alt_outcome = "DRAW"
                
            # Alternatif sonuç için skorlar var mı?
            if alt_outcome in same_outcome_scores and same_outcome_scores[alt_outcome]:
                top_scores_by_outcome = sorted(same_outcome_scores[alt_outcome].items(), key=lambda x: x[1], reverse=True)
                logger.info(f"Alternatif sonuç {alt_outcome} kullanıldı, beklenen goller: {avg_home_goals:.2f}-{avg_away_goals:.2f}")
            else:
                # Herhangi bir uygun skor bulunamazsa, beklenen gollere göre oluştur
                rounded_home = round(avg_home_goals)
                rounded_away = round(avg_away_goals)
                score = f"{rounded_home}-{rounded_away}"
                top_scores_by_outcome = [(score, 1.0)]
                logger.info(f"Hiçbir uygun skor bulunamadı, beklenen gollere göre skor oluşturuldu: {score}")
        
        # Eğer en olası maç sonucu için skorlar varsa, bunları değerlendir
        if top_scores_by_outcome:
            logger.info(f"En olası maç sonucu {most_likely_outcome} için olası skorlar: {[(s, round(p*100, 2)) for s, p in top_scores_by_outcome[:3]]}")
        
        # Düşük gol beklentisinde (1'in altında) form durumuna göre karar ver
        # Ev sahibi takım için
        if avg_home_goals < 1.0:
            # Form ve ev sahibi avantajını değerlendir
            home_form_points = home_form.get('home_performance', {}).get('weighted_form_points', 0.3)
            
            # Son 3 maçtaki gol ortalamasına da bak
            recent_goals_avg = 0
            goals_in_last_matches = 0
            match_count = 0
            
            for match in home_match_data[:3]:
                if match.get('is_home', False):  # Sadece ev sahibi maçlarını dikkate al
                    goals_in_last_matches += match.get('goals_scored', 0)
                    match_count += 1
            
            if match_count > 0:
                recent_goals_avg = goals_in_last_matches / match_count
            
            # Form iyi, ev sahibi avantajı varsa veya son maçlarda gol attıysa
            if (home_form_points > 0.5 and home_advantage > 1.02) or weighted_home_form_points > 0.7 or recent_goals_avg > 0.7:
                # Form iyiyse veya son maçlarda gol attıysa daha yüksek gol olasılığı
                rounded_home_goals = 1
                logger.info(f"Ev sahibi gol beklentisi 1'in altında ({avg_home_goals:.2f}) ama form iyi veya son maçlarda gol ortalaması {recent_goals_avg:.2f}, 1 gol veriliyor")
            else:
                # Form kötüyse ve son maçlarda çok az gol attıysa daha düşük gol olasılığı
                rounded_home_goals = 0
                logger.info(f"Ev sahibi gol beklentisi 1'in altında ({avg_home_goals:.2f}), form düşük ve son maçlarda gol ortalaması {recent_goals_avg:.2f}, 0 gol veriliyor")
        else:
            # Direk yuvarlamak yerine form ve avantajları dikkate al
            # Yüksek gol beklentileri için özel durum (3'ün üzerinde)
            if avg_home_goals >= 3.0:
                # 3 ve üzeri beklentilerde
                if weighted_home_form_points > 0.5 or home_advantage > 1.05:
                    # Form veya ev avantajı iyiyse yukarı yuvarla veya ekstra gol ekle
                    base_goals = int(avg_home_goals)  # Tam kısmı al
                    fraction = avg_home_goals - base_goals  # Ondalık kısmı al
                    
                    if fraction >= 0.4:  # Yüksek ondalık kısım
                        rounded_home_goals = base_goals + 1
                        logger.info(f"Ev sahibi gol beklentisi {avg_home_goals:.2f} (yüksek), form iyi, {rounded_home_goals} gol veriliyor")
                    else:
                        rounded_home_goals = base_goals  # Tam değeri kullan
                        logger.info(f"Ev sahibi gol beklentisi {avg_home_goals:.2f} (yüksek), {rounded_home_goals} gol veriliyor")
                else:
                    # Form düşükse de daha hassas yuvarla
                    # 1.7'den büyük değerleri yukarı yuvarlıyoruz çünkü 2 gol daha olası
                    if avg_home_goals >= 1.7:
                        rounded_home_goals = int(avg_home_goals) + 1  # Yukarı yuvarla
                    else:
                        rounded_home_goals = int(avg_home_goals)  # Aşağı yuvarla
                    logger.info(f"Ev sahibi gol beklentisi {avg_home_goals:.2f} (yüksek), hassas yuvarlamayla {rounded_home_goals} gol veriliyor")
            elif avg_home_goals > 2.5 and avg_home_goals < 3.0:
                # 2.5-3.0 arası değerlerde
                if weighted_home_form_points > 0.6 or home_advantage > 1.1:
                    # Form veya ev avantajı iyiyse 3'e yuvarla
                    rounded_home_goals = 3
                    logger.info(f"Ev sahibi gol beklentisi 2.5-3.0 arasında ({avg_home_goals:.2f}) ve form iyi, 3 gol veriliyor")
                else:
                    # Yoksa 2'ye yuvarla
                    rounded_home_goals = 2
                    logger.info(f"Ev sahibi gol beklentisi 2.5-3.0 arasında ({avg_home_goals:.2f}) ama form düşük, 2 gol veriliyor")
            elif avg_home_goals > 1.5 and avg_home_goals < 1.7:
                # 1.5-1.7 arası değerlerde
                if weighted_home_form_points > 0.5 or home_advantage > 1.05:
                    # Form veya ev avantajı iyiyse 2'ye yuvarla
                    rounded_home_goals = 2
                    logger.info(f"Ev sahibi gol beklentisi 1.5-1.7 arasında ({avg_home_goals:.2f}) ve form iyi, 2 gol veriliyor")
                else:
                    # Yoksa 1'e yuvarla
                    rounded_home_goals = 1
                    logger.info(f"Ev sahibi gol beklentisi 1.5-1.7 arasında ({avg_home_goals:.2f}) ama form düşük, 1 gol veriliyor")
            # 1.7 ve üstü değerleri için daha hassas yuvarlama - sorunumuzu çözen kısım
            elif avg_home_goals >= 1.7 and avg_home_goals < 3.0:
                # Tam sayıya uzaklığı hesapla
                decimal_part = avg_home_goals - int(avg_home_goals)
                
                # 0.3'ten büyükse bir üst sayıya yuvarla (1.7 ve üzeri değerler daha agresif yuvarlanacak)
                if decimal_part >= 0.3:
                    rounded_home_goals = int(avg_home_goals) + 1
                    logger.info(f"Ev sahibi gol beklentisi {avg_home_goals:.2f}, {decimal_part:.2f} ondalık kısmı 0.3'ten büyük olduğu için {rounded_home_goals} olarak yuvarlandı")
                else:
                    rounded_home_goals = int(avg_home_goals)
                    logger.info(f"Ev sahibi gol beklentisi {avg_home_goals:.2f}, {decimal_part:.2f} ondalık kısmı 0.3'ten küçük olduğu için {rounded_home_goals} olarak yuvarlandı")
            
            elif avg_home_goals >= 2.0 and avg_home_goals <= 2.5:
                # 2'nin üstündeki değerler için daha yüksek ihtimal ile 2'ye yuvarla
                if avg_home_goals >= 2.25 or weighted_home_form_points > 0.5:
                    rounded_home_goals = 2
                    logger.info(f"Ev sahibi gol beklentisi 2.0-2.5 arasında ({avg_home_goals:.2f}), 2 gol veriliyor")
                else:
                    # 2.0'a yakın veya form düşükse daha agresif yuvarla
                    rounded_home_goals = 2
                    logger.info(f"Ev sahibi gol beklentisi 2.0-2.5 arasında ({avg_home_goals:.2f}), form hesaba katılarak 2 gol veriliyor")
            
            # 1.0-1.5 arası değerlerde de form bazlı karar ver
            elif avg_home_goals >= 1.0 and avg_home_goals <= 1.5:
                # Eğer 1.35'ten büyükse veya form iyiyse 2'ye yükselterek yuvarla
                if avg_home_goals >= 1.35 or weighted_home_form_points > 0.6:
                    rounded_home_goals = 2
                    logger.info(f"Ev sahibi gol beklentisi 1.0-1.5 arasında ({avg_home_goals:.2f}) ama 1.35+ veya form iyi, 2 gol veriliyor")
                else:
                    # Aksi halde 1'e yuvarla
                    rounded_home_goals = 1
                    logger.info(f"Ev sahibi gol beklentisi 1.0-1.5 arasında ({avg_home_goals:.2f}), 1 gol veriliyor")
            
            # Diğer değerlerde standart yuvarla
            else:
                rounded_home_goals = int(round(avg_home_goals))
                logger.info(f"Ev sahibi gol beklentisi standart yuvarlandı: {avg_home_goals:.2f} -> {rounded_home_goals}")
                
        # Deplasman takımı için
        if avg_away_goals < 1.0:
            # Form ve deplasman performansını değerlendir
            away_form_points = away_form.get('away_performance', {}).get('weighted_form_points', 0.3)
            
            # Son 3 maçtaki gol ortalamasına da bak
            recent_goals_avg = 0
            goals_in_last_matches = 0
            match_count = 0
            
            for match in away_match_data[:3]:
                if not match.get('is_home', True):  # Sadece deplasman maçlarını dikkate al
                    goals_in_last_matches += match.get('goals_scored', 0)
                    match_count += 1
            
            if match_count > 0:
                recent_goals_avg = goals_in_last_matches / match_count
                
            # Son 5 maçta ev sahibine karşı gol atma oranı
            goals_vs_similar = []
            for match in away_match_data[:10]:
                opponent_home_form = None
                opponent_id = None
                if not match.get('is_home', True) and match.get('goals_scored', 0) > 0:
                    # Benzer güçte rakiplere karşı gol atma durumu
                    goals_vs_similar.append(match.get('goals_scored', 0))
            
            avg_vs_similar = sum(goals_vs_similar) / len(goals_vs_similar) if goals_vs_similar else 0
            
            # Form iyi, deplasman avantajı varsa veya son maçlarda gol attıysa
            # Son maç ortalamalarını hesapla
            away_recent_goals_avg = away_recent_avg_goals.get(5, 0)
            if (away_form_points > 0.6 and away_advantage > 1.05) or weighted_away_form_points > 0.8 or away_recent_goals_avg > 0.7 or avg_vs_similar > 0.8:
                # Form iyiyse veya benzer rakiplere karşı gol attıysa daha yüksek gol olasılığı
                rounded_away_goals = 1
                logger.info(f"Deplasman gol beklentisi 1'in altında ({avg_away_goals:.2f}) ama form iyi veya son maçlarda gol ortalaması {away_recent_goals_avg:.2f}, benzer rakiplere karşı {avg_vs_similar:.2f}, 1 gol veriliyor")
            else:
                # Form kötüyse ve son maçlarda çok az gol attıysa daha düşük gol olasılığı
                rounded_away_goals = 0
                # Burada da değişken güncelleme
                away_recent_goals_avg = away_recent_avg_goals.get(5, 0)
                logger.info(f"Deplasman gol beklentisi 1'in altında ({avg_away_goals:.2f}), form düşük ve son maçlarda gol ortalaması {away_recent_goals_avg:.2f}, 0 gol veriliyor")
        else:
            # Değiştirilmiş yuvarlama mantığı - beklenen gol sayısını daha doğru yansıtmak için
            # 3 ve üzeri beklentiler için özel durum
            if avg_away_goals >= 3.0:
                # 3 ve üzeri beklentilerde
                if weighted_away_form_points > 0.5 or away_advantage > 1.05:
                    # Form veya deplasman avantajı iyiyse yukarı yuvarla veya ekstra gol ekle
                    base_goals = int(avg_away_goals)  # Tam kısmı al
                    fraction = avg_away_goals - base_goals  # Ondalık kısmı al
                    
                    if fraction >= 0.4:  # Yüksek ondalık kısım
                        rounded_away_goals = base_goals + 1
                        logger.info(f"Deplasman gol beklentisi {avg_away_goals:.2f} (yüksek), form iyi, {rounded_away_goals} gol veriliyor")
                    else:
                        rounded_away_goals = base_goals  # Tam değeri kullan
                        logger.info(f"Deplasman gol beklentisi {avg_away_goals:.2f} (yüksek), {rounded_away_goals} gol veriliyor")
                else:
                    # Form düşükse de daha hassas yuvarla
                    # 1.7'den büyük değerleri yukarı yuvarlıyoruz çünkü 2 gol daha olası
                    if avg_away_goals >= 1.7:
                        rounded_away_goals = int(avg_away_goals) + 1  # Yukarı yuvarla
                    else:
                        rounded_away_goals = int(avg_away_goals)  # Aşağı yuvarla
                    logger.info(f"Deplasman gol beklentisi {avg_away_goals:.2f} (yüksek), hassas yuvarlamayla {rounded_away_goals} gol veriliyor")
            # 1.8 ve üstü değerleri için daha hassas yuvarlama
            elif avg_away_goals >= 1.8 and avg_away_goals < 3.0:
                # Tam sayıya uzaklığı hesapla
                decimal_part = avg_away_goals - int(avg_away_goals)
                
                # 0.35'ten büyükse bir üst sayıya yuvarla
                if decimal_part >= 0.35:
                    rounded_away_goals = int(avg_away_goals) + 1
                    logger.info(f"Deplasman gol beklentisi {avg_away_goals:.2f}, {decimal_part:.2f} ondalık kısmı 0.35'ten büyük olduğu için {rounded_away_goals} olarak yuvarlandı")
                else:
                    rounded_away_goals = int(avg_away_goals)
                    logger.info(f"Deplasman gol beklentisi {avg_away_goals:.2f}, {decimal_part:.2f} ondalık kısmı 0.35'ten küçük olduğu için {rounded_away_goals} olarak yuvarlandı")
            # 1.5-1.8 arası değerlerde form durumuna göre karar ver
            elif avg_away_goals > 1.5 and avg_away_goals < 1.8:
                # Form faktörü ve son maç ortalamalarını kullan
                away_recent_goals_avg = away_recent_avg_goals.get(5, 0)
                if weighted_away_form_points > 0.55 or away_advantage > 1.1 or away_recent_goals_avg > 1.5:
                    # Form iyi veya son maçlarda gol ortalaması yüksekse 2'ye yuvarla
                    rounded_away_goals = 2
                    logger.info(f"Deplasman gol beklentisi 1.5-1.8 arasında ({avg_away_goals:.2f}) ve form iyi, 2 gol veriliyor")
                else:
                    # Form zayıfsa 1'e yuvarla
                    rounded_away_goals = 1
                    logger.info(f"Deplasman gol beklentisi 1.5-1.8 arasında ({avg_away_goals:.2f}) ama form düşük, 1 gol veriliyor")
            else:
                # Diğer değerlerde daha hassas yuvarlama işlemi kullan
                # 1.7'den büyük değerleri yukarı yuvarlıyoruz
                if avg_away_goals >= 0.7:
                    decimal_part = avg_away_goals - int(avg_away_goals)
                    if decimal_part >= 0.7:
                        rounded_away_goals = int(avg_away_goals) + 1  # Yukarı yuvarla
                    else:
                        rounded_away_goals = int(avg_away_goals)  # Aşağı yuvarla
                else:
                    rounded_away_goals = round(avg_away_goals)
                logger.info(f"Deplasman gol beklentisi {avg_away_goals:.2f}, geliştirilmiş yuvarlamayla {rounded_away_goals} olarak belirlendi")
                
        expected_score = f"{rounded_home_goals}-{rounded_away_goals}"
        
        # Önce beklenen skoru kullan, ancak olasılık çok düşükse top 5 içinden seç
        expected_score_prob = exact_scores.get(expected_score, 0) / simulations
        logger.info(f"Beklenen skor {expected_score} olasılığı: %{round(expected_score_prob*100, 2)}")
        
        # Beklenen skoru birkaç farklı yöntemle değerlendir
        expected_score_methods = {
            "rounded_mean": expected_score,  # Beklenen gollerin yuvarlanması
            # Monte Carlo'dan en yüksek olasılıklı skor, ama beklenen gol değerlerine göre makul sınırlardaki bir skoru seç
            "simulation_top": self._select_reasonable_score_from_simulation(top_5_scores, avg_home_goals, avg_away_goals) if top_5_scores else expected_score,
            "outcome_based": "" # En olası maç sonucuna göre en olası skor (aşağıda doldurulacak)
        }
        
        # Maç sonucu tahminini kullanarak skor belirle
        if most_likely_outcome == "HOME_WIN":
            outcome_scores = sorted(same_outcome_scores["HOME_WIN"].items(), key=lambda x: x[1], reverse=True)
            expected_score_methods["outcome_based"] = outcome_scores[0][0] if outcome_scores else expected_score
        elif most_likely_outcome == "DRAW":
            outcome_scores = sorted(same_outcome_scores["DRAW"].items(), key=lambda x: x[1], reverse=True)
            expected_score_methods["outcome_based"] = outcome_scores[0][0] if outcome_scores else expected_score
        else:  # AWAY_WIN
            outcome_scores = sorted(same_outcome_scores["AWAY_WIN"].items(), key=lambda x: x[1], reverse=True)
            expected_score_methods["outcome_based"] = outcome_scores[0][0] if outcome_scores else expected_score
            
        # Takımların gücünü değerlendirerek skor seçme stratejisini belirle
        home_strength = weighted_home_form_points
        away_strength = weighted_away_form_points
        is_balanced_match = abs(home_strength - away_strength) < 0.2
        is_high_scoring_home = home_recent_avg_goals.get(5, 0) > 1.8
        is_high_scoring_away = away_recent_avg_goals.get(5, 0) > 1.5
        
        # Varsayılan olarak outcome_based yaklaşımı kullan
        most_likely_score = expected_score_methods["outcome_based"]
        
        # Özel durumları ele al
        if is_balanced_match:
            # Dengeli maçlarda simulasyon sonuçlarına daha fazla güven
            if expected_score_prob < 0.07:  # Beklenen skor olasılığı düşükse
                simulation_score = expected_score_methods["simulation_top"]
                simulation_prob = score_histogram.get(simulation_score, 0)
                
                if simulation_prob > expected_score_prob * 1.5:
                    most_likely_score = simulation_score
                    logger.info(f"Dengeli maç, simulasyon tahmini tercih edildi: {simulation_score} (olasılık: %{round(simulation_prob*100, 2)})")
                else:
                    most_likely_score = expected_score_methods["outcome_based"]
                    logger.info(f"Dengeli maç, maç sonucu bazlı tahmin tercih edildi: {most_likely_score}")
        elif is_high_scoring_home and is_high_scoring_away:
            # İki takım da çok gol atıyorsa, daha yüksek skorlu bir tahmin yap
            high_scoring_candidates = []
            for score, prob in score_histogram.items():
                if '-' in score:
                    h, a = map(int, score.split('-'))
                    if h + a >= 3 and prob > 0.05:  # En az 3 gol ve %5 üzeri olasılık
                        high_scoring_candidates.append((score, prob))
            
            if high_scoring_candidates:
                high_scoring_candidates.sort(key=lambda x: x[1], reverse=True)
                most_likely_score = high_scoring_candidates[0][0]
                logger.info(f"Yüksek skorlu maç beklentisi: {most_likely_score} seçildi (olasılık: %{round(high_scoring_candidates[0][1]*100, 2)})")
            else:
                # Yeterli aday yoksa, beklenen skoru kullan
                most_likely_score = expected_score
        else:
            # Takımların gücüne ve maç sonucu tahminlerine dayalı skorları değerlendir
            outcome_score = expected_score_methods["outcome_based"]
            outcome_prob = score_histogram.get(outcome_score, 0)
            
            if outcome_prob > expected_score_prob or outcome_prob > 0.08:  # %8'den yüksek olasılık
                most_likely_score = outcome_score
                logger.info(f"Maç sonucu bazlı tahmin tercih edildi: {outcome_score} (olasılık: %{round(outcome_prob*100, 2)})")
            else:
                # Beklenen goller daha güvenilir, beklenen skoru kullan
                most_likely_score = expected_score
                logger.info(f"Beklenen gollere dayanarak {expected_score} skoru seçildi (olasılık: %{round(expected_score_prob*100, 2)})")
        
        # Özel durum: Eğer goller çok yakınsa ve beraberlik olasılığı yüksekse, beraberlik skorunu değerlendir
        if abs(avg_home_goals - avg_away_goals) < 0.3 and draw_prob > 0.25:
            # Beklenen gol değerlerine göre uygun beraberlik skorunu seç
            # Düşük gol beklentili maçlarda Monte Carlo simülasyon sonuçlarına doğrudan saygı göster
            if avg_home_goals < 1.0 and avg_away_goals < 1.0:
                # İki takım da düşük gol beklentisine sahipse en olası beraberlik skorunu Monte Carlo'dan al
                logger.info(f"Düşük gol beklentili maç için beraberlik skorunu kontrol ediyorum")
                logger.info(f"exact_scores içeriği: {exact_scores}")
                
                # En olası 5 skoru loglama
                top_scores = sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                logger.info(f"En olası 5 skor: {top_scores}")
                
                score_0_0_count = exact_scores.get("0-0", 0)
                score_1_1_count = exact_scores.get("1-1", 0)
                draw_score_0_0_prob = score_0_0_count / simulations if simulations > 0 else 0
                draw_score_1_1_prob = score_1_1_count / simulations if simulations > 0 else 0
                
                logger.info(f"0-0 sayısı: {score_0_0_count}, 1-1 sayısı: {score_1_1_count}, simulations: {simulations}")
                logger.info(f"0-0 olasılığı: {draw_score_0_0_prob}, 1-1 olasılığı: {draw_score_1_1_prob}")
                
                # Ortalama gol değerleri 1'in altındayken, eğer değerler 0.7'den büyükse 1-1'e daha yüksek şans ver
                # Monte Carlo bazen 0-0 veya 1-1 skorları için değer oluşturmuyor olabilir ama skorun mantıklı olması gerekiyor
                if avg_home_goals >= 0.7 and avg_away_goals >= 0.7:
                    logger.info(f"Gol beklentileri 0.7'den yüksek, 1-1 skoru makul bir seçim olabilir")
                    
                    # Monte Carlo değer üretmediyse veya değer çok düşükse, değerleri zorla
                    if score_0_0_count == 0:
                        score_0_0_count = int(simulations * 0.05)  # %5 varsayılan değer
                        logger.info(f"0-0 için Monte Carlo değeri bulunamadı, varsayılan değer eklendi: {score_0_0_count}")
                    
                    if score_1_1_count == 0:
                        score_1_1_count = int(simulations * 0.10)  # %10 varsayılan değer
                        logger.info(f"1-1 için Monte Carlo değeri bulunamadı, varsayılan değer eklendi: {score_1_1_count}")
                    
                    # Değerleri yeniden hesapla
                    draw_score_0_0_prob = score_0_0_count / simulations if simulations > 0 else 0
                    draw_score_1_1_prob = score_1_1_count / simulations if simulations > 0 else 0
                    
                    # Gol beklentileri 0.7'den büyükse, 1-1'e daha yüksek şans ver
                    if score_1_1_count > 0 and draw_score_1_1_prob > draw_score_0_0_prob * 0.8:
                        most_likely_score = "1-1"
                        logger.info(f"Monte Carlo simülasyonu ve gol beklentileri göz önüne alınarak 1-1 skoru seçildi (olasılık: %{round(draw_score_1_1_prob*100, 2)})")
                        # Akışı kesmiyoruz, sadece değeri güncelliyoruz
                
                # Monte Carlo simülasyonunun sonucuna doğrudan saygı göster - hangi skor daha olasıysa onu seç
                if draw_score_0_0_prob >= draw_score_1_1_prob:
                    most_likely_score = "0-0"
                    logger.info(f"Monte Carlo simülasyonu sonucuna göre düşük gol beklentilerinde ({avg_home_goals:.2f} vs {avg_away_goals:.2f}) 0-0 skoru seçildi (olasılık: %{round(draw_score_0_0_prob*100, 2)})")
                else:
                    most_likely_score = "1-1"
                    logger.info(f"Monte Carlo simülasyonu sonucuna göre düşük gol beklentilerinde ({avg_home_goals:.2f} vs {avg_away_goals:.2f}) 1-1 skoru seçildi (olasılık: %{round(draw_score_1_1_prob*100, 2)})")
            else:
                # Normal gol beklentileri için standart değerlendirme
                likely_draw_score = int(round((avg_home_goals + avg_away_goals) / 2))
                draw_score = f"{likely_draw_score}-{likely_draw_score}"
                draw_score_prob = exact_scores.get(draw_score, 0) / simulations
                
                if draw_score_prob > 0.1:  # %10'dan yüksek olasılık
                    most_likely_score = draw_score
                    logger.info(f"Gol beklentileri çok yakın ({avg_home_goals:.2f} vs {avg_away_goals:.2f}) ve beraberlik olasılığı yüksek (%{round(draw_prob*100, 2)}), {draw_score} skoru seçildi")
        
        # GERÇEKÇI SKORLAR IÇIN GERÇEKLİK KONTROLÜ - Skor çok abartılı ise beklenen gollere göre düzeltme yap
        home_score, away_score = map(int, most_likely_score.split('-'))
        
        # DAHA SIKI GERÇEKLİK KONTROLÜ - Son maçlardaki gerçek gol ortalamaları
        recent_home_goals = 0
        recent_away_goals = 0
        home_matches_count = min(5, len(home_match_data))
        away_matches_count = min(5, len(away_match_data))
        
        if home_matches_count > 0:
            recent_home_goals = sum(match.get('goals_scored', 0) for match in home_match_data[:home_matches_count]) / home_matches_count
        
        if away_matches_count > 0:
            recent_away_goals = sum(match.get('goals_scored', 0) for match in away_match_data[:away_matches_count]) / away_matches_count
            
        # Beklenen gollere göre makul skor sınırları - iyileştirilmiş algoritma
        # Düşük beklenen gol değerleri için daha keskin sınırlar kullan
        # Beklenen goller 1'in altındaysa, makul skor maksimum 1 olmalı, 1-2 arasındaysa maksimum 2 olmalı
        if avg_home_goals < 1.0:
            reasonable_home_max = min(1, round(max(avg_home_goals, recent_home_goals)))
        elif avg_home_goals < 2.0:
            reasonable_home_max = min(2, round(max(avg_home_goals * 1.2, recent_home_goals * 1.2)))
        else:
            reasonable_home_max = min(3, round(max(avg_home_goals * 1.3, recent_home_goals * 1.3)))
            
        if avg_away_goals < 1.0:
            reasonable_away_max = min(1, round(max(avg_away_goals, recent_away_goals)))
        elif avg_away_goals < 1.8:
            reasonable_away_max = min(2, round(max(avg_away_goals * 1.2, recent_away_goals * 1.2)))
        elif avg_away_goals < 2.5:
            reasonable_away_max = min(3, round(max(avg_away_goals * 1.2, recent_away_goals * 1.2)))
        else:
            reasonable_away_max = min(3, round(max(avg_away_goals * 1.3, recent_away_goals * 1.3)))
        
        logger.info(f"Makul skor sınırları: Ev={reasonable_home_max}, Deplasman={reasonable_away_max} (son maç gol ort: {recent_home_goals:.2f}-{recent_away_goals:.2f})")
        
        # Gerçekçilik kontrolü - Eğer skor çok abartılı ise düzelt
        if home_score > reasonable_home_max or away_score > reasonable_away_max:
            logger.warning(f"Seçilen skor {most_likely_score} çok abartılı! Beklenen goller: {avg_home_goals:.2f}-{avg_away_goals:.2f}, Son maç gol ort: {recent_home_goals:.2f}-{recent_away_goals:.2f}")
            
            # Beklenen gollere ve son maçlardaki ortalamalara göre daha gerçekçi bir skor belirle
            reasonable_home = min(home_score, reasonable_home_max)
            reasonable_away = min(away_score, reasonable_away_max)
            
            # Sonuç tipini koru (ev sahibi kazanır, deplasman kazanır, beraberlik)
            if home_score > away_score:  # Ev sahibi galibiyet
                if reasonable_home <= reasonable_away:
                    reasonable_home = reasonable_away + 1
            elif away_score > home_score:  # Deplasman galibiyet
                if reasonable_away <= reasonable_home:
                    reasonable_away = reasonable_home + 1
            else:  # Beraberlik
                reasonable_home = reasonable_away = min(reasonable_home, reasonable_away)
            
            adjusted_score = f"{reasonable_home}-{reasonable_away}"
            logger.info(f"Skor beklenen gollere ve son maç ortalamalarına göre düzeltildi: {most_likely_score} -> {adjusted_score}")
            most_likely_score = adjusted_score
        
        # ÖNEMLİ: Tutarlılık kontrolü! En olası sonuçla kesin skor tutarlı mı?
        # Beklenen golleri de hesaba katarak en olası sonucu belirle
        most_likely_outcome = self._get_most_likely_outcome(home_win_prob, draw_prob, away_win_prob, avg_home_goals, avg_away_goals)
        score_outcome = self._get_outcome_from_score(most_likely_score)
        
        # Eğer kesin skor ve tahmin edilen sonuç tutarsızsa, skoru düzelt
        if most_likely_outcome != score_outcome:
            logger.warning(f"Tutarsızlık tespit edildi! Maç sonucu {most_likely_outcome} ama skor tahmini {most_likely_score} ({score_outcome})")
            
            # Beklenen gol değerlerine göre makul skor sınırları belirle
            max_home_score = 1  # Varsayılan
            max_away_score = 1  # Varsayılan
            
            # Ev sahibi için sınır
            if avg_home_goals < 1.0:
                max_home_score = 1  # 1'in altında beklenen gol için maksimum 1 gol
            elif avg_home_goals < 1.8:
                max_home_score = 2  # 1-1.8 arası beklenen gol için maksimum 2 gol
            elif avg_home_goals < 2.5:
                max_home_score = 3  # 1.8-2.5 arası beklenen gol için maksimum 3 gol
            else:
                max_home_score = 4  # 2.5'ten yüksek beklenen gol için maksimum 4 gol
                
            # Deplasman için sınır
            if avg_away_goals < 1.0:
                max_away_score = 1  # 1'in altında beklenen gol için maksimum 1 gol
            elif avg_away_goals < 1.8:
                max_away_score = 2  # 1-1.8 arası beklenen gol için maksimum 2 gol
            elif avg_away_goals < 2.5:
                max_away_score = 3  # 1.8-2.5 arası beklenen gol için maksimum 3 gol
            else:
                max_away_score = 4  # 2.5'ten yüksek beklenen gol için maksimum 4 gol
            
            # En yüksek olasılıklı skoru bul ancak sonuç kısıtlaması ve makul skor sınırları ile
            candidate_scores = []
            
            if most_likely_outcome == "HOME_WIN":
                for score, count in sorted(exact_scores.items(), key=lambda x: x[1], reverse=True):
                    if '-' in score:
                        home, away = map(int, score.split('-'))
                        if home > away and home <= max_home_score and away <= max_away_score:  # Ev sahibi galibiyeti + makul sınırlar
                            candidate_scores.append((score, count))
                
                if candidate_scores:
                    # En yüksek olasılıklı makul skoru seç
                    most_likely_score = candidate_scores[0][0]
                    logger.info(f"Tutarlılık için skor {most_likely_score} olarak güncellendi (Ev galibiyeti), makul sınırlar: {max_home_score}-{max_away_score}")
                else:
                    # Makul sınırlarda bir ev galibiyet skoru oluştur
                    new_home_score = min(2, max_home_score)  # En az 1, en fazla makul sınır
                    new_away_score = min(1, max_away_score)  # En az 0, en fazla makul sınır
                    if new_home_score <= new_away_score:  # Ev sahibi kazanmalı
                        new_home_score = new_away_score + 1
                    most_likely_score = f"{new_home_score}-{new_away_score}"
                    logger.warning(f"Makul sınırlarda bir ev sahibi galibiyet skoru bulunamadı, yeni skor oluşturuldu: {most_likely_score}")
            
            elif most_likely_outcome == "AWAY_WIN":
                for score, count in sorted(exact_scores.items(), key=lambda x: x[1], reverse=True):
                    if '-' in score:
                        home, away = map(int, score.split('-'))
                        if home < away and home <= max_home_score and away <= max_away_score:  # Deplasman galibiyeti + makul sınırlar
                            candidate_scores.append((score, count))
                
                if candidate_scores:
                    # En yüksek olasılıklı makul skoru seç
                    most_likely_score = candidate_scores[0][0]
                    logger.info(f"Tutarlılık için skor {most_likely_score} olarak güncellendi (Deplasman galibiyeti), makul sınırlar: {max_home_score}-{max_away_score}")
                else:
                    # Makul sınırlarda bir deplasman galibiyet skoru oluştur
                    new_home_score = min(1, max_home_score)  # En az 0, en fazla makul sınır
                    new_away_score = min(2, max_away_score)  # En az 1, en fazla makul sınır
                    if new_home_score >= new_away_score:  # Deplasman kazanmalı
                        new_away_score = new_home_score + 1
                    most_likely_score = f"{new_home_score}-{new_away_score}"
                    logger.warning(f"Makul sınırlarda bir deplasman galibiyet skoru bulunamadı, yeni skor oluşturuldu: {most_likely_score}")
            
            elif most_likely_outcome == "DRAW":
                for score, count in sorted(exact_scores.items(), key=lambda x: x[1], reverse=True):
                    if '-' in score:
                        home, away = map(int, score.split('-'))
                        if home == away and home <= max_home_score:  # Beraberlik + makul sınırlar
                            candidate_scores.append((score, count))
                
                if candidate_scores:
                    # En yüksek olasılıklı makul skoru seç
                    most_likely_score = candidate_scores[0][0]
                    logger.info(f"Tutarlılık için skor {most_likely_score} olarak güncellendi (Beraberlik), makul sınırlar: {max_home_score}-{max_away_score}")
                else:
                    # Makul sınırlarda bir beraberlik skoru oluştur
                    new_score = min(1, max_home_score)  # En az 0, en fazla makul sınır
                    most_likely_score = f"{new_score}-{new_score}"
                    logger.warning(f"Makul sınırlarda bir beraberlik skoru bulunamadı, yeni skor oluşturuldu: {most_likely_score}")
        
        # Kesin skoru belirle
        bet_predictions['exact_score'] = most_likely_score
        
        # Kesin skor üzerinden maç sonucu, ÜST/ALT ve KG VAR/YOK tahminlerini güncelle
        score_parts = most_likely_score.split('-')
        if len(score_parts) == 2:
            home_score, away_score = int(score_parts[0]), int(score_parts[1])
            total_goals = home_score + away_score
            
            # Gol beklentileri ile skorlar arasındaki uyumu kontrol et
            home_score_diff = abs(home_score - avg_home_goals)
            away_score_diff = abs(away_score - avg_away_goals)
            
            # Skor beklentiler ile uyumlu değilse uyarı logla
            if home_score_diff > 1.0 or away_score_diff > 1.0:
                logger.warning(f"Seçilen skor {most_likely_score} ile gol beklentileri {avg_home_goals:.2f}-{avg_away_goals:.2f} arasında büyük fark var!")
            else:
                logger.info(f"Seçilen skor {most_likely_score} gol beklentilerine {avg_home_goals:.2f}-{avg_away_goals:.2f} yakın ve uyumlu.")
            
            # Maç sonucu - Önce beklenen golleri karşılaştır, sonra skora göre belirle
            # Beklenen gol sayıları arasındaki fark
            expected_goal_diff = avg_home_goals - avg_away_goals
            
            # Skor ile beklenen golleri karşılaştır
            if home_score > away_score:
                # Beklenen gollere göre kontrol et
                if expected_goal_diff >= 0.2:  # Ev sahibinin beklenen golü en az 0.2 fazlaysa makul
                    bet_predictions['match_result'] = 'HOME_WIN'
                    logger.info(f"Ev galibiyet tahmini beklenen gollerle ({expected_goal_diff:.2f}) uyumlu")
                elif expected_goal_diff <= -0.3:  # Deplasman beklenen golleri çok daha yüksekse uyumsuzluk var
                    # Beklenen gol farkı büyükse ve deplasman lehine ise, skor tutarsız olabilir
                    if abs(expected_goal_diff) > 0.5:
                        # Skoru düzeltmeyi değerlendir
                        logger.warning(f"Tutarsız tahmin: Ev galibiyeti skoru ({home_score}-{away_score}) beklenen gollerle ({avg_home_goals:.2f}-{avg_away_goals:.2f}) uyumsuz")
                        
                        # Gerçek beklenen gollere göre düzeltilmiş skor
                        home_score = rounded_home_goals
                        away_score = rounded_away_goals
                        
                        # Düzeltilmiş skora göre sonucu belirle
                        if home_score > away_score:
                            bet_predictions['match_result'] = 'HOME_WIN'
                        elif away_score > home_score:
                            bet_predictions['match_result'] = 'AWAY_WIN'
                        else:
                            bet_predictions['match_result'] = 'DRAW'
                            
                        bet_predictions['exact_score'] = f"{home_score}-{away_score}"
                        logger.info(f"Skor düzeltildi: {bet_predictions['exact_score']} (beklenen goller temel alındı)")
                    else:
                        bet_predictions['match_result'] = 'HOME_WIN'
                else:
                    bet_predictions['match_result'] = 'HOME_WIN'
                
                # Skor farkına göre olasılığı belirle
                goal_diff = home_score - away_score
                
                # Olasılığı beklenen gol farkıyla dengele
                base_prob = 0.5 + (expected_goal_diff * 0.15)  # Beklenen gol farkı artıkça olasılık artar
                
                # Gerçekçi olasılık ayarlaması
                if goal_diff == 1:  # Minimal fark
                    home_win_prob = max(home_win_prob, min(0.65, base_prob))
                elif goal_diff == 2:  # Orta fark
                    home_win_prob = max(home_win_prob, min(0.75, base_prob + 0.1))
                else:  # Büyük fark
                    home_win_prob = max(home_win_prob, min(0.85, base_prob + 0.2))
                    
                # Diğer olasılıkları dengele
                remaining = 1.0 - home_win_prob
                draw_prob = remaining * 0.6
                away_win_prob = remaining * 0.4
                
            elif away_score > home_score:
                # Beklenen gollere göre kontrol et
                if expected_goal_diff <= -0.2:  # Deplasman beklenen golü en az 0.2 fazlaysa makul
                    bet_predictions['match_result'] = 'AWAY_WIN'
                    logger.info(f"Deplasman galibiyet tahmini beklenen gollerle ({-expected_goal_diff:.2f}) uyumlu")
                elif expected_goal_diff >= 0.3:  # Ev sahibi beklenen golleri çok daha yüksekse uyumsuzluk var
                    # Beklenen gol farkı büyükse ve ev sahibi lehine ise, skor tutarsız olabilir
                    if abs(expected_goal_diff) > 0.5:
                        # Skoru düzeltmeyi değerlendir
                        logger.warning(f"Tutarsız tahmin: Deplasman galibiyeti skoru ({home_score}-{away_score}) beklenen gollerle ({avg_home_goals:.2f}-{avg_away_goals:.2f}) uyumsuz")
                        
                        # Gerçek beklenen gollere göre düzeltilmiş skor
                        home_score = rounded_home_goals
                        away_score = rounded_away_goals
                        
                        # Düzeltilmiş skora göre sonucu belirle
                        if home_score > away_score:
                            bet_predictions['match_result'] = 'HOME_WIN'
                        elif away_score > home_score:
                            bet_predictions['match_result'] = 'AWAY_WIN'
                        else:
                            bet_predictions['match_result'] = 'DRAW'
                            
                        bet_predictions['exact_score'] = f"{home_score}-{away_score}"
                        logger.info(f"Skor düzeltildi: {bet_predictions['exact_score']} (beklenen goller temel alındı)")
                    else:
                        bet_predictions['match_result'] = 'AWAY_WIN'
                else:
                    bet_predictions['match_result'] = 'AWAY_WIN'
                
                # Skor farkına göre olasılığı belirle
                goal_diff = away_score - home_score
                
                # Olasılığı beklenen gol farkıyla dengele
                base_prob = 0.5 + (-expected_goal_diff * 0.15)  # Beklenen gol farkı artıkça olasılık artar
                
                # Gerçekçi olasılık ayarlaması
                if goal_diff == 1:  # Minimal fark
                    away_win_prob = max(away_win_prob, min(0.65, base_prob))
                elif goal_diff == 2:  # Orta fark
                    away_win_prob = max(away_win_prob, min(0.75, base_prob + 0.1))
                else:  # Büyük fark
                    away_win_prob = max(away_win_prob, min(0.85, base_prob + 0.2))
                    
                # Diğer olasılıkları dengele
                remaining = 1.0 - away_win_prob
                draw_prob = remaining * 0.6
                home_win_prob = remaining * 0.4
                
            else:
                bet_predictions['match_result'] = 'DRAW'
                
                # Beklenen gollere göre beraberlik olasılığını değerlendir
                if abs(expected_goal_diff) < 0.3:  # Beklenen goller çok yakınsa beraberlik mantıklı
                    logger.info(f"Beraberlik tahmini beklenen gollerle ({abs(expected_goal_diff):.2f} fark) uyumlu")
                else:
                    # Beklenen gol farkı büyükse, beraberlik tahmini şüpheli olabilir
                    logger.warning(f"Dikkat: Beraberlik skoru ({home_score}-{away_score}) beklenen gollerde önemli fark ({expected_goal_diff:.2f}) var")
                
                # Skor yüksekliğine göre olasılığı ayarla (yüksek skorlu beraberlikler daha nadir)
                if total_goals <= 2:  # 0-0, 1-1
                    draw_prob = max(draw_prob, min(0.65, 0.5 + 0.15 * (1 - abs(expected_goal_diff))))
                else:  # 2-2, 3-3, vs. 
                    draw_prob = max(draw_prob, min(0.55, 0.5 + 0.05 * (1 - abs(expected_goal_diff))))
                    
                # Diğer olasılıkları dengele
                remaining = 1.0 - draw_prob
                # Beklenen gol farkına göre kalan olasılıkları dağıt
                if expected_goal_diff > 0:
                    home_win_prob = remaining * 0.7
                    away_win_prob = remaining * 0.3
                else:
                    home_win_prob = remaining * 0.3
                    away_win_prob = remaining * 0.7
            
            # KG VAR/YOK - Daha hassas olasılık belirle
            if home_score > 0 and away_score > 0:
                bet_predictions['both_teams_to_score'] = 'KG VAR'
                
                # Geçmiş maçlardaki KG VAR/YOK oranlarını değerlendir
                kg_var_home_matches = sum(1 for match in home_match_data[:5] if match.get('goals_scored', 0) > 0 and match.get('goals_conceded', 0) > 0)
                kg_var_away_matches = sum(1 for match in away_match_data[:5] if match.get('goals_scored', 0) > 0 and match.get('goals_conceded', 0) > 0)
                kg_var_rate = (kg_var_home_matches + kg_var_away_matches) / 10 if len(home_match_data) >= 5 and len(away_match_data) >= 5 else 0.5
                
                # Gol beklentilerine ve geçmiş KG VAR/YOK oranlarına göre olasılığı ayarla
                if avg_home_goals > 1.5 and avg_away_goals > 1.0:
                    base_prob = 0.90  # Yüksek gol beklentisinde daha yüksek olasılık
                else:
                    base_prob = 0.85  # Kesin skordan biliyoruz
                    
                # Geçmiş maçlardaki KG VAR/YOK oranlarını da dikkate al
                kg_var_adjusted_prob = base_prob * 0.8 + kg_var_rate * 0.2
                logger.info(f"KG VAR tahmini: Geçmiş maçlarda KG VAR oranı: {kg_var_rate:.2f}, ayarlanmış olasılık: {kg_var_adjusted_prob:.2f}")
            else:
                bet_predictions['both_teams_to_score'] = 'KG YOK'
                
                # Geçmiş maçlardaki KG YOK oranlarını değerlendir
                kg_yok_home_matches = sum(1 for match in home_match_data[:5] if match.get('goals_scored', 0) == 0 or match.get('goals_conceded', 0) == 0)
                kg_yok_away_matches = sum(1 for match in away_match_data[:5] if match.get('goals_scored', 0) == 0 or match.get('goals_conceded', 0) == 0)
                kg_yok_rate = (kg_yok_home_matches + kg_yok_away_matches) / 10 if len(home_match_data) >= 5 and len(away_match_data) >= 5 else 0.5
                
                # Gol beklentilerine ve geçmiş KG YOK oranlarına göre olasılığı ayarla
                if avg_home_goals < 1.0 or avg_away_goals < 0.8:
                    base_prob = 0.10  # Düşük gol beklentisinde daha düşük KG VAR olasılığı
                else:
                    base_prob = 0.15  # Kesin skordan biliyoruz
                    
                # Geçmiş maçlardaki KG YOK oranlarını da dikkate al
                kg_var_adjusted_prob = base_prob * 0.8 + (1 - kg_yok_rate) * 0.2
                logger.info(f"KG YOK tahmini: Geçmiş maçlarda KG YOK oranı: {kg_yok_rate:.2f}, ayarlanmış KG VAR olasılığı: {kg_var_adjusted_prob:.2f}")
            
            # 2.5 ÜST/ALT - Daha hassas olasılık belirle
            if total_goals > 2.5:
                bet_predictions['over_2_5_goals'] = '2.5 ÜST'
                
                # Toplam gol beklentisine göre olasılığı ayarla
                expected_total = avg_home_goals + avg_away_goals
                if expected_total > 3.0:
                    over_25_adjusted_prob = 0.90  # Yüksek toplam gol beklentisinde daha yüksek olasılık
                else:
                    over_25_adjusted_prob = 0.85  # Kesin skordan biliyoruz
            else:
                bet_predictions['over_2_5_goals'] = '2.5 ALT'
                
                # Toplam gol beklentisine göre olasılığı ayarla
                expected_total = avg_home_goals + avg_away_goals
                if expected_total < 2.0:
                    over_25_adjusted_prob = 0.10  # Düşük toplam gol beklentisinde daha düşük olasılık
                else:
                    over_25_adjusted_prob = 0.15  # Kesin skordan biliyoruz
            
            # 3.5 ÜST/ALT - Daha hassas olasılık belirle
            if total_goals > 3.5:
                bet_predictions['over_3_5_goals'] = '3.5 ÜST'
                
                # Toplam gol beklentisine göre olasılığı ayarla
                expected_total = avg_home_goals + avg_away_goals
                if expected_total > 4.0:
                    over_35_adjusted_prob = 0.90  # Çok yüksek toplam gol beklentisinde daha yüksek olasılık
                else:
                    over_35_adjusted_prob = 0.85  # Kesin skordan biliyoruz
            else:
                bet_predictions['over_3_5_goals'] = '3.5 ALT'
                
                # Toplam gol beklentisine göre olasılığı ayarla
                expected_total = avg_home_goals + avg_away_goals
                if expected_total < 3.0:
                    over_35_adjusted_prob = 0.10  # Düşük toplam gol beklentisinde daha düşük olasılık
                else:
                    over_35_adjusted_prob = 0.15  # Kesin skordan biliyoruz
            
            # Kesin skordan doğrudan sonuç belirleme
            match_outcome_from_score = self._get_outcome_from_score(most_likely_score)
            
            # Kritik kontrol: Eşit skorlar için (1-1, 0-0 gibi) kesinlikle beraberlik sonucu olmalı
            # ve sistem genelinde tutarlılık sağlanmalı
            if '-' in str(most_likely_score):
                try:
                    h_goals, a_goals = map(int, str(most_likely_score).split('-'))
                    if h_goals == a_goals:
                        # Eşit skor kontrol edildi, mutlaka beraberlik olmalı
                        match_outcome_from_score = "DRAW"
                        logger.warning(f"EŞİT SKOR KONTROL: {most_likely_score} için sonuç DRAW olarak sabitlendi")
                except Exception as e:
                    logger.error(f"Skor kontrol hatası: {str(e)}")
            
            # Bu değerleri saklıyoruz, prediction değişkeni daha sonra tanımlandığında kullanmak için
            # prediction değişkeni 3854. satırda tanımlanıyor, öncesinde kullanmak hata verir
            saved_most_likely_score = most_likely_score
            saved_match_outcome = match_outcome_from_score
            
            logger.info(f"Kesin skor {most_likely_score} (gol beklentileri: {avg_home_goals:.2f}-{avg_away_goals:.2f}) esas alınarak tüm tahminler güncellendi")
        
        # Üçüncü adım: Maç sonucu ile diğer tahminler arasındaki tutarlılığı sağla
        match_result = bet_predictions['match_result']
        
        # 1. 2.5 ALT ve KG VAR arasındaki uyumsuzluk
        if bet_predictions['over_2_5_goals'] == '2.5 ALT' and bet_predictions['both_teams_to_score'] == 'KG VAR':
            # İstisnai durum: 1-1 skoru
            if bet_predictions['exact_score'] != '1-1':
                # Hangisinin olasılığı daha yüksekse ona göre düzelt
                if over_25_adjusted_prob > kg_var_adjusted_prob:
                    bet_predictions['both_teams_to_score'] = 'KG YOK'
                    logger.info("Mantık düzeltmesi: 2.5 ALT ve KG VAR uyumsuzluğu - KG YOK olarak güncellendi")
                    # Skoru da güncelle
                    if match_result == 'HOME_WIN':
                        bet_predictions['exact_score'] = '2-0'
                    elif match_result == 'AWAY_WIN':
                        bet_predictions['exact_score'] = '0-2'
                    else:  # DRAW
                        bet_predictions['exact_score'] = '0-0'
                else:
                    bet_predictions['over_2_5_goals'] = '2.5 ÜST'
                    logger.info("Mantık düzeltmesi: 2.5 ALT ve KG VAR uyumsuzluğu - 2.5 ÜST olarak güncellendi")
                    # Skoru da güncelle
                    if match_result == 'HOME_WIN':
                        bet_predictions['exact_score'] = '2-1'
                    elif match_result == 'AWAY_WIN':
                        bet_predictions['exact_score'] = '1-2'
                    else:  # DRAW
                        bet_predictions['exact_score'] = '1-1'
        
        # 2. MS1/MS2 ve KG YOK arasındaki uyumsuzluk - skor kontrolü
        if (match_result in ['HOME_WIN', 'AWAY_WIN']) and bet_predictions['both_teams_to_score'] == 'KG YOK':
            score_parts = bet_predictions['exact_score'].split('-')
            if len(score_parts) == 2:
                home_score, away_score = int(score_parts[0]), int(score_parts[1])
                if home_score > 0 and away_score > 0:
                    # Tutarsızlık var, düzelt
                    if match_result == 'HOME_WIN':
                        bet_predictions['exact_score'] = f"{home_score}-0"
                    else:  # AWAY_WIN
                        bet_predictions['exact_score'] = f"0-{away_score}"
                    logger.info(f"Mantık düzeltmesi: {match_result} ve KG YOK tutarsızlık - skor güncellendi")
        
        # 3. MS1/MS2 ve KG VAR arasındaki uyumsuzluk - skor kontrolü
        if (match_result in ['HOME_WIN', 'AWAY_WIN']) and bet_predictions['both_teams_to_score'] == 'KG VAR':
            score_parts = bet_predictions['exact_score'].split('-')
            if len(score_parts) == 2:
                home_score, away_score = int(score_parts[0]), int(score_parts[1])
                if home_score == 0 or away_score == 0:
                    # Tutarsızlık var, düzelt
                    if match_result == 'HOME_WIN':
                        bet_predictions['exact_score'] = '2-1'
                    else:  # AWAY_WIN
                        bet_predictions['exact_score'] = '1-2'
                    logger.info(f"Mantık düzeltmesi: {match_result} ve KG VAR tutarsızlık - skor güncellendi")
        
        # 4. DRAW ve skor uyumsuzluğu
        if match_result == 'DRAW':
            score_parts = bet_predictions['exact_score'].split('-')
            if len(score_parts) == 2:
                home_score, away_score = int(score_parts[0]), int(score_parts[1])
                if home_score != away_score:
                    # Beraberlik skoru değil, düzelt
                    if bet_predictions['both_teams_to_score'] == 'KG VAR':
                        bet_predictions['exact_score'] = '1-1'
                    else:
                        bet_predictions['exact_score'] = '0-0'
                    logger.info("Mantık düzeltmesi: DRAW ve skor uyumsuzluğu - skor güncellendi")
        
        # 5. İlk yarı/maç sonu düzeltmesi
        # Kesin skor üzerinden en olası ilk yarı skorunu tahmin et
        score_parts = bet_predictions['exact_score'].split('-')
        if len(score_parts) == 2:
            home_score, away_score = int(score_parts[0]), int(score_parts[1])
            # İlk yarıda genellikle toplam gollerin %40'ı atılır
            expected_ht_home = round(home_score * 0.4)
            expected_ht_away = round(away_score * 0.4)
            
            # İlk yarı sonucu
            ht_result = "X"
            if expected_ht_home > expected_ht_away:
                ht_result = "MS1"
            elif expected_ht_away > expected_ht_home:
                ht_result = "MS2"
            
            # Maç sonu sonucu
            ft_result = bet_predictions['match_result']
            
            # İlk yarı/maç sonu kombinasyonu artık kullanılmıyor - hibrit model kaldırıldı
            # Not: Sürpriz butonu için İY/MS tahmini api_routes.py içindeki get_htft_prediction() API çağrısıyla yapılıyor
        
        # Rakip gücü analizi yap
        opponent_analysis = self.analyze_opponent_strength(home_form, away_form)
        logger.info(f"Rakip gücü analizi: Göreceli güç = {opponent_analysis['relative_strength']:.2f}")
        
        # H2H analizini yap
        h2h_analysis = self.analyze_head_to_head(home_team_id, away_team_id, home_team_name, away_team_name)
        
        if h2h_analysis and h2h_analysis['total_matches'] > 0:
            logger.info(f"H2H analizi: {h2h_analysis['home_wins']}-{h2h_analysis['draws']}-{h2h_analysis['away_wins']} ({h2h_analysis['total_matches']} maç)")
            
            # H2H analizine dayanarak maç sonucu olasılıklarını ayarla
            h2h_home_win_rate = h2h_analysis['home_wins'] / h2h_analysis['total_matches']
            h2h_draw_rate = h2h_analysis['draws'] / h2h_analysis['total_matches']
            h2h_away_win_rate = h2h_analysis['away_wins'] / h2h_analysis['total_matches']
            
            # H2H analizi ile mevcut tahminleri birleştir (20% H2H, 80% mevcut tahmin)
            if h2h_analysis['total_matches'] >= 3:  # En az 3 H2H maç varsa
                h2h_weight = 0.3  # %30 ağırlık (önceki: %20)
                home_win_prob = home_win_prob * (1 - h2h_weight) + h2h_home_win_rate * h2h_weight
                draw_prob = draw_prob * (1 - h2h_weight) + h2h_draw_rate * h2h_weight
                away_win_prob = away_win_prob * (1 - h2h_weight) + h2h_away_win_rate * h2h_weight
                
                logger.info(f"H2H analizi sonrası MS olasılıkları güncellendi: MS1={home_win_prob:.2f}, X={draw_prob:.2f}, MS2={away_win_prob:.2f}")
            
            # H2H'taki ortalama golleri de değerlendir
            if h2h_analysis['total_matches'] >= 3:  # En az 3 H2H maç varsa
                h2h_goals_weight = 0.25  # %25 ağırlık (önceki: %15)
                avg_home_goals = avg_home_goals * (1 - h2h_goals_weight) + h2h_analysis['avg_home_goals'] * h2h_goals_weight
                avg_away_goals = avg_away_goals * (1 - h2h_goals_weight) + h2h_analysis['avg_away_goals'] * h2h_goals_weight
                
                logger.info(f"H2H analizi sonrası gol beklentileri güncellendi: Ev={avg_home_goals:.2f}, Deplasman={avg_away_goals:.2f}")
                
                # Kesin skor tahminini güncelle
                if abs(avg_home_goals - avg_away_goals) < 0.3 and most_likely_outcome != "DRAW":
                    # Gol beklentileri yakın ama beraberlik tahmini yoksa, yeniden değerlendir
                    logger.info(f"H2H verilerine göre skor yeniden değerlendiriliyor. Gol beklentileri çok yakın, beraberlik olasılığı arttırılıyor.")
                    
                    # Beraberlik olasılığını artırırken gol beklentilerine dayalı mantık kontrolü ekle
                    total_expected_goals = avg_home_goals + avg_away_goals
                    
                    # Eğer toplam gol beklentisi 3.5'ten büyükse, beraberlik olasılığını çok fazla artırma
                    if total_expected_goals > 3.5:
                        draw_prob = min(max(draw_prob, home_win_prob * 0.7, away_win_prob * 0.7), 0.4)
                        logger.info(f"Yüksek gol beklentisi ({total_expected_goals:.2f}) için beraberlik olasılığı sınırlandırıldı: {draw_prob:.2f}")
                    else:
                        draw_prob = max(draw_prob, home_win_prob, away_win_prob) * 1.1
                    
                    home_win_prob = (1 - draw_prob) * (home_win_prob / (home_win_prob + away_win_prob)) if (home_win_prob + away_win_prob) > 0 else 0.25
                    away_win_prob = (1 - draw_prob) * (away_win_prob / (home_win_prob + away_win_prob)) if (home_win_prob + away_win_prob) > 0 else 0.25
                    
                    # Olasılıkları normalize et
                    total = home_win_prob + draw_prob + away_win_prob
                    if total > 0:
                        home_win_prob /= total
                        draw_prob /= total
                        away_win_prob /= total
                        
                    # En olası sonucu güncelle - beklenen golleri de hesaba katarak
                    most_likely_outcome = self._get_most_likely_outcome(home_win_prob, draw_prob, away_win_prob, avg_home_goals, avg_away_goals)
                    
                    # Kesin skoru H2H verisi kullanarak güncelle, ANCAK gol beklentileriyle tutarlı olmalı
                    if most_likely_outcome == "DRAW":
                        # Toplam gol beklentisine göre beraberlik skoru belirle
                        total_expected_goals = avg_home_goals + avg_away_goals
                        
                        if total_expected_goals > 3.5:
                            most_likely_score = "2-2"  # Yüksek skorlu beraberlik
                            logger.info(f"Yüksek gol beklentisi ({total_expected_goals:.2f}) için beraberlik skoru 2-2 belirlendi")
                        elif total_expected_goals > 2.0:
                            most_likely_score = "1-1"  # Orta skorlu beraberlik
                            logger.info(f"Orta gol beklentisi ({total_expected_goals:.2f}) için beraberlik skoru 1-1 belirlendi")
                        elif total_expected_goals > 0.8:
                            most_likely_score = "1-1"  # Düşük-orta skorlu beraberlik
                        else:
                            most_likely_score = "0-0"  # Çok düşük skorlu beraberlik
                            
                        logger.info(f"H2H verilerine göre kesin skor güncellendi: {most_likely_score}")
                        
                        # Bahis tahminlerini güncelle
                        bet_predictions['exact_score'] = most_likely_score
                        bet_predictions['match_result'] = 'DRAW'  # Beraberlik skorları için
                        logger.info(f"H2H sonrası kesin skor {most_likely_score} için maç sonucu DRAW olarak güncellendi")
                        # Not: prediction değişkeni henüz tanımlanmadı, en son saved_match_outcome değişkenine DRAW atanmalı
                        saved_match_outcome = "DRAW"
                        
                        # Olasılıkları dengele
                        draw_prob = max(draw_prob, 0.40)
                        remainder = 1.0 - draw_prob
                        home_win_prob = remainder * 0.5
                        away_win_prob = remainder * 0.5
        
        # Rakip analizi sadece log bırakıyor, skor değiştirilmiyor
        if opponent_analysis['relative_strength'] > 0.6:  # Ev sahibi daha güçlüyse
            # Ev sahibi güç farkı fazlaysa ve ev galibiyeti tahmin ediliyorsa skor farkını loglama (değiştirme)
            if most_likely_outcome == "HOME_WIN":
                home_score, away_score = map(int, most_likely_score.split('-'))
                if home_score - away_score == 1:
                    logger.info(f"Rakip analizi sonrası skor farkı artışı önerildi (ev sahibi daha güçlü): {home_score+1}-{away_score}")
        elif opponent_analysis['relative_strength'] < 0.4:  # Deplasman daha güçlüyse
            # Deplasman güç farkı fazlaysa ve deplasman galibiyeti tahmin ediliyorsa skor farkını loglama (değiştirme)
            if most_likely_outcome == "AWAY_WIN":
                home_score, away_score = map(int, most_likely_score.split('-'))
                if away_score - home_score == 1:
                    logger.info(f"Rakip analizi sonrası skor farkı artışı önerildi (deplasman daha güçlü): {home_score}-{away_score+1}")
        
        # Son olarak, olasılıkları yeniden normalize et
        total_prob = home_win_prob + draw_prob + away_win_prob
        if total_prob > 0:
            home_win_prob = home_win_prob / total_prob
            draw_prob = draw_prob / total_prob
            away_win_prob = away_win_prob / total_prob

        # En yüksek olasılıklı tahmini bul
        # Not: İY/MS tahmini artık kullanılmıyor (hibrit model kaldırıldı)
        most_confident_bet = max(bet_probabilities, key=bet_probabilities.get)

        # Tahmin sonuçlarını hazırla
        prediction = {
            'match': f"{home_team_name} vs {away_team_name}",
            'home_team': {
                'id': home_team_id,
                'name': home_team_name,
                'form': home_form,
                'form_periods': {
                    'last_3': home_form_periods['last_3'],
                    'last_6': home_form_periods['last_6'],
                    'last_9': home_form_periods['last_9']
                }
            },
            'away_team': {
                'id': away_team_id,
                'name': away_team_name,
                'form': away_form,
                'form_periods': {
                    'last_3': away_form_periods['last_3'],
                    'last_6': away_form_periods['last_6'],
                    'last_9': away_form_periods['last_9']
                }
            },
            'enhanced_factors': enhanced_factors if 'enhanced_factors' in locals() else {},
            'head_to_head': h2h_analysis if h2h_analysis else {
                'home_wins': 0,
                'away_wins': 0,
                'draws': 0, 
                'total_matches': 0,
                'avg_home_goals': 0,
                'avg_away_goals': 0,
                'recent_matches': []
            },
            'opponent_analysis': opponent_analysis,
            'predictions': {
                'home_win_probability': round(home_win_prob * 100, 2),
                'draw_probability': round(draw_prob * 100, 2),
                'away_win_probability': round(away_win_prob * 100, 2),
                'expected_goals': {
                    'home': round(avg_home_goals, 2),
                    'away': round(avg_away_goals, 2)
                },
                'confidence': self._calculate_confidence(home_form, away_form),
                # ÖNEMLİ: Maç sonucu kesin skora göre belirleniyor (tutarlılık için)
                # Ek debug bilgileri ekle - hangi skor kullanılıyor?
                'debug_exact_score_used': bet_predictions['exact_score'],
                
                # Kesin skor ve maç sonucu mutlaka birbirine uyumlu olmalı
                'exact_score': bet_predictions['exact_score'],
                
                # KRİTİK: Skoru ve sonucu tutarlı hale getir
                # Önce skordan sonucu türet (tutarlılık için)
                'most_likely_outcome': self._get_outcome_from_score(bet_predictions['exact_score']),
                'match_outcome': self._get_outcome_from_score(bet_predictions['exact_score']),
                
                # Bu noktada 'saved_most_likely_score' ve 'saved_match_outcome' değişkenlerini de kullanabiliriz
                # Ama mevcut bet_predictions['exact_score'] genellikle daha fazla mantık doğrulamasından geçmiştir
                'betting_predictions': {
                    'both_teams_to_score': {
                        'prediction': 'KG VAR' if bet_predictions['both_teams_to_score'] == 'KG VAR' else 'KG YOK',
                        'probability': round(bet_probabilities['both_teams_to_score'] * 100, 2)
                    },
                    'over_2_5_goals': {
                        'prediction': '2.5 ÜST' if bet_predictions['over_2_5_goals'] == '2.5 ÜST' else '2.5 ALT',
                        'probability': round(bet_probabilities['over_2_5_goals'] * 100, 2)
                    },
                    'over_3_5_goals': {
                        'prediction': '3.5 ÜST' if bet_predictions['over_3_5_goals'] == '3.5 ÜST' else '3.5 ALT',
                        'probability': round(bet_probabilities['over_3_5_goals'] * 100, 2)
                    },
                    'exact_score': {
                        'prediction': bet_predictions['exact_score'],
                        'probability': round(bet_probabilities['exact_score'] * 100, 2)
                    },
                    # İY/MS tahmini kaldırıldı - artık sadece sürpriz butonu ile api_routes.py içindeki get_htft_prediction() API çağrısı üzerinden erişilebilir
                    # İlk gol, korner ve kart tahminleri kaldırıldı
                },
                'neural_predictions': {
                    'home_goals': round(neural_home_goals, 2),
                    'away_goals': round(neural_away_goals, 2),
                    'combined_model': {
                        'home_goals': round(expected_home_goals, 2),
                        'away_goals': round(expected_away_goals, 2)
                    }
                },
                'raw_metrics': {
                    'expected_home_goals': round(avg_home_goals, 2),
                    'expected_away_goals': round(avg_away_goals, 2),
                    'p_home_scores': round(p_home_scores * 100, 2),
                    'p_away_scores': round(p_away_scores * 100, 2),
                    'expected_total_goals': round(expected_total_goals, 2),
                    'form_weights': {
                        'last_5_matches': weight_last_5,
                        'last_10_matches': weight_last_10,
                        'last_21_matches': weight_last_21
                    },
                    'weighted_form': {
                        'home_weighted_goals': round(weighted_home_goals, 2),
                        'home_weighted_form': round(weighted_home_form_points, 2),
                        'away_weighted_goals': round(weighted_away_goals, 2),
                        'away_weighted_form': round(weighted_away_form_points, 2)
                    },
                    'bayesian': {
                        'home_attack': round(home_form.get('bayesian', {}).get('home_lambda_scored', 0), 2),
                        'home_defense': round(home_form.get('bayesian', {}).get('home_lambda_conceded', 0), 2),
                        'away_attack': round(away_form.get('bayesian', {}).get('away_lambda_scored', 0), 2),
                        'away_defense': round(away_form.get('bayesian', {}).get('away_lambda_conceded', 0), 2),
                        'prior_home_goals': self.lig_ortalamasi_ev_gol,
                        'prior_away_goals': self.lig_ortalamasi_deplasman_gol
                    },
                    'recent_goals_average': recent_goals_average,
                    'defense_factors': {
                        'home_defense_factor': round(home_defense_factor, 2),
                        'away_defense_factor': round(away_defense_factor, 2)
                    },
                    'z_score_data': {
                        'home_std_dev': round(home_std_dev, 2),
                        'away_std_dev': round(away_std_dev, 2)
                    },
                    'adjusted_thresholds': {
                        'home_threshold': home_threshold,
                        'away_threshold': away_threshold,
                        'home_max': home_max,
                        'away_max': away_max
                    }
                },
                'most_confident_bet': {
                    'market': most_confident_bet,
                    'prediction': bet_predictions[most_confident_bet],
                    'probability': round(bet_probabilities[most_confident_bet] * 100, 2)
                },
                'explanation': {
                    'exact_score': f"Analiz edilen faktörler sonucunda en olası skor {most_likely_score} olarak tahmin edildi. Ev sahibi takımın beklenen gol ortalaması {avg_home_goals:.2f}, deplasman takımının beklenen gol ortalaması {avg_away_goals:.2f}.",
                    'match_result': f"Maç sonucu {self._get_outcome_from_score(bet_predictions['exact_score'])} tahmini, ev sahibi (%{round(home_win_prob*100,1)}), beraberlik (%{round(draw_prob*100,1)}) ve deplasman (%{round(away_win_prob*100,1)}) olasılıklarına dayanmaktadır. Bu sonuç, en olası skor ({bet_predictions['exact_score']}) temel alınarak belirlenmiştir.",
                    'relative_strength': f"Rakip analizi sonucunda, {opponent_analysis['relative_strength'] > 0.5 and home_team_name or away_team_name} göreceli olarak daha güçlü{' değil' if abs(0.5-opponent_analysis['relative_strength']) < 0.01 else ''} bulundu{abs(0.5-opponent_analysis['relative_strength']) >= 0.01 and ' (güç oranı: ' + str(round(abs(0.5-opponent_analysis.get('relative_strength', 0))*2*100, 1)) + '%)' or ' (takımlar eşit güçte)'}",
                    'head_to_head': f"Geçmiş karşılaşmalarda {h2h_analysis and h2h_analysis['total_matches'] or 0} maç oynandı. Sonuçlar: {h2h_analysis and h2h_analysis['home_wins'] or 0} ev sahibi galibiyeti, {h2h_analysis and h2h_analysis['draws'] or 0} beraberlik, {h2h_analysis and h2h_analysis['away_wins'] or 0} deplasman galibiyeti."
                }
            },
            'timestamp': datetime.now().timestamp(),
            'date_predicted': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Gelişmiş modellerin tahminlerini ekle (varsa) - YENİ: Geliştirilmiş tutarlı entegrasyon
        if advanced_prediction:
            # Gelişmiş tahminleri esas al - daha tutarlı yaklaşım
            prediction['predictions']['advanced_models'] = {
                'zero_inflated_poisson': {
                    'expected_goals': advanced_prediction['expected_goals'],
                    'home_win_probability': advanced_prediction['home_win_probability'],
                    'draw_probability': advanced_prediction['draw_probability'],
                    'away_win_probability': advanced_prediction['away_win_probability'],
                    'most_likely_outcome': advanced_prediction['most_likely_outcome'],
                    'most_likely_scores': advanced_prediction['model_details']['top_5_likely_scores'][:3],
                    'zero_zero_probability': advanced_prediction['model_details']['zero_zero_prob']
                },
                'ensemble_predictions': {
                    'home_goals': advanced_prediction['model_details'].get('ensemble_home_goals', 0),
                    'away_goals': advanced_prediction['model_details'].get('ensemble_away_goals', 0)
                }
            }

            # YENİ: Artık gelişmiş tahminlere daha fazla ağırlık ver - tutarlılık için
            # Gelişmiş tahminlere %70, klasik tahminlere %30 ağırlık
            combined_home_goals = (avg_home_goals * 0.3 + advanced_prediction['expected_goals']['home'] * 0.7)
            combined_away_goals = (avg_away_goals * 0.3 + advanced_prediction['expected_goals']['away'] * 0.7)

            # Final tahminleri güncelle
            prediction['predictions']['expected_goals'] = {
                'home': round(combined_home_goals, 2),
                'away': round(combined_away_goals, 2)
            }
            
            # YENİ: Bahis tahminlerini de gelişmiş modelden al eğer mevcutsa - TUTARLILIK için
            if 'betting_predictions' in advanced_prediction:
                adv_betting = advanced_prediction['betting_predictions']
                
                # Eğer gelişmiş model tam bahis tahminleri içeriyorsa bunları kullan
                if all(key in adv_betting for key in ['both_teams_to_score', 'over_2_5_goals', 'exact_score']):
                    logger.info("Bahis tahminleri gelişmiş tutarlı modelden alınıyor")
                    
                    # Kesin skor
                    if 'exact_score' in adv_betting:
                        bet_predictions['exact_score'] = adv_betting['exact_score']['prediction']
                        bet_probabilities['exact_score'] = adv_betting['exact_score']['probability'] / 100
                    
                    # KG VAR/YOK
                    if 'both_teams_to_score' in adv_betting:
                        bet_predictions['both_teams_to_score'] = adv_betting['both_teams_to_score']['prediction']
                        bet_probabilities['both_teams_to_score'] = adv_betting['both_teams_to_score']['probability'] / 100
                    
                    # 2.5 ÜST/ALT
                    if 'over_2_5_goals' in adv_betting:
                        bet_predictions['over_2_5_goals'] = adv_betting['over_2_5_goals']['prediction']
                        bet_probabilities['over_2_5_goals'] = adv_betting['over_2_5_goals']['probability'] / 100
                    
                    # 3.5 ÜST/ALT
                    if 'over_3_5_goals' in adv_betting:
                        bet_predictions['over_3_5_goals'] = adv_betting['over_3_5_goals']['prediction']
                        bet_probabilities['over_3_5_goals'] = adv_betting['over_3_5_goals']['probability'] / 100
                    
                    # Maç sonucu (MS)
                    home_win_prob = advanced_prediction['home_win_probability'] / 100
                    draw_prob = advanced_prediction['draw_probability'] / 100
                    away_win_prob = advanced_prediction['away_win_probability'] / 100
                    
                    # MS tahmini güncelleme
                    prediction['predictions']['home_win_probability'] = advanced_prediction['home_win_probability']
                    prediction['predictions']['draw_probability'] = advanced_prediction['draw_probability']
                    prediction['predictions']['away_win_probability'] = advanced_prediction['away_win_probability']
                    prediction['predictions']['most_likely_outcome'] = advanced_prediction['most_likely_outcome']
                    
                    # İlk yarı/maç sonu tahmini artık kullanılmıyor - hibrit model kaldırıldı
                    
                    # İlk gol kısmı kaldırıldı

        # Tahminlerin tutarlılığını kontrol et ve düzelt
        prediction = self._check_prediction_consistency(prediction)
        
        # Önbelleğe ekle ve kaydet
        # Konsensüs Filtreleme algoritması kaldırıldı - artık orijinal tahmini kullanıyoruz
        # Önbelleğe ekle ve kaydet
        self.predictions_cache[cache_key] = prediction
        # Önbellek değişti, değişiklik bayrağını güncelle
        self._cache_modified = True
        self.save_cache()
        
        logger.info(f"Tahmin yapıldı: {home_team_name} vs {away_team_name}")
        return prediction
    
    def _check_prediction_consistency(self, prediction):
        """
        Tahminlerde tutarlılık kontrolü yapar ve çelişkileri düzeltir
        Kesin skor merkezi bir referans noktası olarak kullanılır ve tüm diğer bahis tahminleri 
        bu skora göre zorunlu olarak güncellenir.
        
        Args:
            prediction: Tahmin sonuçları
            
        Returns:
            dict: Tutarlılığı sağlanmış tahmin sonuçları
        """
        try:
            # Gerekli verileri çıkart
            expected_home_goals = prediction['predictions']['expected_goals']['home']
            expected_away_goals = prediction['predictions']['expected_goals']['away']
            total_expected_goals = expected_home_goals + expected_away_goals
            exact_score = prediction['predictions']['exact_score']
            
            # 2.5 Üst olasılığını hesapla veya varsayılan olarak belirle
            over_2_5_prob = 0.5  # Varsayılan değer
            try:
                over_2_5_prob = prediction['predictions']['betting_predictions']['over_2_5_goals']['probability'] / 100.0
            except:
                # Olasılık bulunamadıysa gol beklentilerine göre hesapla
                over_2_5_prob = 0.65 if total_expected_goals > 2.5 else 0.35
                logger.info(f"2.5 Üst olasılığı tahmin edilemedi, gol beklentisine göre hesaplandı: {over_2_5_prob:.2f}")
            
            # Skor değerlerini ayrıştır
            try:
                home_score, away_score = map(int, exact_score.split('-'))
                total_goals = home_score + away_score
            except:
                home_score, away_score, total_goals = 0, 0, 0
                logger.error(f"Skor ayrıştırma hatası: {exact_score}")
            
            # Tutarsızlık kontrolleri
            
            # 1. Tutarsızlık: Yüksek gol beklentisi ama düşük skorlu tahmin
            if total_expected_goals > 3.0 and total_goals < 2:
                logger.warning(f"Tutarsızlık tespit edildi: Toplam gol beklentisi {total_expected_goals:.2f} ama tahmin edilen skor {exact_score}")
                
                # Yeni skor belirle
                if abs(expected_home_goals - expected_away_goals) < 0.3:
                    # Takımlar yakın güçteyse beraberlik skoru
                    if total_expected_goals > 4.0:
                        new_score = "2-2"  # Yüksek skorlu beraberlik
                    else:
                        new_score = "1-1"  # Orta skorlu beraberlik
                elif expected_home_goals > expected_away_goals:
                    # Ev takımı daha iyiyse
                    new_score = "2-1"  # Ev galibiyeti
                else:
                    # Deplasman daha iyiyse
                    new_score = "1-2"  # Deplasman galibiyeti
                
                logger.info(f"Tutarsızlık giderildi: Skor {exact_score} -> {new_score} olarak güncellendi")
                
                # Tahminleri güncelle
                prediction['predictions']['exact_score'] = new_score
                prediction['predictions']['betting_predictions']['exact_score']['prediction'] = new_score
                prediction['predictions']['most_likely_outcome'] = self._get_outcome_from_score(new_score)
                prediction['predictions']['match_outcome'] = self._get_outcome_from_score(new_score)
            
            # 2. Tutarsızlık: 2.5 Üst olasılığı yüksek ama düşük skorlu tahmin
            elif over_2_5_prob > 0.65 and total_goals < 3:  
                # Düşük gol beklentili maçlarda tutarsızlık düzeltmesi yapma
                # Form puanlarını da dikkate alarak düşük skorlu maç tanımını geliştirdik
                is_low_scoring_match = (expected_home_goals < 1.0 and expected_away_goals < 1.0) or \
                                    (expected_home_goals < 1.2 and expected_away_goals < 1.2 and 
                                     home_form and away_form and
                                     home_form.get('weighted_form_points', 0) < 0.4 and 
                                     away_form.get('weighted_form_points', 0) < 0.4)
                
                # Toplam gol beklentisi düşükse, skor tahminine öncelik ver
                total_expected_goals = expected_home_goals + expected_away_goals
                prioritize_score = (total_expected_goals < 2.5)
                
                if is_low_scoring_match or prioritize_score:
                    # Bu durumda skoru değiştirmek yerine 2.5 Üst/Alt ve KG VAR/YOK bahis tahminlerini skora göre ayarla
                    logger.info(f"Düşük gol beklentili veya düşük toplam gol beklentili maç ({total_expected_goals:.2f})")
                    logger.info(f"Skor tahminine öncelik veriliyor: {exact_score}")
                    
                    # 1-1 skoru artık makul kabul ediliyor - düşük skorlu maçlarda bile
                    if False and is_low_scoring_match and exact_score == "1-1":
                        # Bu kod artık çalıştırılmıyor, 1-1 skorunu koruyoruz
                        logger.info(f"1-1 skoru düşük gollu maçta bile makul kabul ediliyor - skor değiştirilmedi")
                        pass
                    elif False:
                        # Bu kodun eski versiyonu değişiklik yapıyordu, artık yapmıyor
                        logger.info(f"Düşük gol beklentili maçta 1-1 skoru korundu")
                        
                        # Yeni toplam golleri hesapla
                        home_score, away_score = map(int, new_score.split('-'))
                        total_goals = home_score + away_score
                        
                        # 1-1 -> 0-0, 1-0 veya 0-1 değişimi için tüm bahis tahminlerini güncelle
                        # Yeni skora göre tüm bahis tahminleri güncellenecek
                        # KG VAR/YOK, 2.5 ALT/ÜST, 3.5 ALT/ÜST gibi tüm bahis türleri
                        
                        # 1. KG VAR/YOK güncellemesi
                        if 'both_teams_to_score' in prediction['predictions']['betting_predictions']:
                            if home_score > 0 and away_score > 0:
                                # İki takım da gol atarsa KG VAR
                                prediction['predictions']['betting_predictions']['both_teams_to_score']['prediction'] = 'KG VAR'
                                btts_yes_prob = 0.85  # Yüksek KG VAR olasılığı
                                prediction['predictions']['betting_predictions']['both_teams_to_score']['probability'] = round(btts_yes_prob * 100, 1)
                                logger.info(f"{exact_score} -> {new_score} değişimi yapıldı, KG VAR olarak güncellendi, olasılık: {btts_yes_prob:.2f}")
                            else:
                                # Takımlardan biri gol atmazsa KG YOK
                                prediction['predictions']['betting_predictions']['both_teams_to_score']['prediction'] = 'KG YOK'
                                btts_no_prob = 0.9  # Yüksek KG YOK olasılığı
                                prediction['predictions']['betting_predictions']['both_teams_to_score']['probability'] = round(btts_no_prob * 100, 1)
                                logger.info(f"{exact_score} -> {new_score} değişimi yapıldı, KG YOK olarak güncellendi, olasılık: {btts_no_prob:.2f}")
                            
                            # Frontend için uyumluluk - isim formatı kontrol edilmeli
                            # Bazı API sonuçlarında "both_teams_to_score" bazılarında "both_teams_score" kullanılıyor
                            if 'both_teams_score' in prediction['predictions']['betting_predictions']:
                                prediction['predictions']['betting_predictions']['both_teams_score']['prediction'] = prediction['predictions']['betting_predictions']['both_teams_to_score']['prediction']
                                prediction['predictions']['betting_predictions']['both_teams_score']['probability'] = prediction['predictions']['betting_predictions']['both_teams_to_score']['probability']
                        
                        # 2. 2.5 ALT/ÜST güncellemesi
                        if 'over_2_5_goals' in prediction['predictions']['betting_predictions']:
                            if total_goals >= 3:
                                # 2.5 ÜST olmalı
                                prediction['predictions']['betting_predictions']['over_2_5_goals']['prediction'] = '2.5 ÜST'
                                over_25_prob = 0.9  # Yüksek 2.5 ÜST olasılığı
                                prediction['predictions']['betting_predictions']['over_2_5_goals']['probability'] = round(over_25_prob * 100, 1)
                                logger.info(f"{exact_score} -> {new_score} değişimi yapıldı, 2.5 ÜST olarak güncellendi, olasılık: {over_25_prob:.2f}")
                            else:
                                # 2.5 ALT olmalı
                                prediction['predictions']['betting_predictions']['over_2_5_goals']['prediction'] = '2.5 ALT'
                                under_25_prob = 0.85  # Yüksek 2.5 ALT olasılığı
                                prediction['predictions']['betting_predictions']['over_2_5_goals']['probability'] = round(under_25_prob * 100, 1)
                                logger.info(f"{exact_score} -> {new_score} değişimi yapıldı, 2.5 ALT olarak güncellendi, olasılık: {under_25_prob:.2f}")
                                
                        # 3. 3.5 ALT/ÜST güncellemesi
                        if 'over_3_5_goals' in prediction['predictions']['betting_predictions']:
                            if total_goals > 3:
                                # 3.5 ÜST olmalı
                                prediction['predictions']['betting_predictions']['over_3_5_goals']['prediction'] = '3.5 ÜST'
                                over_35_prob = 0.9  # Yüksek 3.5 ÜST olasılığı
                                prediction['predictions']['betting_predictions']['over_3_5_goals']['probability'] = round(over_35_prob * 100, 1)
                                logger.info(f"{exact_score} -> {new_score} değişimi yapıldı, 3.5 ÜST olarak güncellendi, olasılık: {over_35_prob:.2f}")
                            else:
                                # 3.5 ALT olmalı
                                prediction['predictions']['betting_predictions']['over_3_5_goals']['prediction'] = '3.5 ALT'
                                under_35_prob = 0.9  # Yüksek 3.5 ALT olasılığı
                                prediction['predictions']['betting_predictions']['over_3_5_goals']['probability'] = round(under_35_prob * 100, 1)
                                logger.info(f"{exact_score} -> {new_score} değişimi yapıldı, 3.5 ALT olarak güncellendi, olasılık: {under_35_prob:.2f}")
                                
                        # 4. Maç sonucu tahminini de güncelle
                        prediction['predictions']['most_likely_outcome'] = new_outcome
                                
                        # Debug bilgisi ekle
                        logger.info(f"Kesin skor {exact_score} (gol beklentileri: {expected_home_goals:.2f}-{expected_away_goals:.2f}) esas alınarak tüm tahminler güncellendi")
                    else:
                        logger.info(f"Düşük gol beklentili maç veya düşük toplam gol beklentili maç olduğu için skor tahminini koruyorum: {exact_score}")
                        if prioritize_score and not is_low_scoring_match:
                            logger.info(f"Toplam gol beklentisi ({total_expected_goals:.2f}) 2.5'un altında olduğu için skor ({exact_score}) öncelikli tutuldu")
                    
                    # Bahis tahminlerini skorla tutarlı hale getir
                    # 2.5 ALT/ÜST olasılıklarını Monte Carlo sonuçlarına göre hesapla
                    if total_goals < 3:
                        # Monte Carlo'dan 2.5 ALT olasılığını hesapla
                        under_25_prob = self._calculate_over_under_2_5_probability(expected_home_goals, expected_away_goals, False)
                        prediction['predictions']['betting_predictions']['over_2_5_goals']['prediction'] = '2.5 ALT'
                        prediction['predictions']['betting_predictions']['over_2_5_goals']['probability'] = round(under_25_prob * 100, 1)
                        logger.info(f"Bahis tahmini güncellendi: 2.5 ALT olasılığı {under_25_prob:.2f} olarak hesaplandı (skor: {exact_score})")
                    else:
                        # Monte Carlo'dan 2.5 ÜST olasılığını hesapla
                        over_25_prob = self._calculate_over_under_2_5_probability(expected_home_goals, expected_away_goals, True)
                        prediction['predictions']['betting_predictions']['over_2_5_goals']['prediction'] = '2.5 ÜST'
                        prediction['predictions']['betting_predictions']['over_2_5_goals']['probability'] = round(over_25_prob * 100, 1)
                        logger.info(f"Bahis tahmini güncellendi: 2.5 ÜST olasılığı {over_25_prob:.2f} olarak hesaplandı (skor: {exact_score})")
                    
                    # KG VAR/YOK tahminini skora göre ayarla (YES/NO formatında)
                    if 'both_teams_to_score' in prediction['predictions']['betting_predictions']:
                        # Önemli: exact_score yerine prediction['predictions']['exact_score'] kullan
                        # Çünkü 1-1 -> 0-0 değişimi yapıldıysa, exact_score hala eski değeri içerebilir
                        current_score = prediction['predictions']['exact_score']
                        home_score, away_score = map(int, current_score.split('-'))
                        
                        if home_score > 0 and away_score > 0:
                            # Monte Carlo'dan KG VAR olasılığını hesapla
                            btts_prob = self._calculate_kg_var_probability(None, None, expected_home_goals, expected_away_goals, 
                                                                        prediction.get('home_form', {}), prediction.get('away_form', {}))
                            prediction['predictions']['betting_predictions']['both_teams_to_score']['prediction'] = 'KG VAR'
                            prediction['predictions']['betting_predictions']['both_teams_to_score']['probability'] = round(btts_prob * 100, 1)
                            logger.info(f"Bahis tahmini güncellendi: KG VAR olarak ayarlandı, olasılık: {btts_prob:.2f} (skor: {current_score})")
                        else:
                            # Monte Carlo'dan KG YOK olasılığını hesapla
                            btts_no_prob = self._calculate_kg_yok_probability(None, None, expected_home_goals, expected_away_goals,
                                                                           prediction.get('home_form', {}), prediction.get('away_form', {}))
                            prediction['predictions']['betting_predictions']['both_teams_to_score']['prediction'] = 'KG YOK'
                            prediction['predictions']['betting_predictions']['both_teams_to_score']['probability'] = round(btts_no_prob * 100, 1)
                            logger.info(f"Bahis tahmini güncellendi: KG YOK olarak ayarlandı, olasılık: {btts_no_prob:.2f} (skor: {current_score})")
                        
                        # Eğer 'both_teams_score' alanı da varsa, onu da güncelle (frontend uyumluluğu için)
                        if 'both_teams_score' in prediction['predictions']['betting_predictions']:
                            prediction['predictions']['betting_predictions']['both_teams_score']['prediction'] = prediction['predictions']['betting_predictions']['both_teams_to_score']['prediction']
                            prediction['predictions']['betting_predictions']['both_teams_score']['probability'] = prediction['predictions']['betting_predictions']['both_teams_to_score']['probability']
                    
                    # Chelsea-Tottenham gibi düşük gol beklentili maçlarda başka düzeltme yapma
                    if total_expected_goals < 2.5:
                        logger.info(f"Toplam gol beklentisi düşük ({total_expected_goals:.2f}), skorun önceliği korundu: {exact_score}")
                        prediction['predictions']['debug_final_check'] = f"Düşük gol beklentili maç ({total_expected_goals:.2f}): Skor tahmine ({exact_score}) öncelik verildi. Bahis tahminleri skora göre ayarlandı."
                        return prediction
                    
                    # Düşük skorlu maçlarda da daha fazla işlem yapmayarak, bu blok içinden çık
                    return prediction
                else:
                    # Normal maçlarda tutarsızlık düzeltmesi yap
                    logger.warning(f"Tutarsızlık tespit edildi: 2.5 Üst olasılığı {over_2_5_prob:.2f} ama tahmin edilen skor {exact_score}")
                    
                    # Skor ve tahminleri güncelle (normal durumda)
                    if expected_home_goals > expected_away_goals:
                        new_score = "2-1"  # Ev galibiyeti
                    elif expected_away_goals > expected_home_goals:
                        new_score = "1-2"  # Deplasman galibiyeti
                    else:
                        new_score = "2-2"  # Beraberlik
                    
                    logger.info(f"Tutarsızlık giderildi: Skor {exact_score} -> {new_score} olarak güncellendi")
                    
                    # Tahminleri güncelle
                    prediction['predictions']['exact_score'] = new_score
                    prediction['predictions']['betting_predictions']['exact_score']['prediction'] = new_score
                    
                    # Ayrıca 2.5 ÜST bahis tahmini de güncellenmeli
                    prediction['predictions']['betting_predictions']['over_2_5_goals']['prediction'] = 'YES'
                    
                    # BURADAN SONRA KG VAR/YOK VE TOPLAM GOLÜ DE TUTARLI HALE GETİR!
                    home_score = int(new_score.split('-')[0])
                    away_score = int(new_score.split('-')[1])
                    total_goals = home_score + away_score
                    
                    # Hem ev sahibi hem deplasman gol atıyorsa KG VAR, değilse KG YOK
                    if home_score > 0 and away_score > 0:
                        prediction['predictions']['betting_predictions']['both_teams_to_score']['prediction'] = 'KG VAR'
                        logger.info(f"Skor değişimi sonrası KG VAR olarak güncellendi (skor: {home_score}-{away_score})")
                    else:
                        prediction['predictions']['betting_predictions']['both_teams_to_score']['prediction'] = 'KG YOK'
                        logger.info(f"Skor değişimi sonrası KG YOK olarak güncellendi (skor: {home_score}-{away_score})")
                    
                    # Eğer 'both_teams_score' alanı da varsa, onu da güncelle (frontend uyumluluğu için)
                    if 'both_teams_score' in prediction['predictions']['betting_predictions']:
                        prediction['predictions']['betting_predictions']['both_teams_score']['prediction'] = prediction['predictions']['betting_predictions']['both_teams_to_score']['prediction']
                    
                    # Toplam gol 3 veya üzerindeyse 2.5 ÜST, değilse 2.5 ALT
                    if total_goals >= 3:
                        prediction['predictions']['betting_predictions']['over_2_5_goals']['prediction'] = '2.5 ÜST'
                    else:
                        prediction['predictions']['betting_predictions']['over_2_5_goals']['prediction'] = '2.5 ALT'
                        
                    # 3.5 üst/alt kontrolü
                    if 'over_3_5_goals' in prediction['predictions']['betting_predictions']:
                        if total_goals > 3:
                            prediction['predictions']['betting_predictions']['over_3_5_goals']['prediction'] = '3.5 ÜST'
                        else:
                            prediction['predictions']['betting_predictions']['over_3_5_goals']['prediction'] = '3.5 ALT'
                    prediction['predictions']['most_likely_outcome'] = self._get_outcome_from_score(new_score)
                    prediction['predictions']['match_outcome'] = self._get_outcome_from_score(new_score)
                
                # Bu satırlar yukarıda exact_score değişkeninin zaten güncellendiği blok içinde kullanılıyor
                # Bu nedenle bu bölümde new_score değişkeniyle ilgili kod kaldırıldı
                
            # 3. Tutarsızlık: Düşük gol beklentisi ama yüksek skorlu tahmin
            elif total_expected_goals < 1.5 and total_goals > 2:
                logger.warning(f"Tutarsızlık tespit edildi: Toplam gol beklentisi {total_expected_goals:.2f} ama tahmin edilen skor {exact_score}")
                
                # Skor ve tahminleri güncelle
                if abs(expected_home_goals - expected_away_goals) < 0.3:
                    new_score = "0-0"  # Düşük skorlu beraberlik
                elif expected_home_goals > expected_away_goals:
                    new_score = "1-0"  # Ev galibiyeti
                else:
                    new_score = "0-1"  # Deplasman galibiyeti
                
                logger.info(f"Tutarsızlık giderildi: Skor {exact_score} -> {new_score} olarak güncellendi")
                
                # Tahminleri güncelle
                prediction['predictions']['exact_score'] = new_score
                prediction['predictions']['betting_predictions']['exact_score']['prediction'] = new_score
                prediction['predictions']['most_likely_outcome'] = self._get_outcome_from_score(new_score)
                prediction['predictions']['match_outcome'] = self._get_outcome_from_score(new_score)
            
            # KG VAR/YOK hesaplaması - artık tutarsızlık kontrolü yapılmıyor, doğrudan skor üzerinden hesaplanıyor
            if 'both_teams_to_score' in prediction['predictions']['betting_predictions']:
                btts = prediction['predictions']['betting_predictions']['both_teams_to_score']
                
                # Eğer her iki takım da gol attıysa, KG VAR olmalı, değilse KG YOK olmalı
                if home_score > 0 and away_score > 0:
                    # KG VAR
                    prediction['predictions']['betting_predictions']['both_teams_to_score']['prediction'] = 'KG VAR'
                    if 'probability' in btts:
                        # KG VAR olasılığını dinamik hesapla
                        kg_var_prob = self._calculate_kg_var_probability(None, None, expected_home_goals, expected_away_goals,
                                                                      prediction.get('home_form', {}), prediction.get('away_form', {}))
                        prediction['predictions']['betting_predictions']['both_teams_to_score']['probability'] = round(kg_var_prob * 100, 1)
                    logger.info(f"Skor temel alındı: {home_score}-{away_score} skoru için KG VAR tahmini yapıldı")
                else:
                    # KG YOK
                    prediction['predictions']['betting_predictions']['both_teams_to_score']['prediction'] = 'KG YOK'
                    if 'probability' in btts:
                        # KG YOK olasılığını dinamik hesapla
                        kg_yok_prob = self._calculate_kg_yok_probability(None, None, expected_home_goals, expected_away_goals,
                                                                      prediction.get('home_form', {}), prediction.get('away_form', {}))
                        prediction['predictions']['betting_predictions']['both_teams_to_score']['probability'] = round(kg_yok_prob * 100, 1)
                    logger.info(f"Skor temel alındı: {home_score}-{away_score} skoru için KG YOK tahmini yapıldı")
            
            # 2.5 ve 3.5 ALT/ÜST Tutarlılık Kontrolü
            # Skor toplamı 3 veya üstüyse 2.5 ÜST, değilse 2.5 ALT olmalı
            if 'over_2_5_goals' in prediction['predictions']['betting_predictions']:
                over_25 = prediction['predictions']['betting_predictions']['over_2_5_goals']
                home_score = int(home_score)
                away_score = int(away_score)
                total_goals = home_score + away_score
                
                # Artık bu noktada tutarsızlık kontrolü yapmıyoruz, yalnızca tahminleri skordan üretiyoruz
                # Skor artık sabitlendi, skor temel alınarak 2.5 ÜST/ALT değeri üretiliyor
                if total_goals >= 3:
                    # 2.5 ÜST olmalı
                    prediction['predictions']['betting_predictions']['over_2_5_goals']['prediction'] = '2.5 ÜST'
                    if 'probability' in over_25:
                        # 2.5 ÜST olasılığını dinamik hesapla (ama çok yüksek bir değer ver)
                        prediction['predictions']['betting_predictions']['over_2_5_goals']['probability'] = 90.0
                    logger.info(f"Skor temel alındı: {home_score}-{away_score} skoru için 2.5 ÜST tahmini yapıldı")
                else:
                    # 2.5 ALT olmalı
                    prediction['predictions']['betting_predictions']['over_2_5_goals']['prediction'] = '2.5 ALT'
                    if 'probability' in over_25:
                        # 2.5 ALT olasılığını dinamik hesapla (ama çok yüksek bir değer ver)
                        prediction['predictions']['betting_predictions']['over_2_5_goals']['probability'] = 90.0
                    logger.info(f"Skor temel alındı: {home_score}-{away_score} skoru için 2.5 ALT tahmini yapıldı")
            
            # 3.5 ALT/ÜST - Artık skor üzerinden hesaplanıyor, tutarsızlık kontrolü yerine
            if 'over_3_5_goals' in prediction['predictions']['betting_predictions']:
                over_35 = prediction['predictions']['betting_predictions']['over_3_5_goals']
                
                # Skor üzerinden ALT/ÜST tahmini belirle
                if total_goals > 3:
                    # 3.5 ÜST olmalı
                    prediction['predictions']['betting_predictions']['over_3_5_goals']['prediction'] = '3.5 ÜST'
                    if 'probability' in over_35:
                        # 3.5 ÜST olasılığını dinamik hesapla (ama çok yüksek bir değer ver)
                        prediction['predictions']['betting_predictions']['over_3_5_goals']['probability'] = 90.0
                    logger.info(f"Skor temel alındı: {home_score}-{away_score} skoru için 3.5 ÜST tahmini yapıldı")
                else:
                    # 3.5 ALT olmalı 
                    prediction['predictions']['betting_predictions']['over_3_5_goals']['prediction'] = '3.5 ALT'
                    if 'probability' in over_35:
                        # 3.5 ALT olasılığını dinamik hesapla (ama çok yüksek bir değer ver)
                        prediction['predictions']['betting_predictions']['over_3_5_goals']['probability'] = 90.0
                    logger.info(f"Skor temel alındı: {home_score}-{away_score} skoru için 3.5 ALT tahmini yapıldı")

            # Kesin skor değiştikten sonra tüm tahminlerin tutarlılığını sağla
            # Merkezi tahmin tutarlılık fonksiyonu
            self._update_all_predictions_from_score(prediction)
            
            # Yaptığımız düzeltmeleri debug alanına ekleyelim
            prediction['predictions']['debug_final_check'] = f"Tutarlılık giderildi: Skor analiz edildi ve tüm bahis tahminleri (KG, 2.5, 3.5) tutarlı hale getirildi."
            
            # Açıklama metinlerini de güncelle
            prediction['predictions']['explanation']['exact_score'] = f"Analiz edilen faktörler sonucunda en olası skor {prediction['predictions']['exact_score']} olarak tahmin edildi. Ev sahibi takımın beklenen gol ortalaması {expected_home_goals:.2f}, deplasman takımının beklenen gol ortalaması {expected_away_goals:.2f}."
            prediction['predictions']['explanation']['match_result'] = f"Maç sonucu {prediction['predictions']['most_likely_outcome']} tahmini, ev sahibi (%{prediction['predictions']['home_win_probability']}), beraberlik (%{prediction['predictions']['draw_probability']}) ve deplasman (%{prediction['predictions']['away_win_probability']}) olasılıklarına dayanmaktadır. Bu sonuç, en olası skor ({prediction['predictions']['exact_score']}) temel alınarak belirlenmiştir."
            
        except Exception as e:
            logger.error(f"Tahmin tutarlılık kontrolü hatası: {e}")
        
        return prediction

    def _calculate_confidence(self, home_form, away_form):
        """Tahmin güven seviyesini hesapla"""
        # Takımların oynadığı maç sayısına göre güven hesapla
        home_matches = home_form.get('recent_matches', 0)
        away_matches = away_form.get('recent_matches', 0)

        # Eğer takımlar en az 3 maç oynamışsa daha güvenilir tahmin yapabiliriz
        if home_matches >= 3 and away_matches >= 3:
            confidence = min(home_matches, away_matches) / 5.0  # 5 maç üzerinden normalize et
            return round(min(confidence, 1.0) * 100, 2)  # 0-100 arası değer
        else:
            return round((min(home_matches, away_matches) / 5.0) * 70, 2)  # Daha düşük güven
            
    def _calculate_low_scoring_advantage(self, home_form, away_form, home_team_id, away_team_id):
        """
        Düşük skorlu maçlarda (her iki takım da 1.0'ın altında gol beklentisine sahip)
        hangi takımın kazanma olasılığının daha yüksek olduğunu hesaplar.
        
        Args:
            home_form: Ev sahibi takımın form bilgileri
            away_form: Deplasman takımının form bilgileri
            home_team_id: Ev sahibi takım ID'si
            away_team_id: Deplasman takımı ID'si
            
        Returns:
            float: -1.0 ile 1.0 arasında bir değer. 
                  Pozitif değerler ev sahibi avantajını, 
                  negatif değerler deplasman avantajını gösterir.
        """
        # Varsayılan avantaj: Ev sahibi avantajı
        advantage_factor = 0.1
        
        try:
            # 1. Form puanlarını karşılaştır
            home_form_points = home_form.get('weighted_form_points', 0.5)
            away_form_points = away_form.get('weighted_form_points', 0.5)
            
            # Form farkını ölçeklendir (-0.5 ile 0.5 arası bir değer)
            form_advantage = (home_form_points - away_form_points) * 0.5
            advantage_factor += form_advantage
            
            # 2. Defansif performans - Son 8 maçta gol yememe oranı
            home_clean_sheets = 0
            home_matches_count = min(8, len(home_form.get('recent_match_data', [])))
            if home_matches_count > 0:
                home_clean_sheets = sum(1 for match in home_form.get('recent_match_data', [])[:8] 
                                    if match.get('goals_conceded', 1) == 0)
                                    
            away_clean_sheets = 0
            away_matches_count = min(8, len(away_form.get('recent_match_data', [])))
            if away_matches_count > 0:
                away_clean_sheets = sum(1 for match in away_form.get('recent_match_data', [])[:8] 
                                    if match.get('goals_conceded', 1) == 0)
            
            # Gol yememe avantajı (-0.25 ile 0.25 arası)
            clean_sheet_advantage = 0
            if home_matches_count > 0 and away_matches_count > 0:
                clean_sheet_advantage = ((home_clean_sheets/home_matches_count) - (away_clean_sheets/away_matches_count)) * 0.5
            advantage_factor += clean_sheet_advantage
            
            # 3. H2H (kafa kafaya) maçları analiz et
            h2h_advantage = 0
            h2h_results = self.analyze_head_to_head(home_team_id, away_team_id, "", "")
            
            if h2h_results and 'matches' in h2h_results and h2h_results.get('matches'):
                h2h_matches = h2h_results.get('matches', [])
                
                # Son 5 H2H maçını değerlendir
                recent_h2h = h2h_matches[:5]
                home_wins = 0
                away_wins = 0
                
                for match in recent_h2h:
                    if match.get('home_team_id') == home_team_id:
                        # Ev sahibi olarak
                        if match.get('home_score', 0) > match.get('away_score', 0):
                            home_wins += 1
                        elif match.get('home_score', 0) < match.get('away_score', 0):
                            away_wins += 1
                    else:
                        # Deplasman olarak
                        if match.get('away_score', 0) > match.get('home_score', 0):
                            home_wins += 1
                        elif match.get('away_score', 0) < match.get('home_score', 0):
                            away_wins += 1
                
                # H2H avantajı (-0.3 ile 0.3 arası)
                if home_wins + away_wins > 0:
                    h2h_advantage = (home_wins - away_wins) * 0.15  # Her maç ±0.15 değer
                    advantage_factor += h2h_advantage
            
            # 4. Düşük skorlu maçlardaki performansı değerlendir
            home_low_scoring_wins = 0
            home_low_matches_count = 0
            
            for match in home_form.get('recent_match_data', [])[:10]:
                total_goals = match.get('goals_scored', 0) + match.get('goals_conceded', 0)
                if total_goals <= 2:  # Düşük skorlu maç
                    home_low_matches_count += 1
                    if match.get('result', '') == 'W':
                        home_low_scoring_wins += 1
            
            away_low_scoring_wins = 0
            away_low_matches_count = 0
            
            for match in away_form.get('recent_match_data', [])[:10]:
                total_goals = match.get('goals_scored', 0) + match.get('goals_conceded', 0)
                if total_goals <= 2:  # Düşük skorlu maç
                    away_low_matches_count += 1
                    if match.get('result', '') == 'W':
                        away_low_scoring_wins += 1
            
            # Düşük skorlu maçlarda kazanma oranı avantajı (-0.2 ile 0.2 arası)
            low_scoring_advantage = 0
            
            if home_low_matches_count > 0 and away_low_matches_count > 0:
                home_low_win_rate = home_low_scoring_wins / home_low_matches_count
                away_low_win_rate = away_low_scoring_wins / away_low_matches_count
                low_scoring_advantage = (home_low_win_rate - away_low_win_rate) * 0.4  # Max ±0.2
                advantage_factor += low_scoring_advantage
            
            # Avantaj faktörünü makul sınırlar içinde tut
            advantage_factor = max(-1.0, min(1.0, advantage_factor))
            
            logger.info(f"Düşük skorlu maç avantaj faktörü: {advantage_factor:.2f} " +
                      f"(Form: {form_advantage:.2f}, Defans: {clean_sheet_advantage:.2f}, " +
                      f"H2H: {h2h_advantage:.2f}, Düşük Skor: {low_scoring_advantage if 'low_scoring_advantage' in locals() else 0:.2f})")
            
        except Exception as e:
            logger.error(f"Düşük skorlu maç avantajı hesaplanırken hata: {str(e)}")
            # Hata durumunda varsayılan olarak ev sahibi hafif avantajlı
            advantage_factor = 0.1
        
        return advantage_factor

    def _get_most_likely_outcome(self, home_win_prob, draw_prob, away_win_prob, avg_home_goals=None, avg_away_goals=None):
        """
        En olası sonucu belirle - beklenen gol farkını da hesaba katarak
        
        Args:
            home_win_prob: Ev sahibi kazanma olasılığı
            draw_prob: Beraberlik olasılığı
            away_win_prob: Deplasman kazanma olasılığı
            avg_home_goals: Ev sahibi için beklenen gol sayısı (opsiyonel)
            avg_away_goals: Deplasman için beklenen gol sayısı (opsiyonel)
            
        Returns:
            String: "HOME_WIN", "DRAW" veya "AWAY_WIN"
        """
        # En yüksek olasılığı tespit et
        max_prob = max(home_win_prob, draw_prob, away_win_prob)
        
        # Olasılıklar arasında küçük bir fark varsa ve beklenen gol değerleri mevcut ise
        # beklenen golleri dikkate alarak düzeltme yap
        is_close = lambda a, b: abs(a - b) < 0.07  # %7'den az fark varsa "yakın" kabul et
        
        if avg_home_goals is not None and avg_away_goals is not None:
            # Beklenen gol farkı
            goal_diff = avg_home_goals - avg_away_goals
            
            # Beraberlik en olası sonuç olarak görünüyorsa ve önemli gol farkı varsa düzelt
            if max_prob == draw_prob and abs(goal_diff) > 1.5:
                logger.info(f"Beklenen gol farkı büyük ({goal_diff:.2f}) olduğu halde beraberlik en olası sonuç görünüyor, düzeltme yapılıyor")
                
                # Gol farkına göre olası kazananı belirle - artık olasılık farkına bakmaksızın düzeltme yapıyoruz
                if goal_diff > 0:  # Ev sahibi daha fazla gol beklentisine sahip
                    logger.info(f"Beklenen gol farkı ({goal_diff:.2f}) büyük olduğu için HOME_WIN seçildi (olasılıklar: Ev={home_win_prob:.2f}, X={draw_prob:.2f}, Dep={away_win_prob:.2f})")
                    return "HOME_WIN"
                else:  # Deplasman daha fazla gol beklentisine sahip
                    logger.info(f"Beklenen gol farkı ({-goal_diff:.2f}) büyük olduğu için AWAY_WIN seçildi (olasılıklar: Ev={home_win_prob:.2f}, X={draw_prob:.2f}, Dep={away_win_prob:.2f})")
                    return "AWAY_WIN"
            
            # Olasılıklar çok yakınsa, beklenen gol farkı belirleyici olabilir
            if is_close(home_win_prob, draw_prob) or is_close(home_win_prob, away_win_prob) or is_close(draw_prob, away_win_prob):
                logger.info(f"Olasılıklar çok yakın, gol farkı belirleyici olacak: {goal_diff:.2f}")
                
                # Daha hassas gol farkı eşiği kullanarak sonucu belirle
                if goal_diff > 0.5:  # 0.8'den 0.5'e düşürüldü - daha hassas ev sahibi galibiyeti tespiti
                    logger.info(f"Ev sahibi lehine önemli gol farkı ({goal_diff:.2f}) tespit edildi")
                    return "HOME_WIN"
                elif goal_diff < -0.5:  # -0.8'den -0.5'e düşürüldü - daha hassas deplasman galibiyeti tespiti
                    logger.info(f"Deplasman lehine önemli gol farkı ({goal_diff:.2f}) tespit edildi")
                    return "AWAY_WIN"
                else:
                    logger.info(f"Gol farkı ({goal_diff:.2f}) beraberlik eşiği içinde")
                    return "DRAW"
        
        # Standart olası sonuç belirleme
        if max_prob == home_win_prob:
            return "HOME_WIN"
        elif max_prob == draw_prob:
            return "DRAW"
        else:
            return "AWAY_WIN"
    
    def _select_reasonable_score_from_simulation(self, top_scores, avg_home_goals, avg_away_goals):
        """
        Monte Carlo simülasyonundan makul bir skor seç
        Düşük beklenen gol değerleri için çok yüksek skorları engeller
        ve beklenen gol değerlerine uyumlu sonuçlar üretir
        
        DÜŞÜK SKORLU MAÇLAR İÇİN İYİLEŞTİRİLMİŞ DAĞILIM:
        - Hem ev sahibi hem deplasman takımı için gol beklentisi 1.0'dan düşükse özel işlem
        - 0-0, 1-0, 0-1 skorlarının olasılığını dengelenmiş şekilde artırır
        - Form farkına göre kazanan takımı belirler
        - Daha gerçekçi düşük skorlu maç tahminleri yapar
        
        Args:
            top_scores: Monte Carlo'dan gelen en yüksek olasılıklı skorlar ve sayıları
            avg_home_goals: Ev sahibi için beklenen gol
            avg_away_goals: Deplasman için beklenen gol
            
        Returns:
            String: Makul bir skor (örn: "1-0")
        """
        if not top_scores:
            # Varsayılan olarak beklenen gollerin yuvarlanmasını kullan
            return f"{round(avg_home_goals)}-{round(avg_away_goals)}"
            
        # Beklenen gol değerine göre maksimum makul skor sınırları
        max_home_score = 1  # Varsayılan
        max_away_score = 1  # Varsayılan
        
        # Ev sahibi için sınır
        if avg_home_goals < 1.0:
            max_home_score = 1  # 1'in altında beklenen gol için maksimum 1 gol
        elif avg_home_goals < 1.8:
            max_home_score = 2  # 1-1.8 arası beklenen gol için maksimum 2 gol
        elif avg_home_goals < 2.5:
            max_home_score = 3  # 1.8-2.5 arası beklenen gol için maksimum 3 gol
        else:
            max_home_score = 4  # 2.5'ten yüksek beklenen gol için maksimum 4 gol
            
        # Deplasman için sınır
        if avg_away_goals < 1.0:
            max_away_score = 1  # 1'in altında beklenen gol için maksimum 1 gol
        elif avg_away_goals < 1.8:
            max_away_score = 2  # 1-1.8 arası beklenen gol için maksimum 2 gol
        elif avg_away_goals < 2.5:
            max_away_score = 3  # 1.8-2.5 arası beklenen gol için maksimum 3 gol
        else:
            max_away_score = 4  # 2.5'ten yüksek beklenen gol için maksimum 4 gol
            
        # Beklenen gol farkı büyükse özel işlem
        goal_diff = avg_away_goals - avg_home_goals
        if goal_diff >= 2.0:  # Deplasman 2+ gol farkı bekleniyor
            # Leicester vs Newcastle maçı gibi durumlar için özel kural:
            # Ev sahibi takım çok düşük gol beklentisine sahip (1.0'dan az)
            # Deplasman takımı yüksek gol beklentisine sahip (2.5+ gol)
            if avg_home_goals < 1.0 and avg_away_goals > 2.5:
                # Ev sahibi için maximum 0 gol (KG YOK için) zorla
                max_home_score = 0
                logger.info(f"Çok büyük gol farkı tespit edildi: Deplasman avantajı {goal_diff:.2f}. Ev sahibi için 0 gol zorlanıyor.")
                
        # Eğer ev sahibi için 0 gol belirlendiyse monte carlo'dan gelen skorları 0-X şeklinde zorla
        if max_home_score == 0 and len(top_scores) > 0:
            new_top_scores = []
            for score, count in top_scores:
                if '-' in score:
                    home, away = map(int, score.split('-'))
                    # Her skoru 0-X şekline dönüştür, ev sahibi golünü 0 yap
                    if home > 0:
                        new_score = f"0-{away}"
                        new_top_scores.append((new_score, count))
                        logger.info(f"Skor zorla 0 gol olacak şekilde değiştirildi: {score} -> {new_score}")
                    else:
                        new_top_scores.append((score, count))
            
            if new_top_scores:
                # Orijinal listeyi güncelle
                top_scores = new_top_scores
                logger.info(f"Ev sahibi için 0 gol zorlandı, yeni olası skorlar: {top_scores[:3]}")
        
        # Skor adayları
        reasonable_scores = []
        
        # Top 5 skor listesini makul sınırlara göre filtrele
        for score, count in top_scores:
            if '-' in score:
                home, away = map(int, score.split('-'))
                
                # Her iki takım için skor makul sınırlar içinde mi?
                if home <= max_home_score and away <= max_away_score:
                    # Düşük skorlu maç özel durumu
                    if avg_home_goals < 1.0 and avg_away_goals < 1.0:
                        # Takımların gol beklentileri arasındaki farkı kontrol et
                        goal_diff = abs(avg_home_goals - avg_away_goals)
                        
                        # Gol beklentisi farkı büyükse (bir takım daha güçlüyse)
                        if goal_diff >= 0.15:
                            # Güçlü takımın 1-0 kazanması daha olası
                            if avg_home_goals > avg_away_goals:
                                # Ev sahibi daha güçlü
                                if score == "1-0":
                                    # 1-0 skoruna büyük öncelik ver (sayımı %500 artır)
                                    reasonable_scores.append((score, count * 6.0))
                                    logger.info(f"Düşük skorlu ve form farkı olan maçta {score} skoru için ağırlık arttırıldı: {count} -> {count * 6.0}")
                                elif score == "0-0":
                                    # 0-0 skoruna da öncelik ver ama 1-0'dan az
                                    reasonable_scores.append((score, count * 2.5))
                                    logger.info(f"Düşük skorlu ve form farkı olan maçta {score} skoru için ağırlık arttırıldı: {count} -> {count * 2.5}")
                                elif score == "1-1":
                                    # 1-1 skorunu çok azalt
                                    reduced_weight = count * 0.25  # %75 azalt
                                    reasonable_scores.append((score, reduced_weight))
                                    logger.info(f"Düşük skorlu ve form farkı olan maçta {score} skoru için ağırlık azaltıldı: {count} -> {reduced_weight}")
                                else:
                                    # Diğer skorların ağırlığını azalt
                                    reasonable_scores.append((score, count * 0.5))
                            else:
                                # Deplasman daha güçlü
                                if score == "0-1":
                                    # 0-1 skoruna büyük öncelik ver (sayımı %500 artır)
                                    reasonable_scores.append((score, count * 6.0))
                                    logger.info(f"Düşük skorlu ve form farkı olan maçta {score} skoru için ağırlık arttırıldı: {count} -> {count * 6.0}")
                                elif score == "0-0":
                                    # 0-0 skoruna da öncelik ver ama 0-1'den az
                                    reasonable_scores.append((score, count * 2.5))
                                    logger.info(f"Düşük skorlu ve form farkı olan maçta {score} skoru için ağırlık arttırıldı: {count} -> {count * 2.5}")
                                elif score == "1-1":
                                    # 1-1 skorunu çok azalt
                                    reduced_weight = count * 0.25  # %75 azalt
                                    reasonable_scores.append((score, reduced_weight))
                                    logger.info(f"Düşük skorlu ve form farkı olan maçta {score} skoru için ağırlık azaltıldı: {count} -> {reduced_weight}")
                                else:
                                    # Diğer skorların ağırlığını azalt
                                    reasonable_scores.append((score, count * 0.5))
                        else:
                            # Takımlar benzer güçteyse
                            if score in ["0-0", "1-0", "0-1"]:
                                # Düşük skorlu maçlarda bu sonuçlara öncelik ver (sayımı %400 artır)
                                reasonable_scores.append((score, count * 5.0))
                                logger.info(f"Düşük skorlu maçta {score} skoru için ağırlık artırıldı: {count} -> {count * 5.0}")
                            elif score == "1-1":
                                # 1-1 skorunun ağırlığını düşür
                                reduced_weight = count * 0.4  # %60 azalt
                                reasonable_scores.append((score, reduced_weight))
                                logger.info(f"Düşük skorlu maçta {score} skoru için ağırlık azaltıldı: {count} -> {reduced_weight}")
                            else:
                                # Diğer yüksek skorları da düşür
                                reduced_weight = count * 0.5  # %50 azalt
                                reasonable_scores.append((score, reduced_weight))
                                logger.info(f"Düşük skorlu maçta diğer skor {score} için ağırlık azaltıldı: {count} -> {reduced_weight}")
                    else:
                        reasonable_scores.append((score, count))
        
        # Makul adaylar varsa beklenen gol değerleriyle en tutarlı olanı seç
        if reasonable_scores:
            # Adayları maçın beklenen skoruna göre değerlendir - beklenen gol değerlerine uyumu kontrol et
            
            # Beklenen ev sahibi galibiyeti mi?
            expected_home_win = avg_home_goals > avg_away_goals + 0.15  # Form farklılığı için eşik değeri
            # Beklenen beraberlik mi? - Eşik değeri 0.3'ten 0.15'e düşürüldü
            expected_draw = abs(avg_home_goals - avg_away_goals) < 0.15  # Daha hassas beraberlik tespiti
            # Beklenen deplasman galibiyeti mi?
            expected_away_win = avg_away_goals > avg_home_goals + 0.15  # Form farklılığı için eşik değeri
            
            # Beklenen sonuca göre makul skorları sırala
            aligned_scores = []
            
            for score, count in reasonable_scores:
                home, away = map(int, score.split('-'))
                
                # Sonuç beklentiyle uyumlu mu?
                if (expected_home_win and home > away) or \
                   (expected_draw and home == away) or \
                   (expected_away_win and away > home):
                    # Beklenen sonuçla uyumluysa, tam uyum puanı ver
                    # 3 şey kontrol et: 1) Sonuç tipi 2) Ev gol yakınlığı 3) Deplasman gol yakınlığı
                    home_goal_diff = abs(home - round(avg_home_goals))
                    away_goal_diff = abs(away - round(avg_away_goals))
                    # Toplam uyum puanı - düşük = daha iyi
                    alignment_score = home_goal_diff + away_goal_diff
                    
                    aligned_scores.append((score, count, alignment_score))
                else:
                    # Uyumlu değilse, daha yüksek bir uyumsuzluk puanı ver (düşük öncelik)
                    # Ama yine de Monte Carlo sayım değerini de dikkate al
                    alignment_score = 10 + (1000 / (count + 1))  # Yüksek sayım değeri uyumsuzluğu azaltır
                    aligned_scores.append((score, count, alignment_score))
            
            # Önce uyum puanına (düşük=iyi), sonra sayıma (yüksek=iyi) göre sırala
            if aligned_scores:
                aligned_scores.sort(key=lambda x: (x[2], -x[1]))
                logger.info(f"Beklenen gollere göre uyumlu skorlar: {aligned_scores[:3]}")
                most_reasonable = aligned_scores[0][0]
                return most_reasonable
            
            # Uyumlu skor bulunamazsa, en yüksek olasılıklı makul skoru kullan
            most_likely_reasonable = sorted(reasonable_scores, key=lambda x: x[1], reverse=True)[0][0]
            return most_likely_reasonable
        else:
            # Makul alternatif bulunamazsa, beklenen gol değerlerini kullan
            return f"{min(round(avg_home_goals), max_home_score)}-{min(round(avg_away_goals), max_away_score)}"
            
    def _get_outcome_from_score(self, score):
        """
        Belirli bir skordan maç sonucunu belirle - merkezi dinamik tahmin algoritması
        
        Bu fonksiyon, sistemdeki tüm kesin skor-sonuç dönüşümlerini yönetir.
        ConsensusFilter gibi diğer sınıflardaki benzer fonksiyonlar yerine bu merkezi
        fonksiyon kullanılmalıdır. Böylece tüm sistemde tutarlı sonuçlar sağlanır.

        Args:
            score: "3-1" formatında kesin skor string'i
            
        Returns:
            "HOME_WIN", "DRAW", "AWAY_WIN" veya None
        """
        try:
            if not score or '-' not in str(score):
                return None
            
            # Tuple formatından geldiyse (most_likely_score) ilk elemanı al
            if isinstance(score, tuple) and len(score) > 0:
                score = score[0]
                
            # Özel durumlar - en yaygın beraberlik skorları için hızlı kontrol
            score_str = str(score)
            if score_str == "0-0" or score_str == "1-1" or score_str == "2-2" or score_str == "3-3":
                logger.info(f"Beraberlik skoru tespit edildi: {score_str}, sonuç: DRAW")
                return "DRAW"
                
            parts = score_str.split('-')
            if len(parts) != 2:
                return None
                
            home_goals = int(parts[0])
            away_goals = int(parts[1])
            
            # Skorlar eşitse kesinlikle beraberlik
            if home_goals == away_goals:
                logger.info(f"Beraberlik skoru tespit edildi: {score_str}, sonuç: DRAW")
                return "DRAW"
            elif home_goals > away_goals:
                return "HOME_WIN"
            else:
                return "AWAY_WIN"
        except Exception as e:
            logger.error(f"Skor değerlendirme hatası: {str(e)}")
            return None
            
    def _update_all_predictions_from_score(self, prediction):
        """
        Kesin skora göre tüm bahis tahminlerini günceller.
        Bu fonksiyon, tahmin tutarlılığı sağlamak için merkezi bir mekanizmadır.
        Kesin skor değiştikten sonra çağrılmalıdır.
        
        HİYERARŞİK TAHMİN YAPISI (ÖNEMLİ):
        1. Kesin skor, tüm diğer tahminlerin temelidir
        2. Tüm bahis tahminleri kesin skordan türetilir, asla tersine değil
        3. Skordan türetilen tahminler, olasılık hesaplamalarından daha önceliklidir
        
        Args:
            prediction: Tahmin sonuçları sözlüğü
            
        Returns:
            dict: Tutarlı hale getirilmiş tahmin sonuçları
        """
        try:
            # Skor bilgisini al
            exact_score = prediction['predictions']['exact_score']
            home_score, away_score = map(int, exact_score.split('-'))
            total_goals = home_score + away_score
            
            # SKOR -> TÜM DİĞER TAHMİNLER HİYERARŞİSİ
            # --------------------------------------
            
            # 1. Maç sonucu güncelleme (Kesin skordan türetilir)
            match_outcome = self._get_outcome_from_score(exact_score)
            prediction['predictions']['most_likely_outcome'] = match_outcome
            prediction['predictions']['match_outcome'] = match_outcome
            
            logger.info(f"Kesin skor {exact_score} -> Maç sonucu {match_outcome} olarak belirlendi")
            
            # 2. KG VAR/YOK güncelleme (Kesin skordan türetilir)
            if 'both_teams_to_score' in prediction['predictions']['betting_predictions']:
                kg_var_yok = 'KG VAR' if home_score > 0 and away_score > 0 else 'KG YOK'
                prediction['predictions']['betting_predictions']['both_teams_to_score']['prediction'] = kg_var_yok
                
                # Olasılık değerini daha gerçekçi hesapla
                try:
                    # Monte Carlo veya Poisson olasılığını kullan eğer mevcutsa
                    try:
                        p_both_teams_score = prediction.get('raw_metrics', {}).get('p_both_teams_score', None)
                        if p_both_teams_score is not None:
                            # Monte Carlo veya Poisson hesaplaması mevcut
                            kg_var_prob = p_both_teams_score * 100.0  # Yüzdeliğe çevir
                        else:
                            # Gol beklentileriyle hesapla
                            home_exp = prediction['predictions']['expected_goals']['home']
                            away_exp = prediction['predictions']['expected_goals']['away']
                            # Poisson formülüyle her iki takımın da gol atma olasılığı
                            p_home_scores = 1.0 - math.exp(-home_exp)
                            p_away_scores = 1.0 - math.exp(-away_exp)
                            kg_var_prob = p_home_scores * p_away_scores * 100.0  # Yüzdeliğe çevir
                    except:
                        # Hesaplama hatası durumunda kesin skor bilgisini kullan ama %100/%0 yerine daha ılımlı değerler ver
                        if kg_var_yok == 'KG VAR':
                            kg_var_prob = 85.0  # KG VAR için %85 olasılık (daha gerçekçi)
                        else:
                            kg_var_prob = 15.0  # KG YOK için %15 olasılık (karşıt olasılık)
                except Exception as e:
                    logger.error(f"KG VAR/YOK olasılığı hesaplanırken hata: {str(e)}")
                    # Hata durumunda yine de değer belirle
                    if kg_var_yok == 'KG VAR':
                        kg_var_prob = 80.0
                    else:
                        kg_var_prob = 20.0
                
                # Son bir kontrol - olasılığın kesin skorla makul uyumu
                if (kg_var_yok == 'KG VAR' and kg_var_prob < 60.0) or (kg_var_yok == 'KG YOK' and kg_var_prob > 40.0):
                    # Olasılık, kesin skorla çelişiyor gibi görünüyor, düzelt
                    logger.warning(f"KG VAR/YOK olasılığı ({kg_var_prob:.2f}) tahminle ({kg_var_yok}) uyumsuz, düzeltiliyor")
                    if kg_var_yok == 'KG VAR':
                        kg_var_prob = 85.0  # Kesin skorla uyumlu minimum olasılık
                    else:
                        kg_var_prob = 15.0  # Kesin skorla uyumlu minimum olasılık
                
                # Virgülden sonra sadece 2 basamak göster
                kg_var_prob = round(kg_var_prob, 2)
                prediction['predictions']['betting_predictions']['both_teams_to_score']['probability'] = kg_var_prob
                
                logger.info(f"Kesin skor {exact_score} -> {kg_var_yok} olarak belirlendi (olasılık: {kg_var_prob:.2f})")
                
                # KG VAR/YOK için detaylı log
                if kg_var_yok == 'KG VAR':
                    logger.info(f"Her iki takım da gol attığı için ({home_score}>{0} VE {away_score}>{0}) KG VAR")
                else:
                    if home_score == 0:
                        logger.info(f"Ev sahibi gol atamadığı için ({home_score}={0}) KG YOK")
                    if away_score == 0:
                        logger.info(f"Deplasman gol atamadığı için ({away_score}={0}) KG YOK")
                
                # Frontend uyumluluğu için both_teams_score alanını da güncelle
                if 'both_teams_score' in prediction['predictions']['betting_predictions']:
                    prediction['predictions']['betting_predictions']['both_teams_score']['prediction'] = prediction['predictions']['betting_predictions']['both_teams_to_score']['prediction']
                    prediction['predictions']['betting_predictions']['both_teams_score']['probability'] = prediction['predictions']['betting_predictions']['both_teams_to_score']['probability']
            
            # 3. 2.5 ÜST/ALT güncelleme
            if 'over_2_5_goals' in prediction['predictions']['betting_predictions']:
                # Kesin skor tahmini
                if total_goals > 2:
                    # 3 ve üzeri gol varsa 2.5 ÜST
                    over_2_5 = '2.5 ÜST'
                else:
                    # 2 ve altı gol varsa 2.5 ALT
                    over_2_5 = '2.5 ALT'
                    
                prediction['predictions']['betting_predictions']['over_2_5_goals']['prediction'] = over_2_5
                logger.info(f"Skor temel alındı: {exact_score} skoru için {over_2_5} tahmini yapıldı")
                
                # Olasılık değerini daha gerçekçi hesapla
                try:
                    # Monte Carlo veya Poisson olasılığını kullan eğer mevcutsa
                    try:
                        p_over_2_5 = prediction.get('raw_metrics', {}).get('p_over_2_5', None)
                        if p_over_2_5 is not None:
                            # Monte Carlo veya Poisson hesaplaması mevcut
                            over_2_5_prob = p_over_2_5 * 100.0  # Yüzdeliğe çevir
                        else:
                            # Poisson dağılımıyla toplam 3+ gol olasılığını hesapla
                            home_exp = prediction['predictions']['expected_goals']['home']
                            away_exp = prediction['predictions']['expected_goals']['away']
                            total_exp = home_exp + away_exp
                            
                            # Toplam gol beklentisiyle 2.5 üstü olasılığı
                            # 0, 1, 2 gol olmama olasılığı hesapla (bunların dışı 3+ gol demek)
                            p_0_goals = math.exp(-total_exp)
                            p_1_goal = total_exp * math.exp(-total_exp)
                            p_2_goals = (total_exp**2 / 2) * math.exp(-total_exp)
                            
                            # 2.5 üstü olasılığı = 1 - (0, 1, 2 gol olasılığı)
                            over_2_5_prob = (1 - (p_0_goals + p_1_goal + p_2_goals)) * 100.0
                    except Exception as calc_error:
                        logger.warning(f"2.5 ÜST/ALT olasılığı Poisson ile hesaplanamadı: {str(calc_error)}")
                        # Hesaplama hatası durumunda tahmine göre belirle ama %100/%0 yerine daha ılımlı değerler ver
                        if over_2_5 == '2.5 ÜST':
                            over_2_5_prob = 78.0
                        else:
                            over_2_5_prob = 22.0
                except Exception as e:
                    logger.error(f"2.5 ÜST/ALT olasılığı hesaplanırken hata: {str(e)}")
                    # Hata durumunda yine de değer belirle
                    if over_2_5 == '2.5 ÜST':
                        over_2_5_prob = 75.0
                    else:
                        over_2_5_prob = 25.0
                
                # Son bir kontrol - olasılığın kesin skorla makul uyumu
                if (over_2_5 == '2.5 ÜST' and over_2_5_prob < 60.0) or (over_2_5 == '2.5 ALT' and over_2_5_prob > 40.0):
                    # Olasılık, kesin skorla çelişiyor gibi görünüyor, düzelt
                    logger.warning(f"2.5 ÜST/ALT olasılığı ({over_2_5_prob:.2f}) tahminle ({over_2_5}) uyumsuz, düzeltiliyor")
                    if over_2_5 == '2.5 ÜST':
                        over_2_5_prob = 78.0  # Kesin skorla uyumlu minimum olasılık
                    else:
                        over_2_5_prob = 22.0  # Kesin skorla uyumlu minimum olasılık
                
                # Virgülden sonra sadece 2 basamak göster
                over_2_5_prob = round(over_2_5_prob, 2)
                prediction['predictions']['betting_predictions']['over_2_5_goals']['probability'] = over_2_5_prob
            
            # 4. 3.5 ÜST/ALT güncelleme
            if 'over_3_5_goals' in prediction['predictions']['betting_predictions']:
                # Kesin skor tahmini
                if total_goals > 3:
                    # 4 ve üzeri gol varsa 3.5 ÜST
                    over_3_5 = '3.5 ÜST'
                else:
                    # 3 ve altı gol varsa 3.5 ALT
                    over_3_5 = '3.5 ALT'
                    
                prediction['predictions']['betting_predictions']['over_3_5_goals']['prediction'] = over_3_5
                logger.info(f"Skor temel alındı: {exact_score} skoru için {over_3_5} tahmini yapıldı")
                
                # Olasılık değerini daha gerçekçi hesapla
                try:
                    # Monte Carlo veya Poisson olasılığını kullan eğer mevcutsa
                    try:
                        p_over_3_5 = prediction.get('raw_metrics', {}).get('p_over_3_5', None)
                        if p_over_3_5 is not None:
                            # Monte Carlo veya Poisson hesaplaması mevcut
                            over_3_5_prob = p_over_3_5 * 100.0  # Yüzdeliğe çevir
                        else:
                            # Poisson dağılımıyla toplam 4+ gol olasılığını hesapla
                            home_exp = prediction['predictions']['expected_goals']['home']
                            away_exp = prediction['predictions']['expected_goals']['away']
                            total_exp = home_exp + away_exp
                            
                            # Toplam gol beklentisiyle 3.5 üstü olasılığı
                            # 0, 1, 2, 3 gol olmama olasılığı hesapla (bunların dışı 4+ gol demek)
                            p_0_goals = math.exp(-total_exp)
                            p_1_goal = total_exp * math.exp(-total_exp)
                            p_2_goals = (total_exp**2 / 2) * math.exp(-total_exp)
                            p_3_goals = (total_exp**3 / 6) * math.exp(-total_exp)
                            
                            # 3.5 üstü olasılığı = 1 - (0, 1, 2, 3 gol olasılığı)
                            over_3_5_prob = (1 - (p_0_goals + p_1_goal + p_2_goals + p_3_goals)) * 100.0
                    except Exception as calc_error:
                        logger.warning(f"3.5 ÜST/ALT olasılığı Poisson ile hesaplanamadı: {str(calc_error)}")
                        # Hesaplama hatası durumunda tahmine göre belirle ama %100/%0 yerine daha ılımlı değerler ver
                        if over_3_5 == '3.5 ÜST':
                            over_3_5_prob = 65.0
                        else:
                            over_3_5_prob = 35.0
                except Exception as e:
                    logger.error(f"3.5 ÜST/ALT olasılığı hesaplanırken hata: {str(e)}")
                    # Hata durumunda yine de değer belirle
                    if over_3_5 == '3.5 ÜST':
                        over_3_5_prob = 60.0
                    else:
                        over_3_5_prob = 40.0
                
                # Son bir kontrol - olasılığın kesin skorla makul uyumu
                if (over_3_5 == '3.5 ÜST' and over_3_5_prob < 55.0) or (over_3_5 == '3.5 ALT' and over_3_5_prob > 45.0):
                    # Olasılık, kesin skorla çelişiyor gibi görünüyor, düzelt
                    logger.warning(f"3.5 ÜST/ALT olasılığı ({over_3_5_prob:.2f}) tahminle ({over_3_5}) uyumsuz, düzeltiliyor")
                    if over_3_5 == '3.5 ÜST':
                        over_3_5_prob = 65.0  # Kesin skorla uyumlu minimum olasılık
                    else:
                        over_3_5_prob = 35.0  # Kesin skorla uyumlu minimum olasılık
                
                # Virgülden sonra sadece 2 basamak göster
                over_3_5_prob = round(over_3_5_prob, 2)
                prediction['predictions']['betting_predictions']['over_3_5_goals']['probability'] = over_3_5_prob
                    
            return prediction
        except Exception as e:
            logger.error(f"Tahmin güncelleme hatası: {str(e)}")
            return prediction

    def collect_training_data(self):
        """Tüm önbellekteki maçları kullanarak sinir ağları için eğitim verisi topla"""
        try:
            home_features = []
            home_targets = []
            away_features = []
            away_targets = []

            # Tahmin önbelleğindeki tüm maçları kontrol et
            for match_key, prediction in self.predictions_cache.items():
                try:
                    if not prediction or 'home_team' not in prediction or 'away_team' not in prediction:
                        continue

                    home_form = prediction['home_team'].get('form')
                    away_form = prediction['away_team'].get('form')

                    if not home_form or not away_form:
                        continue

                    # Hedef (target) değerleri - gerçek maç sonuçları
                    home_expected_goals = prediction['predictions']['expected_goals']['home']
                    away_expected_goals = prediction['predictions']['expected_goals']['away']

                    # Özellik vektörleri oluştur
                    home_data = self.prepare_data_for_neural_network(home_form, is_home=True)
                    away_data = self.prepare_data_for_neural_network(away_form, is_home=False)

                    if home_data is not None and away_data is not None:
                        home_features.append(home_data[0])
                        home_targets.append(home_expected_goals)

                        away_features.append(away_data[0])
                        away_targets.append(away_expected_goals)
                except Exception as e:
                    logger.error(f"Maç verisi işlenirken hata: {str(e)}")
                    continue

            # Eğer yeterli veri varsa sinir ağlarını eğit (daha az örnek ile de eğitim yapabilmek için)
            min_samples = 2  # Daha az örnekle başlayabilmek için eşiği düşürdük
            if len(home_features) >= min_samples and len(away_features) >= min_samples:
                logger.info(f"Sinir ağları için {len(home_features)} ev sahibi, {len(away_features)} deplasman örneği toplandı.")

                # Verileri numpy array'e dönüştür
                X_home = np.array(home_features)
                y_home = np.array(home_targets)
                X_away = np.array(away_features)
                y_away = np.array(away_targets)

                # Ev sahibi modeli eğit
                self.model_home = self.train_neural_network(X_home, y_home, is_home=True)

                # Deplasman modeli eğit
                self.model_away = self.train_neural_network(X_away, y_away, is_home=False)

                return True
            else:
                logger.info(f"Yeterli eğitim verisi bulunamadı, minimum {min_samples} örnek gerekli. "
                          f"Şu anda {len(home_features)} ev sahibi, {len(away_features)} deplasman örneği var.")
                return False
        except Exception as e:
            logger.error(f"Eğitim verisi toplanırken hata: {str(e)}")
            return False

    def analyze_opponent_strength(self, home_form, away_form):
        """Rakip takım gücünü analiz et ve karşılaştır - Gelişmiş momentum analizi ile"""
        try:
            def calculate_team_power(form_data):
                """Takımın güç puanını hesapla (0-100 arası)"""
                power = 50.0  # Başlangıç puanı
                
                # Son maçlar için ağırlıklı değerlendirme
                all_matches = form_data.get('recent_match_data', [])
                
                # Tüm maçları değerlendirme (son 21 maç - uzun vadeli kapasite)
                all_matches_power = 0
                recent_matches_power = 0
                
                if all_matches:
                    # Son 5 maç (güncel form) - %60-70 ağırlık
                    recent_matches = all_matches[:5]
                    
                    # Son 6-21 maç (genel kapasite) - %30-40 ağırlık
                    older_matches = all_matches[5:21] if len(all_matches) > 5 else []
                    
                    # Güncel form için güç hesapla
                    if recent_matches:
                        for match in recent_matches:
                            if match['result'] == 'W':
                                recent_matches_power += 8 if match['is_home'] else 10  # Deplasman galibiyeti daha değerli
                            elif match['result'] == 'D':
                                recent_matches_power += 4
                            else:  # Mağlubiyet
                                recent_matches_power -= 5 if match['is_home'] else 3  # Evdeki mağlubiyet daha kötü
                                
                            # Gol farkına göre ek puan
                            goal_diff = match['goals_scored'] - match['goals_conceded']
                            if match['result'] == 'W':
                                recent_matches_power += min(5, goal_diff * 2)  # Her fazla gol için +2 puan (max +5)
                            elif match['result'] == 'L':
                                recent_matches_power -= min(5, abs(goal_diff) * 1.5)  # Her fazla yenilen gol için -1.5 puan (max -5)
                    
                    # Genel kapasite için güç hesapla
                    if older_matches:
                        for match in older_matches:
                            if match['result'] == 'W':
                                all_matches_power += 4 if match['is_home'] else 5  # Deplasman galibiyeti daha değerli
                            elif match['result'] == 'D':
                                all_matches_power += 2
                            else:  # Mağlubiyet
                                all_matches_power -= 2.5 if match['is_home'] else 1.5  # Evdeki mağlubiyet daha kötü
                        
                        # Ortalama değere dönüştür
                        all_matches_power = all_matches_power / len(older_matches) * 10 if older_matches else 0
                    
                    # Ağırlıklı güç puanını hesapla
                    # Son 5 maç %65, genel kapasite %35 ağırlığa sahip
                    if recent_matches:
                        recent_weight = 0.65
                        all_weight = 0.35
                        power += recent_matches_power * recent_weight
                        power += all_matches_power * all_weight
                
                # Form verilerinden performans puanı ekle
                performance = form_data.get('form', {})
                if performance:
                    # Ağırlıklı gol ortalamasını kullan (eğer mevcutsa)
                    weighted_goals = performance.get('weighted_avg_goals_scored', performance.get('avg_goals_scored', 0))
                    power += min(12, weighted_goals * 6)  # Her gol ortalaması için +6 puan (max +12)
                    
                    # Ev/Deplasman performansını değerlendir
                    home_perf = performance.get('home_performance', {})
                    away_perf = performance.get('away_performance', {})
                    
                    if home_perf and away_perf:
                        # Ağırlıklı form puanlarını tercih et
                        home_form = home_perf.get('weighted_form_points', home_perf.get('form_points', 0))
                        away_form = away_perf.get('weighted_form_points', away_perf.get('form_points', 0))
                        power += (home_form + away_form) * 12  # Form puanlarına göre ek puan
                
                # Puanı 0-100 aralığında tut
                return max(0, min(100, power))
                
            # Gelişmiş Momentum Analizi için model_validation.py'dan gerekli fonksiyonu import et
            try:
                from model_validation import calculate_advanced_momentum
                momentum_analysis_available = True
                logger.info("Gelişmiş momentum analizi kullanılabilir")
            except ImportError:
                momentum_analysis_available = False
                logger.warning("Gelişmiş momentum analizi kullanılamıyor, temel analiz kullanılacak")
            
            # Momentum analizi sonuçları
            home_momentum = None
            away_momentum = None
            
            # Momentum analizi aktifse ve gerekli veriler mevcutsa uygula
            if momentum_analysis_available:
                # Ev sahibi için momentum analizi
                if 'detailed_data' in home_form and 'all' in home_form['detailed_data']:
                    try:
                        home_matches = home_form['detailed_data']['all']
                        # Son 5 maçın gol ortalaması hesapla
                        home_recent_goals = 0
                        home_recent_matches = min(5, len(home_matches))
                        if home_recent_matches > 0:
                            home_recent_goals = sum(match.get('goals_scored', 0) for match in home_matches[:home_recent_matches]) / home_recent_matches
                            
                        # Momentum hesaplama    
                        home_momentum = calculate_advanced_momentum(
                            home_matches, 
                            window=min(7, len(home_matches)),  # Son 7 maç (veya daha az)
                            recency_weight=1.8,  # Yakın maçları daha fazla ağırlıklandır
                            consider_opponent_strength=True
                        )
                        
                        # Momentum ve gol ortalamasını dengele
                        # Momentum yüksek ama gol ortalaması düşükse, momentum etkisini azalt
                        if home_momentum["momentum_score"] > 0 and home_recent_goals < 1.2:
                            home_momentum_adjusted = home_momentum["momentum_score"] * (0.5 + (home_recent_goals / 2.4))
                            logger.info(f"Ev sahibi momentum puanı gol ortalamasına ({home_recent_goals:.2f}) göre düşürüldü: {home_momentum['momentum_score']:.2f} -> {home_momentum_adjusted:.2f}")
                            home_momentum["momentum_score"] = home_momentum_adjusted
                        logger.info(f"Ev sahibi momentum analizi: {home_momentum}")
                    except Exception as e:
                        logger.error(f"Ev sahibi momentum analizi hatası: {str(e)}")
                
                # Deplasman için momentum analizi
                if 'detailed_data' in away_form and 'all' in away_form['detailed_data']:
                    try:
                        away_matches = away_form['detailed_data']['all']
                        # Son 5 maçın gol ortalaması hesapla
                        away_recent_goals = 0
                        away_recent_matches = min(5, len(away_matches))
                        if away_recent_matches > 0:
                            away_recent_goals = sum(match.get('goals_scored', 0) for match in away_matches[:away_recent_matches]) / away_recent_matches
                        
                        # Momentum hesaplama
                        away_momentum = calculate_advanced_momentum(
                            away_matches, 
                            window=min(7, len(away_matches)),  # Son 7 maç (veya daha az)
                            recency_weight=1.8,  # Yakın maçları daha fazla ağırlıklandır
                            consider_opponent_strength=True
                        )
                        
                        # Momentum ve gol ortalamasını dengele
                        # Momentum yüksek ama gol ortalaması düşükse, momentum etkisini azalt
                        if away_momentum["momentum_score"] > 0 and away_recent_goals < 1.2:
                            away_momentum_adjusted = away_momentum["momentum_score"] * (0.5 + (away_recent_goals / 2.4))
                            logger.info(f"Deplasman momentum puanı gol ortalamasına ({away_recent_goals:.2f}) göre düşürüldü: {away_momentum['momentum_score']:.2f} -> {away_momentum_adjusted:.2f}")
                            away_momentum["momentum_score"] = away_momentum_adjusted
                        logger.info(f"Deplasman momentum analizi: {away_momentum}")
                    except Exception as e:
                        logger.error(f"Deplasman momentum analizi hatası: {str(e)}")
            
            # Her iki takımın güç puanlarını hesapla
            home_power = calculate_team_power(home_form)
            away_power = calculate_team_power(away_form)
            
            # Düşük skorlu maç kontrolü - her iki takımın da gol beklentisi 1.0'in altındaysa momentum etkisini azalt
            is_low_scoring_match = False
            
            # Son maçlardaki ortalama gol sayılarını kontrol et
            if home_form and 'detailed_data' in home_form and 'all' in home_form['detailed_data']:
                home_recent_goals = 0
                recent_matches = home_form['detailed_data']['all'][:5]
                if recent_matches:
                    home_recent_goals = sum(match.get('goals_scored', 0) for match in recent_matches) / len(recent_matches)
                
                if away_form and 'detailed_data' in away_form and 'all' in away_form['detailed_data']:
                    away_recent_goals = 0
                    recent_matches = away_form['detailed_data']['all'][:5]
                    if recent_matches:
                        away_recent_goals = sum(match.get('goals_scored', 0) for match in recent_matches) / len(recent_matches)
                    
                    # Her iki takımın da gol beklentisi 1.0'in altındaysa düşük skorlu maç
                    if home_recent_goals < 1.0 and away_recent_goals < 1.0:
                        is_low_scoring_match = True
                        logger.info(f"Düşük skorlu maç tespit edildi. Ev: {home_recent_goals:.2f}, Dep: {away_recent_goals:.2f}")
            
            # Momentum analizine göre güç puanlarını güncelle
            if home_momentum:
                # Momentum skoru -1 ile 1 arasında, bunu puan ayarlaması olarak dönüştür
                momentum_factor = 0.3 if is_low_scoring_match else 1.0  # Düşük skorlu maçlarda momentum etkisini %70 azalt
                momentum_adjustment = home_momentum['momentum_score'] * 10 * momentum_factor  # -10 ile +10 arasında (veya %30'u)
                
                # Form ivmesine göre ek düzeltme
                acceleration_factor = 0.3 if is_low_scoring_match else 1.0  # Düşük skorlu maçlarda ivme etkisini %70 azalt
                acceleration_adjustment = home_momentum['form_acceleration'] * 5 * acceleration_factor  # -5 ile +5 arasında (veya %30'u)
                
                # Düşük skorlu maçlarda özel mesaj
                if is_low_scoring_match:
                    logger.info(f"Düşük skorlu maç olduğu için momentum ve form ivmesi etkisi %70 azaltıldı")
                
                # Güç puanını güncelle
                home_power = max(0, min(100, home_power + momentum_adjustment + acceleration_adjustment))
                logger.info(f"Ev sahibi güç puanı momentum analizine göre güncellendi: {momentum_adjustment:.2f} + {acceleration_adjustment:.2f} puan")
            
            if away_momentum:
                # Momentum skoru -1 ile 1 arasında, bunu puan ayarlaması olarak dönüştür
                momentum_factor = 0.3 if is_low_scoring_match else 1.0  # Düşük skorlu maçlarda momentum etkisini %70 azalt
                momentum_adjustment = away_momentum['momentum_score'] * 10 * momentum_factor  # -10 ile +10 arasında (veya %30'u)
                
                # Form ivmesine göre ek düzeltme
                acceleration_factor = 0.3 if is_low_scoring_match else 1.0  # Düşük skorlu maçlarda ivme etkisini %70 azalt
                acceleration_adjustment = away_momentum['form_acceleration'] * 5 * acceleration_factor  # -5 ile +5 arasında (veya %30'u)
                
                # Güç puanını güncelle
                away_power = max(0, min(100, away_power + momentum_adjustment + acceleration_adjustment))
                logger.info(f"Deplasman güç puanı momentum analizine göre güncellendi: {momentum_adjustment:.2f} + {acceleration_adjustment:.2f} puan")
            
            # Güç farkını hesapla (100 puan üzerinden)
            power_diff = home_power - away_power
            
            # Relative strength'i 0-1 aralığında normalize et
            relative_strength = 0.5 + (power_diff / 200)  # -100 ile +100 arası farkı 0-1 aralığına dönüştür
            relative_strength = max(0.1, min(0.9, relative_strength))  # Aşırı uç değerleri engelle
            
            # Göreli güç yüzdelerini hesapla
            total_power = home_power + away_power
            home_power_percentage = (home_power / total_power * 100) if total_power > 0 else 50
            away_power_percentage = (away_power / total_power * 100) if total_power > 0 else 50
            
            # Güç farkını yüzde olarak hesapla (0-100 arası)
            strength_ratio = round(abs(power_diff) / max(1, (home_power + away_power) / 2) * 100, 1)
            
            # Sonuç sözlüğü
            result = {
                'home_power': home_power,
                'away_power': away_power,
                'home_power_percentage': home_power_percentage,
                'away_power_percentage': away_power_percentage,
                'power_difference': power_diff,
                'relative_strength': relative_strength,
                'strength_ratio': strength_ratio  # Güç farkı yüzdesi (artık 0'dan büyük olacak)
            }
            
            # Momentum verilerini de sonuca ekle (eğer varsa)
            if home_momentum:
                result['home_momentum'] = home_momentum
            if away_momentum:
                result['away_momentum'] = away_momentum
            
            return result
            
        except Exception as e:
            logger.error(f"Rakip analizi sırasında hata: {str(e)}")
            return {
                'home_power': 50,
                'away_power': 50,
                'home_power_percentage': 50,
                'away_power_percentage': 50,
                'power_difference': 0,
                'relative_strength': 0.5,
                'strength_ratio': 0
            }


    def analyze_head_to_head(self, home_team_id, away_team_id, home_team_name, away_team_name):
        """İki takım arasındaki önceki karşılaşmaları analiz et"""
        try:
            # Son 3 yıllık H2H maçları al
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1095)  # Son 3 yıl

            url = "https://apiv3.apifootball.com/"
            params = {
                'action': 'get_H2H',
                'firstTeamId': home_team_id,
                'secondTeamId': away_team_id,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'APIkey': self.api_key
            }

            response = requests.get(url, params=params)

            if response.status_code != 200:
                logger.error(f"H2H API hatası: {response.status_code}")
                return None

            h2h_data = response.json()

            if not isinstance(h2h_data, dict) or 'firstTeam_VS_secondTeam' not in h2h_data:
                logger.error(f"Beklenmeyen H2H API yanıtı: {h2h_data}")
                return None

            h2h_matches = h2h_data.get('firstTeam_VS_secondTeam', [])

            if not h2h_matches:
                logger.info(f"H2H maç bulunamadı: {home_team_name} vs {away_team_name}")
                return {
                    'home_wins': 0,
                    'away_wins': 0,
                    'draws': 0,
                    'total_matches': 0,
                    'avg_home_goals': 0,
                    'avg_away_goals': 0,
                    'recent_matches': []
                }

            # H2H istatistikleri hesapla
            home_wins = 0
            away_wins = 0
            draws = 0
            total_home_goals = 0
            total_away_goals = 0
            recent_matches = []

            for match in h2h_matches:
                match_home_team = match.get('match_hometeam_name', '')
                match_away_team = match.get('match_awayteam_name', '')
                match_home_score = int(match.get('match_hometeam_score', 0) or 0)
                match_away_score = int(match.get('match_awayteam_score', 0) or 0)
                match_date = match.get('match_date', '')
                match_status = match.get('match_status', '')

                # Sadece tamamlanmış maçları dikkate al
                if match_status not in ['FT', 'AP', 'AET', 'Finished']:
                    continue

                # Maç sonucunu hesapla
                if match_home_team == home_team_name:
                    # Ev sahibi takım aynı
                    if match_home_score > match_away_score:
                        home_wins += 1
                    elif match_home_score < match_away_score:
                        away_wins += 1
                    else:
                        draws += 1

                    total_home_goals += match_home_score
                    total_away_goals += match_away_score
                else:
                    # Takımlar yer değiştirmiş
                    if match_home_score > match_away_score:
                        away_wins += 1
                    elif match_home_score < match_away_score:
                        home_wins += 1
                    else:
                        draws += 1

                    total_home_goals += match_away_score
                    total_away_goals += match_home_score

                # Maç detaylarını ekle
                if match_home_team == home_team_name:
                    result = 'W' if match_home_score > match_away_score else 'D' if match_home_score == match_away_score else 'L'
                    recent_matches.append({
                        'date': match_date,
                        'league': match.get('league_name', ''),
                        'home_score': match_home_score,
                        'away_score': match_away_score,
                        'result': result
                    })
                else:
                    result = 'W' if match_away_score > match_home_score else 'D' if match_home_score == match_away_score else 'L'
                    recent_matches.append({
                        'date': match_date,
                        'league': match.get('league_name', ''),
                        'home_score': match_away_score,
                        'away_score': match_home_score,
                        'result': result
                    })

            # Son maçları tarihe göre sırala - iyileştirilmiş tarih sıralaması
            try:
                # Tarihleri datetime nesnelerine dönüştür, sonra sırala
                for match in recent_matches:
                    # API'den gelen tarih formatı: YYYY-MM-DD
                    match['parsed_date'] = datetime.strptime(match['date'], '%Y-%m-%d')
                
                # Tarihe göre sırala, en yeni maçlar üstte
                recent_matches.sort(key=lambda x: x['parsed_date'], reverse=True)
                
                # Ekstra parsed_date alanını temizle
                for match in recent_matches:
                    if 'parsed_date' in match:
                        del match['parsed_date']
                        
                logger.info(f"H2H maçları başarıyla tarihe göre sıralandı: {len(recent_matches)} maç")
            except Exception as e:
                logger.error(f"H2H maçları sıralanırken hata: {str(e)}")
                # Hata durumunda standart string-based sıralamayı dene
                recent_matches.sort(key=lambda x: x['date'], reverse=True)

            total_matches = home_wins + away_wins + draws
            avg_home_goals = total_home_goals / total_matches if total_matches > 0 else 0
            avg_away_goals = total_away_goals / total_matches if total_matches > 0 else 0

            return {
                'home_wins': home_wins,
                'away_wins': away_wins,
                'draws': draws,
                'total_matches': total_matches,
                'avg_home_goals': avg_home_goals,
                'avg_away_goals': avg_away_goals,
                'recent_matches': recent_matches[:5]  # Sadece son 5 maçı döndür
            }

        except Exception as e:
            logger.error(f"H2H analizi sırasında hata: {str(e)}")
            return None