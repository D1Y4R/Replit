import numpy as np
import os
import pickle
import logging
from scipy.stats import poisson, norm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedScorePredictor:
    """Gelişmiş skor tahmin sınıfı"""

    def __init__(self):
        """Gelişmiş tahmin modellerinin başlatılması"""
        self.home_poisson_model = None
        self.away_poisson_model = None
        self.home_ensemble_model = None 
        self.away_ensemble_model = None
        self.scaler = StandardScaler()

        # Modelleri yükle
        self.load_models()
        logger.info("ZIP ve Ensemble tahmin modeli başlatıldı")

    def load_models(self):
        """Kaydedilmiş modelleri yükle veya yeni modeller oluştur"""
        try:
            # Model dosyaları var mı kontrol et
            model_files_exist = all([
                os.path.exists('home_poisson_model.pkl'),
                os.path.exists('away_poisson_model.pkl'),
                os.path.exists('home_ensemble_model.pkl'),
                os.path.exists('away_ensemble_model.pkl')
            ])

            if model_files_exist:
                logger.info("Kaydedilmiş ZIP ve Ensemble model dosyaları yükleniyor...")
                with open('home_poisson_model.pkl', 'rb') as f:
                    self.home_poisson_model = pickle.load(f)

                with open('away_poisson_model.pkl', 'rb') as f:
                    self.away_poisson_model = pickle.load(f)

                with open('home_ensemble_model.pkl', 'rb') as f:
                    self.home_ensemble_model = pickle.load(f)

                with open('away_ensemble_model.pkl', 'rb') as f:
                    self.away_ensemble_model = pickle.load(f)

                logger.info("Gelişmiş tahmin modelleri başarıyla yüklendi")
            else:
                logger.info("Kaydedilmiş model dosyaları bulunamadı, yeni modeller oluşturuluyor...")
                # Varsayılan modeller oluştur
                self.home_poisson_model = {"intercept": -0.5, "home_attack": 0.2, "away_defense": 0.1}
                self.away_poisson_model = {"intercept": -0.8, "away_attack": 0.2, "home_defense": 0.1}

                # Ensemble modeller için
                self.home_ensemble_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                self.away_ensemble_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")
            # Hata durumunda varsayılan modeller oluştur
            self.home_poisson_model = {"intercept": -0.5, "home_attack": 0.2, "away_defense": 0.1}
            self.away_poisson_model = {"intercept": -0.8, "away_attack": 0.2, "home_defense": 0.1}

            # Ensemble modeller için
            self.home_ensemble_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            self.away_ensemble_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

    def save_models(self):
        """Modelleri kaydet"""
        try:
            if self.home_poisson_model and self.away_poisson_model and self.home_ensemble_model and self.away_ensemble_model:
                with open('home_poisson_model.pkl', 'wb') as f:
                    pickle.dump(self.home_poisson_model, f)

                with open('away_poisson_model.pkl', 'wb') as f:
                    pickle.dump(self.away_poisson_model, f)

                with open('home_ensemble_model.pkl', 'wb') as f:
                    pickle.dump(self.home_ensemble_model, f)

                with open('away_ensemble_model.pkl', 'wb') as f:
                    pickle.dump(self.away_ensemble_model, f)

                logger.info("Tahmin modelleri başarıyla kaydedildi")
                return True
            else:
                logger.warning("Model dosyaları kaydedilemedi, önce modellerin eğitilmesi gerekiyor")
                return False
        except Exception as e:
            logger.error(f"Model kaydetme hatası: {e}")
            return False

    def train_models(self, cached_predictions):
        """Tahmin modellerini eğit"""
        try:
            # Eğitim verileri için gerekli değişkenleri hazırla
            home_features = []
            home_targets = []
            away_features = []
            away_targets = []

            # Önbellekteki tüm tahminleri dolaş
            for match_key, prediction in cached_predictions.items():
                if not isinstance(prediction, dict) or 'home_team' not in prediction or 'away_team' not in prediction:
                    continue

                home_form = prediction.get('home_team', {}).get('form')
                away_form = prediction.get('away_team', {}).get('form')

                if not home_form or not away_form:
                    continue

                # Gol beklentilerini hedef değerler olarak al
                home_goals_expected = prediction.get('predictions', {}).get('expected_goals', {}).get('home', 0)
                away_goals_expected = prediction.get('predictions', {}).get('expected_goals', {}).get('away', 0)

                # Zero-inflated Poisson için özellikler
                try:
                    # Ev sahibi için özellikler
                    home_attack = home_form.get('home_performance', {}).get('weighted_avg_goals_scored', 0)
                    home_defense = home_form.get('home_performance', {}).get('weighted_avg_goals_conceded', 0)
                    away_attack = away_form.get('away_performance', {}).get('weighted_avg_goals_scored', 0)
                    away_defense = away_form.get('away_performance', {}).get('weighted_avg_goals_conceded', 0)

                    # Ensemble model için daha zengin özellik kümesi
                    # Ev sahibi için
                    home_feature_set = [
                        home_form.get('home_performance', {}).get('avg_goals_scored', 0),
                        home_form.get('home_performance', {}).get('avg_goals_conceded', 0),
                        home_form.get('home_performance', {}).get('weighted_avg_goals_scored', 0),
                        home_form.get('home_performance', {}).get('weighted_avg_goals_conceded', 0),
                        home_form.get('home_performance', {}).get('form_points', 0),
                        home_form.get('bayesian', {}).get('home_lambda_scored', 0),
                        home_form.get('bayesian', {}).get('home_lambda_conceded', 0),
                        home_form.get('recent_matches', 0) / 21,  # Normalize
                        away_form.get('away_performance', {}).get('weighted_avg_goals_scored', 0),
                        away_form.get('away_performance', {}).get('weighted_avg_goals_conceded', 0)
                    ]

                    # Deplasman için
                    away_feature_set = [
                        away_form.get('away_performance', {}).get('avg_goals_scored', 0),
                        away_form.get('away_performance', {}).get('avg_goals_conceded', 0),
                        away_form.get('away_performance', {}).get('weighted_avg_goals_scored', 0),
                        away_form.get('away_performance', {}).get('weighted_avg_goals_conceded', 0),
                        away_form.get('away_performance', {}).get('form_points', 0),
                        away_form.get('bayesian', {}).get('away_lambda_scored', 0),
                        away_form.get('bayesian', {}).get('away_lambda_conceded', 0),
                        away_form.get('recent_matches', 0) / 21,  # Normalize
                        home_form.get('home_performance', {}).get('weighted_avg_goals_scored', 0),
                        home_form.get('home_performance', {}).get('weighted_avg_goals_conceded', 0)
                    ]

                    # Poisson Model için basit özellikler
                    poisson_home_features = {
                        "home_attack": home_attack,
                        "away_defense": away_defense
                    }

                    poisson_away_features = {
                        "away_attack": away_attack,
                        "home_defense": home_defense
                    }

                    # Eğitim verilerine ekle
                    if all(x is not None for x in home_feature_set) and all(x is not None for x in away_feature_set):
                        home_features.append(home_feature_set)
                        home_targets.append(home_goals_expected)

                        away_features.append(away_feature_set)
                        away_targets.append(away_goals_expected)

                except Exception as e:
                    logger.error(f"Maç verisi işlenirken hata: {e}")
                    continue

            # Yeterli veri var mı kontrol et
            if len(home_features) < 10 or len(away_features) < 10:
                logger.warning(f"Yeterli eğitim verisi yok: {len(home_features)} ev sahibi, {len(away_features)} deplasman örneği")
                return False

            # Zero-inflated Poisson modelleri için basit regresyon katsayıları
            # Gerçek bir ZIP fit yerine basit katsayılar kullanıyoruz
            self.home_poisson_model = {
                "intercept": np.mean([t - (0.2 * f[0] + 0.1 * f[9]) for t, f in zip(home_targets, home_features)]),
                "home_attack": 0.2,  # Varsayılan katsayı
                "away_defense": 0.1   # Varsayılan katsayı
            }

            self.away_poisson_model = {
                "intercept": np.mean([t - (0.2 * f[2] + 0.1 * f[9]) for t, f in zip(away_targets, away_features)]),
                "away_attack": 0.2,   # Varsayılan katsayı  
                "home_defense": 0.1    # Varsayılan katsayı
            }

            # Ensemble modelleri eğit
            X_home = np.array(home_features)
            y_home = np.array(home_targets)
            X_away = np.array(away_features)
            y_away = np.array(away_targets)

            # Verileri standartlaştır
            X_home_scaled = self.scaler.fit_transform(X_home)
            X_away_scaled = self.scaler.transform(X_away)

            # Ensemble modelleri eğit
            self.home_ensemble_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            self.home_ensemble_model.fit(X_home_scaled, y_home)

            self.away_ensemble_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            self.away_ensemble_model.fit(X_away_scaled, y_away)

            logger.info(f"Modeller {len(home_features)} örnek ile başarıyla eğitildi")

            # Modelleri kaydet
            self.save_models()

            return True
        except Exception as e:
            logger.error(f"Model eğitimi sırasında hata: {e}")
            return False

    def calculate_score_probabilities(self, lambda_home, lambda_away, max_goals=6):
        """Poisson dağılımını kullanarak skor olasılıklarını hesapla"""
        from scipy.stats import poisson

        # Gerçek skor dağılımları için düzeltme faktörleri - GELİŞTİRİLMİŞ VERSİYON
        # Futbol maçlarında gerçekte gözlemlenen skor dağılımları
        # Poisson dağılımından bazı sapmalar gösterir
        # Özellikle beklenen gol değerleri yüksek olduğunda (2.0+) bu faktörler daha dengeli olmalıdır
        
        # Temel düzeltme faktörleri
        correction_factors = {
            "0-0": 1.10,  # 0-0 skoru gerçekte daha sık görünür (1.15 -> 1.10)
            "1-0": 1.08,  # Ev sahibi 1-0 galibiyeti biraz daha sık (1.10 -> 1.08)
            "0-1": 0.98,  # Deplasman 0-1 galibiyeti (0.95 -> 0.98)
            "1-1": 1.12,  # 1-1 beraberliği daha sık (1.20 -> 1.12) (AZALTILDI)
            "2-1": 1.05,  # Ev sahibi 2-1 galibiyeti
            "1-2": 0.98,  # Deplasman 1-2 galibiyeti (0.95 -> 0.98)
            "2-0": 1.03,  # 2-0 galibiyeti (1.05 -> 1.03)
            "0-2": 0.95,  # 0-2 deplasmanı (0.90 -> 0.95)
            "2-2": 1.20,  # 2-2 beraberliği daha sık (1.15 -> 1.20) (ARTIRILDI)
            "3-0": 1.00,  # 3-0 normal
            "0-3": 0.95,  # 0-3 (0.90 -> 0.95)
            "3-1": 1.02,  # 3-1 (1.00 -> 1.02)
            "1-3": 0.95,  # 1-3 (0.90 -> 0.95)
            "3-2": 1.08,  # 3-2 biraz daha sık (1.05 -> 1.08) (ARTIRILDI)
            "2-3": 1.05,  # 2-3 (1.00 -> 1.05) (ARTIRILDI)
            "3-3": 1.18,  # 3-3 daha sık (1.10 -> 1.18) (ARTIRILDI)
            "4-0": 0.98,  # 4-0 (0.95 -> 0.98)
            "0-4": 0.90,  # 0-4 (0.85 -> 0.90)
            "4-1": 1.00,  # 4-1 (0.95 -> 1.00) (ARTIRILDI)
            "1-4": 0.90,  # 1-4 (0.85 -> 0.90)
            "4-2": 1.03,  # 4-2 (0.95 -> 1.03) (ARTIRILDI)
            "2-4": 0.95,  # 2-4 (0.90 -> 0.95)
            "4-3": 1.00,  # 4-3 (0.95 -> 1.00) (ARTIRILDI)
            "3-4": 0.95,  # 3-4 (0.90 -> 0.95)
            "4-4": 1.10,  # 4-4 biraz daha sık (1.05 -> 1.10) (ARTIRILDI)
            "5-0": 0.90,  # 5-0 (0.85 -> 0.90)
            "0-5": 0.85,  # 0-5 (0.80 -> 0.85)
            "5-1": 0.95,  # 5-1 (0.85 -> 0.95) (ARTIRILDI)
            "1-5": 0.90,  # 1-5 (0.80 -> 0.90) (ARTIRILDI)
            "5-2": 1.00,  # 5-2 (EKLENDI)
            "2-5": 0.95,  # 2-5 (EKLENDI)
            "5-3": 1.00,  # 5-3 (EKLENDI)
            "3-5": 0.95,  # 3-5 (EKLENDI)
        }

        score_probs = {}
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                # Her bir skor kombinasyonunun olasılığını hesapla
                score = f"{h}-{a}"
                prob = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)

                # Gerçek dağılımlara uygun düzeltme faktörü uygula
                if score in correction_factors:
                    prob *= correction_factors[score]
                elif h == a and h > 4:  # Yüksek skorlu beraberlikler
                    prob *= 1.0  # Çok yüksek skorlu beraberlikler için normal faktör
                elif h > 4 and a > 4:  # Çok yüksek skorlu maçlar
                    prob *= 0.8  # Çok yüksek skorlu maçları azalt
                elif h > 4:  # Ev sahibi için çok yüksek skorlar
                    prob *= 0.85
                elif a > 4:  # Deplasman için çok yüksek skorlar
                    prob *= 0.80

                score_probs[score] = prob

        # Olasılıkları normalize et
        total_prob = sum(score_probs.values())
        if total_prob > 0:
            for score in score_probs:
                score_probs[score] /= total_prob

        return score_probs

    def predict_match(self, home_form, away_form, cached_predictions=None, model_weight=0.4, simulations=5000):
        """Maç sonucunu tahmin et - Geliştirilmiş tutarlı algoritma"""
        try:
            # Modeller yüklü değilse eğit
            if (not self.home_poisson_model or not self.away_poisson_model or 
                not self.home_ensemble_model or not self.away_ensemble_model) and cached_predictions:
                logger.info("Tahmin modelleri eğitiliyor...")
                self.train_models(cached_predictions)

            # Zero-inflated Poisson için özellik vektörleri
            home_attack = home_form.get('home_performance', {}).get('weighted_avg_goals_scored', 0)
            home_defense = home_form.get('home_performance', {}).get('weighted_avg_goals_conceded', 0)
            away_attack = away_form.get('away_performance', {}).get('weighted_avg_goals_scored', 0)
            away_defense = away_form.get('away_performance', {}).get('weighted_avg_goals_conceded', 0)

            # Zero-inflated Poisson model tahmini
            # log(lambda) = intercept + coef1*x1 + coef2*x2 + ...
            # lambda = exp(intercept + coef1*x1 + coef2*x2 + ...)
            if isinstance(self.home_poisson_model, dict) and isinstance(self.away_poisson_model, dict):
                log_lambda_home = (self.home_poisson_model.get("intercept", 0) + 
                                  self.home_poisson_model.get("home_attack", 0) * home_attack + 
                                  self.home_poisson_model.get("away_defense", 0) * away_defense)

                log_lambda_away = (self.away_poisson_model.get("intercept", 0) + 
                                  self.away_poisson_model.get("away_attack", 0) * away_attack + 
                                  self.away_poisson_model.get("home_defense", 0) * home_defense)

                lambda_home = np.exp(log_lambda_home)
                lambda_away = np.exp(log_lambda_away)
            else:
                # Varsayılan değerler
                lambda_home = 1.5
                lambda_away = 1.0

            # Ensemble model için özellik vektörleri
            home_feature_set = [
                home_form.get('home_performance', {}).get('avg_goals_scored', 0),
                home_form.get('home_performance', {}).get('avg_goals_conceded', 0),
                home_form.get('home_performance', {}).get('weighted_avg_goals_scored', 0),
                home_form.get('home_performance', {}).get('weighted_avg_goals_conceded', 0),
                home_form.get('home_performance', {}).get('form_points', 0),
                home_form.get('bayesian', {}).get('home_lambda_scored', 0),
                home_form.get('bayesian', {}).get('home_lambda_conceded', 0),
                home_form.get('recent_matches', 0) / 21,  # Normalize
                away_form.get('away_performance', {}).get('weighted_avg_goals_scored', 0),
                away_form.get('away_performance', {}).get('weighted_avg_goals_conceded', 0)
            ]

            away_feature_set = [
                away_form.get('away_performance', {}).get('avg_goals_scored', 0),
                away_form.get('away_performance', {}).get('avg_goals_conceded', 0),
                away_form.get('away_performance', {}).get('weighted_avg_goals_scored', 0),
                away_form.get('away_performance', {}).get('weighted_avg_goals_conceded', 0),
                away_form.get('away_performance', {}).get('form_points', 0),
                away_form.get('bayesian', {}).get('away_lambda_scored', 0),
                away_form.get('bayesian', {}).get('away_lambda_conceded', 0),
                away_form.get('recent_matches', 0) / 21,  # Normalize
                home_form.get('home_performance', {}).get('weighted_avg_goals_scored', 0),
                home_form.get('home_performance', {}).get('weighted_avg_goals_conceded', 0)
            ]

            # Ensemble model tahmini
            try:
                X_home = np.array(home_feature_set).reshape(1, -1)
                X_away = np.array(away_feature_set).reshape(1, -1)

                X_home_scaled = self.scaler.transform(X_home)
                X_away_scaled = self.scaler.transform(X_away)

                ensemble_home_goals = float(self.home_ensemble_model.predict(X_home_scaled)[0])
                ensemble_away_goals = float(self.away_ensemble_model.predict(X_away_scaled)[0])

                # Tahmin edilen değerleri pozitif yap
                ensemble_home_goals = max(0.5, ensemble_home_goals)
                ensemble_away_goals = max(0.5, ensemble_away_goals)
            except Exception as e:
                logger.error(f"Ensemble tahmin hatası: {e}")
                ensemble_home_goals = 1.5
                ensemble_away_goals = 1.0

            # Ensemble ve ZIP modellerini birleştir - Modellerin tutarlılığını kontrol et
            # Eğer model tahminleri arasında büyük fark varsa, ağırlıklandırmayı dengele
            if abs(ensemble_home_goals - lambda_home) > 1.5:
                model_weight_home = 0.5  # Eşit ağırlık kullan
                logger.info(f"Ev sahibi için model tahminleri arasında büyük fark: Ensemble={ensemble_home_goals:.2f}, ZIP={lambda_home:.2f}")
            else:
                model_weight_home = model_weight

            if abs(ensemble_away_goals - lambda_away) > 1.5:
                model_weight_away = 0.5  # Eşit ağırlık kullan
                logger.info(f"Deplasman için model tahminleri arasında büyük fark: Ensemble={ensemble_away_goals:.2f}, ZIP={lambda_away:.2f}")
            else:
                model_weight_away = model_weight

            weighted_home_goals = model_weight_home * ensemble_home_goals + (1 - model_weight_home) * lambda_home
            weighted_away_goals = model_weight_away * ensemble_away_goals + (1 - model_weight_away) * lambda_away

            # Yüksek skorlara izin veren DAHA AGRESİF sınırlandırma sistemi
            def realistic_limit(value, is_home=True, lower_bound=0.5, upper_bound=4.5, team_str=None, opp_def=None):
                # Takım güçlerini dikkate alarak sınırları ayarla
                if team_str is not None and opp_def is not None:
                    # Ofansif güç × defansif zayıflık faktörü hesapla
                    offense_defense_factor = team_str * opp_def
                    
                    # Çok golcü takım ve kötü savunma durumunu algıla
                    extreme_case = False
                    
                    # Eğer rakip çok fazla gol yiyorsa (2.0'dan fazla)
                    extreme_conceded = 0
                    if is_home and away_form.get('avg_goals_conceded', 0) > 2.0:
                        extreme_conceded = away_form.get('avg_goals_conceded', 0)
                        extreme_case = True
                    elif not is_home and home_form.get('avg_goals_conceded', 0) > 2.0:
                        extreme_conceded = home_form.get('avg_goals_conceded', 0)
                        extreme_case = True
                    
                    # Extreme durumlarda üst sınırı daha agresif artır
                    if extreme_case:
                        logger.info(f"{'Ev' if is_home else 'Deplasman'} rakibi çok gol yiyor: {extreme_conceded:.2f} gol/maç")
                        upper_bound = max(5.0, extreme_conceded * 1.5)  # En az 5.0 veya ortalama yenilen gol x 1.5
                    
                    # Normal ofansif avantaj durumunda
                    elif offense_defense_factor > 1.2:  # Ofansif avantaj eşiğini düşürdük (1.3 -> 1.2)
                        # Avantaj ne kadar büyükse üst sınır o kadar yüksek
                        upper_bound_adjustment = min(2.0, offense_defense_factor * 0.8)  # Max artış 2.0'a çıkarıldı
                        upper_bound += upper_bound_adjustment
                        # Log mesajı ekle
                        logger.info(f"{'Ev' if is_home else 'Deplasman'} takımı ofansif avantajlı: Üst sınır {upper_bound:.2f}'e ayarlandı")
                
                if value < lower_bound:
                    return lower_bound
                elif value > upper_bound:
                    # Daha az kısıtlayıcı logaritmik düzeltme
                    scaling_factor = 0.7 if is_home else 0.8  # Çok daha esnek sınırlama (0.4/0.5 -> 0.7/0.8)
                    adjusted_value = upper_bound + scaling_factor * np.log1p(value - upper_bound)
                    logger.info(f"{'Ev' if is_home else 'Deplasman'} takımı için yüksek gol beklentisi: {value:.2f} -> {adjusted_value:.2f}")
                    return adjusted_value
                return value

            # Takım güçlerini hesapla (home_form ve away_form içindeki değerlerden)
            home_offensive_strength = home_form.get('avg_goals_scored', 1.5)
            away_offensive_strength = away_form.get('avg_goals_scored', 1.2)
            home_defensive_weakness = 1 / max(0.5, home_form.get('avg_goals_conceded', 1.0))  # Max değeri 0.8 -> 0.5 (daha yüksek defansif zayıflık)
            away_defensive_weakness = 1 / max(0.5, away_form.get('avg_goals_conceded', 1.2))
            
            # Takım özelliklerini kullanarak daha esnek sınırlar uygula
            weighted_home_goals = realistic_limit(
                weighted_home_goals, 
                is_home=True, 
                upper_bound=4.5,  # Temel üst sınır çok daha yüksek (3.5 -> 4.5)
                team_str=home_offensive_strength,
                opp_def=away_defensive_weakness
            )
            weighted_away_goals = realistic_limit(
                weighted_away_goals, 
                is_home=False, 
                upper_bound=4.0,  # Deplasman için de yüksek üst sınır (3.2 -> 4.0)
                team_str=away_offensive_strength,
                opp_def=home_defensive_weakness
            )
            
            # Yüksek gol beklentilerini koruma
            # Eğer weighted_home_goals veya weighted_away_goals 3'ten büyükse, en az %60'ını koru
            # Bu, çok yüksek beklenti değerlerinin fazla düşürülmemesini sağlar
            if weighted_home_goals > 3.0:
                original_home_expect = max(lambda_home, ensemble_home_goals)
                if weighted_home_goals < 0.6 * original_home_expect:
                    weighted_home_goals = 0.6 * original_home_expect
                    logger.info(f"Ev takımı yüksek gol beklentisi korundu: {weighted_home_goals:.2f}")
                    
            if weighted_away_goals > 3.0:
                original_away_expect = max(lambda_away, ensemble_away_goals)
                if weighted_away_goals < 0.6 * original_away_expect:
                    weighted_away_goals = 0.6 * original_away_expect
                    logger.info(f"Deplasman takımı yüksek gol beklentisi korundu: {weighted_away_goals:.2f}")

            logger.info(f"Final gol beklentileri: Ev sahibi={weighted_home_goals:.2f}, Deplasman={weighted_away_goals:.2f}")

            # ---------------------------------------------------------------------
            # YENİ YAKLAŞIM: Poisson dağılımını kullanarak skor olasılık matrisi oluştur
            # ---------------------------------------------------------------------

            # Skor olasılıklarını hesapla (Poisson dağılımını kullanarak)
            score_probs = self.calculate_score_probabilities(weighted_home_goals, weighted_away_goals)

            # Skoru normalleştir
            total_prob = sum(score_probs.values())
            if total_prob > 0:
                for score in score_probs:
                    score_probs[score] /= total_prob

            # En olası 5 skoru bul
            top_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)
            top_5_likely_scores = [(score, round(prob * 100, 2)) for score, prob in top_scores[:5]]

            # En olası skor
            most_likely_score = top_scores[0][0]
            most_likely_score_prob = top_scores[0][1]

            logger.info(f"En olası skor: {most_likely_score} (olasılık: {most_likely_score_prob*100:.2f}%)")

            # MS olasılıklarını doğrudan skor olasılıklarından hesapla
            home_win_probability = sum(prob for score, prob in score_probs.items() 
                                       if int(score.split('-')[0]) > int(score.split('-')[1]))

            draw_probability = sum(prob for score, prob in score_probs.items() 
                                  if int(score.split('-')[0]) == int(score.split('-')[1]))

            away_win_probability = sum(prob for score, prob in score_probs.items() 
                                      if int(score.split('-')[0]) < int(score.split('-')[1]))

            # KG VAR/YOK olasılığını doğrudan skor olasılıklarından hesapla
            kg_var_probability = sum(prob for score, prob in score_probs.items() 
                                    if int(score.split('-')[0]) > 0 and int(score.split('-')[1]) > 0)

            # 2.5 ÜST/ALT olasılığını doğrudan skor olasılıklarından hesapla
            over_25_probability = sum(prob for score, prob in score_probs.items() 
                                     if int(score.split('-')[0]) + int(score.split('-')[1]) > 2.5)

            # 3.5 ÜST/ALT olasılığını doğrudan skor olasılıklarından hesapla
            over_35_probability = sum(prob for score, prob in score_probs.items() 
                                     if int(score.split('-')[0]) + int(score.split('-')[1]) > 3.5)

            # 0-0 skoru olasılığı
            zero_zero_prob = score_probs.get("0-0", 0)

            # İlk yarı/maç sonu tahminleri için
            # İlk yarıda genellikle tüm gollerin yaklaşık %40'ı atılır
            home_ht_goals = weighted_home_goals * 0.4
            away_ht_goals = weighted_away_goals * 0.4

            ht_score_probs = self.calculate_score_probabilities(home_ht_goals, away_ht_goals)

            # İlk yarı sonucu olasılıkları
            ht_home_win_prob = sum(prob for score, prob in ht_score_probs.items() 
                                  if int(score.split('-')[0]) > int(score.split('-')[1]))

            ht_draw_prob = sum(prob for score, prob in ht_score_probs.items() 
                              if int(score.split('-')[0]) == int(score.split('-')[1]))

            ht_away_win_prob = sum(prob for score, prob in ht_score_probs.items() 
                                  if int(score.split('-')[0]) < int(score.split('-')[1]))

            # İlk yarı sonucu
            ht_result = (
                "HOME_WIN" if ht_home_win_prob > max(ht_draw_prob, ht_away_win_prob) else
                "DRAW" if ht_draw_prob > max(ht_home_win_prob, ht_away_win_prob) else
                "AWAY_WIN"
            )

            # Maç sonu sonucu
            ft_result = (
                "HOME_WIN" if home_win_probability > max(draw_probability, away_win_probability) else
                "DRAW" if draw_probability > max(home_win_probability, away_win_probability) else
                "AWAY_WIN"
            )

            # İlk yarı/maç sonu kombinasyonu
            ht_ft = f"{ht_result}/{ft_result}"

            # İlk gol ile ilgili hesaplamalar kaldırıldı
            p_no_goal = zero_zero_prob

            # En olası sonuç
            most_likely_outcome = ft_result

            # Monte Carlo simülasyonu için - retrospective comparison
            # Skor dağılımlarını simüle et
            home_scores = np.random.poisson(weighted_home_goals, size=simulations)
            away_scores = np.random.poisson(weighted_away_goals, size=simulations)

            # Ortalama gol sayısı
            avg_home_goals = np.mean(home_scores)
            avg_away_goals = np.mean(away_scores)

            # Tahmin sonuçları
            prediction = {
                "expected_goals": {
                    "home": round(weighted_home_goals, 2),  # Poisson lambda değeri
                    "away": round(weighted_away_goals, 2)   # Poisson lambda değeri
                },
                "home_win_probability": round(home_win_probability * 100, 2),
                "draw_probability": round(draw_probability * 100, 2),
                "away_win_probability": round(away_win_probability * 100, 2),
                "most_likely_outcome": most_likely_outcome,
                "betting_predictions": {
                    "both_teams_to_score": {
                        "prediction": "YES" if kg_var_probability > 0.5 else "NO",
                        "probability": round(kg_var_probability * 100, 2)
                    },
                    "over_2_5_goals": {
                        "prediction": "YES" if over_25_probability > 0.5 else "NO",
                        "probability": round(over_25_probability * 100, 2)
                    },
                    "over_3_5_goals": {
                        "prediction": "YES" if over_35_probability > max(over_25_probability, kg_var_probability) else "NO",
                        "probability": round(over_35_probability * 100, 2)
                    },
                    "exact_score": {
                        "prediction": most_likely_score,
                        "probability": round(most_likely_score_prob * 100, 2)
                    },
                    "half_time_full_time": {
                        "prediction": ht_ft,
                        "probability": round(max(ht_home_win_prob, ht_draw_prob, ht_away_win_prob) * 
                                           max(home_win_probability, draw_probability, away_win_probability) * 100, 2)
                    }
                },
                "model_details": {
                    "lambda_home": round(lambda_home, 2),
                    "lambda_away": round(lambda_away, 2),
                    "ensemble_home_goals": round(ensemble_home_goals, 2),
                    "ensemble_away_goals": round(ensemble_away_goals, 2),
                    "weighted_home_goals": round(weighted_home_goals, 2),
                    "weighted_away_goals": round(weighted_away_goals, 2),
                    "top_5_likely_scores": top_5_likely_scores,
                    "zero_zero_prob": round(zero_zero_prob * 100, 2),
                    "monte_carlo": {
                        "avg_home_goals": round(avg_home_goals, 2),
                        "avg_away_goals": round(avg_away_goals, 2)
                    }
                },
                "timestamp": datetime.now().timestamp(),
                "date_predicted": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            return prediction

        except Exception as e:
            logger.error(f"Gelişmiş tahmin sırasında hata: {e}")
            return None