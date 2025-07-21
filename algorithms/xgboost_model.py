"""
XGBoost Makine Öğrenmesi Modeli
Gradient boosting ile gelişmiş tahminler
"""
import numpy as np
import logging
import json
import os

logger = logging.getLogger(__name__)

# XGBoost opsiyonel - yüklü değilse basit model kullan
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost bulunamadı, basit ML modeli kullanılacak")
    XGBOOST_AVAILABLE = False

class XGBoostModel:
    """
    XGBoost tabanlı tahmin modeli
    """
    
    def __init__(self):
        self.model_1x2 = None  # 1X2 tahminleri için
        self.model_goals = None  # Toplam gol tahmini için
        self.model_btts = None  # KG var/yok için
        self.models_loaded = False
        self.load_models()
        
    def load_models(self):
        """
        Eğitilmiş modelleri yükle
        """
        if not XGBOOST_AVAILABLE:
            return
            
        try:
            if os.path.exists('models/xgb_1x2.json'):
                self.model_1x2 = xgb.Booster()
                self.model_1x2.load_model('models/xgb_1x2.json')
                self.models_loaded = True
                logger.info("XGBoost modelleri yüklendi")
            else:
                # Hızlı model eğitimi
                self._train_simple_model()
        except Exception as e:
            logger.warning(f"Model yükleme hatası: {e}")
            
    def _train_simple_model(self):
        """
        Basit XGBoost modeli eğit
        """
        try:
            # Önbellekten eğitim verisi al
            if os.path.exists('predictions_cache.json'):
                with open('predictions_cache.json', 'r') as f:
                    cache_data = json.load(f)
                    
                if len(cache_data) >= 10:
                    X_train, y_train = self._prepare_training_data(cache_data)
                    
                    if len(X_train) >= 10:
                        # Basit XGBoost modeli
                        model = xgb.XGBClassifier(
                            n_estimators=50,
                            max_depth=4,
                            learning_rate=0.1,
                            random_state=42,
                            objective='multi:softprob',
                            num_class=3
                        )
                        
                        model.fit(X_train, y_train)
                        
                        # Kaydet
                        model.save_model('models/xgb_1x2.json')
                        self.model_1x2 = model.get_booster()
                        self.models_loaded = True
                        logger.info("XGBoost modeli eğitildi ve kaydedildi")
                        
        except Exception as e:
            logger.error(f"Model eğitim hatası: {e}")
            
    def _prepare_training_data(self, cache_data):
        """
        Önbellekten eğitim verisi hazırla
        """
        X_train = []
        y_train = []
        
        for match_key, match_data in list(cache_data.items())[:100]:  # İlk 100 maç
            if not match_data.get('predictions'):
                continue
                
            predictions = match_data['predictions']
            
            # Özellik vektörü
            features = [
                predictions.get('expected_goals', {}).get('home', 1.5),
                predictions.get('expected_goals', {}).get('away', 1.5),
                predictions.get('home_win_probability', 33) / 100,
                predictions.get('draw_probability', 33) / 100,
                predictions.get('away_win_probability', 34) / 100,
                predictions.get('over_under', {}).get('over_2_5', 50) / 100,
                predictions.get('both_teams_to_score', {}).get('yes', 50) / 100,
                0,  # Elo farkı placeholder
                2.0,  # Form placeholder
                2.0,  # Form placeholder
                1.3,  # Performans placeholder
                1.3,  # Performans placeholder
                1.0,  # Performans placeholder
                1.3   # Performans placeholder
            ]
            
            X_train.append(features)
            
            # Etiket - XGBoost için 0,1,2 sınıfları
            home_prob = predictions.get('home_win_probability', 33)
            draw_prob = predictions.get('draw_probability', 33)
            away_prob = predictions.get('away_win_probability', 34)
            
            if home_prob > draw_prob and home_prob > away_prob:
                y_train.append(0)  # HOME_WIN
            elif draw_prob > home_prob and draw_prob > away_prob:
                y_train.append(1)  # DRAW
            else:
                y_train.append(2)  # AWAY_WIN
        
        # Sınıf dağılımını kontrol et
        y_train_array = np.array(y_train)
        unique_classes = np.unique(y_train_array)
        logger.info(f"Sınıf dağılımı: {unique_classes}")
        
        # Eksik sınıfları 0,1,2 olarak tamamla
        if len(unique_classes) < 3:
            logger.warning("Eksik sınıflar tespit edildi, dengeleme yapılıyor")
            # En az bir örnek her sınıftan ekle
            for missing_class in [0, 1, 2]:
                if missing_class not in unique_classes:
                    X_train.append(X_train[0])  # İlk özellik vektörünü kopyala
                    y_train.append(missing_class)
                    logger.info(f"Sınıf {missing_class} için örnek eklendi")
                
        return np.array(X_train), np.array(y_train)
            
    def prepare_features(self, home_data, away_data, xg_data):
        """
        ML için özellik vektörü hazırla
        
        Args:
            home_data: Ev sahibi takım verileri
            away_data: Deplasman takım verileri
            xg_data: xG/xGA değerleri
            
        Returns:
            numpy.ndarray: Özellik vektörü
        """
        features = []
        
        # xG/xGA özellikleri
        features.extend([
            xg_data.get('home_xg', 1.3),
            xg_data.get('home_xga', 1.3),
            xg_data.get('away_xg', 1.3),
            xg_data.get('away_xga', 1.3),
            xg_data.get('lambda_home', 1.5),
            xg_data.get('lambda_away', 1.0)
        ])
        
        # Elo farkı
        features.append(xg_data.get('elo_diff', 0))
        
        # Form (son 5 maç)
        home_form = self._calculate_form(home_data.get('recent_matches', [])[:5])
        away_form = self._calculate_form(away_data.get('recent_matches', [])[:5])
        features.extend([home_form, away_form])
        
        # Ev/Deplasman performansı
        home_performance = home_data.get('home_performance', {})
        away_performance = away_data.get('away_performance', {})
        
        features.extend([
            home_performance.get('avg_goals', 1.3),
            home_performance.get('avg_conceded', 1.3),
            away_performance.get('avg_goals', 1.0),
            away_performance.get('avg_conceded', 1.3)
        ])
        
        return np.array(features).reshape(1, -1)
        
    def _calculate_form(self, matches):
        """
        Son maçlardan form puanı hesapla
        """
        if not matches:
            return 2.0
            
        points = 0
        for match in matches:
            goals_for = match.get('goals_scored', 0)
            goals_against = match.get('goals_conceded', 0)
            
            if goals_for > goals_against:
                points += 3
            elif goals_for == goals_against:
                points += 1
                
        return points / len(matches)
        
    def predict(self, features):
        """
        XGBoost tahmini yap
        
        Returns:
            dict: Tahmin sonuçları
        """
        if not XGBOOST_AVAILABLE or not self.models_loaded:
            # Fallback: Basit tahmin
            return self._simple_prediction(features)
            
        try:
            # XGBoost tahminleri
            dmatrix = xgb.DMatrix(features)
            
            # 1X2 tahmini
            probs_1x2 = self.model_1x2.predict(dmatrix)[0]
            
            # Lambda değerlerini özelliklerden al
            lambda_home = features[0][4] if features.shape[1] > 4 else 1.5
            lambda_away = features[0][5] if features.shape[1] > 5 else 1.0
            
            # Over/Under ve BTTS tahminleri
            total_goals = lambda_home + lambda_away
            over_2_5 = min(95, max(5, (total_goals - 2.5) * 20 + 50))
            
            # BTTS
            btts_base = min(lambda_home, lambda_away) / max(lambda_home, lambda_away)
            btts_yes = min(90, max(10, btts_base * 70 + 20))
            
            # Dinamik güven hesaplama
            max_prob = max(float(probs_1x2[0]), float(probs_1x2[1]), float(probs_1x2[2]))
            
            # Tahmin keskinliğine göre güven (0.4-0.9 arası)
            if max_prob > 0.6:  # Çok net favori
                base_confidence = 0.75 + (max_prob - 0.6) * 0.5  # Max 0.95
            elif max_prob > 0.45:  # Orta düzey favori
                base_confidence = 0.65 + (max_prob - 0.45) * 0.67  # 0.65-0.75
            else:  # Dengeli maç
                base_confidence = 0.5 + (max_prob - 0.33) * 1.25  # 0.5-0.65
            
            # Model eğitim verisi sayısına göre ayarla
            if hasattr(self, 'training_data_count'):
                if self.training_data_count > 100:
                    base_confidence *= 1.1
                elif self.training_data_count < 30:
                    base_confidence *= 0.9
            
            # Güven değerini sınırla
            dynamic_confidence = max(0.5, min(0.9, base_confidence))
            
            predictions = {
                'home_win': float(probs_1x2[0]) * 100,
                'draw': float(probs_1x2[1]) * 100,
                'away_win': float(probs_1x2[2]) * 100,
                'over_2_5': over_2_5,
                'under_2_5': 100 - over_2_5,
                'btts_yes': btts_yes,
                'btts_no': 100 - btts_yes,
                'expected_goals': {
                    'home': lambda_home,
                    'away': lambda_away
                },
                'confidence': round(dynamic_confidence, 2),
                'model': 'xgboost'
            }
            
            logger.info("XGBoost tahmini tamamlandı")
            return predictions
            
        except Exception as e:
            logger.error(f"XGBoost tahmin hatası: {e}")
            return self._simple_prediction(features)
            
    def _simple_prediction(self, features):
        """
        XGBoost yoksa basit ML benzeri tahmin
        """
        # Özelliklerden basit tahmin
        lambda_home = features[0][4] if features.shape[1] > 4 else 1.5
        lambda_away = features[0][5] if features.shape[1] > 5 else 1.0
        elo_diff = features[0][6] if features.shape[1] > 6 else 0
        
        # Basit lojistik tahmin
        home_advantage = 0.1
        elo_factor = elo_diff / 400  # Elo'yu normalize et
        
        # Olasılıklar
        home_strength = lambda_home / (lambda_home + lambda_away) + home_advantage + elo_factor * 0.1
        away_strength = lambda_away / (lambda_home + lambda_away) - elo_factor * 0.1
        
        # Normalize
        total = home_strength + away_strength
        home_prob = (home_strength / total) * 0.75  # %75 1X2, %25 beraberlik
        away_prob = (away_strength / total) * 0.75
        draw_prob = 0.25
        
        # Düzeltme
        if abs(elo_diff) < 100:  # Yakın maç
            draw_prob = 0.30
            home_prob = (home_strength / total) * 0.70
            away_prob = (away_strength / total) * 0.70
            
        # Over/Under ve BTTS tahminleri
        total_goals = lambda_home + lambda_away
        over_2_5 = min(95, max(5, (total_goals - 2.5) * 20 + 50))  # Basit tahmin
        
        # BTTS - lambda değerlerine göre
        btts_base = min(lambda_home, lambda_away) / max(lambda_home, lambda_away)
        btts_yes = min(90, max(10, btts_base * 70 + 20))
        
        return {
            'home_win': home_prob * 100,
            'draw': draw_prob * 100,
            'away_win': away_prob * 100,
            'over_2_5': over_2_5,
            'under_2_5': 100 - over_2_5,
            'btts_yes': btts_yes,
            'btts_no': 100 - btts_yes,
            'expected_goals': {
                'home': lambda_home,
                'away': lambda_away
            },
            'confidence': 0.65,  # Basit model güveni
            'model': 'simple_ml'
        }
        
    def train_model(self, training_data):
        """
        Model eğitimi (ileride kullanılacak)
        """
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost yüklü değil, eğitim yapılamıyor")
            return
            
        # Eğitim kodu buraya eklenecek
        pass