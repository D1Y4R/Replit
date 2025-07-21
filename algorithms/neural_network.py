"""
Neural Network Tahmin Modeli
TensorFlow/Keras ile derin öğrenme tabanlı tahminler
"""
import numpy as np
import logging
import os
import pickle
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# TensorFlow opsiyonel - yüklü değilse basit model kullan
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler
    TF_AVAILABLE = True
    logger.info("TensorFlow yüklü - Neural Network aktif")
except ImportError:
    logger.warning("TensorFlow bulunamadı, basit NN modeli kullanılacak")
    TF_AVAILABLE = False

class NeuralNetworkModel:
    """
    Neural Network tabanlı tahmin modeli
    """
    
    def __init__(self):
        self.model_1x2 = None
        self.model_goals = None
        self.scaler = None
        self.models_loaded = False
        self.load_models()
        
    def load_models(self):
        """
        Eğitilmiş modelleri yükle veya yeni eğit
        """
        if not TF_AVAILABLE:
            return
            
        try:
            # Ana model
            model_path = 'models/neural_network.h5'
            if os.path.exists(model_path):
                self.model_1x2 = load_model(model_path)
                logger.info("Neural Network modeli yüklendi")
                
            # Scaler
            scaler_path = 'models/scaler.pkl'
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                    
            if self.model_1x2 and self.scaler:
                self.models_loaded = True
                logger.info("Neural Network tamamen yüklendi")
            else:
                logger.warning("Neural Network model bulunamadı, yeni model eğitiliyor")
                self._train_from_cache()
                
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")
            self._train_from_cache()
            
    def prepare_features(self, home_data, away_data, xg_data, match_context):
        """
        Neural Network için özellik hazırla
        
        Args:
            home_data: Ev sahibi takım verileri
            away_data: Deplasman takım verileri
            xg_data: xG/xGA değerleri
            match_context: Maç bağlamı
            
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
        
        # Elo ve güç farkı
        features.extend([
            match_context.get('elo_diff', 0),
            match_context.get('home_advantage', 0.3),
        ])
        
        # Takım form ve performans
        home_form = self._calculate_form_score(home_data.get('recent_matches', [])[:5])
        away_form = self._calculate_form_score(away_data.get('recent_matches', [])[:5])
        features.extend([home_form, away_form])
        
        # Ev/Deplasman özellikler
        home_performance = home_data.get('home_performance', {})
        away_performance = away_data.get('away_performance', {})
        
        features.extend([
            home_performance.get('avg_goals', 1.3),
            home_performance.get('avg_conceded', 1.3),
            away_performance.get('avg_goals', 1.0),
            away_performance.get('avg_conceded', 1.3)
        ])
        
        # Gol istatistikleri
        features.extend([
            home_data.get('avg_goals_scored', 1.3),
            home_data.get('avg_goals_conceded', 1.3),
            away_data.get('avg_goals_scored', 1.0),
            away_data.get('avg_goals_conceded', 1.3)
        ])
        
        # Maç önem faktörleri
        features.extend([
            match_context.get('league_importance', 0.7),
            match_context.get('season_stage', 0.5),
            match_context.get('rivalry_factor', 0.0)
        ])
        
        # Momentum ve trend
        features.extend([
            self._calculate_momentum(home_data.get('recent_matches', [])),
            self._calculate_momentum(away_data.get('recent_matches', [])),
            self._calculate_goal_trend(home_data.get('recent_matches', [])),
            self._calculate_goal_trend(away_data.get('recent_matches', []))
        ])
        
        return np.array(features).reshape(1, -1)
        
    def _calculate_form_score(self, matches):
        """
        Son maçlardan form puanı hesapla
        """
        if not matches:
            return 2.0
            
        points = 0
        weight = 1.0
        
        for match in matches:
            goals_for = match.get('goals_scored', 0)
            goals_against = match.get('goals_conceded', 0)
            
            if goals_for > goals_against:
                points += 3 * weight
            elif goals_for == goals_against:
                points += 1 * weight
                
            weight *= 0.9  # Eski maçlar daha az önemli
            
        return points / len(matches) if matches else 2.0
        
    def _calculate_momentum(self, matches):
        """
        Takım momentumunu hesapla
        """
        if not matches:
            return 0.0
            
        momentum = 0.0
        for i, match in enumerate(matches):  # Tüm mevcut maçlar
            goals_for = match.get('goals_scored', 0)
            goals_against = match.get('goals_conceded', 0)
            
            match_score = goals_for - goals_against
            weight = 1.0 - (i * 0.2)  # Yeni maçlar daha önemli
            momentum += match_score * weight
            
        return momentum / 3
        
    def _calculate_goal_trend(self, matches):
        """
        Gol atma trendini hesapla
        """
        if not matches:
            return 1.0
            
        recent_goals = [match.get('goals_scored', 0) for match in matches]
        
        if len(recent_goals) < 2:
            return 1.0
            
        # Basit trend hesabı
        first_half = sum(recent_goals[:2]) / 2
        second_half = sum(recent_goals[2:]) / max(1, len(recent_goals[2:]))
        
        return second_half / max(0.1, first_half)
        
    def predict(self, features):
        """
        Neural Network tahmini yap
        
        Returns:
            dict: Tahmin sonuçları
        """
        if not TF_AVAILABLE or not self.models_loaded:
            return self._simple_neural_prediction(features)
            
        try:
            # Özellik normalizasyonu
            if self.scaler:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
                
            # 1X2 tahmini
            predictions_1x2 = self.model_1x2.predict(features_scaled, verbose=0)[0]
            
            # Lambda değerlerini özelliklerden al
            lambda_home = features[0][4] if features.shape[1] > 4 else 1.5
            lambda_away = features[0][5] if features.shape[1] > 5 else 1.0
            
            # Ek tahminler
            total_goals = lambda_home + lambda_away
            over_2_5 = min(95, max(5, self._sigmoid(total_goals - 2.5) * 100))
            
            # BTTS tahmini
            btts_factor = min(lambda_home, lambda_away) / max(lambda_home, lambda_away)
            btts_yes = min(90, max(10, btts_factor * 80 + 10))
            
            # Exact score prediction
            exact_scores = self._predict_exact_scores(lambda_home, lambda_away)
            
            # Dinamik güven hesaplama
            max_prob = max(float(predictions_1x2[0]), float(predictions_1x2[1]), float(predictions_1x2[2]))
            
            # Tahmin keskinliğine göre güven (0.4-0.9 arası)
            if max_prob > 0.6:  # Çok net favori
                base_confidence = 0.8 + (max_prob - 0.6) * 0.5  # Max 1.0
            elif max_prob > 0.45:  # Orta düzey favori
                base_confidence = 0.7 + (max_prob - 0.45) * 0.67  # 0.7-0.8
            else:  # Dengeli maç
                base_confidence = 0.55 + (max_prob - 0.33) * 1.25  # 0.55-0.7
            
            # Neural Network modeli genelde daha güvenilir olduğu için biraz daha yüksek tutuyoruz
            base_confidence *= 1.05
            
            # Model eğitim verisi sayısına göre ayarla
            if hasattr(self, 'training_data_count'):
                if self.training_data_count > 100:
                    base_confidence *= 1.1
                elif self.training_data_count < 50:
                    base_confidence *= 0.95
            
            # Güven değerini sınırla
            dynamic_confidence = max(0.5, min(0.9, base_confidence))
            
            predictions = {
                'home_win': float(predictions_1x2[0]) * 100,
                'draw': float(predictions_1x2[1]) * 100,
                'away_win': float(predictions_1x2[2]) * 100,
                'over_2_5': over_2_5,
                'under_2_5': 100 - over_2_5,
                'btts_yes': btts_yes,
                'btts_no': 100 - btts_yes,
                'expected_goals': {
                    'home': lambda_home,
                    'away': lambda_away
                },
                'exact_scores': exact_scores,
                'confidence': round(dynamic_confidence, 2),
                'model': 'neural_network'
            }
            
            logger.info("Neural Network tahmini tamamlandı")
            return predictions
            
        except Exception as e:
            logger.error(f"Neural Network tahmin hatası: {e}")
            return self._simple_neural_prediction(features)
            
    def _sigmoid(self, x):
        """
        Sigmoid aktivasyon fonksiyonu
        """
        return 1 / (1 + np.exp(-x))
        
    def _predict_exact_scores(self, lambda_home, lambda_away):
        """
        Poisson ile exact score tahminleri
        """
        from scipy.stats import poisson
        
        scores = []
        for home_goals in range(6):
            for away_goals in range(6):
                prob = (poisson.pmf(home_goals, lambda_home) * 
                       poisson.pmf(away_goals, lambda_away))
                scores.append({
                    'score': f"{home_goals}-{away_goals}",
                    'probability': prob * 100
                })
                
        return sorted(scores, key=lambda x: x['probability'], reverse=True)[:10]
        
    def _simple_neural_prediction(self, features):
        """
        TensorFlow yoksa basit sinir ağı benzeri tahmin
        """
        try:
            # Özelliklerden basit weighted sum
            lambda_home = features[0][4] if features.shape[1] > 4 else 1.5
            lambda_away = features[0][5] if features.shape[1] > 5 else 1.0
            elo_diff = features[0][6] if features.shape[1] > 6 else 0
            
            # Basit "nöral ağ" hesabı
            # Gizli katman simülasyonu
            hidden_1 = self._sigmoid(lambda_home * 0.8 + lambda_away * 0.2 + elo_diff * 0.01)
            hidden_2 = self._sigmoid(lambda_home * 0.3 + lambda_away * 0.7 - elo_diff * 0.01)
            hidden_3 = self._sigmoid((lambda_home + lambda_away) * 0.5 + elo_diff * 0.005)
            
            # Çıkış katmanı
            home_raw = hidden_1 * 0.6 + hidden_2 * 0.2 + hidden_3 * 0.2
            away_raw = hidden_1 * 0.2 + hidden_2 * 0.6 + hidden_3 * 0.2
            draw_raw = hidden_1 * 0.2 + hidden_2 * 0.2 + hidden_3 * 0.6
            
            # Softmax normalizasyonu
            exp_home = np.exp(home_raw)
            exp_away = np.exp(away_raw)
            exp_draw = np.exp(draw_raw)
            total = exp_home + exp_away + exp_draw
            
            home_prob = exp_home / total
            away_prob = exp_away / total
            draw_prob = exp_draw / total
            
            # Ek tahminler
            total_goals = lambda_home + lambda_away
            over_2_5 = min(95, max(5, (total_goals - 2.5) * 25 + 50))
            btts_yes = min(90, max(10, min(lambda_home, lambda_away) * 50 + 20))
            
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
                'confidence': 0.70,  # Basit model güveni
                'model': 'simple_neural'
            }
            
        except Exception as e:
            logger.error(f"Basit neural prediction hatası: {e}")
            return {
                'home_win': 35.0,
                'draw': 30.0,
                'away_win': 35.0,
                'over_2_5': 55.0,
                'under_2_5': 45.0,
                'btts_yes': 60.0,
                'btts_no': 40.0,
                'expected_goals': {'home': 1.5, 'away': 1.5},
                'confidence': 0.50,
                'model': 'fallback_neural'
            }
            
    def create_and_train_model(self, training_data):
        """
        Yeni model oluştur ve eğit
        """
        if not TF_AVAILABLE:
            logger.error("TensorFlow yüklü değil")
            return False
            
        try:
            # Model mimarisi
            model = Sequential([
                Dense(128, activation='relu', input_shape=(training_data['X'].shape[1],)),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(32, activation='relu'),
                Dropout(0.2),
                
                Dense(3, activation='softmax')
            ])
            
            # Compile
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Eğitim
            model.fit(
                training_data['X'], training_data['y'],
                validation_split=0.2,
                epochs=50,
                batch_size=32,
                verbose=1
            )
            
            # Kaydet
            model.save('models/neural_network.h5')
            logger.info("Neural Network modeli eğitildi ve kaydedildi")
            
            return True
            
        except Exception as e:
            logger.error(f"Model eğitim hatası: {e}")
            return False
    
    def _train_from_cache(self):
        """
        Önbellek verilerinden Neural Network eğit
        """
        if not TF_AVAILABLE:
            return
            
        try:
            import json
            from sklearn.preprocessing import StandardScaler
            
            # Önbellekten eğitim verisi al
            if os.path.exists('predictions_cache.json'):
                with open('predictions_cache.json', 'r') as f:
                    cache_data = json.load(f)
                    
                if len(cache_data) >= 20:  # En az 20 maç
                    X_train, y_train = self._prepare_nn_training_data(cache_data)
                    
                    if len(X_train) >= 20:
                        # Scaler hazırla
                        self.scaler = StandardScaler()
                        X_train_scaled = self.scaler.fit_transform(X_train)
                        
                        # Model oluştur
                        self.model_1x2 = Sequential([
                            Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                            BatchNormalization(),
                            Dropout(0.3),
                            Dense(64, activation='relu'),
                            BatchNormalization(),
                            Dropout(0.3),
                            Dense(32, activation='relu'),
                            Dropout(0.2),
                            Dense(3, activation='softmax')
                        ])
                        
                        # Compile
                        self.model_1x2.compile(
                            optimizer=Adam(learning_rate=0.001),
                            loss='categorical_crossentropy',
                            metrics=['accuracy']
                        )
                        
                        # Eğitim
                        self.model_1x2.fit(
                            X_train_scaled, y_train,
                            validation_split=0.2,
                            epochs=30,
                            batch_size=16,
                            verbose=0
                        )
                        
                        # Kaydet
                        self.model_1x2.save('models/neural_network.h5')
                        with open('models/scaler.pkl', 'wb') as f:
                            pickle.dump(self.scaler, f)
                            
                        self.models_loaded = True
                        logger.info("Neural Network yeni model eğitildi ve kaydedildi")
                    else:
                        logger.warning("Neural Network için yetersiz eğitim verisi")
                        
        except Exception as e:
            logger.error(f"Neural Network eğitim hatası: {e}")
            
    def _prepare_nn_training_data(self, cache_data):
        """
        Neural Network eğitimi için veri hazırla
        """
        import numpy as np
        
        X_train = []
        y_train = []
        
        for match_key, match_data in list(cache_data.items())[:100]:
            if not match_data.get('predictions'):
                continue
                
            predictions = match_data['predictions']
            
            # Özellik vektörü (19 özellik)
            features = [
                predictions.get('expected_goals', {}).get('home', 1.5),
                predictions.get('expected_goals', {}).get('away', 1.5),
                predictions.get('home_win_probability', 33) / 100,
                predictions.get('draw_probability', 33) / 100,
                predictions.get('away_win_probability', 34) / 100,
                predictions.get('over_under', {}).get('over_2_5', 50) / 100,
                predictions.get('both_teams_to_score', {}).get('yes', 50) / 100,
                0,  # Elo farkı
                0.3,  # Home advantage
                2.0,  # Home form
                2.0,  # Away form
                1.3,  # Home avg goals
                1.3,  # Home avg conceded
                1.0,  # Away avg goals
                1.3,  # Away avg conceded
                1.3,  # Home performance
                1.3,  # Away performance
                0.7,  # League importance
                0.5   # Season stage
            ]
            
            X_train.append(features)
            
            # One-hot encoded etiket
            home_prob = predictions.get('home_win_probability', 33)
            draw_prob = predictions.get('draw_probability', 33)
            away_prob = predictions.get('away_win_probability', 34)
            
            if home_prob > draw_prob and home_prob > away_prob:
                y_train.append([1, 0, 0])  # HOME_WIN
            elif draw_prob > home_prob and draw_prob > away_prob:
                y_train.append([0, 1, 0])  # DRAW
            else:
                y_train.append([0, 0, 1])  # AWAY_WIN
                
        return np.array(X_train), np.array(y_train)
    
    def retrain_model(self):
        """
        Modeli yeniden eğit (periyodik güncelleme için)
        """
        logger.info("Neural Network modeli yeniden eğitiliyor...")
        self._train_from_cache()