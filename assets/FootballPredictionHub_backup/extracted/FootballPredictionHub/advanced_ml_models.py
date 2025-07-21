"""
Gelişmiş Makine Öğrenmesi Modelleri
Bu modül, futbol maçları tahmininde kullanılan gelişmiş algoritmaları içerir.

Modeller:
1. Gradient Boosting Machine (GBM) - XGBoost tabanlı
2. LSTM (Long Short-Term Memory) ağları
3. Bayesci Ağlar
4. Geliştirilmiş Monte Carlo simülasyonları
"""

import os
import numpy as np
import pandas as pd
import pickle
import joblib
import logging
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Logging yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedPredictionModels:
    """
    Futbol maçları için gelişmiş tahmin modellerini yöneten sınıf.
    """
    def __init__(self, model_dir="./models"):
        """
        Gelişmiş tahmin modellerini başlatır.
        
        Args:
            model_dir: Modellerin kaydedileceği dizin
        """
        self.model_dir = model_dir
        
        # Eğer model dizini yoksa oluştur
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Model dosya yolları
        self.gbm_model_path = os.path.join(model_dir, "gbm_model.pkl")
        self.lstm_model_path = os.path.join(model_dir, "lstm_model.h5")
        self.scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
        
        # Modelleri yükle veya oluştur
        self.gbm_model = self._load_or_create_gbm()
        self.lstm_model = self._load_or_create_lstm()
        self.scaler = self._load_or_create_scaler()
        
    def _load_or_create_gbm(self):
        """
        GBM modelini yükler veya varsayılan bir model oluşturur.
        """
        try:
            if os.path.exists(self.gbm_model_path):
                logger.info("Var olan GBM modeli yükleniyor...")
                return joblib.load(self.gbm_model_path)
            else:
                logger.info("GBM modeli bulunamadı, yeni model oluşturuluyor...")
                # Takım maç sonuçları tahmini için GBM modeli (sınıflandırma)
                gbm_classifier = GradientBoostingClassifier(
                    n_estimators=100, 
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )
                return gbm_classifier
        except Exception as e:
            logger.error(f"GBM modeli yüklenirken hata: {str(e)}")
            return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    
    def _load_or_create_lstm(self):
        """
        LSTM modelini yükler veya varsayılan bir model oluşturur.
        """
        try:
            if os.path.exists(self.lstm_model_path):
                logger.info("Var olan LSTM modeli yükleniyor...")
                return load_model(self.lstm_model_path)
            else:
                logger.info("LSTM modeli bulunamadı, yeni model oluşturuluyor...")
                # Takım performans trendlerini yakalamak için LSTM modeli
                model = Sequential()
                model.add(LSTM(64, input_shape=(10, 7), return_sequences=True))  # 10 maç, 7 özellik
                model.add(Dropout(0.2))
                model.add(LSTM(32))
                model.add(Dropout(0.2))
                model.add(Dense(3, activation='softmax'))  # 3 sınıf: Galibiyet, Beraberlik, Mağlubiyet
                
                model.compile(
                    loss='categorical_crossentropy',
                    optimizer=Adam(learning_rate=0.001),
                    metrics=['accuracy']
                )
                return model
        except Exception as e:
            logger.error(f"LSTM modeli yüklenirken hata: {str(e)}")
            # Basit yedek model
            model = Sequential()
            model.add(LSTM(32, input_shape=(10, 7), return_sequences=False))
            model.add(Dense(3, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model
    
    def _load_or_create_scaler(self):
        """
        Özellik ölçekleyiciyi (feature scaler) yükler veya yenisini oluşturur.
        """
        try:
            if os.path.exists(self.scaler_path):
                logger.info("Var olan özellik ölçekleyici yükleniyor...")
                return joblib.load(self.scaler_path)
            else:
                logger.info("Özellik ölçekleyici bulunamadı, yeni ölçekleyici oluşturuluyor...")
                return StandardScaler()
        except Exception as e:
            logger.error(f"Özellik ölçekleyici yüklenirken hata: {str(e)}")
            return StandardScaler()
    
    def save_models(self):
        """
        Eğitilmiş modelleri ve ölçekleyiciyi kaydet.
        """
        try:
            logger.info("Modeller kaydediliyor...")
            joblib.dump(self.gbm_model, self.gbm_model_path)
            self.lstm_model.save(self.lstm_model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info("Modeller başarıyla kaydedildi.")
        except Exception as e:
            logger.error(f"Modeller kaydedilirken hata: {str(e)}")
    
    def prepare_data_for_gbm(self, home_form, away_form):
        """
        GBM modeli için veri hazırlar.
        
        Args:
            home_form: Ev sahibi takımın form verisi (match_prediction.py'den)
            away_form: Deplasman takımının form verisi
            
        Returns:
            X: GBM için hazırlanmış özellik vektörü
        """
        features = []
        
        # Ev sahibi takım özellikleri
        home_features = [
            home_form.get('avg_goals_scored', 0),
            home_form.get('avg_goals_conceded', 0),
            home_form.get('form_points', 0) / 9,  # Normalize edilmiş form puanı
            home_form.get('home_advantage', 1.3),  # Ev avantajı faktörü
            home_form.get('bayesian', {}).get('home_lambda_scored', 1.5),
            home_form.get('bayesian', {}).get('home_lambda_conceded', 1.0),
        ]
        
        # Deplasman takımı özellikleri
        away_features = [
            away_form.get('avg_goals_scored', 0),
            away_form.get('avg_goals_conceded', 0),
            away_form.get('form_points', 0) / 9,  # Normalize edilmiş form puanı
            1.0,  # Deplasman için sabit faktör
            away_form.get('bayesian', {}).get('away_lambda_scored', 1.0),
            away_form.get('bayesian', {}).get('away_lambda_conceded', 1.5),
        ]
        
        # Karşılaştırmalı özellikler
        home_power = sum(home_features[:3]) / 3
        away_power = sum(away_features[:3]) / 3
        power_diff = home_power - away_power
        
        # Tüm özellikleri birleştir
        features = home_features + away_features + [power_diff]
        
        # Özellikleri ölçeklendir
        if len(features) > 0:  # Veri kontrolü
            features_array = np.array(features).reshape(1, -1)
            try:
                scaled_features = self.scaler.transform(features_array)
            except:
                # Eğer ölçekleyici eğitilmemişse, fit_transform kullan
                scaled_features = self.scaler.fit_transform(features_array)
            return scaled_features
        else:
            # Varsayılan özellikler
            return np.zeros((1, 13))  # 13 özellik
    
    def prepare_data_for_lstm(self, team_matches, is_home=True, lookback=10):
        """
        LSTM modeli için son maçların serisini hazırlar.
        
        Args:
            team_matches: Takımın son maçları (güncel->geçmiş sırayla)
            is_home: Takımın ev sahibi olup olmadığı
            lookback: Kaç maç geriye bakılacağı
            
        Returns:
            X: LSTM için hazırlanmış zaman serisi verisi
        """
        if not team_matches or len(team_matches) == 0:
            # Veri yoksa boş dizi döndür
            return np.zeros((1, lookback, 7))
        
        # En fazla lookback kadar maç kullan
        matches = team_matches[:min(len(team_matches), lookback)]
        
        # Eğer yeterli maç yoksa, boş verilerle doldur
        while len(matches) < lookback:
            # Varsayılan maç verileri
            default_match = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'goals_scored': 0,
                'goals_conceded': 0,
                'is_home': is_home,
                'result': 'D',  # Beraberlik
                'ht_goals_scored': 0,
                'ht_goals_conceded': 0
            }
            matches.append(default_match)
        
        # Zaman serisi özelliklerini oluştur
        series = []
        for match in matches:
            # Maç başına 7 özellik
            match_features = [
                float(match.get('goals_scored', 0)),
                float(match.get('goals_conceded', 0)),
                1.0 if match.get('is_home', False) else 0.0,
                1.0 if match.get('result', 'D') == 'W' else (0.5 if match.get('result', 'D') == 'D' else 0.0),
                float(match.get('ht_goals_scored', 0)),
                float(match.get('ht_goals_conceded', 0)),
                float(match.get('goals_scored', 0)) - float(match.get('ht_goals_scored', 0))  # İkinci yarı golleri
            ]
            series.append(match_features)
            
        # Seriyi numpy dizisine dönüştür ve yeniden şekillendir
        return np.array(series).reshape(1, lookback, 7)
    
    def predict_with_gbm(self, home_form, away_form):
        """
        GBM kullanarak maç sonucunu tahmin eder.
        
        Args:
            home_form: Ev sahibi takımın form verisi
            away_form: Deplasman takımının form verisi
            
        Returns:
            Dict: Galibiyet, beraberlik, mağlubiyet olasılıkları ve toplam gol beklentisi
        """
        X = self.prepare_data_for_gbm(home_form, away_form)
        
        # GBM modeli eğitilmiş mi kontrol et
        if not hasattr(self.gbm_model, 'n_classes_'):
            # Eğitilmemiş - basit sigmoid ile hesapla
            logger.warning("GBM modeli henüz eğitilmemiş, basit formül kullanılıyor.")
            
            # Ana özellikleri kullan
            home_strength = home_form.get('avg_goals_scored', 1.3) * (1/max(0.5, home_form.get('avg_goals_conceded', 1.0)))
            away_strength = away_form.get('avg_goals_scored', 1.0) * (1/max(0.5, away_form.get('avg_goals_conceded', 1.3)))
            home_form_pts = home_form.get('form_points', 4.5) / 9.0  # 9 üzerinden normalizasyon
            away_form_pts = away_form.get('form_points', 4.5) / 9.0
            
            # Ev avantajı faktörü
            home_advantage = 1.3
            
            # Basit formül ile tahmin
            strength_diff = (home_strength * home_advantage * home_form_pts) - (away_strength * away_form_pts)
            
            # Sigmoid fonksiyonu
            def sigmoid(x, scale=1.5):
                return 1 / (1 + np.exp(-scale * x))
            
            # Olasılıkları hesapla
            home_win_prob = sigmoid(strength_diff, scale=1.5)
            away_win_prob = sigmoid(-strength_diff, scale=1.5)
            
            # Normalizasyon
            total_prob = home_win_prob + away_win_prob
            if total_prob > 1:
                home_win_prob /= total_prob
                away_win_prob /= total_prob
            
            draw_prob = 1.0 - (home_win_prob + away_win_prob)
            
            # Toplam gol beklentisi
            home_goals_exp = home_strength * (1/max(0.5, away_form.get('avg_goals_conceded', 1.0))) * home_advantage
            away_goals_exp = away_strength * (1/max(0.5, home_form.get('avg_goals_conceded', 1.0)))
            
            # Sonuçları döndür
            return {
                'home_win_probability': float(home_win_prob * 100),
                'draw_probability': float(draw_prob * 100),
                'away_win_probability': float(away_win_prob * 100),
                'expected_goals': {
                    'home': float(home_goals_exp),
                    'away': float(away_goals_exp),
                    'total': float(home_goals_exp + away_goals_exp)
                }
            }
        
        try:
            # GBM ile sınıf olasılıklarını tahmin et
            class_probs = self.gbm_model.predict_proba(X)[0]
            
            # Sınıf sayısını kontrol et (3 olmalı)
            if len(class_probs) == 3:
                home_win_prob, draw_prob, away_win_prob = class_probs
            else:
                # Varsayılan değerler
                logger.warning("GBM modelinden beklenmeyen sınıf sayısı, varsayılan olasılıklar kullanılıyor.")
                home_win_prob, draw_prob, away_win_prob = 0.45, 0.25, 0.30
            
            # Toplam gol beklentisi (GBM regresyon modelimiz olmadığı için basit formülle hesaplıyoruz)
            home_goals_exp = home_form.get('bayesian', {}).get('home_lambda_scored', 1.5)
            away_goals_exp = away_form.get('bayesian', {}).get('away_lambda_scored', 1.0)
            
            return {
                'home_win_probability': float(home_win_prob * 100),
                'draw_probability': float(draw_prob * 100),
                'away_win_probability': float(away_win_prob * 100),
                'expected_goals': {
                    'home': float(home_goals_exp),
                    'away': float(away_goals_exp),
                    'total': float(home_goals_exp + away_goals_exp)
                }
            }
            
        except Exception as e:
            logger.error(f"GBM tahmin hatası: {str(e)}")
            # Hata durumunda varsayılan tahmin
            return {
                'home_win_probability': 40.0,
                'draw_probability': 30.0,
                'away_win_probability': 30.0,
                'expected_goals': {'home': 1.5, 'away': 1.2, 'total': 2.7}
            }
    
    def predict_with_lstm(self, home_matches, away_matches, is_home_team=True):
        """
        LSTM ağı kullanarak maç sonucunu tahmin eder.
        
        Args:
            home_matches: Ev sahibi takımın son maçları
            away_matches: Deplasman takımının son maçları
            is_home_team: Tahmini yapılan takımın ev sahibi olup olmadığı
            
        Returns:
            Dict: Galibiyet, beraberlik, mağlubiyet olasılıkları
        """
        # Ev sahibi ve deplasman takımları için veri hazırla
        X_home = self.prepare_data_for_lstm(home_matches, is_home=True)
        X_away = self.prepare_data_for_lstm(away_matches, is_home=False)
        
        # LSTM modeli eğitilmiş mi kontrol et
        if not hasattr(self.lstm_model, 'history'):
            logger.warning("LSTM modeli henüz eğitilmemiş, basit formül kullanılıyor.")
            
            # Ev sahibi takımın son maç performans verilerini çıkart
            home_goals_scored = []
            home_goals_conceded = []
            for match in home_matches[:5]:  # Son 5 maç
                home_goals_scored.append(match.get('goals_scored', 0))
                home_goals_conceded.append(match.get('goals_conceded', 0))
            
            # Deplasman takımının son maç performans verilerini çıkart
            away_goals_scored = []
            away_goals_conceded = []
            for match in away_matches[:5]:  # Son 5 maç
                away_goals_scored.append(match.get('goals_scored', 0))
                away_goals_conceded.append(match.get('goals_conceded', 0))
            
            # Ortalama değerleri hesapla
            avg_home_scored = sum(home_goals_scored) / max(1, len(home_goals_scored))
            avg_home_conceded = sum(home_goals_conceded) / max(1, len(home_goals_conceded))
            avg_away_scored = sum(away_goals_scored) / max(1, len(away_goals_scored))
            avg_away_conceded = sum(away_goals_conceded) / max(1, len(away_goals_conceded))
            
            # Basit formül ile tahmin
            home_strength = avg_home_scored / max(0.5, avg_home_conceded)
            away_strength = avg_away_scored / max(0.5, avg_away_conceded)
            
            # Ev avantajı faktörü
            home_advantage = 1.3
            
            # Güç farkını hesapla
            power_diff = (home_strength * home_advantage) - away_strength
            
            # Sigmoid ile olasılıkları hesapla
            def sigmoid(x, scale=1.0):
                return 1 / (1 + np.exp(-scale * x))
            
            home_win_prob = sigmoid(power_diff)
            away_win_prob = sigmoid(-power_diff)
            
            # Normalizasyon
            total = home_win_prob + away_win_prob
            if total > 1:
                home_win_prob /= total
                away_win_prob /= total
                
            draw_prob = 1.0 - (home_win_prob + away_win_prob)
            
            return {
                'home_win_probability': float(home_win_prob * 100),
                'draw_probability': float(draw_prob * 100),
                'away_win_probability': float(away_win_prob * 100)
            }
        
        try:
            # LSTM ile her iki takım için ayrı tahminler yap
            home_pred = self.lstm_model.predict(X_home)[0]
            away_pred = self.lstm_model.predict(X_away)[0]
            
            # Tahminleri birleştir (ev sahibi ve deplasman perspektiflerini ağırlıklandır)
            if is_home_team:
                # Ev sahibi tahminine daha fazla ağırlık ver (0.6)
                home_win_prob = home_pred[0] * 0.6 + (1 - away_pred[2]) * 0.4
                draw_prob = home_pred[1] * 0.6 + away_pred[1] * 0.4
                away_win_prob = home_pred[2] * 0.6 + (1 - away_pred[0]) * 0.4
            else:
                # Deplasman tahminine daha fazla ağırlık ver (0.6)
                home_win_prob = (1 - away_pred[2]) * 0.4 + home_pred[0] * 0.6
                draw_prob = away_pred[1] * 0.4 + home_pred[1] * 0.6
                away_win_prob = (1 - away_pred[0]) * 0.4 + home_pred[2] * 0.6
            
            # Olasılık toplamının 1 olduğundan emin ol
            total_prob = home_win_prob + draw_prob + away_win_prob
            home_win_prob /= total_prob
            draw_prob /= total_prob
            away_win_prob /= total_prob
            
            return {
                'home_win_probability': float(home_win_prob * 100),
                'draw_probability': float(draw_prob * 100),
                'away_win_probability': float(away_win_prob * 100)
            }
            
        except Exception as e:
            logger.error(f"LSTM tahmin hatası: {str(e)}")
            # Hata durumunda varsayılan tahmin
            return {
                'home_win_probability': 40.0,
                'draw_probability': 30.0,
                'away_win_probability': 30.0
            }
    
    def train_gbm_model(self, X_train, y_train):
        """
        GBM modelini eğitir.
        
        Args:
            X_train: Eğitim özellikleri 
            y_train: Hedef etiketler (0:deplasman galibiyet, 1:beraberlik, 2:ev sahibi galibiyet)
            
        Returns:
            bool: Eğitim başarılı mı
        """
        try:
            logger.info("GBM modeli eğitiliyor...")
            
            # XGBoost modeli
            self.gbm_model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                objective='multi:softproba',
                num_class=3,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
            
            # Modeli eğit
            self.gbm_model.fit(X_train, y_train)
            
            # Modeli kaydet
            joblib.dump(self.gbm_model, self.gbm_model_path)
            
            logger.info("GBM modeli başarıyla eğitildi.")
            return True
            
        except Exception as e:
            logger.error(f"GBM model eğitimi hatası: {str(e)}")
            return False
    
    def train_lstm_model(self, X_train, y_train, epochs=50, batch_size=32):
        """
        LSTM modelini eğitir.
        
        Args:
            X_train: Eğitim verileri (şekil: [örnekler, zaman_adımları, özellikler])
            y_train: Hedef etiketler (one-hot kodlanmış)
            epochs: Eğitim dönemleri sayısı
            batch_size: Parti boyutu
            
        Returns:
            bool: Eğitim başarılı mı
        """
        try:
            logger.info("LSTM modeli eğitiliyor...")
            
            # Erken durdurma
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            # Veriyi böl
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            
            # Modeli eğit
            history = self.lstm_model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Modeli kaydet
            self.lstm_model.save(self.lstm_model_path)
            
            logger.info("LSTM modeli başarıyla eğitildi.")
            return True
            
        except Exception as e:
            logger.error(f"LSTM model eğitimi hatası: {str(e)}")
            return False
    
    def combine_model_predictions(self, gbm_pred, lstm_pred, default_pred=None, weights=None, dynamic_weighting=True):
        """
        Farklı modellerin tahminlerini birleştirir.
        
        Args:
            gbm_pred: GBM modeli tahminleri
            lstm_pred: LSTM modeli tahminleri
            default_pred: Varsayılan model tahminleri (örn. match_prediction.py'den)
            weights: Model ağırlıkları (GBM, LSTM, Varsayılan) - None ise dinamik hesaplanır
            dynamic_weighting: Tahminlerin güvenilirliğine göre ağırlıkları dinamik ayarla
            
        Returns:
            Dict: Birleştirilmiş tahminler
        """
        # Değerler yoksa boş sözlüklerle başlat
        gbm_pred = gbm_pred or {}
        lstm_pred = lstm_pred or {}
        default_pred = default_pred or {}
        
        # Olasılıkları çıkart
        gbm_home = gbm_pred.get('home_win_probability', 0) / 100.0
        gbm_draw = gbm_pred.get('draw_probability', 0) / 100.0
        gbm_away = gbm_pred.get('away_win_probability', 0) / 100.0
        
        lstm_home = lstm_pred.get('home_win_probability', 0) / 100.0
        lstm_draw = lstm_pred.get('draw_probability', 0) / 100.0
        lstm_away = lstm_pred.get('away_win_probability', 0) / 100.0
        
        default_home = default_pred.get('home_win_probability', 0) / 100.0
        default_draw = default_pred.get('draw_probability', 0) / 100.0
        default_away = default_pred.get('away_win_probability', 0) / 100.0
        
        # Veri kalitesi ve model güven skorlarını hesapla
        if dynamic_weighting:
            # Veri kalitesi skorları: Model tahminlerinin güvenilirliğini değerlendir
            # Tahminlerin entropisi düşükse (bir sonuca güçlü eğilim) güvenilirlik yüksektir
            
            # GBM güvenilirlik skoru (0-1 arası)
            gbm_entropy = -((gbm_home * np.log(gbm_home + 1e-10)) + 
                           (gbm_draw * np.log(gbm_draw + 1e-10)) + 
                           (gbm_away * np.log(gbm_away + 1e-10)))
            gbm_confidence = 1.0 - min(1.0, gbm_entropy / 1.1)  # Normalize
            
            # LSTM güvenilirlik skoru (0-1 arası)
            lstm_entropy = -((lstm_home * np.log(lstm_home + 1e-10)) + 
                            (lstm_draw * np.log(lstm_draw + 1e-10)) + 
                            (lstm_away * np.log(lstm_away + 1e-10)))
            lstm_confidence = 1.0 - min(1.0, lstm_entropy / 1.1)  # Normalize
            
            # Default model güvenilirlik skoru
            default_entropy = -((default_home * np.log(default_home + 1e-10)) + 
                               (default_draw * np.log(default_draw + 1e-10)) + 
                               (default_away * np.log(default_away + 1e-10)))
            default_confidence = 1.0 - min(1.0, default_entropy / 1.1)  # Normalize
            
            # DENEYSEL: Modeller arası uyum skoru
            agreement_score = 1.0 - (
                abs(gbm_home - lstm_home) + 
                abs(gbm_draw - lstm_draw) + 
                abs(gbm_away - lstm_away)
            ) / 3.0
            
            # Uyum yüksekse GBM ve LSTM ağırlığını artır
            agreement_boost = 0.1 * agreement_score
            
            # Temel ağırlıkları belirle
            gbm_base_weight = 0.25 + (0.1 * gbm_confidence) + agreement_boost
            lstm_base_weight = 0.25 + (0.1 * lstm_confidence) + agreement_boost
            default_base_weight = 0.5 - (agreement_boost * 2)  # Default ağırlık azalır
            
            # Ağırlıkları normalize et (toplamları 1.0 olacak şekilde)
            total_weight = gbm_base_weight + lstm_base_weight + default_base_weight
            gbm_weight = gbm_base_weight / total_weight
            lstm_weight = lstm_base_weight / total_weight
            default_weight = default_base_weight / total_weight
            
            logger.debug(f"Dinamik ensemble ağırlıkları: GBM={gbm_weight:.2f}, LSTM={lstm_weight:.2f}, Default={default_weight:.2f}, Uyum={agreement_score:.2f}")
        else:
            # Sabit ağırlıkları kullan
            if weights is None:
                # Varsayılan ağırlıklar
                gbm_weight, lstm_weight, default_weight = 0.25, 0.25, 0.5
            else:
                gbm_weight, lstm_weight, default_weight = weights
        
        # Birleştirilmiş olasılıklar
        combined_home = gbm_home * gbm_weight + lstm_home * lstm_weight + default_home * default_weight
        combined_draw = gbm_draw * gbm_weight + lstm_draw * lstm_weight + default_draw * default_weight
        combined_away = gbm_away * gbm_weight + lstm_away * lstm_weight + default_away * default_weight
        
        # Toplam 1.0 olacak şekilde normalize et
        total = combined_home + combined_draw + combined_away
        if total > 0:
            combined_home /= total
            combined_draw /= total
            combined_away /= total
        
        # Beklenen golleri hesapla (varsayılan ve GBM modellerinin ortalaması)
        exp_goals_home = (
            gbm_pred.get('expected_goals', {}).get('home', 0) * gbm_weight + 
            default_pred.get('expected_goals', {}).get('home', 0) * default_weight
        ) / (gbm_weight + default_weight)
        
        exp_goals_away = (
            gbm_pred.get('expected_goals', {}).get('away', 0) * gbm_weight + 
            default_pred.get('expected_goals', {}).get('away', 0) * default_weight
        ) / (gbm_weight + default_weight)
        
        # Sonuçları döndür
        return {
            'home_win_probability': round(combined_home * 100, 1),
            'draw_probability': round(combined_draw * 100, 1),
            'away_win_probability': round(combined_away * 100, 1),
            'expected_goals': {
                'home': round(exp_goals_home, 2),
                'away': round(exp_goals_away, 2),
                'total': round(exp_goals_home + exp_goals_away, 2)
            },
            'model_weights': {
                'gbm': gbm_weight,
                'lstm': lstm_weight,
                'default': default_weight
            }
        }
    
    def improved_monte_carlo(self, home_lambda, away_lambda, simulations=10000):
        """
        Geliştirilmiş Monte Carlo simülasyonu.
        
        Args:
            home_lambda: Ev sahibi takımın gol beklentisi
            away_lambda: Deplasman takımının gol beklentisi
            simulations: Simülasyon sayısı
            
        Returns:
            Dict: Simülasyon sonuçları
        """
        # Sonuç sayaçları
        home_wins = 0
        draws = 0
        away_wins = 0
        
        # Skor dağılımı
        score_distribution = {}
        
        # Yarı skor dağılımları
        half_time_distribution = {}
        half_time_full_time_distribution = {}
        
        # Simülasyonları çalıştır
        for _ in range(simulations):
            # İlk yarı
            # İlk yarı gol beklentileri (toplam beklentinin yaklaşık %40'ı)
            home_lambda_ht = home_lambda * 0.4
            away_lambda_ht = away_lambda * 0.4
            
            # İlk yarı simülasyonu
            home_goals_ht = np.random.poisson(home_lambda_ht)
            away_goals_ht = np.random.poisson(away_lambda_ht)
            
            # İlk yarı sonucu
            ht_score = f"{home_goals_ht}-{away_goals_ht}"
            half_time_distribution[ht_score] = half_time_distribution.get(ht_score, 0) + 1
            
            # İkinci yarı
            # İkinci yarı gol beklentileri (toplam beklentinin yaklaşık %60'ı)
            home_lambda_2h = home_lambda * 0.6
            away_lambda_2h = away_lambda * 0.6
            
            # İkinci yarı simülasyonu
            home_goals_2h = np.random.poisson(home_lambda_2h)
            away_goals_2h = np.random.poisson(away_lambda_2h)
            
            # Toplam goller
            home_goals_ft = home_goals_ht + home_goals_2h
            away_goals_ft = away_goals_ht + away_goals_2h
            
            # Maç sonucu
            if home_goals_ft > away_goals_ft:
                home_wins += 1
            elif home_goals_ft == away_goals_ft:
                draws += 1
            else:
                away_wins += 1
                
            # Skor dağılımı güncelle
            ft_score = f"{home_goals_ft}-{away_goals_ft}"
            score_distribution[ft_score] = score_distribution.get(ft_score, 0) + 1
            
            # İY/MS dağılımı güncelle
            if home_goals_ht > away_goals_ht:
                ht_result = "1"
            elif home_goals_ht == away_goals_ht:
                ht_result = "X"
            else:
                ht_result = "2"
                
            if home_goals_ft > away_goals_ft:
                ft_result = "1"
            elif home_goals_ft == away_goals_ft:
                ft_result = "X"
            else:
                ft_result = "2"
                
            htft_result = f"{ht_result}/{ft_result}"
            half_time_full_time_distribution[htft_result] = half_time_full_time_distribution.get(htft_result, 0) + 1
        
        # Sonuçları hesapla
        home_win_prob = home_wins / simulations * 100
        draw_prob = draws / simulations * 100
        away_win_prob = away_wins / simulations * 100
        
        # En olası skorları bul
        most_likely_scores = sorted(score_distribution.items(), key=lambda x: x[1], reverse=True)[:5]
        most_likely_scores = [(score, count / simulations * 100) for score, count in most_likely_scores]
        
        # En olası İY/MS kombinasyonlarını bul
        most_likely_htft = sorted(half_time_full_time_distribution.items(), key=lambda x: x[1], reverse=True)
        most_likely_htft = [(result, count / simulations * 100) for result, count in most_likely_htft]
        
        # Sonuçları döndür
        return {
            'match_outcome': {
                'home_win': home_win_prob,
                'draw': draw_prob,
                'away_win': away_win_prob
            },
            'score_probabilities': {score: prob for score, prob in most_likely_scores},
            'half_time_full_time': {result: prob for result, prob in most_likely_htft},
            'simulations': simulations,
            'expected_goals': {
                'home': home_lambda,
                'away': away_lambda
            }
        }


class BayesianNetwork:
    """
    Futbol tahminleri için basitleştirilmiş Bayesci Ağ sınıfı.
    Bu sınıf, takımların gol atma ve yeme oranları arasındaki ilişkiyi modellemek için Bayesci yaklaşım kullanır.
    """
    def __init__(self):
        """Bayesci Ağ'ı başlat"""
        # Öncül (prior) parametreler
        self.prior_alpha_home = 1.0  # Ev sahibi gol atma öncül alpha
        self.prior_beta_home = 1.0   # Ev sahibi gol atma öncül beta
        self.prior_alpha_away = 1.0  # Deplasman gol atma öncül alpha
        self.prior_beta_away = 1.0   # Deplasman gol atma öncül beta
        
    def update_beliefs(self, team_data, is_home=True):
        """
        Takım verileri ile Bayesci inançları güncelle
        
        Args:
            team_data: Takımın maç verileri
            is_home: Takımın ev sahibi olup olmadığı
        
        Returns:
            dict: Güncellenmiş Bayesci parametreler
        """
        # Takımın gol atma ve yeme istatistiklerini topla
        goals_scored = []
        goals_conceded = []
        
        for match in team_data:
            # Ev sahibi için ev maçlarını, deplasman için deplasman maçlarını kullan
            if (is_home and match.get('is_home', False)) or (not is_home and not match.get('is_home', False)):
                goals_scored.append(match.get('goals_scored', 0))
                goals_conceded.append(match.get('goals_conceded', 0))
        
        # Veri yoksa varsayılan değerler döndür
        if not goals_scored or not goals_conceded:
            return {
                'alpha_scored': self.prior_alpha_home if is_home else self.prior_alpha_away,
                'beta_scored': self.prior_beta_home if is_home else self.prior_beta_away,
                'alpha_conceded': self.prior_alpha_home if is_home else self.prior_alpha_away,
                'beta_conceded': self.prior_beta_home if is_home else self.prior_beta_away,
                'expected_scored': 1.5 if is_home else 1.0,
                'expected_conceded': 1.0 if is_home else 1.5,
                'variance_scored': 0.5,
                'variance_conceded': 0.5
            }
        
        # Gözlemlenen ortalama ve varyans
        mean_scored = sum(goals_scored) / len(goals_scored)
        mean_conceded = sum(goals_conceded) / len(goals_conceded)
        
        var_scored = np.var(goals_scored) if len(goals_scored) > 1 else 0.5
        var_conceded = np.var(goals_conceded) if len(goals_conceded) > 1 else 0.5
        
        # Gamma dağılımının parametrelerini hesapla
        # Ortalama = alpha/beta, Varyans = alpha/beta^2
        # Bu yüzden alpha = ortalama^2 / varyans, beta = ortalama / varyans
        
        # Aşırı değerleri sınırla
        var_scored = max(0.1, var_scored)
        var_conceded = max(0.1, var_conceded)
        
        # Ev sahibi ve deplasman ayrı ayrı
        if is_home:
            # Ev sahibi için öncül inançları kullan
            prior_alpha = self.prior_alpha_home
            prior_beta = self.prior_beta_home
            
            # Öncül ve gözlemleri birleştir
            posterior_alpha_scored = (prior_alpha + sum(goals_scored)) / (1 + len(goals_scored))
            posterior_beta_scored = (prior_beta + len(goals_scored)) / (1 + sum(goals_scored) + 0.001)
            
            posterior_alpha_conceded = (prior_alpha + sum(goals_conceded)) / (1 + len(goals_conceded))
            posterior_beta_conceded = (prior_beta + len(goals_conceded)) / (1 + sum(goals_conceded) + 0.001)
        else:
            # Deplasman için öncül inançları kullan
            prior_alpha = self.prior_alpha_away
            prior_beta = self.prior_beta_away
            
            # Öncül ve gözlemleri birleştir
            posterior_alpha_scored = (prior_alpha + sum(goals_scored)) / (1 + len(goals_scored))
            posterior_beta_scored = (prior_beta + len(goals_scored)) / (1 + sum(goals_scored) + 0.001)
            
            posterior_alpha_conceded = (prior_alpha + sum(goals_conceded)) / (1 + len(goals_conceded))
            posterior_beta_conceded = (prior_beta + len(goals_conceded)) / (1 + sum(goals_conceded) + 0.001)
        
        # Beklenen değerler ve varyanslar
        expected_scored = posterior_alpha_scored / posterior_beta_scored
        expected_conceded = posterior_alpha_conceded / posterior_beta_conceded
        
        variance_scored = posterior_alpha_scored / (posterior_beta_scored ** 2)
        variance_conceded = posterior_alpha_conceded / (posterior_beta_conceded ** 2)
        
        return {
            'alpha_scored': posterior_alpha_scored,
            'beta_scored': posterior_beta_scored,
            'alpha_conceded': posterior_alpha_conceded,
            'beta_conceded': posterior_beta_conceded,
            'expected_scored': expected_scored,
            'expected_conceded': expected_conceded,
            'variance_scored': variance_scored,
            'variance_conceded': variance_conceded
        }
    
    def predict_match(self, home_team_data, away_team_data):
        """
        Bayesci ağ ile maç sonucunu tahmin et
        
        Args:
            home_team_data: Ev sahibi takım maç verileri
            away_team_data: Deplasman takımı maç verileri
        
        Returns:
            dict: Tahmin sonuçları
        """
        # Ev sahibi ve deplasman inançlarını güncelle
        home_beliefs = self.update_beliefs(home_team_data, is_home=True)
        away_beliefs = self.update_beliefs(away_team_data, is_home=False)
        
        # Ev sahibi ve deplasman gol beklentileri
        # Ev sahibi gol beklentisi için: ev gol atma * deplasman gol yeme
        home_expected_goals = (home_beliefs['expected_scored'] + away_beliefs['expected_conceded']) / 2
        
        # Deplasman gol beklentisi için: deplasman gol atma * ev gol yeme
        away_expected_goals = (away_beliefs['expected_scored'] + home_beliefs['expected_conceded']) / 2
        
        # Ev avantajı ekle
        home_advantage = 1.3  # %30 ev avantajı
        home_expected_goals *= home_advantage
        
        # Monte Carlo simülasyonu (10,000 maç)
        simulations = 10000
        home_wins = 0
        draws = 0
        away_wins = 0
        
        for _ in range(simulations):
            # Her simülasyonda farklı lambda değerleri kullan (belirsizliği yansıtmak için)
            home_lambda = np.random.gamma(home_beliefs['alpha_scored'], 1/home_beliefs['beta_scored']) * home_advantage
            away_lambda = np.random.gamma(away_beliefs['alpha_scored'], 1/away_beliefs['beta_scored'])
            
            # Goller
            home_goals = np.random.poisson(home_lambda)
            away_goals = np.random.poisson(away_lambda)
            
            # Sonuç
            if home_goals > away_goals:
                home_wins += 1
            elif home_goals == away_goals:
                draws += 1
            else:
                away_wins += 1
        
        # Olasılıkları hesapla
        home_win_prob = home_wins / simulations * 100
        draw_prob = draws / simulations * 100
        away_win_prob = away_wins / simulations * 100
        
        return {
            'home_win_probability': home_win_prob,
            'draw_probability': draw_prob,
            'away_win_probability': away_win_prob,
            'expected_goals': {
                'home': home_expected_goals,
                'away': away_expected_goals,
                'total': home_expected_goals + away_expected_goals
            },
            'bayesian_parameters': {
                'home': home_beliefs,
                'away': away_beliefs
            }
        }