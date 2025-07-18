"""
ML Modellerini Gerçek Verilerle Eğitim Scripti
XGBoost, Neural Network ve CRF modellerini hazırlayacağım verilerle eğitir
"""
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# API ve veri modülleri
from api_football import FootballDataAPI
from api_config import APIConfig
from algorithms.elo_system import EloSystem
from algorithms.xg_calculator import XGCalculator

# ML kütüphaneleri
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import sklearn_crfsuite
    from sklearn_crfsuite import CRF
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    ML modellerini eğiten ana sınıf
    """
    
    def __init__(self):
        self.api = FootballDataAPI()
        self.elo_system = EloSystem()
        self.xg_calculator = XGCalculator()
        
        # Veri konteynerları
        self.training_data = []
        self.features = []
        self.labels_1x2 = []
        self.labels_goals = []
        
        # Model dosya yolları
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
    def collect_training_data(self, days_back=180):
        """
        Geçmiş maç verilerini toplayıp eğitim verisi hazırla
        """
        logger.info(f"Son {days_back} günün maç verisi toplanıyor...")
        
        # Tarih aralığı
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Önbellekteki tahminleri kullan
        if os.path.exists('predictions_cache.json'):
            with open('predictions_cache.json', 'r') as f:
                cache_data = json.load(f)
                
            logger.info(f"Önbellekte {len(cache_data)} tahmin bulundu")
            
            # Önbellekteki verilerden özellik çıkar
            for match_key, match_data in cache_data.items():
                if self._is_valid_match_data(match_data):
                    features = self._extract_features_from_cache(match_data)
                    if features:
                        self.features.append(features)
                        
                        # Tahmin sonuçlarını etiket olarak kullan
                        predictions = match_data.get('predictions', {})
                        
                        # 1X2 etiketi
                        home_prob = predictions.get('home_win_probability', 33)
                        draw_prob = predictions.get('draw_probability', 33)
                        away_prob = predictions.get('away_win_probability', 34)
                        
                        if home_prob > draw_prob and home_prob > away_prob:
                            label_1x2 = 0  # HOME_WIN
                        elif away_prob > draw_prob and away_prob > home_prob:
                            label_1x2 = 2  # AWAY_WIN
                        else:
                            label_1x2 = 1  # DRAW
                            
                        self.labels_1x2.append(label_1x2)
                        
                        # Toplam gol etiketi
                        expected_goals = predictions.get('expected_goals', {})
                        total_goals = expected_goals.get('home', 1.5) + expected_goals.get('away', 1.5)
                        self.labels_goals.append(total_goals)
        
        # API'den ek veriler al
        self._fetch_additional_match_data(start_date, end_date)
        
        logger.info(f"Toplam {len(self.features)} maç verisi toplandı")
        return len(self.features)
        
    def _is_valid_match_data(self, match_data):
        """
        Maç verisinin eğitim için uygun olup olmadığını kontrol et
        """
        if not match_data.get('predictions'):
            return False
            
        predictions = match_data['predictions']
        
        # Temel tahmin verilerinin varlığını kontrol et
        required_fields = ['home_win_probability', 'draw_probability', 'away_win_probability']
        return all(field in predictions for field in required_fields)
        
    def _extract_features_from_cache(self, match_data):
        """
        Önbellek verisinden özellik vektörü çıkar
        """
        try:
            predictions = match_data['predictions']
            
            # Temel özellikler
            features = [
                predictions.get('expected_goals', {}).get('home', 1.5),
                predictions.get('expected_goals', {}).get('away', 1.5),
                predictions.get('home_win_probability', 33) / 100,
                predictions.get('draw_probability', 33) / 100,
                predictions.get('away_win_probability', 34) / 100,
                predictions.get('over_under', {}).get('over_2_5', 50) / 100,
                predictions.get('both_teams_to_score', {}).get('yes', 50) / 100,
            ]
            
            # Ek özellikler
            if 'betting_predictions' in predictions:
                betting = predictions['betting_predictions']
                features.extend([
                    betting.get('over_2_5_goals', {}).get('probability', 50) / 100,
                    betting.get('over_3_5_goals', {}).get('probability', 30) / 100,
                ])
            else:
                features.extend([0.5, 0.3])  # Varsayılan değerler
                
            # Skor tahmininden özellikler
            if 'most_likely_score' in predictions:
                score = predictions['most_likely_score']
                if '-' in score:
                    home_goals, away_goals = map(int, score.split('-'))
                    features.extend([home_goals, away_goals, abs(home_goals - away_goals)])
                else:
                    features.extend([1, 1, 0])
            else:
                features.extend([1, 1, 0])
                
            return features
            
        except Exception as e:
            logger.error(f"Özellik çıkarma hatası: {e}")
            return None
            
    def _fetch_additional_match_data(self, start_date, end_date):
        """
        API'den ek maç verisi çek
        """
        try:
            # Son 30 günlük maçları al
            recent_date = datetime.now() - timedelta(days=30)
            fixtures = self.api.get_fixtures(date=recent_date.strftime('%Y-%m-%d'))
            
            if fixtures and 'matches' in fixtures:
                logger.info(f"API'den {len(fixtures['matches'])} maç alındı")
                
                for match in fixtures['matches'][:100]:  # İlk 100 maç
                    if match.get('status') == 'FINISHED':
                        features = self._extract_features_from_api_match(match)
                        if features:
                            self.features.append(features)
                            
                            # Sonuçtan etiket oluştur
                            home_score = match.get('score', {}).get('fullTime', {}).get('home', 0)
                            away_score = match.get('score', {}).get('fullTime', {}).get('away', 0)
                            
                            if home_score > away_score:
                                label_1x2 = 0  # HOME_WIN
                            elif away_score > home_score:
                                label_1x2 = 2  # AWAY_WIN
                            else:
                                label_1x2 = 1  # DRAW
                                
                            self.labels_1x2.append(label_1x2)
                            self.labels_goals.append(home_score + away_score)
                            
        except Exception as e:
            logger.error(f"API veri alma hatası: {e}")
            
    def _extract_features_from_api_match(self, match):
        """
        API maç verisinden özellik çıkar
        """
        try:
            # Temel özellikler
            home_team = match.get('homeTeam', {})
            away_team = match.get('awayTeam', {})
            
            # Skor
            score = match.get('score', {}).get('fullTime', {})
            home_score = score.get('home', 0)
            away_score = score.get('away', 0)
            
            # Basit özellikler
            features = [
                home_score,  # Gerçek ev sahibi gol
                away_score,  # Gerçek deplasman gol
                0.5,  # Varsayılan ev sahibi prob
                0.3,  # Varsayılan beraberlik prob
                0.2,  # Varsayılan deplasman prob
                0.5 if (home_score + away_score) > 2.5 else 0.4,  # Over 2.5
                0.6 if home_score > 0 and away_score > 0 else 0.3,  # BTTS
                0.5,  # Over 2.5 (tekrar)
                0.3,  # Over 3.5
                home_score,  # Tekrar ev sahibi gol
                away_score,  # Tekrar deplasman gol
                abs(home_score - away_score)  # Gol farkı
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"API özellik çıkarma hatası: {e}")
            return None
            
    def train_xgboost_model(self):
        """
        XGBoost modelini eğit
        """
        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost kütüphanesi bulunamadı")
            return False
            
        if len(self.features) < 10:
            logger.error("Yetersiz eğitim verisi")
            return False
            
        logger.info("XGBoost modeli eğitiliyor...")
        
        try:
            # Veriyi hazırla
            X = np.array(self.features)
            y = np.array(self.labels_1x2)
            
            # Eğitim/test ayrımı
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # XGBoost parametreleri
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'random_state': 42,
                'eval_metric': 'mlogloss'
            }
            
            # Model eğitimi
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            
            # Test
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"XGBoost test accuracy: {accuracy:.3f}")
            
            # Modeli kaydet
            model_path = os.path.join(self.model_dir, 'xgb_1x2.json')
            model.save_model(model_path)
            logger.info(f"XGBoost modeli kaydedildi: {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"XGBoost eğitim hatası: {e}")
            return False
            
    def train_neural_network(self):
        """
        Neural Network modelini eğit
        """
        if not TF_AVAILABLE:
            logger.error("TensorFlow kütüphanesi bulunamadı")
            return False
            
        if len(self.features) < 10:
            logger.error("Yetersiz eğitim verisi")
            return False
            
        logger.info("Neural Network modeli eğitiliyor...")
        
        try:
            # Veriyi hazırla
            X = np.array(self.features)
            y_1x2 = np.array(self.labels_1x2)
            
            # One-hot encoding
            y_1x2_onehot = tf.keras.utils.to_categorical(y_1x2, num_classes=3)
            
            # Eğitim/test ayrımı
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_1x2_onehot, test_size=0.2, random_state=42
            )
            
            # Normalizasyon
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Model mimarisi
            model = Sequential([
                Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(32, activation='relu'),
                Dropout(0.2),
                
                Dense(3, activation='softmax')  # 1X2 çıkışı
            ])
            
            # Compile
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
            
            # Eğitim
            history = model.fit(
                X_train_scaled, y_train,
                validation_data=(X_test_scaled, y_test),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            # Test
            test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
            logger.info(f"Neural Network test accuracy: {test_accuracy:.3f}")
            
            # Modeli kaydet
            model_path = os.path.join(self.model_dir, 'neural_network.h5')
            model.save(model_path)
            
            # Scaler'ı kaydet
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
                
            logger.info(f"Neural Network modeli kaydedildi: {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Neural Network eğitim hatası: {e}")
            return False
            
    def train_crf_model(self):
        """
        CRF modelini eğit
        """
        if not CRF_AVAILABLE:
            logger.error("sklearn-crfsuite kütüphanesi bulunamadı")
            return False
            
        if len(self.features) < 10:
            logger.error("Yetersiz eğitim verisi")
            return False
            
        logger.info("CRF modeli eğitiliyor...")
        
        try:
            # CRF için sequence formatında veri hazırla
            X_sequences = []
            y_sequences = []
            
            for i, (features, label) in enumerate(zip(self.features, self.labels_1x2)):
                # Her maç için özellik sözlüğü oluştur
                feature_dict = {
                    f'feature_{j}': str(features[j]) for j in range(len(features))
                }
                
                # Ek özellikler
                feature_dict.update({
                    'lambda_home': str(features[0]),
                    'lambda_away': str(features[1]),
                    'home_prob': str(features[2]),
                    'total_goals': str(features[0] + features[1]),
                    'goal_diff': str(abs(features[0] - features[1])),
                })
                
                X_sequences.append([feature_dict])
                
                # Etiket
                if label == 0:
                    y_sequences.append(['1'])  # HOME_WIN
                elif label == 2:
                    y_sequences.append(['2'])  # AWAY_WIN
                else:
                    y_sequences.append(['X'])  # DRAW
                    
            # Eğitim/test ayrımı
            X_train, X_test, y_train, y_test = train_test_split(
                X_sequences, y_sequences, test_size=0.2, random_state=42
            )
            
            # CRF model
            crf = CRF(
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=100,
                all_possible_transitions=True
            )
            
            # Eğitim
            crf.fit(X_train, y_train)
            
            # Test
            y_pred = crf.predict(X_test)
            
            # Accuracy hesaplama
            correct = 0
            total = 0
            for true_seq, pred_seq in zip(y_test, y_pred):
                for true_label, pred_label in zip(true_seq, pred_seq):
                    if true_label == pred_label:
                        correct += 1
                    total += 1
                    
            accuracy = correct / total if total > 0 else 0
            logger.info(f"CRF test accuracy: {accuracy:.3f}")
            
            # Modeli kaydet
            model_path = os.path.join(self.model_dir, 'crf_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(crf, f)
                
            logger.info(f"CRF modeli kaydedildi: {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"CRF eğitim hatası: {e}")
            return False
            
    def train_all_models(self):
        """
        Tüm modelleri eğit
        """
        logger.info("Tüm ML modelleri eğitiliyor...")
        
        # Veri toplama
        data_count = self.collect_training_data()
        
        if data_count < 10:
            logger.error("Yetersiz eğitim verisi. En az 10 maç verisi gerekli.")
            return False
            
        results = {}
        
        # XGBoost
        logger.info("=== XGBoost Eğitimi ===")
        results['xgboost'] = self.train_xgboost_model()
        
        # Neural Network
        logger.info("=== Neural Network Eğitimi ===")
        results['neural_network'] = self.train_neural_network()
        
        # CRF
        logger.info("=== CRF Eğitimi ===")
        results['crf'] = self.train_crf_model()
        
        # Özet
        logger.info("=== Eğitim Sonuçları ===")
        for model_name, success in results.items():
            status = "✅ Başarılı" if success else "❌ Başarısız"
            logger.info(f"{model_name}: {status}")
            
        return results

if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.train_all_models()
    
    print("\n=== Model Eğitimi Tamamlandı ===")
    for model_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {model_name}")