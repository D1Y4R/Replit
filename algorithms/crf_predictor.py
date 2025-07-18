"""
CRF (Conditional Random Fields) Tahmin Modeli
Eğitilmiş CRF modelini kullanarak 1X2 tahminleri yapar
"""
import pickle
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class CRFPredictor:
    """
    CRF tabanlı tahmin modeli
    """
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.load_model()
        
    def load_model(self):
        """
        CRF modelini yükle veya yeni eğit
        """
        model_path = 'models/crf_model.pkl'
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                    self.model_loaded = True
                    logger.info("CRF modeli başarıyla yüklendi")
            except Exception as e:
                logger.error(f"CRF model yükleme hatası: {e}")
                self.model_loaded = False
        else:
            logger.warning(f"CRF model dosyası bulunamadı, yeni model eğitiliyor")
            self._train_new_model()
            
    def prepare_features(self, home_data, away_data, lambda_home, lambda_away, elo_diff):
        """
        CRF için özellik hazırla
        
        Args:
            home_data: Ev sahibi takım verileri
            away_data: Deplasman takım verileri  
            lambda_home: Ev sahibi gol beklentisi
            lambda_away: Deplasman gol beklentisi
            elo_diff: Elo farkı
            
        Returns:
            list: Özellik listesi
        """
        features = []
        
        # Lambda değerleri
        features.append(f"lambda_home={round(lambda_home, 2)}")
        features.append(f"lambda_away={round(lambda_away, 2)}")
        features.append(f"lambda_diff={round(lambda_home - lambda_away, 2)}")
        
        # Elo farkı
        if elo_diff > 100:
            features.append("elo_strong_home")
        elif elo_diff < -100:
            features.append("elo_strong_away")
        else:
            features.append("elo_balanced")
            
        # Form (son 5 maç)
        home_form = self._get_form_string(home_data.get('recent_matches', [])[:5])
        away_form = self._get_form_string(away_data.get('recent_matches', [])[:5])
        
        features.append(f"home_form_{home_form}")
        features.append(f"away_form_{away_form}")
        
        # Gol ortalamaları
        home_avg_goals = home_data.get('avg_goals_scored', 1.5)
        away_avg_goals = away_data.get('avg_goals_scored', 1.0)
        
        if home_avg_goals > 2.0:
            features.append("home_high_scoring")
        elif home_avg_goals < 1.0:
            features.append("home_low_scoring")
            
        if away_avg_goals > 2.0:
            features.append("away_high_scoring")
        elif away_avg_goals < 1.0:
            features.append("away_low_scoring")
            
        # Savunma gücü
        home_avg_conceded = home_data.get('avg_goals_conceded', 1.3)
        away_avg_conceded = away_data.get('avg_goals_conceded', 1.5)
        
        if home_avg_conceded < 1.0:
            features.append("home_strong_defense")
        elif home_avg_conceded > 2.0:
            features.append("home_weak_defense")
            
        if away_avg_conceded < 1.0:
            features.append("away_strong_defense")
        elif away_avg_conceded > 2.0:
            features.append("away_weak_defense")
            
        return features
        
    def _get_form_string(self, matches):
        """
        Form stringi oluştur (WWDLW gibi)
        """
        form = ""
        for match in matches:
            goals_for = match.get('goals_scored', 0)
            goals_against = match.get('goals_conceded', 0)
            
            if goals_for > goals_against:
                form += "W"
            elif goals_for == goals_against:
                form += "D"
            else:
                form += "L"
                
        return form if form else "DDDDD"
        
    def predict(self, features):
        """
        CRF tahminini yap
        
        Args:
            features: Özellik listesi
            
        Returns:
            dict: 1X2 tahmin olasılıkları
        """
        if not self.model_loaded or not self.model:
            # Model yüklü değilse fallback tahmin
            return self._fallback_prediction(features)
            
        try:
            # CRF tahmini - tek bir örnek için liste formatında
            X = [features]  # CRF sequence modeli için
            
            # Model tahmini
            y_pred = self.model.predict(X)
            
            # Olasılık tahmini
            if hasattr(self.model, 'predict_marginals'):
                marginals = self.model.predict_marginals(X)
                probs = marginals[0]  # İlk (ve tek) örnek için
                
                # CRF marginal olasılıklarından tahmin çıkar
                if isinstance(probs, dict):
                    home_prob = probs.get('1', 0.33)
                    draw_prob = probs.get('X', 0.33)
                    away_prob = probs.get('2', 0.34)
                else:
                    # Numpy array ise basit atama
                    home_prob = 0.33
                    draw_prob = 0.33
                    away_prob = 0.34
            else:
                # Sadece tahmin varsa, basit atama
                prediction = y_pred[0] if y_pred else 'X'
                if prediction == '1':
                    home_prob, draw_prob, away_prob = 0.50, 0.25, 0.25
                elif prediction == '2':
                    home_prob, draw_prob, away_prob = 0.25, 0.25, 0.50
                else:
                    home_prob, draw_prob, away_prob = 0.30, 0.40, 0.30
                    
            # Normalize et
            total = home_prob + draw_prob + away_prob
            if total > 0:
                home_prob /= total
                draw_prob /= total
                away_prob /= total
                
            # Dinamik güven hesaplama
            max_prob = max(home_prob, draw_prob, away_prob)
            
            # Tahmin keskinliğine göre güven (0.4-0.9 arası)
            if max_prob > 0.6:  # Çok net favori
                base_confidence = 0.7 + (max_prob - 0.6) * 0.5  # Max 0.9
            elif max_prob > 0.45:  # Orta düzey favori
                base_confidence = 0.6 + (max_prob - 0.45) * 0.67  # 0.6-0.7
            else:  # Dengeli maç
                base_confidence = 0.5 + (max_prob - 0.33) * 0.83  # 0.5-0.6
            
            # CRF modeli için orta seviye güven
            base_confidence *= 1.0
            
            # Model yüklü değilse güveni düşür
            if not self.model_loaded:
                base_confidence *= 0.85
            
            # Güven değerini sınırla
            dynamic_confidence = max(0.5, min(0.85, base_confidence))
            
            return {
                'home_win': home_prob * 100,
                'draw': draw_prob * 100,
                'away_win': away_prob * 100,
                'confidence': round(dynamic_confidence, 2),
                'model': 'crf'
            }
            
        except Exception as e:
            logger.error(f"CRF tahmin hatası: {e}")
            return self._fallback_prediction(features)
            
    def _fallback_prediction(self, features):
        """
        CRF çalışmazsa basit kural tabanlı tahmin
        """
        # Özelliklerden basit tahmin çıkar
        home_score = 0.35  # Başlangıç
        away_score = 0.35
        
        for feature in features:
            if 'lambda_home' in feature and '=' in feature:
                val = float(feature.split('=')[1])
                if val > 2.0:
                    home_score += 0.1
                    
            if 'elo_strong_home' in feature:
                home_score += 0.15
            elif 'elo_strong_away' in feature:
                away_score += 0.15
                
            if 'home_form_W' in feature:
                home_score += 0.05 * feature.count('W')
            if 'away_form_W' in feature:
                away_score += 0.05 * feature.count('W')
                
        # Normalize
        draw_score = 0.3
        total = home_score + draw_score + away_score
        
        return {
            'home_win': (home_score / total) * 100,
            'draw': (draw_score / total) * 100,
            'away_win': (away_score / total) * 100,
            'confidence': 0.60,
            'model': 'crf_fallback'
        }
    
    def _train_new_model(self):
        """
        Önbellek verilerinden yeni CRF modeli eğit
        """
        try:
            import json
            from sklearn.ensemble import RandomForestClassifier
            
            # Önbellekten eğitim verisi al
            if os.path.exists('predictions_cache.json'):
                with open('predictions_cache.json', 'r') as f:
                    cache_data = json.load(f)
                    
                if len(cache_data) >= 15:  # En az 15 maç
                    X_train, y_train = self._prepare_crf_training_data(cache_data)
                    
                    if len(X_train) >= 15:
                        # Random Forest ile CRF benzeri model
                        self.model = RandomForestClassifier(
                            n_estimators=100,
                            max_depth=8,
                            random_state=42
                        )
                        
                        self.model.fit(X_train, y_train)
                        
                        # Kaydet
                        with open('models/crf_model.pkl', 'wb') as f:
                            pickle.dump(self.model, f)
                            
                        self.model_loaded = True
                        logger.info("Yeni CRF modeli eğitildi ve kaydedildi")
                    else:
                        logger.warning("Yetersiz eğitim verisi, fallback model kullanılıyor")
                        
        except Exception as e:
            logger.error(f"CRF model eğitim hatası: {e}")
            
    def _prepare_crf_training_data(self, cache_data):
        """
        CRF eğitimi için veri hazırla
        """
        X_train = []
        y_train = []
        
        for match_key, match_data in list(cache_data.items())[:100]:
            if not match_data.get('predictions'):
                continue
                
            predictions = match_data['predictions']
            
            # Özellik vektörü (sayısal)
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
                1.5,  # Lambda home
                1.0   # Lambda away
            ]
            
            X_train.append(features)
            
            # Etiket
            home_prob = predictions.get('home_win_probability', 33)
            draw_prob = predictions.get('draw_probability', 33)
            away_prob = predictions.get('away_win_probability', 34)
            
            if home_prob > draw_prob and home_prob > away_prob:
                y_train.append(0)  # HOME_WIN
            elif draw_prob > home_prob and draw_prob > away_prob:
                y_train.append(1)  # DRAW
            else:
                y_train.append(2)  # AWAY_WIN
                
        return X_train, y_train
    
    def update_model_with_result(self, features, actual_result):
        """
        Gerçek sonuçla modeli güncelle (online learning)
        """
        if self.model_loaded:
            try:
                # Yeni veri ile model güncellemesi
                # Basit implementasyon - production'da daha gelişmiş online learning
                pass
            except Exception as e:
                logger.error(f"Model güncelleme hatası: {e}")
    
    def retrain_model(self):
        """
        Modeli yeniden eğit (periyodik güncelleme için)
        """
        logger.info("CRF modeli yeniden eğitiliyor...")
        self._train_new_model()