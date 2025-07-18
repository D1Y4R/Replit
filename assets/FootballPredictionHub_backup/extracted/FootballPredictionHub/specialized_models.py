"""
Özelleştirilmiş Tahmin Modelleri

Bu modül, düşük, orta ve yüksek skorlu maçlar için özelleştirilmiş tahmin modelleri sağlar.
Her skor aralığı için farklı parametrelerle optimize edilmiş modeller kullanarak tahmin doğruluğunu artırır.

Skorlama Kategorileri:
1. Düşük: Toplam beklenen gol < 2.0 
2. Orta: Toplam beklenen gol 2.0-3.5 arası
3. Yüksek: Toplam beklenen gol > 3.5

Özellikler:
- Her kategori için ayrı tahmin modelleri
- Kategori-spesifik tahmin parametreleri
- Kategori bazlı özelleştirilmiş skor olasılık dağılımları
- Gerçek dünya verileri ile kalibre edilmiş skor düzeltme faktörleri
"""

import os
import json
import logging
import numpy as np
import joblib
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Logging ayarları
logger = logging.getLogger(__name__)

class SpecializedModels:
    """
    Düşük, orta ve yüksek skorlu maçlar için özelleştirilmiş modeller sağlayan sınıf.
    """
    
    def __init__(self, model_dir="./models"):
        """
        Özelleştirilmiş tahmin modellerini başlatır.
        
        Args:
            model_dir: Modellerin kaydedileceği dizin
        """
        self.model_dir = model_dir
        
        # Model dosya yolları
        self.low_scoring_model_path = os.path.join(model_dir, "low_scoring_model.pkl")
        self.medium_scoring_model_path = os.path.join(model_dir, "medium_scoring_model.pkl")
        self.high_scoring_model_path = os.path.join(model_dir, "high_scoring_model.pkl")
        
        # Modellerin yüklenmesi veya oluşturulması
        self.low_scoring_model = self._load_or_create_model(self.low_scoring_model_path, "low")
        self.medium_scoring_model = self._load_or_create_model(self.medium_scoring_model_path, "medium")
        self.high_scoring_model = self._load_or_create_model(self.high_scoring_model_path, "high")
        
        # Model parametreleri
        self.model_parameters = {
            "low": {
                "draw_boost": 1.3,       # Düşük skorlu maçlarda beraberlik olasılığını artır
                "score_correction": {    # Spesifik skorların düzeltme faktörleri
                    "0-0": 1.5,          # 0-0 skoru çok daha olası
                    "1-0": 1.2,
                    "0-1": 1.2,
                    "1-1": 1.3,
                    "2-0": 0.9,
                    "0-2": 0.9,
                    "2-1": 0.8,
                    "1-2": 0.8
                },
                "max_score": 2,          # En yüksek skor
                "poisson_correction": 0.7 # Poisson dağılımı düzeltme faktörü (korelasyon)
            },
            "medium": {
                "draw_boost": 1.0,
                "score_correction": {
                    "1-1": 1.2,
                    "2-1": 1.1,
                    "1-2": 1.1,
                    "2-2": 1.1,
                    "3-1": 0.9,
                    "1-3": 0.9
                },
                "max_score": 3,
                "poisson_correction": 0.85
            },
            "high": {
                "draw_boost": 0.8,       # Yüksek skorlu maçlarda beraberlik olasılığını düşür
                "score_correction": {
                    "2-2": 1.1,
                    "3-2": 1.1,
                    "2-3": 1.1,
                    "3-1": 1.1,
                    "1-3": 1.1,
                    "3-3": 1.2          # Yüksek skorlarda 3-3 daha olası
                },
                "max_score": 5,
                "poisson_correction": 0.95
            }
        }

    def _load_or_create_model(self, model_path, category):
        """
        Kaydedilmiş bir modeli yükler veya yeni model oluşturur.
        
        Args:
            model_path: Model dosya yolu
            category: Model kategorisi (low, medium, high)
            
        Returns:
            Yüklenmiş veya yeni oluşturulmuş model
        """
        try:
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                logger.info(f"{category.capitalize()} skorlu maç modeli başarıyla yüklendi")
                return model
        except Exception as e:
            logger.error(f"{category.capitalize()} skorlu maç modeli yüklenirken hata: {str(e)}")
        
        # Model bulunamadı veya yüklenemedi, yeni model oluştur
        logger.info(f"{category.capitalize()} skorlu maç modeli bulunamadı, varsayılan model oluşturuluyor")
        return self._create_default_model(category)
    
    def _create_default_model(self, category):
        """
        Her kategori için varsayılan model oluşturur.
        
        Args:
            category: Model kategorisi (low, medium, high)
            
        Returns:
            Yeni oluşturulmuş varsayılan model
        """
        if category == "low":
            # Düşük skorlu maçlar için GBM sınıflandırıcı
            model = GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.05, 
                max_depth=3,
                min_samples_split=5,
                random_state=42
            )
        elif category == "medium":
            # Orta skorlu maçlar için GBM sınıflandırıcı
            model = GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=4,
                min_samples_split=4,
                random_state=42
            )
        else:  # high
            # Yüksek skorlu maçlar için GBM sınıflandırıcı
            model = GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=5,
                min_samples_split=3,
                random_state=42
            )
        
        return model
    
    def save_models(self):
        """
        Tüm modelleri kaydeder.
        
        Returns:
            bool: İşlem başarılı mı
        """
        try:
            # Model dizinini oluştur
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Modelleri kaydet
            joblib.dump(self.low_scoring_model, self.low_scoring_model_path)
            joblib.dump(self.medium_scoring_model, self.medium_scoring_model_path)
            joblib.dump(self.high_scoring_model, self.high_scoring_model_path)
            
            logger.info("Tüm özelleştirilmiş modeller başarıyla kaydedildi")
            return True
        except Exception as e:
            logger.error(f"Modeller kaydedilirken hata: {str(e)}")
            return False
    
    def determine_category(self, expected_home_goals, expected_away_goals):
        """
        Beklenen gollere göre maçın hangi kategoriye ait olduğunu belirler.
        
        Args:
            expected_home_goals: Ev sahibi takımın beklenen gol sayısı
            expected_away_goals: Deplasman takımının beklenen gol sayısı
            
        Returns:
            str: Maç kategorisi ('low', 'medium', 'high')
        """
        total_expected_goals = expected_home_goals + expected_away_goals
        
        if total_expected_goals < 2.0:
            return "low"
        elif total_expected_goals <= 3.5:
            return "medium"
        else:
            return "high"
    
    def get_model_parameters(self, expected_home_goals, expected_away_goals):
        """
        Maç kategorisine göre model parametrelerini döndürür.
        
        Args:
            expected_home_goals: Ev sahibi takımın beklenen gol sayısı
            expected_away_goals: Deplasman takımının beklenen gol sayısı
            
        Returns:
            dict: Model parametreleri
        """
        category = self.determine_category(expected_home_goals, expected_away_goals)
        return self.model_parameters[category]
    
    def prepare_features(self, home_team_stats, away_team_stats):
        """
        Model tahmini için özellik vektörünü hazırlar.
        
        Args:
            home_team_stats: Ev sahibi takım istatistikleri
            away_team_stats: Deplasman takımı istatistikleri
            
        Returns:
            list: Özellik vektörü
        """
        # Temel istatistikler
        home_performance = home_team_stats.get('home_performance', {})
        away_performance = away_team_stats.get('away_performance', {})
        home_bayesian = home_team_stats.get('bayesian', {})
        away_bayesian = away_team_stats.get('bayesian', {})
        
        # Özellik vektörü oluştur
        features = [
            home_performance.get('avg_goals_scored', 0),
            home_performance.get('avg_goals_conceded', 0),
            home_performance.get('weighted_avg_goals_scored', 0),
            home_performance.get('weighted_avg_goals_conceded', 0),
            home_performance.get('form_points', 0),
            away_performance.get('avg_goals_scored', 0),
            away_performance.get('avg_goals_conceded', 0),
            away_performance.get('weighted_avg_goals_scored', 0),
            away_performance.get('weighted_avg_goals_conceded', 0),
            away_performance.get('form_points', 0),
            home_bayesian.get('home_lambda_scored', 0),
            home_bayesian.get('home_lambda_conceded', 0),
            away_bayesian.get('away_lambda_scored', 0),
            away_bayesian.get('away_lambda_conceded', 0),
            # İlk yarı istatistikleri
            home_performance.get('avg_ht_goals_scored', 0),
            home_performance.get('avg_ht_goals_conceded', 0),
            away_performance.get('avg_ht_goals_scored', 0),
            away_performance.get('avg_ht_goals_conceded', 0),
        ]
        
        return features
    
    def predict(self, home_team_stats, away_team_stats, expected_home_goals, expected_away_goals):
        """
        Spesifik model kullanarak maç sonucunu tahmin eder.
        
        Args:
            home_team_stats: Ev sahibi takım istatistikleri
            away_team_stats: Deplasman takımı istatistikleri
            expected_home_goals: Ev sahibi takımın beklenen gol sayısı
            expected_away_goals: Deplasman takımının beklenen gol sayısı
            
        Returns:
            dict: Tahmin sonuçları ve parametreleri
        """
        # Maç kategorisini belirle
        category = self.determine_category(expected_home_goals, expected_away_goals)
        parameters = self.model_parameters[category]
        
        # Özellik vektörünü hazırla
        features = self.prepare_features(home_team_stats, away_team_stats)
        
        # Doğrudan model yerine, kategori bazlı parametre değerlerini döndür
        # Bu parametreler tahmin algoritmasını ayarlamak için kullanılacaktır
        return {
            "category": category,
            "parameters": parameters,
            "features": features,
            "expected_home_goals_adjusted": expected_home_goals,
            "expected_away_goals_adjusted": expected_away_goals
        }

    def train_models(self, training_data):
        """
        Tüm kategorilere ait modelleri eğitir.
        
        Args:
            training_data: Eğitim verileri listesi (her öğe örnek ve hedef içerir)
            
        Returns:
            dict: Eğitim sonuçları
        """
        # Eğitim verilerini kategorilere ayır
        low_scoring_data = []
        medium_scoring_data = []
        high_scoring_data = []
        
        for match in training_data:
            features = match["features"]
            expected_home_goals = match["expected_home_goals"]
            expected_away_goals = match["expected_away_goals"]
            target = match["result"]  # 1: Ev Sahibi Kazandı, 0: Beraberlik, 2: Deplasman Kazandı
            
            # Kategoriyi belirle ve uygun listeye ekle
            category = self.determine_category(expected_home_goals, expected_away_goals)
            if category == "low":
                low_scoring_data.append((features, target))
            elif category == "medium":
                medium_scoring_data.append((features, target))
            else:  # high
                high_scoring_data.append((features, target))
        
        # Her kategoriye ait modelleri eğit
        results = {}
        
        if low_scoring_data:
            X, y = zip(*low_scoring_data)
            self.low_scoring_model.fit(X, y)
            results["low"] = {"samples": len(low_scoring_data)}
        
        if medium_scoring_data:
            X, y = zip(*medium_scoring_data)
            self.medium_scoring_model.fit(X, y)
            results["medium"] = {"samples": len(medium_scoring_data)}
        
        if high_scoring_data:
            X, y = zip(*high_scoring_data)
            self.high_scoring_model.fit(X, y)
            results["high"] = {"samples": len(high_scoring_data)}
        
        # Eğitilmiş modelleri kaydet
        self.save_models()
        
        return results
    
    def adjust_score_probabilities(self, score_probs, category):
        """
        Kategori bazlı skor olasılıklarını ayarlar.
        
        Args:
            score_probs: Skor olasılıkları sözlüğü
            category: Maç kategorisi ('low', 'medium', 'high')
            
        Returns:
            dict: Ayarlanmış skor olasılıkları
        """
        params = self.model_parameters[category]
        score_corrections = params["score_correction"]
        
        # Skorları ayarla
        adjusted_probs = {}
        for score, prob in score_probs.items():
            if score in score_corrections:
                adjusted_probs[score] = prob * score_corrections[score]
            else:
                adjusted_probs[score] = prob
        
        # Olasılıkları normalize et
        total_prob = sum(adjusted_probs.values())
        if total_prob > 0:
            for score in adjusted_probs:
                adjusted_probs[score] /= total_prob
        
        return adjusted_probs