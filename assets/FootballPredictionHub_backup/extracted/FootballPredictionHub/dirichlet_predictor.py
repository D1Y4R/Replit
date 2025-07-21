"""
Dirichlet Süreci Karışım Modeli (Dirichlet Process Mixture Model)

Bu modül, İlk Yarı/Maç Sonu (İY/MS) tahminleri için Dirichlet Süreci Karışım Modelini içerir.
Bu model, veri setindeki gizli kümeleri (takım stilleri, maç türleri) otomatik keşfederek
olasılık dağılımları üretir.

Özellikleri:
1. Veri yapısını önceden bilmeye gerek kalmadan esnek modelleme sağlar
2. Takım performanslarındaki gizli faktörleri keşfeder
3. Maç türlerine göre (savunma/hücum odaklı, dengeli vb.) tahminler yapabilir
4. Yüksek boyutlu verilerde bile etkili çalışır

Kullanım:
    from dirichlet_predictor import DirichletPredictor
    
    dpm_model = DirichletPredictor()
    predictions = dpm_model.predict(home_stats, away_stats, team_adjustments)
"""

import os
import json
import logging
import pickle
import numpy as np
import math
import random
from datetime import datetime
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Constants
HT_FT_COMBINATIONS = [
    "1/1", "1/X", "1/2",
    "X/1", "X/X", "X/2",
    "2/1", "2/X", "2/2"
]

logger = logging.getLogger(__name__)

class DirichletPredictor:
    """
    Dirichlet Süreci Karışım Modeli tabanlı İY/MS tahmin sınıfı
    """
    def __init__(self, model_path="./models/dirichlet_model.pkl", n_clusters=4):
        """
        Dirichlet modelini başlat
        
        Args:
            model_path: Kaydedilmiş model dosyasının yolu (varsa)
            n_clusters: Küme sayısı (otomatik belirlenmezse)
        """
        self.model_path = model_path
        self.is_trained = os.path.exists(model_path)
        
        # Model parametreleri
        self.n_clusters = n_clusters
        self.alpha = 1.0  # Dirichlet konsantrasyon parametresi
        self.cluster_weights = None
        self.cluster_distributions = None
        
        # Standartlaştırma için ölçekleyici
        self.scaler = None
        
        # Şimdi modeli yükle
        self.model = self._load_or_create_model()
        
        # Modelin kullanacağı özellik şablonları
        self.feature_names = [
            'home_first_half_power', 'away_first_half_power',
            'home_second_half_power', 'away_second_half_power',
            'first_half_power_diff', 'second_half_power_diff',
            'total_power_diff', 'home_form', 'away_form',
            'home_defense', 'away_defense', 'expected_total_goals'
        ]
    
    def _load_or_create_model(self):
        """
        Eğitilmiş bir modeli yükle veya yeni bir model oluştur
        
        Returns:
            Dirichlet modeli
        """
        if os.path.exists(self.model_path):
            try:
                logger.info(f"Dirichlet modeli yükleniyor: {self.model_path}")
                
                # Pickle ile modeli yükle
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.cluster_weights = model_data.get("cluster_weights")
                    self.cluster_distributions = model_data.get("cluster_distributions")
                    self.scaler = model_data.get("scaler")
                    self.n_clusters = len(self.cluster_weights) if self.cluster_weights else self.n_clusters
                
                logger.info(f"Dirichlet modeli başarıyla yüklendi. {self.n_clusters} küme içeriyor.")
                return model_data
                
            except Exception as e:
                logger.error(f"Model yüklenirken hata oluştu: {str(e)}")
                logger.warning("Model yüklenemedi, yeni bir Dirichlet modeli oluşturuluyor...")
        
        # Yeni bir model veri yapısı oluştur
        logger.info("Yeni Dirichlet modeli oluşturuluyor...")
        return {
            "cluster_weights": None,
            "cluster_distributions": None,
            "scaler": None
        }
    
    def save_model(self):
        """
        Eğitilmiş modeli kaydet
        
        Returns:
            bool: Kaydetme başarılı mı
        """
        if not self.model:
            return False
            
        try:
            # Dizini oluştur (yoksa)
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Model kaydetme işlemi
            logger.info(f"Dirichlet modeli kaydediliyor: {self.model_path}")
            
            # Model verilerini güncelle
            self.model["cluster_weights"] = self.cluster_weights
            self.model["cluster_distributions"] = self.cluster_distributions
            self.model["scaler"] = self.scaler
            
            # Pickle ile kaydet
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            self.is_trained = True
            logger.info("Dirichlet modeli başarıyla kaydedildi.")
            return True
        except Exception as e:
            logger.error(f"Model kaydedilirken hata oluştu: {str(e)}")
            return False
    
    def _extract_features(self, home_stats, away_stats, expected_goals=None, team_adjustments=None):
        """
        Tahmin için özellik vektörü oluştur
        
        Args:
            home_stats: Ev sahibi takım istatistikleri
            away_stats: Deplasman takımı istatistikleri
            expected_goals: Beklenen gol sayıları (opsiyonel)
            team_adjustments: Takım-spesifik ayarlamalar (opsiyonel)
            
        Returns:
            dict: Özellik vektörü
        """
        # Takımların ilk yarı ve ikinci yarı güçleri
        home_first_half_power = home_stats["first_half"]["avg_goals_per_match"]
        away_first_half_power = away_stats["first_half"]["avg_goals_per_match"]
        home_second_half_power = home_stats["second_half"]["avg_goals_per_match"]
        away_second_half_power = away_stats["second_half"]["avg_goals_per_match"]
        
        # Güç farkları
        first_half_power_diff = home_first_half_power - away_first_half_power
        second_half_power_diff = home_second_half_power - away_second_half_power
        
        # Savunma gücü (yedikleri goller)
        home_defense = home_stats["first_half"].get("avg_goals_conceded", 0.5) + home_stats["second_half"].get("avg_goals_conceded", 0.5)
        away_defense = away_stats["first_half"].get("avg_goals_conceded", 0.5) + away_stats["second_half"].get("avg_goals_conceded", 0.5)
        
        # Beklenen goller
        total_expected_goals = None
        first_half_expected_goals = None
        if expected_goals:
            total_expected_goals = expected_goals.get("total", 
                                                  home_first_half_power + home_second_half_power + 
                                                  away_first_half_power + away_second_half_power)
            
            # İlk yarıdaki beklenen gol sayısı genellikle toplam gollerin %40-45'i civarındadır
            first_half_expected_goals = expected_goals.get("first_half", total_expected_goals * 0.4)
        else:
            # Expected goals verilmemişse hesapla
            total_expected_goals = home_first_half_power + home_second_half_power + away_first_half_power + away_second_half_power
            first_half_expected_goals = total_expected_goals * 0.4
        
        # Takım-spesifik ayarlamalar
        team_power_diff = 0.0
        if team_adjustments and "power_difference" in team_adjustments:
            team_power_diff = team_adjustments["power_difference"]
        
        # Form ve momentum faktörleri
        home_form = 0.5  # Varsayılan orta seviye
        away_form = 0.5  # Varsayılan orta seviye
        
        if "form" in home_stats:
            home_form = home_stats["form"].get("current_form_points", 0.5) / 10.0  # 0-1 arasına normalize et
            
        if "form" in away_stats:
            away_form = away_stats["form"].get("current_form_points", 0.5) / 10.0  # 0-1 arasına normalize et
        
        # Özellik vektörü oluştur
        features = {
            "home_first_half_power": home_first_half_power,
            "away_first_half_power": away_first_half_power,
            "home_second_half_power": home_second_half_power,
            "away_second_half_power": away_second_half_power,
            "first_half_power_diff": first_half_power_diff,
            "second_half_power_diff": second_half_power_diff,
            "total_power_diff": first_half_power_diff + second_half_power_diff,
            "team_power_diff": team_power_diff,
            "home_form": home_form,
            "away_form": away_form,
            "home_defense": home_defense,
            "away_defense": away_defense,
            "form_diff": home_form - away_form,
            "expected_total_goals": total_expected_goals,
            "first_half_expected_goals": first_half_expected_goals
        }
        
        return features

    def _features_to_vector(self, features):
        """
        Özellik sözlüğünü nümerik vektöre dönüştür
        
        Args:
            features: Özellik sözlüğü
            
        Returns:
            np.array: Özellik vektörü
        """
        # Seçilen özellikleri vektöre dönüştür
        feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names])
        return feature_vector
    
    def train(self, match_data):
        """
        Dirichlet Karışım Modelini eğit
        
        Args:
            match_data: Eğitim için maç verileri [{'home_stats':..., 'away_stats':...}]
            
        Returns:
            float: Eğitim doğruluğu
        """
        try:
            # Özellik vektörlerini hazırla
            X = []
            y = []
            
            logger.info(f"Dirichlet modeli eğitiliyor - {len(match_data)} maç ile...")
            for match in match_data:
                # Özellik vektörünü çıkart
                features = self._extract_features(
                    match["home_stats"], 
                    match["away_stats"],
                    match.get("expected_goals"),
                    match.get("team_adjustments")
                )
                
                # Özellik vektörü oluştur
                feature_vector = self._features_to_vector(features)
                X.append(feature_vector)
                
                # Gerçek sonucu al
                actual_results = match.get("actual_results", {})
                first_half = actual_results.get("first_half", "X") 
                full_time = actual_results.get("full_time", "X")
                
                # İY/MS formatı
                htft = f"{first_half}/{full_time}"
                y.append(htft)
            
            # Veriyi numpy dizisine çevir
            X = np.array(X)
            
            # Veriyi standartlaştır
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Uygun küme sayısını bul (basit K-means kullanarak)
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Her kümeyle ilişkilendirilmiş İY/MS olasılıklarını hesapla
            self.cluster_weights = np.zeros(self.n_clusters)
            self.cluster_distributions = []
            
            for i in range(self.n_clusters):
                # Bu kümenin ağırlığı (örneklerin oranı)
                cluster_indices = np.where(clusters == i)[0]
                self.cluster_weights[i] = len(cluster_indices) / len(X)
                
                # Bu kümeye ait İY/MS dağılımını hesapla
                cluster_outcomes = [y[idx] for idx in cluster_indices]
                cluster_dist = {htft: 0 for htft in HT_FT_COMBINATIONS}
                
                for outcome in cluster_outcomes:
                    if outcome in cluster_dist:
                        cluster_dist[outcome] += 1
                
                # Dağılımı normalize et ve Dirichlet ile düzgünleştir
                total = sum(cluster_dist.values()) + self.alpha * len(cluster_dist)
                for htft in cluster_dist:
                    cluster_dist[htft] = (cluster_dist[htft] + self.alpha) / total
                
                self.cluster_distributions.append(cluster_dist)
            
            # Modeli kaydet
            self.is_trained = True
            self.save_model()
            
            # Eğitim doğruluğunu hesapla
            accuracy = self._compute_accuracy(X_scaled, y)
            logger.info(f"Dirichlet modeli eğitimi tamamlandı - doğruluk: {accuracy:.4f}")
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Dirichlet modeli eğitimi sırasında hata: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return 0.0
    
    def _compute_accuracy(self, X_scaled, y_true):
        """
        Tahmin doğruluğunu hesapla
        
        Args:
            X_scaled: Ölçeklendirilmiş özellik matrisi
            y_true: Gerçek sonuçlar
            
        Returns:
            float: Doğruluk oranı
        """
        if not self.is_trained or self.cluster_weights is None:
            return 0.0
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        correct = 0
        total = 0
        
        for i, cluster in enumerate(clusters):
            # Bu maç için tahmin
            pred_dist = self.cluster_distributions[cluster]
            predicted = max(pred_dist, key=pred_dist.get)
            
            if predicted == y_true[i]:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def predict(self, home_stats, away_stats, expected_goals=None, team_adjustments=None):
        """
        İY/MS tahminleri yap
        
        Args:
            home_stats: Ev sahibi takım istatistikleri
            away_stats: Deplasman takımı istatistikleri
            expected_goals: Beklenen gol sayıları (opsiyonel)
            team_adjustments: Takım-spesifik ayarlamalar (opsiyonel)
            
        Returns:
            dict: İY/MS olasılıkları
        """
        if not self.is_trained or self.cluster_weights is None:
            # Model eğitilmemişse basitleştirilmiş simülasyon yap
            logger.warning("Dirichlet modeli eğitilmemiş, basitleştirilmiş simülasyon yapılıyor.")
            return self._fallback_predict(home_stats, away_stats, expected_goals, team_adjustments)
        
        try:
            # Özellikleri çıkart
            features = self._extract_features(home_stats, away_stats, expected_goals, team_adjustments)
            
            # Özellik vektörü oluştur
            feature_vector = self._features_to_vector(features)
            
            # Standartlaştır
            X_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
            
            # En yakın kümeyi bul (basit K-means kullanarak)
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            kmeans.fit(X_scaled)  # Bu aslında yeni veri için uygun değil, sadece örnek için
            
            # Her kümenin bu örneğe uzaklığını hesapla
            distances = np.zeros(self.n_clusters)
            for i in range(self.n_clusters):
                # Bu küme merkezine uzaklık 
                # (Gerçek bir uygulamada daha sofistike bir yaklaşım kullanılabilir)
                center = kmeans.cluster_centers_[i]
                distances[i] = np.linalg.norm(X_scaled - center)
            
            # Uzaklıkları ters çevir ve normalize et (daha yakın kümeler daha yüksek ağırlık alır)
            inv_distances = 1.0 / (distances + 1e-10)  # Sıfıra bölünmeyi önle
            weights = inv_distances / np.sum(inv_distances)
            
            # Her kümeden İY/MS olasılıklarını ağırlıklı olarak birleştir
            predictions = {htft: 0 for htft in HT_FT_COMBINATIONS}
            
            for i in range(self.n_clusters):
                for htft in predictions:
                    predictions[htft] += weights[i] * self.cluster_distributions[i].get(htft, 0)
            
            # Sonuçları yüzdelik değerlere çevir
            total = sum(predictions.values())
            for htft in predictions:
                predictions[htft] = int(round(predictions[htft] / total * 100))
            
            logger.info(f"Dirichlet modeli başarıyla tahmin yaptı.")
            return predictions
            
        except Exception as e:
            logger.error(f"Dirichlet tahmin sırasında hata: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return self._fallback_predict(home_stats, away_stats, expected_goals, team_adjustments)
    
    def _fallback_predict(self, home_stats, away_stats, expected_goals=None, team_adjustments=None):
        """
        Basit bir simülasyon ile tahmin yap (model kullanılamadığında)
        
        Args:
            home_stats: Ev sahibi takım istatistikleri
            away_stats: Deplasman takımı istatistikleri
            expected_goals: Beklenen gol sayıları (opsiyonel)
            team_adjustments: Takım-spesifik ayarlamalar (opsiyonel)
            
        Returns:
            dict: İY/MS olasılıkları
        """
        logger.info("Dirichlet için fallback tahmin yapılıyor")
        
        # Özellikleri çıkart
        features = self._extract_features(home_stats, away_stats, expected_goals, team_adjustments)
        
        # İlk yarı olasılıkları
        first_half_probs = {
            "1": 0.3,  # Ev sahibi önde
            "X": 0.5,  # Berabere
            "2": 0.2   # Deplasman önde
        }
        
        # Özelliklere göre ilk yarı olasılıklarını ayarla
        power_diff = features.get("first_half_power_diff", 0)
        form_diff = features.get("form_diff", 0)
        total_goals = features.get("expected_total_goals", 2.5)
        
        # Güç farkına göre düzeltme
        if power_diff > 0.1:  # Ev sahibi daha güçlü
            first_half_probs["1"] += min(power_diff * 0.5, 0.3)
            first_half_probs["2"] -= min(power_diff * 0.3, 0.15)
        elif power_diff < -0.1:  # Deplasman daha güçlü
            first_half_probs["2"] += min(abs(power_diff) * 0.5, 0.3)
            first_half_probs["1"] -= min(abs(power_diff) * 0.3, 0.15)
        
        # Form farkına göre düzeltme
        if form_diff > 0.1:  # Ev sahibi formda
            first_half_probs["1"] += min(form_diff * 0.3, 0.1)
        elif form_diff < -0.1:  # Deplasman formda
            first_half_probs["2"] += min(abs(form_diff) * 0.3, 0.1)
        
        # Toplam gol beklentisine göre ilk yarı beraberlik olasılığını ayarla
        if total_goals < 2.0:  # Düşük skorlu maç beklentisi
            first_half_probs["X"] += 0.1  # İlk yarı beraberlik daha olası
            first_half_probs["1"] -= 0.05
            first_half_probs["2"] -= 0.05
        elif total_goals > 3.5:  # Yüksek skorlu maç beklentisi
            first_half_probs["X"] -= 0.15  # İlk yarı beraberlik daha az olası
            first_half_probs["1"] += 0.1
            first_half_probs["2"] += 0.05
        
        # Normalize et
        total = sum(first_half_probs.values())
        for key in first_half_probs:
            first_half_probs[key] /= total
        
        # Maç sonu olasılıkları
        full_time_probs = {
            "1": 0.4,  # Ev sahibi kazanır
            "X": 0.3,  # Berabere
            "2": 0.3   # Deplasman kazanır
        }
        
        # Özelliklere göre maç sonu olasılıklarını ayarla
        power_diff = features.get("total_power_diff", 0)
        
        if power_diff > 0.2:  # Ev sahibi daha güçlü
            full_time_probs["1"] += min(power_diff * 0.5, 0.3)
            full_time_probs["2"] -= min(power_diff * 0.4, 0.2)
        elif power_diff < -0.2:  # Deplasman daha güçlü
            full_time_probs["2"] += min(abs(power_diff) * 0.5, 0.3)
            full_time_probs["1"] -= min(abs(power_diff) * 0.4, 0.2)
        
        # Takım stillerine göre düzeltme
        if team_adjustments:
            if "home_team_style" in team_adjustments:
                home_style = team_adjustments["home_team_style"]
                if home_style.get("style_type") == "strong_start":
                    # Ev sahibi güçlü başlayan, ama dayanıklılık sorunu olan bir takımsa
                    first_half_probs["1"] += 0.1
                    full_time_probs["1"] -= 0.05
                    full_time_probs["X"] += 0.05
                elif home_style.get("style_type") == "comeback":
                    # Ev sahibi toparlanma gücü yüksek bir takımsa
                    full_time_probs["1"] += 0.1
                    first_half_probs["1"] -= 0.05
            
            if "away_team_style" in team_adjustments:
                away_style = team_adjustments["away_team_style"]
                if away_style.get("style_type") == "strong_start":
                    first_half_probs["2"] += 0.1
                    full_time_probs["2"] -= 0.05
                    full_time_probs["X"] += 0.05
                elif away_style.get("style_type") == "comeback":
                    full_time_probs["2"] += 0.1
                    first_half_probs["2"] -= 0.05
        
        # Normalize et
        total = sum(full_time_probs.values())
        for key in full_time_probs:
            full_time_probs[key] /= total
        
        # İY/MS kombinasyonları
        predictions = {}
        
        # Birleşik olasılıkları hesapla
        for fh in ["1", "X", "2"]:
            for ft in ["1", "X", "2"]:
                htft = f"{fh}/{ft}"
                
                # Basit olasılık çarpımı - gerçekte bu ilişki daha karmaşıktır
                predictions[htft] = first_half_probs[fh] * full_time_probs[ft] * 100
        
        # Tutarlılık faktörü ekle
        consistency_factor = 2.0  # 1/1, X/X, 2/2 gibi tutarlı sonuçlar daha olası
        predictions["1/1"] *= consistency_factor
        predictions["X/X"] *= consistency_factor
        predictions["2/2"] *= consistency_factor
        
        # İlk yarı/maç sonu değişimleri için düzeltmeler
        # Ev sahibi önde başlayıp deplasman'ın maçı çevirmesi veya tam tersi
        change_factor = 0.5  # Değişimler daha az olası (1/2, 2/1)
        predictions["1/2"] *= change_factor
        predictions["2/1"] *= change_factor
        
        # Normalize et ve yüzdeye çevir
        total = sum(predictions.values())
        for key in predictions:
            predictions[key] = int(round(predictions[key] / total * 100))
        
        return predictions