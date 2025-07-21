"""
Conditional Random Fields (CRF) Tahmin Modeli

Bu modül, İlk Yarı/Maç Sonu (İY/MS) tahminleri için Conditional Random Fields modelini içerir.
CRF, yapılandırılmış tahmin için güçlü bir olasılıksal modeldir ve özellikle sıralı veri için uygundur.

Özellikleri:
1. İlk yarı ve ikinci yarı arasındaki geçiş yapılarını modelleyebilir
2. Takım formu, güç farkı, lig özellikleri gibi bağlamsal bilgileri kullanır
3. Nadir görülen sonuçlar (1/2, 2/1 gibi) için daha iyi tahminler üretir

Kullanım:
    from crf_predictor import CRFPredictor
    
    crf_model = CRFPredictor()
    predictions = crf_model.predict(home_stats, away_stats, team_adjustments)
"""

import os
import json
import logging
import random
import pickle
import numpy as np
import math
from tqdm import tqdm
from datetime import datetime

# sklearn-crfsuite kütüphanesi
try:
    import sklearn_crfsuite
    from sklearn_crfsuite import metrics
    has_crfsuite = True
except ImportError:
    has_crfsuite = False
    logging.warning("sklearn-crfsuite kütüphanesi bulunamadı, CRF modeli kullanılamayacak.")

# Constants
HT_FT_COMBINATIONS = [
    "1/1", "1/X", "1/2",
    "X/1", "X/X", "X/2",
    "2/1", "2/X", "2/2"
]

logger = logging.getLogger(__name__)

class CRFPredictor:
    """
    Conditional Random Fields (CRF) tabanlı İY/MS tahmin sınıfı
    """
    def __init__(self, model_path="./models/crf_model.pkl"):
        """
        CRF modelini başlat
        
        Args:
            model_path: Kaydedilmiş model dosyasının yolu (varsa)
        """
        self.model_path = model_path
        self.is_trained = os.path.exists(model_path)
        
        # CRF ayarları - önce bu ayarları tanımla, sonra modeli yükle
        self.c1 = 0.1  # L1 düzenlileştirme katsayısı
        self.c2 = 0.1  # L2 düzenlileştirme katsayısı
        self.max_iterations = 100  # Maksimum iterasyon sayısı
        
        # Şimdi modeli yükle
        self.model = self._load_or_create_model()
        
        # Modelin kullanacağı özellik şablonları
        self.feature_templates = {
            'home_first_half_power': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
            'away_first_half_power': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
            'home_second_half_power': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
            'away_second_half_power': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
            'power_difference': [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3],
            'total_expected_goals': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            'first_half_expected_goals': [0.5, 0.8, 1.0, 1.2, 1.5, 1.8],
            'home_form': [0.3, 0.4, 0.5, 0.6, 0.7],
            'away_form': [0.3, 0.4, 0.5, 0.6, 0.7],
            'home_form_trend': [-0.2, -0.1, 0, 0.1, 0.2],  # Form trendi (düşüş/yükseliş) - YENİ
            'away_form_trend': [-0.2, -0.1, 0, 0.1, 0.2],  # Form trendi (düşüş/yükseliş) - YENİ
            'recent_goals_avg': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],  # Son maçlarda ortalama gol - YENİ
            'recent_conceded_avg': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],  # Son maçlarda yenilen gol - YENİ
        }
        
        # HT/FT Geçiş Matrisi - Markov modeli için olasılıkları tanımla
        # Bu matris, ilk yarı sonucundan maç sonu sonucuna geçiş olasılıklarını belirler
        # Kullanıcı geri bildirimine dayalı olarak güncellendi (28.03.2025)
        # İlk yarı ve maç sonu için ayrı ayrı değerlendirme gerekliliğine göre düzenlendi
        self.transition_matrix = {
            # İlk yarı sonucu: {Maç sonu sonuçları olasılıkları}
            "1": {"1": 0.60, "X": 0.30, "2": 0.10},  # İlk yarı ev sahibi önde -> %60 maç sonu ev sahibi kazanır
            "X": {"1": 0.40, "X": 0.35, "2": 0.25},  # İlk yarı berabere -> %40 maç sonu ev sahibi kazanır
            "2": {"1": 0.10, "X": 0.30, "2": 0.60}   # İlk yarı deplasman önde -> %60 maç sonu deplasman kazanır
        }
        # DİKKAT: İlk yarı 1/X/2 ile maç sonu 1/X/2 arasında önceki gibi güçlü bir
        # ilişki modellenmemektedir. Bu sayede 1/2, 2/1, X/1, X/2 gibi "değişen sonuç"
        # kombinasyonları için daha yüksek olasılıklar sağlanır.
        
        # Takım-bazlı HT/FT özellikleri - YENİ
        # Takımlar için karakteristik HT/FT örüntülerini saklar
        self.team_patterns = {}
    
    def _load_or_create_model(self):
        """
        Eğitilmiş bir CRF modelini yükle veya yeni bir model oluştur
        
        Returns:
            CRF modeli
        """
        if not has_crfsuite:
            logger.warning("sklearn-crfsuite kütüphanesi yüklü değil. CRF modeli kullanılamıyor.")
            return None
            
        try:
            if os.path.exists(self.model_path):
                try:
                    logger.info(f"CRF modeli yükleniyor: {self.model_path}")
                    
                    # Pickle ile modeli yükle
                    with open(self.model_path, 'rb') as f:
                        model = pickle.load(f)
                        
                    logger.info("CRF modeli başarıyla yüklendi.")
                    
                    # Model tipi kontrolü yap
                    if not hasattr(model, 'predict'):
                        logger.warning("Yüklenen model doğru formatta değil. Yeni model oluşturuluyor.")
                        raise ValueError("Geçersiz model formatı")
                        
                    return model
                except Exception as e:
                    logger.error(f"CRF modeli yüklenirken hata oluştu: {str(e)}")
                    logger.warning("Yükleme hatası nedeniyle yeni model oluşturuluyor...")
            else:
                logger.warning(f"Model dosyası bulunamadı: {self.model_path}")
            
            # Eğer model dosyası yoksa veya yüklenemezse yeni bir model oluştur
            logger.info("Yeni CRF modeli oluşturuluyor...")
            
            # sklearn-crfsuite kullanarak bir CRF modeli oluştur
            model = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                c1=self.c1,
                c2=self.c2,
                max_iterations=self.max_iterations,
                all_possible_transitions=True
            )
            
            logger.info("Yeni CRF modeli başarıyla oluşturuldu.")
            return model
            
        except Exception as e:
            logger.error(f"CRF modeli oluşturulurken beklenmeyen hata: {str(e)}")
            logger.warning("CRF modeli kullanılamıyor. Alternatif tahmin yöntemleri kullanılacak.")
            return None
    
    def save_model(self):
        """
        Eğitilmiş CRF modelini kaydet
        
        Returns:
            bool: Kaydetme başarılı mı
        """
        if not self.model or not has_crfsuite:
            return False
            
        try:
            # Dizini oluştur (yoksa)
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Model kaydetme işlemi
            logger.info(f"CRF modeli kaydediliyor: {self.model_path}")
            
            # Basit bir pickle kaydetme işlemi
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            self.is_trained = True
            logger.info("CRF modeli başarıyla kaydedildi.")
            return True
        except Exception as e:
            logger.error(f"Model kaydedilirken hata oluştu: {str(e)}")
            return False
    
    def update_team_patterns(self, team_id, match_result):
        """
        Takım-bazlı HT/FT örüntülerini güncelle
        
        Args:
            team_id: Takım ID'si
            match_result: Maç sonucu (HT/FT içermeli)
            
        Returns:
            None
        """
        if team_id not in self.team_patterns:
            # Takım için veri yapısını başlat
            self.team_patterns[team_id] = {
                "matches": 0,
                "patterns": {htft: 0 for htft in HT_FT_COMBINATIONS}
            }
        
        # Maç sayacını artır
        self.team_patterns[team_id]["matches"] += 1
        
        # HT/FT örüntüsünü al
        ht_result = match_result.get("first_half")
        ft_result = match_result.get("full_time")
        
        if ht_result and ft_result:
            htft = f"{ht_result}/{ft_result}"
            if htft in self.team_patterns[team_id]["patterns"]:
                self.team_patterns[team_id]["patterns"][htft] += 1
    
    def get_team_htft_patterns(self, team_id, is_home=True):
        """
        Takımın HT/FT örüntü olasılıklarını döndür
        
        Args:
            team_id: Takım ID'si
            is_home: Takımın ev sahibi olup olmadığı
            
        Returns:
            dict: HT/FT olasılıkları veya None (veri yoksa)
        """
        if team_id not in self.team_patterns or self.team_patterns[team_id]["matches"] < 5:
            return None
            
        # Toplam maç sayısı
        total_matches = self.team_patterns[team_id]["matches"]
        
        # Her kombinasyon için olasılıkları hesapla (minimum değer 0.01)
        probabilities = {}
        for htft in HT_FT_COMBINATIONS:
            count = self.team_patterns[team_id]["patterns"].get(htft, 0)
            # Laplace smoothing uygula (0 olmaması için)
            probabilities[htft] = (count + 0.5) / (total_matches + 4.5)
            
        # Ev sahibi takımlar daha fazla galibiyet alır
        if is_home:
            # Ev sahibi lehine kombinasyonları biraz artır
            for htft in ["1/1", "X/1", "1/X"]:
                probabilities[htft] *= 1.2
        else:
            # Deplasman lehine kombinasyonları biraz artır 
            for htft in ["2/2", "X/2", "2/X"]:
                probabilities[htft] *= 1.2
                
        # Normalize et (toplamları 1 olmalı)
        total = sum(probabilities.values())
        probabilities = {k: v/total for k, v in probabilities.items()}
        
        return probabilities
            
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
        
        # Beklenen goller
        total_expected_goals = None
        first_half_expected_goals = None
        if expected_goals:
            total_expected_goals = expected_goals.get("total", 
                                                      home_first_half_power + home_second_half_power + 
                                                      away_first_half_power + away_second_half_power)
            
            # İlk yarıdaki beklenen gol sayısı genellikle toplam gollerin %40-45'i civarındadır
            first_half_expected_goals = expected_goals.get("first_half", total_expected_goals * 0.4)
        
        # Takım-spesifik ayarlamalar
        home_team_style = None
        away_team_style = None
        team_power_diff = 0.0
        h2h_info = None
        
        if team_adjustments:
            if "home_team_style" in team_adjustments:
                home_team_style = team_adjustments["home_team_style"]
            if "away_team_style" in team_adjustments:
                away_team_style = team_adjustments["away_team_style"]
            if "power_difference" in team_adjustments:
                team_power_diff = team_adjustments["power_difference"]
            if "h2h_analysis" in team_adjustments:
                h2h_info = team_adjustments["h2h_analysis"]
        
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
            "form_diff": home_form - away_form,
        }
        
        # Beklenen gol bilgileri varsa ekle
        if total_expected_goals is not None:
            features["total_expected_goals"] = total_expected_goals
            features["first_half_expected_goals"] = first_half_expected_goals
        
        # Takım stili ve H2H bilgileri varsa ekle
        if home_team_style:
            features["home_strong_start"] = 1 if home_team_style.get("style_type") == "strong_start" else 0
            features["home_comeback"] = 1 if home_team_style.get("style_type") == "comeback" else 0
            
        if away_team_style:
            features["away_strong_start"] = 1 if away_team_style.get("style_type") == "strong_start" else 0
            features["away_comeback"] = 1 if away_team_style.get("style_type") == "comeback" else 0
            
        if h2h_info:
            features["h2h_unpredictable"] = 1 if h2h_info.get("unpredictable", False) else 0
            features["h2h_home_dominant"] = 1 if h2h_info.get("home_dominant", False) else 0
            features["h2h_away_dominant"] = 1 if h2h_info.get("away_dominant", False) else 0
        
        return features
    
    def _extract_sequence_features(self, features, position):
        """
        CRF için sıralı etiketleme özelliklerini çıkart
        
        Args:
            features: Özellik vektörü
            position: Dizideki konum ('first_half' veya 'full_time')
            
        Returns:
            dict: CRF için özellik sözlüğü
        """
        # CRF eğitimi için özellik sözlüğü oluştur
        # Bu model bir dizinin farklı pozisyonları için özellikleri çıkarır
        crf_features = {}
        
        # Her özellik için ayrık aralıkları belirle
        for feat_name, feat_value in features.items():
            if feat_name in self.feature_templates:
                thresholds = self.feature_templates[feat_name]
                # Özelliği aralıklara göre ayrıklaştır
                for i, threshold in enumerate(thresholds):
                    if feat_value <= threshold:
                        bin_id = i
                        break
                else:
                    bin_id = len(thresholds)
                
                # Özellik adı + bin formatında ayrıklaştırılmış özellik oluştur
                crf_features[f"{feat_name}_{position}={bin_id}"] = 1
            else:
                # Ayrıklaştırma şablonu olmayan özellikler için doğrudan değeri kullan
                crf_features[f"{feat_name}_{position}"] = feat_value
        
        # Özel ek özellikler
        # Düşük skorlu maç mı? (<2.0 toplam beklenen gol)
        if "total_expected_goals" in features:
            if features["total_expected_goals"] < 1.5:
                crf_features[f"low_scoring_{position}"] = 1
            elif features["total_expected_goals"] < 2.0:
                crf_features[f"medium_low_scoring_{position}"] = 1
            elif features["total_expected_goals"] > 3.0:
                crf_features[f"high_scoring_{position}"] = 1
        
        # Dengeli maç mı? (güç farkı < 0.2)
        if abs(features.get("team_power_diff", 0)) < 0.2:
            crf_features[f"balanced_match_{position}"] = 1
        
        # Form farkı belirgin mi?
        if abs(features.get("form_diff", 0)) > 0.2:
            crf_features[f"significant_form_diff_{position}"] = 1
            if features.get("form_diff", 0) > 0:
                crf_features[f"home_better_form_{position}"] = 1
            else:
                crf_features[f"away_better_form_{position}"] = 1
        
        return crf_features
    
    def _prepare_training_sequence(self, match_data):
        """
        CRF eğitimi için dizi verileri hazırla
        
        Args:
            match_data: Eğitim için maç verileri
            
        Returns:
            list: Eğitim için özellik dizileri
            list: Etiket dizileri
        """
        X = []  # Özellik dizileri
        y = []  # Etiket dizileri
        
        for match in match_data:
            # Temel özellikleri çıkart
            features = self._extract_features(
                match["home_stats"], 
                match["away_stats"],
                match.get("expected_goals"),
                match.get("team_adjustments")
            )
            
            # Dizi halinde özellikler (first_half ve full_time için)
            match_features = [
                self._extract_sequence_features(features, "first_half"),
                self._extract_sequence_features(features, "full_time")
            ]
            
            # Gerçek sonuçlar
            actual_results = match.get("actual_results", {})
            first_half = actual_results.get("first_half", "X")  # Varsayılan X (berabere)
            full_time = actual_results.get("full_time", "X")   # Varsayılan X (berabere)
            
            match_labels = [first_half, full_time]
            
            X.append(match_features)
            y.append(match_labels)
        
        return X, y
    
    def train(self, match_data):
        """
        CRF modelini eğit
        
        Args:
            match_data: Eğitim için maç verileri
            
        Returns:
            float: Eğitim doğruluğu
        """
        if not has_crfsuite or not self.model:
            logger.error("CRF modeli kullanılamıyor - sklearn-crfsuite kütüphanesi yüklü değil.")
            return 0.0
        
        try:
            # Eğitim verilerini hazırla
            X_train, y_train = self._prepare_training_sequence(match_data)
            
            if not X_train:
                logger.error("Eğitim verileri hazırlanamadı.")
                return 0.0
                
            # Modeli eğit
            logger.info(f"CRF modeli eğitiliyor - {len(X_train)} örnek...")
            self.model.fit(X_train, y_train)
            
            # Modelin doğruluğunu test et
            y_pred = self.model.predict(X_train)
            accuracy = self._compute_accuracy(y_train, y_pred)
            
            logger.info(f"CRF eğitimi tamamlandı - doğruluk: {accuracy:.4f}")
            
            # Modeli kaydet
            self.is_trained = True
            self.save_model()
            
            return accuracy
            
        except Exception as e:
            logger.error(f"CRF eğitimi sırasında hata: {str(e)}")
            return 0.0
    
    def _compute_accuracy(self, y_true, y_pred):
        """
        Tahmin doğruluğunu hesapla
        
        Args:
            y_true: Gerçek etiketler
            y_pred: Tahmin edilen etiketler
            
        Returns:
            float: Doğruluk oranı
        """
        correct = 0
        total = 0
        
        for true_seq, pred_seq in zip(y_true, y_pred):
            for true_label, pred_label in zip(true_seq, pred_seq):
                if true_label == pred_label:
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
        if not has_crfsuite or not self.model:
            # CRF modeli kullanılamıyorsa monte carlo temelli bir simülasyon yap
            logger.warning("CRF modeli kullanılamıyor, basitleştirilmiş simülasyon yapılıyor.")
            return self._fallback_predict(home_stats, away_stats, expected_goals, team_adjustments)
        
        try:
            # Özellikleri çıkart
            features = self._extract_features(home_stats, away_stats, expected_goals, team_adjustments)
            
            # Takım bazlı HT/FT örüntülerini kontrol et - YENİ
            home_team_id = None
            away_team_id = None
            
            if team_adjustments:
                home_team_id = team_adjustments.get("home_team_id")
                away_team_id = team_adjustments.get("away_team_id")
            
            # Takım bazlı örüntüleri al - YENİ
            home_team_patterns = None
            away_team_patterns = None
            
            if home_team_id:
                home_team_patterns = self.get_team_htft_patterns(home_team_id, is_home=True)
                if home_team_patterns:
                    logger.info(f"Ev sahibi takım HT/FT örüntüleri bulundu: {home_team_id}")
                
            if away_team_id:
                away_team_patterns = self.get_team_htft_patterns(away_team_id, is_home=False)
                if away_team_patterns:
                    logger.info(f"Deplasman takımı HT/FT örüntüleri bulundu: {away_team_id}")
            
            # Model kontrolü yap - None olma durumuna karşı ikinci bir kontrol
            if not self.model:
                logger.warning("CRF modeli None. Simülasyon tabanlı tahmin yapılıyor.")
                base_probabilities = self._simulate_probabilities(features)
            else:
                # Dizi halinde özellikler
                match_features = [
                    self._extract_sequence_features(features, "first_half"),
                    self._extract_sequence_features(features, "full_time")
                ]
                
                # CRF modeliyle tahmin
                if hasattr(self.model, 'predict') and self.is_trained:
                    # Eğitilmiş model kullanılabilirse - doğrudan tahmin
                    try:
                        logger.info(f"CRF modeli ile tahmin yapılıyor...")
                        labels = self.model.predict([match_features])[0]
                        logger.info(f"CRF tahmin sonucu: {labels}")
                        base_probabilities = self._compute_probabilities(match_features, labels)
                        logger.info(f"CRF modeli başarıyla tahmin yaptı. Yöntem: trained_model")
                    except Exception as model_error:
                        logger.error(f"Eğitilmiş model tahmin yapamadı: {str(model_error)}")
                        logger.error(f"Hata detayları: {type(model_error).__name__}")
                        # Eğitilmiş model çalışmadıysa simülasyona geri dön
                        logger.warning("Simülasyon tabanlı CRF tahminine dönülüyor.")
                        base_probabilities = self._simulate_probabilities(features)
                else:
                    logger.warning(f"CRF modeli kullanılabilir değil: model={self.model}, is_trained={self.is_trained}, has_predict={hasattr(self.model, 'predict') if self.model else False}")
                    # Model eğitilmemişse veya model tahmin yapamadıysa - simülasyon temelli tahmin yap
                    logger.warning("CRF modeli eğitilmemiş veya çalışmıyor, simülasyon temelli tahmin yapılıyor.")
                    base_probabilities = self._simulate_probabilities(features)
            
            # Takım örüntüleriyle tahminleri birleştir (eğer mevcutsa) - YENİ
            if home_team_patterns or away_team_patterns:
                logger.info("Takım-bazlı HT/FT örüntüleri ile tahminler iyileştiriliyor...")
                return self._combine_with_team_patterns(base_probabilities, home_team_patterns, away_team_patterns)
            else:
                return base_probabilities
                
        except Exception as e:
            logger.error(f"CRF tahmini sırasında hata: {str(e)}")
            return self._fallback_predict(home_stats, away_stats, expected_goals, team_adjustments)
    
    def _simulate_probabilities(self, features):
        """
        Simülasyon temelli olasılık hesaplaması - model eğitilmemişse
        
        Args:
            features: Özellik vektörü
            
        Returns:
            dict: İY/MS olasılıkları
        """
        # İlk yarı olasılıkları
        first_half_probs = {
            "1": 0.3,  # Ev sahibi önde
            "X": 0.5,  # Berabere
            "2": 0.2   # Deplasman önde
        }
        
        # Özelliklere göre ilk yarı olasılıklarını ayarla
        power_diff = features.get("first_half_power_diff", 0)
        form_diff = features.get("form_diff", 0)
        expected_goals = features.get("first_half_expected_goals", 1.2)
        
        # Güç farkına göre düzeltme
        if power_diff > 0.1:  # Ev sahibi daha güçlü
            first_half_probs["1"] += power_diff * 0.5
            first_half_probs["2"] -= power_diff * 0.5
        elif power_diff < -0.1:  # Deplasman daha güçlü
            first_half_probs["2"] += abs(power_diff) * 0.5
            first_half_probs["1"] -= abs(power_diff) * 0.5
            
        # Form farkına göre düzeltme
        if form_diff > 0.1:  # Ev sahibi formda
            first_half_probs["1"] += form_diff * 0.3
            first_half_probs["2"] -= form_diff * 0.3
        elif form_diff < -0.1:  # Deplasman formda
            first_half_probs["2"] += abs(form_diff) * 0.3
            first_half_probs["1"] -= abs(form_diff) * 0.3
            
        # Düşük gollü maçlarda beraberlik olasılığı artar
        if expected_goals < 1.0:
            first_half_probs["X"] += 0.2
            first_half_probs["1"] -= 0.1
            first_half_probs["2"] -= 0.1
            
        # Değerleri normalize et
        total = sum(first_half_probs.values())
        first_half_probs = {k: v/total for k, v in first_half_probs.items()}
        
        # Şartlı ikinci yarı olasılıkları - ilk yarı sonucuna göre değişir
        second_half_probs = {
            "1": {  # İlk yarı ev önde iken
                "1": 0.60,  # MS: ev önde
                "X": 0.30,  # MS: berabere
                "2": 0.10   # MS: deplasman önde
            },
            "X": {  # İlk yarı berabere iken
                "1": 0.40,  # MS: ev önde
                "X": 0.35,  # MS: berabere
                "2": 0.25   # MS: deplasman önde
            }, 
            "2": {  # İlk yarı deplasman önde iken
                "1": 0.15,  # MS: ev önde
                "X": 0.25,  # MS: berabere
                "2": 0.60   # MS: deplasman önde
            }
        }
        
        # Özelliklere göre ikinci yarı olasılıklarını ayarla
        power_diff_second = features.get("second_half_power_diff", 0)
        
        # İkinci yarı güç farkına göre düzeltme
        for first_half in ["1", "X", "2"]:
            if power_diff_second > 0.1:  # Ev sahibi ikinci yarıda daha güçlü
                second_half_probs[first_half]["1"] += power_diff_second * 0.2
                second_half_probs[first_half]["2"] -= power_diff_second * 0.2
            elif power_diff_second < -0.1:  # Deplasman ikinci yarıda daha güçlü
                second_half_probs[first_half]["2"] += abs(power_diff_second) * 0.2
                second_half_probs[first_half]["1"] -= abs(power_diff_second) * 0.2
                
            # Normalize et
            total = sum(second_half_probs[first_half].values())
            second_half_probs[first_half] = {k: v/total for k, v in second_half_probs[first_half].items()}
            
        # Takım stillerine göre ayarla
        home_strong_start = features.get("home_strong_start", 0)
        away_strong_start = features.get("away_strong_start", 0)
        home_comeback = features.get("home_comeback", 0)
        away_comeback = features.get("away_comeback", 0)
        
        # Güçlü başlangıç yapan takımlar ilk yarıda avantajlı
        if home_strong_start:
            adjustment = 0.1
            first_half_probs["1"] += adjustment
            first_half_probs["2"] -= adjustment / 2
            first_half_probs["X"] -= adjustment / 2
            
        if away_strong_start:
            adjustment = 0.1
            first_half_probs["2"] += adjustment
            first_half_probs["1"] -= adjustment / 2
            first_half_probs["X"] -= adjustment / 2
            
        # İkinci yarıda toparlanabilen takımlar için düzenlemeler
        if home_comeback:
            # Ev sahibi ikinci yarıda daha güçlü
            adjustment = 0.15
            second_half_probs["2"]["1"] += adjustment  # 2/1 olasılığı artar
            second_half_probs["2"]["2"] -= adjustment  # 2/2 olasılığı azalır
            
        if away_comeback:
            # Deplasman ikinci yarıda daha güçlü
            adjustment = 0.15
            second_half_probs["1"]["2"] += adjustment  # 1/2 olasılığı artar
            second_half_probs["1"]["1"] -= adjustment  # 1/1 olasılığı azalır
            
        # Son kez normalize et
        for first_half in ["1", "X", "2"]:
            total = sum(second_half_probs[first_half].values())
            second_half_probs[first_half] = {k: v/total for k, v in second_half_probs[first_half].items()}
        
        total = sum(first_half_probs.values())
        first_half_probs = {k: v/total for k, v in first_half_probs.items()}
        
        # İY/MS olasılıklarını hesapla
        htft_probs = {}
        for first_half in ["1", "X", "2"]:
            for full_time in ["1", "X", "2"]:
                htft = f"{first_half}/{full_time}"
                htft_probs[htft] = first_half_probs[first_half] * second_half_probs[first_half][full_time]
        
        # Yüzdeye çevir
        htft_probs = {k: round(v * 100) for k, v in htft_probs.items()}
        
        # Toplamı 100 olacak şekilde düzelt
        total = sum(htft_probs.values())
        if total != 100:
            scale = 100 / total
            for htft in htft_probs:
                htft_probs[htft] = round(htft_probs[htft] * scale)
        
        return htft_probs
    
    def _compute_probabilities(self, features, predicted_labels=None):
        """
        CRF modelinden olasılıkları hesapla
        
        Args:
            features: Özellik dizisi
            predicted_labels: Tahmin edilen etiketler (opsiyonel)
            
        Returns:
            dict: İY/MS olasılıkları
        """
        # Not: Gerçek bir CRF modelinde, modelin marginal olasılıkları hesaplayabilmesi gerekir
        # Ancak sklearn-crfsuite doğrudan bu işlevi sağlamıyor
        # Bu nedenle, bir simülasyon ve Markov zinciri yaklaşımı kullanıyoruz
        
        # Veri yetersiz veya model eğitilmemişse, simülasyon yaklaşımına dön
        if predicted_labels is None or not self.is_trained:
            return self._simulate_probabilities_from_features(features)
            
        # Tahmin edilen diziyi ve olasılıkları belirle
        first_half_pred = predicted_labels[0]
        full_time_pred = predicted_labels[1]
        
        # En yüksek olasılık için tahmin edilen kombinasyon
        best_htft = f"{first_half_pred}/{full_time_pred}"
        
        # Başlangıç olasılıkları (tümü çok düşük)
        htft_probs = {htft: 1 for htft in HT_FT_COMBINATIONS}
        
        # En yüksek olasılığı ayarla
        htft_probs[best_htft] = 30  # En olası tahmin
        
        # 1. İlk yarı olasılıkları - bağımsız hesaplama
        first_half_probs = {
            "1": features.get("home_first_half_power", 0) / (features.get("home_first_half_power", 0) + features.get("away_first_half_power", 0) + 0.5),
            "2": features.get("away_first_half_power", 0) / (features.get("home_first_half_power", 0) + features.get("away_first_half_power", 0) + 0.5),
            "X": 0.5 / (features.get("home_first_half_power", 0) + features.get("away_first_half_power", 0) + 0.5)
        }
        
        # 2. Markov geçiş matrisi kullanarak olasılık hesaplama - YENİ
        for htft in HT_FT_COMBINATIONS:
            first_half, full_time = htft.split('/')
            
            # İlk yarı olasılığını al
            ht_prob = first_half_probs.get(first_half, 0.33)
            
            # İlk yarıdan tam zamana geçiş olasılığını al 
            transition_prob = self.transition_matrix.get(first_half, {}).get(full_time, 0.33)
            
            # Başlangıç ağırlığı hesapla (Markov modeli temelinde)
            markov_weight = ht_prob * transition_prob * 100
            
            # Markov ağırlığını uygula - bu şekilde ilk yarı ve maç sonu arasındaki
            # geçiş ilişkisini kullanarak daha gerçekçi olasılıklar elde ederiz
            htft_probs[htft] = htft_probs[htft] + markov_weight
        
        # Form, güç farkı ve diğer faktörleri olasılıklara yansıt - YENİ
        home_form = features.get("home_form", 0.5)
        away_form = features.get("away_form", 0.5)
        power_diff = features.get("team_power_diff", 0)
        
        # Form ve güç farkına göre bazı kombinasyonları güçlendirelim
        if home_form > 0.6 or power_diff > 0.2:  # Ev sahibi formda veya daha güçlü
            # Ev sahibi lehine sonuçlara ek ağırlık
            htft_probs["1/1"] = htft_probs["1/1"] * 1.3
            htft_probs["X/1"] = htft_probs["X/1"] * 1.2
            # Deplasman lehine sonuçlara ağırlık düşürme
            htft_probs["2/2"] = htft_probs["2/2"] * 0.8
            htft_probs["2/X"] = htft_probs["2/X"] * 0.9
        
        if away_form > 0.6 or power_diff < -0.2:  # Deplasman formda veya daha güçlü
            # Deplasman lehine sonuçlara ek ağırlık
            htft_probs["2/2"] = htft_probs["2/2"] * 1.3
            htft_probs["X/2"] = htft_probs["X/2"] * 1.2
            # Ev sahibi lehine sonuçlara ağırlık düşürme
            htft_probs["1/1"] = htft_probs["1/1"] * 0.8
            htft_probs["1/X"] = htft_probs["1/X"] * 0.9
        
        # Simülasyonla benzer tahminlere daha yüksek olasılıklar ver
        for htft in HT_FT_COMBINATIONS:
            if htft == best_htft:
                continue
                
            first_half, full_time = htft.split('/')
            
            # İlk yarısı doğru tahmin
            if first_half == first_half_pred:
                htft_probs[htft] += 10
                
            # Maç sonu doğru tahmin
            if full_time == full_time_pred:
                htft_probs[htft] += 8
                
            # İlk yarı ve maç sonu tutarlılık ödüllendir
            if (first_half == "1" and full_time == "1") or \
               (first_half == "2" and full_time == "2") or \
               (first_half == "X" and full_time == "X"):
                htft_probs[htft] += 5
        
        # Mantık kontrolleri ve olasılık düzeltmeleri - YENİ
        # Bazı tutarsız kombinasyonları engelleme (örn: 2/1 olasılığı çok yüksekse)
        if htft_probs["2/1"] > htft_probs["2/X"] and htft_probs["2/1"] > 15:
            # 2/1 çok yüksekse, 2/X'i arttır, 2/1'i azalt - ama tamamen sıfırlama
            htft_probs["2/X"] = (htft_probs["2/X"] + htft_probs["2/1"]) / 2
            htft_probs["2/1"] = htft_probs["2/1"] * 0.7
        
        # Sonuçları normalize et
        total = sum(htft_probs.values())
        htft_probs = {k: round(v * 100 / total) for k, v in htft_probs.items()}
        
        # Son bir kez toplamı kontrol et
        total = sum(htft_probs.values())
        if total != 100:
            diff = 100 - total
            # En yüksek olasılıklı tahmini ayarla
            htft_probs[best_htft] += diff
        
        return htft_probs
            
    def _combine_with_team_patterns(self, base_probabilities, home_team_patterns=None, away_team_patterns=None):
        """
        Takım-bazlı HT/FT örüntülerini tahminlerle birleştir
        
        Args:
            base_probabilities: Temel tahmin olasılıkları
            home_team_patterns: Ev sahibi takımın HT/FT örüntü olasılıkları (opsiyonel)
            away_team_patterns: Deplasman takımının HT/FT örüntü olasılıkları (opsiyonel)
            
        Returns:
            dict: İyileştirilmiş İY/MS olasılıkları
        """
        # Hiçbir takım verisi yoksa temel olasılıkları döndür
        if not home_team_patterns and not away_team_patterns:
            return base_probabilities
            
        # Birleştirilmiş olasılıklar için kopya oluştur
        combined_probs = base_probabilities.copy()
        
        # Takımların ana eğilimlerini belirle (26.03.2025 iyileştirmesi)
        home_11_tendency = False
        home_1X_tendency = False
        away_22_tendency = False
        away_2X_tendency = False
        
        if home_team_patterns:
            home_11_tendency = home_team_patterns.get("1/1", 0) > 0.20  # %20'den fazla 1/1 örüntüsü
            home_1X_tendency = home_team_patterns.get("1/X", 0) > 0.15  # %15'den fazla 1/X örüntüsü
            if home_11_tendency:
                logger.info(f"Ev sahibi takımda güçlü 1/1 eğilimi tespit edildi: {home_team_patterns.get('1/1', 0):.2f}")
        
        if away_team_patterns:
            away_22_tendency = away_team_patterns.get("2/2", 0) > 0.20  # %20'den fazla 2/2 örüntüsü
            away_2X_tendency = away_team_patterns.get("2/X", 0) > 0.15  # %15'den fazla 2/X örüntüsü
            if away_22_tendency:
                logger.info(f"Deplasman takımında güçlü 2/2 eğilimi tespit edildi: {away_team_patterns.get('2/2', 0):.2f}")
        
        # Her HT/FT kombinasyonu için birleştirme 
        for htft in HT_FT_COMBINATIONS:
            # Başlangıç ağırlığı
            base_weight = 0.7  # Temel tahminlerin ağırlığı (70%)
            
            # Takım bazlı örüntüleri kullan
            home_weight = 0.0
            away_weight = 0.0
            
            # Güçlü eğilimlere göre ek faktör uygula
            htft_factor = 1.0
            if htft == "1/1" and home_11_tendency:
                htft_factor = 1.3  # Ev sahibi 1/1 eğilimi varsa güçlendir
            elif htft == "1/X" and home_1X_tendency:
                htft_factor = 1.2  # Ev sahibi 1/X eğilimi varsa güçlendir
            elif htft == "2/2" and away_22_tendency:
                htft_factor = 1.3  # Deplasman 2/2 eğilimi varsa güçlendir
            elif htft == "2/X" and away_2X_tendency:
                htft_factor = 1.2  # Deplasman 2/X eğilimi varsa güçlendir
            
            if home_team_patterns:
                home_weight = 0.15  # Ev sahibi örüntülerinin ağırlığı (15%)
                home_prob = home_team_patterns.get(htft, 0) * 100 * htft_factor
                combined_probs[htft] = combined_probs[htft] * base_weight + home_prob * home_weight
                
            if away_team_patterns:
                away_weight = 0.15  # Deplasman örüntülerinin ağırlığı (15%)
                away_prob = away_team_patterns.get(htft, 0) * 100 * htft_factor
                combined_probs[htft] = combined_probs[htft] * (base_weight / (base_weight + home_weight)) + away_prob * away_weight
        
        # Markov modeli kullanarak tutarlılık kontrolü uygula
        # Markov geçiş matrisi, ilk yarı/maç sonu arasındaki ilişkileri gösterir
        for htft in HT_FT_COMBINATIONS:
            first_half, full_time = htft.split('/')
            transition_prob = self.transition_matrix.get(first_half, {}).get(full_time, 0.33)
            
            # Geçiş olasılığı çok düşükse (tutarsız durum), olasılığı azalt
            if transition_prob < 0.1:
                combined_probs[htft] = combined_probs[htft] * 0.6
                logger.info(f"Markov modeli tutarsızlık tespit etti: {htft} (geçiş olasılığı: {transition_prob:.2f})")
            
            # Geçiş olasılığı çok yüksekse (tutarlı durum), olasılığı artır
            elif transition_prob > 0.6:
                combined_probs[htft] = combined_probs[htft] * 1.2
                logger.info(f"Markov modeli yüksek tutarlılık tespit etti: {htft} (geçiş olasılığı: {transition_prob:.2f})")
        
        # Tutarlılık kontrolü - mantık dışı sonuçların olasılıklarını kısıtla
        # Örnek: 2/1 olasılığı çok yüksekse sınırla
        if combined_probs["2/1"] > combined_probs["2/X"] and combined_probs["2/1"] > 15:
            combined_probs["2/1"] = combined_probs["2/1"] * 0.7
            combined_probs["2/X"] = combined_probs["2/X"] * 1.2
            
        if combined_probs["1/2"] > combined_probs["1/X"] and combined_probs["1/2"] > 15:
            combined_probs["1/2"] = combined_probs["1/2"] * 0.7
            combined_probs["1/X"] = combined_probs["1/X"] * 1.2
        
        # Normalize et (toplamları 100 olmalı)
        total = sum(combined_probs.values())
        combined_probs = {k: round(v * 100 / total) for k, v in combined_probs.items()}
        
        # Son bir kez toplamı kontrol et
        total = sum(combined_probs.values())
        if total != 100:
            # Farkı en yüksek olasılıklı kombinasyona ekle
            max_htft = max(combined_probs, key=combined_probs.get)
            combined_probs[max_htft] += (100 - total)
        
        return combined_probs
        
    def _simulate_probabilities_from_features(self, features):
        """
        Özellik dizisinden olasılıkları simüle et
        
        Args:
            features: Özellik dizisi
            
        Returns:
            dict: İY/MS olasılıkları
        """
        # Önceki simülasyon yaklaşımına benzer, ancak dizideki özellikleri kullan
        first_half_features = features[0]
        full_time_features = features[1]
        
        # Düşük skorlu maç mı?
        low_scoring = "low_scoring_first_half" in first_half_features or "medium_low_scoring_first_half" in first_half_features
        
        # Dengeli maç mı?
        balanced = "balanced_match_first_half" in first_half_features
        
        # Form farkı var mı?
        home_better_form = "home_better_form_first_half" in first_half_features
        away_better_form = "away_better_form_first_half" in first_half_features
        
        # İlk yarı olasılıkları
        if low_scoring and balanced:
            # Düşük skorlu ve dengeli maç = Yüksek beraberlik olasılığı
            first_half_probs = {"1": 0.20, "X": 0.65, "2": 0.15}
        elif low_scoring and home_better_form:
            # Düşük skorlu ve ev sahibi formda = Ev sahibi avantajlı
            first_half_probs = {"1": 0.45, "X": 0.45, "2": 0.10}
        elif low_scoring and away_better_form:
            # Düşük skorlu ve deplasman formda = Deplasman avantajlı
            first_half_probs = {"1": 0.10, "X": 0.50, "2": 0.40}
        elif balanced:
            # Dengeli maç
            first_half_probs = {"1": 0.30, "X": 0.45, "2": 0.25}
        elif home_better_form:
            # Ev sahibi formda
            first_half_probs = {"1": 0.50, "X": 0.35, "2": 0.15}
        elif away_better_form:
            # Deplasman formda
            first_half_probs = {"1": 0.15, "X": 0.40, "2": 0.45}
        else:
            # Varsayılan
            first_half_probs = {"1": 0.35, "X": 0.40, "2": 0.25}
        
        # Şartlı ikinci yarı olasılıkları
        second_half_probs = {
            "1": {"1": 0.65, "X": 0.25, "2": 0.10},  # İlk yarı 1
            "X": {"1": 0.40, "X": 0.30, "2": 0.30},  # İlk yarı X
            "2": {"1": 0.15, "X": 0.20, "2": 0.65}   # İlk yarı 2
        }
        
        # Gelişmiş faktörlere göre ikinci yarı olasılıklarını düzelt
        # Örneğin, geri dönüş yapan takımlar
        if "home_comeback_full_time" in full_time_features:
            # Ev sahibi geri dönüş yapabilir
            second_half_probs["2"]["1"] += 0.15  # 2/1 olasılığını artır
            second_half_probs["2"]["2"] -= 0.15  # 2/2 olasılığını azalt
            
        if "away_comeback_full_time" in full_time_features:
            # Deplasman geri dönüş yapabilir
            second_half_probs["1"]["2"] += 0.15  # 1/2 olasılığını artır
            second_half_probs["1"]["1"] -= 0.15  # 1/1 olasılığını azalt
            
        # İkinci yarı güç farkları
        if "significant_form_diff_full_time" in full_time_features:
            if "home_better_form_full_time" in full_time_features:
                # Ev sahibi ikinci yarıda daha güçlü
                for fh in ["1", "X", "2"]:
                    second_half_probs[fh]["1"] += 0.10
                    second_half_probs[fh]["2"] -= 0.10
            elif "away_better_form_full_time" in full_time_features:
                # Deplasman ikinci yarıda daha güçlü
                for fh in ["1", "X", "2"]:
                    second_half_probs[fh]["2"] += 0.10
                    second_half_probs[fh]["1"] -= 0.10
        
        # Değerleri normalize et
        for fh in ["1", "X", "2"]:
            total = sum(second_half_probs[fh].values())
            second_half_probs[fh] = {k: v/total for k, v in second_half_probs[fh].items()}
            
        total = sum(first_half_probs.values())
        first_half_probs = {k: v/total for k, v in first_half_probs.items()}
        
        # İY/MS olasılıkları
        htft_probs = {}
        for first_half in ["1", "X", "2"]:
            for full_time in ["1", "X", "2"]:
                htft = f"{first_half}/{full_time}"
                htft_probs[htft] = first_half_probs[first_half] * second_half_probs[first_half][full_time]
                
        # Yüzdeye çevir
        htft_probs = {k: round(v * 100) for k, v in htft_probs.items()}
        
        # Toplamı 100 olacak şekilde düzelt
        total = sum(htft_probs.values())
        if total != 100:
            scale = 100 / total
            for htft in htft_probs:
                htft_probs[htft] = round(htft_probs[htft] * scale)
                
        return htft_probs
    
    def _fallback_predict(self, home_stats, away_stats, expected_goals=None, team_adjustments=None):
        """
        CRF modeli kullanılamadığında basit bir simülasyon ile tahmin yap
        
        Args:
            home_stats: Ev sahibi takım istatistikleri
            away_stats: Deplasman takımı istatistikleri
            expected_goals: Beklenen gol sayıları (opsiyonel)
            team_adjustments: Takım-spesifik ayarlamalar (opsiyonel)
            
        Returns:
            dict: İY/MS olasılıkları
        """
        logger.info("CRF modeli kullanılamıyor, alternatif tahmin mekanizması kullanılıyor.")
        
        # Varsayılan olasılıklar - ilk yarı ve maç sonuçları için temel olasılıklar
        probabilities = {}
        
        # İstatistikleri çıkart
        home_first = home_stats.get("first_half", {})
        home_second = home_stats.get("second_half", {})
        away_first = away_stats.get("first_half", {})
        away_second = away_stats.get("second_half", {})
        
        # Form ve güç faktörleri
        home_form_score = 0.5
        away_form_score = 0.5
        
        if "form" in home_stats:
            home_form = home_stats["form"]
            if isinstance(home_form, dict):
                home_form_score = home_form.get("form_score", 0.5)
        
        if "form" in away_stats:
            away_form = away_stats["form"]
            if isinstance(away_form, dict):
                away_form_score = away_form.get("form_score", 0.5)
        
        # Takımların güç değerleri
        home_first_power = home_first.get("avg_goals_per_match", 0.5)
        away_first_power = away_first.get("avg_goals_per_match", 0.4)
        home_second_power = home_second.get("avg_goals_per_match", 0.6)
        away_second_power = away_second.get("avg_goals_per_match", 0.5)
        
        # Güç farklarını hesapla
        first_half_power_diff = home_first_power - away_first_power
        second_half_power_diff = home_second_power - away_second_power
        
        # İlk yarı olasılıklarını hesapla
        first_half_probs = {
            "1": 0.40,  # Ev sahibi önde
            "X": 0.45,  # Berabere
            "2": 0.15   # Deplasman önde
        }
        
        # Form ve güç farklarına göre ilk yarı olasılıklarını ayarla
        if first_half_power_diff > 0.25:  # Ev sahibi daha güçlü
            first_half_probs["1"] += 0.15
            first_half_probs["2"] -= 0.10
            first_half_probs["X"] -= 0.05
        elif first_half_power_diff < -0.25:  # Deplasman daha güçlü
            first_half_probs["2"] += 0.15
            first_half_probs["1"] -= 0.10
            first_half_probs["X"] -= 0.05
            
        # Form faktörlerini kullan
        form_diff = home_form_score - away_form_score
        if form_diff > 0.25:  # Ev sahibi formda
            first_half_probs["1"] += 0.1
            first_half_probs["2"] -= 0.1
        elif form_diff < -0.25:  # Deplasman formda
            first_half_probs["2"] += 0.1
            first_half_probs["1"] -= 0.1
            
        # 0-1 arasında sınırla
        first_half_probs = {k: max(0, min(1, v)) for k, v in first_half_probs.items()}
        
        # Toplamı 1 olacak şekilde normalleştir
        total = sum(first_half_probs.values())
        first_half_probs = {k: v / total for k, v in first_half_probs.items()}
            
        # Maç sonu olasılıklarını hesapla
        total_power_diff = (home_first_power + home_second_power) - (away_first_power + away_second_power)
        
        full_time_probs = {
            "1": 0.45,  # Ev sahibi kazanır
            "X": 0.30,  # Beraberlik
            "2": 0.25   # Deplasman kazanır
        }
        
        # Güç farkına göre ayarla
        if total_power_diff > 0.3:  # Ev sahibi daha güçlü
            full_time_probs["1"] += 0.15
            full_time_probs["2"] -= 0.1
            full_time_probs["X"] -= 0.05
        elif total_power_diff < -0.3:  # Deplasman daha güçlü
            full_time_probs["2"] += 0.15
            full_time_probs["1"] -= 0.1
            full_time_probs["X"] -= 0.05
            
        # Form faktörlerini kullan
        if form_diff > 0.3:  # Ev sahibi formda
            full_time_probs["1"] += 0.1
            full_time_probs["2"] -= 0.1
        elif form_diff < -0.3:  # Deplasman formda
            full_time_probs["2"] += 0.1
            full_time_probs["1"] -= 0.1
            
        # 0-1 arasında sınırla
        full_time_probs = {k: max(0, min(1, v)) for k, v in full_time_probs.items()}
        
        # Toplamı 1 olacak şekilde normalleştir
        total = sum(full_time_probs.values())
        full_time_probs = {k: v / total for k, v in full_time_probs.items()}
        
        # Tüm İY/MS kombinasyonlarını hesapla
        base_transition_matrix = {
            # İlk yarı -> Maç sonu geçiş olasılıkları
            # [İlk yarı durum][Maç sonu durum] = geçiş olasılığı
            "1": {"1": 0.65, "X": 0.25, "2": 0.10},  # Ev sahibi önde -> Maç sonu
            "X": {"1": 0.35, "X": 0.40, "2": 0.25},  # Berabere -> Maç sonu
            "2": {"1": 0.15, "X": 0.25, "2": 0.60}   # Deplasman önde -> Maç sonu
        }
        
        # Düşük skorlu maçlarda geçiş matrisi farklı olmalıdır - daha stabil sonuçlar
        # Toplam beklenen gol değeri
        total_expected_goals = 0
        if expected_goals and "total" in expected_goals:
            total_expected_goals = expected_goals["total"]
        else:
            total_expected_goals = home_first_power + home_second_power + away_first_power + away_second_power
        
        # Düşük skorlu maçlar için geçiş matrisini ayarla
        transition_matrix = base_transition_matrix.copy()
        
        # Düşük skorlu maç ayarlamaları
        if total_expected_goals < 2.0:
            # Düşük skorlu maçlarda beraberliğin devam etme olasılığı yüksektir
            transition_matrix["X"]["X"] = 0.55  # X/X olasılığını artır (0.40 -> 0.55)
            transition_matrix["X"]["1"] = 0.25  # X/1 olasılığını azalt (0.35 -> 0.25)
            transition_matrix["X"]["2"] = 0.20  # X/2 olasılığını azalt (0.25 -> 0.20)
            
            # İlk yarı sonuçlarının korunması da daha yüksek olasılıklıdır
            transition_matrix["1"]["1"] = 0.75  # 1/1 olasılığını artır (0.65 -> 0.75)
            transition_matrix["1"]["X"] = 0.20  # 1/X olasılığını azalt (0.25 -> 0.20)
            transition_matrix["1"]["2"] = 0.05  # 1/2 olasılığını azalt (0.10 -> 0.05)
            
            transition_matrix["2"]["2"] = 0.75  # 2/2 olasılığını artır (0.60 -> 0.75)
            transition_matrix["2"]["X"] = 0.20  # 2/X olasılığını azalt (0.25 -> 0.20)
            transition_matrix["2"]["1"] = 0.05  # 2/1 olasılığını azalt (0.15 -> 0.05)
            
            logger.info(f"Düşük skorlu maç (beklenen gol: {total_expected_goals:.2f}), geçiş matrisi ayarlandı.")
        
        # Çok düşük skorlu maç ayarlamaları
        elif total_expected_goals < 1.5:
            # Çok düşük skorlu maçlarda beraberliğin devam etme olasılığı daha da yüksektir
            transition_matrix["X"]["X"] = 0.70  # X/X olasılığını daha fazla artır (0.40 -> 0.70)
            transition_matrix["X"]["1"] = 0.15  # X/1 olasılığını daha fazla azalt (0.35 -> 0.15)
            transition_matrix["X"]["2"] = 0.15  # X/2 olasılığını daha fazla azalt (0.25 -> 0.15)
            
            # İlk yarı sonuçlarının korunması da daha yüksek olasılıklıdır
            transition_matrix["1"]["1"] = 0.80  # 1/1 olasılığını daha fazla artır (0.65 -> 0.80)
            transition_matrix["1"]["X"] = 0.15  # 1/X olasılığını daha fazla azalt (0.25 -> 0.15)
            transition_matrix["1"]["2"] = 0.05  # 1/2 olasılığını azalt (0.10 -> 0.05)
            
            transition_matrix["2"]["2"] = 0.80  # 2/2 olasılığını daha fazla artır (0.60 -> 0.80)
            transition_matrix["2"]["X"] = 0.15  # 2/X olasılığını daha fazla azalt (0.25 -> 0.15)
            transition_matrix["2"]["1"] = 0.05  # 2/1 olasılığını azalt (0.15 -> 0.05)
            
            logger.info(f"Çok düşük skorlu maç (beklenen gol: {total_expected_goals:.2f}), geçiş matrisi büyük ölçüde ayarlandı.")
        
        # Yüksek skorlu maç ayarlamaları
        elif total_expected_goals > 3.0:
            # Yüksek skorlu maçlarda sonuçların değişme olasılığı daha yüksektir
            transition_matrix["X"]["X"] = 0.25  # X/X olasılığını azalt (0.40 -> 0.25)
            transition_matrix["X"]["1"] = 0.40  # X/1 olasılığını artır (0.35 -> 0.40)
            transition_matrix["X"]["2"] = 0.35  # X/2 olasılığını artır (0.25 -> 0.35)
            
            # Tersine dönüşlerin olasılığı da artabilir
            transition_matrix["1"]["2"] = 0.15  # 1/2 olasılığını artır (0.10 -> 0.15)
            transition_matrix["2"]["1"] = 0.20  # 2/1 olasılığını artır (0.15 -> 0.20)
            
            logger.info(f"Yüksek skorlu maç (beklenen gol: {total_expected_goals:.2f}), geçiş matrisi değişken olacak şekilde ayarlandı.")
        
        # İY/MS olasılıklarını hesapla
        for first_half in ["1", "X", "2"]:
            for full_time in ["1", "X", "2"]:
                htft = f"{first_half}/{full_time}"
                
                # İki durumun olasılığı ve geçiş olasılığı ile çarpımı
                p_first = first_half_probs.get(first_half, 0)
                p_transition = transition_matrix[first_half][full_time]
                
                probabilities[htft] = round(p_first * p_transition * 100)
        
        # Olasılıkların toplamının 100 olduğundan emin ol
        total_prob = sum(probabilities.values())
        if total_prob != 100:
            # Oransal olarak ayarla
            scale_factor = 100 / total_prob if total_prob > 0 else 1
            for htft in probabilities:
                probabilities[htft] = round(probabilities[htft] * scale_factor)
                
            # Hala tam 100 değilse, en yüksek olasılıklı sonuca kalan farkı ekle
            new_total = sum(probabilities.values())
            if new_total != 100:
                diff = 100 - new_total
                max_key = max(probabilities, key=probabilities.get)
                probabilities[max_key] += diff
                
        return probabilities
