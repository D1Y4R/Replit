"""
Kendi Kendine Öğrenen Tahmin Modeli

Bu modül, tahmin sonuçlarını ve gerçek maç sonuçlarını değerlendirerek
kendi parametrelerini ve ağırlıklarını otomatik olarak güncelleyen bir model içerir.

Özellikler:
- Tahmin ve gerçek sonuçları karşılaştırır
- Model ağırlıklarını otomatik olarak günceller
- Zamana dayalı performans analizi yapar
- Öğrenme sürecini görselleştirir
- Performansı sürekli izler ve raporlar
"""

import os
import json
import time
import logging
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Yerel modülleri içe aktar
from dynamic_team_analyzer import DynamicTeamAnalyzer

# Logging ayarları
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SelfLearningPredictor:
    """Kendi parametrelerini ve ağırlıklarını otomatik güncelleyen tahmin modeli"""
    
    def __init__(self, 
                 analyzer=None,
                 model_path="self_learning_model.json",
                 learning_rate=0.05,
                 min_validation_matches=10,
                 cache_file="predictions_cache.json"):
        """
        Kendi kendine öğrenen tahmin modelini başlat
        
        Args:
            analyzer: DynamicTeamAnalyzer örneği
            model_path: Model dosyası yolu
            learning_rate: Öğrenme hızı
            min_validation_matches: Minimum doğrulama maç sayısı
            cache_file: Tahmin önbellek dosyası
        """
        self.analyzer = analyzer or DynamicTeamAnalyzer()
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.min_validation_matches = min_validation_matches
        self.cache_file = cache_file
        
        # Model parametreleri
        self.parameters = {
            # Tahmin bileşen ağırlıkları
            'component_weights': {
                'poisson': 0.3,        # Poisson model
                'h2h': 0.3,            # Head-to-head analizi
                'form': 0.25,          # Takım formu
                'home_advantage': 0.15  # Ev sahibi avantajı
            },
            
            # Kategori bazlı çarpanlar
            'low_scoring_draw_boost': 1.25,    # Düşük skorlu maçlarda beraberlik artırma
            'medium_scoring_draw_boost': 1.10, # Orta skorlu maçlarda beraberlik artırma
            'high_scoring_draw_boost': 0.95,   # Yüksek skorlu maçlarda beraberlik artırma
            
            # Gol eşik değerleri
            'low_scoring_threshold': 2.0,      # Düşük skorlu maç eşiği
            'high_scoring_threshold': 3.0,     # Yüksek skorlu maç eşiği
            
            # Skor olasılık çarpanları
            'exact_score_factors': {
                '0-0': 1.1,  # 0-0 skor olasılığı çarpanı
                '1-1': 1.1,  # 1-1 skor olasılığı çarpanı
                '1-0': 1.05, # 1-0 skor olasılığı çarpanı
                '0-1': 1.05, # 0-1 skor olasılığı çarpanı
                '2-1': 1.0,  # 2-1 skor olasılığı çarpanı
                '1-2': 1.0,  # 1-2 skor olasılığı çarpanı
                '2-0': 1.0,  # 2-0 skor olasılığı çarpanı
                '0-2': 1.0,  # 0-2 skor olasılığı çarpanı
                '2-2': 0.95, # 2-2 skor olasılığı çarpanı
                '3-0': 0.9,  # 3-0 skor olasılığı çarpanı
                '0-3': 0.9,  # 0-3 skor olasılığı çarpanı
            },
            
            # Model hiper-parametreleri
            'zero_inflation_factor': 1.2,       # 0-0 skorları için enflasyon faktörü
            'one_inflation_factor': 1.1,        # 1-1 skorları için enflasyon faktörü
            'home_advantage_base': 1.2,         # Temel ev sahibi avantajı
            'form_decay_rate': 0.1,             # Form bozulma hızı
            'h2h_recency_weight': 0.6,          # H2H analizinde yakın tarihli maçların ağırlığı
            
            # Lig kategorileri için varsayılan değerler
            'league_type_factors': {
                'high_scoring': 1.2,   # Yüksek skorlu ligler (Hollanda, Almanya, vb)
                'low_scoring': 0.8,    # Düşük skorlu ligler (İtalya, Fransa, vb)
                'draw_prone': 1.3,     # Beraberliğe yatkın ligler
                'home_advantage': 1.2  # Ev sahibi avantajı yüksek ligler
            },
            
            # Son güncelleme zamanı
            'last_updated': None,
            'validation_metrics': {
                'outcome_accuracy': 0.0,
                'over_under_accuracy': 0.0,
                'exact_score_accuracy': 0.0,
                'log_loss': 0.0,
                'brier_score': 0.0
            },
            'learning_history': []
        }
        
        # Kaydedilmiş modeli yükle
        self._load_model()
    
    def _load_model(self):
        """Kaydedilmiş modeli yükle"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'r', encoding='utf-8') as f:
                    saved_params = json.load(f)
                
                # Mevcut parametreleri güncelle
                for key, value in saved_params.items():
                    if key in self.parameters:
                        self.parameters[key] = value
                
                logger.info(f"Model başarıyla yüklendi: {self.model_path}")
            else:
                logger.info(f"Model dosyası bulunamadı, varsayılan parametreler kullanılacak: {self.model_path}")
        
        except Exception as e:
            logger.error(f"Model yüklenirken hata: {str(e)}")
    
    def save_model(self):
        """Modeli kaydet"""
        try:
            # Şu anki zamanı son güncelleme olarak işaretle
            self.parameters['last_updated'] = datetime.now().isoformat()
            
            with open(self.model_path, 'w', encoding='utf-8') as f:
                json.dump(self.parameters, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Model başarıyla kaydedildi: {self.model_path}")
            return True
        
        except Exception as e:
            logger.error(f"Model kaydedilirken hata: {str(e)}")
            return False
    
    def get_current_parameters(self):
        """Mevcut model parametrelerini döndür"""
        return self.parameters
    
    def analyze_predictions_and_results(self):
        """
        Tahmin önbelleğini analiz ederek tahminler ve gerçek sonuçları karşılaştır
        
        Returns:
            dict: Karşılaştırma sonuçları
        """
        try:
            # Tahmin önbelleğini yükle
            if not os.path.exists(self.cache_file):
                logger.warning(f"Tahmin önbellek dosyası bulunamadı: {self.cache_file}")
                return {}
            
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Analiz için veri hazırla
            analyzed_matches = 0
            correct_outcomes = 0
            correct_over_under = 0
            correct_exact_scores = 0
            
            outcome_probabilities = []
            actual_outcomes = []
            
            # Her tahmin için sonuçları analiz et
            for match_key, prediction in cache_data.items():
                # Tahmin yapısı doğru ve gerçek sonuç kaydedilmiş mi kontrol et
                if (not isinstance(prediction, dict) or 
                    'predictions' not in prediction or 
                    'actual_result' not in prediction):
                    continue
                
                # Gerçek sonucu al
                actual_result = prediction.get('actual_result', {})
                if not actual_result or 'outcome' not in actual_result:
                    continue
                
                # Tahminleri al
                predictions = prediction.get('predictions', {})
                if not predictions:
                    continue
                
                # Gerçek sonuç
                actual_outcome = actual_result.get('outcome')
                actual_home_goals = actual_result.get('home_goals', 0)
                actual_away_goals = actual_result.get('away_goals', 0)
                actual_total_goals = actual_home_goals + actual_away_goals
                actual_exact_score = f"{actual_home_goals}-{actual_away_goals}"
                
                # Tahminleri al
                predicted_outcome = predictions.get('most_likely_outcome')
                
                # Olasılıkları al
                home_win_prob = predictions.get('home_win_probability', 0) / 100
                draw_prob = predictions.get('draw_probability', 0) / 100
                away_win_prob = predictions.get('away_win_probability', 0) / 100
                
                # Doğruluk kontrolü
                if predicted_outcome and actual_outcome:
                    analyzed_matches += 1
                    
                    # Sonuç tahmini doğru mu?
                    if predicted_outcome == actual_outcome:
                        correct_outcomes += 1
                    
                    # Toplam gol tahmini
                    expected_goals = predictions.get('expected_goals', {})
                    expected_home = expected_goals.get('home', 0)
                    expected_away = expected_goals.get('away', 0)
                    expected_total = expected_home + expected_away
                    
                    # 2.5 üst/alt doğru mu?
                    predicted_over_under = expected_total > 2.5
                    actual_over_under = actual_total_goals > 2.5
                    if predicted_over_under == actual_over_under:
                        correct_over_under += 1
                    
                    # Kesin skor doğru mu?
                    betting_predictions = predictions.get('betting_predictions', {})
                    exact_score_prediction = betting_predictions.get('exact_score', {}).get('prediction', '')
                    if exact_score_prediction == actual_exact_score:
                        correct_exact_scores += 1
                    
                    # Log loss ve brier score için olasılıkları kaydet
                    if actual_outcome == "HOME_WIN":
                        outcome_probabilities.append(home_win_prob)
                        actual_outcomes.append(1)
                    elif actual_outcome == "DRAW":
                        outcome_probabilities.append(draw_prob)
                        actual_outcomes.append(1)
                    elif actual_outcome == "AWAY_WIN":
                        outcome_probabilities.append(away_win_prob)
                        actual_outcomes.append(1)
            
            # Yeterli maç yoksa, sonuçları döndürme
            if analyzed_matches < self.min_validation_matches:
                logger.warning(f"Yeterli doğrulama maçı yok: {analyzed_matches} < {self.min_validation_matches}")
                return {
                    'analyzed_matches': analyzed_matches,
                    'sufficient_data': False
                }
            
            # Metrikleri hesapla
            outcome_accuracy = correct_outcomes / analyzed_matches if analyzed_matches > 0 else 0
            over_under_accuracy = correct_over_under / analyzed_matches if analyzed_matches > 0 else 0
            exact_score_accuracy = correct_exact_scores / analyzed_matches if analyzed_matches > 0 else 0
            
            # Log loss ve brier score hesapla
            outcome_probabilities = np.array(outcome_probabilities)
            actual_outcomes = np.array(actual_outcomes)
            
            # Değerler güvenli aralıkta olmalı (0 ve 1 olmamalı)
            outcome_probabilities = np.clip(outcome_probabilities, 0.001, 0.999)
            
            # Log loss hesapla
            log_loss = -np.mean(actual_outcomes * np.log(outcome_probabilities) + 
                                (1 - actual_outcomes) * np.log(1 - outcome_probabilities))
            
            # Brier score hesapla
            brier_score = np.mean((outcome_probabilities - actual_outcomes) ** 2)
            
            # Sonuçları döndür
            results = {
                'analyzed_matches': analyzed_matches,
                'correct_outcomes': correct_outcomes,
                'correct_over_under': correct_over_under,
                'correct_exact_scores': correct_exact_scores,
                'outcome_accuracy': outcome_accuracy,
                'over_under_accuracy': over_under_accuracy,
                'exact_score_accuracy': exact_score_accuracy,
                'log_loss': log_loss,
                'brier_score': brier_score,
                'sufficient_data': True
            }
            
            # Önceki sonuçlarla karşılaştır
            prev_metrics = self.parameters.get('validation_metrics', {})
            prev_outcome_acc = prev_metrics.get('outcome_accuracy', 0)
            prev_log_loss = prev_metrics.get('log_loss', float('inf'))
            
            results['outcome_change'] = outcome_accuracy - prev_outcome_acc
            results['log_loss_change'] = prev_log_loss - log_loss  # Düşüş iyidir
            
            # Doğrulama metriklerini güncelle ve kaydet
            self.parameters['validation_metrics'] = {
                'outcome_accuracy': outcome_accuracy,
                'over_under_accuracy': over_under_accuracy,
                'exact_score_accuracy': exact_score_accuracy,
                'log_loss': log_loss,
                'brier_score': brier_score,
                'timestamp': datetime.now().isoformat()
            }
            
            # Öğrenme geçmişini güncelle
            learning_record = {
                'timestamp': datetime.now().isoformat(),
                'metrics': self.parameters['validation_metrics'],
                'analyzed_matches': analyzed_matches
            }
            self.parameters['learning_history'].append(learning_record)
            
            # Son 10 kaydı tut
            self.parameters['learning_history'] = self.parameters['learning_history'][-10:]
            
            # Yeni parametreleri kaydet
            self.save_model()
            
            return results
            
        except Exception as e:
            logger.error(f"Tahminler ve sonuçlar analiz edilirken hata: {str(e)}")
            return {'error': str(e)}
    
    def update_model_parameters(self):
        """
        Analiz sonuçlarını kullanarak model parametrelerini güncelle
        
        Returns:
            dict: Güncelleme sonuçları
        """
        try:
            # Önce analiz yap
            analysis = self.analyze_predictions_and_results()
            
            if not analysis.get('sufficient_data', False):
                logger.warning("Yeterli doğrulama verisi olmadığından model parametreleri güncellenemiyor.")
                return {'success': False, 'reason': 'insufficient_data'}
            
            # Sonuçları al
            outcome_accuracy = analysis.get('outcome_accuracy', 0)
            over_under_accuracy = analysis.get('over_under_accuracy', 0)
            exact_score_accuracy = analysis.get('exact_score_accuracy', 0)
            log_loss = analysis.get('log_loss', float('inf'))
            brier_score = analysis.get('brier_score', 1.0)
            
            # Değişim oranları
            outcome_change = analysis.get('outcome_change', 0)
            log_loss_change = analysis.get('log_loss_change', 0)
            
            # Performans yeterince iyi mi?
            min_outcome_acc = 0.45  # Minimum başarı oranı
            max_log_loss = 0.7      # Maksimum log loss
            
            if outcome_accuracy < min_outcome_acc or log_loss > max_log_loss:
                logger.info(f"Performans yeterince iyi değil, parametreler güncelleniyor.")
                
                # Parametreleri güncelle
                updates = {}
                
                # Bileşen ağırlıklarını güncelle
                component_weights = self.parameters['component_weights']
                
                # Sonuç doğruluğu düşükse form ağırlığını azalt, H2H ağırlığını arttır
                if outcome_accuracy < 0.45:
                    updates['component_weights.form'] = max(0.15, component_weights['form'] - self.learning_rate * 0.5)
                    updates['component_weights.h2h'] = min(0.4, component_weights['h2h'] + self.learning_rate * 0.5)
                
                # Düşük skorlu maçlarda beraberlik artırıcı faktörü güncelle
                if exact_score_accuracy < 0.1:
                    if '0-0' in analysis.get('missed_exact_scores', []) or '1-1' in analysis.get('missed_exact_scores', []):
                        updates['low_scoring_draw_boost'] = min(1.5, self.parameters['low_scoring_draw_boost'] + self.learning_rate)
                        updates['exact_score_factors.0-0'] = min(1.3, self.parameters['exact_score_factors']['0-0'] + self.learning_rate)
                        updates['exact_score_factors.1-1'] = min(1.3, self.parameters['exact_score_factors']['1-1'] + self.learning_rate)
                
                # Log loss yüksekse sıfır enflasyon faktörünü güncelle
                if log_loss > 0.65:
                    updates['zero_inflation_factor'] = min(1.4, self.parameters['zero_inflation_factor'] + self.learning_rate * 0.5)
                
                # Parametreleri güncelle
                for param_name, new_value in updates.items():
                    if '.' in param_name:
                        # İç içe parametre
                        parts = param_name.split('.')
                        if len(parts) == 2 and parts[0] in self.parameters and parts[1] in self.parameters[parts[0]]:
                            old_value = self.parameters[parts[0]][parts[1]]
                            self.parameters[parts[0]][parts[1]] = new_value
                            logger.info(f"Parametre güncellendi: {param_name} {old_value:.3f} -> {new_value:.3f}")
                    elif param_name in self.parameters:
                        # Birinci seviye parametre
                        old_value = self.parameters[param_name]
                        self.parameters[param_name] = new_value
                        logger.info(f"Parametre güncellendi: {param_name} {old_value:.3f} -> {new_value:.3f}")
                
                # Değişiklik olmamışsa
                if not updates:
                    logger.info("Güncelleme kriterleri karşılanmadı, parametreler değiştirilmedi.")
                    return {'success': True, 'updated': False, 'message': 'No updates needed'}
                
                # Parametreleri kaydet
                self.save_model()
                
                return {
                    'success': True,
                    'updated': True,
                    'updates': updates,
                    'metrics': {
                        'outcome_accuracy': outcome_accuracy,
                        'over_under_accuracy': over_under_accuracy,
                        'exact_score_accuracy': exact_score_accuracy,
                        'log_loss': log_loss,
                        'brier_score': brier_score
                    }
                }
            else:
                logger.info(f"Performans yeterince iyi, parametre güncellemesi gerekmiyor.")
                return {'success': True, 'updated': False, 'message': 'Performance is good enough'}
            
        except Exception as e:
            logger.error(f"Model parametreleri güncellenirken hata: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def train_model_from_history(self, max_iterations=10):
        """
        Geçmiş tahminleri kullanarak modeli eğit
        
        Args:
            max_iterations: Maksimum iterasyon sayısı
            
        Returns:
            dict: Eğitim sonuçları
        """
        try:
            logger.info(f"Model eğitimi başlatılıyor... (maksimum {max_iterations} iterasyon)")
            
            training_history = []
            best_outcome_acc = 0
            best_log_loss = float('inf')
            best_iteration = -1
            
            for i in range(max_iterations):
                logger.info(f"Eğitim iterasyonu {i+1}/{max_iterations}")
                
                # Parametreleri güncelle
                update_result = self.update_model_parameters()
                
                if not update_result.get('success', False):
                    logger.warning(f"İterasyon {i+1}: Güncelleme başarısız: {update_result.get('reason', 'unknown')}")
                    continue
                
                # Sonuçları analiz et
                analysis = self.analyze_predictions_and_results()
                
                if not analysis.get('sufficient_data', False):
                    logger.warning(f"İterasyon {i+1}: Yeterli veri yok")
                    break
                
                # Performans metriklerini al
                outcome_acc = analysis.get('outcome_accuracy', 0)
                over_under_acc = analysis.get('over_under_accuracy', 0)
                exact_score_acc = analysis.get('exact_score_accuracy', 0)
                log_loss = analysis.get('log_loss', float('inf'))
                brier_score = analysis.get('brier_score', 1.0)
                
                # Eğitim geçmişine ekle
                training_history.append({
                    'iteration': i+1,
                    'outcome_accuracy': outcome_acc,
                    'over_under_accuracy': over_under_acc,
                    'exact_score_accuracy': exact_score_acc,
                    'log_loss': log_loss,
                    'brier_score': brier_score,
                    'updated': update_result.get('updated', False)
                })
                
                # En iyi iterasyonu takip et
                if outcome_acc > best_outcome_acc or (outcome_acc == best_outcome_acc and log_loss < best_log_loss):
                    best_outcome_acc = outcome_acc
                    best_log_loss = log_loss
                    best_iteration = i+1
                
                # Gelişme yoksa dur
                if i >= 2 and not update_result.get('updated', False):
                    logger.info(f"İterasyon {i+1}: Gelişme yok, eğitim durduruluyor.")
                    break
            
            logger.info(f"Model eğitimi tamamlandı. En iyi sonuç iterasyon {best_iteration}: "
                      f"Sonuç doğruluğu {best_outcome_acc:.4f}, Log loss {best_log_loss:.4f}")
            
            # Eğitim performansını görselleştir
            if MATPLOTLIB_AVAILABLE and len(training_history) > 1:
                self._visualize_training_history(training_history)
            
            return {
                'iterations': len(training_history),
                'best_iteration': best_iteration,
                'best_outcome_accuracy': best_outcome_acc,
                'best_log_loss': best_log_loss,
                'training_history': training_history
            }
            
        except Exception as e:
            logger.error(f"Model eğitimi sırasında hata: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _visualize_training_history(self, training_history):
        """Eğitim geçmişini görselleştir"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib kütüphanesi bulunamadı, görselleştirme atlanıyor.")
            return
        
        try:
            iterations = [record['iteration'] for record in training_history]
            outcome_acc = [record['outcome_accuracy'] for record in training_history]
            log_loss = [record['log_loss'] for record in training_history]
            
            plt.figure(figsize=(12, 6))
            
            # Doğruluk grafiği
            plt.subplot(1, 2, 1)
            plt.plot(iterations, outcome_acc, 'o-', label='Sonuç doğruluğu')
            plt.axhline(y=0.5, linestyle='--', color='r', label='%50 doğruluk')
            plt.title('Tahmin Doğruluğu')
            plt.xlabel('İterasyon')
            plt.ylabel('Doğruluk')
            plt.legend()
            plt.grid(True)
            
            # Log loss grafiği
            plt.subplot(1, 2, 2)
            plt.plot(iterations, log_loss, 'o-', label='Log loss')
            plt.title('Log Loss')
            plt.xlabel('İterasyon')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('model_training_history.png')
            logger.info("Eğitim geçmişi görselleştirildi: model_training_history.png")
            
        except Exception as e:
            logger.error(f"Eğitim geçmişi görselleştirilirken hata: {str(e)}")
    
    def apply_learned_parameters(self, match_prediction):
        """
        Öğrenilen parametreleri tahmin sonuçlarına uygula
        
        Args:
            match_prediction: Model tahmin sonuçları
            
        Returns:
            dict: Güncellenmiş tahmin sonuçları
        """
        if not match_prediction:
            return match_prediction
        
        try:
            # Elde edilen parametreleri kullan
            params = self.parameters
            
            # Tahmin yapısını değişmemesi için kopyala
            prediction = match_prediction.copy()
            pred_obj = prediction.get('predictions', {})
            
            if not pred_obj:
                return match_prediction
            
            # Beklenen goller
            expected_goals = pred_obj.get('expected_goals', {})
            home_goals = expected_goals.get('home', 0)
            away_goals = expected_goals.get('away', 0)
            total_goals = home_goals + away_goals
            
            # Maç kategorisini belirle
            if total_goals < params.get('low_scoring_threshold', 2.0):
                category = 'low_scoring'
                draw_boost = params.get('low_scoring_draw_boost', 1.25)
            elif total_goals < params.get('high_scoring_threshold', 3.0):
                category = 'medium_scoring'
                draw_boost = params.get('medium_scoring_draw_boost', 1.10)
            else:
                category = 'high_scoring'
                draw_boost = params.get('high_scoring_draw_boost', 0.95)
            
            # Olasılıkları al
            home_win_prob = pred_obj.get('home_win_probability', 0) / 100
            draw_prob = pred_obj.get('draw_probability', 0) / 100
            away_win_prob = pred_obj.get('away_win_probability', 0) / 100
            
            # Beraberlik olasılığını ayarla
            if category == 'low_scoring':
                # Düşük skorlu maçlarda beraberlik artır
                orig_draw_prob = draw_prob
                draw_prob = min(0.70, draw_prob * draw_boost)
                
                # Diğer olasılıkları dengele
                remaining = 1.0 - draw_prob
                if home_win_prob + away_win_prob > 0:
                    home_win_prob = remaining * (home_win_prob / (home_win_prob + away_win_prob))
                    away_win_prob = remaining * (away_win_prob / (home_win_prob + away_win_prob))
                else:
                    home_win_prob = remaining * 0.5
                    away_win_prob = remaining * 0.5
                
                logger.debug(f"Beraberlik olasılığı güncellendi: {orig_draw_prob:.3f} -> {draw_prob:.3f} (kategori: {category})")
            
            # Kesin skor olasılıklarını güncelle
            betting_predictions = pred_obj.get('betting_predictions', {})
            exact_score = betting_predictions.get('exact_score', {})
            exact_score_prob = exact_score.get('probability', 0) / 100
            exact_score_val = exact_score.get('prediction', '')
            
            # Kesin skor olasılık faktörü
            if exact_score_val in params.get('exact_score_factors', {}):
                orig_exact_score_prob = exact_score_prob
                factor = params['exact_score_factors'][exact_score_val]
                exact_score_prob = min(0.30, exact_score_prob * factor)
                
                logger.debug(f"Kesin skor olasılığı güncellendi: {orig_exact_score_prob:.3f} -> {exact_score_prob:.3f} (skor: {exact_score_val})")
            
            # Düşük skorlu maçlarda 0-0 skoru için özel artırma
            if category == 'low_scoring' and exact_score_val == '0-0':
                zero_inflation = params.get('zero_inflation_factor', 1.2)
                exact_score_prob = min(0.35, exact_score_prob * zero_inflation)
                logger.debug(f"Sıfır enflasyonu uygulandı: {exact_score_prob:.3f} (faktör: {zero_inflation})")
            
            # Güncellenmiş olasılıkları kaydet
            pred_obj['home_win_probability'] = round(home_win_prob * 100, 2)
            pred_obj['draw_probability'] = round(draw_prob * 100, 2)
            pred_obj['away_win_probability'] = round(away_win_prob * 100, 2)
            
            if 'betting_predictions' in pred_obj and 'exact_score' in pred_obj['betting_predictions']:
                pred_obj['betting_predictions']['exact_score']['probability'] = round(exact_score_prob * 100, 2)
            
            # En olası sonucu güncelle
            probs = {'HOME_WIN': home_win_prob, 'DRAW': draw_prob, 'AWAY_WIN': away_win_prob}
            most_likely = max(probs, key=probs.get)
            pred_obj['most_likely_outcome'] = most_likely
            
            # Güncellenmiş tahmini döndür
            prediction['predictions'] = pred_obj
            prediction['applied_learning'] = True
            prediction['learning_model_version'] = datetime.now().isoformat()
            
            return prediction
            
        except Exception as e:
            logger.error(f"Öğrenilen parametreler uygulanırken hata: {str(e)}")
            return match_prediction  # Hata durumunda orijinal tahmini döndür
    
    def get_long_term_performance(self):
        """
        Modelin uzun vadeli performans ölçümlerini al
        
        Returns:
            dict: Performans metrikleri
        """
        try:
            # Öğrenme geçmişini al
            learning_history = self.parameters.get('learning_history', [])
            
            if not learning_history:
                return {'history_available': False}
            
            # Metrik serilerini oluştur
            timestamps = []
            outcome_acc = []
            log_loss_vals = []
            
            for record in learning_history:
                timestamps.append(datetime.fromisoformat(record['timestamp']))
                metrics = record.get('metrics', {})
                outcome_acc.append(metrics.get('outcome_accuracy', 0))
                log_loss_vals.append(metrics.get('log_loss', 0))
            
            # Görselleştir (mevcutsa)
            if MATPLOTLIB_AVAILABLE and len(timestamps) > 1:
                plt.figure(figsize=(12, 6))
                
                # Doğruluk grafiği
                plt.subplot(1, 2, 1)
                plt.plot(timestamps, outcome_acc, 'o-', label='Sonuç doğruluğu')
                plt.axhline(y=0.5, linestyle='--', color='r', label='%50 doğruluk')
                plt.title('Uzun Vadeli Tahmin Doğruluğu')
                plt.xlabel('Tarih')
                plt.ylabel('Doğruluk')
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=45)
                
                # Log loss grafiği
                plt.subplot(1, 2, 2)
                plt.plot(timestamps, log_loss_vals, 'o-', label='Log loss')
                plt.title('Uzun Vadeli Log Loss')
                plt.xlabel('Tarih')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                plt.savefig('long_term_performance.png')
                logger.info("Uzun vadeli performans görselleştirildi: long_term_performance.png")
            
            # Performans analizini yap
            trend_analysis = {}
            
            if len(outcome_acc) >= 3:
                # İlk ve son değeri karşılaştır
                first_acc = outcome_acc[0]
                last_acc = outcome_acc[-1]
                acc_trend = last_acc - first_acc
                
                first_loss = log_loss_vals[0]
                last_loss = log_loss_vals[-1]
                loss_trend = last_loss - first_loss  # Negatif değer iyileşmeyi gösterir
                
                # Trend analizi
                trend_analysis = {
                    'accuracy_trend': acc_trend,
                    'loss_trend': loss_trend,
                    'improving': acc_trend > 0 and loss_trend < 0,
                    'stable': abs(acc_trend) < 0.02 and abs(loss_trend) < 0.05,
                    'degrading': acc_trend < 0 and loss_trend > 0
                }
            
            return {
                'history_available': True,
                'data_points': len(learning_history),
                'latest_metrics': {
                    'outcome_accuracy': outcome_acc[-1] if outcome_acc else 0,
                    'log_loss': log_loss_vals[-1] if log_loss_vals else 0
                },
                'trend_analysis': trend_analysis,
                'visualization_saved': MATPLOTLIB_AVAILABLE and len(timestamps) > 1
            }
            
        except Exception as e:
            logger.error(f"Uzun vadeli performans analizi sırasında hata: {str(e)}")
            return {'error': str(e)}

# Test ve örnek kullanım
if __name__ == "__main__":
    try:
        # Öğrenen tahmin modelini başlat
        predictor = SelfLearningPredictor()
        
        # Mevcut parametreleri göster
        params = predictor.get_current_parameters()
        print("Mevcut Model Parametreleri:")
        print(f"Bileşen ağırlıkları: {params['component_weights']}")
        print(f"Düşük skorlu maçlarda beraberlik çarpanı: {params.get('low_scoring_draw_boost', 1.25)}")
        print(f"Sıfır enflasyon faktörü: {params.get('zero_inflation_factor', 1.2)}")
        
        # Tahmin ve sonuçları analiz et
        print("\nTahmin ve sonuçlar analiz ediliyor...")
        analysis = predictor.analyze_predictions_and_results()
        
        if analysis.get('sufficient_data', False):
            print(f"Analiz edilen maç sayısı: {analysis.get('analyzed_matches', 0)}")
            print(f"Sonuç doğruluğu: {analysis.get('outcome_accuracy', 0):.4f}")
            print(f"2.5 Üst/Alt doğruluğu: {analysis.get('over_under_accuracy', 0):.4f}")
            print(f"Kesin skor doğruluğu: {analysis.get('exact_score_accuracy', 0):.4f}")
            print(f"Log loss: {analysis.get('log_loss', 0):.4f}")
            
            # Parametreleri güncelle
            print("\nModel parametreleri güncelleniyor...")
            update_result = predictor.update_model_parameters()
            
            if update_result.get('success', False):
                if update_result.get('updated', False):
                    print("Parametreler güncellendi:")
                    for param, value in update_result.get('updates', {}).items():
                        print(f"- {param}: {value:.4f}")
                else:
                    print("Parametreler yeterince iyi, güncelleme yapılmadı.")
                
                # Uzun vadeli performans
                print("\nUzun vadeli performans analizi:")
                performance = predictor.get_long_term_performance()
                
                if performance.get('history_available', False):
                    print(f"Veri noktaları: {performance.get('data_points', 0)}")
                    latest = performance.get('latest_metrics', {})
                    print(f"Son doğruluk: {latest.get('outcome_accuracy', 0):.4f}")
                    print(f"Son log loss: {latest.get('log_loss', 0):.4f}")
                    
                    trend = performance.get('trend_analysis', {})
                    if trend.get('improving', False):
                        print("Trend: İYİLEŞİYOR")
                    elif trend.get('stable', False):
                        print("Trend: İSTİKRARLI")
                    elif trend.get('degrading', False):
                        print("Trend: KÖTÜLEŞIYOR")
                else:
                    print("Henüz yeterli performans geçmişi yok.")
                
                print("\nModeli kapsamlı eğitmek için:")
                print("result = predictor.train_model_from_history(max_iterations=10)")
                
            else:
                print(f"Parametre güncellemesi başarısız: {update_result.get('error', 'Bilinmeyen hata')}")
        else:
            print(f"Yeterli doğrulama verisi yok ({analysis.get('analyzed_matches', 0)} maç). En az {predictor.min_validation_matches} maç gerekli.")
        
    except Exception as e:
        print(f"Hata: {str(e)}")