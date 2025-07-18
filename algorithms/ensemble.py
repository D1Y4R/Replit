"""
Ensemble Tahmin Birleştirici
Tüm modellerin ağırlıklı ortalamasını alarak nihai tahmin üretir
Dinamik ağırlık sistemi ile çalışır
"""
import numpy as np
import logging
import os
import sys

# Proje root'a ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class EnsemblePredictor:
    """
    Farklı modellerin tahminlerini birleştiren ensemble sistem
    Dinamik ağırlık hesaplama özelliği ile
    """
    
    def __init__(self):
        # Dinamik ağırlık hesaplayıcıyı başlat
        try:
            from dynamic_weight_calculator import DynamicWeightCalculator
            self.dynamic_calculator = DynamicWeightCalculator()
            self.use_dynamic_weights = True
            logger.info("Dinamik ağırlık sistemi aktif")
        except Exception as e:
            logger.warning(f"Dinamik ağırlık sistemi yüklenemedi: {e}")
            self.dynamic_calculator = None
            self.use_dynamic_weights = False
            
        # Varsayılan model ağırlıkları (fallback)
        self.weights = {
            'poisson': 0.25,     # Temel model, güvenilir
            'dixon_coles': 0.18, # Düşük skorlar için iyi
            'xgboost': 0.12,     # ML gücü
            'monte_carlo': 0.15, # Belirsizlik için
            'crf': 0.15,         # CRF tahmin modeli
            'neural_network': 0.15  # Neural Network modeli
        }
        
        # Ekstrem maçlar için ağırlıklar (fallback)
        self.extreme_weights = {
            'poisson': 0.35,     # Yüksek skorları daha iyi modeller
            'dixon_coles': 0.08, # Düşük skor eğilimini azalt
            'xgboost': 0.18,     # Veri tabanlı tahmin
            'monte_carlo': 0.15, # Simülasyon
            'crf': 0.12,         # CRF modeli
            'neural_network': 0.12  # Neural Network modeli
        }
        
        # Durum bazlı ağırlık ayarları (fallback)
        self.adjustments = {
            'low_scoring': {'dixon_coles': +0.10, 'poisson': -0.10},
            'high_elo_diff': {'xgboost': +0.10, 'monte_carlo': -0.05},
            'close_match': {'monte_carlo': +0.05, 'xgboost': -0.05}
        }
        
    def _fallback_weight_calculation(self, match_context, algorithm_weights=None):
        """
        Eski ağırlık hesaplama sistemi (fallback)
        """
        # Ekstrem maç kontrolü
        from algorithms.extreme_detector import ExtremeMatchDetector
        detector = ExtremeMatchDetector()
        
        home_stats = match_context.get('home_stats', {})
        away_stats = match_context.get('away_stats', {})
        
        is_extreme, extreme_details = detector.is_extreme_match(home_stats, away_stats)
        
        # Dinamik ağırlıklar sağlanmışsa öncelik ver
        if algorithm_weights:
            adjusted_weights = algorithm_weights.copy()
            logger.info(f"Self-learning dinamik ağırlıklar kullanılıyor: {adjusted_weights}")
        # Ekstrem maç ise özel ağırlıkları kullan
        elif is_extreme:
            adjusted_weights = self.extreme_weights.copy()
            logger.info(f"Ekstrem maç algılandı, özel ağırlıklar kullanılıyor")
        else:
            adjusted_weights = self.weights.copy()
        
        # Bağlama göre ağırlık ayarla
        adjusted_weights = self._adjust_weights_by_context(adjusted_weights, match_context)
        
        # Normalize et
        total_weight = sum(adjusted_weights.values())
        adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
        
        return adjusted_weights
        
    def combine_predictions(self, model_predictions, match_context, algorithm_weights=None):
        """
        Farklı model tahminlerini birleştir
        
        Args:
            model_predictions: dict - Her modelin tahminleri
            match_context: dict - Maç bağlamı (lambda'lar, elo farkı vb.)
            algorithm_weights: dict - Opsiyonel dinamik ağırlıklar (self-learning'den)
            
        Returns:
            dict: Birleştirilmiş tahmin
        """
        # Dinamik ağırlık sistemi aktifse öncelikli kullan
        if self.use_dynamic_weights and self.dynamic_calculator:
            try:
                # Maç bilgilerini hazırla
                match_info = {
                    'league': match_context.get('league', ''),
                    'home_team': match_context.get('home_team', ''),
                    'away_team': match_context.get('away_team', ''),
                    'elo_diff': match_context.get('elo_diff', 0),
                    'home_stats': match_context.get('home_stats', {}),
                    'away_stats': match_context.get('away_stats', {}),
                    'date': match_context.get('date', ''),
                    'home_position': match_context.get('home_position', 10),
                    'away_position': match_context.get('away_position', 10)
                }
                
                # Dinamik ağırlıkları hesapla
                adjusted_weights = self.dynamic_calculator.calculate_weights(match_info)
                logger.info("YENİ Dinamik ağırlık sistemi kullanıldı")
                
            except Exception as e:
                logger.error(f"Dinamik ağırlık hesaplama hatası: {e}")
                # Fallback: Eski sisteme dön
                adjusted_weights = self._fallback_weight_calculation(match_context, algorithm_weights)
        else:
            # Eski sistem (geriye dönük uyumluluk) - self-learning veya varsayılan
            adjusted_weights = self._fallback_weight_calculation(match_context, algorithm_weights)
        
        logger.info(f"Ensemble ağırlıkları: {adjusted_weights}")
        
        # Birleştirilmiş tahminler
        combined = {
            'home_win': 0.0,
            'draw': 0.0,
            'away_win': 0.0,
            'over_2_5': 0.0,
            'under_2_5': 0.0,
            'btts_yes': 0.0,
            'btts_no': 0.0,
            'expected_goals': {'home': 0.0, 'away': 0.0},
            'most_likely_scores': {},
            'confidence': 0.0
        }
        
        # Her modelin katkısını ekle
        for model_name, predictions in model_predictions.items():
            if model_name not in adjusted_weights:
                continue
                
            weight = adjusted_weights[model_name]
            
            # 1X2 tahminleri
            combined['home_win'] += predictions.get('home_win', 0) * weight
            combined['draw'] += predictions.get('draw', 0) * weight
            combined['away_win'] += predictions.get('away_win', 0) * weight
            
            # Gol tahminleri
            combined['over_2_5'] += predictions.get('over_2_5', 0) * weight
            combined['under_2_5'] += predictions.get('under_2_5', 0) * weight
            combined['btts_yes'] += predictions.get('btts_yes', 0) * weight
            combined['btts_no'] += predictions.get('btts_no', 0) * weight
            
            # Beklenen goller
            if 'expected_goals' in predictions:
                combined['expected_goals']['home'] += predictions['expected_goals'].get('home', 0) * weight
                combined['expected_goals']['away'] += predictions['expected_goals'].get('away', 0) * weight
                
            # Güven seviyesi - modelin kendi güveni ve tahmin keskinliği
            model_confidence = predictions.get('confidence', 0.7)
            # En yüksek olasılığa göre güven ayarla
            max_prob = max(predictions.get('home_win', 0), predictions.get('draw', 0), predictions.get('away_win', 0))
            adjusted_confidence = model_confidence * (max_prob / 100) * 1.2  # Max prob'a göre ayarla
            combined['confidence'] += adjusted_confidence * weight
            
            # Debug: Model güven değerlerini logla
            logger.debug(f"Model {model_name}: confidence={model_confidence}, max_prob={max_prob}, adjusted={adjusted_confidence}, weight={weight}")
            
        # En olası sonucu belirle
        outcomes = {
            'HOME_WIN': combined['home_win'],
            'DRAW': combined['draw'],
            'AWAY_WIN': combined['away_win']
        }
        combined['most_likely_outcome'] = max(outcomes, key=outcomes.get)
        
        # Kesin skor tahminleri (matrislerden)
        combined['most_likely_scores'] = self._combine_score_predictions(model_predictions, adjusted_weights)
        
        # Güven değerini tahmin keskinliğine göre ayarla
        max_outcome_prob = max(outcomes.values())
        
        # Debug: Güven hesaplama öncesi
        logger.info(f"Ensemble güven hesaplama - Başlangıç: {combined['confidence']:.3f}, Max prob: {max_outcome_prob:.1f}%")
        
        # Yeni dinamik güven hesaplama - tahmin keskinliğine göre
        if max_outcome_prob > 60:  # Çok net favori
            # %60+ için güven %75-90 arası
            confidence_boost = (max_outcome_prob - 60) / 40  # 0-1 arası
            combined['confidence'] = 0.75 + (confidence_boost * 0.15)
        elif max_outcome_prob > 45:  # Orta düzey favori  
            # %45-60 için güven %60-75 arası
            confidence_boost = (max_outcome_prob - 45) / 15  # 0-1 arası
            combined['confidence'] = 0.60 + (confidence_boost * 0.15)
        elif max_outcome_prob > 35:  # Hafif favori
            # %35-45 için güven %50-60 arası
            confidence_boost = (max_outcome_prob - 35) / 10  # 0-1 arası
            combined['confidence'] = 0.50 + (confidence_boost * 0.10)
        else:  # Çok dengeli maç
            # %35 altı için güven %45-50 arası
            confidence_boost = max(0, (max_outcome_prob - 25) / 10)  # 0-1 arası
            combined['confidence'] = 0.45 + (confidence_boost * 0.05)
            
        # Model güven değerlerinin ortalamasını da hesaba kat
        model_confidence_avg = 0
        model_count = 0
        for model_name, predictions in model_predictions.items():
            if 'confidence' in predictions:
                model_confidence_avg += predictions['confidence']
                model_count += 1
        
        if model_count > 0:
            model_confidence_avg /= model_count
            # Final güven = %70 tahmin keskinliği + %30 model ortalaması
            combined['confidence'] = (combined['confidence'] * 0.7) + (model_confidence_avg * 0.3)
            
        # Güven değerini sınırla
        combined['confidence'] = max(0.45, min(0.90, combined['confidence']))
        
        # Debug: Final güven değeri
        logger.info(f"Ensemble güven hesaplama - Model ortalaması: {model_confidence_avg:.3f} ({model_count} model)")
        logger.info(f"Ensemble güven hesaplama - Final: {combined['confidence']:.3f}")
        
        # 1X2 olasılıklarını normalize et (toplamı 100'e tamamla)
        match_outcome_total = combined['home_win'] + combined['draw'] + combined['away_win']
        if match_outcome_total > 0:
            combined['home_win'] = (combined['home_win'] / match_outcome_total) * 100
            combined['draw'] = (combined['draw'] / match_outcome_total) * 100
            combined['away_win'] = (combined['away_win'] / match_outcome_total) * 100
        else:
            # Fallback durumu
            combined['home_win'] = 33.3
            combined['draw'] = 33.3
            combined['away_win'] = 33.4
            
        # BTTS değerlerini normalize et (toplamı 100'e tamamla)
        btts_total = combined['btts_yes'] + combined['btts_no']
        if btts_total > 0:
            combined['btts_yes'] = (combined['btts_yes'] / btts_total) * 100
            combined['btts_no'] = (combined['btts_no'] / btts_total) * 100
        else:
            # Fallback durumu
            combined['btts_yes'] = 50.0
            combined['btts_no'] = 50.0
            
        # Over/Under değerlerini normalize et (toplamı 100'e tamamla)
        ou_total = combined['over_2_5'] + combined['under_2_5']
        if ou_total > 0:
            combined['over_2_5'] = (combined['over_2_5'] / ou_total) * 100
            combined['under_2_5'] = (combined['under_2_5'] / ou_total) * 100
        else:
            # Fallback durumu
            combined['over_2_5'] = 45.0
            combined['under_2_5'] = 55.0
        
        return combined
        
    def _adjust_weights_by_context(self, weights, context):
        """
        Maç bağlamına göre ağırlıkları ayarla
        """
        lambda_home = context.get('lambda_home', 1.5)
        lambda_away = context.get('lambda_away', 1.5)
        elo_diff = context.get('elo_diff', 0)
        
        # Düşük skorlu maç (toplam lambda < 2.5)
        if lambda_home + lambda_away < 2.5:
            logger.debug("Düşük skorlu maç tespit edildi")
            for model, adjustment in self.adjustments['low_scoring'].items():
                if model in weights:
                    weights[model] = max(0, weights[model] + adjustment)
                    
        # Yüksek Elo farkı (favori var)
        if abs(elo_diff) > 300:
            logger.debug(f"Yüksek Elo farkı: {elo_diff}")
            for model, adjustment in self.adjustments['high_elo_diff'].items():
                if model in weights:
                    weights[model] = max(0, weights[model] + adjustment)
                    
        # Yakın maç
        elif abs(elo_diff) < 100:
            logger.debug("Yakın maç tespit edildi")
            for model, adjustment in self.adjustments['close_match'].items():
                if model in weights:
                    weights[model] = max(0, weights[model] + adjustment)
                    
        return weights
        
    def _combine_score_predictions(self, model_predictions, weights):
        """
        Farklı modellerin skor tahminlerini birleştir
        """
        combined_scores = {}
        
        # Her modelin skor tahminlerini topla
        for model_name, predictions in model_predictions.items():
            if model_name not in weights or 'score_probabilities' not in predictions:
                continue
                
            weight = weights[model_name]
            
            for score_pred in predictions['score_probabilities']:
                score = score_pred['score']
                prob = score_pred['probability'] * weight
                
                if score in combined_scores:
                    combined_scores[score] += prob
                else:
                    combined_scores[score] = prob
                    
        # En olası 5 skoru sırala
        sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return [{'score': s[0], 'probability': s[1]} for s in sorted_scores]