"""
Self-Learning Model
Dinamik olarak öğrenen ve parametrelerini güncelleyen model
"""
import json
import os
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SelfLearningModel:
    """
    Öğrenen model - tahmin sonuçlarına göre kendini günceller
    """
    
    def __init__(self):
        self.model_path = 'models/self_learning_model.json'
        self.parameters = self.load_parameters()
        
    def load_parameters(self):
        """
        Model parametrelerini yükle
        """
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'r') as f:
                    params = json.load(f)
                    logger.info("Self-learning parametreleri yüklendi")
                    return params
            except Exception as e:
                logger.error(f"Self-learning model yükleme hatası: {e}")
                
        # Varsayılan parametreler
        return self._get_default_parameters()
        
    def _get_default_parameters(self):
        """
        Varsayılan model parametreleri
        """
        return {
            'component_weights': {
                'poisson': 0.25,
                'dixon_coles': 0.18,
                'xgboost': 0.12,
                'monte_carlo': 0.15,
                'crf': 0.15,
                'neural_network': 0.15
            },
            'low_scoring_draw_boost': 1.15,
            'medium_scoring_draw_boost': 1.05,
            'high_scoring_draw_boost': 0.95,
            'low_scoring_threshold': 2.0,
            'high_scoring_threshold': 3.5,
            'exact_score_factors': {
                '0-0': 1.1,
                '1-1': 1.05,
                '0-1': 0.98,
                '1-0': 0.98,
                '2-1': 0.97,
                '1-2': 0.97
            },
            'zero_inflation_factor': 1.08,
            'one_inflation_factor': 1.03,
            'home_advantage_base': 1.1,
            'form_decay_rate': 0.85,
            'h2h_recency_weight': 0.7,
            'league_type_factors': {
                'major': 1.0,
                'minor': 0.95,
                'friendly': 0.85
            },
            'last_updated': datetime.now().isoformat(),
            'validation_metrics': {
                'accuracy': 0.68,
                'precision': 0.65,
                'recall': 0.70
            },
            'learning_history': []
        }
        
    def get_dynamic_weights(self, match_context):
        """
        Maç bağlamına göre dinamik ağırlıklar döndür
        
        Args:
            match_context: Maç bağlamı (takım güçleri, form, vb.)
            
        Returns:
            dict: Algoritma ağırlıkları
        """
        weights = self.parameters['component_weights'].copy()
        
        # Ekstrem maçlar için ağırlık ayarlaması
        if match_context.get('is_extreme', False):
            weights['poisson'] = 0.22
            weights['dixon_coles'] = 0.25
            weights['monte_carlo'] = 0.22
            weights['xgboost'] = 0.10
            weights['crf'] = 0.10
            weights['neural_network'] = 0.11
            
        # Düşük skorlu maçlar için
        elif match_context.get('expected_total_goals', 2.5) < self.parameters['low_scoring_threshold']:
            weights['dixon_coles'] = 0.30
            weights['poisson'] = 0.22
            weights['monte_carlo'] = 0.13
            weights['xgboost'] = 0.10
            weights['crf'] = 0.13
            weights['neural_network'] = 0.12
            
        # Yüksek skorlu maçlar için
        elif match_context.get('expected_total_goals', 2.5) > self.parameters['high_scoring_threshold']:
            weights['monte_carlo'] = 0.25
            weights['poisson'] = 0.25
            weights['dixon_coles'] = 0.13
            weights['xgboost'] = 0.10
            weights['crf'] = 0.13
            weights['neural_network'] = 0.14
            
        # Normalize et
        total = sum(weights.values())
        if total > 0:
            for key in weights:
                weights[key] /= total
                
        return weights
        
    def adjust_draw_probability(self, base_draw_prob, expected_goals):
        """
        Beklenen gol sayısına göre beraberlik olasılığını ayarla
        
        Args:
            base_draw_prob: Temel beraberlik olasılığı
            expected_goals: Beklenen toplam gol
            
        Returns:
            float: Ayarlanmış beraberlik olasılığı
        """
        if expected_goals < self.parameters['low_scoring_threshold']:
            return base_draw_prob * self.parameters['low_scoring_draw_boost']
        elif expected_goals > self.parameters['high_scoring_threshold']:
            return base_draw_prob * self.parameters['high_scoring_draw_boost']
        else:
            return base_draw_prob * self.parameters['medium_scoring_draw_boost']
            
    def get_exact_score_factor(self, home_score, away_score):
        """
        Belirli skorlar için düzeltme faktörü
        
        Args:
            home_score: Ev sahibi skoru
            away_score: Deplasman skoru
            
        Returns:
            float: Düzeltme faktörü
        """
        score_key = f"{home_score}-{away_score}"
        return self.parameters['exact_score_factors'].get(score_key, 1.0)
        
    def apply_home_advantage(self, home_prob, away_prob, league_type='major'):
        """
        Ev sahibi avantajını uygula
        
        Args:
            home_prob: Ev sahibi kazanma olasılığı
            away_prob: Deplasman kazanma olasılığı
            league_type: Lig tipi
            
        Returns:
            tuple: (adjusted_home_prob, adjusted_away_prob)
        """
        league_factor = self.parameters['league_type_factors'].get(league_type, 1.0)
        home_advantage = self.parameters['home_advantage_base'] * league_factor
        
        # Ev sahibi avantajını uygula
        total = home_prob + away_prob
        if total > 0:
            home_ratio = home_prob / total
            away_ratio = away_prob / total
            
            # Avantajı ekle
            home_ratio = home_ratio * home_advantage
            
            # Normalize
            new_total = home_ratio + away_ratio
            home_prob = (home_ratio / new_total) * total
            away_prob = (away_ratio / new_total) * total
            
        return home_prob, away_prob
        
    def update_from_result(self, prediction, actual_result):
        """
        Gerçek sonuçtan öğren ve parametreleri güncelle
        
        Args:
            prediction: Yapılan tahmin
            actual_result: Gerçek maç sonucu
        """
        # Bu fonksiyon ileride gerçek sonuçlarla
        # model parametrelerini güncellemek için kullanılacak
        learning_entry = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'actual': actual_result,
            'accuracy': self._calculate_accuracy(prediction, actual_result)
        }
        
        self.parameters['learning_history'].append(learning_entry)
        
        # Son 100 tahmin üzerinden metrik güncelle
        if len(self.parameters['learning_history']) > 100:
            recent_accuracy = np.mean([
                entry['accuracy'] 
                for entry in self.parameters['learning_history'][-100:]
            ])
            self.parameters['validation_metrics']['accuracy'] = recent_accuracy
            
        self.save_parameters()
        
    def _calculate_accuracy(self, prediction, actual):
        """
        Tahmin doğruluğunu hesapla
        """
        pred_outcome = prediction.get('most_likely_outcome', 'DRAW')
        actual_outcome = actual.get('outcome', 'DRAW')
        
        if pred_outcome == actual_outcome:
            return 1.0
        else:
            # Kısmi puan ver (yakın tahminler için)
            pred_prob = prediction.get('predictions', {}).get(actual_outcome.lower(), 0)
            return min(pred_prob / 100, 0.5)
            
    def save_parameters(self):
        """
        Parametreleri kaydet
        """
        try:
            self.parameters['last_updated'] = datetime.now().isoformat()
            with open(self.model_path, 'w') as f:
                json.dump(self.parameters, f, indent=2)
                logger.info("Self-learning parametreleri güncellendi")
        except Exception as e:
            logger.error(f"Parametre kaydetme hatası: {e}")
            
    def get_feature_importance(self):
        """
        Özellik önem derecelerini döndür
        """
        return {
            'algorithm_weights': self.parameters['component_weights'],
            'scoring_adjustments': {
                'low_scoring_draw_boost': self.parameters['low_scoring_draw_boost'],
                'high_scoring_penalty': 1 / self.parameters['high_scoring_draw_boost']
            },
            'home_advantage': self.parameters['home_advantage_base'],
            'form_decay': self.parameters['form_decay_rate']
        }