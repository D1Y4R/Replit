"""
Sürekli Öğrenme Döngüsü
Model performansını takip eder ve algoritma ağırlıklarını günceller
"""
import json
import numpy as np
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class ContinuousLearner:
    """
    Tahmin sonuçlarından öğrenen ve kendini geliştiren sistem
    """
    
    def __init__(self):
        self.learning_file = 'continuous_learning_data.json'
        self.learning_data = self.load_learning_data()
        self.min_samples_for_update = 10
        
    def load_learning_data(self):
        """Öğrenme verilerini yükle"""
        if os.path.exists(self.learning_file):
            try:
                with open(self.learning_file, 'r') as f:
                    return json.load(f)
            except:
                return self._get_default_learning_data()
        return self._get_default_learning_data()
        
    def _get_default_learning_data(self):
        """Varsayılan öğrenme verileri"""
        return {
            'algorithm_performance': {
                'poisson': {'correct': 0, 'total': 0, 'weight': 0.25},
                'dixon_coles': {'correct': 0, 'total': 0, 'weight': 0.18},
                'xgboost': {'correct': 0, 'total': 0, 'weight': 0.12},
                'monte_carlo': {'correct': 0, 'total': 0, 'weight': 0.15},
                'crf': {'correct': 0, 'total': 0, 'weight': 0.15},
                'neural_network': {'correct': 0, 'total': 0, 'weight': 0.15}
            },
            'context_performance': {
                'extreme_matches': {'correct': 0, 'total': 0},
                'low_scoring': {'correct': 0, 'total': 0},
                'high_elo_diff': {'correct': 0, 'total': 0},
                'close_matches': {'correct': 0, 'total': 0}
            },
            'learning_history': [],
            'last_update': None,
            'total_predictions': 0,
            'correct_predictions': 0
        }
        
    def save_learning_data(self):
        """Öğrenme verilerini kaydet"""
        with open(self.learning_file, 'w') as f:
            json.dump(self.learning_data, f, indent=2)
            
    def update_from_match_result(self, match_id, prediction, actual_result):
        """
        Maç sonucundan öğren ve modeli güncelle
        
        Args:
            match_id: Maç ID'si
            prediction: Yapılan tahmin
            actual_result: Gerçek sonuç
        """
        # Tahmin doğruluğunu kontrol et
        is_correct = self._check_prediction_accuracy(prediction, actual_result)
        
        # Genel istatistikleri güncelle
        self.learning_data['total_predictions'] += 1
        if is_correct:
            self.learning_data['correct_predictions'] += 1
            
        # Her algoritmanın katkısını değerlendir
        if 'algorithm_contributions' in prediction:
            self._update_algorithm_performance(prediction['algorithm_contributions'], is_correct)
            
        # Maç kontekstini değerlendir
        match_context = prediction.get('match_context', {})
        self._update_context_performance(match_context, is_correct)
        
        # Öğrenme geçmişine ekle
        learning_entry = {
            'match_id': match_id,
            'timestamp': datetime.now().isoformat(),
            'prediction_correct': is_correct,
            'predicted_outcome': prediction.get('most_likely_outcome'),
            'actual_outcome': self._get_actual_outcome(actual_result),
            'confidence': prediction.get('confidence', 0),
            'context': match_context
        }
        self.learning_data['learning_history'].append(learning_entry)
        
        # Belirli aralıklarla ağırlıkları güncelle
        if self.learning_data['total_predictions'] % self.min_samples_for_update == 0:
            self._update_algorithm_weights()
            
        self.learning_data['last_update'] = datetime.now().isoformat()
        self.save_learning_data()
        
        logger.info(f"Öğrenme güncellendi - Doğruluk: {self.get_overall_accuracy():.1f}%")
        
    def _check_prediction_accuracy(self, prediction, actual):
        """Tahmin doğruluğunu kontrol et"""
        pred_outcome = prediction.get('most_likely_outcome', '')
        actual_outcome = self._get_actual_outcome(actual)
        return pred_outcome == actual_outcome
        
    def _get_actual_outcome(self, actual):
        """Gerçek maç sonucunu belirle"""
        home_goals = actual.get('home_goals', 0)
        away_goals = actual.get('away_goals', 0)
        
        if home_goals > away_goals:
            return 'HOME_WIN'
        elif home_goals < away_goals:
            return 'AWAY_WIN'
        else:
            return 'DRAW'
            
    def _update_algorithm_performance(self, contributions, is_correct):
        """Algoritma performanslarını güncelle"""
        for algo, contrib in contributions.items():
            if algo in self.learning_data['algorithm_performance']:
                self.learning_data['algorithm_performance'][algo]['total'] += 1
                
                # Algoritmanın tahmini doğruysa ve yüksek katkı sağladıysa
                if is_correct and contrib.get('weight', 0) > 0.1:
                    self.learning_data['algorithm_performance'][algo]['correct'] += 1
                    
    def _update_context_performance(self, context, is_correct):
        """Maç konteksti performansını güncelle"""
        # Ekstrem maç
        if context.get('is_extreme', False):
            self.learning_data['context_performance']['extreme_matches']['total'] += 1
            if is_correct:
                self.learning_data['context_performance']['extreme_matches']['correct'] += 1
                
        # Düşük skorlu maç
        if context.get('expected_total_goals', 2.5) < 2.0:
            self.learning_data['context_performance']['low_scoring']['total'] += 1
            if is_correct:
                self.learning_data['context_performance']['low_scoring']['correct'] += 1
                
        # Yüksek Elo farkı
        if abs(context.get('elo_diff', 0)) > 300:
            self.learning_data['context_performance']['high_elo_diff']['total'] += 1
            if is_correct:
                self.learning_data['context_performance']['high_elo_diff']['correct'] += 1
                
        # Yakın maç
        elif abs(context.get('elo_diff', 0)) < 100:
            self.learning_data['context_performance']['close_matches']['total'] += 1
            if is_correct:
                self.learning_data['context_performance']['close_matches']['correct'] += 1
                
    def _update_algorithm_weights(self):
        """Performansa göre algoritma ağırlıklarını güncelle"""
        logger.info("Algoritma ağırlıkları güncelleniyor...")
        
        # Her algoritmanın başarı oranını hesapla
        success_rates = {}
        for algo, perf in self.learning_data['algorithm_performance'].items():
            if perf['total'] > 0:
                success_rates[algo] = perf['correct'] / perf['total']
            else:
                success_rates[algo] = 0.5  # Varsayılan
                
        # Normalize edilmiş ağırlıklar hesapla
        total_success = sum(success_rates.values())
        if total_success > 0:
            for algo in self.learning_data['algorithm_performance']:
                # Mevcut ağırlık ile yeni performansı birleştir (momentum)
                old_weight = self.learning_data['algorithm_performance'][algo]['weight']
                new_weight = success_rates[algo] / total_success
                
                # Momentum faktörü ile güncelle (ani değişimleri önle)
                momentum = 0.7
                updated_weight = momentum * old_weight + (1 - momentum) * new_weight
                
                # Minimum ve maksimum sınırlar
                updated_weight = max(0.05, min(0.35, updated_weight))
                
                self.learning_data['algorithm_performance'][algo]['weight'] = updated_weight
                
        # Ağırlıkları normalize et
        total_weight = sum(perf['weight'] for perf in self.learning_data['algorithm_performance'].values())
        for algo in self.learning_data['algorithm_performance']:
            self.learning_data['algorithm_performance'][algo]['weight'] /= total_weight
            
        logger.info(f"Yeni ağırlıklar: {self.get_current_weights()}")
        
    def get_current_weights(self):
        """Güncel algoritma ağırlıklarını döndür"""
        weights = {}
        for algo, perf in self.learning_data['algorithm_performance'].items():
            weights[algo] = perf['weight']
        return weights
        
    def get_overall_accuracy(self):
        """Genel tahmin doğruluğunu döndür"""
        if self.learning_data['total_predictions'] > 0:
            return (self.learning_data['correct_predictions'] / 
                   self.learning_data['total_predictions']) * 100
        return 0.0
        
    def get_context_recommendations(self, match_context):
        """Maç kontekstine göre öneriler sun"""
        recommendations = {
            'weight_adjustments': {},
            'confidence_modifier': 1.0,
            'special_considerations': []
        }
        
        # Ekstrem maçlar için
        if match_context.get('is_extreme', False):
            perf = self.learning_data['context_performance']['extreme_matches']
            if perf['total'] > 5:
                accuracy = perf['correct'] / perf['total']
                if accuracy < 0.5:
                    recommendations['confidence_modifier'] *= 0.8
                    recommendations['special_considerations'].append(
                        "Ekstrem maçlarda düşük performans - güven azaltıldı"
                    )
                    
        # Düşük skorlu maçlar için
        if match_context.get('expected_total_goals', 2.5) < 2.0:
            perf = self.learning_data['context_performance']['low_scoring']
            if perf['total'] > 5:
                accuracy = perf['correct'] / perf['total']
                if accuracy > 0.6:
                    recommendations['weight_adjustments']['dixon_coles'] = 1.2
                    recommendations['special_considerations'].append(
                        "Düşük skorlu maçlarda iyi performans - Dixon-Coles ağırlığı artırıldı"
                    )
                    
        return recommendations
        
    def get_learning_report(self):
        """Öğrenme raporu oluştur"""
        report = {
            'overall_statistics': {
                'total_predictions': self.learning_data['total_predictions'],
                'correct_predictions': self.learning_data['correct_predictions'],
                'accuracy': f"{self.get_overall_accuracy():.1f}%",
                'last_update': self.learning_data['last_update']
            },
            'algorithm_performance': {},
            'context_performance': {},
            'recent_trend': self._analyze_recent_trend()
        }
        
        # Algoritma performansları
        for algo, perf in self.learning_data['algorithm_performance'].items():
            if perf['total'] > 0:
                accuracy = (perf['correct'] / perf['total']) * 100
            else:
                accuracy = 0
                
            report['algorithm_performance'][algo] = {
                'accuracy': f"{accuracy:.1f}%",
                'current_weight': f"{perf['weight']:.3f}",
                'total_predictions': perf['total']
            }
            
        # Kontekst performansları
        for context, perf in self.learning_data['context_performance'].items():
            if perf['total'] > 0:
                accuracy = (perf['correct'] / perf['total']) * 100
            else:
                accuracy = 0
                
            report['context_performance'][context] = {
                'accuracy': f"{accuracy:.1f}%",
                'total_matches': perf['total']
            }
            
        return report
        
    def _analyze_recent_trend(self):
        """Son trend analizi"""
        if len(self.learning_data['learning_history']) < 20:
            return "Yetersiz veri"
            
        recent = self.learning_data['learning_history'][-20:]
        first_half = sum(1 for entry in recent[:10] if entry['prediction_correct'])
        second_half = sum(1 for entry in recent[10:] if entry['prediction_correct'])
        
        if second_half > first_half:
            return f"İyileşiyor (+{second_half - first_half} doğru tahmin)"
        elif second_half < first_half:
            return f"Kötüleşiyor ({second_half - first_half} doğru tahmin)"
        else:
            return "Stabil performans"