"""
Otomatik Model Değerlendirme Sistemi
Tahmin doğruluğunu ölçer ve model performansını takip eder
"""
import json
import numpy as np
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss, log_loss
import os

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Model performansını değerlendiren ve takip eden sistem
    """
    
    def __init__(self):
        self.metrics_file = 'model_metrics.json'
        self.evaluation_history = self.load_metrics()
        
    def load_metrics(self):
        """Metrik geçmişini yükle"""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except:
                return {'evaluations': [], 'summary': {}}
        return {'evaluations': [], 'summary': {}}
        
    def save_metrics(self):
        """Metrikleri kaydet"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2)
            
    def evaluate_prediction(self, prediction, actual_result):
        """
        Tahmin ve gerçek sonucu karşılaştırarak metrikler hesapla
        
        Args:
            prediction: Model tahmini
            actual_result: Gerçek maç sonucu
            
        Returns:
            dict: Hesaplanan metrikler
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'match_id': prediction.get('match_id', 'unknown'),
            'outcome_accuracy': self._check_outcome_accuracy(prediction, actual_result),
            'score_accuracy': self._check_score_accuracy(prediction, actual_result),
            'goal_prediction_error': self._calculate_goal_error(prediction, actual_result),
            'btts_accuracy': self._check_btts_accuracy(prediction, actual_result),
            'over_under_accuracy': self._check_over_under_accuracy(prediction, actual_result),
            'probability_calibration': self._calculate_calibration(prediction, actual_result)
        }
        
        # Brier score hesapla
        if 'home_win_probability' in prediction:
            actual_outcome = self._get_actual_outcome(actual_result)
            pred_probs = [
                prediction.get('home_win_probability', 0) / 100,
                prediction.get('draw_probability', 0) / 100,
                prediction.get('away_win_probability', 0) / 100
            ]
            
            try:
                metrics['brier_score'] = self._calculate_brier_score(pred_probs, actual_outcome)
                metrics['log_loss'] = self._calculate_log_loss(pred_probs, actual_outcome)
            except:
                metrics['brier_score'] = None
                metrics['log_loss'] = None
        
        # Değerlendirmeyi kaydet
        self.evaluation_history['evaluations'].append(metrics)
        self._update_summary_metrics()
        self.save_metrics()
        
        logger.info(f"Tahmin değerlendirildi: {metrics}")
        return metrics
        
    def _check_outcome_accuracy(self, prediction, actual):
        """1X2 tahmin doğruluğunu kontrol et"""
        pred_outcome = prediction.get('most_likely_outcome', '')
        actual_home = actual.get('home_goals', 0)
        actual_away = actual.get('away_goals', 0)
        
        if actual_home > actual_away:
            actual_outcome = 'HOME_WIN'
        elif actual_home < actual_away:
            actual_outcome = 'AWAY_WIN'
        else:
            actual_outcome = 'DRAW'
            
        return pred_outcome == actual_outcome
        
    def _check_score_accuracy(self, prediction, actual):
        """Kesin skor tahmin doğruluğunu kontrol et"""
        pred_score = prediction.get('most_likely_score', '')
        actual_score = f"{actual.get('home_goals', 0)}-{actual.get('away_goals', 0)}"
        return pred_score == actual_score
        
    def _calculate_goal_error(self, prediction, actual):
        """Gol tahmin hatasını hesapla"""
        pred_home = prediction.get('expected_goals', {}).get('home', 0)
        pred_away = prediction.get('expected_goals', {}).get('away', 0)
        actual_home = actual.get('home_goals', 0)
        actual_away = actual.get('away_goals', 0)
        
        home_error = abs(pred_home - actual_home)
        away_error = abs(pred_away - actual_away)
        
        return {
            'home_error': home_error,
            'away_error': away_error,
            'total_error': home_error + away_error,
            'mae': (home_error + away_error) / 2
        }
        
    def _check_btts_accuracy(self, prediction, actual):
        """BTTS tahmin doğruluğunu kontrol et"""
        pred_btts = prediction.get('both_teams_to_score', {}).get('yes', 0) > 50
        actual_btts = actual.get('home_goals', 0) > 0 and actual.get('away_goals', 0) > 0
        return pred_btts == actual_btts
        
    def _check_over_under_accuracy(self, prediction, actual):
        """Over/Under tahmin doğruluğunu kontrol et"""
        pred_over = prediction.get('over_under', {}).get('over_2_5', 0) > 50
        actual_total = actual.get('home_goals', 0) + actual.get('away_goals', 0)
        actual_over = actual_total > 2.5
        return pred_over == actual_over
        
    def _calculate_calibration(self, prediction, actual):
        """Olasılık kalibrasyonunu hesapla"""
        # En yüksek olasılıklı tahmin doğru mu?
        probs = {
            'HOME_WIN': prediction.get('home_win_probability', 0),
            'DRAW': prediction.get('draw_probability', 0),
            'AWAY_WIN': prediction.get('away_win_probability', 0)
        }
        
        pred_outcome = max(probs, key=probs.get)
        actual_outcome = self._get_actual_outcome(actual)
        
        if pred_outcome == actual_outcome:
            return probs[pred_outcome] / 100  # Doğru tahmin olasılığı
        else:
            return 1 - (probs[pred_outcome] / 100)  # Yanlış tahmin cezası
            
    def _get_actual_outcome(self, actual):
        """Gerçek maç sonucunu belirle"""
        actual_home = actual.get('home_goals', 0)
        actual_away = actual.get('away_goals', 0)
        
        if actual_home > actual_away:
            return 'HOME_WIN'
        elif actual_home < actual_away:
            return 'AWAY_WIN'
        else:
            return 'DRAW'
            
    def _calculate_brier_score(self, pred_probs, actual_outcome):
        """Brier score hesapla"""
        actual_vector = [0, 0, 0]
        outcome_map = {'HOME_WIN': 0, 'DRAW': 1, 'AWAY_WIN': 2}
        actual_vector[outcome_map[actual_outcome]] = 1
        
        brier = sum((p - a) ** 2 for p, a in zip(pred_probs, actual_vector))
        return brier
        
    def _calculate_log_loss(self, pred_probs, actual_outcome):
        """Log loss hesapla"""
        outcome_map = {'HOME_WIN': 0, 'DRAW': 1, 'AWAY_WIN': 2}
        actual_idx = outcome_map[actual_outcome]
        
        # Küçük bir değer ekleyerek log(0) hatası önle
        epsilon = 1e-15
        pred_prob = max(epsilon, min(1 - epsilon, pred_probs[actual_idx]))
        
        return -np.log(pred_prob)
        
    def _update_summary_metrics(self):
        """Özet metrikleri güncelle"""
        if not self.evaluation_history['evaluations']:
            return
            
        evaluations = self.evaluation_history['evaluations']
        
        # Son 100 tahmin üzerinden metrikler
        recent_evals = evaluations[-100:]
        
        outcome_accuracies = [e['outcome_accuracy'] for e in recent_evals]
        score_accuracies = [e['score_accuracy'] for e in recent_evals]
        btts_accuracies = [e['btts_accuracy'] for e in recent_evals]
        ou_accuracies = [e['over_under_accuracy'] for e in recent_evals]
        
        goal_errors = [e['goal_prediction_error']['mae'] for e in recent_evals]
        brier_scores = [e['brier_score'] for e in recent_evals if e.get('brier_score') is not None]
        
        self.evaluation_history['summary'] = {
            'total_evaluations': len(evaluations),
            'recent_evaluations': len(recent_evals),
            'outcome_accuracy': np.mean(outcome_accuracies) * 100,
            'score_accuracy': np.mean(score_accuracies) * 100,
            'btts_accuracy': np.mean(btts_accuracies) * 100,
            'over_under_accuracy': np.mean(ou_accuracies) * 100,
            'avg_goal_error': np.mean(goal_errors),
            'avg_brier_score': np.mean(brier_scores) if brier_scores else None,
            'last_updated': datetime.now().isoformat()
        }
        
    def get_model_performance_report(self):
        """Model performans raporu oluştur"""
        summary = self.evaluation_history.get('summary', {})
        
        report = {
            'overall_performance': {
                'total_predictions': summary.get('total_evaluations', 0),
                'outcome_accuracy': f"{summary.get('outcome_accuracy', 0):.1f}%",
                'exact_score_accuracy': f"{summary.get('score_accuracy', 0):.1f}%",
                'btts_accuracy': f"{summary.get('btts_accuracy', 0):.1f}%",
                'over_under_accuracy': f"{summary.get('over_under_accuracy', 0):.1f}%"
            },
            'prediction_quality': {
                'avg_goal_error': f"{summary.get('avg_goal_error', 0):.2f}",
                'brier_score': f"{summary.get('avg_brier_score', 0):.3f}" if summary.get('avg_brier_score') else 'N/A',
                'last_updated': summary.get('last_updated', 'Never')
            },
            'recent_performance': self._get_recent_performance_trend()
        }
        
        return report
        
    def _get_recent_performance_trend(self):
        """Son performans trendini hesapla"""
        if len(self.evaluation_history['evaluations']) < 10:
            return {'trend': 'Insufficient data', 'direction': 'neutral'}
            
        recent = self.evaluation_history['evaluations'][-20:]
        first_half = recent[:10]
        second_half = recent[10:]
        
        first_acc = np.mean([e['outcome_accuracy'] for e in first_half])
        second_acc = np.mean([e['outcome_accuracy'] for e in second_half])
        
        if second_acc > first_acc + 0.05:
            return {'trend': 'Improving', 'direction': 'up', 'change': f"+{(second_acc - first_acc)*100:.1f}%"}
        elif second_acc < first_acc - 0.05:
            return {'trend': 'Declining', 'direction': 'down', 'change': f"{(second_acc - first_acc)*100:.1f}%"}
        else:
            return {'trend': 'Stable', 'direction': 'stable', 'change': f"{(second_acc - first_acc)*100:.1f}%"}
            
    def get_algorithm_performance(self):
        """Her algoritmanın performansını analiz et"""
        algorithm_stats = {}
        
        for eval in self.evaluation_history['evaluations']:
            if 'algorithm_contributions' in eval:
                for algo, contrib in eval['algorithm_contributions'].items():
                    if algo not in algorithm_stats:
                        algorithm_stats[algo] = {
                            'correct_predictions': 0,
                            'total_predictions': 0,
                            'avg_confidence': []
                        }
                    
                    algorithm_stats[algo]['total_predictions'] += 1
                    if contrib.get('was_correct', False):
                        algorithm_stats[algo]['correct_predictions'] += 1
                    algorithm_stats[algo]['avg_confidence'].append(contrib.get('confidence', 0))
                    
        # İstatistikleri hesapla
        for algo, stats in algorithm_stats.items():
            if stats['total_predictions'] > 0:
                stats['accuracy'] = stats['correct_predictions'] / stats['total_predictions'] * 100
                stats['avg_confidence'] = np.mean(stats['avg_confidence'])
            else:
                stats['accuracy'] = 0
                stats['avg_confidence'] = 0
                
        return algorithm_stats