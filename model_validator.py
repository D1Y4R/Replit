"""
Kapsamlı Model Doğrulama Sistemi
K-fold cross validation, temporal validation ve performans analizi
"""
import numpy as np
import logging
from sklearn.model_selection import KFold, TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

logger = logging.getLogger(__name__)

class ComprehensiveValidator:
    """
    Gelişmiş model doğrulama ve performans analizi
    """
    
    def __init__(self):
        self.validation_results = {}
        self.metrics_history = []
        self.validation_config = {
            'k_folds': 5,
            'time_series_splits': 3,
            'stratified': True,
            'test_size': 0.2,
            'random_state': 42
        }
        
    def validate_model(self, model, X, y, model_name='model', validation_type='all'):
        """
        Model doğrulama ana fonksiyonu
        
        Args:
            model: Doğrulanacak model
            X: Özellik matrisi
            y: Hedef değişken
            model_name: Model adı
            validation_type: 'kfold', 'temporal', 'stratified' veya 'all'
            
        Returns:
            dict: Doğrulama sonuçları
        """
        logger.info(f"{model_name} doğrulaması başlatılıyor - Tip: {validation_type}")
        
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'data_size': len(X),
            'features': X.shape[1] if hasattr(X, 'shape') else 0
        }
        
        if validation_type in ['kfold', 'all']:
            results['kfold'] = self._kfold_validation(model, X, y)
            
        if validation_type in ['temporal', 'all']:
            results['temporal'] = self._temporal_validation(model, X, y)
            
        if validation_type in ['stratified', 'all']:
            results['stratified'] = self._stratified_validation(model, X, y)
            
        if validation_type == 'all':
            results['liga_based'] = self._liga_based_validation(model, X, y)
            results['holdout'] = self._holdout_validation(model, X, y)
            
        # Özet metrikleri hesapla
        results['summary'] = self._calculate_summary_metrics(results)
        
        # Sonuçları kaydet
        self.validation_results[model_name] = results
        self._save_validation_results()
        
        return results
        
    def _kfold_validation(self, model, X, y):
        """K-fold cross validation"""
        kf = KFold(n_splits=self.validation_config['k_folds'], 
                   shuffle=True, 
                   random_state=self.validation_config['random_state'])
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Model eğit
            model_clone = self._clone_model(model)
            model_clone.fit(X_train, y_train)
            
            # Tahmin yap
            y_pred = model_clone.predict(X_val)
            
            # Metrikleri hesapla
            fold_metrics = self._calculate_metrics(y_val, y_pred)
            fold_metrics['fold'] = fold + 1
            fold_results.append(fold_metrics)
            
        # Ortalama metrikleri hesapla
        avg_metrics = self._average_fold_metrics(fold_results)
        
        return {
            'fold_results': fold_results,
            'average_metrics': avg_metrics,
            'std_metrics': self._calculate_std_metrics(fold_results)
        }
        
    def _temporal_validation(self, model, X, y):
        """Zaman serisi tabanlı doğrulama"""
        tscv = TimeSeriesSplit(n_splits=self.validation_config['time_series_splits'])
        
        temporal_results = []
        
        for split, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Model eğit
            model_clone = self._clone_model(model)
            model_clone.fit(X_train, y_train)
            
            # Tahmin yap
            y_pred = model_clone.predict(X_val)
            
            # Metrikleri hesapla
            split_metrics = self._calculate_metrics(y_val, y_pred)
            split_metrics['split'] = split + 1
            split_metrics['train_size'] = len(train_idx)
            split_metrics['val_size'] = len(val_idx)
            temporal_results.append(split_metrics)
            
        return {
            'split_results': temporal_results,
            'performance_trend': self._analyze_performance_trend(temporal_results)
        }
        
    def _stratified_validation(self, model, X, y):
        """Sınıf dengesini koruyan doğrulama"""
        skf = StratifiedKFold(n_splits=self.validation_config['k_folds'],
                             shuffle=True,
                             random_state=self.validation_config['random_state'])
        
        stratified_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Sınıf dağılımını kontrol et
            train_dist = self._get_class_distribution(y_train)
            val_dist = self._get_class_distribution(y_val)
            
            # Model eğit
            model_clone = self._clone_model(model)
            model_clone.fit(X_train, y_train)
            
            # Tahmin yap
            y_pred = model_clone.predict(X_val)
            
            # Metrikleri hesapla
            fold_metrics = self._calculate_metrics(y_val, y_pred)
            fold_metrics['fold'] = fold + 1
            fold_metrics['train_distribution'] = train_dist
            fold_metrics['val_distribution'] = val_dist
            stratified_results.append(fold_metrics)
            
        return {
            'fold_results': stratified_results,
            'class_balance_preserved': self._check_class_balance(stratified_results)
        }
        
    def _liga_based_validation(self, model, X, y, liga_info=None):
        """Liga bazlı doğrulama - farklı liglerde test"""
        # Liga bilgisi yoksa rastgele grupla
        if liga_info is None:
            # Veriyi 3 gruba böl (major, minor, other)
            n_samples = len(X)
            liga_groups = np.random.choice(['major', 'minor', 'other'], n_samples)
        else:
            liga_groups = liga_info
            
        unique_ligas = np.unique(liga_groups)
        liga_results = {}
        
        for test_liga in unique_ligas:
            # Test ligasını ayır
            test_mask = liga_groups == test_liga
            train_mask = ~test_mask
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
            if len(X_test) < 10:  # Çok az veri varsa atla
                continue
                
            # Model eğit
            model_clone = self._clone_model(model)
            model_clone.fit(X_train, y_train)
            
            # Tahmin yap
            y_pred = model_clone.predict(X_test)
            
            # Metrikleri hesapla
            liga_metrics = self._calculate_metrics(y_test, y_pred)
            liga_metrics['test_size'] = len(X_test)
            liga_metrics['train_size'] = len(X_train)
            
            liga_results[test_liga] = liga_metrics
            
        return {
            'liga_performance': liga_results,
            'cross_liga_generalization': self._analyze_cross_liga_performance(liga_results)
        }
        
    def _holdout_validation(self, model, X, y):
        """Basit holdout doğrulama"""
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.validation_config['test_size'],
            random_state=self.validation_config['random_state'],
            stratify=y if self._is_classification(y) else None
        )
        
        # Model eğit
        model_clone = self._clone_model(model)
        model_clone.fit(X_train, y_train)
        
        # Tahmin yap
        y_pred_train = model_clone.predict(X_train)
        y_pred_test = model_clone.predict(X_test)
        
        # Metrikleri hesapla
        train_metrics = self._calculate_metrics(y_train, y_pred_train)
        test_metrics = self._calculate_metrics(y_test, y_pred_test)
        
        # Overfitting kontrolü
        overfitting_score = self._check_overfitting(train_metrics, test_metrics)
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'overfitting_analysis': overfitting_score
        }
        
    def _calculate_metrics(self, y_true, y_pred):
        """Metrik hesaplama"""
        metrics = {}
        
        if self._is_classification(y_true):
            # Classification metrikleri
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Per-class metrikleri
            if len(np.unique(y_true)) <= 10:  # Çok fazla sınıf yoksa
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                metrics['per_class_metrics'] = report
                
        else:
            # Regression metrikleri
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # Yüzde hata
            mask = y_true != 0
            if np.any(mask):
                metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                metrics['mape'] = None
                
        return metrics
        
    def _clone_model(self, model):
        """Model klonlama"""
        from sklearn.base import clone
        try:
            return clone(model)
        except:
            # Clone başarısız olursa aynı tipte yeni model oluştur
            return type(model)(**model.get_params())
            
    def _is_classification(self, y):
        """Classification mı regression mı kontrol et"""
        # Unique değer sayısına ve tipine bak
        unique_values = np.unique(y)
        
        # Integer ve az sayıda unique değer varsa classification
        if len(unique_values) < 20 and np.all(y == y.astype(int)):
            return True
            
        return False
        
    def _get_class_distribution(self, y):
        """Sınıf dağılımını hesapla"""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        distribution = {}
        for cls, count in zip(unique, counts):
            distribution[str(cls)] = {
                'count': int(count),
                'percentage': float(count / total * 100)
            }
            
        return distribution
        
    def _average_fold_metrics(self, fold_results):
        """Fold metriklerinin ortalamasını al"""
        avg_metrics = {}
        
        # Tüm metrikleri topla
        all_keys = set()
        for result in fold_results:
            all_keys.update(result.keys())
            
        # Her metrik için ortalama hesapla
        for key in all_keys:
            if key in ['fold', 'confusion_matrix', 'per_class_metrics']:
                continue
                
            values = []
            for result in fold_results:
                if key in result and isinstance(result[key], (int, float)):
                    values.append(result[key])
                    
            if values:
                avg_metrics[key] = np.mean(values)
                
        return avg_metrics
        
    def _calculate_std_metrics(self, fold_results):
        """Fold metriklerinin standart sapmasını hesapla"""
        std_metrics = {}
        
        # Sayısal metrikleri bul
        numeric_keys = set()
        for result in fold_results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and key not in ['fold']:
                    numeric_keys.add(key)
                    
        # Her metrik için std hesapla
        for key in numeric_keys:
            values = [r.get(key, 0) for r in fold_results]
            std_metrics[key] = np.std(values)
            
        return std_metrics
        
    def _analyze_performance_trend(self, temporal_results):
        """Zamansal performans trendini analiz et"""
        if not temporal_results:
            return None
            
        # Accuracy veya ana metrik üzerinden trend
        main_metric = 'accuracy' if 'accuracy' in temporal_results[0] else 'rmse'
        
        values = [r.get(main_metric, 0) for r in temporal_results]
        splits = list(range(1, len(values) + 1))
        
        # Linear regression ile trend
        if len(values) >= 2:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(splits, values)
            
            trend = {
                'slope': slope,
                'direction': 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable',
                'r_squared': r_value ** 2,
                'significant': p_value < 0.05,
                'metric_values': values
            }
        else:
            trend = {
                'direction': 'insufficient_data',
                'metric_values': values
            }
            
        return trend
        
    def _check_class_balance(self, stratified_results):
        """Sınıf dengesinin korunup korunmadığını kontrol et"""
        # İlk fold'un train dağılımını referans al
        if not stratified_results:
            return False
            
        reference_dist = stratified_results[0].get('train_distribution', {})
        
        balance_scores = []
        
        for result in stratified_results[1:]:
            train_dist = result.get('train_distribution', {})
            val_dist = result.get('val_distribution', {})
            
            # Dağılım farklarını hesapla
            train_diff = self._calculate_distribution_difference(reference_dist, train_dist)
            val_diff = self._calculate_distribution_difference(reference_dist, val_dist)
            
            balance_scores.append(max(train_diff, val_diff))
            
        # Ortalama fark %5'ten azsa dengeli kabul et
        avg_diff = np.mean(balance_scores) if balance_scores else 0
        
        return {
            'balanced': avg_diff < 5.0,
            'average_deviation': avg_diff,
            'max_deviation': max(balance_scores) if balance_scores else 0
        }
        
    def _calculate_distribution_difference(self, dist1, dist2):
        """İki dağılım arasındaki farkı hesapla"""
        if not dist1 or not dist2:
            return 0.0
            
        total_diff = 0.0
        count = 0
        
        for cls in dist1:
            if cls in dist2:
                diff = abs(dist1[cls]['percentage'] - dist2[cls]['percentage'])
                total_diff += diff
                count += 1
                
        return total_diff / count if count > 0 else 0.0
        
    def _analyze_cross_liga_performance(self, liga_results):
        """Ligler arası performans analizini yap"""
        if not liga_results:
            return None
            
        # Her liga için ana metriği al
        liga_scores = {}
        for liga, metrics in liga_results.items():
            main_metric = metrics.get('accuracy', metrics.get('rmse', 0))
            liga_scores[liga] = main_metric
            
        # En iyi ve en kötü performans
        if liga_scores:
            best_liga = max(liga_scores, key=liga_scores.get)
            worst_liga = min(liga_scores, key=liga_scores.get)
            
            performance_range = liga_scores[best_liga] - liga_scores[worst_liga]
            
            return {
                'best_performing_liga': best_liga,
                'worst_performing_liga': worst_liga,
                'performance_range': performance_range,
                'generalization_score': 1 - (performance_range / max(liga_scores[best_liga], 0.01)),
                'liga_scores': liga_scores
            }
            
        return None
        
    def _check_overfitting(self, train_metrics, test_metrics):
        """Overfitting kontrolü"""
        # Ana metriği belirle
        if 'accuracy' in train_metrics:
            train_score = train_metrics['accuracy']
            test_score = test_metrics['accuracy']
            metric_name = 'accuracy'
        elif 'rmse' in train_metrics:
            train_score = -train_metrics['rmse']  # Negatif çünkü düşük daha iyi
            test_score = -test_metrics['rmse']
            metric_name = 'rmse'
        else:
            return {'status': 'unable_to_check'}
            
        # Performans farkı
        performance_gap = train_score - test_score
        relative_gap = performance_gap / abs(train_score) if train_score != 0 else 0
        
        # Overfitting derecesi
        if relative_gap < 0.05:
            severity = 'none'
        elif relative_gap < 0.1:
            severity = 'mild'
        elif relative_gap < 0.2:
            severity = 'moderate'
        else:
            severity = 'severe'
            
        return {
            'metric': metric_name,
            'train_performance': train_score,
            'test_performance': test_score,
            'gap': performance_gap,
            'relative_gap': relative_gap * 100,
            'severity': severity,
            'recommendation': self._get_overfitting_recommendation(severity)
        }
        
    def _get_overfitting_recommendation(self, severity):
        """Overfitting önerileri"""
        recommendations = {
            'none': 'Model iyi genelleme yapıyor.',
            'mild': 'Hafif overfitting var. Regularizasyon artırılabilir.',
            'moderate': 'Orta düzey overfitting. Dropout veya L2 regularizasyon ekleyin.',
            'severe': 'Ciddi overfitting! Model karmaşıklığını azaltın veya daha fazla veri toplayın.'
        }
        
        return recommendations.get(severity, 'Overfitting durumu belirsiz.')
        
    def _calculate_summary_metrics(self, results):
        """Özet metrikleri hesapla"""
        summary = {
            'validation_types': list(results.keys()),
            'timestamp': results.get('timestamp'),
            'data_size': results.get('data_size'),
            'feature_count': results.get('features')
        }
        
        # K-fold özeti
        if 'kfold' in results:
            kfold_avg = results['kfold'].get('average_metrics', {})
            summary['kfold_performance'] = {
                'accuracy': kfold_avg.get('accuracy', 0),
                'f1_score': kfold_avg.get('f1', 0)
            }
            
        # Temporal özeti
        if 'temporal' in results:
            trend = results['temporal'].get('performance_trend', {})
            summary['temporal_trend'] = trend.get('direction', 'unknown')
            
        # Liga bazlı özet
        if 'liga_based' in results:
            cross_liga = results['liga_based'].get('cross_liga_generalization', {})
            summary['generalization_score'] = cross_liga.get('generalization_score', 0)
            
        # Overfitting özeti
        if 'holdout' in results:
            overfitting = results['holdout'].get('overfitting_analysis', {})
            summary['overfitting_severity'] = overfitting.get('severity', 'unknown')
            
        return summary
        
    def _save_validation_results(self):
        """Doğrulama sonuçlarını kaydet"""
        output_file = 'validation_results.json'
        
        try:
            with open(output_file, 'w') as f:
                json.dump(self.validation_results, f, indent=2, default=str)
            logger.info(f"Doğrulama sonuçları kaydedildi: {output_file}")
        except Exception as e:
            logger.error(f"Sonuç kaydetme hatası: {e}")
            
    def create_validation_report(self, model_name=None):
        """Detaylı doğrulama raporu oluştur"""
        if model_name:
            results = self.validation_results.get(model_name)
            if not results:
                return f"No validation results found for {model_name}"
        else:
            results = self.validation_results
            
        report = {
            'title': 'Model Validation Report',
            'generated_at': datetime.now().isoformat(),
            'models_validated': list(self.validation_results.keys()) if not model_name else [model_name],
            'detailed_results': results,
            'recommendations': self._generate_recommendations(results)
        }
        
        # Raporu kaydet
        report_file = f'validation_report_{model_name or "all"}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        return report
        
    def _generate_recommendations(self, results):
        """Doğrulama sonuçlarına göre öneriler üret"""
        recommendations = []
        
        # Model bazlı öneriler
        if isinstance(results, dict) and 'summary' in results:
            summary = results['summary']
            
            # Overfitting kontrolü
            if summary.get('overfitting_severity') in ['moderate', 'severe']:
                recommendations.append({
                    'type': 'overfitting',
                    'priority': 'high',
                    'suggestion': 'Model aşırı öğrenme gösteriyor. Regularizasyon ekleyin veya model karmaşıklığını azaltın.'
                })
                
            # Temporal trend kontrolü
            if summary.get('temporal_trend') == 'declining':
                recommendations.append({
                    'type': 'temporal_degradation',
                    'priority': 'medium',
                    'suggestion': 'Model performansı zamanla düşüyor. Concept drift olabilir, modeli düzenli güncelleyin.'
                })
                
            # Generalization kontrolü
            if summary.get('generalization_score', 1) < 0.8:
                recommendations.append({
                    'type': 'poor_generalization',
                    'priority': 'high',
                    'suggestion': 'Model farklı veri gruplarında tutarsız performans gösteriyor. Daha çeşitli veri toplayın.'
                })
                
        return recommendations
        
    def plot_validation_results(self, model_name, save_path='validation_plots/'):
        """Doğrulama sonuçlarını görselleştir"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        results = self.validation_results.get(model_name)
        if not results:
            logger.error(f"No results found for {model_name}")
            return
            
        # K-fold sonuçları grafiği
        if 'kfold' in results:
            self._plot_kfold_results(results['kfold'], model_name, save_path)
            
        # Temporal trend grafiği
        if 'temporal' in results:
            self._plot_temporal_trend(results['temporal'], model_name, save_path)
            
        # Confusion matrix
        if 'holdout' in results:
            self._plot_confusion_matrix(results['holdout'], model_name, save_path)
            
    def _plot_kfold_results(self, kfold_results, model_name, save_path):
        """K-fold sonuçlarını görselleştir"""
        fold_results = kfold_results['fold_results']
        
        # Metrik isimlerini al
        metric_names = [k for k in fold_results[0].keys() 
                       if k not in ['fold', 'confusion_matrix', 'per_class_metrics']]
        
        # Her metrik için plot
        fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 4 * len(metric_names)))
        if len(metric_names) == 1:
            axes = [axes]
            
        for i, metric in enumerate(metric_names):
            values = [r.get(metric, 0) for r in fold_results]
            folds = [r['fold'] for r in fold_results]
            
            axes[i].bar(folds, values)
            axes[i].set_xlabel('Fold')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_title(f'{metric.capitalize()} by Fold')
            
            # Ortalama çizgisi
            avg_value = np.mean(values)
            axes[i].axhline(y=avg_value, color='r', linestyle='--', 
                           label=f'Average: {avg_value:.3f}')
            axes[i].legend()
            
        plt.tight_layout()
        plt.savefig(f'{save_path}{model_name}_kfold_results.png')
        plt.close()
        
    def _plot_temporal_trend(self, temporal_results, model_name, save_path):
        """Temporal trend grafiği"""
        trend = temporal_results['performance_trend']
        
        if 'metric_values' not in trend:
            return
            
        values = trend['metric_values']
        splits = list(range(1, len(values) + 1))
        
        plt.figure(figsize=(10, 6))
        plt.plot(splits, values, 'bo-', markersize=8)
        
        # Trend çizgisi
        if trend.get('slope') is not None:
            x = np.array(splits)
            y = trend['slope'] * x + (values[0] - trend['slope'])
            plt.plot(x, y, 'r--', label=f'Trend: {trend["direction"]}')
            
        plt.xlabel('Time Split')
        plt.ylabel('Performance Metric')
        plt.title(f'{model_name} - Temporal Performance Trend')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f'{save_path}{model_name}_temporal_trend.png')
        plt.close()
        
    def _plot_confusion_matrix(self, holdout_results, model_name, save_path):
        """Confusion matrix görselleştirme"""
        test_metrics = holdout_results.get('test_metrics', {})
        cm = test_metrics.get('confusion_matrix')
        
        if cm is None:
            return
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{model_name} - Confusion Matrix')
        
        plt.savefig(f'{save_path}{model_name}_confusion_matrix.png')
        plt.close()