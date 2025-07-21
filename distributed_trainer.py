"""
Dağıtık Model Eğitimi
Paralel model eğitimi ve asenkron veri işleme
"""
import concurrent.futures
import asyncio
import logging
import time
import numpy as np
from datetime import datetime
import multiprocessing as mp
import threading
import queue

logger = logging.getLogger(__name__)

class DistributedTrainer:
    """
    Dağıtık ve paralel model eğitim sistemi
    """
    
    def __init__(self):
        self.max_workers = min(mp.cpu_count() - 1, 4)  # CPU sayısına göre ayarla
        self.training_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_training = False
        
    def train_all_models_parallel(self, training_data):
        """
        Tüm modelleri paralel olarak eğit
        
        Args:
            training_data: Eğitim verisi
            
        Returns:
            dict: Eğitim sonuçları
        """
        start_time = time.time()
        logger.info(f"Paralel model eğitimi başlatılıyor - {self.max_workers} işçi")
        
        # Model eğitim görevleri
        training_tasks = {
            'xgboost': lambda: self._train_xgboost(training_data),
            'neural_network': lambda: self._train_neural_network(training_data),
            'crf': lambda: self._train_crf(training_data),
            'random_forest': lambda: self._train_random_forest(training_data),
            'gradient_boosting': lambda: self._train_gradient_boosting(training_data)
        }
        
        results = {}
        
        # ProcessPoolExecutor kullanarak paralel eğitim
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Görevleri başlat
            future_to_model = {
                executor.submit(task): model 
                for model, task in training_tasks.items()
            }
            
            # Sonuçları topla
            for future in concurrent.futures.as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    results[model_name] = result
                    logger.info(f"{model_name} eğitimi tamamlandı - Süre: {result['duration']:.2f}s")
                except Exception as e:
                    logger.error(f"{model_name} eğitimi başarısız: {str(e)}")
                    results[model_name] = {'status': 'failed', 'error': str(e)}
                    
        total_duration = time.time() - start_time
        logger.info(f"Tüm modeller eğitildi - Toplam süre: {total_duration:.2f}s")
        
        return {
            'models': results,
            'total_duration': total_duration,
            'parallel_speedup': self._calculate_speedup(results, total_duration)
        }
        
    def _train_xgboost(self, training_data):
        """XGBoost modelini eğit"""
        start = time.time()
        
        try:
            # XGBoost import
            import xgboost as xgb
            from sklearn.model_selection import train_test_split
            
            # Veri hazırlama
            X, y = self._prepare_xgboost_data(training_data)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Model parametreleri
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            
            # Model eğitimi
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            # Performans değerlendirme
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            
            # Model kaydet
            model.save_model('models/xgb_distributed.json')
            
            return {
                'status': 'success',
                'duration': time.time() - start,
                'train_accuracy': train_score,
                'val_accuracy': val_score,
                'best_iteration': model.best_iteration
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'duration': time.time() - start,
                'error': str(e)
            }
            
    def _train_neural_network(self, training_data):
        """Neural Network modelini eğit"""
        start = time.time()
        
        try:
            # TensorFlow import
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            
            # Veri hazırlama
            X, y = self._prepare_nn_data(training_data)
            X_train, X_val, y_train, y_val = self._split_data(X, y)
            
            # Model mimarisi
            model = Sequential([
                Dense(128, activation='relu', input_shape=(X.shape[1],)),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dropout(0.1),
                
                Dense(3, activation='softmax')
            ])
            
            # Model derleme
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Callback'ler
            early_stop = EarlyStopping(patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5)
            
            # Model eğitimi
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            # Model kaydet
            model.save('models/nn_distributed.h5')
            
            return {
                'status': 'success',
                'duration': time.time() - start,
                'train_accuracy': history.history['accuracy'][-1],
                'val_accuracy': history.history['val_accuracy'][-1],
                'epochs_trained': len(history.history['loss'])
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'duration': time.time() - start,
                'error': str(e)
            }
            
    def _train_crf(self, training_data):
        """CRF modelini eğit"""
        start = time.time()
        
        try:
            # sklearn-crfsuite yerine RandomForest kullan
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            
            # Veri hazırlama
            X, y = self._prepare_crf_data(training_data)
            
            # Model parametreleri
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1  # Tüm CPU'ları kullan
            )
            
            # Model eğitimi
            model.fit(X, y)
            
            # Cross validation
            cv_scores = cross_val_score(model, X, y, cv=5)
            
            # Model kaydet
            import pickle
            with open('models/crf_distributed.pkl', 'wb') as f:
                pickle.dump(model, f)
                
            return {
                'status': 'success',
                'duration': time.time() - start,
                'train_accuracy': model.score(X, y),
                'cv_mean_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'duration': time.time() - start,
                'error': str(e)
            }
            
    def _train_random_forest(self, training_data):
        """Random Forest modelini eğit"""
        start = time.time()
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import GridSearchCV
            
            # Veri hazırlama
            X, y = self._prepare_regression_data(training_data)
            
            # Hiperparametre grid'i
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            }
            
            # Grid search ile en iyi parametreleri bul
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                model, param_grid, cv=3, 
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            # En iyi model
            best_model = grid_search.best_estimator_
            
            # Model kaydet
            import pickle
            with open('models/rf_distributed.pkl', 'wb') as f:
                pickle.dump(best_model, f)
                
            return {
                'status': 'success',
                'duration': time.time() - start,
                'best_params': grid_search.best_params_,
                'best_score': -grid_search.best_score_,  # MSE'yi pozitife çevir
                'cv_results': len(grid_search.cv_results_)
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'duration': time.time() - start,
                'error': str(e)
            }
            
    def _train_gradient_boosting(self, training_data):
        """Gradient Boosting modelini eğit"""
        start = time.time()
        
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import train_test_split
            
            # Veri hazırlama
            X, y = self._prepare_classification_data(training_data)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Model parametreleri
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                random_state=42
            )
            
            # Model eğitimi
            model.fit(X_train, y_train)
            
            # Performans değerlendirme
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            
            # Feature importance
            feature_importance = model.feature_importances_
            
            # Model kaydet
            import pickle
            with open('models/gb_distributed.pkl', 'wb') as f:
                pickle.dump(model, f)
                
            return {
                'status': 'success',
                'duration': time.time() - start,
                'train_accuracy': train_score,
                'val_accuracy': val_score,
                'top_features': self._get_top_features(feature_importance)
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'duration': time.time() - start,
                'error': str(e)
            }
            
    def _prepare_xgboost_data(self, training_data):
        """XGBoost için veri hazırla"""
        X = []
        y = []
        
        for match in training_data:
            features = self._extract_features(match)
            label = self._extract_label(match)
            
            X.append(features)
            y.append(label)
            
        return np.array(X), np.array(y)
        
    def _prepare_nn_data(self, training_data):
        """Neural Network için veri hazırla"""
        # XGBoost ile aynı ancak normalizasyon ekle
        X, y = self._prepare_xgboost_data(training_data)
        
        # Normalizasyon
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Scaler'ı kaydet
        import pickle
        with open('models/nn_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
            
        return X, y
        
    def _prepare_crf_data(self, training_data):
        """CRF için veri hazırla"""
        # Sequence yerine normal classification data
        return self._prepare_xgboost_data(training_data)
        
    def _prepare_regression_data(self, training_data):
        """Regresyon için veri hazırla"""
        X = []
        y = []
        
        for match in training_data:
            features = self._extract_features(match)
            # Toplam gol tahmini için
            total_goals = match.get('home_goals', 0) + match.get('away_goals', 0)
            
            X.append(features)
            y.append(total_goals)
            
        return np.array(X), np.array(y)
        
    def _prepare_classification_data(self, training_data):
        """Classification için veri hazırla"""
        return self._prepare_xgboost_data(training_data)
        
    def _extract_features(self, match):
        """Maç verisinden özellik çıkar"""
        features = [
            match.get('home_xg', 1.5),
            match.get('away_xg', 1.5),
            match.get('home_xga', 1.3),
            match.get('away_xga', 1.3),
            match.get('elo_diff', 0),
            match.get('home_form', 2.0),
            match.get('away_form', 2.0),
            match.get('home_avg_goals', 1.5),
            match.get('away_avg_goals', 1.3),
            match.get('home_avg_conceded', 1.3),
            match.get('away_avg_conceded', 1.3),
            match.get('h2h_home_wins', 0.33),
            match.get('h2h_draws', 0.33),
            match.get('h2h_away_wins', 0.33),
            match.get('is_home_advantage', 1)
        ]
        
        return features
        
    def _extract_label(self, match):
        """Maç sonucu etiketi"""
        home_goals = match.get('home_goals', 0)
        away_goals = match.get('away_goals', 0)
        
        if home_goals > away_goals:
            return 0  # HOME_WIN
        elif home_goals < away_goals:
            return 2  # AWAY_WIN
        else:
            return 1  # DRAW
            
    def _split_data(self, X, y, test_size=0.2):
        """Veriyi train/validation olarak böl"""
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
    def _calculate_speedup(self, results, total_duration):
        """Paralel hızlanma oranını hesapla"""
        # Seri toplam süre tahmini
        serial_duration = sum(
            r.get('duration', 0) for r in results.values() 
            if r.get('status') == 'success'
        )
        
        if serial_duration > 0:
            speedup = serial_duration / total_duration
            efficiency = speedup / self.max_workers
            
            return {
                'speedup_ratio': speedup,
                'efficiency': efficiency,
                'serial_estimate': serial_duration,
                'parallel_actual': total_duration
            }
            
        return None
        
    def _get_top_features(self, feature_importance, top_n=5):
        """En önemli özellikleri döndür"""
        feature_names = [
            'home_xg', 'away_xg', 'home_xga', 'away_xga', 'elo_diff',
            'home_form', 'away_form', 'home_avg_goals', 'away_avg_goals',
            'home_avg_conceded', 'away_avg_conceded', 'h2h_home_wins',
            'h2h_draws', 'h2h_away_wins', 'is_home_advantage'
        ]
        
        # Önem derecesine göre sırala
        indices = np.argsort(feature_importance)[::-1][:top_n]
        
        top_features = []
        for i in indices:
            if i < len(feature_names):
                top_features.append({
                    'name': feature_names[i],
                    'importance': feature_importance[i]
                })
                
        return top_features
        
    async def train_models_async(self, training_data):
        """Asenkron model eğitimi"""
        logger.info("Asenkron model eğitimi başlatılıyor")
        
        # Eğitim görevlerini oluştur
        tasks = [
            self._train_model_async('xgboost', training_data),
            self._train_model_async('neural_network', training_data),
            self._train_model_async('crf', training_data),
            self._train_model_async('random_forest', training_data),
            self._train_model_async('gradient_boosting', training_data)
        ]
        
        # Tüm görevleri başlat ve sonuçları bekle
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Sonuçları işle
        processed_results = {}
        for i, (model_name, _) in enumerate([
            ('xgboost', None), ('neural_network', None), 
            ('crf', None), ('random_forest', None), 
            ('gradient_boosting', None)
        ]):
            if isinstance(results[i], Exception):
                processed_results[model_name] = {
                    'status': 'failed',
                    'error': str(results[i])
                }
            else:
                processed_results[model_name] = results[i]
                
        return processed_results
        
    async def _train_model_async(self, model_name, training_data):
        """Tek bir modeli asenkron olarak eğit"""
        loop = asyncio.get_event_loop()
        
        # Model eğitim fonksiyonları
        train_functions = {
            'xgboost': self._train_xgboost,
            'neural_network': self._train_neural_network,
            'crf': self._train_crf,
            'random_forest': self._train_random_forest,
            'gradient_boosting': self._train_gradient_boosting
        }
        
        train_func = train_functions.get(model_name)
        if not train_func:
            return {'status': 'failed', 'error': f'Unknown model: {model_name}'}
            
        # CPU-intensive görevi thread pool'da çalıştır
        result = await loop.run_in_executor(None, train_func, training_data)
        
        return result
        
    def create_training_pipeline(self, data_source):
        """Eğitim pipeline'ı oluştur"""
        pipeline = TrainingPipeline(self)
        
        # Pipeline aşamaları
        pipeline.add_stage('data_loading', self._load_training_data, data_source)
        pipeline.add_stage('data_preprocessing', self._preprocess_data)
        pipeline.add_stage('feature_engineering', self._engineer_features)
        pipeline.add_stage('model_training', self.train_all_models_parallel)
        pipeline.add_stage('model_evaluation', self._evaluate_models)
        pipeline.add_stage('model_selection', self._select_best_models)
        
        return pipeline
        
    def _load_training_data(self, data_source):
        """Eğitim verisini yükle"""
        logger.info(f"Veri yükleniyor: {data_source}")
        
        # Farklı veri kaynaklarını destekle
        if data_source.endswith('.json'):
            import json
            with open(data_source, 'r') as f:
                return json.load(f)
        elif data_source.endswith('.csv'):
            import pandas as pd
            return pd.read_csv(data_source).to_dict('records')
        else:
            # Önbellekten yükle
            return self._load_from_cache()
            
    def _load_from_cache(self):
        """Önbellekten eğitim verisi yükle"""
        import json
        
        training_data = []
        
        # predictions_cache.json dosyasından veri al
        try:
            with open('predictions_cache.json', 'r') as f:
                cache_data = json.load(f)
                
            for match_key, match_data in cache_data.items():
                if 'predictions' in match_data and 'match_info' in match_data:
                    # Eğitim verisi oluştur
                    training_sample = {
                        'home_xg': match_data['predictions'].get('expected_goals', {}).get('home', 1.5),
                        'away_xg': match_data['predictions'].get('expected_goals', {}).get('away', 1.5),
                        'home_goals': 0,  # Gerçek sonuç yoksa varsayılan
                        'away_goals': 0,
                        'elo_diff': 0,
                        'home_form': 2.0,
                        'away_form': 2.0
                    }
                    training_data.append(training_sample)
                    
        except Exception as e:
            logger.error(f"Önbellek yükleme hatası: {e}")
            
        return training_data
        
    def _preprocess_data(self, data):
        """Veri ön işleme"""
        logger.info("Veri ön işleme yapılıyor")
        
        # Eksik değerleri doldur
        processed_data = []
        for sample in data:
            if isinstance(sample, dict):
                # Varsayılan değerlerle doldur
                processed_sample = {
                    'home_xg': sample.get('home_xg', 1.5),
                    'away_xg': sample.get('away_xg', 1.5),
                    'home_xga': sample.get('home_xga', 1.3),
                    'away_xga': sample.get('away_xga', 1.3),
                    'home_goals': sample.get('home_goals', 0),
                    'away_goals': sample.get('away_goals', 0),
                    'elo_diff': sample.get('elo_diff', 0),
                    'home_form': sample.get('home_form', 2.0),
                    'away_form': sample.get('away_form', 2.0),
                    'home_avg_goals': sample.get('home_avg_goals', 1.5),
                    'away_avg_goals': sample.get('away_avg_goals', 1.3),
                    'home_avg_conceded': sample.get('home_avg_conceded', 1.3),
                    'away_avg_conceded': sample.get('away_avg_conceded', 1.3),
                    'h2h_home_wins': sample.get('h2h_home_wins', 0.33),
                    'h2h_draws': sample.get('h2h_draws', 0.33),
                    'h2h_away_wins': sample.get('h2h_away_wins', 0.33),
                    'is_home_advantage': 1
                }
                processed_data.append(processed_sample)
                
        return processed_data
        
    def _engineer_features(self, data):
        """Özellik mühendisliği"""
        logger.info("Özellik mühendisliği yapılıyor")
        
        engineered_data = []
        for sample in data:
            # Yeni özellikler ekle
            enhanced_sample = sample.copy()
            
            # Çapraz özellikler
            enhanced_sample['goal_diff_expected'] = sample['home_xg'] - sample['away_xg']
            enhanced_sample['total_goals_expected'] = sample['home_xg'] + sample['away_xg']
            enhanced_sample['form_diff'] = sample['home_form'] - sample['away_form']
            enhanced_sample['defensive_strength_ratio'] = sample['home_xga'] / max(sample['away_xga'], 0.1)
            enhanced_sample['attacking_strength_ratio'] = sample['home_xg'] / max(sample['away_xg'], 0.1)
            
            # Momentum özellikleri
            enhanced_sample['home_momentum'] = sample['home_form'] * (1 + sample.get('elo_diff', 0) / 1000)
            enhanced_sample['away_momentum'] = sample['away_form'] * (1 - sample.get('elo_diff', 0) / 1000)
            
            engineered_data.append(enhanced_sample)
            
        return engineered_data
        
    def _evaluate_models(self, models):
        """Eğitilen modelleri değerlendir"""
        logger.info("Model değerlendirmesi yapılıyor")
        
        evaluation_results = {}
        
        for model_name, model_result in models['models'].items():
            if model_result.get('status') == 'success':
                evaluation = {
                    'accuracy': model_result.get('val_accuracy', model_result.get('cv_mean_accuracy', 0)),
                    'training_time': model_result.get('duration', 0),
                    'model_size': self._get_model_size(model_name),
                    'prediction_speed': self._test_prediction_speed(model_name)
                }
                
                # Genel skor hesapla
                evaluation['overall_score'] = (
                    evaluation['accuracy'] * 0.5 +
                    (1 / (1 + evaluation['training_time'])) * 0.2 +
                    (1 / (1 + evaluation['model_size'] / 1e6)) * 0.1 +
                    evaluation['prediction_speed'] * 0.2
                )
                
                evaluation_results[model_name] = evaluation
                
        return evaluation_results
        
    def _select_best_models(self, evaluation_results):
        """En iyi modelleri seç"""
        logger.info("En iyi modeller seçiliyor")
        
        # Skorlara göre sırala
        sorted_models = sorted(
            evaluation_results.items(),
            key=lambda x: x[1]['overall_score'],
            reverse=True
        )
        
        # En iyi 3 modeli seç
        best_models = {
            'primary': sorted_models[0][0] if len(sorted_models) > 0 else None,
            'secondary': sorted_models[1][0] if len(sorted_models) > 1 else None,
            'tertiary': sorted_models[2][0] if len(sorted_models) > 2 else None
        }
        
        logger.info(f"Seçilen modeller: {best_models}")
        
        return best_models
        
    def _get_model_size(self, model_name):
        """Model dosya boyutunu al (bytes)"""
        import os
        
        model_files = {
            'xgboost': 'models/xgb_distributed.json',
            'neural_network': 'models/nn_distributed.h5',
            'crf': 'models/crf_distributed.pkl',
            'random_forest': 'models/rf_distributed.pkl',
            'gradient_boosting': 'models/gb_distributed.pkl'
        }
        
        file_path = model_files.get(model_name)
        if file_path and os.path.exists(file_path):
            return os.path.getsize(file_path)
            
        return 1e6  # Varsayılan 1MB
        
    def _test_prediction_speed(self, model_name):
        """Model tahmin hızını test et"""
        # Basit bir test verisi ile tahmin hızını ölç
        test_features = np.random.rand(100, 15)  # 100 örnek, 15 özellik
        
        start = time.time()
        
        # Model tipine göre tahmin yap
        if model_name == 'xgboost':
            # XGBoost tahmin simülasyonu
            time.sleep(0.01)  # 10ms simülasyon
        elif model_name == 'neural_network':
            # NN tahmin simülasyonu
            time.sleep(0.02)  # 20ms simülasyon
        else:
            # Diğer modeller
            time.sleep(0.015)  # 15ms simülasyon
            
        duration = time.time() - start
        
        # Saniyede tahmin sayısı
        predictions_per_second = 100 / duration
        
        # Normalize edilmiş hız skoru (0-1)
        speed_score = min(1.0, predictions_per_second / 10000)
        
        return speed_score


class TrainingPipeline:
    """Eğitim pipeline yönetimi"""
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.stages = []
        self.results = {}
        
    def add_stage(self, name, function, *args):
        """Pipeline'a aşama ekle"""
        self.stages.append({
            'name': name,
            'function': function,
            'args': args
        })
        
    def run(self):
        """Pipeline'ı çalıştır"""
        logger.info("Training pipeline başlatılıyor")
        
        data = None
        for stage in self.stages:
            logger.info(f"Aşama çalıştırılıyor: {stage['name']}")
            
            try:
                if data is None:
                    # İlk aşama
                    data = stage['function'](*stage['args'])
                else:
                    # Önceki aşamanın çıktısını kullan
                    data = stage['function'](data, *stage['args'])
                    
                self.results[stage['name']] = {
                    'status': 'success',
                    'output_type': type(data).__name__
                }
                
            except Exception as e:
                logger.error(f"Aşama hatası {stage['name']}: {str(e)}")
                self.results[stage['name']] = {
                    'status': 'failed',
                    'error': str(e)
                }
                break
                
        return self.results