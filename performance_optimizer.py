"""
Performans Optimizasyon Modülü
Redis benzeri önbellekleme, batch işleme ve optimizasyon teknikleri
"""
import json
import time
import logging
from datetime import datetime, timedelta
import hashlib
import threading
from collections import OrderedDict
import os

logger = logging.getLogger(__name__)

class InMemoryCache:
    """
    Redis benzeri in-memory cache implementasyonu
    """
    
    def __init__(self, max_size=1000, default_ttl=3600):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.lock = threading.Lock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expirations': 0
        }
        
    def get(self, key):
        """Cache'den değer al"""
        with self.lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                
                if expiry is None or datetime.now() < expiry:
                    # LRU için sona taşı
                    self.cache.move_to_end(key)
                    self.stats['hits'] += 1
                    return value
                else:
                    # Süresi dolmuş
                    del self.cache[key]
                    self.stats['expirations'] += 1
                    
            self.stats['misses'] += 1
            return None
            
    def set(self, key, value, ttl=None):
        """Cache'e değer ekle"""
        if ttl is None:
            ttl = self.default_ttl
            
        expiry = datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None
        
        with self.lock:
            # Boyut kontrolü
            if len(self.cache) >= self.max_size and key not in self.cache:
                # En eski öğeyi sil (LRU)
                self.cache.popitem(last=False)
                self.stats['evictions'] += 1
                
            self.cache[key] = (value, expiry)
            
    def delete(self, key):
        """Cache'den sil"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
            
    def clear(self):
        """Tüm cache'i temizle"""
        with self.lock:
            self.cache.clear()
            
    def get_stats(self):
        """Cache istatistiklerini döndür"""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': hit_rate,
                'evictions': self.stats['evictions'],
                'expirations': self.stats['expirations']
            }


class PredictionCache:
    """
    Tahmin sonuçları için özelleştirilmiş cache
    """
    
    def __init__(self):
        self.cache = InMemoryCache(max_size=500, default_ttl=1800)  # 30 dakika
        self.file_cache_path = 'predictions_cache.json'
        self.load_file_cache()
        
    def get_cache_key(self, home_team_id, away_team_id, date):
        """Cache key oluştur"""
        key_str = f"{home_team_id}_{away_team_id}_{date}"
        return hashlib.md5(key_str.encode()).hexdigest()
        
    def get_prediction(self, home_team_id, away_team_id, date):
        """Tahmin al"""
        key = self.get_cache_key(home_team_id, away_team_id, date)
        
        # Önce memory cache'e bak
        result = self.cache.get(key)
        if result:
            logger.info(f"Memory cache hit: {key}")
            return result
            
        # File cache'e bak
        result = self._get_from_file_cache(key)
        if result:
            logger.info(f"File cache hit: {key}")
            # Memory cache'e de ekle
            self.cache.set(key, result, ttl=1800)
            return result
            
        logger.info(f"Cache miss: {key}")
        return None
        
    def set_prediction(self, home_team_id, away_team_id, date, prediction):
        """Tahmin kaydet"""
        key = self.get_cache_key(home_team_id, away_team_id, date)
        
        # Memory cache'e ekle
        self.cache.set(key, prediction, ttl=1800)
        
        # File cache'e de ekle
        self._save_to_file_cache(key, prediction)
        
        logger.info(f"Prediction cached: {key}")
        
    def _get_from_file_cache(self, key):
        """Dosya cache'inden al"""
        try:
            with open(self.file_cache_path, 'r') as f:
                file_cache = json.load(f)
                
            if key in file_cache:
                entry = file_cache[key]
                # Süre kontrolü
                if 'timestamp' in entry:
                    timestamp = datetime.fromisoformat(entry['timestamp'])
                    if datetime.now() - timestamp < timedelta(hours=24):
                        return entry.get('prediction')
                        
        except Exception as e:
            logger.error(f"File cache read error: {e}")
            
        return None
        
    def _save_to_file_cache(self, key, prediction):
        """Dosya cache'ine kaydet"""
        try:
            # Mevcut cache'i yükle
            try:
                with open(self.file_cache_path, 'r') as f:
                    file_cache = json.load(f)
            except:
                file_cache = {}
                
            # Yeni entry ekle
            file_cache[key] = {
                'prediction': prediction,
                'timestamp': datetime.now().isoformat()
            }
            
            # Boyut kontrolü - en fazla 1000 entry
            if len(file_cache) > 1000:
                # En eskileri sil
                sorted_entries = sorted(
                    file_cache.items(),
                    key=lambda x: x[1].get('timestamp', ''),
                    reverse=True
                )
                file_cache = dict(sorted_entries[:1000])
                
            # Kaydet
            with open(self.file_cache_path, 'w') as f:
                json.dump(file_cache, f, indent=2)
                
        except Exception as e:
            logger.error(f"File cache write error: {e}")
            
    def load_file_cache(self):
        """Başlangıçta file cache'i memory'ye yükle"""
        try:
            if os.path.exists(self.file_cache_path):
                with open(self.file_cache_path, 'r') as f:
                    file_cache = json.load(f)
                    
                # Son 24 saatteki entry'leri memory cache'e ekle
                now = datetime.now()
                loaded_count = 0
                
                for key, entry in file_cache.items():
                    if 'timestamp' in entry:
                        timestamp = datetime.fromisoformat(entry['timestamp'])
                        if now - timestamp < timedelta(hours=24):
                            self.cache.set(key, entry['prediction'], ttl=1800)
                            loaded_count += 1
                            
                logger.info(f"Loaded {loaded_count} recent predictions from file cache")
                
        except Exception as e:
            logger.error(f"File cache load error: {e}")
            
    def clear_old_entries(self):
        """Eski cache entry'lerini temizle"""
        try:
            with open(self.file_cache_path, 'r') as f:
                file_cache = json.load(f)
                
            now = datetime.now()
            new_cache = {}
            
            for key, entry in file_cache.items():
                if 'timestamp' in entry:
                    timestamp = datetime.fromisoformat(entry['timestamp'])
                    if now - timestamp < timedelta(days=7):  # 7 günden yeni
                        new_cache[key] = entry
                        
            with open(self.file_cache_path, 'w') as f:
                json.dump(new_cache, f, indent=2)
                
            logger.info(f"Cleaned cache: {len(file_cache)} -> {len(new_cache)} entries")
            
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")


class BatchProcessor:
    """
    Batch tahmin işleme
    """
    
    def __init__(self, batch_size=10):
        self.batch_size = batch_size
        self.processing_queue = []
        self.results = {}
        self.lock = threading.Lock()
        
    def add_to_batch(self, match_id, match_data):
        """Batch'e ekle"""
        with self.lock:
            self.processing_queue.append({
                'match_id': match_id,
                'data': match_data
            })
            
            # Batch doldu mu?
            if len(self.processing_queue) >= self.batch_size:
                return True
                
        return False
        
    def process_batch(self, prediction_function):
        """Batch'i işle"""
        with self.lock:
            if not self.processing_queue:
                return {}
                
            batch = self.processing_queue[:self.batch_size]
            self.processing_queue = self.processing_queue[self.batch_size:]
            
        logger.info(f"Processing batch of {len(batch)} predictions")
        
        results = {}
        start_time = time.time()
        
        for item in batch:
            try:
                prediction = prediction_function(item['data'])
                results[item['match_id']] = {
                    'status': 'success',
                    'prediction': prediction
                }
            except Exception as e:
                logger.error(f"Batch processing error for {item['match_id']}: {e}")
                results[item['match_id']] = {
                    'status': 'error',
                    'error': str(e)
                }
                
        processing_time = time.time() - start_time
        logger.info(f"Batch processed in {processing_time:.2f}s")
        
        return results
        
    def get_queue_size(self):
        """Kuyruktaki öğe sayısı"""
        with self.lock:
            return len(self.processing_queue)


class QueryOptimizer:
    """
    Veritabanı ve API sorgu optimizasyonu
    """
    
    def __init__(self):
        self.query_cache = InMemoryCache(max_size=200, default_ttl=300)  # 5 dakika
        self.bulk_fetch_threshold = 5
        
    def optimize_team_data_fetch(self, team_ids):
        """Takım verisi çekimini optimize et"""
        # Cache'de olanları ayır
        cached_data = {}
        missing_ids = []
        
        for team_id in team_ids:
            cached = self.query_cache.get(f"team_{team_id}")
            if cached:
                cached_data[team_id] = cached
            else:
                missing_ids.append(team_id)
                
        logger.info(f"Cache hits: {len(cached_data)}, misses: {len(missing_ids)}")
        
        # Eksikleri bulk olarak çek
        if missing_ids:
            if len(missing_ids) >= self.bulk_fetch_threshold:
                # Bulk fetch öner
                return {
                    'strategy': 'bulk',
                    'cached': cached_data,
                    'to_fetch': missing_ids
                }
            else:
                # Tek tek çek
                return {
                    'strategy': 'individual',
                    'cached': cached_data,
                    'to_fetch': missing_ids
                }
                
        return {
            'strategy': 'all_cached',
            'cached': cached_data,
            'to_fetch': []
        }
        
    def cache_team_data(self, team_id, data):
        """Takım verisini cache'le"""
        self.query_cache.set(f"team_{team_id}", data, ttl=300)
        
    def create_bulk_query(self, team_ids):
        """Bulk sorgu oluştur"""
        # SQL örneği
        placeholders = ','.join(['?'] * len(team_ids))
        query = f"""
        SELECT * FROM teams 
        WHERE team_id IN ({placeholders})
        """
        
        return {
            'query': query,
            'params': team_ids,
            'optimization': 'bulk_fetch'
        }


class ResponseCompressor:
    """
    API yanıt sıkıştırma
    """
    
    def __init__(self):
        self.compression_threshold = 1024  # 1KB
        
    def compress_response(self, data):
        """Yanıtı sıkıştır"""
        import gzip
        import base64
        
        json_str = json.dumps(data)
        
        if len(json_str) < self.compression_threshold:
            return {
                'compressed': False,
                'data': data
            }
            
        # Gzip ile sıkıştır
        compressed = gzip.compress(json_str.encode())
        
        # Base64 encode
        encoded = base64.b64encode(compressed).decode()
        
        compression_ratio = len(encoded) / len(json_str)
        
        return {
            'compressed': True,
            'data': encoded,
            'original_size': len(json_str),
            'compressed_size': len(encoded),
            'compression_ratio': compression_ratio
        }
        
    def decompress_response(self, compressed_data):
        """Sıkıştırılmış yanıtı aç"""
        import gzip
        import base64
        
        if not compressed_data.get('compressed', False):
            return compressed_data['data']
            
        # Base64 decode
        decoded = base64.b64decode(compressed_data['data'])
        
        # Gzip decompress
        decompressed = gzip.decompress(decoded)
        
        # JSON parse
        return json.loads(decompressed.decode())


class PerformanceMonitor:
    """
    Performans izleme ve raporlama
    """
    
    def __init__(self):
        self.metrics = {
            'api_calls': [],
            'cache_performance': [],
            'prediction_times': [],
            'error_rates': {}
        }
        self.lock = threading.Lock()
        
    def record_api_call(self, endpoint, duration, status='success'):
        """API çağrısını kaydet"""
        with self.lock:
            self.metrics['api_calls'].append({
                'endpoint': endpoint,
                'duration': duration,
                'status': status,
                'timestamp': datetime.now().isoformat()
            })
            
            # Son 1000 kaydı tut
            if len(self.metrics['api_calls']) > 1000:
                self.metrics['api_calls'] = self.metrics['api_calls'][-1000:]
                
    def record_cache_access(self, hit=True):
        """Cache erişimini kaydet"""
        with self.lock:
            self.metrics['cache_performance'].append({
                'hit': hit,
                'timestamp': datetime.now().isoformat()
            })
            
            # Son 1000 kaydı tut
            if len(self.metrics['cache_performance']) > 1000:
                self.metrics['cache_performance'] = self.metrics['cache_performance'][-1000:]
                
    def record_prediction_time(self, algorithm, duration):
        """Tahmin süresini kaydet"""
        with self.lock:
            self.metrics['prediction_times'].append({
                'algorithm': algorithm,
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            })
            
            # Son 1000 kaydı tut
            if len(self.metrics['prediction_times']) > 1000:
                self.metrics['prediction_times'] = self.metrics['prediction_times'][-1000:]
                
    def record_error(self, error_type):
        """Hata kaydet"""
        with self.lock:
            if error_type not in self.metrics['error_rates']:
                self.metrics['error_rates'][error_type] = 0
            self.metrics['error_rates'][error_type] += 1
            
    def get_performance_report(self):
        """Performans raporu oluştur"""
        with self.lock:
            # API performansı
            api_calls = self.metrics['api_calls'][-100:]  # Son 100
            if api_calls:
                api_durations = [c['duration'] for c in api_calls if c['status'] == 'success']
                api_errors = sum(1 for c in api_calls if c['status'] != 'success')
                
                api_stats = {
                    'avg_duration': sum(api_durations) / len(api_durations) if api_durations else 0,
                    'max_duration': max(api_durations) if api_durations else 0,
                    'min_duration': min(api_durations) if api_durations else 0,
                    'error_rate': api_errors / len(api_calls) if api_calls else 0
                }
            else:
                api_stats = None
                
            # Cache performansı
            cache_accesses = self.metrics['cache_performance'][-100:]
            if cache_accesses:
                cache_hits = sum(1 for c in cache_accesses if c['hit'])
                cache_stats = {
                    'hit_rate': cache_hits / len(cache_accesses),
                    'total_accesses': len(cache_accesses)
                }
            else:
                cache_stats = None
                
            # Tahmin performansı
            predictions = self.metrics['prediction_times'][-100:]
            if predictions:
                algo_times = {}
                for p in predictions:
                    algo = p['algorithm']
                    if algo not in algo_times:
                        algo_times[algo] = []
                    algo_times[algo].append(p['duration'])
                    
                prediction_stats = {}
                for algo, times in algo_times.items():
                    prediction_stats[algo] = {
                        'avg_time': sum(times) / len(times),
                        'max_time': max(times),
                        'count': len(times)
                    }
            else:
                prediction_stats = None
                
            return {
                'timestamp': datetime.now().isoformat(),
                'api_performance': api_stats,
                'cache_performance': cache_stats,
                'prediction_performance': prediction_stats,
                'error_summary': dict(self.metrics['error_rates'])
            }


# Global instances
prediction_cache = PredictionCache()
batch_processor = BatchProcessor()
query_optimizer = QueryOptimizer()
response_compressor = ResponseCompressor()
performance_monitor = PerformanceMonitor()


def optimize_prediction_request(home_team_id, away_team_id, date):
    """Tahmin isteğini optimize et"""
    start_time = time.time()
    
    # 1. Cache kontrolü
    cached_prediction = prediction_cache.get_prediction(home_team_id, away_team_id, date)
    if cached_prediction:
        performance_monitor.record_cache_access(hit=True)
        performance_monitor.record_api_call('prediction/cached', time.time() - start_time)
        return cached_prediction
        
    performance_monitor.record_cache_access(hit=False)
    
    # 2. Batch'e ekle
    match_data = {
        'home_team_id': home_team_id,
        'away_team_id': away_team_id,
        'date': date
    }
    
    if batch_processor.add_to_batch(f"{home_team_id}_{away_team_id}_{date}", match_data):
        # Batch dolu, işle
        logger.info("Batch full, processing...")
        
    return None


def get_optimization_stats():
    """Optimizasyon istatistiklerini döndür"""
    cache_stats = prediction_cache.cache.get_stats()
    performance_report = performance_monitor.get_performance_report()
    
    return {
        'cache': cache_stats,
        'performance': performance_report,
        'batch_queue': batch_processor.get_queue_size()
    }