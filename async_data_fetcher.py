"""
Asenkron Veri Çekme Modülü
Paralel API çağrıları ve asenkron veri işleme
"""
import asyncio
import aiohttp
import logging
import time
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

class AsyncDataFetcher:
    """
    Asenkron veri çekme ve işleme sistemi
    """
    
    def __init__(self, max_concurrent_requests=10):
        self.max_concurrent_requests = max_concurrent_requests
        self.session = None
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.request_stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'total_time': 0
        }
        # Stats için ek özellikler
        self.stats = {
            'h2h_fetches': 0,
            'team_fetches': 0,
            'match_fetches': 0
        }
        
    async def __aenter__(self):
        """Context manager giriş"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager çıkış"""
        if self.session:
            await self.session.close()
            
    async def fetch_single(self, url, params=None, headers=None):
        """Tek bir URL'den veri çek"""
        async with self.semaphore:
            start_time = time.time()
            
            try:
                async with self.session.get(url, params=params, headers=headers) as response:
                    self.request_stats['total'] += 1
                    
                    if response.status == 200:
                        data = await response.json()
                        self.request_stats['successful'] += 1
                        
                        elapsed = time.time() - start_time
                        self.request_stats['total_time'] += elapsed
                        
                        logger.debug(f"Fetched {url} in {elapsed:.2f}s")
                        
                        return {
                            'status': 'success',
                            'data': data,
                            'url': str(response.url),
                            'duration': elapsed
                        }
                    else:
                        self.request_stats['failed'] += 1
                        logger.error(f"HTTP {response.status} for {url}")
                        
                        return {
                            'status': 'error',
                            'error': f'HTTP {response.status}',
                            'url': str(response.url)
                        }
                        
            except asyncio.TimeoutError:
                self.request_stats['failed'] += 1
                logger.error(f"Timeout for {url}")
                return {
                    'status': 'error',
                    'error': 'Timeout',
                    'url': url
                }
                
            except Exception as e:
                self.request_stats['failed'] += 1
                logger.error(f"Error fetching {url}: {str(e)}")
                return {
                    'status': 'error',
                    'error': str(e),
                    'url': url
                }
                
    async def fetch_multiple(self, urls, params_list=None, headers=None):
        """Birden fazla URL'den paralel veri çek"""
        if params_list is None:
            params_list = [None] * len(urls)
            
        tasks = [
            self.fetch_single(url, params, headers)
            for url, params in zip(urls, params_list)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Exception'ları handle et
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'status': 'error',
                    'error': str(result),
                    'url': urls[i]
                })
            else:
                processed_results.append(result)
                
        return processed_results
        
    async def fetch_team_data_batch(self, team_ids, api_key):
        """Takım verilerini batch olarak çek"""
        base_url = "https://apiv3.apifootball.com/"
        
        # Her takım için URL ve parametreler oluştur
        urls = [base_url] * len(team_ids)
        params_list = [
            {
                'action': 'get_teams',
                'team_id': team_id,
                'APIkey': api_key
            }
            for team_id in team_ids
        ]
        
        logger.info(f"Fetching data for {len(team_ids)} teams asynchronously")
        
        results = await self.fetch_multiple(urls, params_list)
        
        # Sonuçları team_id ile eşle
        team_data = {}
        for team_id, result in zip(team_ids, results):
            if result['status'] == 'success':
                team_data[team_id] = result['data']
            else:
                team_data[team_id] = None
                logger.error(f"Failed to fetch team {team_id}: {result.get('error')}")
                
        return team_data
        
    async def fetch_matches_parallel(self, league_ids, date, api_key):
        """Liglerin maçlarını paralel olarak çek"""
        base_url = "https://apiv3.apifootball.com/"
        
        # Her lig için parametreler
        urls = [base_url] * len(league_ids)
        params_list = [
            {
                'action': 'get_events',
                'league_id': league_id,
                'from': date,
                'to': date,
                'APIkey': api_key
            }
            for league_id in league_ids
        ]
        
        logger.info(f"Fetching matches for {len(league_ids)} leagues on {date}")
        
        results = await self.fetch_multiple(urls, params_list)
        
        # Tüm maçları birleştir
        all_matches = []
        for league_id, result in zip(league_ids, results):
            if result['status'] == 'success' and isinstance(result['data'], list):
                for match in result['data']:
                    match['league_id'] = league_id  # Lig ID'sini ekle
                    all_matches.append(match)
                    
        logger.info(f"Fetched total {len(all_matches)} matches")
        return all_matches
        
    async def fetch_h2h_data(self, home_team_id, away_team_id, api_key, home_team_name=None, away_team_name=None):
        """Head-to-head verilerini çek - önce takım isimleriyle dene, başarısız olursa ID'lerle dene"""
        url = "https://apiv3.apifootball.com/"
        
        # Önce takım isimleriyle dene
        if home_team_name and away_team_name:
            params = {
                'action': 'get_H2H',
                'firstTeam': home_team_name,
                'secondTeam': away_team_name,
                'APIkey': api_key
            }
            
            logger.info(f"H2H verisi çekiliyor (isimle): {home_team_name} vs {away_team_name}")
            result = await self.fetch_single(url, params)
            
            if result['status'] == 'success' and result.get('data'):
                # H2H verisi başarıyla alındı
                h2h_data = result['data']
                if 'firstTeam_VS_secondTeam' in h2h_data and h2h_data['firstTeam_VS_secondTeam']:
                    logger.info(f"H2H verisi başarıyla alındı (isimle): {len(h2h_data['firstTeam_VS_secondTeam'])} maç bulundu")
                    self.stats['h2h_fetches'] += 1
                    return {
                        'success': True,
                        'response': {
                            'total_matches': len(h2h_data['firstTeam_VS_secondTeam']),
                            'matches': h2h_data['firstTeam_VS_secondTeam']
                        }
                    }
        
        # İsimle başarısız olduysa veya isim yoksa ID'lerle dene
        params = {
            'action': 'get_H2H',
            'firstTeamId': home_team_id,
            'secondTeamId': away_team_id,
            'APIkey': api_key
        }
        
        logger.info(f"H2H verisi çekiliyor (ID ile): {home_team_id} vs {away_team_id}")
        result = await self.fetch_single(url, params)
        
        if result['status'] == 'success' and result.get('data'):
            h2h_data = result['data']
            if 'firstTeam_VS_secondTeam' in h2h_data and h2h_data['firstTeam_VS_secondTeam']:
                logger.info(f"H2H verisi başarıyla alındı (ID ile): {len(h2h_data['firstTeam_VS_secondTeam'])} maç bulundu")
                self.stats['h2h_fetches'] += 1
                return {
                    'success': True,
                    'response': {
                        'total_matches': len(h2h_data['firstTeam_VS_secondTeam']),
                        'matches': h2h_data['firstTeam_VS_secondTeam']
                    }
                }
        
        logger.error(f"Failed to fetch H2H data: {result.get('error', 'No data')}")
        return {'success': False, 'response': None}
            
    def get_stats(self):
        """İstatistikleri döndür"""
        avg_time = (self.request_stats['total_time'] / self.request_stats['total'] 
                   if self.request_stats['total'] > 0 else 0)
        
        return {
            'total_requests': self.request_stats['total'],
            'successful': self.request_stats['successful'],
            'failed': self.request_stats['failed'],
            'success_rate': (self.request_stats['successful'] / self.request_stats['total'] * 100
                           if self.request_stats['total'] > 0 else 0),
            'avg_request_time': avg_time
        }


class AsyncPredictionProcessor:
    """
    Asenkron tahmin işleme
    """
    
    def __init__(self, max_workers=5):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_times = []
        
    async def process_predictions_async(self, matches, prediction_function):
        """Tahminleri asenkron olarak işle"""
        logger.info(f"Processing {len(matches)} predictions asynchronously")
        
        loop = asyncio.get_event_loop()
        
        # CPU-intensive tahmin işlemlerini thread pool'da çalıştır
        tasks = []
        for match in matches:
            task = loop.run_in_executor(
                self.executor,
                prediction_function,
                match
            )
            tasks.append(task)
            
        # Tüm tahminleri bekle
        start_time = time.time()
        predictions = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Sonuçları işle
        results = []
        successful = 0
        
        for match, prediction in zip(matches, predictions):
            if isinstance(prediction, Exception):
                logger.error(f"Prediction error for match {match.get('match_id')}: {str(prediction)}")
                results.append({
                    'match_id': match.get('match_id'),
                    'status': 'error',
                    'error': str(prediction)
                })
            else:
                results.append({
                    'match_id': match.get('match_id'),
                    'status': 'success',
                    'prediction': prediction
                })
                successful += 1
                
        self.processing_times.append(total_time)
        
        logger.info(f"Processed {successful}/{len(matches)} predictions in {total_time:.2f}s")
        logger.info(f"Average time per prediction: {total_time/len(matches):.3f}s")
        
        return results
        
    def get_performance_stats(self):
        """Performans istatistiklerini döndür"""
        if not self.processing_times:
            return None
            
        return {
            'total_batches': len(self.processing_times),
            'avg_batch_time': sum(self.processing_times) / len(self.processing_times),
            'min_batch_time': min(self.processing_times),
            'max_batch_time': max(self.processing_times)
        }


class AsyncDatabaseOperations:
    """
    Asenkron veritabanı işlemleri (simüle edilmiş)
    """
    
    def __init__(self):
        self.connection_pool_size = 10
        self.semaphore = asyncio.Semaphore(self.connection_pool_size)
        
    async def fetch_team_stats_async(self, team_ids):
        """Takım istatistiklerini asenkron olarak çek"""
        async with self.semaphore:
            # Veritabanı sorgusunu simüle et
            await asyncio.sleep(0.1)  # DB latency simülasyonu
            
            # Örnek veri döndür
            stats = {}
            for team_id in team_ids:
                stats[team_id] = {
                    'avg_goals_scored': 1.5 + (team_id % 10) * 0.1,
                    'avg_goals_conceded': 1.3 + (team_id % 7) * 0.1,
                    'home_win_rate': 0.45 + (team_id % 5) * 0.02,
                    'away_win_rate': 0.35 + (team_id % 4) * 0.02,
                    'last_updated': datetime.now().isoformat()
                }
                
            return stats
            
    async def save_predictions_batch(self, predictions):
        """Tahminleri batch olarak kaydet"""
        async with self.semaphore:
            # Batch insert simülasyonu
            await asyncio.sleep(0.05 * len(predictions))
            
            # Başarılı kayıt sayısını döndür
            return len(predictions)
            
    async def update_cache_async(self, cache_entries):
        """Cache'i asenkron güncelle"""
        tasks = []
        
        for key, value in cache_entries.items():
            task = self._update_single_cache(key, value)
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        
        successful = sum(1 for r in results if r)
        return {
            'total': len(cache_entries),
            'successful': successful,
            'failed': len(cache_entries) - successful
        }
        
    async def _update_single_cache(self, key, value):
        """Tek bir cache entry'yi güncelle"""
        async with self.semaphore:
            try:
                # Cache güncelleme simülasyonu
                await asyncio.sleep(0.01)
                
                # %95 başarı oranı
                import random
                return random.random() < 0.95
                
            except Exception as e:
                logger.error(f"Cache update error for {key}: {e}")
                return False


class AsyncWorkflowOrchestrator:
    """
    Asenkron iş akışı yönetici
    """
    
    def __init__(self):
        self.data_fetcher = AsyncDataFetcher()
        self.prediction_processor = AsyncPredictionProcessor()
        self.db_ops = AsyncDatabaseOperations()
        
    async def process_match_predictions_workflow(self, match_ids, api_key, prediction_function):
        """
        Komple tahmin iş akışını asenkron olarak yönet
        
        1. Takım verilerini paralel çek
        2. İstatistikleri asenkron yükle
        3. Tahminleri paralel hesapla
        4. Sonuçları kaydet
        """
        workflow_start = time.time()
        
        async with self.data_fetcher:
            # Adım 1: Takım ID'lerini çıkar
            team_ids = set()
            for match_id in match_ids:
                # match_id formatı: "home_away" varsayımı
                home_id, away_id = match_id.split('_')[:2]
                team_ids.add(int(home_id))
                team_ids.add(int(away_id))
                
            logger.info(f"Step 1: Fetching data for {len(team_ids)} unique teams")
            
            # Adım 2: Takım verilerini ve istatistiklerini paralel çek
            team_data_task = self.data_fetcher.fetch_team_data_batch(list(team_ids), api_key)
            team_stats_task = self.db_ops.fetch_team_stats_async(list(team_ids))
            
            team_data, team_stats = await asyncio.gather(team_data_task, team_stats_task)
            
            logger.info("Step 2: Team data and stats fetched")
            
            # Adım 3: Maç verilerini hazırla
            matches = []
            for match_id in match_ids:
                home_id, away_id = match_id.split('_')[:2]
                home_id, away_id = int(home_id), int(away_id)
                
                match_data = {
                    'match_id': match_id,
                    'home_team_id': home_id,
                    'away_team_id': away_id,
                    'home_team_data': team_data.get(home_id, {}),
                    'away_team_data': team_data.get(away_id, {}),
                    'home_team_stats': team_stats.get(home_id, {}),
                    'away_team_stats': team_stats.get(away_id, {})
                }
                matches.append(match_data)
                
            # Adım 4: Tahminleri paralel hesapla
            logger.info(f"Step 3: Processing {len(matches)} predictions")
            predictions = await self.prediction_processor.process_predictions_async(
                matches, prediction_function
            )
            
            # Adım 5: Sonuçları kaydet
            logger.info("Step 4: Saving predictions")
            
            # Cache güncellemeleri hazırla
            cache_updates = {}
            successful_predictions = []
            
            for pred in predictions:
                if pred['status'] == 'success':
                    cache_key = f"prediction_{pred['match_id']}"
                    cache_updates[cache_key] = pred['prediction']
                    successful_predictions.append(pred)
                    
            # Paralel kaydet
            save_task = self.db_ops.save_predictions_batch(successful_predictions)
            cache_task = self.db_ops.update_cache_async(cache_updates)
            
            save_result, cache_result = await asyncio.gather(save_task, cache_task)
            
            workflow_duration = time.time() - workflow_start
            
            logger.info(f"Workflow completed in {workflow_duration:.2f}s")
            logger.info(f"Saved {save_result} predictions, updated {cache_result['successful']} cache entries")
            
            return {
                'total_matches': len(match_ids),
                'successful_predictions': len(successful_predictions),
                'failed_predictions': len(predictions) - len(successful_predictions),
                'workflow_duration': workflow_duration,
                'steps': {
                    'data_fetch': team_data is not None,
                    'stats_load': team_stats is not None,
                    'predictions': len(predictions),
                    'saved': save_result,
                    'cached': cache_result['successful']
                }
            }


# Yardımcı fonksiyonlar
def run_async_workflow(match_ids, api_key, prediction_function):
    """Asenkron iş akışını senkron koddan çalıştır"""
    orchestrator = AsyncWorkflowOrchestrator()
    
    # Event loop oluştur veya mevcut olanı al
    try:
        loop = asyncio.get_running_loop()
        # Zaten bir loop içindeyiz
        task = orchestrator.process_match_predictions_workflow(
            match_ids, api_key, prediction_function
        )
        return loop.create_task(task)
    except RuntimeError:
        # Loop yoksa yeni oluştur
        return asyncio.run(
            orchestrator.process_match_predictions_workflow(
                match_ids, api_key, prediction_function
            )
        )


async def fetch_multiple_endpoints_async(endpoints, api_key):
    """Birden fazla API endpoint'inden veri çek"""
    async with AsyncDataFetcher() as fetcher:
        # Her endpoint için URL ve parametreler
        urls = []
        params_list = []
        
        for endpoint in endpoints:
            urls.append(endpoint['url'])
            params = endpoint.get('params', {}).copy()
            params['APIkey'] = api_key
            params_list.append(params)
            
        results = await fetcher.fetch_multiple(urls, params_list)
        
        # Endpoint isimleriyle eşle
        named_results = {}
        for endpoint, result in zip(endpoints, results):
            named_results[endpoint['name']] = result
            
        return named_results