"""
Takım Performans Güncelleyici

Bu modül, takım performans verilerini düzenli olarak güncelleyen bir planlayıcı (scheduler) içerir.
Bu sayede, tahmin sistemi her zaman güncel takım performans verilerine dayanarak faktörlerini hesaplar.

Özellikler:
- Düzenli olarak maç sonuçlarını kontrol eder
- Takım performans verilerini günceller
- Veritabanı faktörlerini otomatik olarak yeniler
- Düzenli raporlar ve uyarılar oluşturur
"""

import os
import time
import json
import logging
import threading
import sqlite3
import requests
from datetime import datetime, timedelta
from dynamic_team_analyzer import DynamicTeamAnalyzer

# Logging ayarları
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TeamPerformanceUpdater:
    """Takım performanslarını otomatik olarak güncelleyen sınıf"""
    
    def __init__(self, 
                 analyzer=None, 
                 api_key=None, 
                 update_interval=12, # Saat cinsinden
                 cache_file="predictions_cache.json",
                 api_endpoint="https://api.football-data.org/v4"):
        """
        Takım performans güncelleyiciyi başlat
        
        Args:
            analyzer: DynamicTeamAnalyzer örneği
            api_key: Football-Data API anahtarı
            update_interval: Güncelleme aralığı (saat cinsinden)
            cache_file: Tahmin önbellek dosyası
            api_endpoint: API uç noktası
        """
        self.analyzer = analyzer or DynamicTeamAnalyzer()
        self.api_key = api_key
        self.update_interval = update_interval 
        self.cache_file = cache_file
        self.api_endpoint = api_endpoint
        self.is_running = False
        self.thread = None
        self.stop_event = threading.Event()
        
        # API anahtarını kontrol et
        if not self.api_key:
            try:
                # Çevre değişkeninden API anahtarını almaya çalış
                self.api_key = os.environ.get('FOOTBALL_API_KEY')
            except:
                logger.warning("API anahtarı bulunamadı. Veri akışı için bir API anahtarı gereklidir.")
    
    def start(self):
        """Düzenli güncelleme döngüsünü başlat"""
        if self.is_running:
            logger.warning("Takım performans güncelleyici zaten çalışıyor!")
            return False
        
        try:
            # İlk güncellemeyi hemen yap
            self.update_once()
            
            # Arkaplan thread'i başlat
            self.stop_event.clear()
            self.is_running = True
            self.thread = threading.Thread(target=self._update_loop)
            self.thread.daemon = True  # Ana program sonlandığında thread de sonlanır
            self.thread.start()
            
            logger.info(f"Takım performans güncelleyici başlatıldı. Güncelleme aralığı: {self.update_interval} saat")
            return True
        
        except Exception as e:
            logger.error(f"Takım performans güncelleyici başlatılırken hata: {str(e)}")
            self.is_running = False
            return False
    
    def stop(self):
        """Güncelleme döngüsünü durdur"""
        if not self.is_running:
            logger.warning("Takım performans güncelleyici zaten durdurulmuş!")
            return
        
        try:
            self.stop_event.set()
            if self.thread:
                self.thread.join(timeout=5.0)  # Thread'in kapanmasını bekle
            self.is_running = False
            logger.info("Takım performans güncelleyici durduruldu.")
        except Exception as e:
            logger.error(f"Takım performans güncelleyici durdurulurken hata: {str(e)}")
    
    def _update_loop(self):
        """Düzenli güncelleme döngüsü (arkaplan thread'inde çalışır)"""
        while not self.stop_event.is_set():
            try:
                # Belirlenen aralıkta güncelleme yap
                for _ in range(int(self.update_interval * 3600)):  # Saniye cinsinden
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
                
                # Süre dolduğunda güncelleme yap
                if not self.stop_event.is_set():
                    self.update_once()
                
            except Exception as e:
                logger.error(f"Güncelleme döngüsünde hata: {str(e)}")
                time.sleep(60)  # Hata durumunda kısa bir süre bekle
    
    def update_once(self):
        """Tek seferlik performans güncellemesi yap"""
        try:
            logger.info("Takım performans güncellemesi başlatılıyor...")
            
            # Tahmin önbelleğini analiz et
            self.analyzer.load_and_process_cache()
            
            # API'den son maç sonuçlarını al ve faktörleri güncelle
            self._update_from_api()
            
            # Güncel faktörleri JSON'a aktar
            self.analyzer.export_team_factors_json()
            
            logger.info("Takım performans güncellemesi tamamlandı.")
            return True
            
        except Exception as e:
            logger.error(f"Takım performans güncellemesi sırasında hata: {str(e)}")
            return False
    
    def _update_from_api(self):
        """API'den son maç sonuçlarını alarak takım faktörlerini güncelle"""
        if not self.api_key:
            logger.warning("API anahtarı olmadan maç sonuçları güncellenemiyor.")
            return
        
        try:
            # Son 3 gün içindeki maçları al
            end_date = datetime.now()
            start_date = end_date - timedelta(days=3)
            
            # API isteği yap
            headers = {'X-Auth-Token': self.api_key}
            params = {
                'dateFrom': start_date.strftime('%Y-%m-%d'),
                'dateTo': end_date.strftime('%Y-%m-%d'),
                'status': 'FINISHED'
            }
            
            response = requests.get(f"{self.api_endpoint}/matches", headers=headers, params=params)
            
            if response.status_code != 200:
                logger.error(f"API isteği başarısız: {response.status_code} - {response.text}")
                return
            
            # Maç verilerini işle
            match_data = response.json()
            matches = match_data.get('matches', [])
            
            if not matches:
                logger.info("Güncellenecek yeni maç bulunamadı.")
                return
            
            logger.info(f"{len(matches)} maç sonucu bulundu, faktörler güncelleniyor...")
            
            # Sonuçlanan maçları işle ve faktörleri güncelle
            updated_count = 0
            for match in matches:
                try:
                    # Maç sonucu tamamlanmış mı kontrol et
                    if match.get('status') != 'FINISHED':
                        continue
                        
                    # Maç ID'si
                    match_id = str(match.get('id'))
                    
                    # Takım bilgileri
                    home_team = match.get('homeTeam', {})
                    away_team = match.get('awayTeam', {})
                    
                    home_team_id = str(home_team.get('id', ''))
                    away_team_id = str(away_team.get('id', ''))
                    home_team_name = home_team.get('name', 'Unknown')
                    away_team_name = away_team.get('name', 'Unknown')
                    
                    # Skor bilgileri
                    score = match.get('score', {})
                    full_time = score.get('fullTime', {})
                    
                    home_goals = full_time.get('home', 0)
                    away_goals = full_time.get('away', 0)
                    
                    # Eksik veriler varsa atla
                    if not home_team_id or not away_team_id or home_goals is None or away_goals is None:
                        continue
                    
                    # Maç tarihini al
                    match_date = match.get('utcDate', datetime.now().isoformat())
                    
                    # Turnuva bilgisi
                    competition = match.get('competition', {})
                    competition_id = str(competition.get('id', ''))
                    season = match.get('season', {}).get('id', datetime.now().year)
                    
                    # Takım faktörlerini güncelle
                    update_result = self.analyzer.update_from_match_result(
                        match_id=match_id,
                        home_team_id=home_team_id,
                        away_team_id=away_team_id,
                        home_team_name=home_team_name,
                        away_team_name=away_team_name,
                        home_goals=home_goals,
                        away_goals=away_goals,
                        match_date=match_date,
                        competition_id=competition_id,
                        season=season
                    )
                    
                    if update_result:
                        updated_count += 1
                        
                except Exception as e:
                    logger.error(f"Maç işlenirken hata: {str(e)}")
            
            logger.info(f"Toplam {updated_count} maç sonucu ile faktörler güncellendi.")
            
        except Exception as e:
            logger.error(f"API ile güncelleme sırasında hata: {str(e)}")
    
    def generate_performance_report(self, output_file="team_performance_report.json"):
        """Takım performans raporu oluştur"""
        try:
            # Tüm takım faktörlerini al
            factors = self.analyzer.get_all_team_factors()
            
            # Rapor için istatistikleri hesapla
            teams_count = len(factors)
            home_dominated = []
            away_dominated = []
            balanced = []
            
            for team_id, data in factors.items():
                name = data.get('name', 'Unknown')
                home_factor = data.get('home_factor', 1.0)
                away_factor = data.get('away_factor', 1.0)
                
                # Faktör farkına göre kategorize et
                factor_diff = home_factor - away_factor
                if factor_diff > 0.3:
                    home_dominated.append({
                        'id': team_id,
                        'name': name,
                        'home_factor': home_factor,
                        'away_factor': away_factor,
                        'difference': factor_diff
                    })
                elif factor_diff < -0.1:
                    away_dominated.append({
                        'id': team_id,
                        'name': name,
                        'home_factor': home_factor,
                        'away_factor': away_factor,
                        'difference': factor_diff
                    })
                else:
                    balanced.append({
                        'id': team_id,
                        'name': name,
                        'home_factor': home_factor,
                        'away_factor': away_factor,
                        'difference': factor_diff
                    })
            
            # Kategorileri faktör farkına göre sırala
            home_dominated.sort(key=lambda x: x['difference'], reverse=True)
            away_dominated.sort(key=lambda x: x['difference'])
            
            # Raporu oluştur
            report = {
                'generated_at': datetime.now().isoformat(),
                'summary': {
                    'total_teams': teams_count,
                    'home_dominated_count': len(home_dominated),
                    'away_dominated_count': len(away_dominated),
                    'balanced_count': len(balanced)
                },
                'top_home_dominated': home_dominated[:10],
                'top_away_dominated': away_dominated[:10],
                'most_balanced': balanced[:10]
            }
            
            # Raporu JSON dosyasına kaydet
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Takım performans raporu oluşturuldu: {output_file}")
            return report
            
        except Exception as e:
            logger.error(f"Performans raporu oluşturulurken hata: {str(e)}")
            return None

    def detect_performance_changes(self, threshold=0.1):
        """
        Son güncelleme sonrası performans değişikliklerini belirle
        
        Args:
            threshold: Değişiklik için eşik değeri
            
        Returns:
            dict: Değişiklik raporları
        """
        try:
            conn = sqlite3.connect(self.analyzer.db_path)
            cursor = conn.cursor()
            
            # Son güncelleme zamanı
            last_day = datetime.now() - timedelta(days=1)
            last_day_str = last_day.isoformat()
            
            # Son bir günde oynanan maçları al
            cursor.execute("""
            SELECT home_team_id, away_team_id, home_team_name, away_team_name,
            home_goals, away_goals, match_date
            FROM match_results
            WHERE match_date > ?
            ORDER BY match_date DESC
            """, (last_day_str,))
            
            recent_matches = cursor.fetchall()
            
            if not recent_matches:
                return {"message": "Son 24 saatte yeni maç bulunamadı."}
            
            # Performans değişimlerini tespit et
            changes = []
            significant_changes = []
            
            for match in recent_matches:
                home_id, away_id, home_name, away_name, home_goals, away_goals, match_date = match
                
                # Takım faktörlerini al
                cursor.execute("""
                SELECT home_factor, away_factor, last_updated
                FROM team_performances
                WHERE team_id = ?
                """, (home_id,))
                
                home_data = cursor.fetchone()
                
                cursor.execute("""
                SELECT home_factor, away_factor, last_updated
                FROM team_performances
                WHERE team_id = ?
                """, (away_id,))
                
                away_data = cursor.fetchone()
                
                if home_data and away_data:
                    home_factor, _, home_updated = home_data
                    _, away_factor, away_updated = away_data
                    
                    # Maç sonucu analizi
                    if home_goals > away_goals:
                        result = f"{home_name} kazandı ({home_goals}-{away_goals})"
                        expected = "Beklenen" if home_factor > away_factor else "Sürpriz"
                    elif away_goals > home_goals:
                        result = f"{away_name} kazandı ({away_goals}-{home_goals})"
                        expected = "Beklenen" if away_factor > home_factor else "Sürpriz"
                    else:
                        result = f"Beraberlik ({home_goals}-{away_goals})"
                        expected = "Beklenen" if abs(home_factor - away_factor) < 0.1 else "Sürpriz"
                    
                    # Değişim kaydı
                    change_record = {
                        "match": f"{home_name} vs {away_name}",
                        "date": match_date,
                        "result": result,
                        "expectation": expected,
                        "home_factor": home_factor,
                        "away_factor": away_factor
                    }
                    
                    changes.append(change_record)
                    
                    # Önemli değişiklik mi kontrol et
                    if (expected == "Sürpriz" and 
                        ((home_goals > away_goals and home_factor < away_factor and 
                          (away_factor - home_factor) > threshold) or
                         (away_goals > home_goals and away_factor < home_factor and
                          (home_factor - away_factor) > threshold))):
                        
                        significant_changes.append(change_record)
            
            conn.close()
            
            return {
                "total_recent_matches": len(recent_matches),
                "match_changes": changes,
                "significant_changes": significant_changes
            }
            
        except Exception as e:
            logger.error(f"Performans değişiklikleri belirlenirken hata: {str(e)}")
            return {"error": str(e)}

# Komut satırından çalıştırma
if __name__ == "__main__":
    try:
        # API anahtarını al
        api_key = os.environ.get('FOOTBALL_API_KEY')
        
        if not api_key:
            logger.warning("API anahtarı bulunamadı. '--no-api' seçeneğiyle devam edilecek.")
        
        # Analyzer modülünü başlat
        analyzer = DynamicTeamAnalyzer()
        
        # Güncelleyici oluştur
        updater = TeamPerformanceUpdater(analyzer=analyzer, api_key=api_key)
        
        # Tek seferlik güncelleme yap
        print("Takım performans verilerini güncelleme...")
        success = updater.update_once()
        
        if success:
            print("Takım performans verileri başarıyla güncellendi!")
            
            # Performans raporu oluştur
            report = updater.generate_performance_report()
            
            if report:
                print(f"Toplam {report['summary']['total_teams']} takım analiz edildi.")
                print(f"Ev avantajı olan takımlar: {report['summary']['home_dominated_count']}")
                print(f"Deplasman avantajı olan takımlar: {report['summary']['away_dominated_count']}")
                print(f"Dengeli takımlar: {report['summary']['balanced_count']}")
                
                # En yüksek ev avantajı olan takımlar
                print("\nEn Yüksek Ev Avantajı Olan Takımlar:")
                for i, team in enumerate(report['top_home_dominated'][:5], 1):
                    print(f"{i}. {team['name']}: Ev {team['home_factor']:.2f}, Deplasman {team['away_factor']:.2f}")
                
                # Değişim tespiti
                changes = updater.detect_performance_changes()
                if changes.get("significant_changes"):
                    print("\nÖnemli Performans Değişiklikleri:")
                    for change in changes["significant_changes"]:
                        print(f"- {change['match']}: {change['result']} ({change['expectation']})")
                
            print("\nGüncelleyiciyi arkaplanda çalıştırmak için:")
            print("from team_performance_updater import TeamPerformanceUpdater")
            print("updater = TeamPerformanceUpdater()")
            print("updater.start()  # Arkaplanda periyodik güncelleme başlatır")
        else:
            print("Takım performans verileri güncellenemedi.")
    
    except Exception as e:
        print(f"Hata: {str(e)}")