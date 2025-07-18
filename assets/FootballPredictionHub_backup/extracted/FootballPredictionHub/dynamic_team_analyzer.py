"""
Dinamik Takım Analizörü

Bu modül, takımların ev sahibi ve deplasman performanslarını otomatik olarak analiz ederek
faktör değerlerini hesaplar. Bu şekilde tahmin sistemi, statik değerler yerine güncel performansa
dayalı dinamik faktörler kullanabilir.

Özellikler:
- Takımların son maçlarını analiz eder
- Ev/deplasman performans asimetrilerini hesaplar
- Değişen formu yakalar ve faktörleri günceller
- Performans verilerini veritabanında saklar
"""

import os
import json
import logging
import sqlite3
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicTeamAnalyzer:
    """Takımların ev/deplasman performanslarını dinamik olarak analiz eden sınıf"""
    
    def __init__(self, db_path="./team_performance.db", cache_file="predictions_cache.json"):
        """
        Analizör sınıfını başlat
        
        Args:
            db_path: Veritabanı dosya yolu
            cache_file: Tahmin önbellek dosyası yolu
        """
        self.db_path = db_path
        self.cache_file = cache_file
        self.performances = defaultdict(lambda: {
            'home': {'matches': 0, 'goals_scored': 0, 'goals_conceded': 0, 'wins': 0, 'draws': 0, 'losses': 0},
            'away': {'matches': 0, 'goals_scored': 0, 'goals_conceded': 0, 'wins': 0, 'draws': 0, 'losses': 0}
        })
        
        # Veritabanını oluştur
        self._init_database()
        
    def _init_database(self):
        """Veritabanını oluştur veya bağlan"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Takım performans tablosunu oluştur
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS team_performances (
                team_id TEXT,
                team_name TEXT,
                home_factor REAL,
                away_factor REAL,
                home_matches INTEGER,
                away_matches INTEGER,
                home_goals_scored REAL,
                home_goals_conceded REAL,
                away_goals_scored REAL,
                away_goals_conceded REAL,
                home_wins INTEGER,
                home_draws INTEGER,
                home_losses INTEGER,
                away_wins INTEGER,
                away_draws INTEGER,
                away_losses INTEGER,
                last_updated TIMESTAMP,
                PRIMARY KEY (team_id)
            )
            ''')
            
            # Maç sonuçları tablosunu oluştur
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS match_results (
                match_id TEXT,
                competition_id TEXT,
                home_team_id TEXT,
                away_team_id TEXT,
                home_team_name TEXT,
                away_team_name TEXT,
                home_goals INTEGER,
                away_goals INTEGER,
                match_date TIMESTAMP,
                season TEXT,
                PRIMARY KEY (match_id)
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Veritabanı başarıyla oluşturuldu/bağlandı")
            
        except Exception as e:
            logger.error(f"Veritabanı oluşturulurken hata: {str(e)}")
    
    def load_and_process_cache(self):
        """Tahmin önbelleğini yükle ve işle"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                logger.info(f"Tahmin önbelleği yüklendi: {len(cache_data)} tahmin")
                return self._process_cache_data(cache_data)
            else:
                logger.warning(f"Tahmin önbellek dosyası bulunamadı: {self.cache_file}")
                return {}
        except Exception as e:
            logger.error(f"Tahmin önbelleği işlenirken hata: {str(e)}")
            return {}
    
    def _process_cache_data(self, cache_data):
        """Önbellek verilerinden takım performanslarını hesapla"""
        # Performans verileri sözlüğü
        performances = defaultdict(lambda: {
            'home': {'matches': 0, 'goals_scored': 0, 'goals_conceded': 0, 'wins': 0, 'draws': 0, 'losses': 0},
            'away': {'matches': 0, 'goals_scored': 0, 'goals_conceded': 0, 'wins': 0, 'draws': 0, 'losses': 0}
        })
        
        # Önbellek verilerini işle
        for match_key, prediction in cache_data.items():
            if not isinstance(prediction, dict) or 'home_team' not in prediction or 'away_team' not in prediction:
                continue
            
            # Takım bilgilerini al
            home_team = prediction.get('home_team', {})
            away_team = prediction.get('away_team', {})
            
            home_team_id = home_team.get('id')
            away_team_id = away_team.get('id')
            
            if not home_team_id or not away_team_id:
                continue
            
            # Takım formlarını al
            home_form = home_team.get('form', {})
            away_form = away_team.get('form', {})
            
            # Son maçları analiz et
            if 'recent_match_data' in home_form:
                self._analyze_recent_matches(home_team_id, home_team.get('name', ''), home_form.get('recent_match_data', []), performances)
                
            if 'recent_match_data' in away_form:
                self._analyze_recent_matches(away_team_id, away_team.get('name', ''), away_form.get('recent_match_data', []), performances)
        
        # Performansları veritabanına kaydet
        self._update_performances_in_db(performances)
        
        # Faktörleri hesapla
        return self._calculate_factors(performances)
    
    def _analyze_recent_matches(self, team_id, team_name, matches, performances):
        """Takımın son maçlarını analiz et"""
        for match in matches:
            is_home = match.get('is_home', False)
            goals_scored = match.get('goals_scored', 0)
            goals_conceded = match.get('goals_conceded', 0)
            
            # Sonucu belirle
            result = 'draws'  # Varsayılan: beraberlik
            if goals_scored > goals_conceded:
                result = 'wins'
            elif goals_scored < goals_conceded:
                result = 'losses'
            
            # Performans verilerini güncelle
            location = 'home' if is_home else 'away'
            performances[team_id][location]['matches'] += 1
            performances[team_id][location]['goals_scored'] += goals_scored
            performances[team_id][location]['goals_conceded'] += goals_conceded
            performances[team_id][location][result] += 1
            
            # Maçı veritabanına ekle - sadece yeterli bilgi varsa
            if 'match_id' in match and 'match_date' in match:
                self._add_match_to_db(
                    match_id=match.get('match_id'),
                    home_team_id=team_id if is_home else match.get('opponent_id', 'unknown'),
                    away_team_id=match.get('opponent_id', 'unknown') if is_home else team_id,
                    home_team_name=team_name if is_home else match.get('opponent', 'Unknown'),
                    away_team_name=match.get('opponent', 'Unknown') if is_home else team_name,
                    home_goals=goals_scored if is_home else goals_conceded,
                    away_goals=goals_conceded if is_home else goals_scored,
                    match_date=match.get('match_date'),
                    competition_id=match.get('competition_id', 'unknown'),
                    season=match.get('season', datetime.now().year)
                )
    
    def _add_match_to_db(self, match_id, home_team_id, away_team_id, home_team_name, away_team_name, 
                         home_goals, away_goals, match_date, competition_id, season):
        """Maçı veritabanına ekle"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Maç zaten var mı kontrol et
            cursor.execute("SELECT 1 FROM match_results WHERE match_id = ?", (match_id,))
            if cursor.fetchone():
                conn.close()
                return  # Maç zaten var, işlemi atla
            
            # Yeni maçı ekle
            cursor.execute('''
            INSERT INTO match_results 
            (match_id, competition_id, home_team_id, away_team_id, home_team_name, away_team_name,
             home_goals, away_goals, match_date, season)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (match_id, competition_id, home_team_id, away_team_id, home_team_name, away_team_name,
                  home_goals, away_goals, match_date, season))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Maç veritabanına eklenirken hata: {str(e)}")
    
    def _update_performances_in_db(self, performances):
        """Takım performanslarını veritabanına kaydet"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            update_count = 0
            insert_count = 0
            
            for team_id, data in performances.items():
                home_data = data['home']
                away_data = data['away']
                
                # Faktörleri hesapla
                home_factor, away_factor = self._compute_team_factors(home_data, away_data)
                
                # Takım adını al
                team_name = self._get_team_name_from_db(cursor, team_id)
                
                # Takım veritabanında var mı kontrol et
                cursor.execute("SELECT 1 FROM team_performances WHERE team_id = ?", (team_id,))
                if cursor.fetchone():
                    # Takım verilerini güncelle
                    cursor.execute('''
                    UPDATE team_performances SET
                    team_name = ?,
                    home_factor = ?,
                    away_factor = ?,
                    home_matches = ?,
                    away_matches = ?,
                    home_goals_scored = ?,
                    home_goals_conceded = ?,
                    away_goals_scored = ?,
                    away_goals_conceded = ?,
                    home_wins = ?,
                    home_draws = ?,
                    home_losses = ?,
                    away_wins = ?,
                    away_draws = ?,
                    away_losses = ?,
                    last_updated = ?
                    WHERE team_id = ?
                    ''', (
                        team_name, home_factor, away_factor,
                        home_data['matches'], away_data['matches'],
                        home_data['goals_scored'], home_data['goals_conceded'],
                        away_data['goals_scored'], away_data['goals_conceded'],
                        home_data['wins'], home_data['draws'], home_data['losses'],
                        away_data['wins'], away_data['draws'], away_data['losses'],
                        datetime.now().isoformat(), team_id
                    ))
                    update_count += 1
                else:
                    # Yeni takım ekle
                    cursor.execute('''
                    INSERT INTO team_performances
                    (team_id, team_name, home_factor, away_factor, 
                    home_matches, away_matches, 
                    home_goals_scored, home_goals_conceded, 
                    away_goals_scored, away_goals_conceded,
                    home_wins, home_draws, home_losses,
                    away_wins, away_draws, away_losses,
                    last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        team_id, team_name, home_factor, away_factor,
                        home_data['matches'], away_data['matches'],
                        home_data['goals_scored'], home_data['goals_conceded'],
                        away_data['goals_scored'], away_data['goals_conceded'],
                        home_data['wins'], home_data['draws'], home_data['losses'],
                        away_data['wins'], away_data['draws'], away_data['losses'],
                        datetime.now().isoformat()
                    ))
                    insert_count += 1
            
            conn.commit()
            conn.close()
            
            logger.info(f"Takım performansları güncellendi: {update_count} güncelleme, {insert_count} ekleme")
            
        except Exception as e:
            logger.error(f"Takım performansları güncellenirken hata: {str(e)}")
    
    def _get_team_name_from_db(self, cursor, team_id):
        """Takım ID'sinden takım adını bul"""
        # Önce match_results tablosunda ara
        cursor.execute("""
        SELECT home_team_name FROM match_results 
        WHERE home_team_id = ? 
        ORDER BY match_date DESC LIMIT 1
        """, (team_id,))
        
        result = cursor.fetchone()
        if result:
            return result[0]
        
        # Bulunamazsa away_team olarak ara
        cursor.execute("""
        SELECT away_team_name FROM match_results 
        WHERE away_team_id = ? 
        ORDER BY match_date DESC LIMIT 1
        """, (team_id,))
        
        result = cursor.fetchone()
        if result:
            return result[0]
        
        # Bulunamazsa "Unknown" döndür
        return "Unknown"
    
    def _compute_team_factors(self, home_data, away_data):
        """Takımın ev sahibi ve deplasman faktörlerini hesapla"""
        # Varsayılan faktörler
        default_home_factor = 1.0
        default_away_factor = 1.0
        
        # Yeterli maç yoksa varsayılan değerleri kullan
        min_matches = 3
        if home_data['matches'] < min_matches or away_data['matches'] < min_matches:
            return default_home_factor, default_away_factor
        
        # Ev sahibi performansı
        home_points = home_data['wins'] * 3 + home_data['draws']
        home_max_points = home_data['matches'] * 3
        home_performance = home_points / home_max_points if home_max_points > 0 else 0.5
        
        # Deplasman performansı
        away_points = away_data['wins'] * 3 + away_data['draws']
        away_max_points = away_data['matches'] * 3
        away_performance = away_points / away_max_points if away_max_points > 0 else 0.5
        
        # Gol performansı
        home_goal_rate = home_data['goals_scored'] / home_data['matches'] if home_data['matches'] > 0 else 1.0
        away_goal_rate = away_data['goals_scored'] / away_data['matches'] if away_data['matches'] > 0 else 1.0
        
        # Faktörleri hesapla
        # Ev performansı çok iyiyse ev faktörünü artır (1.0-1.4 arasında)
        home_factor = 1.0 + (home_performance - 0.5) * 0.8  # 0.6->1.08, 0.7->1.16, 0.8->1.24
        
        # Deplasman performansı kötüyse away faktörünü azalt (0.7-1.1 arasında)
        away_factor = 0.9 + (away_performance - 0.5) * 0.4  # 0.4->0.86, 0.3->0.82, 0.2->0.78
        
        # Gol performansıyla da ağırlıklandır
        league_avg_home_goals = 1.5  # Lig ortalaması varsayılan değer
        league_avg_away_goals = 1.2  # Lig ortalaması varsayılan değer
        
        home_goal_ratio = home_goal_rate / league_avg_home_goals if league_avg_home_goals > 0 else 1.0
        away_goal_ratio = away_goal_rate / league_avg_away_goals if league_avg_away_goals > 0 else 1.0
        
        # Faktörleri gol oranlarıyla düzelt
        home_factor = home_factor * 0.7 + home_goal_ratio * 0.3
        away_factor = away_factor * 0.7 + away_goal_ratio * 0.3
        
        # Faktörleri sınırla (aşırı değerlerden kaçın)
        home_factor = max(0.8, min(1.5, home_factor))
        away_factor = max(0.6, min(1.2, away_factor))
        
        return home_factor, away_factor
    
    def _calculate_factors(self, performances):
        """Tüm takımlar için faktörleri hesapla ve döndür"""
        factors = {}
        
        for team_id, data in performances.items():
            home_data = data['home']
            away_data = data['away']
            
            # Faktörleri hesapla
            home_factor, away_factor = self._compute_team_factors(home_data, away_data)
            
            # Takım verilerini ekle
            factors[team_id] = {
                "home_factor": home_factor,
                "away_factor": away_factor,
                "home_data": {
                    "matches": home_data['matches'],
                    "wins": home_data['wins'],
                    "draws": home_data['draws'],
                    "losses": home_data['losses'],
                    "goals_scored": home_data['goals_scored'],
                    "goals_conceded": home_data['goals_conceded']
                },
                "away_data": {
                    "matches": away_data['matches'],
                    "wins": away_data['wins'],
                    "draws": away_data['draws'],
                    "losses": away_data['losses'],
                    "goals_scored": away_data['goals_scored'],
                    "goals_conceded": away_data['goals_conceded']
                }
            }
        
        return factors
    
    def get_all_team_factors(self):
        """Tüm takımlar için faktörleri veritabanından al"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT team_id, team_name, home_factor, away_factor,
            home_matches, away_matches, home_wins, away_wins
            FROM team_performances
            """)
            
            factors = {}
            for row in cursor.fetchall():
                team_id, team_name, home_factor, away_factor, home_matches, away_matches, home_wins, away_wins = row
                factors[team_id] = {
                    "name": team_name,
                    "home_factor": home_factor,
                    "away_factor": away_factor,
                    "home_matches": home_matches,
                    "away_matches": away_matches,
                    "home_win_rate": home_wins / home_matches if home_matches > 0 else 0.0,
                    "away_win_rate": away_wins / away_matches if away_matches > 0 else 0.0
                }
            
            conn.close()
            return factors
            
        except Exception as e:
            logger.error(f"Takım faktörleri alınırken hata: {str(e)}")
            return {}
    
    def get_team_factors(self, team_id):
        """Belirli bir takım için faktörleri veritabanından al"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT team_name, home_factor, away_factor, 
            home_matches, away_matches,
            home_goals_scored, home_goals_conceded,
            away_goals_scored, away_goals_conceded,
            home_wins, home_draws, home_losses,
            away_wins, away_draws, away_losses,
            last_updated
            FROM team_performances
            WHERE team_id = ?
            """, (team_id,))
            
            row = cursor.fetchone()
            if not row:
                conn.close()
                return None
                
            team_name, home_factor, away_factor, home_matches, away_matches, \
            home_goals_scored, home_goals_conceded, away_goals_scored, away_goals_conceded, \
            home_wins, home_draws, home_losses, away_wins, away_draws, away_losses, last_updated = row
            
            # Son maçları da al
            cursor.execute("""
            SELECT home_team_id, away_team_id, home_team_name, away_team_name,
            home_goals, away_goals, match_date
            FROM match_results
            WHERE home_team_id = ? OR away_team_id = ?
            ORDER BY match_date DESC
            LIMIT 10
            """, (team_id, team_id))
            
            recent_matches = []
            for match_row in cursor.fetchall():
                home_id, away_id, home_name, away_name, home_goals, away_goals, match_date = match_row
                is_home = (home_id == team_id)
                
                opponent_id = away_id if is_home else home_id
                opponent_name = away_name if is_home else home_name
                goals_scored = home_goals if is_home else away_goals
                goals_conceded = away_goals if is_home else home_goals
                
                recent_matches.append({
                    "is_home": is_home,
                    "opponent_id": opponent_id,
                    "opponent_name": opponent_name,
                    "goals_scored": goals_scored,
                    "goals_conceded": goals_conceded,
                    "match_date": match_date,
                    "result": "W" if goals_scored > goals_conceded else ("D" if goals_scored == goals_conceded else "L")
                })
            
            conn.close()
            
            # Takım faktör ve veri yapısını oluştur
            return {
                "id": team_id,
                "name": team_name,
                "home_factor": home_factor,
                "away_factor": away_factor,
                "home_data": {
                    "matches": home_matches,
                    "wins": home_wins,
                    "draws": home_draws,
                    "losses": home_losses,
                    "goals_scored": home_goals_scored,
                    "goals_conceded": home_goals_conceded,
                    "points_per_game": (home_wins * 3 + home_draws) / home_matches if home_matches > 0 else 0.0,
                    "goals_per_game": home_goals_scored / home_matches if home_matches > 0 else 0.0,
                    "win_rate": home_wins / home_matches if home_matches > 0 else 0.0
                },
                "away_data": {
                    "matches": away_matches,
                    "wins": away_wins,
                    "draws": away_draws,
                    "losses": away_losses,
                    "goals_scored": away_goals_scored,
                    "goals_conceded": away_goals_conceded,
                    "points_per_game": (away_wins * 3 + away_draws) / away_matches if away_matches > 0 else 0.0,
                    "goals_per_game": away_goals_scored / away_matches if away_matches > 0 else 0.0,
                    "win_rate": away_wins / away_matches if away_matches > 0 else 0.0
                },
                "recent_matches": recent_matches,
                "last_updated": last_updated
            }
            
        except Exception as e:
            logger.error(f"Takım faktörleri alınırken hata: {str(e)}")
            return None
    
    def update_from_match_result(self, match_id, home_team_id, away_team_id, home_team_name, away_team_name, 
                                 home_goals, away_goals, match_date, competition_id=None, season=None):
        """Yeni bir maç sonucuyla faktörleri güncelle"""
        try:
            # Maçı veritabanına ekle
            self._add_match_to_db(
                match_id=match_id,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                home_team_name=home_team_name,
                away_team_name=away_team_name,
                home_goals=home_goals,
                away_goals=away_goals,
                match_date=match_date,
                competition_id=competition_id or 'unknown',
                season=season or datetime.now().year
            )
            
            # Ev sahibi takım faktörlerini güncelle
            home_team_data = self.get_team_factors(home_team_id) or {
                'home_data': {'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0, 
                             'goals_scored': 0, 'goals_conceded': 0},
                'name': home_team_name
            }
            
            # Deplasman takım faktörlerini güncelle
            away_team_data = self.get_team_factors(away_team_id) or {
                'away_data': {'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0, 
                             'goals_scored': 0, 'goals_conceded': 0},
                'name': away_team_name
            }
            
            # Sonucu belirle
            home_result = 'draws'
            if home_goals > away_goals:
                home_result = 'wins'
            elif home_goals < away_goals:
                home_result = 'losses'
                
            away_result = 'draws'
            if away_goals > home_goals:
                away_result = 'wins'
            elif away_goals < home_goals:
                away_result = 'losses'
            
            # Ev sahibi takım verilerini güncelle
            home_data = home_team_data.get('home_data', {
                'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0, 
                'goals_scored': 0, 'goals_conceded': 0
            })
            
            home_data['matches'] += 1
            home_data['goals_scored'] += home_goals
            home_data['goals_conceded'] += away_goals
            home_data[home_result] += 1
            
            # Deplasman takım verilerini güncelle
            away_data = away_team_data.get('away_data', {
                'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0, 
                'goals_scored': 0, 'goals_conceded': 0
            })
            
            away_data['matches'] += 1
            away_data['goals_scored'] += away_goals
            away_data['goals_conceded'] += home_goals
            away_data[away_result] += 1
            
            # Faktörleri yeniden hesapla
            home_factor, _ = self._compute_team_factors(home_data, {})
            _, away_factor = self._compute_team_factors({}, away_data)
            
            # Veritabanını güncelle
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Ev sahibi takımı güncelle
            cursor.execute("""
            UPDATE team_performances SET
            team_name = ?,
            home_factor = ?,
            home_matches = ?,
            home_goals_scored = ?,
            home_goals_conceded = ?,
            home_wins = ?,
            home_draws = ?,
            home_losses = ?,
            last_updated = ?
            WHERE team_id = ?
            """, (
                home_team_name, home_factor,
                home_data['matches'], home_data['goals_scored'], home_data['goals_conceded'],
                home_data['wins'], home_data['draws'], home_data['losses'],
                datetime.now().isoformat(), home_team_id
            ))
            
            # Ev sahibi takım yoksa ekle
            if cursor.rowcount == 0:
                cursor.execute("""
                INSERT INTO team_performances
                (team_id, team_name, home_factor, away_factor, 
                home_matches, away_matches, 
                home_goals_scored, home_goals_conceded, 
                away_goals_scored, away_goals_conceded,
                home_wins, home_draws, home_losses,
                away_wins, away_draws, away_losses,
                last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    home_team_id, home_team_name, home_factor, 1.0,  # Varsayılan away_factor
                    home_data['matches'], 0,  # away_matches
                    home_data['goals_scored'], home_data['goals_conceded'], 
                    0, 0,  # away goals
                    home_data['wins'], home_data['draws'], home_data['losses'],
                    0, 0, 0,  # away results
                    datetime.now().isoformat()
                ))
            
            # Deplasman takımı güncelle
            cursor.execute("""
            UPDATE team_performances SET
            team_name = ?,
            away_factor = ?,
            away_matches = ?,
            away_goals_scored = ?,
            away_goals_conceded = ?,
            away_wins = ?,
            away_draws = ?,
            away_losses = ?,
            last_updated = ?
            WHERE team_id = ?
            """, (
                away_team_name, away_factor,
                away_data['matches'], away_data['goals_scored'], away_data['goals_conceded'],
                away_data['wins'], away_data['draws'], away_data['losses'],
                datetime.now().isoformat(), away_team_id
            ))
            
            # Deplasman takımı yoksa ekle
            if cursor.rowcount == 0:
                cursor.execute("""
                INSERT INTO team_performances
                (team_id, team_name, home_factor, away_factor, 
                home_matches, away_matches, 
                home_goals_scored, home_goals_conceded, 
                away_goals_scored, away_goals_conceded,
                home_wins, home_draws, home_losses,
                away_wins, away_draws, away_losses,
                last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    away_team_id, away_team_name, 1.0, away_factor,  # Varsayılan home_factor
                    0, away_data['matches'],  # home_matches
                    0, 0,  # home goals
                    away_data['goals_scored'], away_data['goals_conceded'],
                    0, 0, 0,  # home results
                    away_data['wins'], away_data['draws'], away_data['losses'],
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Takım faktörleri maç sonucuyla güncellendi: {home_team_name} vs {away_team_name}")
            return True
            
        except Exception as e:
            logger.error(f"Maç sonucuyla faktörler güncellenirken hata: {str(e)}")
            return False
    
    def export_team_factors_json(self, output_file="team_factors.json"):
        """Takım faktörlerini JSON dosyasına aktar"""
        try:
            factors = self.get_all_team_factors()
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(factors, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Takım faktörleri JSON dosyasına aktarıldı: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Takım faktörleri JSON'a aktarılırken hata: {str(e)}")
            return False
    
    def import_team_factors_json(self, input_file="team_factors.json"):
        """Takım faktörlerini JSON dosyasından içe aktar"""
        try:
            if not os.path.exists(input_file):
                logger.warning(f"JSON dosyası bulunamadı: {input_file}")
                return False
                
            with open(input_file, 'r', encoding='utf-8') as f:
                factors = json.load(f)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for team_id, data in factors.items():
                # Takım veritabanında var mı kontrol et
                cursor.execute("SELECT 1 FROM team_performances WHERE team_id = ?", (team_id,))
                if cursor.fetchone():
                    # Takımı güncelle
                    cursor.execute("""
                    UPDATE team_performances SET
                    team_name = ?,
                    home_factor = ?,
                    away_factor = ?,
                    last_updated = ?
                    WHERE team_id = ?
                    """, (
                        data.get('name', 'Unknown'),
                        data.get('home_factor', 1.0),
                        data.get('away_factor', 1.0),
                        datetime.now().isoformat(),
                        team_id
                    ))
                else:
                    # Yeni takım ekle
                    cursor.execute("""
                    INSERT INTO team_performances
                    (team_id, team_name, home_factor, away_factor, 
                    home_matches, away_matches, 
                    home_goals_scored, home_goals_conceded, 
                    away_goals_scored, away_goals_conceded,
                    home_wins, home_draws, home_losses,
                    away_wins, away_draws, away_losses,
                    last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        team_id, data.get('name', 'Unknown'),
                        data.get('home_factor', 1.0), data.get('away_factor', 1.0),
                        data.get('home_matches', 0), data.get('away_matches', 0),
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        datetime.now().isoformat()
                    ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Takım faktörleri JSON dosyasından içe aktarıldı: {input_file}")
            return True
            
        except Exception as e:
            logger.error(f"Takım faktörleri JSON'dan içe aktarılırken hata: {str(e)}")
            return False
    
    def apply_dynamic_factors(self, home_team_id, away_team_id, home_goals, away_goals):
        """
        Tahmin için dinamik faktörleri uygula
        
        Args:
            home_team_id: Ev sahibi takım ID'si
            away_team_id: Deplasman takım ID'si
            home_goals: Ev sahibi takım beklenen golleri
            away_goals: Deplasman takım beklenen golleri
            
        Returns:
            tuple: Güncellenmiş (home_goals, away_goals)
        """
        try:
            # Takım faktörlerini al
            home_team = self.get_team_factors(home_team_id)
            away_team = self.get_team_factors(away_team_id)
            
            # Faktörler yoksa orijinal değerleri döndür
            if not home_team or not away_team:
                return home_goals, away_goals
            
            # Faktörleri uygula
            home_factor = home_team.get('home_factor', 1.0)
            away_factor = away_team.get('away_factor', 1.0)
            
            # Katsayıları sınırla (aşırı düzeltmelerden kaçın)
            home_factor = max(0.8, min(1.5, home_factor))
            away_factor = max(0.6, min(1.2, away_factor))
            
            # Beklenen golleri güncelle
            adjusted_home_goals = home_goals * home_factor
            adjusted_away_goals = away_goals * away_factor
            
            logger.info(f"Dinamik faktörler uygulandı: {home_team.get('name', 'Home')} ({home_factor:.2f}) vs "
                      f"{away_team.get('name', 'Away')} ({away_factor:.2f})")
            logger.info(f"Beklenen goller: Ev {home_goals:.2f}->{adjusted_home_goals:.2f}, "
                      f"Deplasman {away_goals:.2f}->{adjusted_away_goals:.2f}")
            
            return adjusted_home_goals, adjusted_away_goals
            
        except Exception as e:
            logger.error(f"Dinamik faktörler uygulanırken hata: {str(e)}")
            return home_goals, away_goals  # Hata durumunda orijinal değerleri döndür
    
    def analyze_and_update(self):
        """Tahmin önbelleğini yükle, analiz et ve faktörleri güncelle"""
        self.load_and_process_cache()
        factors = self.get_all_team_factors()
        logger.info(f"Takım faktörleri hesaplandı: {len(factors)} takım")
        return factors

# Test ve örnek kullanım
if __name__ == "__main__":
    analyzer = DynamicTeamAnalyzer()
    factors = analyzer.analyze_and_update()
    
    # Faktörleri dışa aktar
    analyzer.export_team_factors_json()
    
    # Örnek dinamik faktör uygulaması
    home_id = "610"  # Galatasaray
    away_id = "1005" # Fenerbahçe
    
    # Orijinal beklenen goller
    home_goals = 1.8
    away_goals = 1.2
    
    # Dinamik faktörleri uygula
    adj_home, adj_away = analyzer.apply_dynamic_factors(home_id, away_id, home_goals, away_goals)
    
    print(f"Orijinal beklenen goller: Ev {home_goals:.2f}, Deplasman {away_goals:.2f}")
    print(f"Dinamik faktörlerle: Ev {adj_home:.2f}, Deplasman {adj_away:.2f}")