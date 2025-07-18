"""
Takım Özel Tahmin Modelleri

Bu modül, belirli takımlar veya ligler için özelleştirilmiş tahmin modelleri içerir.
Bu modeller, genel tahmin modellerinden daha yüksek doğruluk sağlamak için 
belirli takım veya lig davranışlarına uyarlanmıştır.

Özellikler:
1. Lig-spesifik özellikler ve katsayılar
2. Takım-spesifik tahmin modelleri ve davranış analizi
3. Büyük takımlar için özel tahmin ayarlamaları
4. Gol dağılımı ve oyun stili analizi

Kullanım:
from team_specific_models import TeamSpecificPredictor
predictor = TeamSpecificPredictor()
team_adjustments = predictor.get_team_adjustments(home_team_id, away_team_id)
"""

import logging
import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TeamSpecificPredictor:
    """
    Takıma özel tahmin modellerini yöneten sınıf.
    Bu sınıf, farklı lig ve takımlar için özelleştirilmiş tahmin parametreleri sağlar.
    """
    
    def __init__(self, config_file="team_specific_config.json"):
        """
        Takıma özel tahmin modellerini başlatır.
        
        Args:
            config_file: Takım konfigürasyon dosyası
        """
        self.config_file = config_file
        self.config = self._load_config()
        self.team_cache = {}
        self.league_cache = {}
        
        # Lig türlerini tanımla
        self.league_types = {
            "HIGH_SCORING": ["Eredivisie", "Bundesliga", "Serie A"],
            "LOW_SCORING": ["La Liga", "Ligue 1", "English Premier League"],
            "HIGH_HOME_ADVANTAGE": ["Turkish Süper Lig", "Greek Super League", "Scottish Premiership"],
            "UNPREDICTABLE": ["MLS", "Chinese Super League", "Indian Super League"]
        }
        
        # Takım türlerini tanımla
        self.team_styles = {
            "DEFENSIVE": {
                "goal_factor": 0.8,
                "concede_factor": 0.7,
                "draw_bias": 1.2,
                "description": "Savunma ağırlıklı"
            },
            "OFFENSIVE": {
                "goal_factor": 1.3,
                "concede_factor": 1.2,
                "draw_bias": 0.8,
                "description": "Hücum ağırlıklı"
            },
            "BALANCED": {
                "goal_factor": 1.0,
                "concede_factor": 1.0,
                "draw_bias": 1.0,
                "description": "Dengeli"
            },
            "HOME_STRONG": {
                "home_goal_factor": 1.25,
                "home_concede_factor": 0.8,
                "away_goal_factor": 0.9,
                "away_concede_factor": 1.1,
                "description": "Evinde güçlü"
            },
            "AWAY_STRONG": {
                "home_goal_factor": 0.95,
                "home_concede_factor": 1.05,
                "away_goal_factor": 1.15,
                "away_concede_factor": 0.9,
                "description": "Deplasmanda güçlü"
            },
            "INCONSISTENT": {
                "variance_factor": 1.3,
                "description": "Tutarsız performans"
            }
        }
        
        # Büyük takım listesi
        self.big_teams = self._load_big_teams()
    
    def _load_config(self):
        """
        Takım konfigürasyon dosyasını yükler.
        Dosya yoksa varsayılan değerlerle oluşturur.
        
        Returns:
            dict: Takım konfigürasyonu
        """
        if not os.path.exists(self.config_file):
            # Varsayılan konfigürasyon oluştur
            default_config = {
                "teams": {},
                "leagues": {},
                "updated_at": datetime.now().isoformat()
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=4)
            
            return default_config
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Konfigürasyon dosyası yüklenirken hata: {str(e)}")
            return {"teams": {}, "leagues": {}, "updated_at": datetime.now().isoformat()}
    
    def _save_config(self):
        """
        Takım konfigürasyonunu dosyaya kaydeder.
        """
        try:
            self.config["updated_at"] = datetime.now().isoformat()
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=4)
                
            logger.info(f"Takım konfigürasyonu kaydedildi: {self.config_file}")
        except Exception as e:
            logger.error(f"Konfigürasyon kaydedilirken hata: {str(e)}")
    
    def _load_big_teams(self):
        """
        Büyük takımlar listesini yükler.
        
        Returns:
            dict: Lig ve büyük takımlar listesi
        """
        return {
            "Premier League": [33, 42, 40, 50, 36, 47],  # Top 6 İngiliz takımları
            "La Liga": [530, 541, 546, 536],             # Real, Barça, Atletico, Sevilla
            "Serie A": [496, 505, 497, 492, 502, 499],   # Juventus, Inter, Milan, Roma, Napoli, Lazio
            "Bundesliga": [157, 159, 169, 168, 170],     # Bayern, Dortmund, Leverkusen, Leipzig, Gladbach
            "Ligue 1": [85, 77, 79, 81, 80],             # PSG, Lyon, Marseille, Lille, Monaco
            "Süper Lig": [621, 627, 614, 631]            # Galatasaray, Fenerbahçe, Beşiktaş, Trabzonspor
        }
    
    def is_big_team(self, team_id, league_name=None):
        """
        Takımın büyük takım olup olmadığını kontrol eder.
        
        Args:
            team_id: Takım ID'si
            league_name: Lig adı (opsiyonel)
            
        Returns:
            bool: Büyük takım mı
        """
        team_id = int(team_id)
        
        # Belirli bir lig verildiyse sadece o ligi kontrol et
        if league_name and league_name in self.big_teams:
            return team_id in self.big_teams[league_name]
        
        # Tüm liglerde kontrol et
        for league, teams in self.big_teams.items():
            if team_id in teams:
                return True
        
        return False
    
    def get_league_type(self, league_name):
        """
        Lig türünü belirler.
        
        Args:
            league_name: Lig adı
            
        Returns:
            str: Lig türü
        """
        for league_type, leagues in self.league_types.items():
            if any(league.lower() in league_name.lower() for league in leagues):
                return league_type
        
        return "STANDARD"  # Varsayılan lig türü
    
    def register_team_data(self, team_id, team_name, league_id, league_name, team_data):
        """
        Takım verilerini kaydeder ve analiz eder.
        
        Args:
            team_id: Takım ID'si
            team_name: Takım adı
            league_id: Lig ID'si
            league_name: Lig adı
            team_data: Takım performans verileri
            
        Returns:
            dict: Takım stili ve özellikleri
        """
        # String IDs'yi int'e dönüştür
        team_id = str(team_id)
        league_id = str(league_id)
        
        # Takımın oyun stilini belirle
        team_style = self._determine_team_style(team_data)
        
        # Takımı konfigürasyona ekle
        if team_id not in self.config["teams"]:
            self.config["teams"][team_id] = {
                "name": team_name,
                "league_id": league_id,
                "league_name": league_name,
                "style": team_style,
                "is_big_team": self.is_big_team(team_id, league_name),
                "last_updated": datetime.now().isoformat(),
                "performance_history": []
            }
        else:
            # Mevcut takım bilgilerini güncelle
            self.config["teams"][team_id]["name"] = team_name
            self.config["teams"][team_id]["league_id"] = league_id
            self.config["teams"][team_id]["league_name"] = league_name
            self.config["teams"][team_id]["style"] = team_style
            self.config["teams"][team_id]["is_big_team"] = self.is_big_team(team_id, league_name)
            self.config["teams"][team_id]["last_updated"] = datetime.now().isoformat()
        
        # Performans verilerini ekle
        performance = {
            "date": datetime.now().isoformat(),
            "avg_goals_scored": team_data.get("avg_goals_scored", 0),
            "avg_goals_conceded": team_data.get("avg_goals_conceded", 0),
            "home_avg_goals_scored": team_data.get("home_performance", {}).get("avg_goals_scored", 0),
            "home_avg_goals_conceded": team_data.get("home_performance", {}).get("avg_goals_conceded", 0),
            "away_avg_goals_scored": team_data.get("away_performance", {}).get("avg_goals_scored", 0),
            "away_avg_goals_conceded": team_data.get("away_performance", {}).get("avg_goals_conceded", 0),
            "form_points": team_data.get("form_points", 0)
        }
        
        # Performans geçmişini güncelle (son 10 kayıt)
        self.config["teams"][team_id]["performance_history"].append(performance)
        self.config["teams"][team_id]["performance_history"] = self.config["teams"][team_id]["performance_history"][-10:]
        
        # Konfigürasyonu kaydet
        self._save_config()
        
        # Takım verisini ön belleğe al
        self.team_cache[team_id] = self.config["teams"][team_id]
        
        return team_style
    
    def register_league_data(self, league_id, league_name, league_data):
        """
        Lig verilerini kaydeder ve analiz eder.
        
        Args:
            league_id: Lig ID'si
            league_name: Lig adı
            league_data: Lig performans verileri
            
        Returns:
            str: Lig türü
        """
        league_id = str(league_id)
        
        # Lig türünü belirle
        league_type = self.get_league_type(league_name)
        
        # Ligi konfigürasyona ekle
        if league_id not in self.config["leagues"]:
            self.config["leagues"][league_id] = {
                "name": league_name,
                "type": league_type,
                "last_updated": datetime.now().isoformat(),
                "performance_metrics": {}
            }
        else:
            # Mevcut lig bilgilerini güncelle
            self.config["leagues"][league_id]["name"] = league_name
            self.config["leagues"][league_id]["type"] = league_type
            self.config["leagues"][league_id]["last_updated"] = datetime.now().isoformat()
        
        # Performans metriklerini ekle
        self.config["leagues"][league_id]["performance_metrics"] = {
            "avg_home_goals": league_data.get("avg_home_goals", 0),
            "avg_away_goals": league_data.get("avg_away_goals", 0),
            "avg_total_goals": league_data.get("avg_total_goals", 0),
            "home_win_percentage": league_data.get("home_win_percentage", 0),
            "draw_percentage": league_data.get("draw_percentage", 0),
            "away_win_percentage": league_data.get("away_win_percentage", 0),
            "btts_percentage": league_data.get("btts_percentage", 0),
            "over_2_5_percentage": league_data.get("over_2_5_percentage", 0)
        }
        
        # Konfigürasyonu kaydet
        self._save_config()
        
        # Lig verisini ön belleğe al
        self.league_cache[league_id] = self.config["leagues"][league_id]
        
        return league_type
    
    def _determine_team_style(self, team_data):
        """
        Takımın oyun stilini belirler.
        
        Args:
            team_data: Takım performans verileri
            
        Returns:
            dict: Takım stili ve özellikleri
        """
        # Genel ortalamalar
        avg_goals_scored = team_data.get("avg_goals_scored", 0)
        avg_goals_conceded = team_data.get("avg_goals_conceded", 0)
        
        # Ev sahibi performansı
        home_performance = team_data.get("home_performance", {})
        home_avg_goals_scored = home_performance.get("avg_goals_scored", 0)
        home_avg_goals_conceded = home_performance.get("avg_goals_conceded", 0)
        
        # Deplasman performansı
        away_performance = team_data.get("away_performance", {})
        away_avg_goals_scored = away_performance.get("avg_goals_scored", 0)
        away_avg_goals_conceded = away_performance.get("avg_goals_conceded", 0)
        
        # Ev vs deplasman gol farkı
        home_away_score_diff = (home_avg_goals_scored - away_avg_goals_scored)
        home_away_concede_diff = (home_avg_goals_conceded - away_avg_goals_conceded)
        
        # Sınıflandırma için eşik değerler
        offensive_threshold = 2.0  # Hücum ağırlıklı
        defensive_threshold = 1.0  # Savunma ağırlıklı
        home_strong_threshold = 0.7  # Evinde güçlü
        away_strong_threshold = 0.3  # Deplasmanda güçlü
        
        # Takım stilini belirle
        team_style = {}
        
        # Hücum/savunma dengesi
        if avg_goals_scored >= offensive_threshold and avg_goals_conceded >= 1.3:
            primary_style = "OFFENSIVE"
        elif avg_goals_scored <= defensive_threshold and avg_goals_conceded <= 1.0:
            primary_style = "DEFENSIVE"
        else:
            primary_style = "BALANCED"
            
        # Ev/deplasman dengesi
        if home_away_score_diff >= home_strong_threshold and home_away_concede_diff <= -0.2:
            location_style = "HOME_STRONG"
        elif home_away_score_diff <= -away_strong_threshold and home_away_concede_diff >= 0.2:
            location_style = "AWAY_STRONG"
        else:
            location_style = "BALANCED"
        
        # Maç sonuçlarında tutarsızlık kontrolü
        consistency_factor = team_data.get("consistency_factor", 1.0)  # Varsayılan: tutarlı
        consistency_style = "STANDARD"
        if consistency_factor > 1.2:
            consistency_style = "INCONSISTENT"
        
        # Stil ve faktörleri birleştir
        team_style = {
            "primary_style": primary_style,
            "location_style": location_style,
            "consistency_style": consistency_style,
            "factors": {
                # Ana stil faktörleri
                "goal_factor": self.team_styles[primary_style].get("goal_factor", 1.0),
                "concede_factor": self.team_styles[primary_style].get("concede_factor", 1.0),
                "draw_bias": self.team_styles[primary_style].get("draw_bias", 1.0),
                
                # Ev/deplasman faktörleri
                "home_goal_factor": self.team_styles[location_style].get("home_goal_factor", 1.0) 
                    if location_style in ["HOME_STRONG", "AWAY_STRONG"] else 1.0,
                "home_concede_factor": self.team_styles[location_style].get("home_concede_factor", 1.0)
                    if location_style in ["HOME_STRONG", "AWAY_STRONG"] else 1.0,
                "away_goal_factor": self.team_styles[location_style].get("away_goal_factor", 1.0)
                    if location_style in ["HOME_STRONG", "AWAY_STRONG"] else 1.0,
                "away_concede_factor": self.team_styles[location_style].get("away_concede_factor", 1.0)
                    if location_style in ["HOME_STRONG", "AWAY_STRONG"] else 1.0,
                
                # Tutarsızlık faktörü
                "variance_factor": self.team_styles["INCONSISTENT"].get("variance_factor", 1.0) 
                    if consistency_style == "INCONSISTENT" else 1.0
            },
            "description": f"{self.team_styles[primary_style]['description']}, "
                        f"{self.team_styles[location_style]['description'] if location_style != 'BALANCED' else 'normal ev/deplasman dengesi'}"
                        f"{', tutarsız performans' if consistency_style == 'INCONSISTENT' else ''}"
        }
        
        return team_style
    
    def get_team_adjustments(self, home_team_id, away_team_id, home_team_data=None, away_team_data=None):
        """
        İki takım arasındaki maç için tahmin ayarlamalarını yapar.
        
        Args:
            home_team_id: Ev sahibi takım ID'si
            away_team_id: Deplasman takım ID'si
            home_team_data: Ev sahibi takım verileri (opsiyonel, verilmezse önbellekten alınır)
            away_team_data: Deplasman takım verileri (opsiyonel, verilmezse önbellekten alınır)
            
        Returns:
            dict: Tahmin ayarlamaları
        """
        home_team_id = str(home_team_id)
        away_team_id = str(away_team_id)
        
        # Takım verilerini al (önbellekten veya parametre olarak verilen)
        home_team = home_team_data or self.get_team_data(home_team_id)
        away_team = away_team_data or self.get_team_data(away_team_id)
        
        # Takım verileri yoksa varsayılan ayarlamalar kullan
        if not home_team or not away_team:
            logger.warning(f"Takım verileri bulunamadı: ev={home_team_id}, deplasman={away_team_id}")
            return self._get_default_adjustments()
        
        # Takım stillerini al
        home_style = home_team.get("style", {})
        away_style = away_team.get("style", {})
        
        # Eğer takım stili yoksa ancak form verileri varsa, form verilerinden stil tahmin et
        if (not home_style or not home_style.get("factors")) and isinstance(home_team_data, dict) and home_team_data.get("recent_match_data"):
            logger.info(f"Ev sahibi takım için form verilerinden stil belirleniyor: {home_team_id}")
            home_style = self._determine_team_style_from_form(home_team_data)
            # Eğer takım objesine stil henüz eklenmemişse, ekle
            if not home_team.get("style"):
                home_team["style"] = home_style
        
        if (not away_style or not away_style.get("factors")) and isinstance(away_team_data, dict) and away_team_data.get("recent_match_data"):
            logger.info(f"Deplasman takımı için form verilerinden stil belirleniyor: {away_team_id}")
            away_style = self._determine_team_style_from_form(away_team_data)
            # Eğer takım objesine stil henüz eklenmemişse, ekle
            if not away_team.get("style"):
                away_team["style"] = away_style
        
        # Varsayılan değerler
        if not home_style or not away_style:
            logger.warning(f"Takım stilleri bulunamadı ve hesaplanamadı: ev={home_team_id}, deplasman={away_team_id}")
            return self._get_default_adjustments()
        
        # Lig verilerini al
        home_league_id = home_team.get("league_id")
        away_league_id = away_team.get("league_id")
        
        home_league = self.get_league_data(home_league_id) if home_league_id else None
        away_league = self.get_league_data(away_league_id) if away_league_id else None
        
        # Lig verisi yoksa varsayılan değerler kullan
        league_type = "STANDARD"
        if home_league and away_league and home_league_id == away_league_id:
            # Aynı ligdeyse doğrudan lig türünü al
            league_type = home_league.get("type", "STANDARD")
        elif home_league:
            # Farklı liglerdeyse ev sahibi ligini kullan
            league_type = home_league.get("type", "STANDARD")
        
        # Stil faktörlerini al
        home_factors = home_style.get("factors", {})
        away_factors = away_style.get("factors", {})
        
        # Takımlar arası göreli güç analizi
        team_power_comparison = self._analyze_team_power_difference(home_team, away_team)
        
        # Takım trendlerini analiz et (son maçlardaki form trendi)
        home_trend = self._analyze_team_trend(home_team_data) if isinstance(home_team_data, dict) else 0
        away_trend = self._analyze_team_trend(away_team_data) if isinstance(away_team_data, dict) else 0
        
        # Form trendi bazlı ayarlamalar
        home_trend_factor = 1.0 + (home_trend * 0.1)  # +/- %10 etki
        away_trend_factor = 1.0 + (away_trend * 0.1)  # +/- %10 etki
        
        # Tahmin ayarlamalarını oluştur
        adjustments = {
            # Gol faktörleri - hem atma hem yeme dengesi için
            "home_goal_multiplier": home_factors.get("goal_factor", 1.0) * home_factors.get("home_goal_factor", 1.0) * home_trend_factor,
            "away_goal_multiplier": away_factors.get("goal_factor", 1.0) * away_factors.get("away_goal_factor", 1.0) * away_trend_factor,
            
            # Beraberlik yanlılığı - takım stillerinden ve maç önemine göre
            "draw_bias": (home_factors.get("draw_bias", 0.0) + away_factors.get("draw_bias", 0.0)) / 2,
            
            # Skor varyansı - yüksek varyanslı takımlar daha değişken skorlar üretir
            "variance_factor": max(home_factors.get("variance_factor", 1.0), away_factors.get("variance_factor", 1.0)),
            
            # Büyük takım yanlılığı
            "big_team_bias": self._calculate_big_team_bias(home_team, away_team),
            
            # Lig-spesifik ayarlamalar
            "league_type": league_type,
            "score_inflation": self._get_league_adjustments(league_type).get("score_inflation", 1.0),
            "score_deflation": self._get_league_adjustments(league_type).get("score_deflation", 1.0),
            "home_advantage": self._get_league_adjustments(league_type).get("home_advantage", 1.0),
            "away_advantage": self._get_league_adjustments(league_type).get("away_advantage", 1.0),
            
            # Takım güç karşılaştırması
            "power_difference": team_power_comparison,
            
            # Form trendleri
            "home_trend": home_trend,
            "away_trend": away_trend,
            
            # Takım stilleri
            "home_team_style": {
                "defensive": home_style.get("defensive", 0),
                "offensive": home_style.get("offensive", 0),
                "possession": home_style.get("possession", 0),
                "defensive_factor": home_factors.get("defensive_factor", 0.9),
                "offensive_factor": home_factors.get("offensive_factor", 1.1),
                "possession_factor": home_factors.get("possession_factor", 1.05)
            },
            "away_team_style": {
                "defensive": away_style.get("defensive", 0),
                "offensive": away_style.get("offensive", 0),
                "possession": away_style.get("possession", 0),
                "defensive_factor": away_factors.get("defensive_factor", 0.9),
                "offensive_factor": away_factors.get("offensive_factor", 1.1),
                "possession_factor": away_factors.get("possession_factor", 1.05)
            },
            
            # Takım açıklamaları
            "team_descriptions": {
                "home": home_style.get("description", "Standart"),
                "away": away_style.get("description", "Standart")
            }
        }
        
        return adjustments
        
    def _determine_team_style_from_form(self, team_data):
        """
        Takımın form verilerinden oyun stilini belirler.
        
        Args:
            team_data: Takımın form verileri
            
        Returns:
            dict: Takım stili ve faktörleri
        """
        if not team_data or not team_data.get("recent_match_data"):
            return {"description": "Belirlenemedi", "factors": {}}
        
        matches = team_data.get("recent_match_data", [])
        if len(matches) < 3:
            return {"description": "Yetersiz veri", "factors": {}}
        
        # Temel istatistikleri hesapla
        total_matches = len(matches)
        goals_scored = sum(match.get("goals_scored", 0) for match in matches)
        goals_conceded = sum(match.get("goals_conceded", 0) for match in matches)
        clean_sheets = sum(1 for match in matches if match.get("goals_conceded", 0) == 0)
        high_scoring = sum(1 for match in matches if match.get("goals_scored", 0) + match.get("goals_conceded", 0) > 2.5)
        wins = sum(1 for match in matches if match.get("result", "") == "W")
        
        # Stil belirteçleri
        is_defensive = clean_sheets / total_matches > 0.35  # %35+ temiz kapanan maç
        is_offensive = goals_scored / total_matches > 1.5   # Maç başı 1.5+ gol
        high_variance = high_scoring / total_matches > 0.6  # %60+ maç yüksek skorlu
        successful = wins / total_matches > 0.5             # %50+ kazanma oranı
        
        # Stil ve faktörleri belirle
        style = {
            "defensive": round(clean_sheets / total_matches, 2),
            "offensive": round(goals_scored / total_matches / 1.5, 2),
            "possession": 0.5,  # Varsayılan değer, ileride geliştirilecek
            "description": ""
        }
        
        # Faktörleri hesapla
        factors = {
            "goal_factor": max(0.8, min(1.3, goals_scored / total_matches / 1.2)),
            "concede_factor": max(0.8, min(1.3, 1.0 / max(0.5, goals_conceded / total_matches))),
            "home_goal_factor": 1.0,
            "away_goal_factor": 1.0,
            "draw_bias": 0.0,
            "variance_factor": 1.0 + (high_variance * 0.2)
        }
        
        # Stil açıklaması
        if is_defensive and not is_offensive:
            style["description"] = "Defansif"
            factors["defensive_factor"] = 0.85
            factors["draw_bias"] = 0.15
        elif is_offensive and not is_defensive:
            style["description"] = "Ofansif"
            factors["offensive_factor"] = 1.15
            factors["defensive_factor"] = 1.0
        elif is_offensive and is_defensive:
            style["description"] = "Dengeli Güçlü"
            factors["defensive_factor"] = 0.9
            factors["offensive_factor"] = 1.1
        elif high_variance:
            style["description"] = "Değişken"
            factors["variance_factor"] = 1.2
        elif successful:
            style["description"] = "Etkili"
            factors["goal_factor"] = factors["goal_factor"] * 1.1
        else:
            style["description"] = "Standart"
        
        style["factors"] = factors
        return style
        
    def _analyze_team_power_difference(self, home_team, away_team):
        """
        İki takım arasındaki güç farkını analiz eder.
        
        Args:
            home_team: Ev sahibi takım verileri
            away_team: Deplasman takımı verileri
            
        Returns:
            float: Güç farkı (-1 ile 1 arasında, pozitif değer ev sahibi avantajını gösterir)
        """
        # Takımların büyüklüğünü kontrol et
        home_is_big = self.is_big_team(home_team.get("id"))
        away_is_big = self.is_big_team(away_team.get("id"))
        
        # Başlangıç güç değerleri
        home_power = 0.5
        away_power = 0.5
        
        # Büyük takım avantajı
        if home_is_big and not away_is_big:
            home_power += 0.15
        elif away_is_big and not home_is_big:
            away_power += 0.15
        
        # Takım stillerine göre güç değerini ayarla
        home_style = home_team.get("style", {})
        away_style = away_team.get("style", {})
        
        if home_style.get("description") == "Dengeli Güçlü":
            home_power += 0.1
        elif home_style.get("description") == "Ofansif" and away_style.get("description") == "Defansif":
            # Ofansif ev sahibi takım defansif deplasman takımına karşı avantajlı
            home_power += 0.05
        elif home_style.get("description") == "Defansif" and away_style.get("description") == "Ofansif":
            # Defansif ev sahibi takım ofansif deplasman takımına karşı dezavantajlı
            away_power += 0.05
        
        # Son güç değerlerini hesapla
        power_difference = home_power - away_power
        return round(power_difference, 2)
        
    def _analyze_team_trend(self, team_data):
        """
        Takımın son maçlardaki form trendini analiz eder.
        
        Args:
            team_data: Takım form verileri
            
        Returns:
            float: Trend değeri (-1 ile 1 arasında, pozitif değer yükselen form)
        """
        if not team_data or not team_data.get("recent_match_data"):
            return 0.0
        
        matches = team_data.get("recent_match_data", [])
        if len(matches) < 5:
            return 0.0
            
        # Sadece son 10 maçı değerlendir
        recent_matches = matches[:10]
        
        # İlk 5 ve son 5 maç performanslarını karşılaştır
        first_half = recent_matches[5:10] if len(recent_matches) > 5 else []
        second_half = recent_matches[:5]
        
        if not first_half:
            return 0.0
            
        # Puan hesabı (W=3, D=1, L=0)
        first_half_points = sum(3 if m.get("result") == "W" else 1 if m.get("result") == "D" else 0 for m in first_half)
        second_half_points = sum(3 if m.get("result") == "W" else 1 if m.get("result") == "D" else 0 for m in second_half)
        
        # Gol performansı
        first_half_goals = sum(m.get("goals_scored", 0) for m in first_half)
        second_half_goals = sum(m.get("goals_scored", 0) for m in second_half)
        
        # Kasa performansı
        first_half_conceded = sum(m.get("goals_conceded", 0) for m in first_half)
        second_half_conceded = sum(m.get("goals_conceded", 0) for m in second_half)
        
        # Trend hesaplama faktörleri
        points_trend = (second_half_points - first_half_points) / 15.0  # Maksimum puan farkı 15
        goals_trend = (second_half_goals - first_half_goals) / 10.0     # Normalleştirme faktörü
        defense_trend = (first_half_conceded - second_half_conceded) / 10.0  # Pozitif değer savunma iyileşmesi
        
        # Toplam trend değeri (-1 ile 1 arasında)
        total_trend = (points_trend * 0.5) + (goals_trend * 0.3) + (defense_trend * 0.2)
        return max(-1.0, min(1.0, total_trend))
    
    def get_team_data(self, team_id):
        """
        Takım verilerini döndürür.
        
        Args:
            team_id: Takım ID'si
            
        Returns:
            dict: Takım verileri
        """
        team_id = str(team_id)
        
        # Önbellekte varsa oradan al
        if team_id in self.team_cache:
            return self.team_cache[team_id]
        
        # Konfigürasyondan al
        if team_id in self.config["teams"]:
            # Ön belleğe al
            self.team_cache[team_id] = self.config["teams"][team_id]
            return self.config["teams"][team_id]
        
        return None
    
    def get_league_data(self, league_id):
        """
        Lig verilerini döndürür.
        
        Args:
            league_id: Lig ID'si
            
        Returns:
            dict: Lig verileri
        """
        league_id = str(league_id)
        
        # Önbellekte varsa oradan al
        if league_id in self.league_cache:
            return self.league_cache[league_id]
        
        # Konfigürasyondan al
        if league_id in self.config["leagues"]:
            # Ön belleğe al
            self.league_cache[league_id] = self.config["leagues"][league_id]
            return self.config["leagues"][league_id]
        
        return None
    
    def _calculate_big_team_bias(self, home_team, away_team):
        """
        Büyük takım bias değerini hesaplar.
        
        Args:
            home_team: Ev sahibi takım verileri
            away_team: Deplasman takım verileri
            
        Returns:
            dict: Büyük takım bias değerleri
        """
        home_is_big = home_team.get("is_big_team", False)
        away_is_big = away_team.get("is_big_team", False)
        
        if home_is_big and not away_is_big:
            # Ev sahibi büyük takım, deplasman değil
            return {
                "home_goals": 1.15,  # Ev sahibi büyük takımın gol atma avantajı
                "away_goals": 0.9,   # Deplasman takımının gol atma dezavantajı
                "home_win": 1.1,     # Ev sahibi kazanma olasılığı artışı
                "description": "Ev sahibi büyük takım"
            }
        elif away_is_big and not home_is_big:
            # Deplasman büyük takım, ev sahibi değil
            return {
                "home_goals": 0.95,  # Ev sahibi takımının gol atma dezavantajı
                "away_goals": 1.1,   # Deplasman büyük takımın gol atma avantajı
                "away_win": 1.15,    # Deplasman kazanma olasılığı artışı
                "description": "Deplasman büyük takım"
            }
        elif home_is_big and away_is_big:
            # İki takım da büyük
            return {
                "home_goals": 1.05,  # Ev sahibi avantajı ile küçük bir artış
                "away_goals": 1.05,  # Deplasman takımı da büyük olduğu için artış
                "tight_match": 1.1,  # Sıkı maç olasılığı artışı
                "description": "İki takım da büyük"
            }
        else:
            # İki takım da büyük değil
            return {
                "home_goals": 1.0,
                "away_goals": 1.0,
                "description": "Standart maç"
            }
    
    def _get_league_adjustments(self, league_type):
        """
        Lig türüne göre ayarlamaları döndürür.
        
        Args:
            league_type: Lig türü
            
        Returns:
            dict: Lig ayarlamaları
        """
        if league_type == "HIGH_SCORING":
            return {
                "goals_factor": 1.15,
                "over_2_5_bias": 1.2,
                "btts_bias": 1.15,
                "description": "Yüksek gollü lig"
            }
        elif league_type == "LOW_SCORING":
            return {
                "goals_factor": 0.9,
                "under_2_5_bias": 1.15,
                "btts_bias": 0.9,
                "description": "Düşük gollü lig"
            }
        elif league_type == "HIGH_HOME_ADVANTAGE":
            return {
                "home_win_bias": 1.2,
                "away_win_bias": 0.85,
                "description": "Ev sahibi avantajı yüksek lig"
            }
        elif league_type == "UNPREDICTABLE":
            return {
                "variance_factor": 1.2,
                "draw_bias": 1.1,
                "description": "Tahmin edilmesi zor lig"
            }
        else:
            return {
                "goals_factor": 1.0,
                "home_win_bias": 1.0,
                "away_win_bias": 1.0,
                "draw_bias": 1.0,
                "variance_factor": 1.0,
                "description": "Standart lig"
            }
    
    def _get_default_adjustments(self):
        """
        Varsayılan tahmin ayarlamalarını döndürür.
        
        Returns:
            dict: Varsayılan ayarlamalar
        """
        return {
            "home_goals_factor": 1.0,
            "home_concede_factor": 1.0,
            "away_goals_factor": 1.0,
            "away_concede_factor": 1.0,
            "draw_factor": 1.0,
            "variance_factor": 1.0,
            "big_team_bias": {
                "home_goals": 1.0,
                "away_goals": 1.0,
                "description": "Standart maç"
            },
            "league_adjustments": {
                "goals_factor": 1.0,
                "home_win_bias": 1.0,
                "away_win_bias": 1.0,
                "draw_bias": 1.0,
                "variance_factor": 1.0,
                "description": "Standart lig"
            },
            "team_descriptions": {
                "home": "Standart",
                "away": "Standart"
            }
        }