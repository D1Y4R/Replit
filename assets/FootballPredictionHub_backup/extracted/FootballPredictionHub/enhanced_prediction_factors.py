"""
Gelişmiş Tahmin Faktörleri Modülü

Bu modül, kesin skor tahmin doğruluğunu artırmak için ek faktörler sağlar:
1. Maç Önemi Analizi - Ligin durumuna göre takımların motivasyonunu ve stratejilerini hesaplar
2. Özel Maç Örüntü Tanıma - Takımlar arasındaki geçmiş karşılaşmaların skorlarını analiz eder
3. Momentum Analizi - Takımların son 90 günlük performans trendini hesaplar

Bu faktörler tahmin doğruluğunu artırmak için diğer modellerin sonuçlarıyla birleştirilebilir.
"""

import os
import json
import logging
import datetime
import math
import numpy as np
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPredictionFactors:
    """
    Kesin skor tahminleri için gelişmiş faktörler sunan sınıf.
    """
    def __init__(self, league_data_path="./league_data.json", historical_matches_path="./historical_matches.json"):
        """
        Gelişmiş tahmin faktörleri için gerekli verileri yükle.
        
        Args:
            league_data_path: Lig verilerinin bulunduğu dosya yolu (opsiyonel)
            historical_matches_path: Geçmiş maç verilerinin bulunduğu dosya yolu (opsiyonel)
        """
        self.league_data = self._load_league_data(league_data_path)
        self.historical_matches = self._load_historical_matches(historical_matches_path)
        logger.info("Gelişmiş Tahmin Faktörleri modülü başlatıldı")
        
    def _load_league_data(self, file_path):
        """
        Lig verilerini yükle veya dosya bulunamazsa varsayılan veri oluştur.
        
        Args:
            file_path: Lig verilerinin bulunduğu dosya yolu
            
        Returns:
            dict: Lig verileri sözlüğü
        """
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    logger.info(f"Lig verileri yüklendi: {file_path}")
                    return json.load(f)
            else:
                # Varsayılan veri oluştur
                logger.warning(f"Lig verileri dosyası bulunamadı: {file_path}, varsayılan veriler kullanılacak.")
                default_leagues = {
                    # İspanya La Liga
                    "140": {
                        "name": "La Liga",
                        "country": "Spain",
                        "current_week": self._estimate_week_number(),
                        "total_weeks": 38,
                        "teams_count": 20,
                        "standings": {}  # Burada takımların sıralamaları olacak
                    },
                    # İngiltere Premier Lig
                    "39": {
                        "name": "Premier League",
                        "country": "England",
                        "current_week": self._estimate_week_number(),
                        "total_weeks": 38,
                        "teams_count": 20,
                        "standings": {} 
                    },
                    # Türkiye Süper Lig
                    "203": {
                        "name": "Süper Lig",
                        "country": "Turkey",
                        "current_week": self._estimate_week_number(),
                        "total_weeks": 38,
                        "teams_count": 20,
                        "standings": {}
                    },
                    # İtalya Serie A
                    "135": {
                        "name": "Serie A",
                        "country": "Italy",
                        "current_week": self._estimate_week_number(),
                        "total_weeks": 38,
                        "teams_count": 20,
                        "standings": {}
                    },
                    # Almanya Bundesliga
                    "78": {
                        "name": "Bundesliga",
                        "country": "Germany",
                        "current_week": self._estimate_week_number(),
                        "total_weeks": 34,
                        "teams_count": 18,
                        "standings": {}
                    },
                    # Fransa Ligue 1
                    "61": {
                        "name": "Ligue 1",
                        "country": "France",
                        "current_week": self._estimate_week_number(),
                        "total_weeks": 38,
                        "teams_count": 20,
                        "standings": {}
                    }
                }
                return default_leagues
        except Exception as e:
            logger.error(f"Lig verileri yüklenirken hata: {str(e)}")
            return {}
        
    def _load_historical_matches(self, file_path):
        """
        Geçmiş maç verilerini yükle veya dosya bulunamazsa boş liste döndür.
        
        Args:
            file_path: Geçmiş maç verilerinin bulunduğu dosya yolu
            
        Returns:
            list: Geçmiş maç verileri listesi
        """
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    logger.info(f"Geçmiş maç verileri yüklendi: {file_path}")
                    return json.load(f)
            else:
                logger.warning(f"Geçmiş maç verileri dosyası bulunamadı: {file_path}, boş liste kullanılacak.")
                return []
        except Exception as e:
            logger.error(f"Geçmiş maç verileri yüklenirken hata: {str(e)}")
            return []
        
    def _estimate_week_number(self):
        """
        Ligin mevcut hafta numarasını tahmin et.
        Genellikle ağustos ayında yeni sezon başladığını varsayarak.
        
        Returns:
            int: Tahmini lig hafta numarası (1-38 arası)
        """
        now = datetime.datetime.now()
        # Ağustos 1'i sezon başlangıcı olarak kabul et
        season_start = datetime.datetime(now.year if now.month >= 8 else now.year - 1, 8, 1)
        days_passed = (now - season_start).days
        
        # Ortalama olarak, her 7 günde bir hafta maçı oynanır
        estimated_week = min(38, max(1, int(days_passed / 7) + 1))
        
        # Kış arası ve bazı liglerde daha az hafta olduğunu hesaba kat
        if estimated_week > 19 and now.month in [12, 1]:  # Kış arası
            estimated_week = min(estimated_week, 17)  # Genellikle 17. haftada kış arası
        
        return estimated_week
        
    def get_team_position(self, team_id, league_id=None):
        """
        Takımın ligdeki mevcut pozisyonunu döndür.
        
        Args:
            team_id: Takım ID'si
            league_id: Lig ID'si (opsiyonel, belirtilmezse otomatik bulunmaya çalışılır)
            
        Returns:
            int: Takımın ligdeki pozisyonu (1 = ilk sırada, vb.) veya veri yoksa 10 (orta sıra)
        """
        # Lig ID'si verilmediyse, tahmini bir değer kullan
        if not league_id:
            # Takım ID'sine göre tahmin et - belirli takım ID aralıkları belirli liglere ait olabilir
            # Bu sadece bir örnek, gerçek veriye göre ayarlanmalı
            if str(team_id) in ["610", "614", "611", "636", "1005"]:  # Türk takımları
                league_id = "203"  # Türkiye Süper Lig
            elif int(team_id) < 100:  # İngiltere tahmin
                league_id = "39"  # Premier Lig
            elif int(team_id) < 200:  # İspanya tahmin
                league_id = "140"  # La Liga
            elif int(team_id) < 300:  # İtalya tahmin
                league_id = "135"  # Serie A
            elif int(team_id) < 400:  # Almanya tahmin
                league_id = "78"  # Bundesliga
            elif int(team_id) < 500:  # Fransa tahmin
                league_id = "61"  # Ligue 1
            else:
                # Varsayılan lig
                league_id = "203"  # Varsayılan olarak Türkiye Süper Lig
        
        # İlgili lig verisini bul
        league = self.league_data.get(str(league_id), {})
        
        # Takımın pozisyonunu bul
        standings = league.get("standings", {})
        
        if str(team_id) in standings:
            return standings[str(team_id)].get("position", 10)  # Pozisyon yoksa 10 (orta sıra) döndür
        else:
            # Veri yoksa, orta sıra varsay
            return 10
        
    def get_league_size(self, league_id):
        """
        Ligdeki takım sayısını döndür.
        
        Args:
            league_id: Lig ID'si
            
        Returns:
            int: Ligdeki takım sayısı veya veri yoksa 20 (varsayılan büyük lig)
        """
        league = self.league_data.get(str(league_id), {})
        return league.get("teams_count", 20)  # Varsayılan büyük lig
        
    def calculate_match_importance(self, home_team_id, away_team_id, league_id=None, date=None):
        """
        Maçın önemi faktörünü hesapla.
        Sezonun ilerleyen dönemlerinde ve takımların lig pozisyonlarına göre değişir.
        
        Args:
            home_team_id: Ev sahibi takım ID'si
            away_team_id: Deplasman takımı ID'si
            league_id: Lig ID'si (opsiyonel)
            date: Maç tarihi (opsiyonel, None ise güncel tarih kullanılır)
            
        Returns:
            dict: Maç önem faktörü
        """
        now = datetime.datetime.now() if date is None else date
        
        # Lig bilgisini bul
        if league_id and str(league_id) in self.league_data:
            league = self.league_data[str(league_id)]
        else:
            # Takım ID'leri üzerinden lig tahmin et (varsayılan olarak)
            if str(home_team_id) in ["610", "614", "611", "636", "1005"]:  # Türk takımları
                league_id = "203"  # Türkiye Süper Lig
                league = self.league_data.get(league_id, {})
            else:
                # Varsayılan değerler kullan
                league = {"current_week": self._estimate_week_number(), "total_weeks": 38}
        
        current_week = league.get("current_week", self._estimate_week_number())
        total_weeks = league.get("total_weeks", 38)
        
        # Takımların lig pozisyonlarını al
        home_position = self.get_team_position(home_team_id, league_id)
        away_position = self.get_team_position(away_team_id, league_id)
        league_size = self.get_league_size(league_id)
        
        # Sezon sonu etkisi (son 8 hafta)
        season_end_effect = 1.0
        if current_week > (total_weeks - 8):
            # Sezon sonuna yaklaştıkça önem artar
            weeks_to_end = total_weeks - current_week
            season_end_effect = 1.0 + ((8 - weeks_to_end) / 8) * 0.5  # Sezon sonunda %50 daha önemli
        
        # Takımların ligi bitirme pozisyonlarının potansiyel önemi
        home_importance = 1.0
        away_importance = 1.0
        
        # Üst sıra mücadelesi (ilk 4 veya 6)
        # Çoğu ligde ilk 4 şampiyonlar ligi, 5-6 avrupa ligi
        if home_position <= 6:
            home_importance += (7 - home_position) * 0.05  # Daha üst sıradaysa daha önemli
        if away_position <= 6:
            away_importance += (7 - away_position) * 0.05  # Daha üst sıradaysa daha önemli
            
        # Düşme hattı mücadelesi (son 3 veya 4)
        if home_position >= (league_size - 4):
            home_importance += (home_position - (league_size - 5)) * 0.1  # Düşme hattında daha önemli
        if away_position >= (league_size - 4):
            away_importance += (away_position - (league_size - 5)) * 0.1  # Düşme hattında daha önemli
            
        # Derbi veya büyük maç etkisi (takım ID'lerine göre)
        derby_effect = 1.0
        
        # Türk futbolu derbileri
        turkish_big_teams = ["610", "614", "611", "636", "1005"]  # GS, BJK, TS, FB
        if str(home_team_id) in turkish_big_teams and str(away_team_id) in turkish_big_teams:
            derby_effect = 1.4  # Türk derbileri çok önemli
            logger.info(f"Türk derbisi tespit edildi: {str(home_team_id)} vs {str(away_team_id)}")
            
        # Maçın toplam önemi
        match_importance = (home_importance + away_importance) / 2 * season_end_effect * derby_effect
        
        # Açıklama oluştur
        description = ""
        factors = []
        
        if season_end_effect > 1.05:
            factors.append(f"sezon sonuna yaklaşım (x{season_end_effect:.2f})")
        
        if home_importance > 1.05:
            if home_position <= 6:
                factors.append(f"ev sahibi üst sıra mücadelesi ({home_position}. sıra)")
            if home_position >= (league_size - 4):
                factors.append(f"ev sahibi düşme hattı mücadelesi ({home_position}. sıra)")
        
        if away_importance > 1.05:
            if away_position <= 6:
                factors.append(f"deplasman üst sıra mücadelesi ({away_position}. sıra)")
            if away_position >= (league_size - 4):
                factors.append(f"deplasman düşme hattı mücadelesi ({away_position}. sıra)")
        
        if derby_effect > 1.05:
            factors.append(f"derbi/büyük maç etkisi (x{derby_effect:.2f})")
        
        if factors:
            description = "Maç önemi yüksek: " + ", ".join(factors)
        else:
            description = "Normal maç önemi (standart lig maçı)"
        
        return {
            "factor": match_importance,
            "description": description,
            "home_importance": home_importance,
            "away_importance": away_importance,
            "derby_effect": derby_effect,
            "season_end_effect": season_end_effect
        }
        
    def get_historical_matchups(self, home_team_id, away_team_id, num_years=3):
        """
        İki takım arasındaki geçmiş karşılaşmaları bul.
        
        Args:
            home_team_id: Ev sahibi takım ID'si
            away_team_id: Deplasman takımı ID'si
            num_years: Kaç yıl geriye gidileceği (varsayılan: 3)
            
        Returns:
            list: Geçmiş karşılaşmaların listesi
        """
        now = datetime.datetime.now()
        cutoff_date = now - datetime.timedelta(days=num_years * 365)
        
        # Geçmiş maçları filtrele
        matchups = []
        
        for match in self.historical_matches:
            # Tarih formatına dikkat et ve doğru parse et
            match_date = None
            try:
                if 'date' in match:
                    # Tarih formatına bağlı olarak parse et
                    if isinstance(match['date'], str):
                        try:
                            match_date = datetime.datetime.strptime(match['date'], "%Y-%m-%d")
                        except ValueError:
                            try:
                                match_date = datetime.datetime.strptime(match['date'], "%d.%m.%Y")
                            except ValueError:
                                pass
            except Exception as e:
                logger.warning(f"Tarih ayrıştırma hatası: {str(e)}, match={match.get('date', 'No date')}")
                continue
                
            # Tarih yoksa veya ayrıştırılamadıysa, atla
            if not match_date:
                continue
                
            # Karşılaşma tarihimizden yeni ise
            if match_date >= cutoff_date:
                # Her iki yönde de maçı kontrol et (ev sahibi/deplasman değişebilir)
                if (str(match.get('home_team_id', '')) == str(home_team_id) and 
                    str(match.get('away_team_id', '')) == str(away_team_id)):
                    matchups.append(match)
                elif (str(match.get('home_team_id', '')) == str(away_team_id) and 
                      str(match.get('away_team_id', '')) == str(home_team_id)):
                    # Ev/deplasmanı ters çevir ve ekle
                    reversed_match = match.copy()
                    reversed_match['is_reversed'] = True  # Ters çevrildiğini belirt
                    matchups.append(reversed_match)
        
        return matchups
        
    def analyze_historical_pattern(self, home_team_id, away_team_id, num_years=3):
        """
        İki takım arasındaki geçmiş karşılaşmaların skor örüntülerini analiz et.
        
        Args:
            home_team_id: Ev sahibi takım ID'si
            away_team_id: Deplasman takımı ID'si
            num_years: Kaç yıl geriye gidileceği (varsayılan: 3)
            
        Returns:
            dict: Skor örüntü analizi
        """
        matchups = self.get_historical_matchups(home_team_id, away_team_id, num_years)
        
        if not matchups:
            return {
                "has_pattern": False,
                "description": "Geçmiş karşılaşma verisi bulunamadı",
                "score_adjustment": (1.0, 1.0)  # Ayarlama yok (ev, deplasman çarpanı)
            }
        
        # Skorları analiz et
        scores = []
        total_home_goals = 0
        total_away_goals = 0
        home_wins = 0
        away_wins = 0
        draws = 0
        
        for match in matchups:
            # Veriyi doğru şekilde işle
            home_goals = match.get('home_score', 0)
            away_goals = match.get('away_score', 0)
            
            # Sayı değeri olduğundan emin ol
            try:
                home_goals = int(home_goals)
                away_goals = int(away_goals)
            except (TypeError, ValueError):
                continue
            
            # Eğer ters çevrilmiş maç ise, skorları da ters çevir
            if match.get('is_reversed', False):
                home_goals, away_goals = away_goals, home_goals
            
            total_home_goals += home_goals
            total_away_goals += away_goals
            
            if home_goals > away_goals:
                home_wins += 1
            elif home_goals < away_goals:
                away_wins += 1
            else:
                draws += 1
            
            scores.append((home_goals, away_goals))
        
        # En sık tekrarlanan skorları bul
        score_counter = Counter(scores)
        most_common_scores = score_counter.most_common(3)  # En yaygın 3 skor
        
        # Ortalama skorları hesapla
        num_matches = len(scores)
        avg_home_goals = total_home_goals / num_matches if num_matches > 0 else 0
        avg_away_goals = total_away_goals / num_matches if num_matches > 0 else 0
        
        # Belirli bir örüntü var mı?
        pattern_threshold = 0.5  # %50 veya daha fazla aynı skor ise örüntü vardır
        has_pattern = False
        pattern_description = ""
        score_adjustment = (1.0, 1.0)  # Varsayılan değer
        
        # En yaygın skorun toplam maçlara oranı
        if most_common_scores and num_matches >= 3:  # En az 3 maç
            most_common_score, most_common_count = most_common_scores[0]
            most_common_ratio = most_common_count / num_matches
            
            if most_common_ratio >= pattern_threshold:
                has_pattern = True
                pattern_home_goals, pattern_away_goals = most_common_score
                pattern_description = f"Belirgin skor örüntüsü: {pattern_home_goals}-{pattern_away_goals} skoru {most_common_count}/{num_matches} maçta tekrarlanmış ({most_common_ratio*100:.0f}%)"
                
                # Örüntüye uygun skor ayarlaması
                if avg_home_goals > 0 and avg_away_goals > 0:
                    score_adjustment = (
                        pattern_home_goals / avg_home_goals,
                        pattern_away_goals / avg_away_goals
                    )
            else:
                # Genel eğilim analizi (ev-away denge)
                if num_matches >= 3:
                    # Toplam gol üretimi
                    if avg_home_goals > 0 or avg_away_goals > 0:
                        pattern_description = f"Son {num_matches} karşılaşmada ortalama skor: {avg_home_goals:.1f}-{avg_away_goals:.1f}"
                        
                        # Sonuç dağılımına göre ek bilgi
                        if home_wins > away_wins and home_wins > draws:
                            pattern_description += f", ev sahibi üstünlüğü ({home_wins}/{num_matches})"
                            # Ev sahibi üstünlüğünü vurgula (en fazla %20)
                            score_adjustment = (min(1.2, 1.0 + (home_wins/num_matches) * 0.3), 
                                              max(0.8, 1.0 - (home_wins/num_matches) * 0.3))
                        elif away_wins > home_wins and away_wins > draws:
                            pattern_description += f", deplasman üstünlüğü ({away_wins}/{num_matches})"
                            # Deplasman üstünlüğünü vurgula (en fazla %20)
                            score_adjustment = (max(0.8, 1.0 - (away_wins/num_matches) * 0.3),
                                              min(1.2, 1.0 + (away_wins/num_matches) * 0.3))
                        elif draws > home_wins and draws > away_wins:
                            pattern_description += f", beraberlik eğilimi ({draws}/{num_matches})"
                            # Beraberlik eğilimini vurgula (skorları birbirine yaklaştır)
                            if avg_home_goals > avg_away_goals:
                                score_adjustment = (
                                    max(0.9, 1.0 - (draws/num_matches) * 0.2),
                                    min(1.1, 1.0 + (draws/num_matches) * 0.2)
                                )
                            else:
                                score_adjustment = (
                                    min(1.1, 1.0 + (draws/num_matches) * 0.2),
                                    max(0.9, 1.0 - (draws/num_matches) * 0.2)
                                )
                    else:
                        pattern_description = f"Son {num_matches} karşılaşmada belirgin bir örüntü yok"
                else:
                    pattern_description = f"Sadece {num_matches} karşılaşma verisi mevcut, yeterli örüntü analizi yapılamıyor"
        else:
            if num_matches > 0:
                pattern_description = f"Sadece {num_matches} karşılaşma verisi mevcut, yeterli örüntü analizi yapılamıyor"
            else:
                pattern_description = "Geçmiş karşılaşma verisi bulunamadı"
        
        return {
            "has_pattern": has_pattern,
            "description": pattern_description,
            "score_adjustment": score_adjustment,
            "avg_home_goals": avg_home_goals,
            "avg_away_goals": avg_away_goals,
            "num_matches": num_matches,
            "home_wins": home_wins,
            "away_wins": away_wins,
            "draws": draws,
            "most_common_scores": most_common_scores
        }
        
    def calculate_momentum(self, team_id, recent_matches, days=90):
        """
        Takımın momentum skorunu hesapla.
        
        Args:
            team_id: Takım ID'si
            recent_matches: Takımın son maçları
            days: Kaç günlük maçları analiz edeceği (varsayılan: 90)
            
        Returns:
            float: 0.7 ile 1.3 arasında momentum faktörü (1.0 = nötr momentum)
        """
        now = datetime.datetime.now()
        cutoff_date = now - datetime.timedelta(days=days)
        
        # Son maçlar listesi
        matches = []
        
        # Verileri hazırla
        if isinstance(recent_matches, list):
            # Zaten liste formatındaysa doğrudan kullan
            matches = recent_matches
        elif isinstance(recent_matches, dict):
            # Form sözlüğünden son maçları çıkar
            matches = recent_matches.get('recent_match_data', [])
        else:
            # Veri formatı uygun değilse boş liste kullan
            logger.warning(f"Geçersiz form veri formatı: {type(recent_matches)}")
            matches = []
        
        # Yeterli maç yoksa nötr momentum döndür
        if not matches:
            return 1.0
        
        # Güncelliğe göre ağırlıklı puan hesapla
        total_weight = 0
        weighted_score = 0
        
        for i, match in enumerate(matches[:10]):  # Son 10 maçı kullan (daha fazla varsa)
            # Sonucu kontrol et
            result = match.get('result', 'unknown')
            
            # Zamanı kontrol et (yoksa son maçlardan sayıldığını varsay)
            match_date = None
            if 'date' in match:
                try:
                    if isinstance(match['date'], str):
                        try:
                            match_date = datetime.datetime.strptime(match['date'], "%Y-%m-%d")
                        except ValueError:
                            try:
                                match_date = datetime.datetime.strptime(match['date'], "%d.%m.%Y")
                            except ValueError:
                                pass
                except Exception:
                    pass
            
            # Tarih yoksa varsayılan olarak güncel kabul et
            is_recent = True
            if match_date and match_date < cutoff_date:
                is_recent = False
            
            if not is_recent:
                continue
            
            # Güncelliğe göre ağırlık hesapla (daha yeni maçlar daha önemli)
            recency_weight = 1.0 - (i * 0.05)  # Her bir eski maç için %5 azalt
            recency_weight = max(0.5, recency_weight)  # En az %50 ağırlık
            
            # Sonuca göre puan
            match_score = 0
            if result == 'win':
                match_score = 1.0
            elif result == 'draw':
                match_score = 0.5
            elif result == 'loss':
                match_score = 0.0
            else:
                # Sonuç bilinmiyorsa, gol farklarına bakarak belirle
                goals_scored = match.get('goals_scored', 0)
                goals_conceded = match.get('goals_conceded', 0)
                
                try:
                    goals_scored = int(goals_scored)
                    goals_conceded = int(goals_conceded)
                    
                    if goals_scored > goals_conceded:
                        match_score = 1.0
                    elif goals_scored == goals_conceded:
                        match_score = 0.5
                    else:
                        match_score = 0.0
                except (TypeError, ValueError):
                    # Sayısal veri yoksa nötr puan
                    match_score = 0.5
            
            weighted_score += match_score * recency_weight
            total_weight += recency_weight
        
        # Yeterli maç yoksa nötr momentum döndür
        if total_weight == 0:
            return 1.0
        
        # Ağırlıklı ortalama (0-1 arası)
        avg_score = weighted_score / total_weight
        
        # 0.7 ile 1.3 arasında momentum faktörüne dönüştür
        # 0.5 -> 1.0 (nötr momentum)
        # 0.0 -> 0.7 (negatif momentum)
        # 1.0 -> 1.3 (pozitif momentum)
        momentum = 1.0 + (avg_score - 0.5) * 0.6
        
        return momentum
        
    def get_enhanced_prediction_factors(self, home_team_id, away_team_id, home_recent_matches, away_recent_matches, league_id=None):
        """
        Bir maç için gelişmiş tahmin faktörlerini hesaplar ve döndürür.
        
        Args:
            home_team_id: Ev sahibi takım ID'si
            away_team_id: Deplasman takımı ID'si
            home_recent_matches: Ev sahibi takımın son maçları
            away_recent_matches: Deplasman takımının son maçları
            league_id: Lig ID'si (opsiyonel)
            
        Returns:
            dict: Gelişmiş tahmin faktörleri
        """
        # Maç önemi analizi
        match_importance = self.calculate_match_importance(home_team_id, away_team_id, league_id)
        
        # Önceki maç örüntü analizi
        historical_pattern = self.analyze_historical_pattern(home_team_id, away_team_id)
        
        # Momentum analizi
        home_momentum = self.calculate_momentum(home_team_id, home_recent_matches)
        away_momentum = self.calculate_momentum(away_team_id, away_recent_matches)
        
        # Momentumları logla
        logger.info(f"Momentum faktörleri: Ev={home_momentum:.2f}, Deplasman={away_momentum:.2f}")
        
        # Tüm faktörleri birleştir
        enhanced_factors = {
            "match_importance": match_importance,
            "historical_pattern": historical_pattern,
            "momentum": {
                "home_momentum": home_momentum,
                "away_momentum": away_momentum,
                "description": f"Ev momentum: {home_momentum:.2f}, Deplasman momentum: {away_momentum:.2f}"
            }
        }
        
        return enhanced_factors
        
    def adjust_score_prediction(self, base_home_goals, base_away_goals, enhanced_factors):
        """
        Temel gol beklentilerini gelişmiş faktörlere göre ayarla.
        
        Args:
            base_home_goals: Ev sahibi takımın temel gol beklentisi
            base_away_goals: Deplasman takımının temel gol beklentisi
            enhanced_factors: Gelişmiş tahmin faktörleri
            
        Returns:
            tuple: Ayarlanmış (ev_sahibi_goller, deplasman_goller)
        """
        adjusted_home_goals = base_home_goals
        adjusted_away_goals = base_away_goals
        
        # Başlangıç değerlerini kaydet
        original_home_goals = base_home_goals
        original_away_goals = base_away_goals
        
        # 1. Maç önemi faktörü
        if "match_importance" in enhanced_factors:
            importance_factor = enhanced_factors["match_importance"]["factor"]
            
            # Maç önemi her iki takım için toplam gol beklentisini artırır
            # Çok önemli maçlarda takımlar daha motive olur ve daha çok risk alırlar
            if importance_factor > 1.1:  # %10'dan fazla önem
                # Maç önemli ise, daha fazla gol beklentisi (en fazla %20 artış)
                goal_multiplier = min(1.2, 1.0 + (importance_factor - 1.0))
                
                # İki takımın önemini ayrı ayrı değerlendir
                home_importance = enhanced_factors["match_importance"].get("home_importance", 1.0)
                away_importance = enhanced_factors["match_importance"].get("away_importance", 1.0)
                
                # Daha önemli olan takım lehine hafif bir ağırlık ver
                if home_importance > away_importance:
                    home_multiplier = goal_multiplier * (1.0 + (home_importance - away_importance) * 0.1)
                    away_multiplier = goal_multiplier
                elif away_importance > home_importance:
                    home_multiplier = goal_multiplier
                    away_multiplier = goal_multiplier * (1.0 + (away_importance - home_importance) * 0.1)
                else:
                    home_multiplier = goal_multiplier
                    away_multiplier = goal_multiplier
                
                adjusted_home_goals *= home_multiplier
                adjusted_away_goals *= away_multiplier
                
                logger.info(f"Maç önemi faktörü ({importance_factor:.2f}) uygulandı: "
                          f"Ev {original_home_goals:.2f}->{adjusted_home_goals:.2f}, "
                          f"Deplasman {original_away_goals:.2f}->{adjusted_away_goals:.2f}")
        
        # 2. Tarihsel örüntü faktörü
        if "historical_pattern" in enhanced_factors:
            pattern = enhanced_factors["historical_pattern"]
            
            # Örüntü varsa ayarla
            if pattern["has_pattern"] or pattern["num_matches"] >= 3:
                home_adj, away_adj = pattern["score_adjustment"]
                
                # Ayarlama faktörlerini uygula (maksimum etki sınırlaması ile)
                home_adj = max(0.7, min(1.3, home_adj))  # %30 artış/azalış limiti
                away_adj = max(0.7, min(1.3, away_adj))  # %30 artış/azalış limiti
                
                original_home_goals = adjusted_home_goals  # Mevcut değeri kaydet
                original_away_goals = adjusted_away_goals  # Mevcut değeri kaydet
                
                adjusted_home_goals *= home_adj
                adjusted_away_goals *= away_adj
                
                logger.info(f"Tarihsel örüntü faktörü uygulandı: "
                          f"Ev {original_home_goals:.2f}->{adjusted_home_goals:.2f}, "
                          f"Deplasman {original_away_goals:.2f}->{adjusted_away_goals:.2f}")
        
        # 3. Momentum faktörü
        if "momentum" in enhanced_factors:
            home_momentum = enhanced_factors["momentum"]["home_momentum"]
            away_momentum = enhanced_factors["momentum"]["away_momentum"]
            
            # Momentumlar nötr (1.0) değerinden uzaksa etki et
            if abs(home_momentum - 1.0) > 0.05 or abs(away_momentum - 1.0) > 0.05:
                original_home_goals = adjusted_home_goals  # Mevcut değeri kaydet
                original_away_goals = adjusted_away_goals  # Mevcut değeri kaydet
                
                # Momentum faktörlerini uygula
                adjusted_home_goals *= home_momentum
                adjusted_away_goals *= away_momentum
                
                logger.info(f"Momentum faktörleri uygulandı (Ev: {home_momentum:.2f}, Deplasman: {away_momentum:.2f}): "
                          f"Ev {original_home_goals:.2f}->{adjusted_home_goals:.2f}, "
                          f"Deplasman {original_away_goals:.2f}->{adjusted_away_goals:.2f}")
                
        # Aşırı değer kontrolü
        MAX_ADJUSTMENT = 1.5  # Orijinal değerin en fazla %50 artışı/azalışı
        
        if adjusted_home_goals > base_home_goals * MAX_ADJUSTMENT:
            adjusted_home_goals = base_home_goals * MAX_ADJUSTMENT
            logger.warning(f"Ev gol beklentisi aşırı yüksek, sınırlandırıldı: {adjusted_home_goals:.2f}")
            
        if adjusted_away_goals > base_away_goals * MAX_ADJUSTMENT:
            adjusted_away_goals = base_away_goals * MAX_ADJUSTMENT
            logger.warning(f"Deplasman gol beklentisi aşırı yüksek, sınırlandırıldı: {adjusted_away_goals:.2f}")
            
        if adjusted_home_goals < base_home_goals / MAX_ADJUSTMENT:
            adjusted_home_goals = base_home_goals / MAX_ADJUSTMENT
            logger.warning(f"Ev gol beklentisi aşırı düşük, sınırlandırıldı: {adjusted_home_goals:.2f}")
            
        if adjusted_away_goals < base_away_goals / MAX_ADJUSTMENT:
            adjusted_away_goals = base_away_goals / MAX_ADJUSTMENT
            logger.warning(f"Deplasman gol beklentisi aşırı düşük, sınırlandırıldı: {adjusted_away_goals:.2f}")
        
        # Toplam ayarlama etkisini logla
        total_home_change = adjusted_home_goals / base_home_goals
        total_away_change = adjusted_away_goals / base_away_goals
        logger.info(f"Gelişmiş faktörler sonrası toplam değişim: "
                  f"Ev x{total_home_change:.2f}, Deplasman x{total_away_change:.2f}")
        
        return adjusted_home_goals, adjusted_away_goals

# Singleton pattern - aynı örneği her yerden kullanabilmek için
_instance = None

def get_instance():
    """
    Sınıfın global örneğini döndür (Singleton pattern).
    """
    global _instance
    if _instance is None:
        _instance = EnhancedPredictionFactors()
    return _instance