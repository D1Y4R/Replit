"""
Gol Trend İvmesi Analiz Modülü

Bu modül, takımların son maçlarındaki gol atma ve gol yeme trendlerinin değişim 
ivmesini analiz eder. Takımın yükselen, düşen veya stabil gol formu, tahmin
doğruluğunu artırmak için kullanılır.

Özellikler:
1. Gol İvmesi Analizi: Takımın son maçlardaki gol atma/yeme eğilimini ölçer
2. Gol Patlaması Tespiti: Son maçlarda ani gol artışları veya azalışlarını tespit eder
3. Rakip Bazlı Gol Analizi: Rakip gücüne göre düzeltilmiş gol trendi
4. Gol İvme Faktörü: Kesin skor tahminlerini etkileyen sayısal faktörler üretir
"""

import numpy as np
import logging
import math
from datetime import datetime, timedelta
from collections import defaultdict

# Logging yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoalTrendAnalyzer:
    """
    Takımların gol atma ve yeme trendlerinin ivmesini analiz eden sınıf.
    """
    def __init__(self):
        """
        Gol trendi analizörünü başlat
        """
        self.team_stats_cache = {}  # Takım istatistikleri önbelleği
        self.trend_results_cache = {}  # Trend analiz sonuçları önbelleği
        self.trend_half_life = 5  # Son 5 maç daha yüksek ağırlığa sahip (yarılanma)
        logger.info("Gol Trend Analizörü başlatıldı")
        
    def _exponential_weight(self, match_index, half_life=5):
        """
        Maçlara üstel azalma ağırlığı uygula (yeni maçlar daha önemli)
        
        Args:
            match_index: Maç indeksi (0 = en yeni maç)
            half_life: Yarılanma periyodu
            
        Returns:
            float: Ağırlık faktörü
        """
        return math.exp(-match_index / half_life)
    
    def _calculate_weighted_goals(self, goals_list, weights=None):
        """
        Maçlardaki gollerin ağırlıklı ortalamasını hesapla
        
        Args:
            goals_list: Maçlardaki gol sayıları listesi
            weights: Ağırlıklar listesi, None ise üstel ağırlık kullanılır
            
        Returns:
            float: Ağırlıklı ortalama gol sayısı
        """
        if not goals_list:
            return 0.0
            
        if weights is None:
            weights = [self._exponential_weight(i, self.trend_half_life) for i in range(len(goals_list))]
            
        weighted_sum = sum(g * w for g, w in zip(goals_list, weights))
        sum_weights = sum(weights)
        
        return weighted_sum / sum_weights if sum_weights > 0 else 0.0
    
    def _calculate_trend_acceleration(self, goals_list, window_size=3):
        """
        Gol trendindeki ivmeyi hesapla
        
        Args:
            goals_list: Maçlardaki gol sayıları listesi
            window_size: Karşılaştırılacak pencere boyutu
            
        Returns:
            dict: Trend ivmesi analiz sonuçları
        """
        if len(goals_list) < window_size * 2:
            # Yeterli veri yoksa sıfır ivme göster
            return {
                "trend": "stable",
                "acceleration": 0.0,
                "confidence": 0.5,
                "recent_avg": np.mean(goals_list[:window_size]) if goals_list else 0.0,
                "previous_avg": np.mean(goals_list[window_size:window_size*2]) if len(goals_list) >= window_size*2 else 0.0
            }
            
        # Son pencere ve önceki pencere maçlarındaki ortalama gol sayıları
        recent_window = goals_list[:window_size]
        previous_window = goals_list[window_size:window_size*2]
        
        recent_avg = np.mean(recent_window)
        previous_avg = np.mean(previous_window)
        
        # İvme: Son ortalama - önceki ortalama
        acceleration = recent_avg - previous_avg
        
        # İvme yönü ve büyüklüğüne göre trend belirle
        if abs(acceleration) < 0.3:  # 0.3 gol değişim eşiği
            trend = "stable"
        elif acceleration > 0:
            trend = "rising"
        else:
            trend = "falling"
            
        # Veri miktarına göre güven faktörü
        confidence = min(1.0, len(goals_list) / 10)  # Maksimum 10 maç için tam güven
        
        return {
            "trend": trend,
            "acceleration": acceleration,
            "confidence": confidence,
            "recent_avg": recent_avg,
            "previous_avg": previous_avg
        }
        
    def _detect_goal_burst(self, goals_list, threshold=1.5):
        """
        Ani gol patlaması veya düşüşü tespit et
        
        Args:
            goals_list: Maçlardaki gol sayıları listesi
            threshold: Patlama/düşüş eşik faktörü
            
        Returns:
            dict: Patlama/düşüş analiz sonuçları
        """
        if not goals_list or len(goals_list) < 3:
            return {"detected": False}
            
        # Son 3 maçın ortalaması
        recent_avg = np.mean(goals_list[:3])
        
        # Önceki 3-10 maçın ortalaması (varsa)
        if len(goals_list) >= 6:
            previous_avg = np.mean(goals_list[3:min(10, len(goals_list))])
        else:
            previous_avg = np.mean(goals_list[3:]) if len(goals_list) > 3 else recent_avg
            
        # Ortalamalar 0'a yakınsa karşılaştırma yanıltıcı olabilir
        if previous_avg < 0.1:
            previous_avg = 0.1  # Bölme hatasını önlemek için
            
        # Oran hesapla
        ratio = recent_avg / previous_avg
        
        # Patlama veya düşüş var mı?
        burst_detected = ratio >= threshold
        slump_detected = ratio <= (1.0 / threshold)
        
        return {
            "detected": burst_detected or slump_detected,
            "type": "burst" if burst_detected else "slump" if slump_detected else "normal",
            "ratio": ratio,
            "recent_avg": recent_avg,
            "previous_avg": previous_avg
        }
        
    def _adjust_for_opponent_strength(self, goals_list, opponent_strength_list):
        """
        Rakip gücüne göre gol verilerini düzelt
        
        Args:
            goals_list: Maçlardaki gol sayıları listesi
            opponent_strength_list: Rakip güçleri listesi (0-1 arası)
            
        Returns:
            list: Rakip gücüne göre düzeltilmiş gol listesi
        """
        if not goals_list or not opponent_strength_list or len(goals_list) != len(opponent_strength_list):
            return goals_list
            
        # Rakip gücüne göre düzeltilmiş goller
        adjusted_goals = []
        
        for goals, strength in zip(goals_list, opponent_strength_list):
            # Güçlü rakiplere karşı atılan goller daha değerli, zayıf rakiplere karşı daha az değerli
            if strength > 0.6:  # Güçlü rakip
                adjusted = goals * (1 + (strength - 0.6))  # Artış faktörü
            elif strength < 0.4:  # Zayıf rakip
                adjusted = goals * (1 - (0.4 - strength))  # Azalış faktörü
            else:
                adjusted = goals  # Orta seviye rakip, düzeltme yok
                
            adjusted_goals.append(adjusted)
            
        return adjusted_goals
        
    def analyze_scoring_trend(self, team_form_data, adjusted_for_opponent=True):
        """
        Takımın gol atma trendini analiz et
        
        Args:
            team_form_data: Takımın form verileri (match_prediction.py'dan)
            adjusted_for_opponent: Rakip gücüne göre düzeltme yapılsın mı
            
        Returns:
            dict: Gol atma trendi analiz sonuçları
        """
        # Önbellekte varsa, önbellekten döndür
        team_id = team_form_data.get('team_id')
        if team_id and team_id in self.trend_results_cache:
            return self.trend_results_cache[team_id].get('scoring_trend')
            
        if 'recent_match_data' not in team_form_data:
            logger.warning("Recent match data not found in team form data")
            return {
                "trend": "stable",
                "acceleration": 0.0,
                "burst": {"detected": False},
                "confidence": 0.0
            }
            
        # En son maçları al (en yeni maç ilk sırada)
        matches = team_form_data['recent_match_data']
        
        # Gol sayılarını ve rakip güçlerini liste haline getir
        goals_scored = [match.get('goals_scored', 0) for match in matches]
        opponent_strength = [match.get('opponent_strength', 0.5) for match in matches]
        
        # Rakip gücüne göre düzeltme yap (opsiyonel)
        if adjusted_for_opponent:
            adjusted_goals = self._adjust_for_opponent_strength(goals_scored, opponent_strength)
        else:
            adjusted_goals = goals_scored
            
        # Trend ivmesini hesapla
        trend_data = self._calculate_trend_acceleration(adjusted_goals)
        
        # Ani gol patlaması veya düşüşü tespit et
        burst_data = self._detect_goal_burst(adjusted_goals)
        
        # Sonuçları birleştir
        results = {
            "trend": trend_data["trend"],
            "acceleration": trend_data["acceleration"],
            "burst": burst_data,
            "confidence": trend_data["confidence"],
            "recent_avg": trend_data["recent_avg"],
            "previous_avg": trend_data["previous_avg"]
        }
        
        # Önbelleğe kaydet
        if team_id:
            if team_id not in self.trend_results_cache:
                self.trend_results_cache[team_id] = {}
            self.trend_results_cache[team_id]['scoring_trend'] = results
            
        return results
        
    def analyze_conceding_trend(self, team_form_data, adjusted_for_opponent=True):
        """
        Takımın gol yeme trendini analiz et
        
        Args:
            team_form_data: Takımın form verileri (match_prediction.py'dan)
            adjusted_for_opponent: Rakip gücüne göre düzeltme yapılsın mı
            
        Returns:
            dict: Gol yeme trendi analiz sonuçları
        """
        # Önbellekte varsa, önbellekten döndür
        team_id = team_form_data.get('team_id')
        if team_id and team_id in self.trend_results_cache:
            return self.trend_results_cache[team_id].get('conceding_trend')
            
        if 'recent_match_data' not in team_form_data:
            logger.warning("Recent match data not found in team form data")
            return {
                "trend": "stable",
                "acceleration": 0.0,
                "burst": {"detected": False},
                "confidence": 0.0
            }
            
        # En son maçları al (en yeni maç ilk sırada)
        matches = team_form_data['recent_match_data']
        
        # Yenilen gol sayılarını ve rakip güçlerini liste haline getir
        goals_conceded = [match.get('goals_conceded', 0) for match in matches]
        opponent_strength = [match.get('opponent_strength', 0.5) for match in matches]
        
        # Rakip gücüne göre düzeltme yap (opsiyonel)
        if adjusted_for_opponent:
            adjusted_goals = self._adjust_for_opponent_strength(goals_conceded, opponent_strength)
        else:
            adjusted_goals = goals_conceded
            
        # Trend ivmesini hesapla
        trend_data = self._calculate_trend_acceleration(adjusted_goals)
        
        # Ani gol yeme patlaması veya düşüşü tespit et
        burst_data = self._detect_goal_burst(adjusted_goals)
        
        # Sonuçları birleştir
        results = {
            "trend": trend_data["trend"],
            "acceleration": trend_data["acceleration"],
            "burst": burst_data,
            "confidence": trend_data["confidence"],
            "recent_avg": trend_data["recent_avg"],
            "previous_avg": trend_data["previous_avg"]
        }
        
        # Önbelleğe kaydet
        if team_id:
            if team_id not in self.trend_results_cache:
                self.trend_results_cache[team_id] = {}
            self.trend_results_cache[team_id]['conceding_trend'] = results
            
        return results
        
    def calculate_goal_trend_factors(self, home_form, away_form):
        """
        Gol trend faktörlerini hesapla - tahminlerde kullanılacak
        
        Args:
            home_form: Ev sahibi takımın form verileri
            away_form: Deplasman takımının form verileri
            
        Returns:
            dict: Gol trend faktörleri
        """
        # Her iki takımın gol atma ve yeme trendlerini analiz et
        home_scoring_trend = self.analyze_scoring_trend(home_form)
        home_conceding_trend = self.analyze_conceding_trend(home_form)
        away_scoring_trend = self.analyze_scoring_trend(away_form)
        away_conceding_trend = self.analyze_conceding_trend(away_form)
        
        # Faktörleri hesapla
        factors = {
            "home_scoring_factor": self._trend_to_factor(home_scoring_trend),
            "home_conceding_factor": self._trend_to_factor(home_conceding_trend),
            "away_scoring_factor": self._trend_to_factor(away_scoring_trend),
            "away_conceding_factor": self._trend_to_factor(away_conceding_trend),
            "trends": {
                "home_scoring": home_scoring_trend,
                "home_conceding": home_conceding_trend,
                "away_scoring": away_scoring_trend,
                "away_conceding": away_conceding_trend
            }
        }
        
        # Gol trendine dayalı maç olasılık ayarlaması hesapla
        factors["match_outcome_adjustment"] = self._calculate_match_outcome_adjustment(factors)
        
        return factors
        
    def _trend_to_factor(self, trend_data):
        """
        Trend verilerini sayısal faktöre dönüştür
        
        Args:
            trend_data: Trend analiz verileri
            
        Returns:
            float: Sayısal faktör (0.5-2.0 aralığında)
        """
        if not trend_data:
            return 1.0  # Varsayılan nötr değer
            
        # Temel faktör: İvmeye bağlı
        acceleration = trend_data.get("acceleration", 0.0)
        confidence = trend_data.get("confidence", 0.5)
        
        # İvmeyi faktöre dönüştür: 0 ivme = 1.0 faktör (nötr)
        # Pozitif ivme: 1.0'dan büyük faktör, Negatif ivme: 1.0'dan küçük faktör
        base_factor = 1.0 + (acceleration * 0.15)  # Her 1 birim ivme için ±%15 değişim
        
        # Faktörü makul aralığa sınırla
        factor = max(0.5, min(2.0, base_factor))
        
        # Güven düzeyine göre faktör ağırlığı ayarla
        weighted_factor = 1.0 + ((factor - 1.0) * confidence)
        
        # Ani gol patlaması veya düşüşü varsa ek ayarlama
        burst = trend_data.get("burst", {})
        if burst.get("detected", False):
            burst_type = burst.get("type", "normal")
            burst_ratio = burst.get("ratio", 1.0)
            
            if burst_type == "burst":
                # Ani gol patlaması: Faktörü yükselt
                burst_adjustment = min(0.25, (burst_ratio - 1.0) * 0.1)  # En fazla +0.25
                weighted_factor += burst_adjustment
            elif burst_type == "slump":
                # Ani gol düşüşü: Faktörü düşür
                burst_adjustment = min(0.25, (1.0 - burst_ratio) * 0.1)  # En fazla -0.25
                weighted_factor -= burst_adjustment
                
        return weighted_factor
        
    def _calculate_match_outcome_adjustment(self, factors):
        """
        Gol trendlerine dayalı maç sonucu olasılık düzeltmelerini hesapla
        
        Args:
            factors: Hesaplanan gol trend faktörleri
            
        Returns:
            dict: Maç sonucu olasılık düzeltmeleri
        """
        # Faktörlerden değerleri çıkart
        home_scoring = factors["home_scoring_factor"]
        home_conceding = factors["home_conceding_factor"]
        away_scoring = factors["away_scoring_factor"]
        away_conceding = factors["away_conceding_factor"]
        
        # Takım avantajlarını hesapla
        home_advantage = (home_scoring * away_conceding) / 2
        away_advantage = (away_scoring * home_conceding) / 2
        
        # İki takım arasındaki güç farkı
        power_diff = home_advantage - away_advantage
        
        # MS1, X, MS2 olasılık düzeltmeleri hesapla
        # Power diff pozitifse ev sahibi lehine, negatifse deplasman lehine
        if power_diff > 0:
            # Ev sahibi lehine durum
            home_win_adj = min(0.15, power_diff * 0.1)  # En fazla +%15
            draw_adj = -home_win_adj * 0.5  # Beraberlik olasılığı biraz düşer
            away_win_adj = -home_win_adj * 0.5  # Deplasman galibiyet olasılığı biraz düşer
        else:
            # Deplasman lehine durum
            away_win_adj = min(0.15, -power_diff * 0.1)  # En fazla +%15
            draw_adj = -away_win_adj * 0.5  # Beraberlik olasılığı biraz düşer
            home_win_adj = -away_win_adj * 0.5  # Ev sahibi galibiyet olasılığı biraz düşer
            
        # Toplam gol beklentisi ayarlaması
        total_goals_factor = (home_scoring * away_scoring + home_conceding * away_conceding) / 4
        over_under_adj = (total_goals_factor - 1.0) * 0.2  # Trend faktörü 1.0'dan büyükse ÜST lehine
        
        return {
            "home_win": home_win_adj,
            "draw": draw_adj,
            "away_win": away_win_adj,
            "over_under": over_under_adj,
            "description": self._generate_trend_description(factors)
        }
        
    def _generate_trend_description(self, factors):
        """
        Gol trendleri için anlaşılır açıklama oluştur
        
        Args:
            factors: Gol trend faktörleri
            
        Returns:
            str: Trend açıklaması
        """
        # Trendleri al
        trends = factors["trends"]
        home_scoring = trends["home_scoring"]
        home_conceding = trends["home_conceding"]
        away_scoring = trends["away_scoring"]
        away_conceding = trends["away_conceding"]
        
        # Ev sahibi açıklaması
        home_desc = ""
        if home_scoring["trend"] == "rising":
            home_desc += "Ev sahibi takım son maçlarda daha fazla gol atmaya başladı"
            if home_scoring["burst"]["detected"] and home_scoring["burst"]["type"] == "burst":
                home_desc += " (ani gol patlaması var)"
        elif home_scoring["trend"] == "falling":
            home_desc += "Ev sahibi takımın gol atma trendi düşüşte"
            if home_scoring["burst"]["detected"] and home_scoring["burst"]["type"] == "slump":
                home_desc += " (ani gol düşüşü var)"
                
        if home_conceding["trend"] == "rising":
            if home_desc:
                home_desc += " ve "
            home_desc += "daha fazla gol yemeye başladı"
        elif home_conceding["trend"] == "falling":
            if home_desc:
                home_desc += " ama "
            home_desc += "gol yeme trendi düşüşte (savunması güçleniyor)"
            
        # Deplasman açıklaması
        away_desc = ""
        if away_scoring["trend"] == "rising":
            away_desc += "Deplasman takımı son maçlarda daha fazla gol atmaya başladı"
            if away_scoring["burst"]["detected"] and away_scoring["burst"]["type"] == "burst":
                away_desc += " (ani gol patlaması var)"
        elif away_scoring["trend"] == "falling":
            away_desc += "Deplasman takımının gol atma trendi düşüşte"
            if away_scoring["burst"]["detected"] and away_scoring["burst"]["type"] == "slump":
                away_desc += " (ani gol düşüşü var)"
                
        if away_conceding["trend"] == "rising":
            if away_desc:
                away_desc += " ve "
            away_desc += "daha fazla gol yemeye başladı"
        elif away_conceding["trend"] == "falling":
            if away_desc:
                away_desc += " ama "
            away_desc += "gol yeme trendi düşüşte (savunması güçleniyor)"
            
        # Genel açıklama
        description = ""
        if home_desc:
            description += home_desc + ". "
        if away_desc:
            description += away_desc + "."
            
        # Açıklama yoksa nötr bir açıklama ekle
        if not description:
            description = "Her iki takımın da gol trendlerinde önemli bir değişiklik gözlenmiyor."
            
        return description
            
    def adjust_expected_goals(self, home_expected_goals, away_expected_goals, trend_factors):
        """
        Gol trendlerine göre beklenen gol tahminlerini düzelt
        
        Args:
            home_expected_goals: Ev sahibi takımın beklenen gol sayısı
            away_expected_goals: Deplasman takımının beklenen gol sayısı
            trend_factors: Gol trend faktörleri
            
        Returns:
            tuple: Düzeltilmiş (ev sahibi gol, deplasman gol) değerleri
        """
        # Trend faktörlerini al
        home_scoring_factor = trend_factors["home_scoring_factor"]
        away_conceding_factor = trend_factors["away_conceding_factor"]
        away_scoring_factor = trend_factors["away_scoring_factor"]
        home_conceding_factor = trend_factors["home_conceding_factor"]
        
        # Faktör ağırlıkları (toplam 1.0)
        BASE_WEIGHT = 0.7  # Orijinal tahmin ağırlığı
        TREND_WEIGHT = 1.0 - BASE_WEIGHT  # Trend faktörü ağırlığı
        
        # Düzeltilmiş gol tahminleri
        adjusted_home_goals = (home_expected_goals * BASE_WEIGHT) + (home_expected_goals * home_scoring_factor * away_conceding_factor * TREND_WEIGHT)
        adjusted_away_goals = (away_expected_goals * BASE_WEIGHT) + (away_expected_goals * away_scoring_factor * home_conceding_factor * TREND_WEIGHT)
        
        # Aşırı sapmaları sınırla
        if adjusted_home_goals > home_expected_goals * 1.5:
            adjusted_home_goals = home_expected_goals * 1.5
        elif adjusted_home_goals < home_expected_goals * 0.5:
            adjusted_home_goals = home_expected_goals * 0.5
            
        if adjusted_away_goals > away_expected_goals * 1.5:
            adjusted_away_goals = away_expected_goals * 1.5
        elif adjusted_away_goals < away_expected_goals * 0.5:
            adjusted_away_goals = away_expected_goals * 0.5
            
        return adjusted_home_goals, adjusted_away_goals

# Singleton instance için yardımcı fonksiyon
_instance = None

def get_instance():
    """
    GoalTrendAnalyzer'ın singleton instance'ını döndür
    """
    global _instance
    if _instance is None:
        _instance = GoalTrendAnalyzer()
    return _instance