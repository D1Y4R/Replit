"""
İlk Yarı / Maç Sonu tahmin algoritmaları
Takımların ilk ve ikinci yarı performanslarına göre İY/MS olasılıklarını hesaplar
Monte Carlo simülasyonu, yapay sinir ağı, Conditional Random Fields ve Dirichlet Süreci
Karışım Modelleri kullanılarak gelişmiş tahminler yapılır
"""

import logging
import math
import random
import numpy as np

# Geliştirilmiş Poisson ve Skellam dağılımları için SciPy kütüphanesi
try:
    from scipy.stats import poisson, skellam
    SCIPY_AVAILABLE = True
    logging.getLogger(__name__).info("SciPy kütüphanesi kullanılabilir, gelişmiş istatistiksel modeller etkin.")
except ImportError:
    SCIPY_AVAILABLE = False
    logging.getLogger(__name__).warning("SciPy kütüphanesi bulunamadı, basit istatistiksel modeller kullanılacak.")

# Conditional Random Fields (CRF) modeli
try:
    from crf_predictor import CRFPredictor
    CRF_AVAILABLE = True
    logging.getLogger(__name__).info("CRF modeli kullanılabilir, gelişmiş yapısal tahmin modeli etkin.")
except ImportError:
    CRF_AVAILABLE = False
    logging.getLogger(__name__).warning("CRF modeli bulunamadı, geleneksel tahmin modelleri kullanılacak.")
    
# Dirichlet Süreci Karışım Modeli
try:
    from dirichlet_predictor import DirichletPredictor
    DIRICHLET_AVAILABLE = True
    logging.getLogger(__name__).info("Dirichlet Süreci Karışım Modeli kullanılabilir, gizli küme keşfi etkin.")
except ImportError:
    DIRICHLET_AVAILABLE = False
    logging.getLogger(__name__).warning("Dirichlet modeli bulunamadı, geleneksel tahmin modelleri kullanılacak.")
    
import json
import os
from collections import defaultdict
from datetime import datetime
from scipy.stats import poisson, skellam

# Team-specific models entegrasyonu
# Configure logger
logger = logging.getLogger(__name__)

try:
    from team_specific_models import TeamSpecificPredictor
except ImportError:
    logger.warning("TeamSpecificPredictor import edilemedi, takım-spesifik ayarlamalar devre dışı.")
    
# Log konfigurasyon satırı duplicate - kaldırıldı

# Tüm olası İY/MS kombinasyonları
HT_FT_COMBINATIONS = ['1/1', '1/X', '1/2', 'X/1', 'X/X', 'X/2', '2/1', '2/X', '2/2']

# Form durumu hesaplamaya yardımcı fonksiyonlar
def calculate_form_points(matches, count=5):
    """
    Takımın son maçlardaki form durumunu hesapla
    
    Args:
        matches: Takımın maç verileri
        count: Kaç maç geriye gidileceği (5, 10, 15 vs.)
        
    Returns:
        Form durumu (galibiyet, beraberlik, mağlubiyet sayıları ve puan)
    """
    if not matches or not isinstance(matches, list) or len(matches) == 0:
        return {
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "points": 0,
            "form_score": 0,
            "recent_matches": 0
        }
    
    # En son maçları al (max count kadar)
    recent_matches = matches[:min(count, len(matches))]
    
    wins = 0
    draws = 0
    losses = 0
    
    for match in recent_matches:
        result = match.get("result", "")
        if result == "W":
            wins += 1
        elif result == "D":
            draws += 1
        elif result == "L":
            losses += 1
    
    # Toplam puan (W=3, D=1, L=0)
    points = wins * 3 + draws * 1
    max_possible_points = len(recent_matches) * 3
    
    # Form skoru (0-1 arası normalize edilmiş)
    form_score = points / max_possible_points if max_possible_points > 0 else 0
    
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "points": points,
        "form_score": form_score,
        "recent_matches": len(recent_matches)
    }

def calculate_motivation_factor(matches, weighted=True):
    """
    Takımın motivasyon/moral durumunu hesapla
    Yakın zamandaki maçların uzak geçmişteki maçlardan daha önemli olduğunu varsayar
    
    Args:
        matches: Takımın maç verileri
        weighted: Yakın zamandaki maçlara daha fazla ağırlık ver
        
    Returns:
        Motivasyon faktörü (0-1 arasında)
    """
    if not matches or not isinstance(matches, list) or len(matches) == 0:
        return 0.5  # Nötr değer
    
    # Son 5 maçı al
    recent_matches = matches[:min(5, len(matches))]
    
    # Maç sonuçlarına göre puanlar (W=3, D=1, L=0)
    points = []
    for match in recent_matches:
        result = match.get("result", "")
        if result == "W":
            points.append(3)
        elif result == "D":
            points.append(1)
        elif result == "L":
            points.append(0)
    
    # Eğer ağırlıklı hesaplama isteniyorsa, son maçlara daha fazla ağırlık ver
    if weighted and points:
        weights = [1.0, 0.8, 0.6, 0.4, 0.2][:len(points)]
        weighted_points = sum(p * w for p, w in zip(points, weights))
        max_possible = sum(3 * w for w in weights)
        motivation = weighted_points / max_possible if max_possible > 0 else 0.5
    else:
        # Ağırlıksız hesaplama
        max_possible = len(points) * 3
        motivation = sum(points) / max_possible if max_possible > 0 else 0.5
    
    # 0.1-0.9 aralığına normalize et (çok uç değerlerden kaçın)
    motivation = 0.1 + (motivation * 0.8)
    
    return motivation

def analyze_ht_ft_combinations(ht_ft_combinations, is_home=True):
    """
    YENİ: Takımın İY/MS kombinasyonlarını analiz eder
    
    Args:
        ht_ft_combinations: İY/MS kombinasyon verileri (1/1, 1/X, 1/2, X/1, vb.)
        is_home: Takımın ev sahibi olup olmadığı
        
    Returns:
        dict: Analiz sonuçları
    """
    if not ht_ft_combinations or not isinstance(ht_ft_combinations, dict):
        return {
            "most_common_combination": "X/X",
            "most_common_combinations": [],
            "rare_combinations": [],
            "switches": {
                "first_to_second": 0.0,  # 1/2 veya 2/1 gibi tersine dönüşler
                "draw_to_result": 0.0,   # X/1 veya X/2 gibi beraberlikten sonuca
                "result_to_draw": 0.0    # 1/X veya 2/X gibi sonuçtan beraberliğe
            },
            "home_patterns": {},
            "away_patterns": {}
        }
    
    try:
        # is_home parametresine göre home, away veya total kategorisini seç
        category = "home" if is_home else "away"
        
        # Seçilen kategoriye göre İY/MS kombinasyonları
        results = ht_ft_combinations.get(category, {})
        home_results = ht_ft_combinations.get("home", {})
        away_results = ht_ft_combinations.get("away", {})
        
        # Ev sahibi ve deplasman maç sayıları
        home_matches = sum(home_results.values()) if home_results else 0
        away_matches = sum(away_results.values()) if away_results else 0
        
        # Seçilen kategoriye göre maç sayısı
        selected_matches = sum(results.values())
        
        if selected_matches == 0:
            return {
                "most_common_combination": "X/X",
                "most_common_combinations": [],
                "rare_combinations": [],
                "switches": {
                    "first_to_second": 0.0,
                    "draw_to_result": 0.0,
                    "result_to_draw": 0.0
                },
                "home_patterns": {},
                "away_patterns": {}
            }
        
        # En sık görülen kombinasyonlar
        sorted_combinations = sorted(results.items(), key=lambda x: x[1], reverse=True)
        most_common_combination = sorted_combinations[0][0] if sorted_combinations else "X/X"
        
        # En sık görülen üç kombinasyon (oranları ile birlikte)
        most_common_combinations = []
        for combo, count in sorted_combinations[:3]:
            most_common_combinations.append({
                "combination": combo,
                "count": count,
                "rate": count / selected_matches if selected_matches > 0 else 0.0
            })
        
        # En nadir görülen kombinasyonlar (en az 1 kez görülmüş olanlar)
        rare_combinations = []
        for combo, count in sorted(results.items(), key=lambda x: x[1])[:3]:
            if count > 0:
                rare_combinations.append({
                    "combination": combo,
                    "count": count,
                    "rate": count / selected_matches if selected_matches > 0 else 0.0
                })
        
        # İlk yarı/ikinci yarı geçiş analizleri
        switches = {
            "first_to_second": 0.0,  # 1/2 veya 2/1 gibi tersine dönüşler
            "draw_to_result": 0.0,   # X/1 veya X/2 gibi beraberlikten sonuca
            "result_to_draw": 0.0    # 1/X veya 2/X gibi sonuçtan beraberliğe
        }
        
        # Tersine dönüşler (1/2, 2/1)
        first_to_second_count = results.get("1/2", 0) + results.get("2/1", 0)
        # Beraberlikten sonuca (X/1, X/2)
        draw_to_result_count = results.get("X/1", 0) + results.get("X/2", 0)
        # Sonuçtan beraberliğe (1/X, 2/X)
        result_to_draw_count = results.get("1/X", 0) + results.get("2/X", 0)
        
        switches["first_to_second"] = first_to_second_count / selected_matches if selected_matches > 0 else 0.0
        switches["draw_to_result"] = draw_to_result_count / selected_matches if selected_matches > 0 else 0.0
        switches["result_to_draw"] = result_to_draw_count / selected_matches if selected_matches > 0 else 0.0
        
        # Ev sahibi ve deplasman olarak kombinasyon örüntüleri
        home_patterns = {}
        away_patterns = {}
        
        # Ev sahibi örüntüleri
        if home_matches > 0:
            home_patterns = {
                "consistent": (home_results.get("1/1", 0) + home_results.get("X/X", 0) + home_results.get("2/2", 0)) / home_matches,
                "improving": (home_results.get("X/1", 0) + home_results.get("2/1", 0) + home_results.get("2/X", 0)) / home_matches,
                "declining": (home_results.get("1/X", 0) + home_results.get("1/2", 0) + home_results.get("X/2", 0)) / home_matches
            }
        
        # Deplasman örüntüleri
        if away_matches > 0:
            away_patterns = {
                "consistent": (away_results.get("1/1", 0) + away_results.get("X/X", 0) + away_results.get("2/2", 0)) / away_matches,
                "improving": (away_results.get("X/1", 0) + away_results.get("2/1", 0) + away_results.get("2/X", 0)) / away_matches,
                "declining": (away_results.get("1/X", 0) + away_results.get("1/2", 0) + away_results.get("X/2", 0)) / away_matches
            }
        
        # Kombinasyonların tipine göre gruplama
        combination_types = {
            "stable": {},     # 1/1, X/X, 2/2 - aynı kalan sonuçlar
            "reversed": {},   # 1/2, 2/1 - tersine dönen sonuçlar
            "partial": {}     # 1/X, X/1, 2/X, X/2 - kısmen değişen sonuçlar
        }
        
        # Her kombinasyon tipi için oran hesapla
        for combo, count in results.items():
            if combo in ["1/1", "X/X", "2/2"]:
                combination_types["stable"][combo] = count / selected_matches if selected_matches > 0 else 0.0
            elif combo in ["1/2", "2/1"]:
                combination_types["reversed"][combo] = count / selected_matches if selected_matches > 0 else 0.0
            elif combo in ["1/X", "X/1", "2/X", "X/2"]:
                combination_types["partial"][combo] = count / selected_matches if selected_matches > 0 else 0.0
        
        return {
            "most_common_combination": most_common_combination,
            "most_common_combinations": most_common_combinations,
            "rare_combinations": rare_combinations,
            "switches": switches,
            "home_patterns": home_patterns,
            "away_patterns": away_patterns,
            "combination_types": combination_types
        }
    except Exception as e:
        logger.error(f"İY/MS kombinasyonları analizinde hata: {str(e)}")
        return {
            "most_common_combination": "X/X",
            "most_common_combinations": [],
            "rare_combinations": [],
            "switches": {
                "first_to_second": 0.0,
                "draw_to_result": 0.0,
                "result_to_draw": 0.0
            },
            "home_patterns": {},
            "away_patterns": {},
            "error": str(e)
        }

def get_form_trend(matches, first_half=True):
    """
    Takımın ilk yarı veya ikinci yarı form trendini hesapla
    
    Args:
        matches: Takımın maç verileri
        first_half: İlk yarı için mi, ikinci yarı için mi
        
    Returns:
        Form trendi (yükselen, düşen, kararlı)
    """
    if not matches or not isinstance(matches, list) or len(matches) < 3:
        return "steady"  # Yeterli veri yoksa kararlı sayalım
    
    # Son 3 maçı al
    recent_matches = matches[:3]
    
    goals_key = "ht_goals_scored" if first_half else "goals_scored"
    conceded_key = "ht_goals_conceded" if first_half else "goals_conceded"
    
    # Son 3 maçtaki gol bilgilerini topla
    goals_per_match = []
    for match in recent_matches:
        scored = match.get(goals_key, 0)
        conceded = match.get(conceded_key, 0)
        
        # Net skor farkını hesapla
        diff = scored - conceded
        goals_per_match.append(diff)
    
    # Trend hesaplaması
    if len(goals_per_match) >= 3:
        # Son maçtan ilk maça doğru gidiyoruz (0=en son maç)
        if goals_per_match[0] > goals_per_match[1] > goals_per_match[2]:
            return "rising"  # Yükselen trend
        elif goals_per_match[0] < goals_per_match[1] < goals_per_match[2]:
            return "falling"  # Düşen trend
    
    return "steady"  # Kararlı/belirsiz trend

def analyze_ht_results(ht_results, is_home=True):
    """
    Takımın ilk yarı ve maç sonu sonuçlarını analiz eder
    
    Args:
        ht_results: İlk yarı sonuç verileri (1=önde, X=berabere, 2=geride)
        is_home: Takımın ev sahibi olup olmadığı
        
    Returns:
        dict: Analiz sonuçları
    """
    if not ht_results:
        return {
            "ht_win_rate": 0.0,
            "ht_draw_rate": 0.0,
            "ht_loss_rate": 0.0,
            "trend": "steady",
            "most_common_result": "X",
            "home_tendency": "neutral",
            "away_tendency": "neutral",
            "overall_tendency": "neutral"
        }
    
    try:
        # Ev sahibi, deplasman ve genel sonuçları analiz et
        home_results = ht_results.get("home", {"1": 0, "X": 0, "2": 0})
        away_results = ht_results.get("away", {"1": 0, "X": 0, "2": 0})
        total_results = ht_results.get("total", {"1": 0, "X": 0, "2": 0})
        
        # Kullanılacak kategoriyi belirle
        category = "home" if is_home else "away"
        if category not in ht_results:
            category = "total"  # Eğer spesifik kategori yoksa toplam kullan
            
        results = ht_results[category]
        
        # Toplam maç sayıları
        home_matches = sum(home_results.values())
        away_matches = sum(away_results.values())
        total_matches = sum(total_results.values())
        
        # Seçilen kategoriye göre maç sayısı
        selected_matches = sum(results.values())
        
        if selected_matches == 0:
            return {
                "ht_win_rate": 0.0,
                "ht_draw_rate": 0.0,
                "ht_loss_rate": 0.0,
                "trend": "steady",
                "most_common_result": "X",
                "home_tendency": "neutral",
                "away_tendency": "neutral",
                "overall_tendency": "neutral"
            }
        
        # İlk yarı sonuçları oranları (seçilen kategoriye göre)
        ht_win_rate = results.get("1", 0) / selected_matches if selected_matches > 0 else 0.0
        ht_draw_rate = results.get("X", 0) / selected_matches if selected_matches > 0 else 0.0
        ht_loss_rate = results.get("2", 0) / selected_matches if selected_matches > 0 else 0.0
        
        # Ev sahibi olarak ilk yarı sonuçları oranları (eğer veri varsa)
        home_win_rate = home_results.get("1", 0) / home_matches if home_matches > 0 else 0.0
        home_draw_rate = home_results.get("X", 0) / home_matches if home_matches > 0 else 0.0
        home_loss_rate = home_results.get("2", 0) / home_matches if home_matches > 0 else 0.0
        
        # Deplasman olarak ilk yarı sonuçları oranları (eğer veri varsa)
        away_win_rate = away_results.get("1", 0) / away_matches if away_matches > 0 else 0.0
        away_draw_rate = away_results.get("X", 0) / away_matches if away_matches > 0 else 0.0
        away_loss_rate = away_results.get("2", 0) / away_matches if away_matches > 0 else 0.0
        
        # En çok tekrarlanan sonucu bulma
        most_common_result = max(results.items(), key=lambda x: x[1])[0] if results else "X"
        
        # Ev sahibi eğilimi
        home_tendency = "neutral"
        if home_matches > 0:
            if home_win_rate > 0.5:
                home_tendency = "usually_leading_at_home"
            elif home_draw_rate > 0.5:
                home_tendency = "usually_drawing_at_home"
            elif home_loss_rate > 0.5:
                home_tendency = "usually_trailing_at_home"
            elif home_win_rate > home_draw_rate and home_win_rate > home_loss_rate:
                home_tendency = "mostly_leading_at_home"
            elif home_draw_rate > home_win_rate and home_draw_rate > home_loss_rate:
                home_tendency = "mostly_drawing_at_home"
            elif home_loss_rate > home_win_rate and home_loss_rate > home_draw_rate:
                home_tendency = "mostly_trailing_at_home"
        
        # Deplasman eğilimi
        away_tendency = "neutral"
        if away_matches > 0:
            if away_win_rate > 0.5:
                away_tendency = "usually_leading_away"
            elif away_draw_rate > 0.5:
                away_tendency = "usually_drawing_away"
            elif away_loss_rate > 0.5:
                away_tendency = "usually_trailing_away"
            elif away_win_rate > away_draw_rate and away_win_rate > away_loss_rate:
                away_tendency = "mostly_leading_away"
            elif away_draw_rate > away_win_rate and away_draw_rate > away_loss_rate:
                away_tendency = "mostly_drawing_away"
            elif away_loss_rate > away_win_rate and away_loss_rate > away_draw_rate:
                away_tendency = "mostly_trailing_away"
        
        # Genel ilk yarı trendi
        trend = "steady"
        if ht_win_rate > 0.5:  # %50'den fazla ilk yarı önde
            trend = "strong_first_half"
        elif ht_loss_rate > 0.5:  # %50'den fazla ilk yarı geride
            trend = "weak_first_half"
        elif ht_draw_rate > 0.5:  # %50'den fazla ilk yarı berabere
            trend = "balanced_first_half"
        elif ht_win_rate > ht_draw_rate and ht_win_rate > ht_loss_rate:
            trend = "mostly_leading"
        elif ht_draw_rate > ht_win_rate and ht_draw_rate > ht_loss_rate:
            trend = "mostly_drawing"
        elif ht_loss_rate > ht_win_rate and ht_loss_rate > ht_draw_rate:
            trend = "mostly_trailing"
        
        # Sonuç olarak daha detaylı bir analiz döndür
        return {
            "ht_win_rate": round(ht_win_rate * 100, 1),
            "ht_draw_rate": round(ht_draw_rate * 100, 1), 
            "ht_loss_rate": round(ht_loss_rate * 100, 1),
            "trend": trend,
            "most_common_result": most_common_result,
            "home_tendency": home_tendency,
            "away_tendency": away_tendency,
            "home_stats": {
                "win_rate": round(home_win_rate * 100, 1),
                "draw_rate": round(home_draw_rate * 100, 1),
                "loss_rate": round(home_loss_rate * 100, 1),
                "matches": home_matches
            },
            "away_stats": {
                "win_rate": round(away_win_rate * 100, 1),
                "draw_rate": round(away_draw_rate * 100, 1),
                "loss_rate": round(away_loss_rate * 100, 1),
                "matches": away_matches
            },
            "total_matches": total_matches,
            "selected_category": category
        }
    except Exception as e:
        logging.error(f"İlk yarı sonuçları analiz edilirken hata: {str(e)}")
        return {
            "ht_win_rate": 0.0,
            "ht_draw_rate": 0.0,
            "ht_loss_rate": 0.0,
            "trend": "steady",
            "most_common_result": "X",
            "home_tendency": "neutral",
            "away_tendency": "neutral",
            "overall_tendency": "neutral",
            "error": str(e)
        }

def analyze_ht_ft_combinations(ht_ft_combinations, is_home=True):
    """
    YENİ: Takımın İY/MS kombinasyonlarını analiz eder
    
    Args:
        ht_ft_combinations: İY/MS kombinasyon verileri (1/1, 1/X, 1/2, X/1, vb.)
        is_home: Takımın ev sahibi olup olmadığı
        
    Returns:
        dict: Analiz sonuçları
    """
    if not ht_ft_combinations:
        return {
            "most_common_combination": None,
            "switches": {
                "first_to_second": 0.0,
                "draw_to_result": 0.0
            },
            "consistency": 0.0,
            "patterns": {}
        }
    
    try:
        # Ev sahibi, deplasman ve genel sonuçları analiz et
        home_results = ht_ft_combinations.get("home", {})
        away_results = ht_ft_combinations.get("away", {})
        total_results = ht_ft_combinations.get("total", {})
        
        # Kullanılacak kategoriyi belirle
        category = "home" if is_home else "away"
        if category not in ht_ft_combinations or not ht_ft_combinations[category]:
            category = "total"  # Eğer spesifik kategori yoksa toplam kullan
            
        results = ht_ft_combinations[category]
        
        # Toplam maç sayıları
        home_matches = sum(home_results.values()) if home_results else 0
        away_matches = sum(away_results.values()) if away_results else 0
        total_matches = sum(total_results.values()) if total_results else 0
        
        # Seçilen kategoriye göre maç sayısı
        selected_matches = sum(results.values()) if results else 0
        
        if selected_matches == 0:
            return {
                "most_common_combination": None,
                "switches": {
                    "first_to_second": 0.0,
                    "draw_to_result": 0.0
                },
                "consistency": 0.0,
                "patterns": {}
            }
        
        # En çok tekrarlanan kombinasyonu bulma
        most_common_combination = max(results.items(), key=lambda x: x[1])[0] if results else None
        
        # Tutarlılık: 1/1, X/X, 2/2 kombinasyonlarının toplam oranı
        consistent_combinations = ["1/1", "X/X", "2/2"]
        consistent_count = sum(results.get(combo, 0) for combo in consistent_combinations)
        consistency = consistent_count / selected_matches if selected_matches > 0 else 0.0
        
        # Tersine dönüş kombinasyonları: 1/2, 2/1
        reverse_combinations = ["1/2", "2/1"]
        reverse_count = sum(results.get(combo, 0) for combo in reverse_combinations)
        first_to_second_switch = reverse_count / selected_matches if selected_matches > 0 else 0.0
        
        # Beraberlikten sonuca dönüş kombinasyonları: X/1, X/2
        draw_to_result_combinations = ["X/1", "X/2"]
        draw_to_result_count = sum(results.get(combo, 0) for combo in draw_to_result_combinations)
        draw_to_result_switch = draw_to_result_count / selected_matches if selected_matches > 0 else 0.0
        
        # Tüm kombinasyonların oranları
        pattern_rates = {}
        for combo, count in results.items():
            pattern_rates[combo] = count / selected_matches if selected_matches > 0 else 0.0
        
        return {
            "most_common_combination": most_common_combination,
            "switches": {
                "first_to_second": first_to_second_switch,
                "draw_to_result": draw_to_result_switch
            },
            "consistency": consistency,
            "patterns": pattern_rates,
            "total_matches": selected_matches
        }
        
    except Exception as e:
        logging.error(f"İY/MS kombinasyonları analiz edilirken hata: {str(e)}")
        return {
            "most_common_combination": None,
            "switches": {
                "first_to_second": 0.0,
                "draw_to_result": 0.0
            },
            "consistency": 0.0,
            "patterns": {},
            "error": str(e)
        }

def sigmoid(x):
    """Sigmoid aktivasyon fonksiyonu"""
    return 1 / (1 + math.exp(-x))

def tanh(x):
    """Tanh aktivasyon fonksiyonu"""
    return math.tanh(x)

def relu(x):
    """ReLU aktivasyon fonksiyonu"""
    return max(0, x)

def softmax(arr):
    """Softmax fonksiyonu - olasılık dağılımı için"""
    max_val = max(arr)
    exp_values = [math.exp(val - max_val) for val in arr]
    sum_exp_values = sum(exp_values)
    return [val / sum_exp_values for val in exp_values]

def predict_half_time_full_time(home_stats, away_stats, global_outcome=None, home_form=None, away_form=None, 
                        home_team_id=None, away_team_id=None, team_adjustments=None, use_crf=True):
    """
    İlk yarı/maç sonu olasılıklarını hesapla
    
    Args:
        home_stats: Ev sahibi takım yarı istatistikleri
        away_stats: Deplasman takımı yarı istatistikleri
        global_outcome: Genel maç sonucu tahmini (Ana tahmin butonundan gelen)
        home_form: Ev sahibi takımın form verileri (opsiyonel)
        away_form: Deplasman takımının form verileri (opsiyonel)
        home_team_id: Ev sahibi takım ID'si (opsiyonel, takım-spesifik ayarlar için)
        away_team_id: Deplasman takım ID'si (opsiyonel, takım-spesifik ayarlar için)
        team_adjustments: Takım-spesifik ayarlamalar (team_specific_models.py modülünden)
        use_crf: CRF modelini kullan (varsayılan: True)
    
    Returns:
        İY/MS tahminleri ve olasılıkları
    """
    # SÜRPRİZ BUTONU İYİLEŞTİRME 4:
    # Ana tahmin fonksiyonundaki iyileştirmeler - algoritmanın giriş noktası burası
    
    # İlk yarı sonuçlarını ve İY/MS kombinasyonlarını analiz et
    home_iy_performansi = None
    away_iy_performansi = None
    home_iyms_analizi = None
    away_iyms_analizi = None
    
    # Eğer home_stats içerisinde ht_results varsa, ilk yarı sonuç analizini kullan
    # Önce home_stats ve away_stats'ın sözlük olduğundan emin olalım
    if not isinstance(home_stats, dict):
        home_stats = {}
    if not isinstance(away_stats, dict):
        away_stats = {}
    
    # Şimdi kontrolü yapalım, Response tipinde olmamalı
    if home_stats and isinstance(home_stats, dict) and 'ht_results' in home_stats:
        logging.info("İlk yarı sonuç analizini kullanarak İY/MS tahmini yapılıyor")
        home_ht_results = home_stats.get('ht_results', {})
        away_ht_results = away_stats.get('ht_results', {})
        
        # Takımların ilk yarı ve maç sonu performanslarını analiz et
        # Bu veriler takımların ev sahibi, deplasman ve toplam istatistiklerini içerir
        home_iy_performansi = analyze_ht_results(home_ht_results, True)  # True = ev sahibi
        away_iy_performansi = analyze_ht_results(away_ht_results, False)  # False = deplasman
        
        logging.info(f"Ev sahibi ilk yarı performansı: {home_iy_performansi}")
        logging.info(f"Deplasman ilk yarı performansı: {away_iy_performansi}")
        
        # YENİ: İY/MS kombinasyonlarını analiz et
        if isinstance(home_stats, dict) and 'ht_ft_combinations' in home_stats:
            logger.info("İY/MS kombinasyon analizini kullanarak geliştirilmiş tahmin yapılıyor")
            home_ht_ft_combinations = home_stats.get('ht_ft_combinations', {})
            away_ht_ft_combinations = away_stats.get('ht_ft_combinations', {})
            
            home_iyms_analizi = analyze_ht_ft_combinations(home_ht_ft_combinations, True)
            away_iyms_analizi = analyze_ht_ft_combinations(away_ht_ft_combinations, False)
            
            logger.info(f"Ev sahibi İY/MS analizi: {home_iyms_analizi['most_common_combination']}, " +
                    f"tersine dönüş oranı: %{round(home_iyms_analizi['switches']['first_to_second']*100, 1)}")
            logger.info(f"Deplasman İY/MS analizi: {away_iyms_analizi['most_common_combination']}, " +
                    f"tersine dönüş oranı: %{round(away_iyms_analizi['switches']['first_to_second']*100, 1)}")
        
        # İlk yarı sonuç verilerini istatistiksel model ağırlıklarını ayarlamak için kullan
        # Bu, ilk yarı önde bitiren takımların kazanma şansını artırmalı
    
    # İstatistik verileri yoksa hata döndür
    if not home_stats or not away_stats:
        logger.error("İY/MS tahmini için yarı istatistikleri eksik veya null: %s, %s", home_stats, away_stats)
        return {
            "error": "Yarı istatistik verileri eksik",
            "predictions": {}
        }
    
    # Takım istatistikleri içinde hata var mı kontrol et
    if "error" in home_stats:
        logger.error("Ev sahibi takım istatistikleri alınamadı veya hatalı: %s", home_stats)
        return {
            "error": "Ev sahibi takım istatistikleri alınamadı",
            "predictions": {}
        }
    
    if "error" in away_stats:
        logger.error("Deplasman takımı istatistikleri alınamadı veya hatalı: %s", away_stats)
        return {
            "error": "Deplasman takımı istatistikleri alınamadı",
            "predictions": {}
        }
    
    # Takımların istatistikleri
    try:
        # Tuple kontrolü - API yanıtı (data, status_code) formatında olabilir
        if isinstance(home_stats, tuple) and len(home_stats) > 0:
            home_stats = home_stats[0]  # İlk eleman veriyi içerir
        if isinstance(away_stats, tuple) and len(away_stats) > 0:
            away_stats = away_stats[0]  # İlk eleman veriyi içerir
            
        # Home_stats sözlük değilse JSON'a dönüştürmeyi dene
        if hasattr(home_stats, 'json'):
            home_stats = home_stats.json()
        if hasattr(away_stats, 'json'):
            away_stats = away_stats.json()
            
        home_first = home_stats["statistics"]["first_half"]
        home_second = home_stats["statistics"]["second_half"]
        away_first = away_stats["statistics"]["first_half"]
        away_second = away_stats["statistics"]["second_half"]
        
        # Takımların toplam maç sayısı
        home_matches = home_stats.get("total_matches_analyzed", 1)  # Default 1 to avoid division by zero
        away_matches = away_stats.get("total_matches_analyzed", 1)  # Default 1 to avoid division by zero
        
        # Takım form verileri için özel İY/MS eğilim analizleri (28.03.2025 İyileştirmesi)
        # Bu veriler 1/2 ve 2/1 gibi "tersine dönüş" kombinasyonları için kullanılacak
        home_form_data = None
        away_form_data = None
        
        if home_form and isinstance(home_form, dict):
            # İlk ve ikinci yarı performans analizlerini ekle (yoksa oluştur)
            if 'second_half_performance' not in home_form:
                home_form['second_half_performance'] = 0.5
            if 'first_half_performance' not in home_form:
                home_form['first_half_performance'] = 0.5
            if 'comeback_ratio' not in home_form:
                home_form['comeback_ratio'] = 0.1  # %10 varsayılan
            if 'scores_in_second_half_ratio' not in home_form:
                home_form['scores_in_second_half_ratio'] = 0.5
                
            home_form_data = home_form
        
        if away_form and isinstance(away_form, dict):
            # İlk ve ikinci yarı performans analizlerini ekle (yoksa oluştur)
            if 'second_half_performance' not in away_form:
                away_form['second_half_performance'] = 0.5
            if 'first_half_performance' not in away_form:
                away_form['first_half_performance'] = 0.5
            if 'comeback_ratio' not in away_form:
                away_form['comeback_ratio'] = 0.1  # %10 varsayılan
            if 'scores_in_second_half_ratio' not in away_form:
                away_form['scores_in_second_half_ratio'] = 0.5
                
            away_form_data = away_form
                
        # Form verilerini geliştirmek için ilk yarı/ikinci yarı goller verisini kullan
        if home_form_data and 'ht_matches' in home_stats:
            try:
                # Geliştirilmiş yarı analizi - takımın maç içindeki performans değişimini daha iyi anlamak için
                # Özellikle 1/2 ve 2/1 gibi "tersine dönüş" kombinasyonlarına odaklanılmıştır
                ht_matches = home_stats['ht_matches']
                recent_matches = ht_matches[:min(10, len(ht_matches))]  # Son 10 maça bak (varsa)
                
                first_half_goals = 0
                second_half_goals = 0
                comeback_count = 0
                second_half_scored_count = 0
                late_goals_count = 0  # 75. dakikadan sonra gol
                
                # Tersine dönüş senaryoları - özel olarak izle
                ht_win_ft_loss = 0   # İlk yarı önde, maç sonu geride (1/2)
                ht_loss_ft_win = 0   # İlk yarı geride, maç sonu önde (2/1)
                ht_draw_ft_win = 0   # İlk yarı berabere, maç sonu galibiyet (X/1)
                ht_draw_ft_loss = 0  # İlk yarı berabere, maç sonu mağlubiyet (X/2)
                
                match_count = len(recent_matches)
                
                if match_count > 0:
                    for match in recent_matches:
                        # İlk yarı/ikinci yarı gollerin yerini kontrol et
                        ht_result = match.get('ht_result', '')
                        ft_result = match.get('ft_result', '')
                        ht_code = match.get('ht_ft_code', '')
                        
                        # Tüm maçlarda ilk yarı ve ikinci yarı golleri topla
                        first_half_goals += match.get('ht_goals_scored', 0)
                        second_half_goals += match.get('goals_scored', 0) - match.get('ht_goals_scored', 0)
                        
                        # İkinci yarıda gol atma durumu
                        if match.get('goals_scored', 0) > match.get('ht_goals_scored', 0):
                            second_half_scored_count += 1
                        
                        # Tersine dönüş senaryolarını kontrol et
                        if ht_code == '1/2':  # İlk yarı önde, maç sonu geride
                            ht_win_ft_loss += 1
                        elif ht_code == '2/1':  # İlk yarı geride, maç sonu önde
                            ht_loss_ft_win += 1
                            comeback_count += 1
                        elif ht_code == 'X/1':  # İlk yarı berabere, maç sonu galibiyet
                            ht_draw_ft_win += 1
                        elif ht_code == 'X/2':  # İlk yarı berabere, maç sonu mağlubiyet
                            ht_draw_ft_loss += 1
                        
                        # Ek tersine dönüş durumları
                        # İlk yarı gerideyken maçı çevirme durumları (ht_code'dan bağımsız)
                        if match.get('ht_goals_scored', 0) < match.get('ht_goals_conceded', 0) and ft_result in ['W', 'D']:
                            comeback_count += 1
                    
                    # Performans değerlerini güncelle
                    if first_half_goals + second_half_goals > 0:
                        home_form_data['first_half_performance'] = first_half_goals / (first_half_goals + second_half_goals)
                        home_form_data['second_half_performance'] = second_half_goals / (first_half_goals + second_half_goals)
                    
                    # Tersine dönüş istatistiklerini kaydet
                    home_form_data['comeback_ratio'] = comeback_count / match_count
                    home_form_data['scores_in_second_half_ratio'] = second_half_scored_count / match_count
                    
                    # Ek analiz verileri
                    home_form_data['ht_win_ft_loss_ratio'] = ht_win_ft_loss / match_count
                    home_form_data['ht_loss_ft_win_ratio'] = ht_loss_ft_win / match_count
                    home_form_data['ht_draw_ft_win_ratio'] = ht_draw_ft_win / match_count
                    home_form_data['ht_draw_ft_loss_ratio'] = ht_draw_ft_loss / match_count
                    
                    # En olası tersine dönüş senaryosu
                    home_form_data['most_likely_switch'] = max(
                        [('1/2', ht_win_ft_loss), ('2/1', ht_loss_ft_win), 
                         ('X/1', ht_draw_ft_win), ('X/2', ht_draw_ft_loss)], 
                        key=lambda x: x[1]
                    )[0]
                    
                    logger.info(f"Ev sahibi tersine dönüş analizleri: Comeback: {home_form_data['comeback_ratio']:.2f}, " +
                                f"En olası dönüş: {home_form_data['most_likely_switch']}")
            except Exception as e:
                logger.error(f"Ev sahibi takım için ilk yarı/ikinci yarı gol analizleri yapılırken hata: {str(e)}")
                # Hataya rağmen varsayılan değerler kullanarak devam et
        
        # Deplasman takımı için aynı analizi yap - Ev sahibi analizlerinden sonra eklendi
        if away_form_data and 'ht_matches' in away_stats:
            try:
                # Ev sahibi ile aynı analizi deplasman için de yap
                ht_matches = away_stats['ht_matches']
                recent_matches = ht_matches[:min(10, len(ht_matches))]  # Son 10 maça bak (varsa)
                
                first_half_goals = 0
                second_half_goals = 0
                comeback_count = 0
                second_half_scored_count = 0
                late_goals_count = 0  # 75. dakikadan sonra gol
                
                # Tersine dönüş senaryoları - özel olarak izle
                ht_win_ft_loss = 0   # İlk yarı önde, maç sonu geride (1/2)
                ht_loss_ft_win = 0   # İlk yarı geride, maç sonu önde (2/1)
                ht_draw_ft_win = 0   # İlk yarı berabere, maç sonu galibiyet (X/1)
                ht_draw_ft_loss = 0  # İlk yarı berabere, maç sonu mağlubiyet (X/2)
                
                match_count = len(recent_matches)
                
                if match_count > 0:
                    for match in recent_matches:
                        # İlk yarı/ikinci yarı gollerin yerini kontrol et
                        ht_result = match.get('ht_result', '')
                        ft_result = match.get('ft_result', '')
                        ht_code = match.get('ht_ft_code', '')
                        
                        # Tüm maçlarda ilk yarı ve ikinci yarı golleri topla
                        first_half_goals += match.get('ht_goals_scored', 0)
                        second_half_goals += match.get('goals_scored', 0) - match.get('ht_goals_scored', 0)
                        
                        # İkinci yarıda gol atma durumu
                        if match.get('goals_scored', 0) > match.get('ht_goals_scored', 0):
                            second_half_scored_count += 1
                        
                        # Tersine dönüş senaryolarını kontrol et
                        if ht_code == '1/2':  # İlk yarı önde, maç sonu geride
                            ht_win_ft_loss += 1
                        elif ht_code == '2/1':  # İlk yarı geride, maç sonu önde
                            ht_loss_ft_win += 1
                            comeback_count += 1
                        elif ht_code == 'X/1':  # İlk yarı berabere, maç sonu galibiyet
                            ht_draw_ft_win += 1
                        elif ht_code == 'X/2':  # İlk yarı berabere, maç sonu mağlubiyet
                            ht_draw_ft_loss += 1
                        
                        # Ek tersine dönüş durumları
                        # İlk yarı gerideyken maçı çevirme durumları (ht_code'dan bağımsız)
                        if match.get('ht_goals_scored', 0) < match.get('ht_goals_conceded', 0) and ft_result in ['W', 'D']:
                            comeback_count += 1
                    
                    # Performans değerlerini güncelle
                    if first_half_goals + second_half_goals > 0:
                        away_form_data['first_half_performance'] = first_half_goals / (first_half_goals + second_half_goals)
                        away_form_data['second_half_performance'] = second_half_goals / (first_half_goals + second_half_goals)
                    
                    # Tersine dönüş istatistiklerini kaydet
                    away_form_data['comeback_ratio'] = comeback_count / match_count
                    away_form_data['scores_in_second_half_ratio'] = second_half_scored_count / match_count
                    
                    # Ek analiz verileri
                    away_form_data['ht_win_ft_loss_ratio'] = ht_win_ft_loss / match_count
                    away_form_data['ht_loss_ft_win_ratio'] = ht_loss_ft_win / match_count
                    away_form_data['ht_draw_ft_win_ratio'] = ht_draw_ft_win / match_count
                    away_form_data['ht_draw_ft_loss_ratio'] = ht_draw_ft_loss / match_count
                    
                    # En olası tersine dönüş senaryosu
                    away_form_data['most_likely_switch'] = max(
                        [('1/2', ht_win_ft_loss), ('2/1', ht_loss_ft_win), 
                         ('X/1', ht_draw_ft_win), ('X/2', ht_draw_ft_loss)], 
                        key=lambda x: x[1]
                    )[0]
                    
                    logger.info(f"Deplasman tersine dönüş analizleri: Comeback: {away_form_data['comeback_ratio']:.2f}, " +
                                f"En olası dönüş: {away_form_data['most_likely_switch']}")
            except Exception as e:
                logger.error(f"Deplasman takımı için ilk yarı/ikinci yarı gol analizleri yapılırken hata: {str(e)}")
                # Hataya rağmen varsayılan değerler kullanarak devam et
        
        logger.info("İY/MS Tahmin - Takım istatistikleri: %s", {
            "ev": {
                "ilk_yari": home_first,
                "ikinci_yari": home_second,
                "mac_sayisi": home_matches
            },
            "deplasman": {
                "ilk_yari": away_first,
                "ikinci_yari": away_second,
                "mac_sayisi": away_matches
            }
        })
        
        # İlk yarı/ikinci yarı performans değerlerini logla
        logger.info("Takım İY/MS performans değerleri: %s", {
            "ev": {
                "ilk_yarı_performans": home_form_data.get('first_half_performance', 0.5) if home_form_data else 0.5,
                "ikinci_yarı_performans": home_form_data.get('second_half_performance', 0.5) if home_form_data else 0.5,
                "geri_dönüş_oranı": home_form_data.get('comeback_ratio', 0.1) if home_form_data else 0.1,
                "ikinci_yarıda_gol_atma_oranı": home_form_data.get('scores_in_second_half_ratio', 0.5) if home_form_data else 0.5
            },
            "deplasman": {
                "ilk_yarı_performans": away_form_data.get('first_half_performance', 0.5) if away_form_data else 0.5,
                "ikinci_yarı_performans": away_form_data.get('second_half_performance', 0.5) if away_form_data else 0.5,
                "geri_dönüş_oranı": away_form_data.get('comeback_ratio', 0.1) if away_form_data else 0.1,
                "ikinci_yarıda_gol_atma_oranı": away_form_data.get('scores_in_second_half_ratio', 0.5) if away_form_data else 0.5
            }
        })
        
        # İlk yarı performansları (maç başına gol) - güvenli erişim
        try:
            home_first_half_avg = home_first.get("avg_goals_per_match", 0.5)
        except (AttributeError, KeyError, TypeError):
            home_first_half_avg = 0.5  # Veri yoksa varsayılan değer
            
        try:
            away_first_half_avg = away_first.get("avg_goals_per_match", 0.5)
        except (AttributeError, KeyError, TypeError):
            away_first_half_avg = 0.5  # Veri yoksa varsayılan değer
        
        # İlk yarı gol farkı
        first_half_diff = home_first_half_avg - away_first_half_avg
        
        # İkinci yarı performansları (maç başına gol) - güvenli erişim
        try:
            home_second_half_avg = home_second.get("avg_goals_per_match", 0.5)
        except (AttributeError, KeyError, TypeError):
            home_second_half_avg = 0.5  # Veri yoksa varsayılan değer
            
        try:
            away_second_half_avg = away_second.get("avg_goals_per_match", 0.5)
        except (AttributeError, KeyError, TypeError):
            away_second_half_avg = 0.5  # Veri yoksa varsayılan değer
        
        # İkinci yarı gol farkı
        second_half_diff = home_second_half_avg - away_second_half_avg
        
        # İlk yarı tahminini hesapla
        if first_half_diff > 0.5:
            first_half_prediction = '1'  # Ev sahibi önde bitiriyor
        elif first_half_diff < -0.3:
            first_half_prediction = '2'  # Deplasman önde bitiriyor
        else:
            first_half_prediction = 'X'  # Berabere
        
        # İkinci yarıda gol sayısı farkına göre tahmin
        # ÖNEMLİ: Eğer global sonuç tahmini varsa, MS kısmını onunla tutarlı hale getir
        if global_outcome:
            # Tahmin butonu verilerini kullan (maç sonucu ile tutarlılık için)
            logger.info("Tahmin butonu sonucu: %s", global_outcome)
            
            match_outcome = None
            expected_home_goals = None
            expected_away_goals = None
            
            # Dict formatındaki global_outcome için
            if isinstance(global_outcome, dict):
                # Maç sonucu tahmini
                match_outcome = global_outcome.get("match_outcome")
                
                # Beklenen gol değerleri al
                expected_home_goals = global_outcome.get("expected_home_goals")
                if expected_home_goals is None:
                    expected_home_goals = global_outcome.get("score_prediction", {}).get("beklenen_home_gol")
                    
                expected_away_goals = global_outcome.get("expected_away_goals") 
                if expected_away_goals is None:
                    expected_away_goals = global_outcome.get("score_prediction", {}).get("beklenen_away_gol")
                    
                # Gol beklentilerine göre maç sonucu tahmini yap
                if match_outcome is None and expected_home_goals is not None and expected_away_goals is not None:
                    # Gol değerlerine göre maç sonucu tahmini
                    if expected_home_goals > expected_away_goals + 0.3:
                        match_outcome = "HOME_WIN"
                    elif expected_away_goals > expected_home_goals + 0.3:
                        match_outcome = "AWAY_WIN"
                    else:
                        match_outcome = "DRAW"
                        
                    logger.info(f"Beklenen gol değerlerine göre maç sonucu tahmini yapıldı: {match_outcome}")
            else:
                # String formatlı global_outcome (eski tip)
                match_outcome = global_outcome
            
            # MS kısmını tutarlı hale getir
            if match_outcome == "HOME_WIN":
                second_half_prediction = '1'  # Ev sahibi kazanıyor
            elif match_outcome == "AWAY_WIN":
                second_half_prediction = '2'  # Deplasman kazanıyor
            elif match_outcome == "DRAW":
                second_half_prediction = 'X'  # Berabere
            else:
                # Bilinmeyen durum, varsayılan olarak kombinasyon seç
                logger.warning(f"Beklenmeyen tahmin sonucu: {match_outcome}")
                second_half_prediction = 'X'  # Varsayılan olarak berabere
            
            logger.info(f"İY/MS tahminleri hesaplandı (tahmin butonu ile uyumlu): {first_half_prediction}/{second_half_prediction}")
        else:
            # Tahmin verisi yoksa, istatistiklerden hesapla
            # İkinci yarı + ilk yarıdan gelen avantaj
            combined_diff = second_half_diff + (first_half_diff * 0.3)
            
            if combined_diff > 0.4:
                second_half_prediction = '1'  # Ev sahibi kazanıyor
            elif combined_diff < -0.3:
                second_half_prediction = '2'  # Deplasman kazanıyor
            else:
                second_half_prediction = 'X'  # Berabere
        
        # İY/MS tahminini oluştur
        htft_prediction = f"{first_half_prediction}/{second_half_prediction}"
        
        # Tahmin butonundan gelen beklenen gol değerlerini al
        expected_home_goals = None
        expected_away_goals = None
        is_surprise_button = False
        
        if isinstance(global_outcome, dict):
            # Doğrudan expected_home_goals ve expected_away_goals anahtarlarını kontrol et
            expected_home_goals = global_outcome.get("expected_home_goals")
            expected_away_goals = global_outcome.get("expected_away_goals")
            
            # Eğer bulunamadıysa, eski API formatını dene
            if expected_home_goals is None:
                expected_home_goals = global_outcome.get("score_prediction", {}).get("beklenen_home_gol")
            if expected_away_goals is None:
                expected_away_goals = global_outcome.get("score_prediction", {}).get("beklenen_away_gol")
            
            # Sürpriz butonu için mi kontrol et
            is_surprise_button = global_outcome.get("is_surprise_button", False)
            
            # Düşük gol beklentisi kontrolü
            if expected_home_goals is not None and expected_away_goals is not None:
                total_expected = expected_home_goals + expected_away_goals
                if total_expected < 1.2:
                    logger.info(f"DÜŞÜK GOL BEKLENTİSİ: {total_expected:.2f} - İY/MS algoritması ayarlandı")
                    # Düşük gol beklentisinde 0-0 ilk yarı olasılığını artır
                    # İlk yarı tahmini güncellenir
                    first_half_prediction = 'X'  # Düşük gol beklentisinde ilk yarı berabere olasılığı yüksek
        
        # Varsayılan değerler ata (eğer değerler None ise)
        if expected_home_goals is None:
            expected_home_goals = 1.5
        if expected_away_goals is None:
            expected_away_goals = 1.0
            
        logger.info(f"İY/MS tahmini için gol beklentileri: Ev={expected_home_goals:.2f}, Deplasman={expected_away_goals:.2f}, Sürpriz={is_surprise_button}")
            
        # Deplasman kazanma ihtimali yüksek, düşük skorlu maçta ise
        if isinstance(global_outcome, dict) and expected_away_goals > expected_home_goals + 0.4 and (expected_home_goals + expected_away_goals) < 2.5:
            # Bu durumda X/2 ve 2/2 olasılıkları artmalı
            logger.info("Düşük skorlu maçta deplasman avantajlı - X/2 ve 2/2 olasılıkları arttırıldı")
        
        # YENİ PERFORMANS ALGORİTMASI: Takımların ilk ve ikinci yarı dağılımlarına göre
        # golleri ayırıp, Monte Carlo simülasyonu ile daha tutarlı olasılıklar hesapla
        final_probabilities = implement_gol_distribution_algorithm(
            home_stats, away_stats,
            expected_home_goals, expected_away_goals,
            is_surprise_button=is_surprise_button,
            team_adjustments=team_adjustments
        )
        
        logger.info(f"YENİ ALGORİTMA İLE HESAPLANAN İY/MS OLASILIKLARINI: {final_probabilities}")
        
        # En yüksek olasılıklı tahmini tekrar bulalım - bu sefer ayarlanmış olasılıklar kullanılacak!
        sorted_predictions = sorted(final_probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # En yüksek olasılıklı tahmin
        adjusted_prediction = sorted_predictions[0][0] if sorted_predictions else htft_prediction
        
        # En yüksek olasılıklı 3 tahmini bul
        top_predictions = find_top_predictions(final_probabilities, 3)
        
        # Eğer global maç sonucu tanımlanmışsa, İY/MS tahmini mutlaka bununla uyumlu olmalı
        if global_outcome:
            ms_outcome = second_half_prediction
            
            # Tahminimizin MS kısmı
            prediction_ms = adjusted_prediction.split('/')[1]
            
            # Eğer MS kısmı global sonuçla uyumlu değilse
            if prediction_ms != ms_outcome:
                logger.warning("Tahmin düzeltiliyor: Hesaplanan olasılıklarda bile uyumsuzluk var!")
                # MS ile uyumlu olan en yüksek olasılıklı tahmini bul
                compatible_predictions = [pred for pred in sorted_predictions if pred[0].split('/')[1] == ms_outcome]
                if compatible_predictions:
                    # En yüksek olasılıklı uyumlu tahmini kullan
                    return {
                        "prediction": compatible_predictions[0][0],
                        "top_predictions": top_predictions,
                        "all_probabilities": final_probabilities
                    }
        
        return {
            "prediction": adjusted_prediction,
            "top_predictions": top_predictions,
            "all_probabilities": final_probabilities
        }
        
    except Exception as e:
        logger.error("İY/MS tahmini hesaplanırken hata oluştu: %s", str(e), exc_info=True)
        return {
            "error": f"İY/MS tahmini hesaplanırken hata: {str(e)}",
            "predictions": {}
        }

def calculate_all_htft_probabilities(home_first, home_second, away_first, away_second, home_matches, away_matches, 
                           home_team_id=None, away_team_id=None, home_form=None, away_form=None):
    """
    Tüm İY/MS olasılıklarını hesapla
    Tahmin butonu ile tutarlı olacak şekilde daha dengeli olasılıklar üretir
    
    Args:
        home_first: Ev sahibi ilk yarı istatistikleri
        home_second: Ev sahibi ikinci yarı istatistikleri
        away_first: Deplasman ilk yarı istatistikleri
        away_second: Deplasman ikinci yarı istatistikleri
        home_matches: Ev sahibi maç sayısı
        away_matches: Deplasman maç sayısı
        home_team_id: Ev sahibi takım ID'si (opsiyonel, takım-spesifik ayarlar için)
        away_team_id: Deplasman takım ID'si (opsiyonel, takım-spesifik ayarlar için)
        home_form: Ev sahibi takım form verileri (opsiyonel)
        away_form: Deplasman takım form verileri (opsiyonel)
        
    Returns:
        İY/MS olasılıkları sözlüğü
    """
    # CRF modelini kullanarak tahmini destekle
    if CRF_AVAILABLE:
        try:
            # CRF modelini yükle
            crf_model = CRFPredictor()
            
            # Takım istatistiklerini hazırla
            home_stats = {
                "first_half": home_first,
                "second_half": home_second,
                "form": home_form
            }
            
            away_stats = {
                "first_half": away_first,
                "second_half": away_second,
                "form": away_form
            }
            
            # Takım-spesifik ayarlamaları al
            team_adjustments = None
            try:
                team_predictor = TeamSpecificPredictor()
                team_adjustments = team_predictor.get_team_adjustments(
                    home_team_id, away_team_id
                )
            except Exception as e:
                logger.warning(f"Takım-spesifik ayarlamalar alınamadı: {str(e)}")
            
            # Beklenen gol değerlerini hesapla - güvenli erişim
            try:
                home_first_avg = home_first.get("avg_goals_per_match", 0.5)
                home_second_avg = home_second.get("avg_goals_per_match", 0.5)
                away_first_avg = away_first.get("avg_goals_per_match", 0.5)
                away_second_avg = away_second.get("avg_goals_per_match", 0.5)
            except (AttributeError, KeyError, TypeError):
                home_first_avg = 0.5
                home_second_avg = 0.5
                away_first_avg = 0.5
                away_second_avg = 0.5
            
            expected_home_goals = home_first_avg + home_second_avg
            expected_away_goals = away_first_avg + away_second_avg
            
            expected_goals = {
                "total": expected_home_goals + expected_away_goals,
                "first_half": (home_first_avg + away_first_avg),
                "home": expected_home_goals,
                "away": expected_away_goals
            }
            
            # CRF tahmini için takım bilgilerini ekle - YENİ
            if team_adjustments is None:
                team_adjustments = {}
                
            # Takım ID'lerini ekle - YENİ
            if home_team_id:
                team_adjustments["home_team_id"] = home_team_id
            if away_team_id:
                team_adjustments["away_team_id"] = away_team_id
                
            # CRF modeli ile tahmin yap
            crf_predictions = crf_model.predict(home_stats, away_stats, expected_goals, team_adjustments)
            logger.info(f"CRF modeliyle tahmin sonuçları: {crf_predictions}")
            
            # CRF tahminleri ile diğer modelleri birleştirerek daha iyi sonuç elde ederiz
            # ancak önce geleneksel modellerle de hesaplama yapalım
        except Exception as e:
            logger.error(f"CRF tahmini sırasında hata: {str(e)}")
            # Hata durumunda CRF tahminleri kullanılamaz, klasik hesaplama ile devam et
    """
    Tüm İY/MS olasılıklarını hesapla
    Tahmin butonu ile tutarlı olacak şekilde daha dengeli olasılıklar üretir
    
    Args:
        home_first: Ev sahibi ilk yarı istatistikleri
        home_second: Ev sahibi ikinci yarı istatistikleri
        away_first: Deplasman ilk yarı istatistikleri
        away_second: Deplasman ikinci yarı istatistikleri
        home_matches: Ev sahibi maç sayısı
        away_matches: Deplasman maç sayısı
        home_team_id: Ev sahibi takım ID'si (opsiyonel, takım-spesifik ayarlar için)
        away_team_id: Deplasman takım ID'si (opsiyonel, takım-spesifik ayarlar için)
        home_form: Ev sahibi takım form verileri (opsiyonel)
        away_form: Deplasman takım form verileri (opsiyonel)
    """
    probabilities = {}
    total_weight = 100  # Toplam ağırlık
    
    # Log olarak maç sayılarını da yazdıralım
    logger.info("Total matches for each team: %s", {
        "ev_sahibi": home_matches or "unknown",
        "deplasman": away_matches or "unknown"
    })
    
    # Takım spesifik ayarlamaları uygula
    team_adjustments = None
    if home_team_id and away_team_id:
        try:
            team_specific_predictor = TeamSpecificPredictor()
            team_adjustments = team_specific_predictor.get_team_adjustments(
                home_team_id, away_team_id, home_form, away_form
            )
            logger.info("Takım-spesifik ayarlamalar uygulanıyor: %s", team_adjustments)
        except Exception as e:
            logger.warning("Takım-spesifik ayarlamalar uygulanırken hata: %s", str(e))
    
    # Gelişmiş modellerle tahminleri gerçekleştir
    
    # Takım istatistiklerini hazırla (hem CRF hem de Dirichlet için kullanılacak)
    home_stats = {
        "first_half": home_first,
        "second_half": home_second,
        "form": home_form
    }
    
    away_stats = {
        "first_half": away_first,
        "second_half": away_second,
        "form": away_form
    }
    
    # Beklenen gol değerlerini hesapla - güvenli erişim
    try:
        home_first_avg = home_first.get("avg_goals_per_match", 0.5)
        home_second_avg = home_second.get("avg_goals_per_match", 0.5)
        away_first_avg = away_first.get("avg_goals_per_match", 0.5)
        away_second_avg = away_second.get("avg_goals_per_match", 0.5)
    except (AttributeError, KeyError, TypeError):
        home_first_avg = 0.5
        home_second_avg = 0.5
        away_first_avg = 0.5
        away_second_avg = 0.5
        
    expected_home_goals = home_first_avg + home_second_avg
    expected_away_goals = away_first_avg + away_second_avg
    
    expected_goals = {
        "total": expected_home_goals + expected_away_goals,
        "first_half": (home_first_avg + away_first_avg),
        "home": expected_home_goals,
        "away": expected_away_goals
    }
    
    # CRF modeliyle tahmini gerçekleştir
    crf_predictions = None
    if CRF_AVAILABLE:
        try:
            # CRF modelini hazırla
            crf_model = CRFPredictor()
            
            # CRF tahmini için takım bilgilerini ekle - YENİ
            if team_adjustments is None:
                team_adjustments = {}
                
            # Takım ID'lerini ekle - YENİ
            if home_team_id:
                team_adjustments["home_team_id"] = home_team_id
            if away_team_id:
                team_adjustments["away_team_id"] = away_team_id
                
            # CRF modeli ile tahmin yap
            crf_predictions = crf_model.predict(home_stats, away_stats, expected_goals, team_adjustments)
            logger.info(f"CRF modeliyle tahmin sonuçları: {crf_predictions}")
        except Exception as e:
            logger.error(f"CRF tahmini sırasında hata: {str(e)}")
            
    # Dirichlet modeliyle tahmini gerçekleştir
    dirichlet_predictions = None
    if DIRICHLET_AVAILABLE:
        try:
            # Dirichlet modelini hazırla
            dirichlet_model = DirichletPredictor()
            
            # Dirichlet modeli ile tahmin yap
            dirichlet_predictions = dirichlet_model.predict(home_stats, away_stats, expected_goals, team_adjustments)
            logger.info(f"Dirichlet modeliyle tahmin sonuçları: {dirichlet_predictions}")
        except Exception as e:
            logger.error(f"Dirichlet tahmini sırasında hata: {str(e)}")
            # Hata durumunda CRF tahminleri kullanılamaz
    
    # Monte Carlo simülasyonu ve yapay sinir ağı uygulaması ekleyelim
    monte_carlo_probs = run_monte_carlo_simulation(home_first, home_second, away_first, away_second, 
                                                 10000, team_adjustments)
    neural_net_probs = predict_with_neural_network(home_first, home_second, away_first, away_second, 
                                                 team_adjustments)
                                                 
    # Form verilerini varsayılan değerlerle başlat
    form_motivation_factors = {
        "stat_weight_factor": 1.0,
        "neural_weight_factor": 1.0,
        "home_form_score": 0.5,
        "away_form_score": 0.5,
        "home_motivation": 0.5,
        "away_motivation": 0.5
    }
    
    # Form verilerini geçirelim (eğer varsa)
    if home_form is not None and away_form is not None:
        logger.info("Form verileri monte_carlo ve neural_network modellere başarıyla geçirildi")
    
    # Form verileri doğrudan olarak try-except bloğu içinde işle
    try:
        # Eğer form verileri doğru biçimde geldiyse kullan
        if home_form is not None and away_form is not None and isinstance(home_form, dict) and isinstance(away_form, dict):
            if "form_score" in home_form and "form_score" in away_form:
                # Form farkını hesapla
                form_diff = abs(home_form["form_score"] - away_form["form_score"])
                
                # Form farkı büyükse istatistiksel modele daha fazla güven
                stat_weight_factor = 1.0
                neural_weight_factor = 1.0
                
                if form_diff > 0.5:  # Büyük form farkı
                    stat_weight_factor = 1.2  # İstatistiksel modele %20 daha fazla güven
                    logger.info(f"Büyük form farkı tespit edildi: {form_diff:.2f} - İstatistiksel model ağırlığı artırıldı")
                
                # Form ve motivasyon faktörlerini güncelle
                form_motivation_factors.update({
                    "stat_weight_factor": stat_weight_factor,
                    "neural_weight_factor": neural_weight_factor,
                    "home_form_score": home_form.get("form_score", 0.5),
                    "away_form_score": away_form.get("form_score", 0.5),
                    "home_motivation": home_form.get("motivation", 0.5),
                    "away_motivation": away_form.get("motivation", 0.5)
                })
                
                logger.info("Form verileri monte_carlo ve neural_network modellere başarıyla geçirildi")
    except Exception as e:
        logger.warning(f"Form ve motivasyon parametreleri oluşturulurken hata: {str(e)}")
        # Hata durumunda varsayılan değerler kullanılacak    
    # Takımların toplam performansını hesapla - güvenli erişim
    try:
        home_first_avg = home_first.get("avg_goals_per_match", 0.5)
        home_second_avg = home_second.get("avg_goals_per_match", 0.5)
        away_first_avg = away_first.get("avg_goals_per_match", 0.5)
        away_second_avg = away_second.get("avg_goals_per_match", 0.5)
    except (AttributeError, KeyError, TypeError):
        home_first_avg = 0.5
        home_second_avg = 0.5
        away_first_avg = 0.5
        away_second_avg = 0.5
        
    home_total_avg = home_first_avg + home_second_avg
    away_total_avg = away_first_avg + away_second_avg
    
    # Ev avantajı - genelde ev sahibi takımlar %10-15 daha avantajlı 
    home_advantage = 1.15
    
    # İlk yarı ev sahibi önde bitirme olasılığı için temel değerler - güvenli erişim
    try:
        home_first_avg_goals = home_first.get("avg_goals_per_match", 0.5)
        away_first_avg_goals = away_first.get("avg_goals_per_match", 0.5)
    except (AttributeError, KeyError, TypeError):
        home_first_avg_goals = 0.5
        away_first_avg_goals = 0.5
        
    home_first_half_strength = (home_first_avg_goals * home_advantage - away_first_avg_goals * 0.9)
    
    # İlk yarı deplasman önde bitirme olasılığı için temel değerler
    away_first_half_strength = (away_first_avg_goals - home_first_avg_goals * 0.85)
    
    # İlk yarı berabere bitme olasılığı - futbolun doğasında ilk yarıda beraberlikler yaygındır
    draw_first_half_base = 0.45  # 0.35'ten 0.45'e yükseltildi
    
    # Düşük gol beklentili maçlarda ilk yarı beraberlik olasılığını ÇOK DAHA FAZLA artır
    total_first_half_goal_exp = home_first_avg_goals + away_first_avg_goals
    
    # Maçın toplam beklenen gol sayısına göre kademeli arttırma
    if total_first_half_goal_exp < 1.0:  # Çok düşük gol beklentisi (0.5-1.0 arası)
        draw_first_half_base = 0.75  # %75 beraberlik olasılığı (artırıldı - daha çok 0-0 ilk yarı)
        logger.info(f"Çok düşük gol beklentili maç, ilk yarı beraberlik olasılığı artırıldı: {draw_first_half_base}")
    elif total_first_half_goal_exp < 1.5:  # Düşük gol beklentisi (1.0-1.5 arası)
        draw_first_half_base = 0.65  # %65 beraberlik olasılığı (artırıldı)
        logger.info(f"Düşük gol beklentili maç, ilk yarı beraberlik olasılığı artırıldı: {draw_first_half_base}")
    elif total_first_half_goal_exp < 2.0:  # Normal-düşük gol beklentisi (1.5-2.0 arası)
        draw_first_half_base = 0.55  # %55 beraberlik olasılığı (artırıldı)
        logger.info(f"Normal-düşük gol beklentili maç, ilk yarı beraberlik olasılığı hafif artırıldı: {draw_first_half_base}")
    
    # Güçler arasındaki fark çok büyükse beraberlik olasılığını azalt - güvenli erişim
    first_half_diff = abs(home_first_avg_goals - away_first_avg_goals)
    if first_half_diff > 1.5:
        draw_first_half_base = max(0.2, draw_first_half_base - (first_half_diff * 0.15))
    
    # Gol beklentisi yüksekse beraberlik olasılığı düşer
    if total_first_half_goal_exp > 3:
        draw_first_half_base = max(0.2, draw_first_half_base - 0.2)  # 0.1'den 0.2'ye değiştirildi - daha agresif düşüş
    
    # İlk yarı berabere olasılığı
    draw_first_half_strength = draw_first_half_base - (abs(home_first_half_strength - away_first_half_strength) * 0.2)
    draw_first_half_strength = max(0.15, draw_first_half_strength)  # Minimum %15 olasılık
    
    # Negatif değerlerin önüne geç
    home_first_half_strength = max(0.1, home_first_half_strength)
    away_first_half_strength = max(0.1, away_first_half_strength)
    
    # Tam maç sonucu için olasılıklar
    home_win_strength = (home_total_avg * home_advantage - away_total_avg * 0.9)
    away_win_strength = (away_total_avg - home_total_avg * 0.85)
    
    # Beraberlik olasılığı için temel değer
    draw_full_time_base = 0.3
    
    # Tam maç gol sayısı düşükse beraberlik olasılığını arttır
    if home_total_avg < 2.5 and away_total_avg < 2.5:
        draw_full_time_base = 0.45  # 0.4'ten 0.45'e yükseltildi - daha fazla beraberlik
    
    # Çok düşük gol beklentili maçlarda beraberlik olasılığını daha da artır (yeni)
    if home_total_avg < 1.5 and away_total_avg < 1.5:
        draw_full_time_base = 0.55  # 0-0 ve 1-1 sonuçları daha olası
    
    # Güçler arasındaki fark çok büyükse beraberlik olasılığını azalt
    full_time_diff = abs(home_total_avg - away_total_avg)
    if full_time_diff > 2:
        draw_full_time_base = max(0.15, draw_full_time_base - (full_time_diff * 0.1))
    
    # Tam maç için beraberlik olasılığı
    draw_full_time_strength = draw_full_time_base - (abs(home_win_strength - away_win_strength) * 0.15)
    draw_full_time_strength = max(0.12, draw_full_time_strength)  # Minimum %12 olasılık
    
    # Negatif değerlerin önüne geç
    home_win_strength = max(0.1, home_win_strength)
    away_win_strength = max(0.1, away_win_strength)
    
    # Takım karakteristiklerini dikkate al - birçok takım için özel durumları modelle
    # Örnek: Atletico Madrid gibi takımlar ilk yarıyı genelde önde bitirip ikinci yarı düşüş yaşar
    
    # İkinci yarı düşüş yaşayan takımları tespit et - güvenli erişim
    try:
        home_first_avg_goals = home_first.get("avg_goals_per_match", 0.5)
        home_second_avg_goals = home_second.get("avg_goals_per_match", 0.5)
        away_first_avg_goals = away_first.get("avg_goals_per_match", 0.5)
        away_second_avg_goals = away_second.get("avg_goals_per_match", 0.5)
    except (AttributeError, KeyError, TypeError):
        home_first_avg_goals = 0.5
        home_second_avg_goals = 0.5
        away_first_avg_goals = 0.5
        away_second_avg_goals = 0.5
        
    home_second_half_drop = home_first_avg_goals > (home_second_avg_goals * 1.3)
    away_second_half_drop = away_first_avg_goals > (away_second_avg_goals * 1.3)
    
    # İkinci yarı yükselen takımları tespit et
    home_second_half_rise = home_second_avg_goals > (home_first_avg_goals * 1.3)
    away_second_half_rise = away_second_avg_goals > (away_first_avg_goals * 1.3)
    
    # İlk yarı olasılıkları için normalleştirme
    first_half_total = home_first_half_strength + away_first_half_strength + draw_first_half_strength
    home_first_prob = home_first_half_strength / first_half_total
    away_first_prob = away_first_half_strength / first_half_total
    draw_first_prob = draw_first_half_strength / first_half_total
    
    # Tam maç olasılıkları için normalleştirme
    full_time_total = home_win_strength + away_win_strength + draw_full_time_strength
    home_win_prob = home_win_strength / full_time_total
    away_win_prob = away_win_strength / full_time_total
    draw_full_time_prob = draw_full_time_strength / full_time_total
    
    # Tüm kombinasyonlar için başlangıç olasılıklarını hesapla
    # İY/MS kombinasyonları için olasılıkları hesapla (daha gerçekçi korelasyon faktörleriyle)
    probabilities['1/1'] = round((home_first_prob * home_win_prob * 1.1) * 100)  # 1.3'ten 1.1'e düşürüldü
    probabilities['1/X'] = round((home_first_prob * draw_full_time_prob * 1.3) * 100)  # 1.1'den 1.3'e yükseltildi
    probabilities['1/2'] = round((home_first_prob * away_win_prob * 1.2) * 100)  # 0.9'dan 1.2'ye yükseltildi
    probabilities['X/1'] = round((draw_first_prob * home_win_prob * 1.4) * 100)  # 1.2'den 1.4'e yükseltildi
    probabilities['X/X'] = round((draw_first_prob * draw_full_time_prob * 1.3) * 100)  # Değişmedi
    probabilities['X/2'] = round((draw_first_prob * away_win_prob * 1.4) * 100)  # 1.2'den 1.4'e yükseltildi
    probabilities['2/1'] = round((away_first_prob * home_win_prob * 1.2) * 100)  # 0.8'den 1.2'ye yükseltildi
    probabilities['2/X'] = round((away_first_prob * draw_full_time_prob * 1.3) * 100)  # 1.1'den 1.3'e yükseltildi
    probabilities['2/2'] = round((away_first_prob * away_win_prob * 1.15) * 100)  # 1.4'ten 1.15'e düşürüldü
    
    # Takım özelliklerine göre ayarlamalar yap
    # İlk yarı önde bitirip ikinci yarı düşüş yaşayan takımlar için
    if home_second_half_drop:
        # Ev sahibi ilk yarı önde bitirip sonra berabere/yenilme olasılığı artar
        # 1/2 özel güçlendirme - ev sahibi ilk yarı önde bitirip maçı kaybetme olasılığı
        probabilities['1/X'] = round(probabilities['1/X'] * 1.4)
        probabilities['1/2'] = round(probabilities['1/2'] * 1.8)  # 1.3'ten 1.8'e güçlendirildi
        probabilities['1/1'] = round(probabilities['1/1'] * 0.85)  # 0.9'dan 0.85'e düşürüldü
    
    if away_second_half_drop:
        # Deplasman ilk yarı önde bitirip sonra berabere/yenilme olasılığı artar
        # 2/1 özel güçlendirme - deplasman ilk yarı önde bitirip maçı kaybetme olasılığı
        probabilities['2/X'] = round(probabilities['2/X'] * 1.4)
        probabilities['2/1'] = round(probabilities['2/1'] * 1.8)  # 1.3'ten 1.8'e güçlendirildi
        probabilities['2/2'] = round(probabilities['2/2'] * 0.85)  # 0.9'dan 0.85'e düşürüldü
    
    # İkinci yarı yükselen takımlar için
    if home_second_half_rise:
        # Ev sahibi ilk yarı geride/berabere olup kazanma olasılığı artar
        probabilities['X/1'] = round(probabilities['X/1'] * 1.6)  # 1.5'ten 1.6'ya güçlendirildi
        probabilities['2/1'] = round(probabilities['2/1'] * 1.9)  # 1.4'ten 1.9'a güçlendirildi
    
    if away_second_half_rise:
        # Deplasman ilk yarı geride/berabere olup kazanma olasılığı artar
        probabilities['X/2'] = round(probabilities['X/2'] * 1.6)  # 1.5'ten 1.6'ya güçlendirildi
        probabilities['1/2'] = round(probabilities['1/2'] * 1.9)  # 1.4'ten 1.9'a güçlendirildi
    
    # ÖZEL 1/2 ve 2/1 SENARYOLARI - Erken gol atma eğilimli takımlar
    home_early_goal_tendency = False
    away_early_goal_tendency = False
    home_comeback_ability = False
    away_comeback_ability = False
    
    # Takım verilerini güvenli bir şekilde al
    try:
        # Ev sahibinin son maçlarında erken gol atma eğilimi
        home_recent_match_data = []
        if isinstance(home_form, dict) and 'recent_match_data' in home_form:
            home_recent_match_data = home_form.get('recent_match_data', [])
            
        if home_recent_match_data and len(home_recent_match_data) > 0:
            for match in home_recent_match_data[:5]:  # Son 5 maç
                # İlk yarı gol attıysa ve ilk yarı önde bitirdiyse
                if match.get('ht_goals_scored', 0) > 0 and match.get('ht_goals_scored', 0) > match.get('ht_goals_conceded', 0):
                    home_early_goal_tendency = True
                    break
                    
            # Ev sahibinin son maçlarında geride başlayıp kazanma/berabere kalma yeteneği
            for match in home_recent_match_data[:5]:  # Son 5 maç
                # İlk yarı gerideyken maçı kazandı veya berabere kaldıysa
                if (match.get('ht_goals_scored', 0) < match.get('ht_goals_conceded', 0)) and \
                   (match.get('goals_scored', 0) >= match.get('goals_conceded', 0)):
                    home_comeback_ability = True
                    break
                    
        # Deplasman takımının son maçlarında erken gol atma eğilimi
        away_recent_match_data = []
        if isinstance(away_form, dict) and 'recent_match_data' in away_form:
            away_recent_match_data = away_form.get('recent_match_data', [])
            
        if away_recent_match_data and len(away_recent_match_data) > 0:
            for match in away_recent_match_data[:5]:  # Son 5 maç
                # İlk yarı gol attıysa ve ilk yarı önde bitirdiyse
                if match.get('ht_goals_scored', 0) > 0 and match.get('ht_goals_scored', 0) > match.get('ht_goals_conceded', 0):
                    away_early_goal_tendency = True
                    break
            
            # Deplasman takımının son maçlarında geride başlayıp kazanma/berabere kalma yeteneği
            for match in away_recent_match_data[:5]:  # Son 5 maç
                # İlk yarı gerideyken maçı kazandı veya berabere kaldıysa
                if (match.get('ht_goals_scored', 0) < match.get('ht_goals_conceded', 0)) and \
                   (match.get('goals_scored', 0) >= match.get('goals_conceded', 0)):
                    away_comeback_ability = True
                    break
                    
        # Geri dönüş analizlerinin sonuçlarını logla
        logger.info(f"HTFT Özel Faktörler - Ev sahibi: Erken gol={home_early_goal_tendency}, Comeback={home_comeback_ability}, " + 
                   f"Deplasman: Erken gol={away_early_goal_tendency}, Comeback={away_comeback_ability}")
                   
        # Erken gol atma eğilimi + rakip comeback yeteneği varsa, 1/2 veya 2/1 olasılığı artırılır
        if home_early_goal_tendency and away_comeback_ability:
            probabilities['1/2'] = round(probabilities['1/2'] * 2.2)  # 1/2 olasılığını %120 artır
            logger.info(f"Ev sahibi erken gol, deplasman comeback yeteneği tespit edildi: 1/2 olasılığı artırıldı")
        
        if away_early_goal_tendency and home_comeback_ability:
            probabilities['2/1'] = round(probabilities['2/1'] * 2.2)  # 2/1 olasılığını %120 artır
            logger.info(f"Deplasman erken gol, ev sahibi comeback yeteneği tespit edildi: 2/1 olasılığı artırıldı")
            
        # Özel 1/2 ve 2/1 senaryoları için ek analizler
        home_dominant_start_weak_finish = False
        away_dominant_start_weak_finish = False
        
        # Ev sahibi ilk yarı iyi ikinci yarı kötü performans gösterme eğilimi - güvenli erişim
        if isinstance(home_first, dict) and isinstance(home_second, dict):
            home_first_avg = home_first.get("avg_goals_per_match", 0)
            home_second_avg = home_second.get("avg_goals_per_match", 0)
            if home_first_avg > 1.3 and home_first_avg > home_second_avg * 1.3:
                home_dominant_start_weak_finish = True
                logger.info(f"Ev sahibi ilk yarı güçlü, ikinci yarı zayıf eğilim tespit edildi")
                probabilities['1/2'] = round(probabilities['1/2'] * 1.7)  # 1/2 olasılığını %70 artır
                probabilities['1/X'] = round(probabilities['1/X'] * 1.4)  # 1/X olasılığını %40 artır
                
        # Deplasman ilk yarı iyi ikinci yarı kötü performans gösterme eğilimi - güvenli erişim
        if isinstance(away_first, dict) and isinstance(away_second, dict):
            away_first_avg = away_first.get("avg_goals_per_match", 0)
            away_second_avg = away_second.get("avg_goals_per_match", 0)
            if away_first_avg > 1.3 and away_first_avg > away_second_avg * 1.3:
                away_dominant_start_weak_finish = True
                logger.info(f"Deplasman ilk yarı güçlü, ikinci yarı zayıf eğilim tespit edildi")
                probabilities['2/1'] = round(probabilities['2/1'] * 1.7)  # 2/1 olasılığını %70 artır
                probabilities['2/X'] = round(probabilities['2/X'] * 1.4)  # 2/X olasılığını %40 artır
    except Exception as e:
        logger.warning(f"HTFT özel faktör analizinde hata: {e}")
    
    # En düşük olasılık oranlarını dengele (bir tahmin çok baskın olmasın)
    max_probability = max(probabilities.values())
    
    # Eğer bir olasılık çok yüksekse (%70'den fazla), diğerlerini biraz artır
    if max_probability > 70:
        for key in probabilities:
            if probabilities[key] < 5:
                probabilities[key] = 5 + random.randint(0, 4)  # 5-9 arası bir değer
    
    # İstatistiklere göre bazı özel durumları ele al
    # 1. Çok gollü takımlar için (ortalama 3+ gol/maç)
    if home_total_avg > 3 and away_total_avg > 2.5:
        # Her iki takım da çok gol atıyorsa, X/X olasılığını azalt
        probabilities['X/X'] = max(5, round(probabilities['X/X'] * 0.8))
    
    # 2. Az gollü takımlar için (ortalama 1- gol/maç)
    if home_total_avg < 1 and away_total_avg < 1:
        # Her iki takım da az gol atıyorsa, X/X olasılığını artır
        probabilities['X/X'] = min(60, round(probabilities['X/X'] * 1.6))
        
        # İkinci olarak en olası X/1 veya X/2 olasılıklarını azalt
        if home_win_prob > away_win_prob:
            probabilities['X/1'] = round(probabilities['X/1'] * 0.7)
        else:
            probabilities['X/2'] = round(probabilities['X/2'] * 0.7)
    
    # Üç farklı modeli birleştirerek daha güvenilir tahminler üret
    statistical_probs = probabilities.copy()  # İstatistik temelli model - %40 ağırlık
    
    # Modelleri birleştir - form ve motivasyon faktörlerini ve takım ayarlamalarını parametre olarak geçir
    # CRF ve Dirichlet modellerini de dahil ediyoruz (mevcutsa)
    combined_probs = combine_model_results(statistical_probs, monte_carlo_probs, neural_net_probs, form_motivation_factors, team_adjustments, crf_predictions, dirichlet_predictions)
    
    # YENİ YAKLAŞIM (28.03.2025) - İlk yarı ve maç sonu performanslarını ayrı ayrı değerlendir
    # Kullanıcı geri bildirimine göre güncellendi
    
    # İlk yarı olasılıkları toplamları
    ht_1_total = combined_probs.get('1/1', 0) + combined_probs.get('1/X', 0) + combined_probs.get('1/2', 0)
    ht_X_total = combined_probs.get('X/1', 0) + combined_probs.get('X/X', 0) + combined_probs.get('X/2', 0)
    ht_2_total = combined_probs.get('2/1', 0) + combined_probs.get('2/X', 0) + combined_probs.get('2/2', 0)
    
    # Maç sonu olasılıkları toplamları
    ft_1_total = combined_probs.get('1/1', 0) + combined_probs.get('X/1', 0) + combined_probs.get('2/1', 0)
    ft_X_total = combined_probs.get('1/X', 0) + combined_probs.get('X/X', 0) + combined_probs.get('2/X', 0)
    ft_2_total = combined_probs.get('1/2', 0) + combined_probs.get('X/2', 0) + combined_probs.get('2/2', 0)
    
    # İlk yarı favorisi
    ht_favorite = "1" if ht_1_total > max(ht_X_total, ht_2_total) + 5 else ("2" if ht_2_total > max(ht_1_total, ht_X_total) + 5 else "X")
    
    # Maç sonu favorisi
    ft_favorite = "1" if ft_1_total > max(ft_X_total, ft_2_total) + 5 else ("2" if ft_2_total > max(ft_1_total, ft_X_total) + 5 else "X")
    
    logger.info(f"İlk yarı favorisi: {ht_favorite}, Maç sonu favorisi: {ft_favorite}")
    
    # 1. Maç sonu sonuçlarını tutarlı hale getir
    if ft_favorite == "1":  # Ev sahibi maç sonu favorisi
        # Ev sahibinin kazandığı tüm kombinasyonların olasılıklarını güçlendir
        for htft in ["1/1", "X/1", "2/1"]:
            combined_probs[htft] = int(combined_probs.get(htft, 0) * 1.15)
        logger.info("Tutarlılık: Ev sahibi (1) maç sonu favorisi, tüm */1 kombinasyonları güçlendirildi")
    
    elif ft_favorite == "2":  # Deplasman maç sonu favorisi
        # Deplasmanın kazandığı tüm kombinasyonların olasılıklarını güçlendir  
        for htft in ["1/2", "X/2", "2/2"]:
            combined_probs[htft] = int(combined_probs.get(htft, 0) * 1.15)
        logger.info("Tutarlılık: Deplasman (2) maç sonu favorisi, tüm */2 kombinasyonları güçlendirildi")
    
    elif ft_favorite == "X":  # Beraberlik maç sonu favorisi
        # Beraberlikle biten tüm kombinasyonların olasılıklarını güçlendir
        for htft in ["1/X", "X/X", "2/X"]:  
            combined_probs[htft] = int(combined_probs.get(htft, 0) * 1.15)
        logger.info("Tutarlılık: Beraberlik (X) maç sonu favorisi, tüm */X kombinasyonları güçlendirildi")
    
    # 2. İlk yarı sonuçlarını tutarlı hale getir
    if ht_favorite == "1":  # Ev sahibi ilk yarı favorisi
        # Ev sahibinin ilk yarıda önde olduğu tüm kombinasyonları güçlendir
        for htft in ["1/1", "1/X", "1/2"]:
            combined_probs[htft] = int(combined_probs.get(htft, 0) * 1.15)
        logger.info("Tutarlılık: Ev sahibi (1) ilk yarı favorisi, tüm 1/* kombinasyonları güçlendirildi")
    
    elif ht_favorite == "2":  # Deplasman ilk yarı favorisi
        # Deplasmanın ilk yarıda önde olduğu tüm kombinasyonları güçlendir
        for htft in ["2/1", "2/X", "2/2"]:
            combined_probs[htft] = int(combined_probs.get(htft, 0) * 1.15)
        logger.info("Tutarlılık: Deplasman (2) ilk yarı favorisi, tüm 2/* kombinasyonları güçlendirildi")
    
    elif ht_favorite == "X":  # İlk yarı beraberlik favorisi
        # İlk yarı beraberlik olan tüm kombinasyonları güçlendir
        for htft in ["X/1", "X/X", "X/2"]:
            combined_probs[htft] = int(combined_probs.get(htft, 0) * 1.15)
        logger.info("Tutarlılık: Beraberlik (X) ilk yarı favorisi, tüm X/* kombinasyonları güçlendirildi")
    
    # 3. En tutarlı kombinasyonu ekstra güçlendir - hem ilk yarı hem maç sonu favorileriyle uyumlu
    if ht_favorite != "?" and ft_favorite != "?":
        best_combo = f"{ht_favorite}/{ft_favorite}"
        if best_combo in combined_probs:
            # Her iki favoriyle uyumlu kombinasyon
            combined_probs[best_combo] = int(combined_probs.get(best_combo, 0) * 1.1)
            logger.info(f"En tutarlı kombinasyon güçlendirildi: {best_combo}")
            
    # 4. "Tersine dönüş" kombinasyonlarını (1/2 ve 2/1) analiz et
    # NOT: 28.03.2025 İlk yarı / maç sonu tersine dönüş kombinasyonları için özel analiz
    # İlk yarı ve maç sonu favorileri farklı kutupta ise (1 ve 2)
    has_reverse_potential = False
    
    # Takımların form verilerini ve ikinci yarı performanslarını kontrol et
    if isinstance(home_form, dict) and isinstance(away_form, dict):
        # İkinci yarı performans değerlerini al (0-1 arası, yüksek değer daha iyi)
        home_second_half_strong = home_form.get('second_half_performance', 0.5) > 0.6
        away_second_half_strong = away_form.get('second_half_performance', 0.5) > 0.6
        
        # Son 5 maçta ikinci yarıda gol atabilme
        home_scores_in_second_half = home_form.get('scores_in_second_half_ratio', 0.5) > 0.6
        away_scores_in_second_half = away_form.get('scores_in_second_half_ratio', 0.5) > 0.6
        
        # Son 5 maçta geri dönüş (kayıp durumda iken beraberlik/galibiyet alabilme)
        home_comeback_ability = home_form.get('comeback_ratio', 0.0) > 0.2  # %20'den fazla geri dönebilme
        away_comeback_ability = away_form.get('comeback_ratio', 0.0) > 0.2  # %20'den fazla geri dönebilme
        
        # Tersine dönüş potansiyelini belirle
        # 1/2 kombinasyonu: Ev sahibi ilk yarı iyi ama ikinci yarı kötü, deplasman ikinci yarı güçlü
        potential_for_1_2 = home_form.get('first_half_performance', 0.5) > 0.6 and away_second_half_strong and away_scores_in_second_half
        
        # 2/1 kombinasyonu: Deplasman ilk yarı iyi ama ikinci yarı kötü, ev sahibi ikinci yarı güçlü
        potential_for_2_1 = away_form.get('first_half_performance', 0.5) > 0.6 and home_second_half_strong and home_scores_in_second_half
        
        # Geri dönüş yeteneği olan takım
        if (ft_favorite == "1" and ht_favorite == "2" and home_comeback_ability) or \
           (ft_favorite == "2" and ht_favorite == "1" and away_comeback_ability):
            has_reverse_potential = True
            logger.info(f"Tersine dönüş potansiyeli tespit edildi: {ht_favorite}/{ft_favorite}, ev sahibi geri dönüş: {home_comeback_ability}, deplasman geri dönüş: {away_comeback_ability}")
            
        # İkinci yarı performansına dayalı tersine dönüş
        if potential_for_1_2 or potential_for_2_1:
            has_reverse_potential = True
            logger.info(f"İkinci yarı performansına dayalı tersine dönüş potansiyeli tespit edildi: potential_for_1_2={potential_for_1_2}, potential_for_2_1={potential_for_2_1}")
    
    # Tersine dönüş potansiyeli varsa 1/2 ve 2/1 kombinasyonlarını güçlendir
    if has_reverse_potential:
        # Tersine dönüş potansiyeline göre olasılıkları güçlendir
        if ht_favorite == "1" and ft_favorite == "2":
            # 1/2 kombinasyonu (ev sahibi önde başlayıp deplasman kazanıyor)
            boost_factor = 1.35  # %35 artış
            combined_probs["1/2"] = int(combined_probs.get("1/2", 0) * boost_factor)
            logger.info(f"1/2 kombinasyonu güçlendirildi: Tersine dönüş potansiyeli nedeniyle. Yeni olasılık: {combined_probs['1/2']}")
            
        elif ht_favorite == "2" and ft_favorite == "1":
            # 2/1 kombinasyonu (deplasman önde başlayıp ev sahibi kazanıyor)
            boost_factor = 1.35  # %35 artış
            combined_probs["2/1"] = int(combined_probs.get("2/1", 0) * boost_factor)
            logger.info(f"2/1 kombinasyonu güçlendirildi: Tersine dönüş potansiyeli nedeniyle. Yeni olasılık: {combined_probs['2/1']}")
            
    else:
        # Tersine dönüş potansiyeli yoksa, bu kombinasyonları zayıflat
        # Diğer durumlar için tam zıt sonuçları biraz zayıflat
        if (ht_favorite == "1" and ft_favorite != "2") and combined_probs.get("1/2", 0) > 7:
            combined_probs["1/2"] = int(combined_probs.get("1/2", 0) * 0.8)
            logger.info("1/2 kombinasyonu indirgendi: Tersine dönüş potansiyeli tespit edilmedi.")
            
        elif (ht_favorite == "2" and ft_favorite != "1") and combined_probs.get("2/1", 0) > 7:
            combined_probs["2/1"] = int(combined_probs.get("2/1", 0) * 0.8)
            logger.info("2/1 kombinasyonu indirgendi: Tersine dönüş potansiyeli tespit edilmedi.")
    
    # Toplam 100'e olacak şekilde ayarla
    total_prob = sum(combined_probs.values())
    
    # Toplam tam 100 değilse, en yüksek olasılığı ayarla
    if total_prob != 100:
        diff = 100 - total_prob
        # En yüksek olasılıklı tahmini bul - max() kullanmak yerine loop kullanarak
        highest_key = None
        highest_val = -1
        for key, val in combined_probs.items():
            if val > highest_val:
                highest_val = val
                highest_key = key
        
        if highest_key:
            combined_probs[highest_key] += diff
    
    # 1/2 ve 2/1 sonuçlarının olasılıklarını analiz et (sürpriz sonuçlar)
    surprise_ht_ft_predictions = {
        "1/2": combined_probs.get("1/2", 0),
        "2/1": combined_probs.get("2/1", 0)
    }
    
    # Eğer sürpriz tahminler yüksek olasılıklı ise bunun nedenini açıkla
    if surprise_ht_ft_predictions["1/2"] > 12:  # %12'den fazla ise önemli bir olasılık
        logger.info(f"Yüksek 1/2 olasılığı tespit edildi: %{surprise_ht_ft_predictions['1/2']}")
        # Bu durumun oluşma nedenlerini analiz et
        if isinstance(home_form, dict) and isinstance(away_form, dict):
            home_first_half_good = home_form.get('first_half_performance', 0) > 0.6
            away_second_half_good = away_form.get('second_half_performance', 0) > 0.7
            logger.info(f"1/2 analizi: Ev sahibi ilk yarı performans={home_form.get('first_half_performance', 0)}, " +
                       f"Deplasman ikinci yarı performans={away_form.get('second_half_performance', 0)}")
    
    if surprise_ht_ft_predictions["2/1"] > 12:  # %12'den fazla ise önemli bir olasılık
        logger.info(f"Yüksek 2/1 olasılığı tespit edildi: %{surprise_ht_ft_predictions['2/1']}")
        # Bu durumun oluşma nedenlerini analiz et
        if isinstance(home_form, dict) and isinstance(away_form, dict):
            away_first_half_good = away_form.get('first_half_performance', 0) > 0.6
            home_second_half_good = home_form.get('second_half_performance', 0) > 0.7
            logger.info(f"2/1 analizi: Deplasman ilk yarı performans={away_form.get('first_half_performance', 0)}, " +
                       f"Ev sahibi ikinci yarı performans={home_form.get('second_half_performance', 0)}")
        
    # Modelleri logla
    logger.info("İY/MS Tahmin Modelleri Sonuçları: %s", {
        "istatistik": statistical_probs,
        "monteCarlo": monte_carlo_probs,
        "neuralNetwork": neural_net_probs,
        "combined": combined_probs,
        "sürpriz_tahminler": surprise_ht_ft_predictions
    })
    
    return combined_probs

def find_top_predictions(probabilities, count):
    """
    En yüksek olasılıklı tahminleri bul
    """
    # Olasılıkları sırala
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_items[:count]
    
    # Dict formatına dönüştür
    return [{"prediction": pred, "probability": prob} for pred, prob in top_items]

def get_htft_description(prediction):
    """
    İY/MS tahminin açıklamasını al
    """
    descriptions = {
        '1/1': 'Ev Sahibi Önde/Ev Sahibi Kazanır',
        '1/X': 'Ev Sahibi Önde/Berabere',
        '1/2': 'Ev Sahibi Önde/Deplasman Kazanır',
        'X/1': 'İlk Yarı Berabere/Ev Sahibi Kazanır',
        'X/X': 'İlk Yarı Berabere/Maç Berabere',
        'X/2': 'İlk Yarı Berabere/Deplasman Kazanır',
        '2/1': 'Deplasman Önde/Ev Sahibi Kazanır',
        '2/X': 'Deplasman Önde/Berabere',
        '2/2': 'Deplasman Önde/Deplasman Kazanır'
    }
    
    return descriptions.get(prediction, 'Bilinmeyen Tahmin')

def run_monte_carlo_simulation(home_first, home_second, away_first, away_second, simulation_count=10000, team_adjustments=None):
    """
    Monte Carlo simülasyonu ile İY/MS tahminleri yap
    
    Args:
        home_first: Ev sahibi ilk yarı istatistikleri
        home_second: Ev sahibi ikinci yarı istatistikleri 
        away_first: Deplasman ilk yarı istatistikleri
        away_second: Deplasman ikinci yarı istatistikleri
        simulation_count: Simülasyon sayısı (default: 10000)
        team_adjustments: Takım-spesifik ayarlamalar (team_specific_models.py modülünden)
    
    Returns:
        İY/MS olasılıkları
        
    Not: İY/MS kombinasyonlarının tutarlılığını artırmak için, maç sonu favorisi olan takımın 
    ilk yarıda da önde olduğu kombinasyonlar daha yüksek olasılık alır. Tersine kombinasyonlar
    (1/2, 2/1 gibi) için ise takımların ikinci yarı performansı ve geri dönüş istatistikleri
    özel olarak analiz edilir.
    """
    # SÜRPRİZ BUTONU İYİLEŞTİRME 2:
    # Monte Carlo simülasyonunda çeşitliliği artıracak faktörler ekleyelim
    # Bu iyileştirme, nadir İY/MS kombinasyonlarına daha yüksek olasılıklar vererek
    # tahminlerin 1/1 ve 2/2'den uzaklaşmasını sağlayacak
    
    # Trend analizine göre ek faktörler
    # Varsayılan faktörler - sürpriz sonuçları artırıyoruz
    special_factors = {
        "goal_diff_variability": 1.5,     # İlk ve ikinci yarı arasındaki gol farkı değişkenliği (1.0'dan 1.5'e yükseltildi)
        "htft_1_2_boost": 2.0,            # 1/2 için özel katsayı - %100 artış (1.25'ten 2.0'a yükseltildi)
        "htft_2_1_boost": 2.0,            # 2/1 için özel katsayı - %100 artış (1.25'ten 2.0'a yükseltildi)
        "htft_X_1_boost": 1.8,            # X/1 için özel katsayı - %80 artış (yeni eklendi)
        "htft_X_2_boost": 1.8,            # X/2 için özel katsayı - %80 artış (yeni eklendi)
        "htft_1_X_boost": 1.6,            # 1/X için özel katsayı - %60 artış (yeni eklendi)
        "htft_2_X_boost": 1.6             # 2/X için özel katsayı - %60 artış (yeni eklendi)
    }
    
    # Takım-spesifik ayarlamalar
    home_factor = 1.0
    away_factor = 1.0
    comeback_factor = 1.5  # İkinci yarı toparlanma faktörü (2/1 veya 1/2 olasılıklarını etkiler) - 1.0'dan 1.5'e çıkarıldı
    reversal_boost = 1.8  # Skor tersine dönüş faktörü - 1.0'dan 1.8'e çıkarıldı
    
    if team_adjustments:
        try:
            # Takım stil faktörlerini al
            home_style = team_adjustments.get("home_team_style", {})
            away_style = team_adjustments.get("away_team_style", {})
            
            # Takım-spesifik faktörleri al
            home_factors = home_style.get("factors", {})
            away_factors = away_style.get("factors", {})
            
            # Gol atma/yeme faktörleri
            home_goal_factor = home_factors.get("goal_factor", 1.0)
            home_concede_factor = home_factors.get("concede_factor", 1.0)
            away_goal_factor = away_factors.get("goal_factor", 1.0) 
            away_concede_factor = away_factors.get("concede_factor", 1.0)
            
            # Takım güç farkı
            team_power_diff = team_adjustments.get("power_difference", 0.0)
            
            # Stil faktörleri
            home_second_half_power = home_style.get("second_half_strength", 1.0)
            away_second_half_power = away_style.get("second_half_strength", 1.0)
            
            # Takım faktörlerini ayarla
            home_factor = home_goal_factor
            away_factor = away_goal_factor
            
            # İkinci yarı toparlanma faktörü
            # Eğer bir takım ikinci yarıda daha güçlü ise, 2/1 veya 1/2 olasılıkları artar
            comeback_factor = (home_second_half_power / 1.0) * 1.2 if home_second_half_power > 1.1 else 1.0
            
            # Skor tersine dönüş faktörü
            # Ev sahibi ve deplasman güçleri arasında büyük fark varsa, 1/2 veya 2/1 olasılıkları azaltır
            if abs(team_power_diff) > 0.4:
                reversal_boost = 0.8  # Takımlar arası güç farkı büyükse, skor tersine dönüş olasılığı azalır
            
            # Takım stili "comeback" ise, ikinci yarı toparlanma faktörünü artır
            if home_style.get("style_type") == "comeback" or away_style.get("style_type") == "comeback":
                comeback_factor *= 1.3
                
            # Takım stili "strong_start" ise, 1/1 ve 1/X olasılıklarını artır
            if home_style.get("style_type") == "strong_start":
                special_factors["htft_1_2_boost"] = 0.8  # 1/2 olasılığını azalt
                
            # Takım stili "defensive" ise, X/X olasılığını artır
            if home_style.get("style_type") == "defensive" or away_style.get("style_type") == "defensive":
                special_factors["goal_diff_variability"] = 0.8  # Gol değişkenliğini azalt
            
            logger.info("Monte Carlo için takım-spesifik faktörler: ev=%s, deplasman=%s, comeback=%s, reversal=%s", 
                      home_factor, away_factor, comeback_factor, reversal_boost)
        except Exception as e:
            logger.warning(f"Monte Carlo'da takım ayarlamaları uygulanırken hata: {str(e)}")
    
    # Takımların ilk ve ikinci yarı gol beklentileri
    home_first_exp = home_first["avg_goals_per_match"] * home_factor
    home_second_exp = home_second["avg_goals_per_match"] * home_factor
    away_first_exp = away_first["avg_goals_per_match"] * away_factor
    away_second_exp = away_second["avg_goals_per_match"] * away_factor
    
    # Özel faktörleri uygula - 1/2 ve 2/1 senaryoları için
    if special_factors["htft_1_2_boost"] != 1.0:
        # 1/2 faktörü - Ev sahibi ilk yarı iyi, ikinci yarı kötü, deplasman tam tersi
        home_second_exp *= special_factors["htft_1_2_boost"]
        away_second_exp /= special_factors["htft_1_2_boost"]
        
    if special_factors["htft_2_1_boost"] != 1.0:
        # 2/1 faktörü - Deplasman ilk yarı iyi, ikinci yarı kötü, ev sahibi tam tersi
        away_second_exp *= special_factors["htft_2_1_boost"]
        home_second_exp /= special_factors["htft_2_1_boost"]
        
    # Bayes güncellemesi için ek düzeltme faktörleri
    # Daha az maç olan takımlarda tahminleri düzeltmek için
    conf_factor_home = 1
    conf_factor_away = 1
    
    # Bu değerleri Bayesci yöntemle biraz düzeltelim
    # Verinin azlığından dolayı aşırı uç değerler tehlikeli olabilir
    base_home_first = 1.0  # Prior for home team first half
    base_away_first = 0.8  # Prior for away team first half
    base_home_second = 1.2  # Prior for home team second half
    base_away_second = 1.0  # Prior for away team second half
    
    # Daha gerçekçi simülasyon için son maçları inceleyelim
    # Bayesci yaklaşım - az veri varsa prior'a daha çok güven
    home_first_exp_adjusted = (home_first_exp * conf_factor_home + base_home_first) / (conf_factor_home + 1)
    home_second_exp_adjusted = (home_second_exp * conf_factor_home + base_home_second) / (conf_factor_home + 1)
    away_first_exp_adjusted = (away_first_exp * conf_factor_away + base_away_first) / (conf_factor_away + 1)
    away_second_exp_adjusted = (away_second_exp * conf_factor_away + base_away_second) / (conf_factor_away + 1)
    
    # Toplam maç sayısı
    match_count_home = home_first.get("total_matches", 10)
    match_count_away = away_first.get("total_matches", 10)
    
    # Log olarak hesaplanan beklentileri yazdır
    logger.info("Monte Carlo - Bayesian düzeltilmiş beklentiler: %s", {
        "ev": {
            "ilk_yari": home_first_exp_adjusted,
            "ikinci_yari": home_second_exp_adjusted,
            "ham_ilk_yari": home_first_exp,
            "ham_ikinci_yari": home_second_exp,
            "maç_sayısı": match_count_home,
            "güven_faktörü": conf_factor_home
        },
        "deplasman": {
            "ilk_yari": away_first_exp_adjusted,
            "ikinci_yari": away_second_exp_adjusted,
            "ham_ilk_yari": away_first_exp,
            "ham_ikinci_yari": away_second_exp,
            "maç_sayısı": match_count_away,
            "güven_faktörü": conf_factor_away
        }
    })
    
    # Sonuçlar için counter
    results = defaultdict(int)
    
    # Monte Carlo simülasyonu
    for _ in range(simulation_count):
        # Düşük skorlu maç için özel durum - GELİŞTİRİLMİŞ VERSİYON
        total_expected_goals = home_first_exp_adjusted + away_first_exp_adjusted + home_second_exp_adjusted + away_second_exp_adjusted
        total_first_half_goals = home_first_exp_adjusted + away_first_exp_adjusted
        
        # İlk yarı 0-0 olasılığını kademeli olarak artır (düşük skorlu maçlar için)
        force_first_half_draw = False
        
        # Kademeli yaklaşım - gol beklentisine göre farklı olasılıklar
        if total_expected_goals < 1.5 and total_first_half_goals < 0.7:
            # Çok düşük skorlu maç - %70 olasılıkla ilk yarı 0-0
            if random.random() < 0.7:
                home_first_goals = 0
                away_first_goals = 0
                force_first_half_draw = True
                logger.debug("Çok düşük skorlu maç - İlk yarı 0-0 zorlandı (%70 olasılık)")
            else:
                # İlk yarı simülasyonu - beklenti %80 düşürüldü
                home_first_goals = poisson_random(home_first_exp_adjusted * 0.2)
                away_first_goals = poisson_random(away_first_exp_adjusted * 0.2)
        elif total_expected_goals < 2.0 and total_first_half_goals < 1.0:
            # Düşük skorlu maç - %60 olasılıkla ilk yarı 0-0
            if random.random() < 0.6:
                home_first_goals = 0
                away_first_goals = 0
                force_first_half_draw = True
                logger.debug("Düşük skorlu maç - İlk yarı 0-0 zorlandı (%60 olasılık)")
            else:
                # İlk yarı simülasyonu - beklenti %70 düşürüldü
                home_first_goals = poisson_random(home_first_exp_adjusted * 0.3)
                away_first_goals = poisson_random(away_first_exp_adjusted * 0.3)
        elif total_expected_goals < 2.5 and total_first_half_goals < 1.2:
            # Normal-düşük skorlu maç - %50 olasılıkla ilk yarı 0-0
            if random.random() < 0.5:
                home_first_goals = 0
                away_first_goals = 0
                force_first_half_draw = True
                logger.debug("Normal-düşük skorlu maç - İlk yarı 0-0 zorlandı (%50 olasılık)")
            else:
                # İlk yarı simülasyonu - beklenti %60 düşürüldü
                home_first_goals = poisson_random(home_first_exp_adjusted * 0.4)
                away_first_goals = poisson_random(away_first_exp_adjusted * 0.4)
        else:
            # Normal veya yüksek skorlu maç - normal simülasyon, 
            # ancak ilk yarıda yine de az gol beklendiği için küçük bir azaltma
            home_first_goals = poisson_random(home_first_exp_adjusted * 0.9)
            away_first_goals = poisson_random(away_first_exp_adjusted * 0.9)
        
        # İkinci yarı simülasyonu - SÜRPRIZ BUTONU ÖZEL FAKTÖRÜ - GELİŞTİRİLMİŞ VERSİYON
        # Daha fazla dramatik değişim için olasılık uygula
        first_half_dif = home_first_goals - away_first_goals
        
        # Sürpriz seviyesi - toplam beklenen gol sayısına göre
        # Düşük skorlu maçlarda sürprizler daha yaygın olduğu için daha agresif
        surprise_factor = 1.0
        if total_expected_goals < 2.0:
            surprise_factor = 1.5  # %50 daha fazla sürpriz
        elif total_expected_goals < 2.5:
            surprise_factor = 1.3  # %30 daha fazla sürpriz
            
        # İlk yarı yarı öne geçen takım için comeback senaryoları - daha yüksek olasılıklarla
        # Eğer ilk yarı farkı varsa, ikinci yarıda comeback olasılığını artır
        if first_half_dif >= 1:  # 2'den 1'e düşürüldü - daha sık sürpriz
            # Ev sahibi önde - deplasman ikinci yarıda toparlanabilir
            # Her fark seviyesi için farklı olasılıklar
            comeback_chance = 0.35  # Baz olasılık
            
            if first_half_dif >= 2:  # Çok fark varsa daha yüksek comeback şansı 
                comeback_chance = 0.50
            
            # Düşük skorlu maçlar için ek comeback şansı
            comeback_chance *= surprise_factor
            
            if random.random() < comeback_chance:  # Dinamik comeback olasılığı
                # Ev sahibi önde ve fark var - deplasman ikinci yarıda toparlanabilir
                away_second_goals = poisson_random(away_second_exp_adjusted * 2.0)  # %100 artış (%80'den %100'e)
                home_second_goals = poisson_random(home_second_exp_adjusted * 0.5)  # %50 azalış (%30'dan %50'ye)
                logger.debug(f"COMEBACK SENARYOSU UYGULANDI: İlk yarı farkı {first_half_dif}, deplasman ikinci yarıda toparlanıyor")
            else:
                # Normal simülasyon - hafif düzeltme
                home_second_goals = poisson_random(home_second_exp_adjusted * 0.9)  # %10 azalış
                away_second_goals = poisson_random(away_second_exp_adjusted * 1.1)  # %10 artış
                
        elif first_half_dif <= -1:  # -2'den -1'e değiştirildi - daha sık sürpriz
            # Deplasman önde - ev sahibi ikinci yarıda toparlanabilir
            comeback_chance = 0.35  # Baz olasılık
            
            if first_half_dif <= -2:  # Çok fark varsa daha yüksek comeback şansı
                comeback_chance = 0.50
                
            # Düşük skorlu maçlar için ek comeback şansı
            comeback_chance *= surprise_factor
            
            if random.random() < comeback_chance:  # Dinamik comeback olasılığı
                # Deplasman önde ve fark var - ev sahibi ikinci yarıda toparlanabilir
                home_second_goals = poisson_random(home_second_exp_adjusted * 2.0)  # %100 artış
                away_second_goals = poisson_random(away_second_exp_adjusted * 0.5)  # %50 azalış
                logger.debug(f"COMEBACK SENARYOSU UYGULANDI: İlk yarı farkı {first_half_dif}, ev sahibi ikinci yarıda toparlanıyor")
            else:
                # Normal simülasyon - hafif düzeltme
                home_second_goals = poisson_random(home_second_exp_adjusted * 1.1)  # %10 artış
                away_second_goals = poisson_random(away_second_exp_adjusted * 0.9)  # %10 azalış
            
        # İlk yarı berabere ise, ikinci yarıda genelde bir takım üstünlük kurabilir
        elif first_half_dif == 0:
            # Beraberlikten sonra değişim olasılığı - düşük skorlu maçlarda daha yüksek
            draw_break_chance = 0.70  # %70 olasılıkla beraberlik bozulur (%60'tan %70'e)
            draw_break_chance *= surprise_factor  # Düşük skorlu maçlarda daha da yüksek
            
            if random.random() < draw_break_chance:
                if random.random() < 0.5:  # Ev veya deplasman rastgele seçiliyor
                    # Ev sahibi üstünlük kuruyor
                    home_second_goals = poisson_random(home_second_exp_adjusted * 1.8)  # %80 artış (%50'den %80'e)
                    away_second_goals = poisson_random(away_second_exp_adjusted * 0.6)  # %40 azalış (yeni)
                    logger.debug("BERABERLİK BOZULUYOR: Ev sahibi ikinci yarıda üstünlük kuruyor")
                else:
                    # Deplasman üstünlük kuruyor
                    away_second_goals = poisson_random(away_second_exp_adjusted * 1.8)  # %80 artış (%50'den %80'e)
                    home_second_goals = poisson_random(home_second_exp_adjusted * 0.6)  # %40 azalış (yeni)
                    logger.debug("BERABERLİK BOZULUYOR: Deplasman ikinci yarıda üstünlük kuruyor")
            else:
                # Beraberlik devam ediyor - daha düşük skorlu ikinci yarı
                home_second_goals = poisson_random(home_second_exp_adjusted * 0.7)  # %30 azalış
                away_second_goals = poisson_random(away_second_exp_adjusted * 0.7)  # %30 azalış
                logger.debug("BERABERLİK DEVAM EDİYOR: İkinci yarıda az gol bekleniyor")
        else:
            # Normal simülasyon
            home_second_goals = poisson_random(home_second_exp_adjusted)
            away_second_goals = poisson_random(away_second_exp_adjusted)
        
        # Toplam goller
        home_total = home_first_goals + home_second_goals
        away_total = away_first_goals + away_second_goals
        
        # İlk yarı sonucu
        if home_first_goals > away_first_goals:
            first_half = '1'
        elif home_first_goals < away_first_goals:
            first_half = '2'
        else:
            first_half = 'X'
        
        # Maç sonu sonucu
        if home_total > away_total:
            full_time = '1'
        elif home_total < away_total:
            full_time = '2'
        else:
            full_time = 'X'
        
        # İY/MS kombinasyonu
        htft = f"{first_half}/{full_time}"
        results[htft] += 1
    
    # Sonuçları olasılığa dönüştür - yüzdelik
    probabilities = {}
    for htft in HT_FT_COMBINATIONS:
        probabilities[htft] = round((results[htft] / simulation_count) * 100)
    
    return probabilities

def poisson_random(lambda_val):
    """Poisson dağılımından rastgele değer üret"""
    # scipy.stats.poisson kullanarak daha doğru ve verimli örnekleme
    try:
        # Scipy Poisson dağılımını kullan
        return poisson.rvs(lambda_val)
    except:
        # Hata durumunda klasik implementasyona geri dön
        L = math.exp(-lambda_val)
        k = 0
        p = 1.0
        
        while p > L:
            k += 1
            p *= random.random()
        
        return k - 1

def implement_gol_distribution_algorithm(home_stats, away_stats, expected_home_goals, expected_away_goals, is_surprise_button=False, team_adjustments=None):
    """
    GELİŞTİRİLMİŞ ALGORİTMA: Beklenen golleri ilk yarı/ikinci yarı dağılımlarına göre bölerek İY/MS olasılıkları hesaplar.
    Takımların ilk yarı sonuçlarını (önde/berabere/geride) ve İY/MS kombinasyonlarını da analize dahil eder.
    
    Args:
        home_stats: Ev sahibi takım yarı istatistikleri
        away_stats: Deplasman takımı yarı istatistikleri
        expected_home_goals: Beklenen ev sahibi golü (tahmin butonundan)
        expected_away_goals: Beklenen deplasman golü (tahmin butonundan)
        is_surprise_button: Sürpriz butonu için mi
        team_adjustments: Takım-spesifik ayarlamalar
        
    Returns:
        İY/MS olasılıkları sözlüğü
    """
    # Geliştirilmiş model: Poisson ve Skellam dağılımları kullanarak daha doğru ilk/ikinci yarı dağılımları
    logger = logging.getLogger(__name__)
    
    # İlk yarı sonuç verilerini (önde/berabere/geride) analiz et
    home_ht_results = None
    away_ht_results = None
    
    if home_stats and 'ht_results' in home_stats:
        home_ht_results = home_stats.get('ht_results', {})
        away_ht_results = away_stats.get('ht_results', {})
        
        # Takımların ilk yarı performanslarını analiz et
        home_iy_performansi = analyze_ht_results(home_ht_results, True)  # True = ev sahibi
        away_iy_performansi = analyze_ht_results(away_ht_results, False)  # False = deplasman
        
        logger.info(f"İY/MS Dağılım Algoritması - Ev sahibi ilk yarı performansı: {home_iy_performansi}")
        logger.info(f"İY/MS Dağılım Algoritması - Deplasman ilk yarı performansı: {away_iy_performansi}")
        
        # Takımların ilk yarı performans eğilimlerini değerlendir
        # Bu veriler model ağırlıklarını ayarlamak için kullanılabilir
        home_trend = home_iy_performansi.get('trend', 'steady')
        away_trend = away_iy_performansi.get('trend', 'steady')
        
        # Ev sahibi takım ilk yarıları genellikle önde bitiriyorsa, 1/* olasılıklarını artır
        home_ht_win_rate = home_iy_performansi.get('ht_win_rate', 0.0)
        away_ht_win_rate = away_iy_performansi.get('ht_win_rate', 0.0)
        
        # Bu faktörler daha sonra her bir olasılık için çarpan olarak kullanılacak
        first_half_1_factor = 1.0
        first_half_x_factor = 1.0
        first_half_2_factor = 1.0
        
        # Ev sahibi ilk yarıyı %50'den fazla önde bitiriyorsa
        if home_trend == "strong_first_half":
            logger.info("Ev sahibi ilk yarıları genellikle önde bitiriyor - 1/* olasılıkları artırıldı")
            first_half_1_factor = 1.3  # %30 artış
        # Deplasman ilk yarıyı %50'den fazla önde bitiriyorsa
        elif away_trend == "strong_first_half":
            logger.info("Deplasman ilk yarıları genellikle önde bitiriyor - 2/* olasılıkları artırıldı")
            first_half_2_factor = 1.3  # %30 artış
        # İki takım da ilk yarıları genellikle berabere bitiriyorsa
        elif home_trend == "balanced_first_half" and away_trend == "balanced_first_half":
            logger.info("Her iki takım da ilk yarıları genellikle berabere bitiriyor - X/* olasılıkları artırıldı")
            first_half_x_factor = 1.3  # %30 artış
    
    # İstatistikleri çıkar
    home_first = home_stats["statistics"]["first_half"]
    home_second = home_stats["statistics"]["second_half"]
    away_first = away_stats["statistics"]["first_half"]
    away_second = away_stats["statistics"]["second_half"]
    
    # 1. Takımların ilk yarı ve ikinci yarı gol oranlarını hesapla
    # Ev sahibi için
    home_first_half_ratio = 0.45  # Varsayılan: %45 ilk yarı, %55 ikinci yarı
    if home_first["total_goals"] > 0 and (home_first["total_goals"] + home_second["total_goals"]) > 0:
        home_first_half_ratio = home_first["total_goals"] / (home_first["total_goals"] + home_second["total_goals"])
    
    # Deplasman için 
    away_first_half_ratio = 0.45  # Varsayılan: %45 ilk yarı, %55 ikinci yarı
    if away_first["total_goals"] > 0 and (away_first["total_goals"] + away_second["total_goals"]) > 0:
        away_first_half_ratio = away_first["total_goals"] / (away_first["total_goals"] + away_second["total_goals"])
    
    logger.info(f"GOL DAĞILIM ORANLARI - EV: {home_first_half_ratio:.2f} ilk yarı / {1-home_first_half_ratio:.2f} ikinci yarı, " + 
               f"DEPLASMAN: {away_first_half_ratio:.2f} ilk yarı / {1-away_first_half_ratio:.2f} ikinci yarı")
    
    # 2. Beklenen golleri ilk ve ikinci yarıya dağıt
    # YENİ: Düşük gol beklentilerinde ilk yarı daha düşük olur
    # Düşük gol beklentisi olan maçlarda, ilk yarının 0-0 olma olasılığı artar
    
    # Toplam beklenen gol değerlerini kontrol et
    total_expected_goals = expected_home_goals + expected_away_goals
    
    # Maç kategorisi (düşük/orta/yüksek skorlu)
    match_category = "normal"  # Varsayılan kategori
    if total_expected_goals < 2.0:
        match_category = "low"  # Düşük skorlu maç
    elif total_expected_goals <= 3.5:
        match_category = "medium"  # Orta skorlu maç
    else:
        match_category = "high"  # Yüksek skorlu maç
    
    # Kategori bazlı beraberlik faktörlerini tanımla
    draw_boost_factors = {
        "low": 1.3,      # Düşük skorlu maçlarda beraberlik olasılığını artır
        "medium": 1.0,   # Orta skorlu maçlarda normal
        "high": 0.8      # Yüksek skorlu maçlarda beraberlik olasılığını azalt
    }
    
    # Kategori bazlı draw boost faktörünü al
    category_draw_boost = draw_boost_factors.get(match_category, 1.0)
    logger.info(f"Maç kategorisi: {match_category.upper()}, beraberlik çarpanı: {category_draw_boost}")
    
    # Düşük ve orta gol beklentisi için daha dinamik düzeltme sistemi - Kriterleri daha agresif hale getirdik
    low_scoring_match = match_category == "low"
    very_low_scoring_match = total_expected_goals < 1.2  # Eşik değeri 1.0'dan 1.2'ye yükseltildi
    balanced_match = abs(expected_home_goals - expected_away_goals) < 0.4  # Dengeli maç kriteri gevşetildi
    
    # Deplasman avantajlı düşük skorlu maç? (Osasuna-Getafe örneği)
    deplasman_avantajli = expected_away_goals > expected_home_goals + 0.3
    deplasman_cok_avantajli = expected_away_goals > expected_home_goals + 0.5
    
    # İkinci yarı performans çarpanları (varsayılan)
    away_second_strength = 1.0
    home_second_strength = 1.0
    
    # İlk yarı beraberlik faktörü - başlangıçta maç kategorisine göre ayarla
    first_half_draw_boost = category_draw_boost
    
    # Regresyon bazlı dinamik ilk yarı beraberlik faktörü
    # Düşük toplam gol beklentisine göre ilk yarı 0-0 olasılığını artır
    if low_scoring_match:
        # Polinomiyal regresyon: y = -1.0x² + 0.5x + 1.2 (daha agresif formül)
        first_half_draw_boost = max(1.0, -1.0 * (total_expected_goals ** 2) + 0.5 * total_expected_goals + 1.2)
        logger.info(f"DÜŞÜK GOL BEKLENTİLİ MAÇ ALGALANDI: Toplam {total_expected_goals:.2f} gol, İY beraberlik faktörü: {first_half_draw_boost:.2f}")
        
        # Çok düşük skorlu maçlarda daha agresif ayarlama
        if very_low_scoring_match:
            first_half_draw_boost *= 2.0  # %100 daha fazla artış (1.5 → 2.0)
            
        # Dengeli maçlarda daha da yüksek beraberlik olasılığı
        if balanced_match:
            first_half_draw_boost *= 1.3  # %30 daha fazla artış
            logger.info(f"DENGELİ MAÇ ALGALANDI: Gol farkı < 0.3, İY beraberlik faktörü artırıldı: {first_half_draw_boost:.2f}")
        
        # Düşük skorlu maçlarda, ilk yarıların daha az gollü olma eğilimi var
        # İlk yarı gol oranlarını ÇOK DAHA FAZLA azalt - ULTRA azaltma oranları
        home_first_half_ratio *= 0.3  # %70 daha az (eski: %60 daha az)
        away_first_half_ratio *= 0.3  # %70 daha az (eski: %60 daha az)
        
        # Deplasman avantajlı düşük skorlu maçsa, X/2 olasılığını artırmak için daha fazla ayarlama yap
        if deplasman_avantajli:
            logger.info(f"DEPLASMAN AVANTAJLI DÜŞÜK SKORLU MAÇ: Ev={expected_home_goals:.2f}, Deplasman={expected_away_goals:.2f}")
            # Deplasman ikinci yarıda daha güçlü, ilk yarıda beraberlik olasılığı yüksek
            away_first_half_ratio *= 0.7  # Deplasman ilk yarıda daha da düşük gol atar
            home_first_half_ratio *= 0.7  # Ev sahibi de ilk yarıda düşük gol atar
            
            # Deplasman ikinci yarıda daha etkili olacak
            away_second_strength = 1.4  # Deplasman ikinci yarıda daha güçlü
            
            # Deplasman çok avantajlı ise ikinci yarı performansını daha da artır
            if deplasman_cok_avantajli:
                logger.info(f"DEPLASMAN ÇOK AVANTAJLI DÜŞÜK SKORLU MAÇ: Ev={expected_home_goals:.2f}, Deplasman={expected_away_goals:.2f}")
                away_second_strength = 1.6  # Deplasman ikinci yarıda daha da güçlü
                # X/2 olasılığını maksimize etmek için
                if is_surprise_button:
                    away_second_strength = 1.8  # Sürpriz butonunda X/2 olasılığını artır
        
        logger.info(f"DÜŞÜK GOL DÜZELTME SONRASI - EV: {home_first_half_ratio:.2f} ilk yarı, DEPLASMAN: {away_first_half_ratio:.2f} ilk yarı")
    
    # Ev sahibi için
    home_first_expected = expected_home_goals * home_first_half_ratio
    home_second_expected = expected_home_goals * (1 - home_first_half_ratio)
    
    # Deplasman için
    away_first_expected = expected_away_goals * away_first_half_ratio
    away_second_expected = expected_away_goals * (1 - away_first_half_ratio)
    
    logger.info(f"YARI BAZLI BEKLENEN GOLLER - EV: {home_first_expected:.2f} ilk yarı / {home_second_expected:.2f} ikinci yarı, " + 
               f"DEPLASMAN: {away_first_expected:.2f} ilk yarı / {away_second_expected:.2f} ikinci yarı")
    
    # Takım stillerine göre gol beklentilerini ayarla
    momentum_factor_home = 1.0
    momentum_factor_away = 1.0
    
    if team_adjustments:
        if "home_team_style" in team_adjustments:
            home_style = team_adjustments["home_team_style"]
            if "first_half_boost" in home_style:
                home_first_expected *= home_style["first_half_boost"]
            if "second_half_boost" in home_style:
                home_second_expected *= home_style["second_half_boost"]
            
        if "away_team_style" in team_adjustments:
            away_style = team_adjustments["away_team_style"]
            if "first_half_boost" in away_style:
                away_first_expected *= away_style["first_half_boost"]
            if "second_half_boost" in away_style:
                away_second_expected *= away_style["second_half_boost"]
                
        # Momentum faktörleri için takım güç farkını kullan
        if "power_difference" in team_adjustments:
            power_diff = team_adjustments["power_difference"]
            # Pozitif değer ev sahibi lehine, negatif deplasman lehine
            if power_diff > 0.2:  # Ev sahibi avantajlı
                momentum_factor_home = 1.1
                momentum_factor_away = 0.9
            elif power_diff < -0.2:  # Deplasman avantajlı
                momentum_factor_home = 0.9
                momentum_factor_away = 1.1
    
    # 3. Monte Carlo simülasyonu 
    # a) İlk yarı simülasyonu (10,000 simülasyon) - İlk yarı beraberlik boost faktörü uygulanmış
    first_half_results = {"1": 0, "X": 0, "2": 0}
    
    # Dinamik beraberlik faktörünü uygulama yöntemi seçimi
    # 1. Eğer 0-0 skor olasılığı artırılacaksa, Poisson çıktılarını doğrudan değiştirme
    # 2. Alternatif olarak, ilk yarı beraberlik sonuçlarını ağırlıklandırma
    
    # İlk yarı beraberlik boost faktörünü logla
    logger.info(f"İlk yarı beraberlik boost faktörü: {first_half_draw_boost:.2f}")
    
    for _ in range(10000):
        # İlk yarı golleri için Poisson dağılımı
        home_goals_first = poisson_random(home_first_expected)
        away_goals_first = poisson_random(away_first_expected)
        
        # Düşük skorlu maçta ve boost faktörü > 1.0 ise boost uygula
        if first_half_draw_boost > 1.0 and low_scoring_match and random.random() < (first_half_draw_boost - 1.0) / first_half_draw_boost:
            # Beraberlik olasılığını artırmak için zorla 0-0 ata (boost faktörüne bağlı olasılıkla)
            home_goals_first = 0
            away_goals_first = 0
            logger.debug("Düşük skorlu maç için 0-0 ilk yarı skoru zorlandı")
        
        if home_goals_first > away_goals_first:
            first_half_results["1"] += 1
        elif home_goals_first == away_goals_first:
            # Beraberlik sonuçlarını boost faktörüyle ağırlıklandır - daha agresif artış
            boost_weight = int(first_half_draw_boost * 1.5) if first_half_draw_boost > 1.0 else 1
            first_half_results["X"] += boost_weight  # %50 daha fazla ağırlık
        else:
            first_half_results["2"] += 1
    
    # b) İkinci yarı simülasyonu (ilk yarı sonucuna bağlı, 3x10,000 simülasyon)
    second_half_conditional = {
        "1": {"1": 0, "X": 0, "2": 0},  # İlk yarı 1 iken ikinci yarı sonuçları
        "X": {"1": 0, "X": 0, "2": 0},  # İlk yarı X iken ikinci yarı sonuçları
        "2": {"1": 0, "X": 0, "2": 0}   # İlk yarı 2 iken ikinci yarı sonuçları
    }
    
    for first_result in ["1", "X", "2"]:
        # İlk yarı sonucuna göre ikinci yarı için gol beklentilerini ayarla
        adjusted_home_second = home_second_expected
        adjusted_away_second = away_second_expected
        
        # Psikolojik faktörler - ilk yarı sonucuna göre ikinci yarı performansı değişir
        if first_result == "1":  # Ev sahibi önde
            # Ev sahibi daha defansif, deplasman daha ofansif
            adjusted_home_second *= 0.9 * momentum_factor_home
            adjusted_away_second *= 1.2 * momentum_factor_away
        elif first_result == "2":  # Deplasman önde
            # Ev sahibi daha ofansif, deplasman daha defansif
            adjusted_home_second *= 1.2 * momentum_factor_home
            adjusted_away_second *= 0.9 * momentum_factor_away
        
        for _ in range(10000):
            # İkinci yarı golleri
            home_goals_second = poisson_random(adjusted_home_second)
            away_goals_second = poisson_random(adjusted_away_second)
            
            if home_goals_second > away_goals_second:
                second_half_conditional[first_result]["1"] += 1
            elif home_goals_second == away_goals_second:
                second_half_conditional[first_result]["X"] += 1
            else:
                second_half_conditional[first_result]["2"] += 1
    
    # 4. İY/MS olasılıklarını hesapla
    htft_probabilities = {}
    
    # İlk yarı sonuç analizini kullanmak için ilk yarı faktörleri uygula
    if 'home_ht_results' in locals() and home_ht_results and 'away_ht_results' in locals() and away_ht_results:
        logger.info(f"İY/MS olasılıkları hesaplanırken ilk yarı faktörleri uygulanıyor: 1={first_half_1_factor:.2f}, X={first_half_x_factor:.2f}, 2={first_half_2_factor:.2f}")
        
        # İlk yarı faktörlerini uygula
        first_half_results["1"] = int(first_half_results["1"] * first_half_1_factor)
        first_half_results["X"] = int(first_half_results["X"] * first_half_x_factor)
        first_half_results["2"] = int(first_half_results["2"] * first_half_2_factor)
        
        # Toplam yeniden normalize et
        total = first_half_results["1"] + first_half_results["X"] + first_half_results["2"]
        if total > 0:
            first_half_results["1"] = first_half_results["1"] * 10000 // total
            first_half_results["X"] = first_half_results["X"] * 10000 // total
            first_half_results["2"] = first_half_results["2"] * 10000 // total
            
    # YENİ: İY/MS kombinasyon analizi verilerini kullan
    iyms_adjustments = {}
    
    # İY/MS kombinasyon verisi varsa, belirli İY/MS kombinasyonları için ek ayarlamalar yap
    if home_stats and 'ht_ft_combinations' in home_stats:
        home_ht_ft_combinations = home_stats.get('ht_ft_combinations', {})
        away_ht_ft_combinations = away_stats.get('ht_ft_combinations', {})
        
        # Takımların İY/MS kombinasyon eğilimlerini analiz et
        home_iyms_analizi = analyze_ht_ft_combinations(home_ht_ft_combinations, True)
        away_iyms_analizi = analyze_ht_ft_combinations(away_ht_ft_combinations, False)
        
        logger.info(f"İY/MS Kombinasyon Analizi kullanılıyor - Ev: {home_iyms_analizi['most_common_combination']}, Deplasman: {away_iyms_analizi['most_common_combination']}")
        
        # Her takımın en yaygın İY/MS kombinasyonunu belirle
        home_most_common = home_iyms_analizi['most_common_combination']
        away_most_common = away_iyms_analizi['most_common_combination']
        
        # Tersine dönüş oranlarını al (1/2 veya 2/1 gibi)
        home_switch_rate = home_iyms_analizi['switches']['first_to_second']
        away_switch_rate = away_iyms_analizi['switches']['first_to_second']
        
        # Her takımın en yaygın İY/MS kombinasyonu için boost faktörü
        if home_most_common:
            iyms_adjustments[home_most_common] = iyms_adjustments.get(home_most_common, 1.0) + 0.3
        
        if away_most_common:
            iyms_adjustments[away_most_common] = iyms_adjustments.get(away_most_common, 1.0) + 0.3
        
        # Takımlar yüksek tersine dönüş oranlarına sahipse, bu kombinasyonları güçlendir
        if home_switch_rate > 0.25:  # %25'ten fazla tersine dönüş
            # 1/2 ve 2/1 gibi tersine dönüş kombinasyonlarını güçlendir
            iyms_adjustments["1/2"] = iyms_adjustments.get("1/2", 1.0) + 0.4
            iyms_adjustments["2/1"] = iyms_adjustments.get("2/1", 1.0) + 0.4
            logger.info(f"Ev sahibi yüksek tersine dönüş oranına sahip (%{home_switch_rate*100:.1f}), 1/2 ve 2/1 olasılıkları artırıldı")
            
        if away_switch_rate > 0.25:  # %25'ten fazla tersine dönüş
            # 1/2 ve 2/1 gibi tersine dönüş kombinasyonlarını güçlendir
            iyms_adjustments["1/2"] = iyms_adjustments.get("1/2", 1.0) + 0.4
            iyms_adjustments["2/1"] = iyms_adjustments.get("2/1", 1.0) + 0.4
            logger.info(f"Deplasman yüksek tersine dönüş oranına sahip (%{away_switch_rate*100:.1f}), 1/2 ve 2/1 olasılıkları artırıldı")
    
    # YENİ: İY/MS kombinsayon analizi ayarlamalarını uygula
    for fh in ["1", "X", "2"]:
        first_half_prob = first_half_results[fh] / 10000
        
        for sh in ["1", "X", "2"]:
            second_half_prob = second_half_conditional[fh][sh] / 10000
            htft = f"{fh}/{sh}"
            
            # Olasılıklar: ilk yarı olasılığı * ikinci yarı şartlı olasılığı
            raw_probability = first_half_prob * second_half_prob * 100
            
            # Kategori bazlı beraberlik boost faktörünü uygula
            # Düşük skorlu maçlarda X/X olasılığını artır, yüksek skorlu maçlarda azalt
            if htft == "X/X" and match_category in ["low", "medium", "high"]:
                category_boost = draw_boost_factors.get(match_category, 1.0)
                if category_boost != 1.0:
                    original_probability = raw_probability
                    raw_probability *= category_boost
                    logger.info(f"Maç kategorisine göre X/X olasılığı düzeltmesi: {original_probability:.2f} -> {raw_probability:.2f} (çarpan: {category_boost:.2f})")
            
            # İY/MS kombinasyon ayarlamalarını uygula (eğer varsa)
            if htft in iyms_adjustments:
                adjustment_factor = iyms_adjustments[htft]
                if adjustment_factor != 1.0:
                    logger.info(f"İY/MS kombinasyon ayarlaması uygulanıyor: {htft} için {adjustment_factor:.2f} faktörü")
                    raw_probability *= adjustment_factor
            
            htft_probabilities[htft] = round(raw_probability)
    
    # Her bir İY/MS durumu için yüzdeyi hesapla
    total_probability = sum(htft_probabilities.values())
    if total_probability != 100:
        scale_factor = 100 / total_probability
        for htft in htft_probabilities:
            htft_probabilities[htft] = round(htft_probabilities[htft] * scale_factor)
    
    # 5. Sürpriz butonu için özel ayarlamalar
    if is_surprise_button:
        # En olası tahmini al
        sorted_probs = sorted(htft_probabilities.items(), key=lambda x: x[1], reverse=True)
        most_likely = sorted_probs[0][0]
        most_likely_prob = sorted_probs[0][1]
        
        # a) 1/1 ve 2/2 olasılıklarını ÇOK DAHA FAZLA azalt
        if "1/1" in htft_probabilities:
            htft_probabilities["1/1"] = round(htft_probabilities["1/1"] * 0.2)  # %80 azalt (eski: %70 azalt)
        
        if "2/2" in htft_probabilities:
            htft_probabilities["2/2"] = round(htft_probabilities["2/2"] * 0.15)  # %85 azalt (eski: %70 azalt)
        
        # b) Değişim içeren olasılıkları artır
        # Düşük gol beklentisi varsa (total_expected_goals < 1.2) X/X olasılığını artır
        # Bu değer tahmin butonundan alınan parametrelerle hesaplanıyor
        total_expected_goals = expected_home_goals + expected_away_goals
        low_scoring_match = total_expected_goals < 1.2
        
        # İlk yarı beraberlik olasılığını artır
        if low_scoring_match:
            logger.info(f"Sürpriz butonu düşük gol düzeltmesi uygulanıyor: Toplam beklenen gol {total_expected_goals:.2f}")
            # X/X en yüksek olasılık olmalı - düşük gol beklentisinde en gerçekçi tahmin
            if "X/X" in htft_probabilities:
                htft_probabilities["X/X"] = round(htft_probabilities["X/X"] * 3.0)  # %300 artır
                # Ayrıca X/1 ve X/2 tahminlerini de artır - ilk yarı 0-0 bitip ikinci yarı gol olabilir
                if "X/1" in htft_probabilities:
                    htft_probabilities["X/1"] = round(htft_probabilities["X/1"] * 2.5)
                if "X/2" in htft_probabilities:
                    htft_probabilities["X/2"] = round(htft_probabilities["X/2"] * 2.5)
        
        # ULTRA BOOST faktörleri - değişim içeren tahminleri ÇOK DAHA FAZLA artır
        boost_factors = {
            "1/2": 3.5,  # İlk yarı ev sahibi önde, sonra deplasman önde (2.8 -> 3.5)
            "2/1": 3.5,  # İlk yarı deplasman önde, sonra ev sahibi önde (2.8 -> 3.5)
            "X/1": 3.0,  # İlk yarı berabere, sonra ev sahibi önde (2.2 -> 3.0)
            "X/2": 3.0,  # İlk yarı berabere, sonra deplasman önde (2.2 -> 3.0)
            "1/X": 2.5,  # İlk yarı ev sahibi önde, sonra berabere (2.0 -> 2.5)
            "2/X": 2.5   # İlk yarı deplasman önde, sonra berabere (2.0 -> 2.5)
        }
        
        # Düşük skorlu maçların deplasman avantajlı olması durumunda X/2 olasılığını ÇOK DAHA FAZLA artır
        if low_scoring_match and deplasman_avantajli:
            logger.info("Deplasman avantajlı düşük skorlu maç için X/2 olasılığı ULTRA artırılıyor")
            if "X/2" in htft_probabilities:
                boost_factors["X/2"] = 4.5  # X/2 için ULTRA artış (3.5 -> 4.5)
        
        for htft, boost in boost_factors.items():
            if htft in htft_probabilities:
                htft_probabilities[htft] = round(htft_probabilities[htft] * boost)
        
        # c) Takım güç farkına göre ek ayarlamalar
        if team_adjustments and "power_difference" in team_adjustments:
            power_diff = team_adjustments["power_difference"]
            
            if power_diff > 0.1:  # Ev sahibi daha güçlü
                # 2/1 daha olası (dönüş)
                if "2/1" in htft_probabilities:
                    htft_probabilities["2/1"] = round(htft_probabilities["2/1"] * 1.5)
            elif power_diff < -0.1:  # Deplasman daha güçlü
                # 1/2 daha olası (dönüş)
                if "1/2" in htft_probabilities:
                    htft_probabilities["1/2"] = round(htft_probabilities["1/2"] * 1.5)
        
        # d) Emin olmak için 1/1 ve 2/2 kontrolü
        new_sorted = sorted(htft_probabilities.items(), key=lambda x: x[1], reverse=True)
        new_most_likely = new_sorted[0][0]
        
        if new_most_likely in ["1/1", "2/2"]:
            surprise_candidates = [(k, v) for k, v in htft_probabilities.items() if k not in ["1/1", "2/2"]]
            if surprise_candidates:
                best_surprise = max(surprise_candidates, key=lambda x: x[1])
                # En iyi sürprizi en olası tahminden %20 daha yüksek yap
                htft_probabilities[best_surprise[0]] = round(htft_probabilities[new_most_likely] * 1.2)
                htft_probabilities[new_most_likely] = round(htft_probabilities[new_most_likely] * 0.8)
        
        # e) Toplamı 100 yap
        total = sum(htft_probabilities.values())
        if total != 100:
            scale = 100 / total
            for htft in htft_probabilities:
                htft_probabilities[htft] = round(htft_probabilities[htft] * scale)
                
        logger.info(f"SÜRPRİZ BUTONU YENİ ALGORITMA SONUÇLARI: {htft_probabilities}")
        logger.info(f"Sürpriz butonu: Orijinal en olası {most_likely} (%{most_likely_prob}) iken, " +
                 f"değişiklik sonrası en olası {max(htft_probabilities, key=htft_probabilities.get)} " +
                 f"(%{max(htft_probabilities.values())})")
    
    return htft_probabilities

def predict_with_neural_network(home_first, home_second, away_first, away_second, team_adjustments=None):
    """
    Yapay sinir ağı ile İY/MS tahminleri yap
    Genişletilmiş sinir ağı modeli - form ve motivasyon faktörleri eklendi
    
    Args:
        home_first: Ev sahibi ilk yarı istatistikleri
        home_second: Ev sahibi ikinci yarı istatistikleri
        away_first: Deplasman ilk yarı istatistikleri
        away_second: Deplasman ikinci yarı istatistikleri
        team_adjustments: Takım-spesifik ayarlamalar (team_specific_models.py modülünden)
    
    Returns:
        İY/MS olasılıkları
    """
    # SÜRPRİZ BUTONU İYİLEŞTİRME 3:
    # Yapay sinir ağı modeli, yarı bazlı analizlerde daha ince ayrımlar yapabilir
    # İlk ve ikinci yarı dinamiklerini ayrı ayrı analiz ederek, 1/2 ve 2/1 gibi
    # "tersine çevirme" tahminlerinin olasılığını daha yüksek hesaplayacağız
    
    # Manuel yapay sinir ağı - eğitilmiş modeli taklit eden ağırlıklar ve biaslar
    # Gerçek uygulamada bu değerler eğitim ile elde edilir
    
    # Takım-spesifik ayarlamalar
    home_factor = 1.0
    away_factor = 1.0
    
    if team_adjustments:
        try:
            # Takım stil faktörlerini al
            home_style = team_adjustments.get("home_team_style", {})
            away_style = team_adjustments.get("away_team_style", {})
            
            # Home team factors
            home_factors = home_style.get("factors", {})
            home_goal_factor = home_factors.get("goal_factor", 1.0)
            home_concede_factor = home_factors.get("concede_factor", 1.0)
            
            # Away team factors
            away_factors = away_style.get("factors", {})
            away_goal_factor = away_factors.get("goal_factor", 1.0)
            away_concede_factor = away_factors.get("concede_factor", 1.0)
            
            # Takım güç karşılaştırması (home vs away)
            team_power_diff = team_adjustments.get("power_difference", 0.0)
            
            # Faktörleri ayarla
            home_factor = home_goal_factor
            away_factor = away_goal_factor
            
            # Takım güç farkı yüksekse daha fazla ağırlık ver
            if abs(team_power_diff) > 0.3:
                if team_power_diff > 0:  # Ev sahibi avantajlı
                    home_factor *= (1 + team_power_diff/2)
                    away_factor *= (1 - team_power_diff/3)
                else:  # Deplasman avantajlı
                    away_factor *= (1 - team_power_diff/2)  # power_diff negatif olduğu için eklemek olacak
                    home_factor *= (1 + team_power_diff/3)  # power_diff negatif olduğu için çıkarmak olacak
            
            logger.info("Yapay sinir ağı için takım-spesifik faktörler: ev=%s, deplasman=%s, güç farkı=%s", 
                      home_factor, away_factor, team_power_diff)
        except Exception as e:
            logger.warning(f"Yapay sinir ağında takım ayarlamaları uygulanırken hata: {str(e)}")
    
    # Giriş verileri
    features = [
        home_first["avg_goals_per_match"] * home_factor,
        home_second["avg_goals_per_match"] * home_factor,
        away_first["avg_goals_per_match"] * away_factor,
        away_second["avg_goals_per_match"] * away_factor,
        home_first["total_goals"],
        home_second["total_goals"],
        away_first["total_goals"],
        away_second["total_goals"],
        (home_first["avg_goals_per_match"] * home_factor) - (away_first["avg_goals_per_match"] * away_factor),
        (home_second["avg_goals_per_match"] * home_factor) - (away_second["avg_goals_per_match"] * away_factor)
    ]
    
    # Ağırlıklar - manuel olarak ayarlandı
    # Gerçekte bu ağırlıklar eğitim sürecinde öğrenilir
    weights_layer1 = [
        [0.2, 0.3, -0.1, 0.4, 0.1, -0.2, 0.3, -0.4, 0.5, -0.3],  # Nöron 1
        [0.15, 0.25, -0.15, 0.35, 0.05, -0.25, 0.15, -0.35, 0.45, -0.25],  # Nöron 2
        [-0.1, -0.2, 0.3, -0.15, -0.05, 0.25, -0.3, 0.2, -0.4, 0.35],  # Nöron 3
        [0.3, -0.1, 0.2, -0.25, 0.15, 0.1, -0.2, 0.25, -0.1, 0.2]  # Nöron 4
    ]
    
    # İlk katman bias değerleri
    biases_layer1 = [0.1, 0.05, -0.1, 0.15]
    
    # Çıkış katmanı için ağırlıklar (9 sınıf için)
    weights_output = [
        [0.3, -0.2, 0.1, -0.1],  # 1/1
        [0.1, 0.2, -0.2, 0.3],   # 1/X
        [-0.2, 0.25, 0.15, -0.1], # 1/2
        [0.2, -0.15, -0.1, 0.25], # X/1
        [0.1, 0.15, 0.1, -0.1],  # X/X
        [-0.15, -0.1, 0.25, 0.2], # X/2
        [0.15, -0.2, -0.1, 0.3],  # 2/1
        [0.05, 0.15, 0.15, -0.2], # 2/X
        [-0.3, 0.1, 0.25, -0.1]   # 2/2
    ]
    
    # Çıkış katmanı bias değerleri
    biases_output = [0.05, -0.1, 0.1, 0.15, 0.0, -0.05, 0.2, 0.1, -0.15]
    
    # İlk katman - 4 nöron
    layer1_output = []
    for i in range(len(weights_layer1)):
        # Ağırlıklı toplam
        weighted_sum = biases_layer1[i]
        for j in range(len(features)):
            weighted_sum += features[j] * weights_layer1[i][j]
        
        # ReLU aktivasyon fonksiyonu
        layer1_output.append(relu(weighted_sum))
    
    # Çıkış katmanı - 9 sınıf (İY/MS kombinasyonları)
    output_values = []
    for i in range(len(weights_output)):
        # Ağırlıklı toplam
        weighted_sum = biases_output[i]
        for j in range(len(layer1_output)):
            weighted_sum += layer1_output[j] * weights_output[i][j]
        
        # Lineer çıkış (daha sonra softmax uygulanacak)
        output_values.append(weighted_sum)
    
    # Softmax ile olasılık dağılımına dönüştür
    probabilities = softmax(output_values)
    
    # Yüzdelik değerlere dönüştür
    result = {}
    for i, htft in enumerate(HT_FT_COMBINATIONS):
        result[htft] = round(probabilities[i] * 100)
    
    return result

def combine_model_results(statistical_probs, monte_carlo_probs, neural_net_probs, form_motivation_factors=None, team_adjustments=None, crf_probs=None, dirichlet_probs=None):
    """
    Tüm modellerin sonuçlarını ağırlıklı olarak birleştir ve tutarlılığı arttır
    
    Her modele eşit ağırlık (%20) verilir:
    1. İstatistik temelli model (%20)
    2. Monte Carlo simülasyonu (%20)
    3. Yapay sinir ağı modeli (%20)
    4. CRF modeli (%20)
    5. Dirichlet modeli (%20)
    
    Args:
        statistical_probs: İstatistik temelli olasılıklar
        monte_carlo_probs: Monte Carlo simülasyonu olasılıkları
        neural_net_probs: Yapay sinir ağı olasılıkları
        form_motivation_factors: Form ve motivasyon faktörleri (opsiyonel)
        team_adjustments: Takım-spesifik ayarlamalar (team_specific_models.py modülünden)
        crf_probs: CRF modeli olasılıkları (opsiyonel)
        dirichlet_probs: Dirichlet Süreci Karışım Modeli olasılıkları (opsiyonel)
    
    Returns:
        Birleştirilmiş olasılıklar
        
    Not:
        İY/MS kombinasyonları arasındaki tutarlılığı arttırmak için:
        1. Eğer bir takımın maç kazanma olasılığı yüksekse, o takımın ilk yarıda da önde olma kombinasyonları güçlendirilir
        2. Tersine kombinasyonlar (1/2, 2/1) sadece takımların ikinci yarı performansları ve geri dönüş yetenekleri yüksekse güçlendirilir
        3. Markov zinciri analizi ile geçiş olasılıkları kontrol edilir ve tutarsız kombinasyonlar kısıtlanır
    """
    # SÜRPRİZ BUTONU GELİŞTİRME 5: Gelişmiş ağırlıklı kombinasyon
    # Farklı yaklaşımları eşit ağırlıklandırarak birleştir (%20 her biri)
    # Varsayılan ağırlıklar: Tüm modeller için eşit %20 ağırlık
    # Kullanıcı isteği üzerine güncellendi (26.03.2025)
    
    # Form ve motivasyon faktörlerini kullan (eğer mevcutsa)
    stat_weight_factor = 1.0
    neural_weight_factor = 1.0 
    if form_motivation_factors is not None:
        stat_weight_factor = form_motivation_factors.get("stat_weight_factor", 1.0)
        neural_weight_factor = form_motivation_factors.get("neural_weight_factor", 1.0)
        
        # Log olarak faktörleri göster
        logger.info(f"Form ve motivasyon faktörleri model ağırlıklarını etkiliyor: stat_factor={stat_weight_factor}, neural_factor={neural_weight_factor}")
    
    # İY/MS olasılık matrisindeki şablonları incele
    # 1. Modeller arasında tutarlılık var mı?
    consistent_predictions = []
    all_models = [statistical_probs, monte_carlo_probs, neural_net_probs]
    
    # Her model için en olası üç tahmini al
    top_predictions = []
    for model in all_models:
        sorted_preds = sorted(model.items(), key=lambda x: x[1], reverse=True)
        top_predictions.append([p[0] for p in sorted_preds[:3]])
    
    # Tüm modellerin top-3 tahminleri arasında ortak olanlar
    for pred in HT_FT_COMBINATIONS:
        if all(pred in model_top for model_top in top_predictions):
            consistent_predictions.append(pred)
    
    # SÜRPRİZ BUTONU İYİLEŞTİRME 1: 
    # Önceki sistem çok tutarlı sonuçlar lehine ağırlıklandırma yapıyordu, 
    # şimdi daha çeşitli sonuçları teşvik eden bir yaklaşım uyguluyoruz
    
    # Tutarlı sonuçlar varsa bile, sürpriz butonu olduğu için çeşitliliği teşvik et
    # YENİ AĞIRLIK DAĞILIMI - Kullanıcı isteğine göre güncellendi (26.03.2025)
    # CRF ve Dirichlet modellerini de dahil eden eşit ağırlıklı dağılım
    weight_statistical = 0.20  # İstatistik temelli model - %20 
    weight_monte_carlo = 0.20  # Monte Carlo simülasyonu - %20
    weight_neural_net = 0.20   # Yapay sinir ağı - %20
    weight_crf = 0.20          # CRF modeli - %20
    weight_dirichlet = 0.20    # Dirichlet modeli - %20
    
    # Eğer hiç tutarlı tahmin yoksa ya da nadir sonuçları daha fazla teşvik etmek istiyorsak
    # Monte Carlo'ya daha da fazla ağırlık ver çünkü en çeşitli tahminleri o üretiyor
    if not consistent_predictions or random.random() < 0.80:  # %80 olasılıkla her zaman çeşitlilik teşvik edilir (önceki %70)
        weight_statistical = 0.05  # %15'ten %5'e düşürüldü (sürpriz sonuçlara daha çok yer açmak için)
        weight_monte_carlo = 0.75  # %60'tan %75'e artırıldı (daha çeşitli sonuçlar için) 
        weight_neural_net = 0.20   # %25'ten %20'ye düşürüldü
        
        logger.info("Sürpriz butonu için çeşitlilik modu etkinleştirildi: Monte Carlo ağırlığı artırıldı")
    # Eğer belirli bir örüntü başat ise (tüm modeller aynı en olası sonucu veriyorsa)
    elif len(consistent_predictions) == 1:
        # Bu durumda bütün modeller aynı şeyi söylüyor, bu olasılığı daha da güçlendir
        for htft in statistical_probs:
            if htft in consistent_predictions:
                statistical_probs[htft] = round(statistical_probs[htft] * 1.2)  # %20 artış
        
        for htft in monte_carlo_probs:
            if htft in consistent_predictions:
                monte_carlo_probs[htft] = round(monte_carlo_probs[htft] * 1.15)  # %15 artış
    
    # Eğer ilk yarı gol verileri çok düşükse (0.7'den az) ve iki takım da eşit güçlü ise
    if any("X/" in htft for htft in top_predictions[0][:1]):
        # İlk yarı berabere olasılığı yüksek
        weight_statistical = 0.30
        weight_monte_carlo = 0.40  # Monte Carlo bu durumda daha iyi çalışır
        weight_neural_net = 0.30
    
    # Beklenen gol değeri yüksek maçlarda (toplam 3.5+) yapay sinir ağı daha iyi performans gösterir
    # Bu durumda yapay sinir ağı ağırlığını artır
    high_scoring_match = False
    expected_total_goals = 0
    
    # Monte Carlo olasılıklarından toplam gol beklentisini hesapla
    if hasattr(neural_net_probs, 'expected_total_goals'):
        expected_total_goals = neural_net_probs.expected_total_goals
    
    # Form ve motivasyon faktörlerinden ağırlıkları ayarla
    weight_statistical = weight_statistical * stat_weight_factor
    weight_neural_net = weight_neural_net * neural_weight_factor
    
    # Yüksek skorlu maçlar için yapay sinir ağına daha fazla güven
    if expected_total_goals > 3.5:
        high_scoring_match = True
        weight_statistical = weight_statistical * 0.9  # %10 azalt
        weight_monte_carlo = weight_monte_carlo * 0.95  # %5 azalt
        weight_neural_net = weight_neural_net * 1.15  # %15 artır
    
    # Düşük skorlu maçlar için Monte Carlo'ya daha fazla güven - çok düşük skorlu maçlarda daha agresif
    if expected_total_goals < 1.5:
        weight_monte_carlo = weight_monte_carlo * 1.2  # %20 artır
        # İlk yarı berabere (X) olasılıkları daha yüksek
        for htft in monte_carlo_probs:
            if htft.startswith('X/'):
                monte_carlo_probs[htft] = round(monte_carlo_probs[htft] * 1.3)  # %30 artış
    
    # Çok düşük skorlu maçlar için ekstra iyileştirme (1.2'den az toplam gol beklentisi)
    if expected_total_goals < 1.2:
        logger.info(f"Çok düşük skorlu maç tespit edildi! (Toplam beklenen gol: {expected_total_goals:.2f})")
        weight_monte_carlo = weight_monte_carlo * 1.3  # %30 daha artır
        # İlk yarı berabere (X) olasılıkları çok daha yüksek - X/X özellikle
        for htft in monte_carlo_probs:
            if htft == 'X/X':
                monte_carlo_probs[htft] = round(monte_carlo_probs[htft] * 1.8)  # %80 artış
    
    # Ağırlıkların toplamının 1.0 olmasını sağla
    total_weight = weight_statistical + weight_monte_carlo + weight_neural_net
    if total_weight != 1.0:
        weight_statistical /= total_weight
        weight_monte_carlo /= total_weight
        weight_neural_net /= total_weight
    
    logger.info(f"Model ağırlıkları (son): İstatistiksel={weight_statistical:.2f}, Monte Carlo={weight_monte_carlo:.2f}, Yapay Sinir Ağı={weight_neural_net:.2f} (Tutarlı tahminler: {consistent_predictions}, Yüksek skorlu maç: {high_scoring_match})")
    
    # Ağırlıklı birleştirme (CRF modelini de dahil et)
    combined = {}
    
    # CRF ve Dirichlet modeli mevcutsa, sonuçlara dahil et
    if crf_probs is not None and dirichlet_probs is not None:
        # Modellerin ağırlıklarını belirle - kullanıcı isteği üzerine eşit ağırlık
        weight_crf = 0.20  # %20 ağırlık (eşit dağılım)
        weight_dirichlet = 0.20  # %20 ağırlık (eşit dağılım)
        
        # Artık düşük skorlu maçlarda özel ayarlama yapmayacağız, her durumda tüm modellere eşit ağırlık verilecek
        logger.info(f"Tüm modeller kullanıcı isteği üzerine eşit ağırlıklarla kullanılacak (Her model için %20)")
        
        # Kullanıcı isteği üzerine tüm modellere eşit ağırlık verilecek (26.03.2025)
        weight_statistical = 0.20
        weight_monte_carlo = 0.20
        weight_neural_net = 0.20
        weight_crf = 0.20
        weight_dirichlet = 0.20
        
        logger.info(f"Tüm modeller tahminlere dahil edildi. Ağırlıklar: CRF={weight_crf:.3f}, Dirichlet={weight_dirichlet:.3f}, İstatistik={weight_statistical:.2f}, Monte Carlo={weight_monte_carlo:.2f}, YSA={weight_neural_net:.2f}")
        
        # Tüm modelleri birleştir (CRF ve Dirichlet dahil)
        for htft in HT_FT_COMBINATIONS:
            combined[htft] = round(
                statistical_probs.get(htft, 0) * weight_statistical +
                monte_carlo_probs.get(htft, 0) * weight_monte_carlo +
                neural_net_probs.get(htft, 0) * weight_neural_net +
                crf_probs.get(htft, 0) * weight_crf +
                dirichlet_probs.get(htft, 0) * weight_dirichlet
            )
    elif crf_probs is not None:
        # Sadece CRF modeli mevcutsa
        # Kullanıcı isteği üzerine tüm modellere eşit ağırlık (%25) verilecek
        weight_statistical = 0.25
        weight_monte_carlo = 0.25
        weight_neural_net = 0.25
        weight_crf = 0.25
        
        logger.info(f"Kullanıcı isteği üzerine her modele eşit ağırlık verilecek (%25)")
        
        logger.info(f"CRF modeli tahminlere dahil edildi. Ağırlıklar: CRF={weight_crf:.2f}, İstatistik={weight_statistical:.2f}, Monte Carlo={weight_monte_carlo:.2f}, YSA={weight_neural_net:.2f}")
        
        # Modelleri birleştir (CRF dahil)
        for htft in HT_FT_COMBINATIONS:
            combined[htft] = round(
                statistical_probs.get(htft, 0) * weight_statistical +
                monte_carlo_probs.get(htft, 0) * weight_monte_carlo +
                neural_net_probs.get(htft, 0) * weight_neural_net +
                crf_probs.get(htft, 0) * weight_crf
            )
    elif dirichlet_probs is not None:
        # Sadece Dirichlet modeli mevcutsa
        # Kullanıcı isteği üzerine tüm modellere eşit ağırlık (%25) verilecek
        weight_statistical = 0.25
        weight_monte_carlo = 0.25
        weight_neural_net = 0.25
        weight_dirichlet = 0.25
        
        logger.info(f"Kullanıcı isteği üzerine her modele eşit ağırlık verilecek (%25)")
        
        logger.info(f"Dirichlet modeli tahminlere dahil edildi. Ağırlıklar: Dirichlet={weight_dirichlet:.2f}, İstatistik={weight_statistical:.2f}, Monte Carlo={weight_monte_carlo:.2f}, YSA={weight_neural_net:.2f}")
        
        # Modelleri birleştir (Dirichlet dahil)
        for htft in HT_FT_COMBINATIONS:
            combined[htft] = round(
                statistical_probs.get(htft, 0) * weight_statistical +
                monte_carlo_probs.get(htft, 0) * weight_monte_carlo +
                neural_net_probs.get(htft, 0) * weight_neural_net +
                dirichlet_probs.get(htft, 0) * weight_dirichlet
            )
    else:
        # Hiçbir gelişmiş model yoksa, temel modellere eşit ağırlık ver
        weight_statistical = 0.33
        weight_monte_carlo = 0.33
        weight_neural_net = 0.34  # Tam 100% yapabilmek için
        
        logger.info(f"Gelişmiş modeller kullanılamadığı için temel modellere eşit ağırlık veriliyor: İstatistik={weight_statistical:.2f}, Monte Carlo={weight_monte_carlo:.2f}, YSA={weight_neural_net:.2f}")
        
        # Temel modelleri birleştir
        for htft in HT_FT_COMBINATIONS:
            combined[htft] = round(
                statistical_probs.get(htft, 0) * weight_statistical +
                monte_carlo_probs.get(htft, 0) * weight_monte_carlo +
                neural_net_probs.get(htft, 0) * weight_neural_net
            )
    
    # SÜRPRIZ BUTONU ÖZELLİĞİ - ÖNEMLİ DEĞİŞİKLİK
    # Butonun amacı 1/1 ve 2/2 dışındaki tahminlere öncelik vermek
    
    # 1. En olası tahminleri bul (sıralı olarak)
    sorted_probs = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    most_likely = sorted_probs[0][0]
    most_likely_prob = sorted_probs[0][1]
    
    # SÜRPRİZ BUTONU İYİLEŞTİRME 6: RADIKAL YENİDEN FORMÜLASYON
    # Sürpriz butonu için tamamen yeni bir yaklaşım uyguluyoruz
    # FORMÜL: İlk yarı ve ikinci yarı için ayrı faktörler hesaplayarak h2h, güç analizi ve form etkisini artırıyoruz
    logger.info(f"SÜRPRİZ BUTONU YENİ ALGORİTMA UYGULANACAK - CURRENT PROBS: {combined}")
    
    # Önce mevcut tahminleri tamamen sıfırla - çok radikal bir değişiklik
    for key in combined:
        combined[key] = 1  # Herkese eşit şans ver başlangıçta
    
    # İlk yarı ve maç sonu için faktörleri hesapla
    first_half_factors = {}
    full_time_factors = {}
    
    # 1. İlk Yarı Faktörlerini Hesapla
    # a) Ev sahibi gücü
    home_first_half_power = home_first["avg_goals_per_match"] * 0.3  # %30 ağırlık
    away_first_half_power = away_first["avg_goals_per_match"] * 0.3  # %30 ağırlık
    
    # b) Eğer takım-spesifik ayarlamalar varsa, kullan
    if team_adjustments:
        if "home_team_style" in team_adjustments:
            home_style = team_adjustments["home_team_style"]
            # İlk yarı güçlü takımları belirle
            if home_style.get("style_type") == "strong_start":
                home_first_half_power *= 1.5  # %50 artış
                logger.info(f"Ev sahibi 'strong_start' stiline sahip, ilk yarı gücü artırıldı: {home_first_half_power}")
                
        if "away_team_style" in team_adjustments:
            away_style = team_adjustments["away_team_style"]
            if away_style.get("style_type") == "strong_start":
                away_first_half_power *= 1.5  # %50 artış
                logger.info(f"Deplasman 'strong_start' stiline sahip, ilk yarı gücü artırıldı: {away_first_half_power}")
    
    # c) Takımların güç farkını hesapla
    power_diff_first_half = home_first_half_power - away_first_half_power
    
    # d) İlk yarı faktörlerini hesapla
    # Pozitif güç farkı ev sahibi lehine, negatif deplasman lehine
    if power_diff_first_half > 0.3:  # Ev sahibi ilk yarıda avantajlı
        first_half_factors["1"] = 5.0  # Ev sahibi ilk yarı önde
        first_half_factors["X"] = 2.0  # İlk yarı berabere
        first_half_factors["2"] = 1.0  # Deplasman ilk yarı önde
    elif power_diff_first_half < -0.3:  # Deplasman ilk yarıda avantajlı
        first_half_factors["1"] = 1.0
        first_half_factors["X"] = 2.0
        first_half_factors["2"] = 5.0
    else:  # Takımlar ilk yarı için dengeli
        first_half_factors["1"] = 2.0
        first_half_factors["X"] = 4.0  # Beraberliği teşvik et
        first_half_factors["2"] = 2.0
    
    # 2. Maç Sonu Faktörlerini Hesapla - ikinci yarı güçleriyle de ilgili
    # a) Ev sahibi gücü
    home_second_half_power = home_second["avg_goals_per_match"] * 0.3  # %30 ağırlık 
    away_second_half_power = away_second["avg_goals_per_match"] * 0.3  # %30 ağırlık
    
    # b) Eğer takım-spesifik ayarlamalar varsa, kullan
    if team_adjustments:
        # İkinci yarı güçlü takımları belirle
        if "home_team_style" in team_adjustments:
            home_style = team_adjustments["home_team_style"]
            if home_style.get("style_type") == "comeback":
                home_second_half_power *= 1.5  # %50 artış
                logger.info(f"Ev sahibi 'comeback' stiline sahip, ikinci yarı gücü artırıldı: {home_second_half_power}")
                
        if "away_team_style" in team_adjustments:
            away_style = team_adjustments["away_team_style"]
            if away_style.get("style_type") == "comeback":
                away_second_half_power *= 1.5  # %50 artış
                logger.info(f"Deplasman 'comeback' stiline sahip, ikinci yarı gücü artırıldı: {away_second_half_power}")
    
    # c) Takımların ikinci yarı güç farkını hesapla
    power_diff_second_half = home_second_half_power - away_second_half_power
    
    # d) Maç sonu faktörlerini hesapla
    if power_diff_second_half > 0.3:  # Ev sahibi ikinci yarıda avantajlı
        full_time_factors["1"] = 5.0
        full_time_factors["X"] = 2.0
        full_time_factors["2"] = 1.0
    elif power_diff_second_half < -0.3:  # Deplasman ikinci yarıda avantajlı
        full_time_factors["1"] = 1.0
        full_time_factors["X"] = 2.0
        full_time_factors["2"] = 5.0
    else:  # Takımlar ikinci yarı için dengeli
        full_time_factors["1"] = 2.0
        full_time_factors["X"] = 4.0
        full_time_factors["2"] = 2.0
    
    # 3. Takımların form ve motivasyon faktörlerini kullan (sürpriz sonuçlar için)
    # Farklı gol faktörlerine göre sürpriz faktörlerini belirle
    # Çekişmeli maçların daha değişken olduğunu düşünelim
    h2h_surprise_factor = 1.0
    if team_adjustments and "h2h_analysis" in team_adjustments:
        h2h = team_adjustments["h2h_analysis"]
        if h2h.get("unpredictable", False):
            h2h_surprise_factor = 2.0  # İki takım arasındaki maçlar tahmin edilemez
            logger.info(f"H2H analizi: Takımlar arasında tahmin edilemez sonuçlar var, sürpriz faktörü artırıldı: {h2h_surprise_factor}")
    
    # 4. Kombinasyon faktörlerini hesapla ve uygula
    # Her bir kombinasyon için faktör hesapla
    for first in ["1", "X", "2"]:
        for second in ["1", "X", "2"]:
            htft = f"{first}/{second}"
            # Eğer ilk yarı ve ikinci yarı faktörleri uyumluysa (örn. 1/1, 2/2, X/X) bu faktörü biraz azalt
            combo_factor = first_half_factors[first] * full_time_factors[second]
            if first == second:
                combo_factor *= 0.5  # Aynı sonuçların ağırlığını azalt - sürpriz tahmini teşvik et
            
            # Özellikle 1/2 ve 2/1 gibi tersine dönüşleri teşvik et
            if (first == "1" and second == "2") or (first == "2" and second == "1"):
                combo_factor *= h2h_surprise_factor * 2.0  # Tersine dönüşleri iki kat daha fazla teşvik et
                logger.info(f"Tersine dönüş teşvik edildi: {htft}, yeni faktör: {combo_factor}")
            
            # Combo faktörünü uygula
            combined[htft] = round(combo_factor)
            
    # 5. ÇOK ÖNEMLİ: 1/1 ve 2/2 için özel azaltma - ULTRA AGRESİF regresyon yaklaşımı
    # Beklenen gol seviyelerini kullanarak regresif azaltma faktörü belirle
    # Düşük gol durumlarında daha fazla azaltma, yüksek gol durumlarında daha az azaltma
    total_goals = sum([home_first["avg_goals_per_match"], home_second["avg_goals_per_match"],
                      away_first["avg_goals_per_match"], away_second["avg_goals_per_match"]])
    
    # ULTRA AGRESİF: Tamamen yeni formül: y = -0.04x² + 0.06x + 0.03
    # Burada x toplam beklenen gol, y azaltma faktörü (0.03 ile 0.15 arasında) - DAHA DA DÜŞÜK FAKTÖR
    reduction_factor = min(max(-0.04 * (total_goals ** 2) + 0.06 * total_goals + 0.03, 0.03), 0.15)
    
    logger.info(f"1/1 ve 2/2 için regresif azaltma faktörü (ULTRA AGRESİF): {reduction_factor:.2f} (toplam gol beklentisi: {total_goals:.2f})")
    
    # 1/1 için daha agresif azaltma (özellikle düşük skorlu maçlarda daha nadir)
    combined["1/1"] = round(combined["1/1"] * reduction_factor)  # Dinamik azaltma
    # 2/2 için çok daha agresif azaltma - %40 daha fazla azaltma (0.8 -> 0.6)
    combined["2/2"] = round(combined["2/2"] * (reduction_factor * 0.6))  # Dinamik azaltma, %40 daha fazla azaltma
    
    # X/X (berabere/berabere) için de daha düşük azaltma faktörü uygula, ancak diğerlerinden daha az azaltma
    combined["X/X"] = round(combined["X/X"] * (reduction_factor * 1.8))  # X/X'i daha da az azalt (1.5'ten 1.8'e yükseltildi)
    
    # 1/X, X/1, 1/2, 2/1 gibi değişken sonuçlar için boost uygula
    # Değişken sonuçlar için özel boost - özellikle bunları daha da artır
    combined["1/X"] = round(combined["1/X"] * 1.5)  # %50 artış
    combined["X/1"] = round(combined["X/1"] * 1.5)  # %50 artış
    combined["X/2"] = round(combined["X/2"] * 1.5)  # %50 artış
    combined["2/X"] = round(combined["2/X"] * 1.5)  # %50 artış
    
    logger.info(f"Sürpriz butonu yeni algoritma sonuçları: {combined}")
    
    # Ek boost denetimleri: Eğer hala yeterince sürpriz değilse
    # İlk yarı/ikinci yarı değişimi yüksek olasın
    
    # Özellikle 1/2 ve 2/1 gibi değişimlere ek boost uygula - Lojistik fonksiyon temelli
    def logistic_boost(base_value, boost_factor, total_expected_goals):
        # Lojistik fonksiyon: 1/(1+e^(-k*(x-x0))) formülü
        # Düşük gol beklentisinde daha yüksek boost, yüksek gol beklentisinde normal boost
        k = -0.5  # Eğim faktörü
        x0 = 3.0  # Orta nokta
        logistic_factor = 1 + (boost_factor - 1) * (1 / (1 + math.exp(-k * (total_goals - x0))))
        return round(base_value * logistic_factor)
    
    # Dinamik boost faktörleri - maçın beklenen gol seviyesine göre uyarlanır
    combined["1/2"] = logistic_boost(combined["1/2"], 2.0, total_goals)
    combined["2/1"] = logistic_boost(combined["2/1"], 2.0, total_goals)
    combined["X/1"] = logistic_boost(combined["X/1"], 1.6, total_goals)
    combined["X/2"] = logistic_boost(combined["X/2"], 1.6, total_goals)
    combined["1/X"] = logistic_boost(combined["1/X"], 1.5, total_goals)
    combined["2/X"] = logistic_boost(combined["2/X"], 1.5, total_goals)
    
    # Takım güç faktörlerini kullan (eğer mevcutsa) - ek bir katman olarak
    if team_adjustments and "power_difference" in team_adjustments:
        power_diff = team_adjustments["power_difference"]
        
        # Güç farkına göre hangi tür sürpriz daha olası olabilir
        if power_diff > 0.1:  # Ev sahibi biraz daha güçlü
            # 2/1 daha olası (ilk yarı deplasman, sonra ev toparlanır)
            combined["2/1"] = round(combined["2/1"] * 1.5)
        elif power_diff < -0.1:  # Deplasman biraz daha güçlü  
            # 1/2 daha olası (ilk yarı ev sahibi, sonra deplasman toparlanır)
            combined["1/2"] = round(combined["1/2"] * 1.5)
        else:  # Takımlar eşit güçte
            # X/1 ve X/2 gibi beraberlikten galibiyete durumları artır
            combined["X/1"] = round(combined["X/1"] * 1.5)
            combined["X/2"] = round(combined["X/2"] * 1.5)
    
    # Yeni en olası tahmini kontrol et (artık 1/1 veya 2/2 olmamalı)
    new_most_likely = max(combined, key=combined.get)
    
    # Eğer hala 1/1 veya 2/2 en yüksekse, zorunlu değişiklik yap
    if new_most_likely in ["1/1", "2/2"]:
        # Zorunlu olarak en yüksek sürpriz tahminini en olası yap
        surprise_candidates = {k: v for k, v in combined.items() if k not in ["1/1", "2/2"]}
        if surprise_candidates:
            best_surprise = max(surprise_candidates, key=surprise_candidates.get)
            # Sürpriz tahmini en olası tahminden %10 daha yüksek yap
            combined[best_surprise] = round(combined[new_most_likely] * 1.1)
    
    logger.info(f"Sürpriz butonu: Orijinal en olası tahmin {most_likely} (%{most_likely_prob}) iken, değişiklik sonrası en olası tahmin {max(combined, key=combined.get)} (%{max(combined.values())})")
    
    # Toplamın 100 olduğundan emin ol
    total = sum(combined.values())
    if total != 100:
        scale_factor = 100 / total
        for htft in combined:
            combined[htft] = round(combined[htft] * scale_factor)
    
    return combined