import requests
import json
import os
import logging
import flask
import time
import numpy as np
from datetime import datetime
from flask import Blueprint, jsonify, request, current_app

# Main Flask application (needed for directly accessing routes)
# Using app directly from main can cause circular imports
try:
    from main import app
except ImportError:
    # If imported by main.py, use current_app instead
    from flask import current_app as app

# NumPy değerlerini Python'a dönüştürmek için yardımcı fonksiyon
def numpy_to_python(obj):
    """NumPy değerlerini Python'a dönüştür (JSON serileştirme için)"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    return obj
from datetime import datetime, timedelta
from functools import wraps

# Logging
logger = logging.getLogger(__name__)

# İlk Yarı / Maç Sonu tahmini modülünü içe aktar
import halfTime_fullTime_predictor

# API Keys
FOOTBALL_DATA_API_KEY = os.environ.get('FOOTBALL_DATA_API_KEY', '668dd03e0aea41b58fce760cdf4eddc8')
API_FOOTBALL_KEY = os.environ.get('API_FOOTBALL_KEY', '2f0c06f149e51424f4c9be24eb70cb8f')

# Blueprint definition
api_v3_bp = Blueprint('api_v3', __name__, url_prefix='/api/v3')

# Tahmin önbelleğini temizleme API'si
# API blueprint erişimi için eski versiyonu çıkarıyoruz, çünkü aşağıda yenisi tanımlanmış
# Bu fonksiyon API Blueprint'i üzerinden erişilen eski versiyondur
# Yeni sürüm 995-1031 satırları arasındadır
@api_v3_bp.route('/clear-prediction-cache', methods=['GET'])
def clear_prediction_cache_v3():  # Fonksiyon ismi değiştirildi
    """Tahmin önbelleğini temizler (v3 API Blueprint versiyonu)"""
    try:
        from match_prediction import MatchPredictor
        predictor = MatchPredictor()
        predictor.clear_cache()
        logger.info("Tahmin önbelleği başarıyla temizlendi (v3 API)")
        return jsonify({"success": True, "message": "Tahmin önbelleği başarıyla temizlendi"})
    except Exception as e:
        logger.error(f"Tahmin önbelleği temizlenirken hata: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

# Önbellek kontrolü için yardımcı fonksiyon
def api_cache(timeout=300):
    """
    Flask route için önbellek dekoratörü
    
    Args:
        timeout: Önbellek süresi (saniye), varsayılan: 5 dakika
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Flask cache nesnesi mevcut mu kontrol et
            if hasattr(current_app, 'cache'):
                cache = current_app.cache
                # Önbellek anahtarı oluştur
                cache_key = f.__name__
                # Fonksiyon parametrelerini cache anahtarına ekle
                for arg in args:
                    if isinstance(arg, (str, int, float, bool)):
                        cache_key += f"_{arg}"
                for key, value in kwargs.items():
                    if isinstance(value, (str, int, float, bool)):
                        cache_key += f"_{key}_{value}"
                
                # Önbellekte var mı kontrol et
                cached_result = cache.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {cache_key}")
                    return cached_result
                
                # Yoksa fonksiyonu çalıştır ve sonucu önbelleğe al
                start_time = time.time()
                result = f(*args, **kwargs)
                elapsed_time = time.time() - start_time
                logger.debug(f"Cache miss for {cache_key}, execution took {elapsed_time:.6f} seconds")
                
                # Sonucu cache'e kaydet
                cache.set(cache_key, result, timeout=timeout)
                return result
            else:
                # Cache yok, normal şekilde fonksiyonu çalıştır
                return f(*args, **kwargs)
        return decorated_function
    return decorator

# API Football endpoints
@api_v3_bp.route('/fixtures', methods=['GET'])
def get_fixtures():
    try:
        date = request.args.get('date', datetime.now().strftime("%Y-%m-%d"))
        league = request.args.get('league', '')  # Default to all leagues
        logger.info(f"Fetching fixtures for date: {date}, league: {league if league else 'all leagues'}")
        
        timezone = request.args.get('timezone', 'Europe/Istanbul')
        # Burada kodu tamamlamanız gerekiyor
        return jsonify({"message": "API is under construction"})
    except Exception as e:
        logger.error(f"Error fetching fixtures: {str(e)}")
        return jsonify({"error": str(e)}), 500

def convert_apifootball_to_standard(matches_data):
    """API-Football verilerini standart formata dönüştür"""
    converted_data = {
        "response": []
    }

    for match in matches_data:
        fixture = {
            "fixture": {
                "id": match.get('match_id'),
                "date": match.get('match_date') + 'T' + match.get('match_time') + 'Z',
                "status": {
                    "short": "NS" if match.get('match_status') == '' else match.get('match_status')[:2],
                    "long": match.get('match_status') or 'SCHEDULED'
                },
                "venue": {
                    "name": match.get('match_stadium', '')
                }
            },
            "league": {
                "name": match.get('league_name', ''),
                "logo": match.get('league_logo', '')
            },
            "teams": {
                "home": {
                    "name": match.get('match_hometeam_name', ''),
                    "logo": match.get('team_home_badge', '')
                },
                "away": {
                    "name": match.get('match_awayteam_name', ''),
                    "logo": match.get('team_away_badge', '')
                }
            },
            "goals": {
                "home": match.get('match_hometeam_score', ''),
                "away": match.get('match_awayteam_score', '')
            }
        }
        converted_data["response"].append(fixture)

    return converted_data

# Football-data.org fallback
def get_football_data_fixtures(date):
    try:
        # Önce API-Football ile deneyelim
        url = "https://apiv3.apifootball.com/"
        params = {
            'action': 'get_events',
            'from': date,
            'to': date,
            'APIkey': API_FOOTBALL_KEY
        }

        response = requests.get(url, params=params)
        if response.status_code == 200 and isinstance(response.json(), list) and len(response.json()) > 0:
            # API-Football verisi başarıyla alındı, converter'a gönder
            return convert_apifootball_to_standard(response.json())

        # API-Football çalışmazsa football-data.org ile devam et
        url = "https://api.football-data.org/v4/matches"
        headers = {'X-Auth-Token': FOOTBALL_DATA_API_KEY}
        params = {"date": date}

        response = requests.get(url, headers=headers, params=params)
        data = response.json()

        # Convert football-data.org format to api-football format
        converted_data = {
            "response": []
        }

        if 'matches' in data:
            for match in data['matches']:
                fixture = {
                    "fixture": {
                        "id": match.get('id'),
                        "date": match.get('utcDate'),
                        "status": {
                            "short": "NS" if match.get('status') == 'SCHEDULED' else match.get('status')[:2],
                            "long": match.get('status')
                        },
                        "venue": {
                            "name": match.get('venue')
                        }
                    },
                    "league": {
                        "name": match.get('competition', {}).get('name'),
                        "logo": match.get('competition', {}).get('emblem')
                    },
                    "teams": {
                        "home": {
                            "name": match.get('homeTeam', {}).get('shortName'),
                            "logo": match.get('homeTeam', {}).get('crest')
                        },
                        "away": {
                            "name": match.get('awayTeam', {}).get('shortName'),
                            "logo": match.get('awayTeam', {}).get('crest')
                        }
                    },
                    "goals": {
                        "home": match.get('score', {}).get('fullTime', {}).get('home'),
                        "away": match.get('score', {}).get('fullTime', {}).get('away')
                    }
                }
                converted_data["response"].append(fixture)

        return jsonify(converted_data)
    except Exception as e:
        logger.error(f"Error getting football-data fixtures: {str(e)}")
        return jsonify({"errors": True, "message": str(e)}), 500



@api_v3_bp.route('/team/half-time-stats/<team_id>', methods=['GET'])
@api_cache(timeout=1800)  # 30 dakika önbellekleme (takım istatistikleri sık değişmez)
def get_team_half_time_stats(team_id):
    """Takımın son 21 maçındaki ilk yarı (0-45) ve ikinci yarı (46-90) gol istatistiklerini ve galibiyet/beraberlik/mağlubiyet durumlarını döndür"""
    try:
        # team_id string olarak gelmişse int'e çevirmeyi dene
        if isinstance(team_id, str) and team_id.isdigit():
            team_id = int(team_id)
        
        # Eğer team_id sıfır veya geçersizse hemen uygun cevap dön
        if not team_id or not isinstance(team_id, int) or team_id <= 0:
            team_id_value = team_id if isinstance(team_id, int) else str(team_id)
            logger.warning(f"Invalid team_id={team_id_value} provided")
            return {
                "team_id": str(team_id_value),
                "status": "Veri bulunamadı",
                "message": "Geçersiz takım ID'si",
                "first_half": {"for": 0, "against": 0, "matches": 0},
                "second_half": {"for": 0, "against": 0, "matches": 0},
                "total_matches": 0,
                "matches": [],
                "ht_results": {
                    "home": {"1": 0, "X": 0, "2": 0},  # Ev sahibiyken ilk yarı önde/berabere/geride
                    "away": {"1": 0, "X": 0, "2": 0},  # Deplasmanken ilk yarı önde/berabere/geride
                    "total": {"1": 0, "X": 0, "2": 0}  # Toplam ilk yarı önde/berabere/geride
                },
                "ft_results": {
                    "home": {"1": 0, "X": 0, "2": 0},  # Ev sahibiyken maç sonu galibiyet/beraberlik/mağlubiyet
                    "away": {"1": 0, "X": 0, "2": 0},  # Deplasmanken maç sonu galibiyet/beraberlik/mağlubiyet
                    "total": {"1": 0, "X": 0, "2": 0}  # Toplam maç sonu galibiyet/beraberlik/mağlubiyet
                }
            }
            
        # Doğrudan API'den takımın son maçlarını alalım
        url = "https://apiv3.apifootball.com/"
        last_matches = 21  # Son 21 maçı al
        
        # API anahtarını tanımla
        api_key = '39bc8dd65153ff5c7c0f37b4939009de04f4a70c593ee1aea0b8f70dd33268c0'
                    
        # API'den takım maç verilerini alalım    
        params = {
            'action': 'get_events',
            'team_id': team_id,
            'APIkey': API_FOOTBALL_KEY
        }
        
        logger.info(f"Takımın son {last_matches} maçındaki yarı istatistikleri isteniyor: team_id={team_id}")
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            logger.error(f"API-Football error getting team matches: {response.status_code}")
            # Veritabanı benzeri bir yapı döndürerek analiz kodlarının çalışmasını sağlayalım
            return jsonify({
                "statistics": {
                    "first_half": {
                        "goals_scored": 0,
                        "goals_conceded": 0,
                        "matches": 0
                    },
                    "second_half": {
                        "goals_scored": 0,
                        "goals_conceded": 0,
                        "matches": 0
                    }
                },
                "errors": True, 
                "message": f"API error: {response.status_code}"
            })
            
        matches_data = response.json()
        
        if not matches_data or not isinstance(matches_data, list) or len(matches_data) == 0:
            logger.warning(f"No matches found for team_id={team_id}")
            # Veritabanı benzeri bir yapı döndürerek analiz kodlarının çalışmasını sağlayalım
            return jsonify({
                "statistics": {
                    "first_half": {
                        "goals_scored": 0,
                        "goals_conceded": 0,
                        "matches": 0
                    },
                    "second_half": {
                        "goals_scored": 0,
                        "goals_conceded": 0,
                        "matches": 0
                    }
                },
                "team_id": team_id,
                "status": "Veri bulunamadı",
                "message": "Bu takım için maç verisi bulunamadı"
            })
        
        # Maçları en yeniden en eskiye sırala
        matches_data = sorted(
            matches_data, 
            key=lambda x: x.get('match_date', ''), 
            reverse=True
        )[:last_matches]  # Sadece son 21 maçı al
        
        # Her maç için ilk ve ikinci yarı gol istatistiklerini hesapla
        processed_matches = []
        
        first_half_home_goals = 0
        first_half_away_goals = 0
        second_half_home_goals = 0
        second_half_away_goals = 0
        
        for match in matches_data:
            # Maç durumunu kontrol et - sadece tamamlanmış maçları hesaplamaya dahil et
            match_status = match.get('match_status', '')
            
            # Maç bilgilerini işlenmiş veriye ekle
            processed_match = {
                "match_id": match.get('match_id'),
                "match_date": match.get('match_date', ''),
                "league_name": match.get('league_name', ''),
                "match_status": match_status
            }
            
            # Sadece tamamlanmış maçlar için istatistik hesapla
            if match_status == 'Finished':
                # İlk yarı skorları
                ht_home_score = int(match.get('match_hometeam_halftime_score', 0) or 0)
                ht_away_score = int(match.get('match_awayteam_halftime_score', 0) or 0)
                
                # Tam maç skorları
                ft_home_score = int(match.get('match_hometeam_score', 0) or 0)
                ft_away_score = int(match.get('match_awayteam_score', 0) or 0)
                
                # İkinci yarı skorlarını hesapla (tam skor - ilk yarı skoru)
                second_half_home = ft_home_score - ht_home_score
                second_half_away = ft_away_score - ht_away_score
                
                # İlgili takım ev sahibi mi yoksa deplasman takımı mı belirle
                # Debug için ekstra log ekle
                home_team_id = match.get('match_hometeam_id', '')
                away_team_id = match.get('match_awayteam_id', '')
                logger.debug(f"Comparing team IDs - API team_id: {team_id}, match_hometeam_id: {home_team_id}, match_awayteam_id: {away_team_id}")
                
                is_home_team = home_team_id == str(team_id)
                
                # Takımın attığı golleri toplam sayılara ekle
                if is_home_team:
                    first_half_home_goals += ht_home_score
                    second_half_home_goals += second_half_home
                else:
                    first_half_away_goals += ht_away_score
                    second_half_away_goals += second_half_away
                    
                # İlgili takım verilerini ekle
                processed_match["home_team"] = {
                    "team_id": match.get('match_hometeam_id', ''),
                    "name": match.get('match_hometeam_name', ''),
                    "is_selected_team": is_home_team
                }
                processed_match["away_team"] = {
                    "team_id": match.get('match_awayteam_id', ''),
                    "name": match.get('match_awayteam_name', ''),
                    "is_selected_team": not is_home_team
                }
                
                # Skor bilgilerini ekle
                processed_match["scores"] = {
                    "first_half": {
                        "home": ht_home_score,
                        "away": ht_away_score
                    },
                    "second_half": {
                        "home": second_half_home,
                        "away": second_half_away
                    },
                    "full_time": {
                        "home": ft_home_score,
                        "away": ft_away_score
                    }
                }
            else:
                # Oynanmamış maç için varsayılan bilgileri ekle
                processed_match["home_team"] = {
                    "team_id": match.get('match_hometeam_id', ''),
                    "name": match.get('match_hometeam_name', '')
                }
                processed_match["away_team"] = {
                    "team_id": match.get('match_awayteam_id', ''),
                    "name": match.get('match_awayteam_name', '')
                }
            
            processed_matches.append(processed_match)
        
        # İstatistik toplamları
        total_matches = len(processed_matches)
        
        # Toplam goller
        total_first_half_goals = first_half_home_goals + first_half_away_goals
        total_second_half_goals = second_half_home_goals + second_half_away_goals
        
        # Maç başına ortalama goller
        avg_first_half_goals = total_first_half_goals / total_matches if total_matches > 0 else 0
        avg_second_half_goals = total_second_half_goals / total_matches if total_matches > 0 else 0
        
        # İlk yarı/maç sonu sonuçları izleme (galibiyet, beraberlik, mağlubiyet)
        ht_results = {
            "home": {"1": 0, "X": 0, "2": 0},  # Ev sahibiyken ilk yarı önde/berabere/geride
            "away": {"1": 0, "X": 0, "2": 0},  # Deplasmanken ilk yarı önde/berabere/geride
            "total": {"1": 0, "X": 0, "2": 0}  # Toplam ilk yarı önde/berabere/geride
        }
        
        ft_results = {
            "home": {"1": 0, "X": 0, "2": 0},  # Ev sahibiyken maç sonu galibiyet/beraberlik/mağlubiyet
            "away": {"1": 0, "X": 0, "2": 0},  # Deplasmanken maç sonu galibiyet/beraberlik/mağlubiyet
            "total": {"1": 0, "X": 0, "2": 0}  # Toplam maç sonu galibiyet/beraberlik/mağlubiyet
        }
        
        # YENİ: İY/MS (HT/FT) kombinasyonları takip etme (9 farklı kombinasyon)
        ht_ft_combinations = {
            "home": {"1/1": 0, "1/X": 0, "1/2": 0, "X/1": 0, "X/X": 0, "X/2": 0, "2/1": 0, "2/X": 0, "2/2": 0},
            "away": {"1/1": 0, "1/X": 0, "1/2": 0, "X/1": 0, "X/X": 0, "X/2": 0, "2/1": 0, "2/X": 0, "2/2": 0},
            "total": {"1/1": 0, "1/X": 0, "1/2": 0, "X/1": 0, "X/X": 0, "X/2": 0, "2/1": 0, "2/X": 0, "2/2": 0}
        }
        
        # İşlenmiş maçları tekrar analiz ederek sonuçları sınıflandır
        for match in processed_matches:
            # Sadece tamamlanmış maçları değerlendir
            if match.get('match_status') != 'Finished' or 'scores' not in match:
                continue
                
            # İlk yarı sonucu
            ht_home = match['scores']['first_half']['home']
            ht_away = match['scores']['first_half']['away']
            
            # Maç sonu sonucu
            ft_home = match['scores']['full_time']['home']
            ft_away = match['scores']['full_time']['away']
            
            # İlk yarı sonucunu belirle (1=önde, X=berabere, 2=geride)
            ht_result = "1" if ht_home > ht_away else "2" if ht_home < ht_away else "X"
            
            # Maç sonu sonucunu belirle (1=galibiyet, X=beraberlik, 2=mağlubiyet)
            ft_result = "1" if ft_home > ft_away else "2" if ft_home < ft_away else "X"
            
            # Takım ev sahibi mi yoksa deplasman mı kontrol et
            is_home = match['home_team']['is_selected_team']
            
            # Sonuçları ilgili kategorilere ekle
            if is_home:
                ht_results['home'][ht_result] += 1
                ft_results['home'][ft_result] += 1
                
                # İY/MS kombinasyonunu ekle (ev sahibi için)
                ht_ft_key = f"{ht_result}/{ft_result}"
                ht_ft_combinations['home'][ht_ft_key] += 1
                ht_ft_combinations['total'][ht_ft_key] += 1
            else:
                # Deplasman takımıyız, sonuçları tersine çevir (1->2, 2->1)
                away_ht_result = "1" if ht_result == "2" else "2" if ht_result == "1" else "X"
                away_ft_result = "1" if ft_result == "2" else "2" if ft_result == "1" else "X"
                
                ht_results['away'][away_ht_result] += 1  
                ft_results['away'][away_ft_result] += 1
                
                # İY/MS kombinasyonunu ekle (deplasman için, tersine çevrilmiş sonuçlarla)
                away_ht_ft_key = f"{away_ht_result}/{away_ft_result}"
                ht_ft_combinations['away'][away_ht_ft_key] += 1
                ht_ft_combinations['total'][away_ht_ft_key] += 1
                
            # Toplam sonuçları güncelle (takımın bakış açısından)
            if is_home:
                ht_results['total'][ht_result] += 1
                ft_results['total'][ft_result] += 1
            else:
                ht_results['total'][away_ht_result] += 1
                ft_results['total'][away_ft_result] += 1
        
        # Sonuç verisini oluştur
        result = {
            "team_id": team_id,
            "total_matches_analyzed": total_matches,
            "statistics": {
                "first_half": {
                    "total_goals": total_first_half_goals,
                    "avg_goals_per_match": round(avg_first_half_goals, 2),
                    "home_goals": first_half_home_goals,
                    "away_goals": first_half_away_goals
                },
                "second_half": {
                    "total_goals": total_second_half_goals,
                    "avg_goals_per_match": round(avg_second_half_goals, 2),
                    "home_goals": second_half_home_goals,
                    "away_goals": second_half_away_goals
                },
                "full_time": {
                    "total_goals": total_first_half_goals + total_second_half_goals,
                    "avg_goals_per_match": round(avg_first_half_goals + avg_second_half_goals, 2)
                }
            },
            "ht_results": ht_results,  # İlk yarı sonuçları (1/X/2)
            "ft_results": ft_results,  # Maç sonu sonuçları (1/X/2)
            "ht_ft_combinations": ht_ft_combinations,  # YENİ: İY/MS kombinasyonları (1/1, 1/X, 1/2, vb.)
            "matches": processed_matches
        }
        
        logger.info(f"Takımın yarı istatistikleri başarıyla alındı: team_id={team_id}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting team half-time stats: {str(e)}")
        return jsonify({"errors": True, "message": str(e)}), 500

@api_v3_bp.route('/matches/half-time-stats/<int:match_id>', methods=['GET']) 
@api_cache(timeout=3600)  # 1 saat önbellekleme (oynanan maç sonuçları değişmez)
def get_half_time_stats(match_id):
    """Maçın ilk yarı (0-45) ve ikinci yarı (46-90) gol istatistlerini döndür"""
    try:
        # API-Football'dan maç bilgilerini çek
        url = "https://apiv3.apifootball.com/"
        params = {
            'action': 'get_events',
            'match_id': match_id,
            'APIkey': API_FOOTBALL_KEY
        }

        logger.info(f"İlk yarı/ikinci yarı gol istatistikleri isteniyor: match_id={match_id}")
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            logger.error(f"API-Football error getting match stats: {response.status_code}")
            return jsonify({"errors": True, "message": f"API error: {response.status_code}"}), 500
            
        data = response.json()
        
        if not data or not isinstance(data, list) or len(data) == 0:
            logger.warning(f"No data found for match_id={match_id}")
            # Maç bulunamadıysa kişiselleştirilmiş boş veri
            match_data = {
                "match_id": str(match_id),
                "match_date": "Veri bulunamadı",
                "match_status": "Veri yok",
                "league_name": "Veri yok",
                "match_hometeam_name": "Bilinmiyor",
                "match_awayteam_name": "Bilinmiyor",
                "match_hometeam_id": "0",
                "match_awayteam_id": "0",
                "match_hometeam_score": "0",
                "match_awayteam_score": "0",
                "match_hometeam_halftime_score": "0", 
                "match_awayteam_halftime_score": "0",
                "team_home_badge": "",
                "team_away_badge": ""
            }
            
            # Bilgilendirici bir 202 yanıtı döndür (veri yok ama istek kabul edildi)
            result = {
                "match_id": match_id,
                "match_date": match_data.get('match_date', ''),
                "match_status": "Veri bulunamadı",
                "league_name": "Bilinmiyor",
                "home_team": {
                    "team_id": "0",
                    "name": "Bilinmiyor",
                    "logo": ""
                },
                "away_team": {
                    "team_id": "0",
                    "name": "Bilinmiyor",
                    "logo": ""
                },
                "scores": {
                    "full_time": {"home": "0", "away": "0"},
                    "half_time": {"home": "0", "away": "0"},
                    "second_half": {"home": 0, "away": 0}
                },
                "info": "Bu maç için veri bulunamadı"
            }
            return jsonify(result), 202  # 202 Accepted
            
        # Veri varsa, her iki takımın ID'lerini al ve yarı istatistiklerini öyle göster
        match_data = data[0]  # İlk eleman maç verilerini içerir
        
        # Takım ID'lerini çıkar
        home_team_id = match_data.get('match_hometeam_id')
        away_team_id = match_data.get('match_awayteam_id')
        
        logger.info(f"Takım ID'leri alındı, her iki takımın ayrı ayrı istatistikleri gösterilecek: home_team_id={home_team_id}, away_team_id={away_team_id}")
        
        # Şimdi her iki takımın son 21 maçını takım istatistiklerinden getir
        if home_team_id and home_team_id != '0' and away_team_id and away_team_id != '0':
            # Takım endpointine git
            return jsonify({
                "match_id": match_id,
                "redirect": "Bu endpoint yakında kaldırılacak. Lütfen takım ID'leri ile sorgulama yapın.",
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_team_name": match_data.get('match_hometeam_name', 'Bilinmiyor'),
                "away_team_name": match_data.get('match_awayteam_name', 'Bilinmiyor'),
                "home_team_endpoint": f"/api/v3/team/half-time-stats/{home_team_id}",
                "away_team_endpoint": f"/api/v3/team/half-time-stats/{away_team_id}"
            })
        
        # İlk yarı ve ikinci yarı skorları çıkar (bu eskiden kalma kod, şimdi takım ID'leri ile çalışmaya öncelik vermeliyiz)
        ht_home_score = match_data.get('match_hometeam_halftime_score', 0)
        ht_away_score = match_data.get('match_awayteam_halftime_score', 0)
        
        ft_home_score = match_data.get('match_hometeam_score', 0)
        ft_away_score = match_data.get('match_awayteam_score', 0)
        
        # İkinci yarı skorlarını hesapla (tam skor - ilk yarı skoru)
        second_half_home = int(ft_home_score) - int(ht_home_score) if ft_home_score is not None and ht_home_score is not None else 0
        second_half_away = int(ft_away_score) - int(ht_away_score) if ft_away_score is not None and ht_away_score is not None else 0
        
        # Maç hakkında bazı genel bilgiler
        result = {
            "match_id": match_id,
            "match_date": match_data.get('match_date', ''),
            "match_status": match_data.get('match_status', ''),
            "league_name": match_data.get('league_name', ''),
            "home_team": {
                "team_id": home_team_id or "0",
                "name": match_data.get('match_hometeam_name', 'Bilinmiyor'),
                "logo": match_data.get('team_home_badge', '')
            },
            "away_team": {
                "team_id": away_team_id or "0",
                "name": match_data.get('match_awayteam_name', 'Bilinmiyor'),
                "logo": match_data.get('team_away_badge', '')
            },
            "scores": {
                "full_time": {
                    "home": ft_home_score,
                    "away": ft_away_score
                },
                "half_time": {
                    "home": ht_home_score,
                    "away": ht_away_score
                },
                "second_half": {
                    "home": second_half_home,
                    "away": second_half_away
                }
            }
        }
        
        logger.info(f"İlk yarı/ikinci yarı verileri başarıyla alındı: match_id={match_id}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting match half-time stats: {str(e)}")
        return jsonify({"errors": True, "message": str(e)}), 500

@api_v3_bp.route('/fixtures/team/<int:team_id>', methods=['GET'])
@api_cache(timeout=1800)  # 30 dakika önbellek süresi
def get_team_stats(team_id):
    """
    Takımın detaylı istatistiklerini döndüren API endpoint
    Popup takım istatistikleri için kullanılır
    """
    try:
        # Takımın son maçlarını al
        api_key = os.environ.get('API_FOOTBALL_KEY', '39bc8dd65153ff5c7c0f37b4939009de04f4a70c593ee1aea0b8f70dd33268c0')
        url = "https://apiv3.apifootball.com/"
        
        # Son 10 maçı çek
        params = {
            'action': 'get_events',
            'team_id': team_id,
            'from': '2023-01-01',  # Yeteri kadar geriye git
            'to': datetime.now().strftime('%Y-%m-%d'),
            'APIkey': api_key
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return jsonify([])
            
        matches = response.json()
        if not isinstance(matches, list):
            return jsonify([])
            
        # Son 10 maçı format­la ve döndür
        formatted_matches = []
        for match in matches[:10]:  # Son 10 maç
            match_date = match.get('match_date', '')
            try:
                # Tarihi daha okunabilir formata dönüştür
                date_obj = datetime.strptime(match_date, '%Y-%m-%d')
                formatted_date = date_obj.strftime('%d %b %Y')
            except Exception:
                formatted_date = match_date
                
            formatted_match = {
                'date': formatted_date,
                'match': f"{match.get('match_hometeam_name', '')} vs {match.get('match_awayteam_name', '')}",
                'score': f"{match.get('match_hometeam_score', '')} - {match.get('match_awayteam_score', '')}"
            }
            formatted_matches.append(formatted_match)
            
        return jsonify(formatted_matches)
        
    except Exception as e:
        print(f"Takım istatistikleri alınırken hata: {str(e)}")
        return jsonify([])

def get_team_matches(team_id):
    """
    Takımın son maçlarını döndüren API endpoint
    Frontend'den backend'e taşınan hesaplamalarla geliştirilmiş versiyonu
    """
    try:
        # Geçersiz takım ID kontrolü
        if not team_id or not isinstance(team_id, int) or team_id <= 0:
            team_id_value = team_id if isinstance(team_id, int) else str(team_id)
            logger.warning(f"Invalid team_id={team_id_value} provided to get_team_matches")
            
            # Boş ama geçerli bir yanıt döndür - hata yerine varsayılan değerler kullan
            return jsonify({
                "team_id": str(team_id_value),
                "team_name": "Bilinmeyen Takım",
                "status": "Veri bulunamadı",
                "message": "Geçersiz takım ID'si",
                "matches": [],
                "total_matches": 0,
                "form": {
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "goals_scored": 0,
                    "goals_conceded": 0
                }
            }), 202  # 202 Accepted
            
        # Query parametrelerini al
        last_count = int(request.args.get('last', 5))  # Son kaç maçı alacağız
        include_stats = request.args.get('stats', 'true').lower() == 'true'  # İstatistikler dahil edilsin mi
        team_name = request.args.get('team_name', f'Takım {team_id}')
        
        logger.info(f"Takım maçları isteniyor: team_id={team_id}, team_name={team_name}, last_count={last_count}")
        
        # API Football kullanarak takımın maçlarını al
        url = "https://apiv3.apifootball.com/"
        params = {
            'action': 'get_events',
            'team_id': team_id,
            'APIkey': API_FOOTBALL_KEY
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            logger.error(f"Error getting team matches: {response.status_code}")
            return jsonify({
                "errors": True, 
                "message": f"API error: {response.status_code}",
                "team_id": str(team_id),
                "team_name": team_name,
                "matches": [],
                "total_matches": 0
            }), 500
            
        data = response.json()
        
        # API yanıtı boş veya geçersiz ise
        if not data or not isinstance(data, list):
            logger.warning(f"No or invalid data returned for team_id={team_id}")
            return jsonify({
                "team_id": str(team_id),
                "team_name": team_name,
                "status": "Veri bulunamadı",
                "message": "Bu takım için maç verisi bulunamadı",
                "matches": [],
                "total_matches": 0,
                "form": {
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "goals_scored": 0,
                    "goals_conceded": 0
                }
            }), 202
        
        # Maç verilerini sırala ve son maçları al
        matches_data = []
        all_match_data = []
        
        if isinstance(data, list) and len(data) > 0:
            # Tarihe göre sırala
            sorted_data = sorted(
                data, 
                key=lambda x: x.get('match_date', ''), 
                reverse=True
            )
            
            # İstatistikler için hesaplamalar
            team_stats = {
                "total_matches": len(sorted_data),
                "finished_matches": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "goals_scored": 0,
                "goals_conceded": 0,
                "clean_sheets": 0,
                "failed_to_score": 0,
                "form_streak": "",
                "last_5_results": [],
                "last_10_results": [],
                "home_matches": 0,
                "away_matches": 0,
                "home_wins": 0,
                "away_wins": 0,
                "home_goals_for": 0,
                "away_goals_for": 0,
                "home_goals_against": 0,
                "away_goals_against": 0,
                "goal_difference": 0
            }
            
            # Tüm maçları işle
            for match in sorted_data:
                # Takımın ev sahibi mi yoksa deplasman mı olduğunu belirle
                is_home = str(match.get('match_hometeam_id', '')) == str(team_id)
                team_goals = int(match.get('match_hometeam_score', 0)) if is_home else int(match.get('match_awayteam_score', 0))
                opponent_goals = int(match.get('match_awayteam_score', 0)) if is_home else int(match.get('match_hometeam_score', 0))
                
                match_date = match.get('match_date', '')
                try:
                    date_obj = datetime.strptime(match_date, '%Y-%m-%d')
                    formatted_date = date_obj.strftime('%d.%m.%Y')
                except:
                    formatted_date = match_date
                    
                # Her maç için standart format
                formatted_match = {
                    "date": formatted_date,
                    "raw_date": match_date,
                    "league": match.get('league_name', ''),
                    "match": f"{match.get('match_hometeam_name', '')} vs {match.get('match_awayteam_name', '')}",
                    "home_team": match.get('match_hometeam_name', ''),
                    "away_team": match.get('match_awayteam_name', ''),
                    "score": f"{match.get('match_hometeam_score', '0')} - {match.get('match_awayteam_score', '0')}",
                    "half_time_score": f"{match.get('match_hometeam_halftime_score', '?')} - {match.get('match_awayteam_halftime_score', '?')}",
                    "status": match.get('match_status', ''),
                    "is_home": is_home,
                    "team_score": team_goals,
                    "opponent_score": opponent_goals
                }
                
                all_match_data.append(formatted_match)
                
                # Maç durumu kontrolü - yalnızca tamamlanmış maçları hesapla
                match_status = match.get('match_status', '')
                if match_status == 'Finished':
                    team_stats["finished_matches"] += 1
                    
                    # Gollerle ilgili istatistikler
                    team_stats["goals_scored"] += team_goals
                    team_stats["goals_conceded"] += opponent_goals
                    team_stats["goal_difference"] += (team_goals - opponent_goals)
                    
                    # Temiz kale ve gol atamama
                    if opponent_goals == 0:
                        team_stats["clean_sheets"] += 1
                    if team_goals == 0:
                        team_stats["failed_to_score"] += 1
                    
                    # Ev/deplasman istatistikleri
                    if is_home:
                        team_stats["home_matches"] += 1
                        team_stats["home_goals_for"] += team_goals
                        team_stats["home_goals_against"] += opponent_goals
                    else:
                        team_stats["away_matches"] += 1
                        team_stats["away_goals_for"] += team_goals
                        team_stats["away_goals_against"] += opponent_goals
                    
                    # Sonuçlar (W/D/L)
                    if team_goals > opponent_goals:
                        team_stats["wins"] += 1
                        result = "W"
                        if is_home:
                            team_stats["home_wins"] += 1
                        else:
                            team_stats["away_wins"] += 1
                    elif team_goals == opponent_goals:
                        team_stats["draws"] += 1
                        result = "D"
                    else:
                        team_stats["losses"] += 1
                        result = "L"
                    # Son 5 ve 10 maç sonuçları
                    if len(team_stats["last_5_results"]) < 5:
                        team_stats["last_5_results"].append(result)
                    if len(team_stats["last_10_results"]) < 10:
                        team_stats["last_10_results"].append(result)
                else:
                    # Oynanmamış maçları form verilerinde kullanma
                    continue
            
            # Son n maçı formatla ve listele
            matches_data = all_match_data[:last_count]
            
            # Form istatistiklerini hesapla
            team_stats["form_streak"] = "".join(team_stats["last_5_results"])
            
            # Ortalama değerleri hesapla
            if team_stats["finished_matches"] > 0:
                team_stats["avg_goals_scored"] = round(team_stats["goals_scored"] / team_stats["finished_matches"], 2)
                team_stats["avg_goals_conceded"] = round(team_stats["goals_conceded"] / team_stats["finished_matches"], 2)
                
                # Ev/deplasman ortalama goller
                if team_stats["home_matches"] > 0:
                    team_stats["avg_home_goals_for"] = round(team_stats["home_goals_for"] / team_stats["home_matches"], 2)
                    team_stats["avg_home_goals_against"] = round(team_stats["home_goals_against"] / team_stats["home_matches"], 2)
                else:
                    team_stats["avg_home_goals_for"] = 0
                    team_stats["avg_home_goals_against"] = 0
                
                if team_stats["away_matches"] > 0:
                    team_stats["avg_away_goals_for"] = round(team_stats["away_goals_for"] / team_stats["away_matches"], 2)
                    team_stats["avg_away_goals_against"] = round(team_stats["away_goals_against"] / team_stats["away_matches"], 2)
                else:
                    team_stats["avg_away_goals_for"] = 0
                    team_stats["avg_away_goals_against"] = 0
            
            # Kazanma oranları
            if team_stats["finished_matches"] > 0:
                team_stats["win_ratio"] = round(team_stats["wins"] / team_stats["finished_matches"] * 100, 2)
                team_stats["draw_ratio"] = round(team_stats["draws"] / team_stats["finished_matches"] * 100, 2)
                team_stats["loss_ratio"] = round(team_stats["losses"] / team_stats["finished_matches"] * 100, 2)
            
            if team_stats["home_matches"] > 0:
                team_stats["home_win_ratio"] = round(team_stats["home_wins"] / team_stats["home_matches"] * 100, 2)
            
            if team_stats["away_matches"] > 0:
                team_stats["away_win_ratio"] = round(team_stats["away_wins"] / team_stats["away_matches"] * 100, 2)
        
        response_data = {
            "response": matches_data,
            "team_id": team_id,
            "team_name": team_name
        }
        
        # İstatistikleri dahil et
        if include_stats:
            response_data["stats"] = team_stats
            response_data["all_matches"] = all_match_data
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error getting team matches: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"errors": True, "message": str(e)}), 500

@api_v3_bp.route('/status')
def api_status():
    return jsonify({"status": "ok", "message": "API is working"})
    
@api_v3_bp.route('/clear-prediction-cache', methods=['GET', 'POST'])
@api_v3_bp.route('/clear-cache', methods=['GET', 'POST'])
@app.route('/api/clear-prediction-cache', methods=['GET', 'POST'])
@app.route('/api/clear-cache', methods=['GET', 'POST'])
def clear_prediction_cache():
    """Tahmin önbelleğini temizle - sorunlu tahminleri yenilemek için kullanılır"""
    from match_prediction import MatchPredictor
    
    try:
        # Önce ana uygulamadan predictor'ı alma
        try:
            from main import predictor
            # Ana predictor nesnesini tamamen temizle
            predictor.predictions_cache = {}
            predictor.save_cache()
            logger.info("Ana uygulama predictor önbelleği temizlendi")
        except ImportError:
            logger.warning("Main predictor'a erişilemedi, muhtemelen ana rotalardan çağrıldı")
        
        # API yollarındaki global match_predictor'ı sıfırla (eğer tanımlıysa)
        global match_predictor
        try:
            match_predictor = MatchPredictor()
            match_predictor.predictions_cache = {}
            match_predictor.save_cache()
            logger.info("API rotaları match_predictor önbelleği temizlendi")
        except Exception as predictor_error:
            logger.warning(f"Global match_predictor temizlenirken hata: {str(predictor_error)}")
        
        # Flask önbelleğini temizlemeyi dene
        try:
            # Eğer current_app içinde cache varsa temizle
            from flask import current_app
            if hasattr(current_app, 'cache'):
                current_app.cache.clear()
                logger.info("Flask önbelleği temizlendi")
        except Exception as cache_error:
            logger.warning(f"Flask önbelleği temizlenirken hata: {str(cache_error)}")
            
        # Önbellek dosyasını doğrudan temizle
        try:
            with open('predictions_cache.json', 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False)
            logger.info("Önbellek dosyası başarıyla temizlendi.")
        except Exception as file_error:
            logger.error(f"Önbellek dosyası yazılırken hata: {str(file_error)}")
        
        message = "Tahmin önbelleği başarıyla temizlendi. Yeni tahminler güncel algoritma ile hesaplanacak. KG VAR/YOK tutarsızlığı giderilecek."
    except Exception as e:
        logger.error(f"Önbellek temizleme işlemi sırasında genel hata: {str(e)}")
        message = f"Tahmin önbelleği temizlenirken hata: {str(e)}"
    
    return jsonify({
        "status": "success",
        "message": message,
        "cache_cleared": True,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
@api_v3_bp.route('/team-stats/<team_id>')
@api_cache(timeout=1800)  # 30 dakika önbellek süresi
def get_team_stats_api(team_id):
    """
    Takımın detaylı istatistiklerini döndüren API endpoint (v3 API ile)
    Popup takım istatistikleri için kullanılır
    """
    try:
        # Önce tahmin verilerinden maçları kontrol edelim
        from main import predictor
        try:
            # Tahmin önbelleğini kontrol et
            cache_data = predictor.load_cache()
            
            # Tüm maçları dön ve belirtilen takım ID'sini içeren maçları bul
            team_matches = []
            team_name = None
            
            for key, match_data in cache_data.items():
                # Maç anahtarını analiz et (genellikle "home_team_id-away_team_id" formatındadır)
                match_teams = key.split('-')
                if len(match_teams) == 2:
                    home_id, away_id = match_teams
                    
                    # Takım ID'sini kontrol et
                    if home_id == str(team_id) or away_id == str(team_id):
                        # Maç bilgilerini al
                        if 'match_data' in match_data:
                            home_team_name = match_data.get('home_team_name', 'Bilinmeyen Takım')
                            away_team_name = match_data.get('away_team_name', 'Bilinmeyen Takım')
                            
                            # Takım adını belirle
                            if home_id == str(team_id):
                                team_name = home_team_name
                                is_home = True
                            else:
                                team_name = away_team_name
                                is_home = False
                            
                            # Maç detaylarını al
                            match_date = match_data.get('date', datetime.now().strftime('%Y-%m-%d'))
                            status = match_data.get('status', 'Tamamlandı')
                            
                            # Skorları formatla
                            home_score = match_data.get('home_score', '?')
                            away_score = match_data.get('away_score', '?')
                            
                            # Tarihi formatla
                            try:
                                date_obj = datetime.strptime(match_date, '%Y-%m-%d')
                                formatted_date = date_obj.strftime('%d %b %Y')
                            except:
                                formatted_date = match_date
                            
                            # Maç bilgilerini ekle
                            team_matches.append({
                                'date': formatted_date,
                                'match': f"{home_team_name} vs {away_team_name}",
                                'score': f"{home_score} - {away_score}",
                                'status': status,
                                'is_home': is_home
                            })
            
            # Eğer önbellekte veriler bulunduysa, doğrudan döndür
            if team_matches:
                logger.info(f"Takım {team_id} için önbellekten {len(team_matches)} maç bulundu")
                return jsonify(team_matches)
        except Exception as cache_error:
            logger.warning(f"Önbellekten takım maçları alınırken hata: {str(cache_error)}")
        
        # Önbellekte veri bulunamadıysa devam et
        # Takımın son maçlarını al
        api_key = os.environ.get('API_FOOTBALL_KEY', '39bc8dd65153ff5c7c0f37b4939009de04f4a70c593ee1aea0b8f70dd33268c0')
        url = "https://apiv3.apifootball.com/"
        
        # Son 10 maçı çek (günümüzden 3 yıl öncesine kadar)
        three_years_ago = (datetime.now() - timedelta(days=1095)).strftime('%Y-%m-%d')
        
        params = {
            'action': 'get_events',
            'team_id': team_id,
            'from': three_years_ago,  # 3 yıl öncesinden
            'to': datetime.now().strftime('%Y-%m-%d'),
            'APIkey': api_key
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            logger.error(f"API yanıt hatası {response.status_code}: Takım: {team_id}")
            # İlk API başarısız olduğunda, backup API'yi deneyelim
            try:
                return get_team_stats_backup(team_id)
            except Exception as e2:
                logger.error(f"Backup API ile takım verileri alınamadı: {str(e2)}")
                # Alternatif olarak dummy_matches kullan
                return get_team_stats_dummy(team_id)
            
        matches = response.json()
        if not isinstance(matches, list) or len(matches) == 0:
            logger.warning(f"API yanıtı boş veya geçersiz format: Takım: {team_id}")
            # API yanıtı boş ise, backup API'yi deneyelim
            try:
                return get_team_stats_backup(team_id)
            except Exception as e2:
                logger.error(f"Backup API ile takım verileri alınamadı: {str(e2)}")
                # Alternatif olarak dummy_matches kullan
                return get_team_stats_dummy(team_id)
            
        # Son 10 maçı format­la ve döndür
        formatted_matches = []
        for match in matches[:10]:  # Son 10 maç
            match_date = match.get('match_date', '')
            try:
                # Tarihi daha okunabilir formata dönüştür
                date_obj = datetime.strptime(match_date, '%Y-%m-%d')
                formatted_date = date_obj.strftime('%d %b %Y')
            except Exception:
                formatted_date = match_date
                
            home_team = match.get('match_hometeam_name', '')
            away_team = match.get('match_awayteam_name', '')
            home_score = match.get('match_hometeam_score', '')
            away_score = match.get('match_awayteam_score', '')
            
            # Maç durumunu ekleyelim
            match_status = match.get('match_status', '')
            status_text = ""
            if match_status != "Finished" and match_status != "":
                status_text = f" ({match_status})"
                
            formatted_match = {
                'date': formatted_date,
                'match': f"{home_team} vs {away_team}",
                'score': f"{home_score} - {away_score}{status_text}",
                'raw_date': match_date,
                'status': match_status,
                'league': match.get('league_name', '')
            }
            formatted_matches.append(formatted_match)
            
        return jsonify(formatted_matches)
        
    except Exception as e:
        logger.error(f"Takım istatistikleri alınırken hata: {str(e)}")
        # Ana API ile bir hata olursa, backup API'yi deneyelim
        try:
            return get_team_stats_backup(team_id)
        except Exception as e2:
            logger.error(f"Backup API ile takım verileri alınamadı: {str(e2)}")
            # Alternatif olarak dummy_matches kullan
            return get_team_stats_dummy(team_id)

def get_team_stats_dummy(team_id):
    """
    API'den veri alınamadığında kullanılacak varsayılan istatistikler için
    tahmin edilmiş verileri döndürür. 
    Bu, takımın ad bilgileri ile asgari düzeyde veri oluşturulur.
    """
    try:
        # Takım adını bulmaya çalış
        team_name = "Takım " + str(team_id)
        
        # Tahmin önbelleğinden takım adını bulmaya çalış
        from main import predictor
        cache_data = predictor.load_cache()
        
        # Tüm tahminleri dolaş ve takım adını bul
        for key, prediction in cache_data.items():
            match_teams = key.split('-')
            if len(match_teams) == 2:
                home_id, away_id = match_teams
                
                if home_id == str(team_id) and 'home_team_name' in prediction:
                    team_name = prediction['home_team_name']
                    break
                elif away_id == str(team_id) and 'away_team_name' in prediction:
                    team_name = prediction['away_team_name']
                    break
        
        # Bugünün tarihini al
        today = datetime.now().strftime('%d %b %Y')
        
        # Son tarih maçı oluştur
        return jsonify([
            {
                'date': today,
                'match': f"{team_name} - Form/Maç verisi bulunamadı",
                'score': "Veri yok",
                'status': "Bilgi",
                'league': "Veri bulunamadı"
            }
        ])
    
    except Exception as e:
        logger.error(f"Varsayılan takım istatistikleri oluşturulurken hata: {str(e)}")
        return jsonify([])
            
def get_team_stats_backup(team_id):
    """
    Birinci API kaynak başarısız olduğunda yedek kaynak olarak farklı bir 
    API endpoint'i kullanarak takım istatistiklerini getirmeyi dener
    """
    try:
        # Alternatif API için API anahtarı
        api_key = os.environ.get('FOOTBALL_DATA_API_KEY', '85c1a3c16af54ce687b76479261b6e73')
        headers = {'X-Auth-Token': api_key}
        
        # Football Data API kullanarak takımın son maçlarını al
        url = f"https://api.football-data.org/v4/teams/{team_id}/matches?limit=10"
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            logger.warning(f"Backup API yanıt hatası {response.status_code}: Takım: {team_id}")
            return jsonify([])
            
        data = response.json()
        matches = data.get('matches', [])
        
        if not matches or len(matches) == 0:
            logger.warning(f"Backup API yanıtı boş: Takım: {team_id}")
            return jsonify([])
            
        # Son 10 maçı formatla ve döndür
        formatted_matches = []
        for match in matches[:10]:
            utc_date = match.get('utcDate', '')
            try:
                # Tarihi daha okunabilir formata dönüştür
                date_obj = datetime.strptime(utc_date, '%Y-%m-%dT%H:%M:%SZ')
                formatted_date = date_obj.strftime('%d %b %Y')
            except Exception:
                formatted_date = utc_date.split('T')[0] if 'T' in utc_date else utc_date
                
            home_team = match.get('homeTeam', {}).get('name', '')
            away_team = match.get('awayTeam', {}).get('name', '')
            
            score = match.get('score', {})
            full_time = score.get('fullTime', {})
            home_score = full_time.get('home', '-')
            away_score = full_time.get('away', '-')
            
            # Maç durumunu ekleyelim
            match_status = match.get('status', '')
            status_text = ""
            if match_status != "FINISHED":
                status_text = f" ({match_status})"
                
            formatted_match = {
                'date': formatted_date,
                'match': f"{home_team} vs {away_team}",
                'score': f"{home_score} - {away_score}{status_text}",
                'status': match_status,
                'league': match.get('competition', {}).get('name', '')
            }
            formatted_matches.append(formatted_match)
            
        return jsonify(formatted_matches)
        
    except Exception as e:
        logger.error(f"Backup API ile takım istatistikleri alınırken hata: {str(e)}")
        return jsonify([])

@api_v3_bp.route('/train-neural-network', methods=['POST'])
def train_neural_network():
    """Sinir ağı modelini eğit (artık otomatik yapılıyor)"""
    try:
        from main import predictor
        success = predictor.collect_training_data()
        if success:
            return jsonify({"success": True, "message": "Sinir ağı modelleri başarıyla eğitildi."})
        else:
            return jsonify({"success": False, "message": "Yeterli veri bulunamadı veya eğitim başarısız oldu. Sinir ağları tahmin sırasında otomatik olarak eğitilecektir."})
    except Exception as e:
        logger.error(f"Sinir ağı eğitimi sırasında hata: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_v3_bp.route('/predict-match/<home_team_id>/<away_team_id>')
@api_cache(timeout=600)  # 10 dakika önbellek süresi  
def api_v3_predict_match(home_team_id, away_team_id):
    try:
        from main import predictor
        home_name = request.args.get('home_name', '')
        away_name = request.args.get('away_name', '')
        force_update = request.args.get('force_update', 'false').lower() == 'true'
        use_goal_trend = request.args.get('use_goal_trend', 'true').lower() == 'true'  # Varsayılan olarak aktif
        
        # İlk yarı/maç sonu tahminleri için takım-spesifik ayarlamaları alalım
        team_adjustments = None
        try:
            # Takım-spesifik modelleri yükle
            from team_specific_models import TeamSpecificPredictor
            team_specific_predictor = TeamSpecificPredictor()
            
            # Takımların form verilerini al
            home_form = predictor.get_team_form(home_team_id)
            away_form = predictor.get_team_form(away_team_id)
            
            # Takım ayarlamalarını al
            team_adjustments = team_specific_predictor.get_team_adjustments(
                home_team_id, away_team_id, home_form, away_form
            )
            logger.info(f"Ana tahmin için takım-spesifik ayarlamalar: {home_team_id} vs {away_team_id}")
        except Exception as e:
            logger.warning(f"Ana tahmin için takım-spesifik ayarlamalar yapılırken hata: {str(e)}")
            
        # Tahmin fonksiyonunu çağır (Gol Trend İvmesi analizi parametresi ile)
        try:
            # Gol Trend İvmesi analizini kontrol parametresi ile yapılandır
            prediction = predictor.predict_match(
                home_team_id, away_team_id, 
                home_name, away_name, 
                force_update=force_update,
                use_goal_trend_analysis=use_goal_trend
            )
            
            if not prediction:
                logger.error(f"Tahmin null döndü: {home_name} vs {away_name}")
                return jsonify({"error": "Tahmin yapılamadı. Yeterli veri bulunmuyor."}), 400
        except Exception as predict_error:
            logger.error(f"Tahmin fonksiyonu çağrılırken hata oluştu: {str(predict_error)}", exc_info=True)
            return jsonify({
                "error": "Tahmin işlemi sırasında teknik bir hata oluştu, lütfen daha sonra tekrar deneyin",
                "match": f"{home_name} vs {away_name}",
                "timestamp": time.time()
            }), 500
            
        if prediction:
            # Simplify prediction data by removing complex card and corner predictions
            # to ensure lighter response payload
            if 'predictions' in prediction and 'betting_predictions' in prediction['predictions']:
                betting_predictions = prediction['predictions']['betting_predictions']
                # Remove corner and card predictions to reduce complexity
                if 'cards_over_3_5' in betting_predictions:
                    del betting_predictions['cards_over_3_5']
                if 'corners_over_9_5' in betting_predictions:
                    del betting_predictions['corners_over_9_5']
                
                # İY/MS tahminlerini en yüksek olasılıklı tahminden çıkar
                # Çok önemli: İY/MS tahmini half_time_full_time market'ında saklanır ve
                # most_confident_bet'ten kaldırılmalıdır
                # HER DURUMDA kontrol edelim ki tam emin olalım 
                if 'most_confident_bet' in prediction['predictions']:
                    
                    logger.info(f"İY/MS tahmini en yüksek olasılıklı tahminden kaldırılıyor: {prediction['predictions']['most_confident_bet']}")
                    
                    # En yüksek olasılıklı alternatif tahmini bul (İY/MS hariç)
                    next_best_bet = None
                    next_best_prob = 0
                    
                    # İY/MS ve corner/card tahminleri hariç diğer tahminlere bak
                    for market, bet_data in betting_predictions.items():
                        if market != 'half_time_full_time' and \
                           market != 'corners_over_9_5' and \
                           market != 'cards_over_3_5' and \
                           'probability' in bet_data and \
                           bet_data['probability'] > next_best_prob:
                            
                            next_best_bet = {
                                'market': market,
                                'prediction': bet_data['prediction'],
                                'probability': bet_data['probability']
                            }
                            next_best_prob = bet_data['probability']
                    
                    # Alternatif tahmin bulunamazsa maç sonucu tahminini kullan
                    if not next_best_bet:
                        # Maç sonucu olasılıklarını karşılaştır
                        home_prob = prediction['predictions'].get('home_win_probability', 0)
                        draw_prob = prediction['predictions'].get('draw_probability', 0)
                        away_prob = prediction['predictions'].get('away_win_probability', 0)
                        
                        if home_prob >= draw_prob and home_prob >= away_prob:
                            result = 'HOME_WIN'
                            prob = home_prob
                        elif draw_prob >= away_prob:
                            result = 'DRAW'
                            prob = draw_prob
                        else:
                            result = 'AWAY_WIN'
                            prob = away_prob
                        
                        next_best_bet = {
                            'market': 'match_result',
                            'prediction': result,
                            'probability': prob
                        }
                    
                    # En yüksek olasılıklı tahmini değiştir
                    prediction['predictions']['most_confident_bet'] = next_best_bet
                    logger.info(f"En yüksek olasılıklı tahmin şuna değiştirildi: {next_best_bet}")
                    
                    # Eğer exact_score ve most_likely_outcome varsa, tutarlı olduklarından emin ol
                    if 'exact_score' in prediction['predictions']:
                        exact_score = prediction['predictions']['exact_score']
                        # Kesin skordan sonuç çıkarma
                        from match_prediction import MatchPredictor
                        predictor = MatchPredictor()
                        # Object formatındaki exact_score için prediction anahtarını kontrol et
                        if isinstance(exact_score, dict) and 'prediction' in exact_score:
                            exact_score_value = exact_score['prediction']
                        else:
                            exact_score_value = exact_score
                            
                        derived_outcome = predictor._get_outcome_from_score(exact_score_value)
                        
                        # Sonucu güncelle
                        if derived_outcome:
                            if 'most_likely_outcome' in prediction['predictions']:
                                logger.info(f"Kesin skordan ({exact_score_value}) hesaplanan sonuç: {derived_outcome}")
                                prediction['predictions']['most_likely_outcome'] = derived_outcome
                            
                            if 'match_outcome' in prediction['predictions']:
                                prediction['predictions']['match_outcome'] = derived_outcome
            return jsonify(prediction)
        else:
            return jsonify({"error": "Tahmin yapılamadı. Yeterli veri bulunmuyor."}), 400
    except Exception as e:
        logger.error(f"API v3 tahmin yapılırken hata: {str(e)}", exc_info=True)
        # Kullanıcıya daha anlaşılır ve güvenli bir hata mesajı döndür
        error_message = "Tahmin işlemi sırasında teknik bir hata oluştu, lütfen daha sonra tekrar deneyin"
        # Değişken kontrolü yaparak daha güvenli bir yaklaşım
        try:
            home_name_val = home_name if 'home_name' in locals() else f"Takım {home_team_id}"
            away_name_val = away_name if 'away_name' in locals() else f"Takım {away_team_id}"
            match_info = f"{home_name_val} vs {away_name_val}"
        except:
            match_info = f"{home_team_id} vs {away_team_id}"
        return jsonify({
            "error": error_message,
            "match": match_info,
            "timestamp": time.time()
        }), 500

# Advanced predictions endpoint
@api_v3_bp.route('/htft-prediction/<home_team_id>/<away_team_id>', methods=['GET'])
@api_cache(timeout=600)  # 10 dakika önbellek süresi
def get_htft_prediction(home_team_id, away_team_id):
    """
    İlk Yarı/Maç Sonu (İY/MS) tahmini döndüren API endpoint
    Bu endpoint, Sürpriz Butonu için Python tabanlı backend tahmin modelini kullanır
    Frontend'den JavaScript hesaplamalarını tamamen backend'e taşır
    """
    # Performans ölçümü
    start_time = time.time()
    
    # Geçersiz takım ID'leri için hata kontrolü
    if not home_team_id or not isinstance(home_team_id, str) and not isinstance(home_team_id, int):
        logger.warning(f"Geçersiz ev sahibi takım ID: {home_team_id}")
        home_team_id = str(home_team_id or "0")
        
    if not away_team_id or not isinstance(away_team_id, str) and not isinstance(away_team_id, int):
        logger.warning(f"Geçersiz deplasman takım ID: {away_team_id}")
        away_team_id = str(away_team_id or "0")
        
    # URL parametrelerini fonksiyon başlangıcında tanımla (tüm fonksiyon kapsamında kullanılabilir)
    home_name = request.args.get('home_name', f'Takım {home_team_id}')
    away_name = request.args.get('away_name', f'Takım {away_team_id}')
    # Yeni eklenen parametre: Önbelleği atlayıp zorla güncel tahmin oluşturma 
    force_update = request.args.get('force_update', 'false').lower() == 'true'
    
    # Talep edilen hesaplama yöntemi
    calculation_method = request.args.get('method', 'all')  # all, statistical, monte_carlo, neural_net
    
    logger.info(f"İY/MS tahmini isteniyor: {home_name}({home_team_id}) vs {away_name}({away_team_id})")
    
    try:
        # Maç sonucu tahmini için global beklenti
        global_outcome_str = request.args.get('global_outcome', None)  # Ana tahmin butonundan gelen sonuç (string)
        global_outcome = None
        
        # Sürpriz butonu parametresi
        is_surprise_button = request.args.get('surprise_button', 'false').lower() == 'true'
        
        # String'i JSON'a çevir (eğer var ise)
        if global_outcome_str:
            try:
                import json
                global_outcome = json.loads(global_outcome_str)
                # Sürpriz butonu flag'ini ekle
                global_outcome['is_surprise_button'] = is_surprise_button
                
                logger.info(f"Global outcome JSON başarıyla parse edildi, is_surprise_button={is_surprise_button}")
            except Exception as e:
                logger.warning(f"Global outcome JSON parse hatası: {str(e)}")
                # Parse hatası durumunda manuel bir global_outcome objesi oluştur
                global_outcome = {
                    "is_surprise_button": is_surprise_button,
                    "score_prediction": {
                        "beklenen_home_gol": 1.5,
                        "beklenen_away_gol": 1.0
                    }
                }
        else:
            # Global outcome yoksa, API'den tahmin bilgilerini alalım
            try:
                from main import predictor
                # Tahmin butonundan verileri almak için API çağrısı yap (Gol Trend İvmesi analizini kullanarak)
                basic_prediction = predictor.predict_match(
                    home_team_id, away_team_id, 
                    home_name, away_name,
                    use_goal_trend_analysis=True
                )
                
                if basic_prediction and 'expected_goals' in basic_prediction:
                    expected_home_goals = basic_prediction['expected_goals'].get('home', 1.5)
                    expected_away_goals = basic_prediction['expected_goals'].get('away', 1.0)
                    
                    logger.info(f"Ana tahmin modülünden gol beklentileri alındı: Ev={expected_home_goals:.2f}, Deplasman={expected_away_goals:.2f}")
                    
                    # Tahmin butonundan alınan gol beklentileri ile global_outcome oluştur
                    global_outcome = {
                        "is_surprise_button": is_surprise_button,
                        "expected_home_goals": expected_home_goals,
                        "expected_away_goals": expected_away_goals,
                        "score_prediction": {
                            "beklenen_home_gol": expected_home_goals,
                            "beklenen_away_gol": expected_away_goals
                        }
                    }
                else:
                    # Tahmin bilgisi yoksa varsayılan değerlerle devam et
                    global_outcome = {
                        "is_surprise_button": is_surprise_button,
                        "expected_home_goals": 1.5,
                        "expected_away_goals": 1.0,
                        "score_prediction": {
                            "beklenen_home_gol": 1.5,
                            "beklenen_away_gol": 1.0
                        }
                    }
            except Exception as e:
                logger.warning(f"Tahmin bilgileri alınırken hata: {str(e)}")
                # Hata durumunda varsayılan değerler kullan
                global_outcome = {
                    "is_surprise_button": is_surprise_button,
                    "expected_home_goals": 1.5,
                    "expected_away_goals": 1.0,
                    "score_prediction": {
                        "beklenen_home_gol": 1.5,
                        "beklenen_away_gol": 1.0
                    }
                }
        
        logger.info(f"İY/MS tahmini isteniyor: {home_name} vs {away_name}, is_surprise_button={is_surprise_button}")
        
        # Her iki takımın da ilk yarı/ikinci yarı istatistiklerini almamız gerekiyor
        # /api/v3/team/half-time-stats/{team_id} endpointini kullanıyoruz
        
        # Doğrudan get_team_half_time_stats fonksiyonunu çağıralım
        # yerel URL kullanmak yerine (connection refused hatası)
        try:
            # İç fonksiyonu doğrudan çağırarak yarı istatistiklerini alalım
            # team_id string olabilir, o yüzden int'e çevirelim
            home_team_id_int = int(home_team_id) if isinstance(home_team_id, str) and home_team_id.isdigit() else home_team_id
            home_stats = get_team_half_time_stats(home_team_id_int)
            # jsonify edilmiş cevabı düzgün bir dict'e çevirelim
            if hasattr(home_stats, 'json'):
                home_stats = home_stats.json
            
            home_team_data = {
                "id": home_team_id,
                "name": home_name,
                "stats": home_stats
            }
        except Exception as e:
            logger.warning(f"Ev sahibi takım için form verisi bulunamadı: team_id={home_team_id}, error={str(e)}")
            home_team_data = {
                "id": home_team_id,
                "name": home_name,
                "stats": {"error": "Veri işlenirken hata oluştu"}
            }
        
        try:
            # İç fonksiyonu doğrudan çağırarak yarı istatistiklerini alalım
            # team_id string olabilir, o yüzden int'e çevirelim
            away_team_id_int = int(away_team_id) if isinstance(away_team_id, str) and away_team_id.isdigit() else away_team_id
            away_stats = get_team_half_time_stats(away_team_id_int)
            # jsonify edilmiş cevabı düzgün bir dict'e çevirelim
            if hasattr(away_stats, 'json'):
                away_stats = away_stats.json
                
            away_team_data = {
                "id": away_team_id,
                "name": away_name,
                "stats": away_stats
            }
        except Exception as e:
            logger.warning(f"Deplasman takımı için form verisi bulunamadı: team_id={away_team_id}, error={str(e)}")
            away_team_data = {
                "id": away_team_id,
                "name": away_name,
                "stats": {"error": "Veri işlenirken hata oluştu"}
            }
            logger.warning(f"Deplasman takımı için form verisi bulunamadı: team_id={away_team_id}")
            return jsonify({
                "error": "Deplasman takımı istatistikleri alınamadı",
                "home_team": {
                    "id": home_team_id,
                    "name": home_name,
                    "stats": {"error": "Veri işlenirken hata oluştu"}
                },
                "away_team": {
                    "id": away_team_id,
                    "name": away_name,
                    "stats": {"error": "Bu takım için maç verisi bulunamadı"}
                },
                "htft_prediction": {"error": "Deplasman takımı istatistikleri alınamadı"}
            }), 202
        
        # Takım istatistiklerini kullan, artık home_team_data ve away_team_data'dan alıyoruz
        home_team_stats = home_team_data.get('stats', {})
        away_team_stats = away_team_data.get('stats', {})
        
        # Takım formlarını alma - match_prediction'dan
        from match_prediction import MatchPredictor
        predictor = MatchPredictor()
        home_team_form = predictor.get_team_form(home_team_id)
        away_team_form = predictor.get_team_form(away_team_id)
        
        logger.info(f"Form verileri elde edildi: Ev sahibi form={home_team_form is not None}, Deplasman form={away_team_form is not None}")
        
        # Takım-spesifik modellerden ayarlamalar için TeamSpecificPredictor sınıfı
        team_specific_predictor = None
        team_adjustments = None
        
        try:
            # team_specific_models modülünden TeamSpecificPredictor'ı içe aktar
            from team_specific_models import TeamSpecificPredictor
            team_specific_predictor = TeamSpecificPredictor()
            
            # Takım-spesifik ayarlamaları al
            team_adjustments = team_specific_predictor.get_team_adjustments(
                home_team_id, away_team_id, home_team_form, away_team_form
            )
            logger.info(f"Takım-spesifik ayarlamalar başarıyla elde edildi: {home_team_id} vs {away_team_id}")
            
        except Exception as e:
            logger.warning(f"Takım-spesifik modellerden ayarlamalar yapılırken hata: {str(e)}")
            
        # Tahmin butonundan gelen beklenen gol değerlerini al (eğer mevcutsa)
        expected_home_goals = None
        expected_away_goals = None
        
        # Geliştirilmiş beklenen gol ve sürpriz butonu kontrolü
        is_surprise_button = False
        match_outcome = None
        
        if global_outcome:
            if isinstance(global_outcome, dict):
                # Sürpriz butonu için mi?
                is_surprise_button = global_outcome.get("is_surprise_button", False)
                
                # Maç sonucu tahmini
                match_outcome = global_outcome.get("match_outcome")
                
                # Beklenen gol değerleri al (farklı formatları dene)
                expected_home_goals = global_outcome.get("expected_home_goals")
                expected_away_goals = global_outcome.get("expected_away_goals")
                
                # Eski API formatını da dene
                if expected_home_goals is None:
                    expected_home_goals = global_outcome.get("score_prediction", {}).get("beklenen_home_gol")
                if expected_away_goals is None:
                    expected_away_goals = global_outcome.get("score_prediction", {}).get("beklenen_away_gol")
                
                # JavaScript'ten gelen gol değerleri string olabilir, float'a çevir
                if expected_home_goals is not None and isinstance(expected_home_goals, str):
                    try:
                        expected_home_goals = float(expected_home_goals)
                    except (ValueError, TypeError):
                        expected_home_goals = 1.5
                        
                if expected_away_goals is not None and isinstance(expected_away_goals, str):
                    try:
                        expected_away_goals = float(expected_away_goals)
                    except (ValueError, TypeError):
                        expected_away_goals = 1.0
                        
                # Sıfır veya negatif değerler için düzeltme
                if expected_home_goals is not None and expected_home_goals <= 0:
                    expected_home_goals = 0.5
                if expected_away_goals is not None and expected_away_goals <= 0:
                    expected_away_goals = 0.5
            else:
                # String formatından (eski tip) maç sonucunu al
                match_outcome = global_outcome
        
        # Beklenen gol değerleri alındı mı?        
        if expected_home_goals is not None and expected_away_goals is not None:
            logger.info(f"Tahmin butonundan beklenen gol değerleri alındı: Ev={expected_home_goals:.2f}, Deplasman={expected_away_goals:.2f}, Sürpriz={is_surprise_button}")
            
            # Düşük gol beklentisi kontrolü
            total_expected_goals = expected_home_goals + expected_away_goals
            if total_expected_goals < 1.2:
                logger.info(f"DÜŞÜK GOL BEKLENTİSİ TESPİT EDİLDİ: Toplam={total_expected_goals:.2f}")
                
            # Deplasman avantajlı düşük skorlu maç mı?
            if expected_away_goals > expected_home_goals + 0.3 and total_expected_goals < 2.0:
                logger.info(f"DEPLASMAN AVANTAJLI DÜŞÜK SKORLU MAÇ: Ev={expected_home_goals:.2f}, Deplasman={expected_away_goals:.2f}")
                # global_outcome nesnesine maç sonucu ekle (halfTime_fullTime_predictor için)
                if isinstance(global_outcome, dict) and match_outcome is None:
                    global_outcome["match_outcome"] = "AWAY_WIN"
        
        # İY/MS tahminini hesapla - form verilerini ve takım-spesifik ayarlamaları geçirerek
        htft_prediction = halfTime_fullTime_predictor.predict_half_time_full_time(
            home_team_stats,
            away_team_stats,
            global_outcome,
            home_form=home_team_form,        # Form verilerini ekle
            away_form=away_team_form,        # Form verilerini ekle
            home_team_id=home_team_id,       # Takım ID'lerini ekle
            away_team_id=away_team_id,       # Takım ID'lerini ekle
            team_adjustments=team_adjustments # Takım-spesifik ayarlamaları ekle
        )
        
        # Diğer maç bilgilerini ekle (halihazırda başlamış maç ise)
        scores = {
            "half_time": {"home": 0, "away": 0},
            "full_time": {"home": 0, "away": 0},
            "second_half": {"home": 0, "away": 0}
        }
        
        # Sürpriz tahminler için özel içgörüler ekle - 1/2 ve 2/1 tahminleri
        special_insight = None
        surprise_predictions = {
            "1/2": htft_prediction.get("all_probabilities", {}).get("1/2", 0),
            "2/1": htft_prediction.get("all_probabilities", {}).get("2/1", 0)
        }
        
        # Yüksek olasılıklı 1/2 tahmini
        if surprise_predictions["1/2"] > 15:  # %15'den yüksek olasılık
            special_insight = {
                "type": "1/2",
                "title": "Sürpriz Tahmin: İlk Yarı Ev Önde / Maç Sonu Deplasman Kazanır",
                "description": "Ev sahibi takımın ilk yarıda öne geçme eğilimi, ancak deplasman takımının ikinci yarı performansı güçlü.",
                "probability": surprise_predictions["1/2"],
                "explanation": "Ev sahibi takım ilk yarıda formda, deplasman takımı ikinci yarıda daha etkili oyun sergiliyor."
            }
            logger.info(f"Yüksek 1/2 olasılığı tespit edildi: %{surprise_predictions['1/2']}")
            
        # Yüksek olasılıklı 2/1 tahmini
        elif surprise_predictions["2/1"] > 15:  # %15'den yüksek olasılık
            special_insight = {
                "type": "2/1",
                "title": "Sürpriz Tahmin: İlk Yarı Deplasman Önde / Maç Sonu Ev Kazanır",
                "description": "Deplasman takımının ilk yarıda öne geçme eğilimi, ancak ev sahibi takımın ikinci yarı performansı güçlü.",
                "probability": surprise_predictions["2/1"],
                "explanation": "Deplasman takımı ilk yarıda formda, ev sahibi takımı ikinci yarıda daha etkili oyun sergiliyor."
            }
            logger.info(f"Yüksek 2/1 olasılığı tespit edildi: %{surprise_predictions['2/1']}")
        
        # Response objelerini JSON'a dönüştürelim
        # Eğer home_team_stats ve away_team_stats Response objesi ise, düzeltelim
        try:
            # Önce home_team_stats kontrolü
            if hasattr(home_team_stats, 'json'):
                try:
                    home_team_stats = home_team_stats.json()
                except Exception:
                    home_team_stats = {"error": "Veri işlenirken hata oluştu"}
            
            # Sonra away_team_stats kontrolü
            if hasattr(away_team_stats, 'json'):
                try:
                    away_team_stats = away_team_stats.json()
                except Exception:
                    away_team_stats = {"error": "Veri işlenirken hata oluştu"}
                    
            # Sonuç oluştur
            result = {
                "home_team": {
                    "id": home_team_id,
                    "name": home_name,
                    "stats": home_team_stats
                },
                "away_team": {
                    "id": away_team_id,
                    "name": away_name,
                    "stats": away_team_stats
                },
                "htft_prediction": htft_prediction,
                "scores": scores,
                "special_insight": special_insight  # Sürpriz tahmin içgörüsü ekle
            }
            
            # JSON serileştirilebilirlik kontrolü
            from flask.json import dumps
            try:
                # Deneme amaçlı serileştirme
                dumps(result)
            except TypeError as json_error:
                logger.error(f"JSON serileştirme hatası: {str(json_error)}")
                # Daha basit bir yanıt döndür
                result = {
                    "home_team": {
                        "id": home_team_id,
                        "name": home_name,
                        "stats": {"error": "Veri işlenirken hata oluştu"}
                    },
                    "away_team": {
                        "id": away_team_id,
                        "name": away_name,
                        "stats": {"error": "Veri işlenirken hata oluştu"}
                    },
                    "htft_prediction": {
                        "prediction": "İY/MS tahmini oluşturulamadı",
                        "probabilities": {}
                    }
                }
        except Exception as result_error:
            logger.error(f"Sonuç oluşturulurken hata: {str(result_error)}")
            # En basit yanıt
            result = {
                "error": "Sonuç oluşturulurken hata",
                "message": str(result_error)
            }
            
        logger.info(f"İY/MS tahmini başarıyla oluşturuldu: {home_name} vs {away_name}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"İY/MS tahmini oluşturulurken hata: {str(e)}", exc_info=True)
        # home_name ve away_name değişkenleri zaten fonksiyon başında tanımlandı
        # Böylece exception durumunda da erişilebilir olacak
        
        return jsonify({
            "error": f"İY/MS tahmini oluşturulurken hata: {str(e)}",
            "home_team": {
                "id": home_team_id,
                "name": home_name,
                "stats": {"error": "Veri işlenirken hata oluştu"}
            },
            "away_team": {
                "id": away_team_id,
                "name": away_name,
                "stats": {"error": "Veri işlenirken hata oluştu"}
            },
            "htft_prediction": {"error": str(e)}
        }), 500

@api_v3_bp.route('/team-form-analysis/<team_id>')
@api_cache(timeout=1800)  # 30 dakika önbellek süresi
def team_form_analysis(team_id):
    """
    Takımın form analizi ve detaylı istatistiklerini döndüren API endpoint
    
    Frontend'den JavaScript hesaplamalarını backend'e taşıyan bu endpoint,
    takımların istatistik ve form hesaplamalarını Python tarafında yapar.
    """
    try:
        from match_prediction import MatchPredictor
        
        team_name = request.args.get('team_name', 'Bilinmeyen Takım')
        
        # MatchPredictor sınıfını başlat
        predictor = MatchPredictor()
        
        # Takım formunu hesapla
        team_form = predictor.get_team_form(team_id)
        
        if not team_form:
            return jsonify({
                "error": "Takım formu hesaplanamadı, veriler eksik olabilir",
                "team_id": team_id,
                "team_name": team_name
            }), 400
        
        # Form periyotlarını hesapla
        last_5_form = {
            "win_ratio": 0,
            "draw_ratio": 0,
            "loss_ratio": 0,
            "total_matches": 0
        }
        
        last_10_form = {
            "win_ratio": 0,
            "draw_ratio": 0,
            "loss_ratio": 0,
            "total_matches": 0
        }
        
        try:
            # Son 5 ve 10 maçtaki form istatistiklerini hesapla
            if 'recent_match_data' in team_form:
                recent_matches = team_form['recent_match_data'][:10]
                
                # Son 5 maç
                if len(recent_matches) >= 5:
                    last_5_matches = recent_matches[:5]
                    wins_5 = sum(1 for m in last_5_matches if m.get('result') == 'W')
                    draws_5 = sum(1 for m in last_5_matches if m.get('result') == 'D')
                    losses_5 = sum(1 for m in last_5_matches if m.get('result') == 'L')
                    total_5 = len(last_5_matches)
                    
                    last_5_form = {
                        "win_ratio": round(wins_5 / total_5 * 100, 2) if total_5 > 0 else 0,
                        "draw_ratio": round(draws_5 / total_5 * 100, 2) if total_5 > 0 else 0,
                        "loss_ratio": round(losses_5 / total_5 * 100, 2) if total_5 > 0 else 0,
                        "total_matches": total_5,
                        "wins": wins_5,
                        "draws": draws_5,
                        "losses": losses_5
                    }
                
                # Son 10 maç
                if len(recent_matches) > 0:
                    wins_10 = sum(1 for m in recent_matches if m.get('result') == 'W')
                    draws_10 = sum(1 for m in recent_matches if m.get('result') == 'D')
                    losses_10 = sum(1 for m in recent_matches if m.get('result') == 'L')
                    total_10 = len(recent_matches)
                    
                    last_10_form = {
                        "win_ratio": round(wins_10 / total_10 * 100, 2) if total_10 > 0 else 0,
                        "draw_ratio": round(draws_10 / total_10 * 100, 2) if total_10 > 0 else 0,
                        "loss_ratio": round(losses_10 / total_10 * 100, 2) if total_10 > 0 else 0,
                        "total_matches": total_10,
                        "wins": wins_10,
                        "draws": draws_10,
                        "losses": losses_10
                    }
        except Exception as e:
            logger.warning(f"Form periyotları hesaplanırken hata: {str(e)}")
        
        # Gol istatistiklerini hesapla
        goal_stats = {
            "home": {
                "scored": 0,
                "conceded": 0,
                "average_scored": 0,
                "average_conceded": 0,
                "matches": 0
            },
            "away": {
                "scored": 0,
                "conceded": 0,
                "average_scored": 0,
                "average_conceded": 0,
                "matches": 0
            },
            "overall": {
                "scored": 0,
                "conceded": 0,
                "average_scored": 0,
                "average_conceded": 0,
                "matches": 0
            }
        }
        
        try:
            # Ev sahibi ve deplasman gol istatistiklerini hesapla
            home_matches = [m for m in team_form.get('recent_match_data', []) if m.get('is_home', False)]
            away_matches = [m for m in team_form.get('recent_match_data', []) if not m.get('is_home', False)]
            
            # Ev sahibi istatistikleri
            if home_matches:
                home_goals_scored = sum(m.get('goals_scored', 0) for m in home_matches)
                home_goals_conceded = sum(m.get('goals_conceded', 0) for m in home_matches)
                home_matches_count = len(home_matches)
                
                goal_stats["home"] = {
                    "scored": home_goals_scored,
                    "conceded": home_goals_conceded,
                    "average_scored": round(home_goals_scored / home_matches_count, 2) if home_matches_count > 0 else 0,
                    "average_conceded": round(home_goals_conceded / home_matches_count, 2) if home_matches_count > 0 else 0,
                    "matches": home_matches_count
                }
            
            # Deplasman istatistikleri
            if away_matches:
                away_goals_scored = sum(m.get('goals_scored', 0) for m in away_matches)
                away_goals_conceded = sum(m.get('goals_conceded', 0) for m in away_matches)
                away_matches_count = len(away_matches)
                
                goal_stats["away"] = {
                    "scored": away_goals_scored,
                    "conceded": away_goals_conceded,
                    "average_scored": round(away_goals_scored / away_matches_count, 2) if away_matches_count > 0 else 0,
                    "average_conceded": round(away_goals_conceded / away_matches_count, 2) if away_matches_count > 0 else 0,
                    "matches": away_matches_count
                }
            
            # Genel istatistikler
            all_matches = team_form.get('recent_match_data', [])
            if all_matches:
                total_goals_scored = sum(m.get('goals_scored', 0) for m in all_matches)
                total_goals_conceded = sum(m.get('goals_conceded', 0) for m in all_matches)
                total_matches_count = len(all_matches)
                
                goal_stats["overall"] = {
                    "scored": total_goals_scored,
                    "conceded": total_goals_conceded,
                    "average_scored": round(total_goals_scored / total_matches_count, 2) if total_matches_count > 0 else 0,
                    "average_conceded": round(total_goals_conceded / total_matches_count, 2) if total_matches_count > 0 else 0,
                    "matches": total_matches_count
                }
        except Exception as e:
            logger.warning(f"Gol istatistikleri hesaplanırken hata: {str(e)}")
        
        # Sonucu derle
        result = {
            "team_id": team_id,
            "team_name": team_name,
            "form": team_form,
            "form_periods": {
                "last_5": last_5_form,
                "last_10": last_10_form
            },
            "goal_stats": goal_stats
        }
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Takım form analizi yapılırken hata: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            "error": f"Takım form analizi yapılırken hata: {str(e)}",
            "team_id": team_id,
            "team_name": request.args.get('team_name', 'Bilinmeyen Takım')
        }), 500

@api_v3_bp.route('/advanced-predictions/<home_team_id>/<away_team_id>')
@api_cache(timeout=900)  # 15 dakika önbellek süresi
def advanced_predictions(home_team_id, away_team_id):
    """Gelişmiş tahminleri döndüren API endpoint"""
    try:
        from zip_and_ensemble_predictor import AdvancedScorePredictor
        from main import predictor
        
        home_name = request.args.get('home_name', '')
        away_name = request.args.get('away_name', '')
        
        # Önce tahmin verilerini al
        basic_prediction = predictor.predict_match(
            home_team_id, away_team_id, 
            home_name, away_name,
            use_goal_trend_analysis=True
        )
        
        if basic_prediction and basic_prediction.get('home_form') and basic_prediction.get('away_form'):
            # Gelişmiş tahmin modellerini oluştur
            advanced_predictor = AdvancedScorePredictor()
            
            # Gelişmiş tahminleri hesapla
            advanced_predictions = advanced_predictor.predict_match(
                basic_prediction['home_form'], 
                basic_prediction['away_form']
            )
            
            # İki tahmin setini birleştir
            response = {
                "basic_predictions": basic_prediction,
                "advanced_predictions": advanced_predictions
            }
            
            return jsonify(response)
        else:
            return jsonify({"error": "Basic prediction data not available"}), 400
            
    except Exception as e:
        logger.error(f"Error getting advanced predictions: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
# Model Doğrulama ve Değerlendirme için yeni API rotaları
@api_v3_bp.route('/validation/cross-validate', methods=['GET', 'POST'])
def run_cross_validation():
    """Çapraz doğrulama gerçekleştirme API endpoint'i"""
    try:
        from main import model_validator
        
        # İsteğe bağlı parametreleri al
        if request.method == 'POST':
            data = request.get_json() or {}
        else:  # GET metodu
            data = {}
            
        k_folds = data.get('k_folds', 5)
        random_state = data.get('random_state', 42)
        max_days = data.get('max_days', 180)  # Varsayılan olarak 180 gün
        use_time_weights = data.get('use_time_weights', True)  # Varsayılan olarak zaman ağırlıklı
        
        # Parametreleri doğrula
        if not isinstance(k_folds, int) or k_folds < 2 or k_folds > 10:
            return jsonify({"error": "k_folds 2 ile 10 arasında bir tam sayı olmalıdır"}), 400
            
        if not isinstance(max_days, int) or max_days < 1:
            return jsonify({"error": "max_days pozitif bir tam sayı olmalıdır"}), 400
            
        # Çapraz doğrulama gerçekleştir
        result = model_validator.cross_validate(
            k_folds=k_folds, 
            random_state=random_state,
            use_time_weights=use_time_weights,
            max_days=max_days
        )
        
        # NumPy değerlerini Python'a dönüştür
        result = numpy_to_python(result)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Çapraz doğrulama gerçekleştirilirken hata: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_v3_bp.route('/validation/backtesting', methods=['POST'])
def run_backtesting():
    """Geliştirilmiş geriye dönük test gerçekleştirme API endpoint'i"""
    try:
        from main import model_validator
        
        # İsteğe bağlı parametreleri al
        data = request.get_json() or {}
        days_back = data.get('days_back', 180)  # Varsayılan olarak 180 gün
        test_ratio = data.get('test_ratio', 0.25)  # Eğitim için %75, test için %25
        season_weight_factor = data.get('season_weight_factor', 1.5)  # Mevcut sezon için ağırlık faktörü
        time_based_split = data.get('time_based_split', True)  # Zamansal bölünme kullanımı
        validation_type = data.get('validation_type', 'standard')  # Doğrulama tipi
        ensemble_type = data.get('ensemble_type', 'weighted')  # Ensemble tipi
        
        # Parametreleri doğrula
        if not isinstance(days_back, int) or days_back < 7 or days_back > 365:
            return jsonify({"error": "days_back 7 ile 365 arasında bir tam sayı olmalıdır"}), 400
            
        if not isinstance(test_ratio, (int, float)) or test_ratio <= 0 or test_ratio >= 1:
            return jsonify({"error": "test_ratio 0 ile 1 arasında bir ondalık sayı olmalıdır"}), 400
        
        if not isinstance(season_weight_factor, (int, float)) or season_weight_factor < 0:
            return jsonify({"error": "season_weight_factor 0'dan büyük bir sayı olmalıdır"}), 400
            
        # Doğrulama tipini kontrol et
        valid_validation_types = ['standard', 'rolling', 'expanding', 'nested']
        if validation_type not in valid_validation_types:
            return jsonify({"error": f"validation_type şu değerlerden biri olmalıdır: {', '.join(valid_validation_types)}"}), 400
            
        # Ensemble tipini kontrol et
        valid_ensemble_types = ['voting', 'weighted', 'stacking', 'tuned', 'blending']
        if ensemble_type not in valid_ensemble_types:
            return jsonify({"error": f"ensemble_type şu değerlerden biri olmalıdır: {', '.join(valid_ensemble_types)}"}), 400
            
        # Geliştirilmiş geriye dönük test gerçekleştir
        result = model_validator.backtesting(
            days_back=days_back, 
            test_ratio=test_ratio,
            season_weight_factor=season_weight_factor,
            time_based_split=time_based_split,
            validation_type=validation_type,
            ensemble_type=ensemble_type
        )
        
        # NumPy değerlerini Python'a dönüştür
        result = numpy_to_python(result)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Geriye dönük test gerçekleştirilirken hata: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_v3_bp.route('/validation/results', methods=['GET'])
@api_v3_bp.route('/run_cross_validation', methods=['GET'])  # Eski API rotaları için geriye uyumluluk
@api_cache(timeout=3600)  # 1 saat önbellek süresi (doğrulama sonuçları sık değişmez)
def get_validation_results():
    """Doğrulama sonuçlarını döndüren API endpoint'i"""
    try:
        from main import model_validator
        
        # İsteğe bağlı parametreleri al
        result_type = request.args.get('type', 'all')
        count = int(request.args.get('count', 5))
        
        # Parametreleri doğrula
        if result_type not in ['cross_validation', 'backtesting', 'all']:
            return jsonify({"error": "type parametresi 'cross_validation', 'backtesting' veya 'all' olmalıdır"}), 400
            
        if count < 1 or count > 50:
            return jsonify({"error": "count parametresi 1 ile 50 arasında bir tam sayı olmalıdır"}), 400
            
        # Sonuçları getir
        results = model_validator.get_latest_results(result_type=result_type, count=count)
        
        # NumPy değerlerini Python'a dönüştür
        results = numpy_to_python(results)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Doğrulama sonuçları alınırken hata: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_v3_bp.route('/validation/report', methods=['GET'])
@api_cache(timeout=3600)  # 1 saat önbellek süresi (doğrulama raporları sık değişmez)
def get_validation_report():
    """Doğrulama raporu döndüren API endpoint'i"""
    try:
        from main import model_validator
        
        # Rapor oluştur
        report = model_validator.generate_validation_report()
        
        # NumPy değerlerini Python'a dönüştür
        report = numpy_to_python(report)
        return jsonify(report)
    except Exception as e:
        logger.error(f"Doğrulama raporu oluşturulurken hata: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@api_v3_bp.route('/validation/feature-importance-analysis', methods=['GET'])
@api_cache(timeout=3600)  # 1 saat önbellek süresi
def get_feature_importance_analysis():
    """Özellik önem dereceleri analizi döndüren API endpoint'i"""
    try:
        from main import model_validator
        
        # Özellik önem derecelerini analiz et
        importance_data = model_validator.analyze()
        
        if not importance_data:
            return jsonify({
                'status': 'error',
                'message': 'Özellik önemi analizi için yeterli veri bulunamadı',
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # NumPy değerlerini Python'a dönüştür
        importance_data = numpy_to_python(importance_data)
        
        return jsonify({
            'status': 'success',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'importance': importance_data
        })
    except Exception as e:
        logger.error(f"Özellik önemi analizi alınırken hata: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_v3_bp.route('/validation/ensemble', methods=['POST'])
def run_ensemble_validation():
    """Gelişmiş ensemble modelleri ile çapraz doğrulama gerçekleştiren API endpoint'i"""
    try:
        from main import model_validator
        
        # POST verisini al
        data = request.get_json(silent=True) or {}
        ensemble_type = data.get('ensemble_type', 'stacking')  # 'voting', 'stacking', 'blending'
        k_folds = int(data.get('k_folds', 5))
        random_state = int(data.get('random_state', 42))
        
        # Kısıtlamaları kontrol et
        if ensemble_type not in ['voting', 'stacking', 'blending']:
            return jsonify({
                "error": "Geçersiz ensemble tipi. 'voting', 'stacking' veya 'blending' olmalıdır."
            }), 400
            
        if k_folds < 2 or k_folds > 10:
            return jsonify({
                "error": "k_folds 2 ile 10 arasında olmalıdır."
            }), 400
        
        # Zaman ağırlıkları parametreleri kontrol et
        use_time_weights = data.get('use_time_weights', True)
        max_days = data.get('max_days', 180)
        
        # Ensemble çapraz doğrulama çalıştır
        result = model_validator.ensemble_cross_validate(
            ensemble_type=ensemble_type,
            k_folds=k_folds,
            random_state=random_state,
            use_time_weights=use_time_weights,
            max_days=max_days
        )
        
        # NaN değerleri temizle
        import math
        if 'metrics' in result and 'ensemble_improvement' in result['metrics'] and isinstance(result['metrics']['ensemble_improvement'], dict):
            improvement = result['metrics']['ensemble_improvement']
            for key in list(improvement.keys()):
                if isinstance(improvement[key], (int, float)) and math.isnan(improvement[key]):
                    improvement[key] = 0.0
        
        # NumPy değerlerini Python'a dönüştür
        result = numpy_to_python(result)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Ensemble doğrulama yapılırken hata: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
        
@api_v3_bp.route('/validation/hyperparameter-tuning', methods=['POST'])
def run_hyperparameter_tuning():
    """Hiperparametre optimizasyonu gerçekleştiren API endpoint'i"""
    try:
        from main import model_validator
        
        # POST verisini al
        data = request.get_json(silent=True) or {}
        model_type = data.get('model_type', 'rf')  # 'rf', 'gbm', 'linear'
        cv = int(data.get('cv', 5))
        scoring = data.get('scoring', 'neg_mean_squared_error')
        
        # Param grid
        param_grid = data.get('param_grid', None)
        
        # Kısıtlamaları kontrol et
        if model_type not in ['rf', 'gbm', 'linear']:
            return jsonify({
                "error": "Geçersiz model tipi. 'rf', 'gbm' veya 'linear' olmalıdır."
            }), 400
            
        if cv < 2 or cv > 10:
            return jsonify({
                "error": "cv 2 ile 10 arasında olmalıdır."
            }), 400
        
        # Zaman ağırlıkları parametreleri kontrol et
        use_time_weights = data.get('use_time_weights', True)
        max_days = data.get('max_days', 180)
        
        # Hiperparametre optimizasyonu çalıştır
        result = model_validator.optimize_hyperparameters(
            model_type=model_type,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            use_time_weights=use_time_weights,
            max_days=max_days
        )
        # NumPy değerlerini Python'a dönüştür
        result = numpy_to_python(result)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Hiperparametre optimizasyonu yapılırken hata: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@api_v3_bp.route('/validation/feature-importance', methods=['POST'])
def get_feature_importance():
    """Özellik önem derecelerini döndüren API endpoint'i"""
    try:
        from main import model_validator
        
        # POST verisini al
        data = request.get_json(silent=True) or {}
        model_type = data.get('model_type', 'rf')  # 'rf', 'gbm', 'ensemble'
        ensemble_type = data.get('ensemble_type', 'stacking')  # ensemble için
        k_folds = int(data.get('k_folds', 5))
        
        # Kısıtlamaları kontrol et
        if model_type not in ['rf', 'gbm', 'ensemble']:
            return jsonify({
                "error": "Geçersiz model tipi. 'rf', 'gbm' veya 'ensemble' olmalıdır."
            }), 400
        
        # Özellik önem analizi yapıldı
        result = {
            'status': 'success',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': model_type
        }
        
        # Zaman ağırlıkları parametreleri kontrol et
        use_time_weights = data.get('use_time_weights', True)
        max_days = data.get('max_days', 180)
        
        # Önce veriyi hazırla
        df = model_validator._prepare_data_from_cache(use_time_weights=use_time_weights, max_days=max_days)
        if df.empty:
            return jsonify({
                'status': 'error',
                'message': 'Yeterli veri bulunamadı',
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }), 400
        
        # Model tipine göre işlem yap
        if model_type == 'ensemble':
            # Ensemble model oluştur ve çapraz doğrulama ile eğit
            ensemble_result = model_validator.ensemble_cross_validate(
                ensemble_type="voting",  # Küçük veri setleri için voting ensemble kullan
                k_folds=k_folds
            )
            
            # Eğer blending ise özellik önemleri var
            if ensemble_type == 'blending':
                result['ensemble_type'] = ensemble_type
                result['ensemble_result'] = ensemble_result
                # Özellik önemleri burada gelecek
                
            else:
                # Voting ve stacking için hızlı bir model eğit
                X_home, y_home, X_away, y_away = model_validator._prepare_features_for_ensemble(df)
                
                home_model = model_validator._create_ensemble_model(ensemble_type)
                home_model.fit(X_home, y_home)
                
                away_model = model_validator._create_ensemble_model(ensemble_type)
                away_model.fit(X_away, y_away)
                
                # Feature importance analizi
                # Sadece base modellerin feature importance değerleri çıkarılabilir
                if hasattr(home_model, 'estimators_'):
                    home_importances = {}
                    away_importances = {}
                    
                    for i, (name, _) in enumerate(home_model.estimators):
                        estimator = home_model.estimators_[i]
                        if hasattr(estimator, 'feature_importances_'):
                            home_importances[name] = {
                                'importances': estimator.feature_importances_.tolist(),
                                'features': X_home.columns.tolist()
                            }
                    
                    for i, (name, _) in enumerate(away_model.estimators):
                        estimator = away_model.estimators_[i]
                        if hasattr(estimator, 'feature_importances_'):
                            away_importances[name] = {
                                'importances': estimator.feature_importances_.tolist(),
                                'features': X_away.columns.tolist()
                            }
                    
                    result['feature_importances'] = {
                        'home': home_importances,
                        'away': away_importances
                    }
                
                result['ensemble_type'] = ensemble_type
                
        else:  # rf veya gbm
            # Basit model oluştur ve eğit
            X_home, y_home, X_away, y_away = model_validator._prepare_features_for_ensemble(df)
            
            if model_type == 'rf':
                from sklearn.ensemble import RandomForestRegressor
                home_model = RandomForestRegressor(n_estimators=100, random_state=42)
                away_model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:  # gbm
                from sklearn.ensemble import GradientBoostingRegressor
                home_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                away_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            
            home_model.fit(X_home, y_home)
            away_model.fit(X_away, y_away)
            
            # Feature importance analizi
            result['feature_importances'] = {
                'home': {
                    'importances': home_model.feature_importances_.tolist(),
                    'features': X_home.columns.tolist()
                },
                'away': {
                    'importances': away_model.feature_importances_.tolist(),
                    'features': X_away.columns.tolist()
                }
            }
        
        # NumPy değerlerini Python'a dönüştür
        result = numpy_to_python(result)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Özellik önem derecesi analizi yapılırken hata: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
        
@api_v3_bp.route('/validation/hyperparameter-optimization', methods=['POST'])
def hyperparameter_optimization():
    """Gelişmiş hiperparametre optimizasyonu gerçekleştirir"""
    try:
        from main import model_validator
        
        # POST verisini al - hata durumunda varsayılan değerler kullan
        try:
            data = request.get_json(silent=True) or {}
        except Exception as json_err:
            logger.error(f"JSON verisi işlenirken hata: {str(json_err)}")
            # Bağlantı veya istemci hatası durumunda varsayılan değerler kullan
            data = {}
        
        # Parametreleri al - Daha az hesaplama gerektiren değerler kullan
        model_type = data.get('model_type', 'rf')
        cv = data.get('cv', 3)  # 5 yerine 3 kat çapraz doğrulama (daha hızlı)
        scoring = data.get('scoring', 'neg_mean_squared_error')
        use_time_weights = data.get('use_time_weights', True)
        max_days = data.get('max_days', 180)
        search_type = data.get('search_type', 'random')  # RandomSearch öntanımlı (GridSearch çok yavaş)
        n_iter = data.get('n_iter', 10)  # 20 yerine 10 iterasyon (daha hızlı)
        season_weight_factor = data.get('season_weight_factor', 1.5)
        
        # İlerleme bildirimi
        logger.info(f"Hiperparametre optimizasyonu başlatılıyor: {model_type} modeli, {search_type} arama, {cv} kat doğrulama")
        
        # Model tipini kontrol et
        if model_type not in ['rf', 'gbm', 'xgb', 'linear', 'elasticnet']:
            return jsonify({
                "error": f"Geçersiz model tipi: {model_type}. Desteklenen tipler: 'rf', 'gbm', 'xgb', 'linear', 'elasticnet'"
            }), 400
            
        # Arama tipini kontrol et
        if search_type not in ['grid', 'random']:
            return jsonify({
                "error": f"Geçersiz arama tipi: {search_type}. Desteklenen tipler: 'grid', 'random'"
            }), 400
        
        # Optimizasyonu çalıştır
        result = model_validator.optimize_hyperparameters(
            model_type=model_type,
            cv=cv,
            scoring=scoring,
            use_time_weights=use_time_weights,
            max_days=max_days,
            search_type=search_type,
            n_iter=n_iter,
            season_weight_factor=season_weight_factor
        )
        
        # NumPy değerlerini Python'a dönüştür
        result = numpy_to_python(result)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Hiperparametre optimizasyonu sırasında hata: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
        
@api_v3_bp.route('/validation/detailed-report', methods=['GET'])
def get_detailed_validation_report():
    """Gelişmiş model doğrulama raporu oluşturur"""
    try:
        from main import model_validator
        
        # Raporu oluştur
        report = model_validator.generate_validation_report()
        
        # NumPy değerlerini Python'a dönüştür
        report = numpy_to_python(report)
        return jsonify(report)
    except Exception as e:
        logger.error(f"Doğrulama raporu oluşturulurken hata: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500