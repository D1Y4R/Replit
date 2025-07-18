import logging
import os
import requests
import threading
import socket
from datetime import datetime, timedelta
import pytz
from flask import Flask, render_template, jsonify, request, flash, redirect, url_for
from flask_caching import Cache
from match_prediction import MatchPredictor
# Create and load api_routes only after setting up the Flask app
# This avoids circular imports
api_v3_bp = None  # Will be set after app creation
from model_validation import ModelValidator
from dynamic_team_analyzer import DynamicTeamAnalyzer
from team_performance_updater import TeamPerformanceUpdater
from self_learning_predictor import SelfLearningPredictor

# Global değişkenler - Modüller arası paylaşım için
team_analyzer = None
self_learning_model = None
performance_updater = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")

# Flask-Caching konfigürasyonu
cache_config = {
    "CACHE_TYPE": "SimpleCache",  # Basit bellek içi önbellek
    "CACHE_DEFAULT_TIMEOUT": 300,  # Varsayılan 5 dakika (300 saniye) önbellek süresi
    "CACHE_THRESHOLD": 500,        # Maksimum önbellek öğe sayısı
}
cache = Cache(app, config=cache_config)

# API Blueprint'leri kaydet - moved below
# api_v3_bp will be imported after app creation

# Tahmin modelini oluştur
predictor = MatchPredictor()

# Model doğrulama ve değerlendirme için validator oluştur
model_validator = ModelValidator(predictor)

def get_matches(selected_date=None):
    try:
        # Create timezone objects
        utc = pytz.UTC
        turkey_tz = pytz.timezone('Europe/Istanbul')

        if not selected_date:
            selected_date = datetime.now().strftime('%Y-%m-%d')

        matches = []
        api_key = '39bc8dd65153ff5c7c0f37b4939009de04f4a70c593ee1aea0b8f70dd33268c0'

        # Get matches from APIFootball
        url = "https://apiv3.apifootball.com/"
        params = {
            'action': 'get_events',
            'APIkey': api_key,
            'from': selected_date,
            'to': selected_date,
            'timezone': 'Europe/Istanbul'
        }
        logger.info(f"Sending API request to {url} with params: {params}")

        logger.info(f"Fetching matches for date: {selected_date}")
        response = requests.get(url, params=params)
        logger.debug(f"API Response status: {response.status_code}")
        logger.debug(f"API Response content: {response.content}") # Added logging for debugging

        if response.status_code == 200:
            data = response.json()
            logger.info(f"API response received. Type: {type(data)}")
            if data == []:
                logger.warning("API returned empty data array")

            if isinstance(data, list):
                logger.info(f"Total matches in API response: {len(data)}")
                for match in data:
                    match_obj = process_match(match, utc, turkey_tz)
                    if match_obj:
                        matches.append(match_obj)
                        logger.debug(f"Added match: {match_obj['competition']['name']} - {match_obj['homeTeam']['name']} vs {match_obj['awayTeam']['name']}")
            elif isinstance(data, dict):
                logger.error(f"API returned error: {data.get('message', 'Unknown error')}")

        # Group matches by league
        league_matches = {}
        for match in matches:
            league_id = match['competition']['id']
            league_name = match['competition']['name']

            if league_id not in league_matches:
                league_matches[league_id] = {
                    'name': league_name,
                    'matches': []
                }
            league_matches[league_id]['matches'].append(match)

        # Sort matches within each league
        for league_data in league_matches.values():
            league_data['matches'].sort(key=lambda x: (
                0 if x['is_live'] else (1 if x['status'] == 'FINISHED' else 2),
                x['turkish_time']
            ))

        # Format leagues for template
        formatted_leagues = []
        for league_id, league_data in league_matches.items():
            formatted_leagues.append({
                'id': league_id,
                'name': league_data['name'],
                'matches': league_data['matches'],
                'priority': get_league_priority(league_id)
            })

        # Sort leagues by priority (high to low) and then by name
        formatted_leagues.sort(key=lambda x: (-x['priority'], x['name']))

        logger.info(f"Total leagues found: {len(formatted_leagues)}")
        for league in formatted_leagues:
            logger.info(f"League: {league['name']} - {len(league['matches'])} matches")

        return {'leagues': formatted_leagues}

    except Exception as e:
        logger.error(f"Error fetching matches: {str(e)}")
        return {'leagues': []}

def get_league_priority(league_id):
    """Return priority for league sorting. Higher number means higher priority."""

    # Convert league_id to string for comparison
    league_id_str = str(league_id)

    # Favorite leagues with their IDs from API-Football
    favorite_leagues = {
        "3": 100,    # UEFA Champions League
        "4": 90,     # UEFA Europa League
        "683": 80,   # UEFA Conference League
        "302": 70,   # La Liga
        "152": 65,   # Premier League
        "207": 60,   # Serie A
        "175": 55,   # Bundesliga
        "168": 50,   # Ligue 1
        "322": 45,   # Türk Süper Lig
        "266": 25,   # Primeira Liga
        "128": 40,   # Gana Premier Lig
        "567": 39,   # Brezilya Série A
        "164": 38,   # Hollanda Eredivisie
        "358": 37,   # Arjantin Primera División
        "196": 36,   # İskoçya Premiership
        "179": 35,   # İsviçre Süper Ligi
        "144": 34,   # Belçika Pro League
        "182": 33    # Portekiz Primeira Liga
    }

    # Doğrudan ID ile kontrol et
    if league_id_str in favorite_leagues:
        return favorite_leagues[league_id_str]

    return 0

def process_match(match, utc, turkey_tz):
    try:
        # Get team names
        home_name = match.get('match_hometeam_name', '')
        away_name = match.get('match_awayteam_name', '')

        if not home_name or not away_name:
            return None

        # Get match time and convert to Turkish time
        match_date = match.get('match_date', '')
        match_time = match.get('match_time', '')
        league_name = match.get('league_name', '')

        # Log raw API response for debugging
        logger.info(f"Raw API match data for {home_name} vs {away_name}:")
        logger.info(f"Match date: {match_date}")
        logger.info(f"Match time: {match_time}")

        turkish_time_str = "Belirlenmedi"
        try:
            if match_date and match_time and match_time != "00:00":
                # API'den gelen zamanı doğrudan kullan, çünkü params 'timezone': 'Europe/Istanbul' zaten ayarlanmış
                turkish_time_str = match_time
                
                logger.info(f"Time conversion details for {home_name} vs {away_name}:")
                logger.info(f"Original time (from API): {match_time}")
                logger.info(f"Using as Turkish time (TSİ): {turkish_time_str}")

        except ValueError as e:
            logger.error(f"Time conversion error: {e}")
            logger.error(f"Input date={match_date}, time={match_time}")

        # Get match status and scores
        match_status = match.get('match_status', '')
        match_live = match.get('match_live', '0')
        home_score = '0'
        away_score = '0'
        is_live = False
        live_minute = ''

        if match_status == 'Finished':
            home_score = match.get('match_hometeam_score', '0')
            away_score = match.get('match_awayteam_score', '0')
            is_live = False
        elif match_live == '1' or match_status in ['LIVE', 'HALF TIME BREAK', 'PENALTY IN PROGRESS']:
            home_score = match.get('match_hometeam_score', '0')
            away_score = match.get('match_awayteam_score', '0')
            is_live = True
            if match_status.isdigit():
                live_minute = match_status

        return {
            'id': match.get('match_id', ''),
            'competition': {
                'id': match.get('league_id', ''),
                'name': match.get('league_name', '')
            },
            'utcDate': match_date,
            'status': 'LIVE' if is_live else ('FINISHED' if match_status == 'Finished' else 'SCHEDULED'),
            'homeTeam': {
                'name': home_name,
                'id': match.get('match_hometeam_id', '')
            },
            'awayTeam': {
                'name': away_name,
                'id': match.get('match_awayteam_id', '')
            },
            'score': {
                'fullTime': {
                    'home': int(home_score) if home_score.isdigit() else 0,
                    'away': int(away_score) if away_score.isdigit() else 0
                },
                'halfTime': {
                    'home': match.get('match_hometeam_halftime_score', '-'),
                    'away': match.get('match_awayteam_halftime_score', '-')
                }
            },
            'turkish_time': turkish_time_str,
            'is_live': is_live,
            'live_minute': live_minute
        }

    except Exception as e:
        logger.error(f"Error processing match: {str(e)}")
        return None

@app.route('/')
@cache.cached(timeout=300, query_string=True)  # 5 dakika önbellek, query string parametrelerine duyarlı
def index():
    """
    Ana sayfa - Günün maçlarını listeler
    Cache ile performans artırılmıştır (5 dakika önbellek)
    query_string=True sayesinde farklı tarihler için farklı önbellek oluşturulur
    """
    selected_date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    start_time = datetime.now()
    matches_data = get_matches(selected_date)
    elapsed_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Maç listesi yüklendi, süre: {elapsed_time:.2f} saniye")
    return render_template('index.html', matches=matches_data, selected_date=selected_date)

@app.route('/api/team-stats/<team_id>')
def team_stats(team_id):
    try:
        # APIFootball API anahtarı
        api_key = '39bc8dd65153ff5c7c0f37b4939009de04f4a70c593ee1aea0b8f70dd33268c0'

        # Son 6 aylık maçları al
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # 6 ay öncesine kadar maçları al

        # APIFootball'dan takımın son maçlarını al
        url = "https://apiv3.apifootball.com/"
        params = {
            'action': 'get_events',
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'team_id': team_id,
            'APIkey': api_key
        }

        logger.debug(f"Fetching team stats for team_id: {team_id}")
        response = requests.get(url, params=params)
        logger.debug(f"API Response status: {response.status_code}")

        if response.status_code == 200:
            matches = response.json()
            logger.debug(f"Total matches found: {len(matches)}")

            # Maçları tarihe göre sırala (en yeniden en eskiye)
            matches.sort(key=lambda x: x.get('match_date', ''), reverse=True)

            # Son 5 maçı al ve formatla
            last_5_matches = []
            for match in matches[:5]:  # Son 5 maç
                match_date = match.get('match_date', '')
                try:
                    # Tarihi düzgün formata çevir
                    date_obj = datetime.strptime(match_date, '%Y-%m-%d')
                    formatted_date = date_obj.strftime('%d.%m.%Y')
                except ValueError:
                    formatted_date = match_date

                match_data = {
                    'date': formatted_date,
                    'match': f"{match.get('match_hometeam_name', '')} vs {match.get('match_awayteam_name', '')}",
                    'score': f"{match.get('match_hometeam_score', '0')} - {match.get('match_awayteam_score', '0')}"
                }
                last_5_matches.append(match_data)

            return jsonify(last_5_matches)

        return jsonify([])

    except Exception as e:
        logger.error(f"Error fetching team stats: {str(e)}")
        return jsonify([])


@app.route('/test_half_time_stats')
def test_half_time_stats():
    """Test sayfası - İlk yarı/ikinci yarı istatistiklerini test etmek için"""
    return render_template('half_time_stats_test.html')
    
def get_league_standings(league_id):
    """Get standings for a specific league"""
    try:
        logger.info(f"Attempting to fetch standings for league_id: {league_id}")

        api_key = os.environ.get('FOOTBALL_DATA_API_KEY')
        if not api_key:
            logger.error("FOOTBALL_DATA_API_KEY is not set")
            return None

        # Football-data.org API endpoint
        url = f"https://api.football-data.org/v4/competitions/{league_id}/standings"
        headers = {'X-Auth-Token': api_key}

        logger.info(f"Making API request to {url}")
        response = requests.get(url, headers=headers)

        # Yanıt başlıklarını kontrol et
        logger.info(f"API Response headers: {response.headers}")

        # Yanıt içeriğini kontrol et
        try:
            data = response.json()
            logger.info(f"API Response data: {data}")
        except Exception as e:
            logger.error(f"Error parsing JSON response: {e}")
            return None

        if response.status_code != 200:
            logger.error(f"API request failed with status code: {response.status_code}")
            logger.error(f"Error message: {data.get('message', 'No error message provided')}")
            return None

        if 'standings' not in data:
            logger.error("API response doesn't contain standings data")
            logger.error(f"Full response: {data}")
            return None

        standings = []
        for standing_type in data['standings']:
            if standing_type['type'] == 'TOTAL':  # Ana puan durumu
                for team in standing_type['table']:
                    team_data = {
                        'rank': team['position'],
                        'name': team['team']['name'],
                        'logo': team['team']['crest'],
                        'played': team['playedGames'],
                        'won': team['won'],
                        'draw': team['draw'],
                        'lost': team['lost'],
                        'goals_for': team['goalsFor'],
                        'goals_against': team['goalsAgainst'],
                        'goal_diff': team['goalDifference'],
                        'points': team['points']
                    }
                    standings.append(team_data)
                break

        if not standings:
            logger.error("No standings data was processed")
            return None

        logger.info(f"Successfully processed standings data. Found {len(standings)} teams.")
        return standings

    except Exception as e:
        logger.error(f"Error in get_league_standings: {str(e)}")
        logger.exception("Full traceback:")
        return None

def get_available_leagues():
    """Get list of available leagues"""
    return [
        {'id': 2021, 'name': 'Premier League'},
        {'id': 2014, 'name': 'La Liga'},
        {'id': 2019, 'name': 'Serie A'},
        {'id': 2002, 'name': 'Bundesliga'},
        {'id': 2015, 'name': 'Ligue 1'}
    ]

@app.route('/leagues')
@cache.cached(timeout=3600, query_string=True)  # 1 saat önbellek, query string parametrelerine duyarlı
def leagues():
    try:
        # league_id'yi GET parametresinden al
        league_id = request.args.get('league_id', type=int)  # Changed back to int for new IDs
        logger.info(f"Received request for /leagues with league_id: {league_id}")

        available_leagues = get_available_leagues()
        logger.info(f"Available leagues: {available_leagues}")

        selected_league_name = None
        standings = None

        if league_id:
            logger.info(f"Processing request for league_id: {league_id}")

            # Find selected league name
            for league in available_leagues:
                if league['id'] == league_id:
                    selected_league_name = league['name']
                    logger.info(f"Found matching league: {selected_league_name}")
                    break

            if not selected_league_name:
                logger.error(f"No matching league found for league_id: {league_id}")
                flash("Seçtiğiniz lig için puan durumu verisi şu anda mevcut değil.", "error")
                return render_template('leagues.html',
                                    available_leagues=available_leagues,
                                    selected_league=None,
                                    selected_league_name=None,
                                    standings=None)

            # Get standings for selected league
            standings = get_league_standings(league_id)

            if standings is None:
                logger.error(f"Failed to fetch standings for league: {selected_league_name} (ID: {league_id})")
                flash("Puan durumu verisi alınamadı. Lütfen daha sonra tekrar deneyin.", "error")
            else:
                logger.info(f"Successfully fetched standings for {selected_league_name}")

        return render_template('leagues.html',
                            available_leagues=available_leagues,
                            selected_league=league_id,
                            selected_league_name=selected_league_name,
                            standings=standings)

    except Exception as e:
        logger.error(f"Unexpected error in leagues route: {str(e)}")
        logger.exception("Full traceback:")
        flash("Bir hata oluştu. Lütfen daha sonra tekrar deneyin.", "error")
        return render_template('leagues.html',
                            available_leagues=get_available_leagues(),
                            selected_league=None,
                            selected_league_name=None,
                            standings=None)

@app.route('/api/predict', methods=['POST'])
def predict_match_post():
    """POST metodu ile maç tahmini yap"""
    try:
        # JSON verisi al
        data = request.json
        if not data:
            return jsonify({"error": "JSON verisi eksik"}), 400
        
        # Takım ID ve adları
        home_team_id = data.get('home_team_id')
        away_team_id = data.get('away_team_id')
        home_team_name = data.get('home_team_name', 'Ev Sahibi')
        away_team_name = data.get('away_team_name', 'Deplasman')
        force_update = data.get('force_update', False)
        
        # Takım ID'lerini doğrula
        if not home_team_id or not away_team_id:
            return jsonify({"error": "Takım ID'leri eksik"}), 400
            
        # Tahmin yap
        prediction = predictor.predict_match(
            home_team_id, 
            away_team_id, 
            home_team_name, 
            away_team_name, 
            force_update=force_update
        )
        
        return jsonify(prediction)
    except Exception as e:
        logger.error(f"Tahmin POST işlemi sırasında hata: {str(e)}", exc_info=True)
        return jsonify({"error": f"Tahmin yapılırken hata oluştu: {str(e)}"}), 500

@app.route('/api/predict-match/<home_team_id>/<away_team_id>')
def predict_match(home_team_id, away_team_id):
    """Belirli bir maç için tahmin yap"""
    try:
        # Takım adlarını alın
        home_team_name = request.args.get('home_name', 'Ev Sahibi')
        away_team_name = request.args.get('away_name', 'Deplasman')
        force_update = request.args.get('force_update', 'false').lower() == 'true'
        
        # Takım ID'lerini doğrula
        if not home_team_id or not away_team_id or not home_team_id.isdigit() or not away_team_id.isdigit():
            return jsonify({"error": "Geçersiz takım ID'leri"}), 400

        # Önbellek anahtarı oluştur
        cache_key = f"predict_match_{home_team_id}_{away_team_id}_{home_team_name}_{away_team_name}"
        
        # Önbellekten getir (eğer force_update değilse)
        cached_prediction = None
        if not force_update:
            cached_prediction = cache.get(cache_key)
            
        if cached_prediction and not force_update:
            logger.info(f"Önbellekten tahmin alındı: {home_team_name} vs {away_team_name}")
            # Önbellekteki veriyi timestampli olarak işaretle
            cached_prediction['from_cache'] = True
            cached_prediction['cache_timestamp'] = datetime.now().timestamp()
            return jsonify(cached_prediction)
            
        # Eğer önbellekte değilse veya force_update ise yeni tahmin yap
        logger.info(f"Yeni tahmin yapılıyor. Force update: {force_update}, Takımlar: {home_team_name} vs {away_team_name}")
            
        try:
            # Tahmin yap
            prediction = predictor.predict_match(home_team_id, away_team_id, home_team_name, away_team_name, force_update)
            
            # Yeni tahmini önbelleğe ekle (10 dakika süreyle)
            if prediction and (isinstance(prediction, dict) and not prediction.get('error')):
                prediction['from_cache'] = False
                prediction['cache_timestamp'] = datetime.now().timestamp()
                # Önbelleğe 10 dakika süreyle kaydet
                cache.set(cache_key, prediction, timeout=600)

            if not prediction:
                return jsonify({"error": "Tahmin yapılamadı, takım verileri eksik olabilir", 
                               "match": f"{home_team_name} vs {away_team_name}"}), 400
                
            # Tahmin hata içeriyorsa
            if isinstance(prediction, dict) and "error" in prediction:
                return jsonify(prediction), 400

            # Maksimum yanıt boyutu kontrolü - büyük tahmin verilerinde hata olmasını önle
            import json
            response_size = len(json.dumps(prediction))
            
            if response_size > 1000000:  # 1MB'dan büyükse
                logger.warning(f"Çok büyük yanıt boyutu: {response_size} byte. Gereksiz detaylar kırpılıyor.")
                # Bazı gereksiz alanları kırp
                if 'home_team' in prediction and 'form' in prediction['home_team']:
                    # Form detaylarını azalt
                    prediction['home_team']['form'].pop('detailed_data', None)
                    prediction['home_team'].pop('form_periods', None)
                
                if 'away_team' in prediction and 'form' in prediction['away_team']:
                    # Form detaylarını azalt
                    prediction['away_team']['form'].pop('detailed_data', None)
                    prediction['away_team'].pop('form_periods', None)
                
                if 'predictions' in prediction and 'raw_metrics' in prediction['predictions']:
                    # Raw metrikleri kaldır
                    prediction['predictions'].pop('raw_metrics', None)

            return jsonify(prediction)
        except Exception as predict_error:
            logger.error(f"Tahmin işlemi sırasında hata: {str(predict_error)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Daha basit bir yanıt dön - veri boyutu nedenli hatalar için
            return jsonify({
                "error": "Tahmin işlemi sırasında teknik bir hata oluştu, lütfen daha sonra tekrar deneyin",
                "match": f"{home_team_name} vs {away_team_name}",
                "timestamp": datetime.now().timestamp()
            }), 500

    except Exception as e:
        logger.error(f"Tahmin yapılırken beklenmeyen hata: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Güvenli erişim - değişkenler tanımlanmamış veya None olabilir
        home_name = home_team_name if 'home_team_name' in locals() and home_team_name is not None else f"Takım {home_team_id}"
        away_name = away_team_name if 'away_team_name' in locals() and away_team_name is not None else f"Takım {away_team_id}"
        
        return jsonify({"error": "Sistem hatası. Lütfen daha sonra tekrar deneyin.", 
                        "match": f"{home_name} vs {away_name}"}), 500

@app.route('/api/clear-cache', methods=['POST'])
def clear_predictions_cache():
    """Tahmin önbelleğini temizle (hem dosya tabanlı önbelleği hem de Flask-Cache önbelleğini)"""
    try:
        # Predictor dosya tabanlı önbelleğini temizle
        success_file_cache = predictor.clear_cache()
        
        # Flask-Cache önbelleğini temizle
        with app.app_context():
            success_flask_cache = cache.clear()
        
        # Her iki önbelleğin de temizlenme durumunu değerlendir
        success = success_file_cache and success_flask_cache
        
        if success:
            logger.info("Hem dosya tabanlı önbellek hem de Flask-Cache önbelleği başarıyla temizlendi.")
            return jsonify({
                "success": True, 
                "message": "Tüm önbellekler temizlendi, yeni tahminler yapılabilir",
                "flask_cache_cleared": success_flask_cache,
                "file_cache_cleared": success_file_cache
            })
        else:
            logger.warning(f"Önbellek temizleme kısmen başarılı oldu. Dosya önbelleği: {success_file_cache}, Flask-Cache: {success_flask_cache}")
            return jsonify({
                "success": False, 
                "message": "Önbellek temizlenirken bazı sorunlar oluştu, ancak işlem devam edebilir", 
                "flask_cache_cleared": success_flask_cache,
                "file_cache_cleared": success_file_cache
            }), 200
    except Exception as e:
        error_msg = f"Önbellek temizlenirken beklenmeyen hata: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg, "success": False}), 500

@app.route('/predictions')
def predictions_page():
    """Tüm tahminleri gösteren sayfa"""
    return render_template('predictions.html')

@app.route('/cache-table')
def cache_table():
    """Önbellekteki tahminleri tabloda gösteren sayfa"""
    import json
    from tabulate import tabulate
    
    try:
        with open('predictions_cache.json', 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        # Sonuçları saklamak için liste
        results = []
        
        # Tüm tahminleri dolaş
        for match_key, prediction in predictions.items():
            if 'home_team' not in prediction or 'away_team' not in prediction:
                continue
                
            home_name = prediction.get('home_team', {}).get('name', '')
            away_name = prediction.get('away_team', {}).get('name', '')
            match_name = f'{home_name} vs {away_name}'
            
            # Tahmin edilen skor
            exact_score = prediction.get('predictions', {}).get('betting_predictions', {}).get('exact_score', {}).get('prediction', 'N/A')
            
            # Gerçek sonucu bul
            actual_home_goals = None
            actual_away_goals = None
            actual_result = 'Henüz oynanmadı'
            
            # Ev sahibi takımın son maçlarında ara
            if 'home_team' in prediction and 'form' in prediction['home_team'] and 'recent_match_data' in prediction['home_team']['form']:
                for match in prediction['home_team']['form']['recent_match_data']:
                    if match.get('opponent') == away_name and match.get('is_home', False) and match.get('result') in ['W', 'D', 'L']:
                        actual_home_goals = match.get('goals_scored', 'N/A')
                        actual_away_goals = match.get('goals_conceded', 'N/A')
                        actual_result = f'{actual_home_goals}-{actual_away_goals}'
                        break
                        
            # Deplasman takımının son maçlarında ara
            if actual_result == 'Henüz oynanmadı' and 'away_team' in prediction and 'form' in prediction['away_team'] and 'recent_match_data' in prediction['away_team']['form']:
                for match in prediction['away_team']['form']['recent_match_data']:
                    if match.get('opponent') == home_name and not match.get('is_home', True) and match.get('result') in ['W', 'D', 'L']:
                        actual_away_goals = match.get('goals_scored', 'N/A')
                        actual_home_goals = match.get('goals_conceded', 'N/A')
                        actual_result = f'{actual_home_goals}-{actual_away_goals}'
                        break
            
            # Tahmin doğruluğunu kontrol et
            accuracy = 'Doğru' if exact_score == actual_result and actual_result != 'Henüz oynanmadı' else 'Yanlış' if actual_result != 'Henüz oynanmadı' else 'Henüz oynanmadı'
            
            # Ev sahibi ve deplasman takımlarının gol beklentileri
            expected_home = prediction.get('predictions', {}).get('expected_goals', {}).get('home', 'N/A')
            expected_away = prediction.get('predictions', {}).get('expected_goals', {}).get('away', 'N/A')
            expected_goals = f'{expected_home}-{expected_away}'
            
            # Tahmin tarihi
            date_predicted = prediction.get('date_predicted', 'Bilinmiyor')
            
            # Sonucu ekle
            results.append([match_name, exact_score, actual_result, expected_goals, accuracy, date_predicted])
                
        # Özet istatistikler hesapla
        completed_matches = [r for r in results if r[4] != 'Henüz oynanmadı']
        correct_predictions = [r for r in completed_matches if r[4] == 'Doğru']
        
        # İstatistikler
        total_matches = len(results)
        completed_count = len(completed_matches)
        correct_ratio = round(len(correct_predictions)/completed_count*100, 2) if completed_count > 0 else 0
        correct_count = len(correct_predictions)
        
        return render_template('cache_table.html', 
                              results=results, 
                              total_matches=total_matches, 
                              completed_count=completed_count, 
                              correct_ratio=correct_ratio, 
                              correct_count=correct_count)
    except FileNotFoundError:
        return render_template('cache_table.html', error="Önbellek dosyası (predictions_cache.json) bulunamadı.")
    except json.JSONDecodeError:
        return render_template('cache_table.html', error="Önbellek dosyası geçerli bir JSON formatında değil.")
    except Exception as e:
        return render_template('cache_table.html', error=f"Bir hata oluştu: {str(e)}")

@app.route('/model-validation')
@cache.cached(timeout=1800)  # 30 dakika önbellek
def model_validation_page():
    """
    Model doğrulama ve değerlendirme sayfasını göster
    Bu sayfa, doğrulama raporlarını ve sonuçlarını görselleştirir
    Doğrulama verileri sık değişmediği için 30 dakikalık bir önbellek uygun
    """
    return render_template('model_validation.html')

# AI İçgörüleri route'u
@app.route('/insights/<home_team_id>/<away_team_id>', methods=['GET'])
def match_insights(home_team_id, away_team_id):
    """Maç için AI içgörüleri ve doğal dil açıklamaları göster"""
    try:
        from match_insights import MatchInsightsGenerator
        insights_generator = MatchInsightsGenerator()
        
        # Takım verilerini al
        home_team_name = request.args.get('home_name', 'Ev Sahibi')
        away_team_name = request.args.get('away_name', 'Deplasman')
        
        # İçgörüleri oluştur
        insights = insights_generator.generate_match_insights(
            home_team_id, away_team_id, 
            additional_data={
                'home_team_name': home_team_name,
                'away_team_name': away_team_name
            }
        )
        
        # Eğer içgörüler başarıyla oluşturulursa şablonu render et
        if insights and 'error' not in insights:
            template_data = {
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'home_team_name': home_team_name,
                'away_team_name': away_team_name,
                'insights': insights
            }
            return render_template('match_insights.html', **template_data)
        else:
            # Hata durumunda ana sayfaya yönlendir
            flash('İçgörüler oluşturulamadı. Lütfen daha sonra tekrar deneyin.', 'warning')
            return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"İçgörüler oluşturulurken hata: {str(e)}")
        flash('Bir hata oluştu. Lütfen daha sonra tekrar deneyin.', 'danger')
        return redirect(url_for('index'))

def find_available_port(preferred_ports=None):
    """
    Kullanılabilir bir port bul
    
    Args:
        preferred_ports: Tercih edilen portların listesi, önce bunlar denenecek
        
    Returns:
        int: Kullanılabilir port numarası
    """
    import socket
    
    # Hiç tercih edilen port belirtilmemişse varsayılan listeyi kullan
    if preferred_ports is None:
        # Sırasıyla denenecek portlar
        preferred_ports = [80, 8080, 5000, 3000, 8000, 8888, 9000]
    
    # Önce çevre değişkeninden PORT değerini kontrol et
    env_port = os.environ.get('PORT')
    if env_port:
        try:
            env_port = int(env_port)
            if env_port not in preferred_ports:
                # Çevre değişkeni varsa onu listenin başına ekle
                preferred_ports.insert(0, env_port)
        except ValueError:
            logger.warning(f"Çevre değişkenindeki PORT değeri ({env_port}) geçerli bir sayı değil, yok sayılıyor")
    
    # Her bir portu dene ve kullanılabilir olanı bul
    for port in preferred_ports:
        try:
            # Port müsait mi kontrol et
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result != 0:  # Port açık değilse (bağlantı başarısız oldu)
                logger.info(f"Port {port} kullanılabilir, bu port kullanılacak")
                return port
            else:
                logger.warning(f"Port {port} zaten kullanımda, başka port deneniyor")
        except Exception as e:
            logger.warning(f"Port {port} kontrolü sırasında hata: {str(e)}")
    
    # Hiçbir tercih edilen port kullanılamıyorsa, rastgele bir port ata
    logger.warning("Tercih edilen portların hiçbiri kullanılamıyor, rastgele bir port atanacak")
    return 0  # 0 verilirse, sistem otomatik olarak kullanılabilir bir port atar

if __name__ == '__main__':
    # Dinamik takım analizörünü başlat
    try:
        team_analyzer = DynamicTeamAnalyzer()
        logger.info("Dinamik Takım Analizörü başlatıldı")
        
        # Performans verilerini analiz et ve güncelle
        team_analyzer.analyze_and_update()
        logger.info("Takım performans analizi tamamlandı")
        
        # Kendi kendine öğrenen tahmin modelini başlat
        self_learning = SelfLearningPredictor(analyzer=team_analyzer)
        logger.info("Kendi Kendine Öğrenen Tahmin Modeli başlatıldı")
        
        # Takım faktörlerini analiz et
        analysis_result = self_learning.analyze_predictions_and_results()
        if analysis_result.get('sufficient_data', False):
            logger.info(f"Model analizi tamamlandı: {analysis_result.get('analyzed_matches', 0)} maç analiz edildi")
            logger.info(f"Doğruluk: {analysis_result.get('outcome_accuracy', 0):.4f}")
        else:
            logger.info("Yeterli doğrulama verisi yok, model analizi atlandı")
            
        # Performans güncelleyiciyi arkaplanda başlat
        updater = TeamPerformanceUpdater(analyzer=team_analyzer)
        updater.start()
        logger.info("Takım Performans Güncelleyici arkaplanda başlatıldı")
    except Exception as e:
        logger.error(f"Dinamik analiz sistemleri başlatılırken hata: {str(e)}")
        logger.info("Uygulama temel tahmin modelleriyle çalışmaya devam edecek")
    
    # Şimdi api_routes modülündeki blueprint'i içe aktar
    try:
        from api_routes import api_v3_bp
        # Register the blueprint after it's properly imported
        app.register_blueprint(api_v3_bp)
        logger.info("API Blueprint başarıyla kaydedildi")
    except Exception as e:
        logger.error(f"API Blueprint kaydedilirken hata: {str(e)}")
    
    # PORT çevre değişkeni veya varsayılan portları kontrol et
    try:
        import socket
        
        port = None
        # İlk olarak PORT çevre değişkenini dene
        port_env = os.environ.get('PORT')
        if port_env:
            try:
                port = int(port_env)
                logger.info(f"PORT çevre değişkeni bulundu: {port}")
            except ValueError:
                logger.warning(f"PORT çevre değişkeni geçerli bir sayı değil: {port_env}")
                port = None
                
        # Tercih edilen portları dene
        preferred_ports = [8080, 3000, 5000]
        
        # PORT çevre değişkeni geçerli değilse tercih edilen portları dene
        if port is None:
            for test_port in preferred_ports:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.bind(('0.0.0.0', test_port))
                    sock.close()
                    port = test_port
                    logger.info(f"Kullanılabilir port bulundu: {port}")
                    break
                except OSError:
                    logger.warning(f"Port {test_port} kullanılamıyor, sonraki deneniyor...")
                    continue
                    
        # Hala port bulunamadıysa, sisteme rastgele bir port seçtir
        if port is None:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('0.0.0.0', 0))
                port = sock.getsockname()[1]
                sock.close()
                logger.info(f"Rastgele port seçildi: {port}")
            except OSError as e:
                logger.error(f"Rastgele port seçerken hata: {str(e)}")
                # Son çare olarak 8888 portunu dene
                port = 8888
        
        # Son kontrol - port hala None ise 3000 ile dene
        if port is None:
            port = 3000
            
        logger.info(f"Uygulama {port} portunda başlatılıyor")
        print(f"Uygulama {port} portunda başlatılıyor")
            
        # Uygulamayı çalıştır
        app.run(host='0.0.0.0', port=port, debug=True)
        
    except Exception as e:
        logger.error(f"Uygulama başlatılırken kritik hata: {str(e)}")
        # Son çare - 5000 portunu güvenli modda dene
        try:
            logger.info("Son çare: 5000 portu güvenli modda deneniyor...")
            print("Son çare: 5000 portu güvenli modda deneniyor...")
            # Debug modunu kapatarak dene
            app.run(host='0.0.0.0', port=5000, debug=False)
        except Exception as final_e:
            logger.error(f"Son çare girişimi de başarısız: {str(final_e)}")
            print(f"Uygulama başlatılamadı. Hata: {str(final_e)}")