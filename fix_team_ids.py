import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API key'i config dosyasından al
try:
    with open('api_config.json', 'r') as f:
        config = json.load(f)
        API_KEY = config.get('api_key', '')
except:
    logger.error("API key bulunamadı!")
    exit(1)

# La Liga ID: 302 (dokümantasyondan)
SPAIN_LALIGA_ID = 302

def get_teams_from_league(league_id):
    """Belirli bir ligdeki takımları çek"""
    url = "https://apiv3.apifootball.com/"
    params = {
        'action': 'get_teams',
        'league_id': league_id,
        'APIkey': API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            teams = response.json()
            return teams
        else:
            logger.error(f"API hatası: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Takım çekme hatası: {e}")
        return []

def find_correct_team_ids():
    """Barcelona ve Real Madrid'in doğru ID'lerini bul"""
    logger.info("La Liga takımları çekiliyor...")
    teams = get_teams_from_league(SPAIN_LALIGA_ID)
    
    barcelona_id = None
    real_madrid_id = None
    
    for team in teams:
        team_name = team.get('team_name', '').lower()
        team_id = team.get('team_key', '')
        
        logger.info(f"Takım: {team.get('team_name')} - ID: {team_id}")
        
        if 'barcelona' in team_name:
            barcelona_id = team_id
            logger.info(f"✓ Barcelona bulundu! Doğru ID: {barcelona_id}")
            
        elif 'real madrid' in team_name:
            real_madrid_id = team_id
            logger.info(f"✓ Real Madrid bulundu! Doğru ID: {real_madrid_id}")
    
    return barcelona_id, real_madrid_id

def test_h2h_with_correct_ids(team1_id, team2_id, team1_name, team2_name):
    """Doğru ID'lerle H2H verisi test et"""
    url = "https://apiv3.apifootball.com/"
    params = {
        'action': 'get_H2H',
        'firstTeamId': team1_id,
        'secondTeamId': team2_id,
        'APIkey': API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            h2h_data = response.json()
            
            if 'firstTeam_VS_secondTeam' in h2h_data:
                matches = h2h_data['firstTeam_VS_secondTeam']
                logger.info(f"\nH2H Test Sonucu: {team1_name} vs {team2_name}")
                logger.info(f"Toplam maç sayısı: {len(matches)}")
                
                if matches:
                    # İlk maçı kontrol et
                    first_match = matches[0]
                    home_team = first_match.get('match_hometeam_name', '')
                    away_team = first_match.get('match_awayteam_name', '')
                    date = first_match.get('match_date', '')
                    
                    logger.info(f"İlk maç: {date} - {home_team} vs {away_team}")
                    
                    # Takım isimlerini kontrol et
                    if team1_name.lower() in home_team.lower() or team1_name.lower() in away_team.lower():
                        logger.info("✓ H2H verisi doğru!")
                        return True
                    else:
                        logger.error("✗ H2H verisi yanlış takımlar içeriyor!")
                        return False
                        
    except Exception as e:
        logger.error(f"H2H test hatası: {e}")
        return False

# Ana işlem
if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Takım ID Düzeltme İşlemi Başlıyor")
    logger.info("=" * 50)
    
    # 1. Doğru ID'leri bul
    barcelona_id, real_madrid_id = find_correct_team_ids()
    
    if barcelona_id and real_madrid_id:
        logger.info(f"\nBulunan ID'ler:")
        logger.info(f"Barcelona: {barcelona_id}")
        logger.info(f"Real Madrid: {real_madrid_id}")
        
        # 2. H2H verisini test et
        logger.info("\n" + "=" * 50)
        logger.info("H2H Verisi Test Ediliyor")
        logger.info("=" * 50)
        
        test_h2h_with_correct_ids(barcelona_id, real_madrid_id, "Barcelona", "Real Madrid")
        
        # 3. Düzeltme önerileri
        logger.info("\n" + "=" * 50)
        logger.info("Düzeltme Önerileri:")
        logger.info("=" * 50)
        logger.info(f"Mevcut yanlış ID'ler:")
        logger.info(f"  Barcelona: 529 (Australia) → {barcelona_id}")
        logger.info(f"  Real Madrid: 541 (Ecuador) → {real_madrid_id}")
        
    else:
        logger.error("Takım ID'leri bulunamadı!")