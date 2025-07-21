import json

# predictions_cache.json dosyasını oku
with open('predictions_cache.json', 'r', encoding='utf-8') as f:
    cache_data = json.load(f)

print("Takım ID Kontrolü")
print("=" * 50)

# Tüm tahminlerdeki takım bilgilerini kontrol et
for key, data in cache_data.items():
    prediction = data.get('prediction', {})
    match_info = prediction.get('match_info', {})
    
    home_team = match_info.get('home_team', {})
    away_team = match_info.get('away_team', {})
    
    home_name = home_team.get('name', 'Unknown')
    away_name = away_team.get('name', 'Unknown')
    home_id = home_team.get('id', 'Unknown')
    away_id = away_team.get('id', 'Unknown')
    
    print(f"\nMaç: {home_name} vs {away_name}")
    print(f"  Ev Sahibi ID: {home_id}")
    print(f"  Deplasman ID: {away_id}")
    
    # H2H verisi varsa kontrol et
    if 'h2h_data' in prediction and 'matches' in prediction['h2h_data']:
        h2h_matches = prediction['h2h_data']['matches']
        if h2h_matches:
            first_match = h2h_matches[0]
            h2h_home_name = first_match.get('match_hometeam_name', '')
            h2h_away_name = first_match.get('match_awayteam_name', '')
            h2h_home_id = first_match.get('match_hometeam_id', '')
            h2h_away_id = first_match.get('match_awayteam_id', '')
            
            print(f"  H2H'de gelen takımlar: {h2h_home_name} (ID: {h2h_home_id}) vs {h2h_away_name} (ID: {h2h_away_id})")
            
            # ID uyuşmazlığı kontrolü
            if (home_name.lower() not in h2h_home_name.lower() and 
                home_name.lower() not in h2h_away_name.lower() and
                away_name.lower() not in h2h_home_name.lower() and
                away_name.lower() not in h2h_away_name.lower()):
                print(f"  ⚠️ UYUŞMAZLIK: Takım isimleri H2H verileriyle eşleşmiyor!")
                print(f"     Beklenen: {home_name} vs {away_name}")
                print(f"     Gelen: {h2h_home_name} vs {h2h_away_name}")