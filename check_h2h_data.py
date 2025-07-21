import json

# predictions_cache.json dosyasını oku
with open('predictions_cache.json', 'r', encoding='utf-8') as f:
    cache_data = json.load(f)

# H2H veri durumunu analiz et
total_predictions = len(cache_data)
h2h_available = 0
h2h_empty = 0
h2h_not_present = 0

print("H2H Veri Durumu Analizi")
print("=" * 50)

for key, data in cache_data.items():
    prediction = data.get('prediction', {})
    home_team = prediction.get('match_info', {}).get('home_team', {}).get('name', 'Unknown')
    away_team = prediction.get('match_info', {}).get('away_team', {}).get('name', 'Unknown')
    
    if 'h2h_data' in prediction:
        h2h_data = prediction['h2h_data']
        if isinstance(h2h_data, dict) and 'matches' in h2h_data:
            matches_count = len(h2h_data['matches'])
            if matches_count > 0:
                h2h_available += 1
                print(f"✓ {home_team} vs {away_team}: {matches_count} H2H maç bulundu")
                # İlk 3 maçın sonuçlarını göster
                for i, match in enumerate(h2h_data['matches'][:3]):
                    if 'match_hometeam_name' in match:
                        home_score = match.get('match_hometeam_score', '0')
                        away_score = match.get('match_awayteam_score', '0')
                        date = match.get('match_date', '')
                        print(f"   • {date}: {match['match_hometeam_name']} {home_score}-{away_score} {match['match_awayteam_name']}")
            else:
                h2h_empty += 1
                print(f"✗ {home_team} vs {away_team}: H2H maç bulunamadı (boş liste)")
        else:
            h2h_empty += 1
            print(f"✗ {home_team} vs {away_team}: H2H veri yapısı bozuk")
    else:
        h2h_not_present += 1
        print(f"? {home_team} vs {away_team}: H2H verisi yok")

print("\n" + "=" * 50)
print(f"Toplam tahmin: {total_predictions}")
print(f"H2H verisi olan: {h2h_available} ({h2h_available/total_predictions*100:.1f}%)")
print(f"H2H verisi boş: {h2h_empty} ({h2h_empty/total_predictions*100:.1f}%)")
print(f"H2H verisi olmayan: {h2h_not_present} ({h2h_not_present/total_predictions*100:.1f}%)")

print("\n🔍 H2H Veri Analizi Özeti:")
if h2h_available > 0:
    print(f"• %{h2h_available/total_predictions*100:.0f} oranında başarılı H2H veri çekimi")
    print(f"• Ortalama {sum([len(data.get('prediction', {}).get('h2h_data', {}).get('matches', [])) for data in cache_data.values()])/h2h_available:.1f} H2H maç/takım")
if h2h_empty > 0:
    print(f"• %{h2h_empty/total_predictions*100:.0f} oranında H2H verisi bulunamadı (iki takım daha önce karşılaşmamış olabilir)")