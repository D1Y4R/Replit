#!/usr/bin/env python3
"""
H2H API çağrısını debug et
"""

import requests
from api_config import APIConfig

def debug_h2h_api():
    """H2H API çağrısını debug et"""
    
    # API anahtarını al
    api_config = APIConfig()
    api_key = api_config.get_api_key()
    
    print(f"API Key: {api_key[:10]}...")
    
    # H2H API çağrısı - takım isimleriyle
    url = "https://apiv3.apifootball.com/"
    params = {
        'action': 'get_H2H',
        'firstTeam': 'Barcelona',
        'secondTeam': 'Real Madrid',
        'APIkey': api_key
    }
    
    print(f"\nAPI URL: {url}")
    print(f"Params: {params}")
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"\nResponse Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Veri yapısını kontrol et
            if isinstance(data, dict) and 'error' in data:
                print(f"API Error: {data['error']}")
            elif isinstance(data, list) and data:
                print(f"\nToplam maç sayısı: {len(data)}")
                # İlk maçı göster
                if data[0]:
                    first_match = data[0]
                    print(f"\nİlk maç:")
                    print(f"Tarih: {first_match.get('match_date', '?')}")
                    print(f"Ev sahibi: {first_match.get('match_hometeam_name', '?')}")
                    print(f"Deplasman: {first_match.get('match_awayteam_name', '?')}")
                    print(f"Skor: {first_match.get('match_hometeam_score', '?')}-{first_match.get('match_awayteam_score', '?')}")
            elif data:
                print(f"\nBeklenmeyen veri formatı: {type(data)}")
                print(f"İlk 200 karakter: {str(data)[:200]}")
            else:
                print("Boş veri döndü")
        else:
            print(f"API Error Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"Hata: {str(e)}")
    
    # Takım ID'leriyle de deneyelim
    print("\n" + "="*50)
    print("Takım ID'leriyle deneniyor...")
    
    params2 = {
        'action': 'get_H2H',
        'firstTeamId': '97',  # Barcelona
        'secondTeamId': '76',  # Real Madrid
        'APIkey': api_key
    }
    
    print(f"Params: {params2}")
    
    try:
        response2 = requests.get(url, params=params2, timeout=10)
        print(f"\nResponse Status: {response2.status_code}")
        
        if response2.status_code == 200:
            data2 = response2.json()
            if isinstance(data2, list) and data2:
                print(f"Toplam maç sayısı: {len(data2)}")
            else:
                print(f"Veri: {str(data2)[:200]}")
                
    except Exception as e:
        print(f"Hata: {str(e)}")

if __name__ == "__main__":
    debug_h2h_api()