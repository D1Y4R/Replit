#!/usr/bin/env python3
"""
H2H veri çekimini hızlı test et
"""

import asyncio
from async_data_fetcher import AsyncDataFetcher
from api_config import APIConfig

async def test_h2h():
    """H2H veri çekimini test et"""
    
    # API anahtarını al
    api_config = APIConfig()
    api_key = api_config.get_api_key()
    
    print(f"API Key: {api_key[:10]}...")
    
    # AsyncDataFetcher ile H2H verisi çek
    async with AsyncDataFetcher() as fetcher:
        # Barcelona (97) vs Real Madrid (76) H2H verisi
        h2h_data = await fetcher.fetch_h2h_data(
            home_team_id=97,
            away_team_id=76,
            api_key=api_key,
            home_team_name="Barcelona",
            away_team_name="Real Madrid"
        )
        
        if h2h_data and 'response' in h2h_data:
            total_matches = h2h_data.get('response', {}).get('total_matches', 0)
            print(f"\nToplam H2H maç sayısı: {total_matches}")
            
            matches = h2h_data.get('response', {}).get('matches', [])
            if matches:
                print("\nİlk 3 H2H maçı:")
                for i, match in enumerate(matches[:3]):
                    home = match.get('match_hometeam_name', '?')
                    away = match.get('match_awayteam_name', '?')
                    date = match.get('match_date', '?')
                    score = f"{match.get('match_hometeam_score', '?')}-{match.get('match_awayteam_score', '?')}"
                    print(f"{i+1}. {date}: {home} vs {away} - Skor: {score}")
            else:
                print("Maç verisi bulunamadı")
        else:
            print("H2H verisi alınamadı")

if __name__ == "__main__":
    asyncio.run(test_h2h())