"""
Dynamic Team Analyzer Test
"""
import logging
from dynamic_team_analyzer import DynamicTeamAnalyzer

# Loglama ayarla
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def create_test_matches():
    """Test maç verileri oluştur"""
    # Son 20 maç verisi (form trendi için)
    matches = []
    
    # İyi form dönemi (son 5 maç)
    for i in range(5):
        matches.append({
            'date': f'2025-07-{18-i}',
            'venue': 'home' if i % 2 == 0 else 'away',
            'goals_scored': 2 + (i % 2),
            'goals_conceded': 0 if i < 2 else 1,
            'opponent_position': 8 + i,
            'match_type': 'derby' if i == 0 else 'normal'
        })
    
    # Orta form (sonraki 10 maç)
    for i in range(5, 15):
        matches.append({
            'date': f'2025-07-{18-i}',
            'venue': 'home' if i % 2 == 0 else 'away',
            'goals_scored': 1,
            'goals_conceded': 1,
            'opponent_position': 10
        })
    
    # Kötü form (en eski 5 maç)
    for i in range(15, 20):
        matches.append({
            'date': f'2025-06-{30-i+15}',
            'venue': 'home' if i % 2 == 0 else 'away',
            'goals_scored': 0,
            'goals_conceded': 2,
            'opponent_position': 15
        })
    
    return matches

def test_single_team_analysis():
    """Tek takım analizi testi"""
    print("\n=== TEK TAKIM ANALİZİ ===")
    
    analyzer = DynamicTeamAnalyzer()
    test_matches = create_test_matches()
    
    # Takım bilgileri
    team_info = {
        'position': 4,
        'recent_form': 'WWWDL',
        'matches_played': 25,
        'total_matches': 38
    }
    
    # Lig bilgileri
    league_info = {
        'total_teams': 20,
        'name': 'Premier League'
    }
    
    # Analiz yap
    analysis = analyzer.analyze_team(
        team_id=1,
        team_matches=test_matches,
        team_info=team_info,
        league_info=league_info,
        is_home=True
    )
    
    # Sonuçları göster
    print(f"\nGenel Skor: {analysis['overall_score']}")
    print(f"Momentum: {analysis['momentum']['overall_score']} ({analysis['momentum']['trend']})")
    print(f"Form - Genel: {analysis['momentum']['overall_form']:.1f}, Ev: {analysis['momentum']['venue_form']:.1f}")
    print(f"Mevcut Seri: {analysis['momentum']['current_streak']['description']}")
    
    print(f"\nTaktiksel Profil: {analysis['tactical_profile']['style']}")
    print(f"Tempo: {analysis['tactical_profile']['tempo']} ({analysis['tactical_profile']['tempo_details']['avg_total_goals']:.1f} gol/maç)")
    print(f"Savunma: {analysis['tactical_profile']['defensive_solidity']}")
    
    print(f"\nMotivation: {analysis['situational_factors']['motivation_level']}")
    print(f"Büyük Maç Performansı: {'Evet' if analysis['situational_factors']['big_match_performer'] else 'Hayır'}")
    
    print(f"\nAdaptasyon Skoru: {analysis['adaptation']['adaptation_score']}")
    print(f"Form Evrimi: {analysis['adaptation']['form_evolution']['trend']}")
    
    print(f"\nTahmin Ayarlamaları:")
    adj = analysis['prediction_adjustments']
    print(f"  Gol Beklentisi: {adj['goals_expectation']:+.2f}")
    print(f"  KG Olasılığı: {adj['btts_probability']:+.0f}%")
    print(f"  2.5 Üst: {adj['over_2_5_probability']:+.0f}%")
    print(f"  Güven: {adj['confidence_modifier']:+.0f}%")
    
    print(f"\nÖzet: {analysis['summary']}")

def test_team_comparison():
    """İki takım karşılaştırma testi"""
    print("\n\n=== İKİ TAKIM KARŞILAŞTIRMASI ===")
    
    analyzer = DynamicTeamAnalyzer()
    
    # Ev sahibi - iyi formda
    home_matches = create_test_matches()
    home_info = {
        'position': 3,
        'recent_form': 'WWWWD',
        'matches_played': 25
    }
    
    # Deplasman - kötü formda
    away_matches = []
    for i in range(20):
        away_matches.append({
            'date': f'2025-07-{18-i}',
            'venue': 'away' if i % 2 == 0 else 'home',
            'goals_scored': 0 if i < 10 else 1,
            'goals_conceded': 2,
            'opponent_position': 5
        })
    
    away_info = {
        'position': 17,
        'recent_form': 'LLLLD',
        'matches_played': 25
    }
    
    # Analizleri yap
    home_analysis = analyzer.analyze_team(1, home_matches, home_info, is_home=True)
    away_analysis = analyzer.analyze_team(2, away_matches, away_info, is_home=False)
    
    # Karşılaştır
    comparison = analyzer.compare_teams(home_analysis, away_analysis)
    
    print(f"\nMomentum Avantajı: {comparison['momentum_advantage']} (Fark: {comparison['momentum_diff']:.1f})")
    print(f"Taktiksel Eşleşme: {comparison['tactical_matchup']['style_compatibility']}")
    print(f"Motivasyon Farkı: {comparison['motivation_diff']:.1f}")
    
    print(f"\nMaç Dinamikleri:")
    print(f"  Beklenen Patern: {comparison['match_dynamics']['expected_pattern']}")
    print(f"  Anahtar Faktörler: {', '.join(comparison['match_dynamics']['key_factors'])}")
    print(f"  Sürpriz Potansiyeli: {comparison['match_dynamics']['surprise_potential']}")
    
    print(f"\nKombine Ayarlamalar:")
    comb = comparison['combined_adjustments']
    print(f"  Toplam Gol: {comb['total_goals_modifier']:+.2f}")
    print(f"  KG: {comb['btts_modifier']:+.0f}%")
    print(f"  2.5 Üst: {comb['over_2_5_modifier']:+.0f}%")
    print(f"  Güven: {comb['confidence_modifier']:+.0f}%")
    print(f"  Volatilite: x{comb['volatility_factor']:.2f}")

if __name__ == "__main__":
    test_single_team_analysis()
    test_team_comparison()
    
    print("\n\n=== TEST TAMAMLANDI ===")
    print("Dynamic Team Analyzer sistemi başarıyla çalışıyor!")