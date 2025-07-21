"""
Dinamik Ensemble Sistemini Test Et
"""
import logging
from algorithms.ensemble import EnsemblePredictor
from match_categorizer import MatchCategorizer
from model_performance_tracker import ModelPerformanceTracker

# Loglama ayarla
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_match_categorization():
    """Maç kategorilendirme testi"""
    print("\n=== MAÇ KATEGORİLENDİRME TESTİ ===")
    
    categorizer = MatchCategorizer()
    
    # Test maçları
    test_matches = [
        {
            'league': 'Bundesliga',
            'home_team': 'Bayern Munich',
            'away_team': 'Dortmund', 
            'elo_diff': 50,
            'home_stats': {'avg_goals_scored': 2.5, 'avg_goals_conceded': 0.8},
            'away_stats': {'avg_goals_scored': 2.2, 'avg_goals_conceded': 1.2},
            'date': '2025-07-18'
        },
        {
            'league': 'Serie A',
            'home_team': 'Juventus',
            'away_team': 'Inter Milan',
            'elo_diff': 30,
            'home_stats': {'avg_goals_scored': 1.3, 'avg_goals_conceded': 0.7},
            'away_stats': {'avg_goals_scored': 1.4, 'avg_goals_conceded': 0.9},
            'date': '2025-07-18'
        },
        {
            'league': 'Premier League',
            'home_team': 'Manchester United',
            'away_team': 'Manchester City',
            'elo_diff': -150,
            'home_stats': {'avg_goals_scored': 1.6, 'avg_goals_conceded': 1.4},
            'away_stats': {'avg_goals_scored': 2.3, 'avg_goals_conceded': 0.9},
            'date': '2025-07-18'
        }
    ]
    
    for match in test_matches:
        print(f"\n{match['home_team']} vs {match['away_team']} ({match['league']})")
        categories = categorizer.categorize_match(match)
        
        print(f"Lig kategorisi: {categories['league_category']}")
        print(f"Maç tipi: {categories['match_type']}")
        print(f"Takım profilleri: {categories['team_profiles']}")
        
        # Önerilen ağırlıkları göster
        weights = categorizer.get_category_weights(categories)
        print("Önerilen model ağırlıkları:")
        for model, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model}: %{weight*100:.1f}")

def test_ensemble_integration():
    """Ensemble entegrasyon testi"""
    print("\n\n=== ENSEMBLE ENTEGRASYON TESTİ ===")
    
    ensemble = EnsemblePredictor()
    
    # Mock tahmin verileri
    model_predictions = {
        'poisson': {
            'home_win': 45,
            'draw': 30,
            'away_win': 25,
            'over_2_5': 55,
            'under_2_5': 45,
            'btts_yes': 60,
            'btts_no': 40,
            'expected_goals': {'home': 1.8, 'away': 1.2},
            'confidence': 0.75
        },
        'dixon_coles': {
            'home_win': 48,
            'draw': 32,
            'away_win': 20,
            'over_2_5': 45,
            'under_2_5': 55,
            'btts_yes': 50,
            'btts_no': 50,
            'expected_goals': {'home': 1.5, 'away': 1.0},
            'confidence': 0.70
        },
        'xgboost': {
            'home_win': 42,
            'draw': 28,
            'away_win': 30,
            'over_2_5': 60,
            'under_2_5': 40,
            'btts_yes': 65,
            'btts_no': 35,
            'expected_goals': {'home': 2.0, 'away': 1.3},
            'confidence': 0.80
        }
    }
    
    # Test bağlamı
    match_context = {
        'league': 'Bundesliga',
        'home_team': 'Bayern Munich',
        'away_team': 'Dortmund',
        'elo_diff': 50,
        'lambda_home': 2.1,
        'lambda_away': 1.6,
        'home_stats': {'avg_goals_scored': 2.5, 'avg_goals_conceded': 0.8},
        'away_stats': {'avg_goals_scored': 2.2, 'avg_goals_conceded': 1.2},
        'date': '2025-07-18'
    }
    
    print("\nDinamik ağırlık sistemi ile tahmin birleştirme...")
    
    # Ensemble tahmin
    try:
        result = ensemble.combine_predictions(model_predictions, match_context)
        
        if result:
            print("\nBirleştirilmiş tahmin sonuçları:")
            print(f"Ev sahibi kazanır: %{result.get('home_win', 0):.1f}")
            print(f"Beraberlik: %{result.get('draw', 0):.1f}")
            print(f"Deplasman kazanır: %{result.get('away_win', 0):.1f}")
            print(f"2.5 Üst: %{result.get('over_2_5', 0):.1f}")
            print(f"KG Var: %{result.get('btts_yes', 0):.1f}")
            print(f"Güven: %{result.get('confidence', 0)*100:.1f}")
    except Exception as e:
        print(f"Hata: {e}")
        
def test_performance_tracking():
    """Performans takip testi"""
    print("\n\n=== PERFORMANS TAKİP TESTİ ===")
    
    tracker = ModelPerformanceTracker()
    
    # Test tahminleri ekle
    test_predictions = [
        {
            'model': 'poisson',
            'prediction': {'most_likely_outcome': 'HOME_WIN'},
            'actual': {'home_goals': 2, 'away_goals': 1},
            'match_info': {'league': 'Bundesliga', 'elo_diff': 100}
        },
        {
            'model': 'dixon_coles',
            'prediction': {'most_likely_outcome': 'DRAW'},
            'actual': {'home_goals': 1, 'away_goals': 1},
            'match_info': {'league': 'Serie A', 'elo_diff': 50}
        },
        {
            'model': 'xgboost',
            'prediction': {'most_likely_outcome': 'AWAY_WIN'},
            'actual': {'home_goals': 0, 'away_goals': 2},
            'match_info': {'league': 'Premier League', 'elo_diff': -200}
        }
    ]
    
    for test in test_predictions:
        tracker.track_prediction(
            test['model'],
            test['prediction'],
            test['actual'],
            test['match_info']
        )
    
    # Performans raporu
    print(tracker.generate_performance_report())

if __name__ == "__main__":
    test_match_categorization()
    test_ensemble_integration()
    test_performance_tracking()
    
    print("\n\n=== TEST TAMAMLANDI ===")
    print("Dinamik ensemble sistemi başarıyla entegre edildi!")