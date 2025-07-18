import json
import sys
from tabulate import tabulate
from datetime import datetime

def analyze_form_data(team_data, team_name, is_home):
    """Takımın form durumunu analiz et"""
    if not team_data or 'form' not in team_data:
        return "Form verisi bulunamadı"

    form = team_data['form']
    recent_matches = form.get('recent_match_data', [])

    # Son 5 maç analizi
    last_5 = []
    home_matches = []
    away_matches = []

    for match in recent_matches:
        match_info = {
            'opponent': match.get('opponent', 'Bilinmiyor'),
            'score': f"{match.get('goals_scored', 0)}-{match.get('goals_conceded', 0)}",
            'result': match.get('result', 'N/A'),
            'is_home': match.get('is_home', False)
        }

        if len(last_5) < 5:
            last_5.append(match_info)

        if match.get('is_home', False):
            home_matches.append(match_info)
        else:
            away_matches.append(match_info)

    # Son 5 iç saha/deplasman maçı
    relevant_matches = home_matches[:5] if is_home else away_matches[:5]

    analysis = f"\n{team_name} Form Analizi:\n"
    analysis += "Son 5 Maç:\n"
    for m in last_5:
        analysis += f"{'(E)' if m['is_home'] else '(D)'} vs {m['opponent']}: {m['score']} ({m['result']})\n"

    location_text = "İç Saha" if is_home else "Deplasman"
    analysis += f"\nSon 5 {location_text} Maçı:\n"
    for m in relevant_matches:
        analysis += f"vs {m['opponent']}: {m['score']} ({m['result']})\n"

    return analysis

def display_predictions_cache():
    try:
        with open('predictions_cache.json', 'r', encoding='utf-8') as f:
            predictions = json.load(f)

        # Analiz sonuçları
        analysis_results = {
            'correct': [],
            'incorrect': [],
            'pending': []
        }

        expected_goals_accuracy = {
            'total': 0,
            'accurate': 0  # Beklenen gol farkı ile gerçek gol farkı arasında 1'den az fark olanlar
        }

        for match_key, prediction in predictions.items():
            if 'home_team' not in prediction or 'away_team' not in prediction:
                continue

            home_name = prediction.get('home_team', {}).get('name', '')
            away_name = prediction.get('away_team', {}).get('name', '')
            match_name = f'{home_name} vs {away_name}'

            # Tahmin edilen skor
            exact_score = prediction.get('predictions', {}).get('betting_predictions', {}).get('exact_score', {}).get('prediction', 'N/A')

            # Beklenen goller
            expected_home = prediction.get('predictions', {}).get('expected_goals', {}).get('home', 0)
            expected_away = prediction.get('predictions', {}).get('expected_goals', {}).get('away', 0)
            expected_goals = f'{expected_home}-{expected_away}'

            # Gerçek sonuç
            actual_result = 'Henüz oynanmadı'
            actual_home_goals = None
            actual_away_goals = None

            # Maç sonucunu bul
            if 'home_team' in prediction and 'form' in prediction['home_team']:
                for match in prediction['home_team']['form'].get('recent_match_data', []):
                    if match.get('opponent') == away_name and match.get('is_home', False):
                        actual_home_goals = match.get('goals_scored')
                        actual_away_goals = match.get('goals_conceded')
                        if actual_home_goals is not None and actual_away_goals is not None:
                            actual_result = f'{actual_home_goals}-{actual_away_goals}'
                            break

            match_data = {
                'match_name': match_name,
                'predicted_score': exact_score,
                'actual_result': actual_result,
                'expected_goals': expected_goals,
                'date_predicted': prediction.get('date_predicted', 'Bilinmiyor'),
                'home_team_data': prediction.get('home_team', {}),
                'away_team_data': prediction.get('away_team', {})
            }

            if actual_result != 'Henüz oynanmadı':
                expected_goals_accuracy['total'] += 1
                expected_home_diff = abs(float(expected_home) - actual_home_goals)
                expected_away_diff = abs(float(expected_away) - actual_away_goals)

                if expected_home_diff < 1 and expected_away_diff < 1:
                    expected_goals_accuracy['accurate'] += 1

                if exact_score == actual_result:
                    analysis_results['correct'].append(match_data)
                else:
                    analysis_results['incorrect'].append(match_data)
            else:
                analysis_results['pending'].append(match_data)

        # Sonuçları yazdır
        print("\n=== TAHMİN ANALİZİ ===\n")

        # Doğru tahminler
        print(f"\nDoğru Tahminler ({len(analysis_results['correct'])} maç):")
        for match in analysis_results['correct']:
            print(f"\nMaç: {match['match_name']}")
            print(f"Tahmin: {match['predicted_score']} | Gerçek: {match['actual_result']}")
            print(f"Beklenen Goller: {match['expected_goals']}")
            print("-" * 50)

        # Yanlış tahminler (detaylı analiz)
        print(f"\nYanlış Tahminler ({len(analysis_results['incorrect'])} maç):")
        for match in analysis_results['incorrect']:
            print(f"\nMaç: {match['match_name']}")
            print(f"Tahmin: {match['predicted_score']} | Gerçek: {match['actual_result']}")
            print(f"Beklenen Goller: {match['expected_goals']}")

            # Form analizleri
            print("\nDetaylı Form Analizi:")
            print(analyze_form_data(match['home_team_data'], match['match_name'].split(' vs ')[0], True))
            print(analyze_form_data(match['away_team_data'], match['match_name'].split(' vs ')[1], False))
            print("-" * 50)

        # Beklenen gol analizi
        if expected_goals_accuracy['total'] > 0:
            accuracy_percentage = (expected_goals_accuracy['accurate'] / expected_goals_accuracy['total']) * 100
            print(f"\nBeklenen Gol Tahmin Doğruluğu: %{accuracy_percentage:.2f}")
            print(f"({expected_goals_accuracy['accurate']}/{expected_goals_accuracy['total']} maçta 1 golden az farkla tahmin)")

        return True

    except FileNotFoundError:
        print("Predictions cache dosyası bulunamadı.")
        return False
    except json.JSONDecodeError:
        print("Cache dosyası geçerli bir JSON formatında değil.")
        return False
    except Exception as e:
        print(f"Bir hata oluştu: {str(e)}")
        return False

if __name__ == "__main__":
    display_predictions_cache()