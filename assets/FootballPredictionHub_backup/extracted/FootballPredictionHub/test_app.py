import os
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
app.secret_key = "test_secret_key"

@app.route('/')
def index():
    """Test anasayfası"""
    return render_template('index.html')

@app.route('/api/test-prediction', methods=['GET'])
def test_prediction():
    """Test tahmini API endpoint'i"""
    response = {
        "home_team": {
            "id": "123",
            "name": "Test Takım 1",
            "form": {
                "last_5_matches": ["W", "L", "D", "W", "W"],
                "points": 10
            }
        },
        "away_team": {
            "id": "456",
            "name": "Test Takım 2",
            "form": {
                "last_5_matches": ["L", "W", "W", "D", "L"],
                "points": 7
            }
        },
        "predictions": {
            "match_winner": {
                "home_win": 0.65,
                "draw": 0.20,
                "away_win": 0.15
            },
            "exact_score": {
                "score": "2-1",
                "probability": 0.18
            },
            "betting_predictions": {
                "over_2_5_goals": {
                    "prediction": "YES",
                    "display_value": "2.5 ÜST",
                    "probability": 0.70
                },
                "btts": {
                    "prediction": "YES",
                    "display_value": "KG VAR",
                    "probability": 0.75
                },
                "total_goals": {
                    "value": "2.8"
                }
            },
            "additional_data": {
                "home_recent_avg_goals": 1.8,
                "away_recent_avg_goals": 1.2,
                "home_recent_form": 0.8,
                "away_recent_form": 0.6,
                "home_motivation": 0.9,
                "away_motivation": 0.7,
                "home_advantage": 0.6,
                "fatigue_factor": {
                    "home": 0.3,
                    "away": 0.5
                },
                "key_players": {
                    "home": ["Oyuncu 1", "Oyuncu 2"],
                    "away": ["Oyuncu 3"]
                }
            },
            "htft_predictions": {
                "1/1": 0.40,
                "1/X": 0.05,
                "1/2": 0.02,
                "X/1": 0.15,
                "X/X": 0.20,
                "X/2": 0.03,
                "2/1": 0.05,
                "2/X": 0.04,
                "2/2": 0.06
            }
        },
        "timestamp": "2025-04-04T22:00:00Z"
    }
    
    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 80))
    app.run(host='0.0.0.0', port=port, debug=True)