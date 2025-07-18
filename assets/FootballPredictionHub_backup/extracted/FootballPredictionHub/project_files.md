# Football Scores Application - Tüm Dosyalar

## 1. main.py
```python
from flask import Flask, render_template, jsonify
import logging
import os
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")

def get_matches():
    headers = {
        'X-Auth-Token': os.environ.get('FOOTBALL_DATA_API_KEY', '2f0c06f149e51424f4c9be24eb70cb8f')
    }

    try:
        # Get today's matches from football-data.org v3 API
        response = requests.get(
            'https://api.football-data.org/v3/matches',
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Error fetching matches: {str(e)}")
        return None

@app.route('/api/team-matches/<int:team_id>')
def get_team_matches(team_id):
    headers = {
        'X-Auth-Token': os.environ.get('FOOTBALL_DATA_API_KEY', '2f0c06f149e51424f4c9be24eb70cb8f')
    }

    try:
        # Get matches for the team (API will return most recent matches first)
        response = requests.get(
            f'https://api.football-data.org/v3/teams/{team_id}/matches',
            params={
                'status': 'FINISHED',
                'limit': 5,
                'season': 2025  # Specifically get matches from 2025 season
            },
            headers=headers
        )
        response.raise_for_status()

        # Get the response data and return the matches
        data = response.json()
        return jsonify({"matches": data.get('matches', [])})

    except Exception as e:
        logging.error(f"Error fetching team matches: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    matches_data = get_matches()
    return render_template('index.html', matches=matches_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

## 2. templates/base.html
```html
<!DOCTYPE html>
<html lang="tr" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Futbol Skorları & Fikstür</title>
    <link rel="stylesheet" href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/custom.css') }}">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/"><i class="fas fa-futbol"></i> Futbol Portalı</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link" href="/"><i class="fas fa-home"></i> Ana Sayfa</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        {% block content %}{% endblock %}
    </div>

    <footer class="footer mt-5 py-3 bg-dark">
        <div class="container text-center">
            <span class="text-muted">© 2024 Futbol Portalı</span>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
    {% block extra_js %}{% endblock %}
</body>
</html>
```

## 3. templates/index.html
```html
{% extends "base.html" %}

{% block title %}Fikstür{% endblock %}

{% block content %}
<div class="row">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <h2 class="h5 mb-0"><i class="fas fa-calendar"></i> Fikstür</h2>
            </div>
            <div class="card-body">
                {% if matches and matches.matches %}
                    {% set competitions = {} %}
                    {% for match in matches.matches %}
                        {% if match.competition.name not in competitions %}
                            {% set _ = competitions.update({match.competition.name: []}) %}
                        {% endif %}
                        {% set _ = competitions[match.competition.name].append(match) %}
                    {% endfor %}

                    {% for competition, competition_matches in competitions.items() %}
                        <div class="league-section mb-4">
                            <h3 class="h6 league-header mb-3">
                                <i class="fas fa-trophy me-2"></i>{{ competition }}
                            </h3>
                            {% for match in competition_matches %}
                                <div class="card mb-2 match-item" data-status="{{ match.status }}" 
                                     data-home-team="{{ match.homeTeam.id }}" 
                                     data-away-team="{{ match.awayTeam.id }}"
                                     data-home-name="{{ match.homeTeam.name }}"
                                     data-away-name="{{ match.awayTeam.name }}"
                                     style="cursor: pointer;">
                                    <div class="card-body py-2">
                                        <div class="d-flex justify-content-between align-items-center">
                                            <div class="team-home">
                                                {{ match.homeTeam.name }}
                                            </div>
                                            <div class="score">
                                                {% if match.status == 'FINISHED' %}
                                                    {{ match.score.fullTime.home }} - {{ match.score.fullTime.away }}
                                                {% elif match.status == 'IN_PLAY' %}
                                                    {{ match.score.fullTime.home }} - {{ match.score.fullTime.away }}
                                                    <span class="badge bg-danger">CANLI</span>
                                                {% else %}
                                                    {{ match.utcDate|replace("T", " ")|replace("Z", "") }}
                                                {% endif %}
                                            </div>
                                            <div class="team-away">
                                                {{ match.awayTeam.name }}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            {% endfor %}
                        </div>
                    {% endfor %}
                {% else %}
                    <div class="alert alert-info">
                        <i class="fas fa-info-circle"></i> Şu anda görüntülenecek maç bulunmuyor.
                    </div>
                {% endif %}
            </div>
        </div>
    </div>
</div>

<!-- Team History Modal -->
<div class="modal fade" id="teamHistoryModal" tabindex="-1">
    <div class="modal-dialog modal-lg">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">Takım Geçmişi</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
            </div>
            <div class="modal-body">
                <div class="row">
                    <div class="col-md-6">
                        <h6 class="home-team-name mb-3"></h6>
                        <div class="home-team-history">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Yükleniyor...</span>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <h6 class="away-team-name mb-3"></h6>
                        <div class="away-team-history">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Yükleniyor...</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script>
document.addEventListener('DOMContentLoaded', function() {
    const modal = new bootstrap.Modal(document.getElementById('teamHistoryModal'));

    // Format date function
    function formatDate(dateStr) {
        const date = new Date(dateStr);
        return date.toLocaleDateString('tr-TR');
    }

    // Fetch team history
    async function fetchTeamHistory(teamId) {
        try {
            const response = await fetch(`/api/team-matches/${teamId}`);
            const data = await response.json();
            return data.matches || [];
        } catch (error) {
            console.error('Error fetching team history:', error);
            return [];
        }
    }

    // Display team history
    function displayTeamHistory(matches, containerSelector) {
        const container = document.querySelector(containerSelector);
        if (matches.length === 0) {
            container.innerHTML = '<div class="alert alert-info">Geçmiş maç bulunamadı.</div>';
            return;
        }

        const html = matches.map(match => `
            <div class="card mb-2">
                <div class="card-body py-2">
                    <small class="text-muted">${formatDate(match.utcDate)}</small><br>
                    <div class="d-flex justify-content-between align-items-center">
                        <div class="team-home">${match.homeTeam.name}</div>
                        <div class="score">${match.score.fullTime.home} - ${match.score.fullTime.away}</div>
                        <div class="team-away">${match.awayTeam.name}</div>
                    </div>
                </div>
            </div>
        `).join('');

        container.innerHTML = html;
    }

    // Handle match click
    document.querySelectorAll('.match-item').forEach(match => {
        match.addEventListener('click', async function() {
            const homeTeamId = this.dataset.homeTeam;
            const awayTeamId = this.dataset.awayTeam;
            const homeTeamName = this.dataset.homeName;
            const awayTeamName = this.dataset.awayName;

            // Update modal titles
            document.querySelector('.home-team-name').textContent = `${homeTeamName} - Son 5 Maç`;
            document.querySelector('.away-team-name').textContent = `${awayTeamName} - Son 5 Maç`;

            // Reset history containers
            document.querySelector('.home-team-history').innerHTML = `
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Yükleniyor...</span>
                </div>`;
            document.querySelector('.away-team-history').innerHTML = `
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Yükleniyor...</span>
                </div>`;

            // Show modal
            modal.show();

            // Fetch and display team histories
            const [homeTeamMatches, awayTeamMatches] = await Promise.all([
                fetchTeamHistory(homeTeamId),
                fetchTeamHistory(awayTeamId)
            ]);

            displayTeamHistory(homeTeamMatches, '.home-team-history');
            displayTeamHistory(awayTeamMatches, '.away-team-history');
        });
    });
});
</script>
{% endblock %}
```

## 4. static/css/custom.css
```css
/* Custom styles to enhance the dark theme */
body {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    background-color: var(--bs-dark);
}

.footer {
    margin-top: auto;
}

.card {
    background: var(--bs-dark);
    border: 1px solid var(--bs-gray-700);
    margin-bottom: 1rem;
}

.card-header {
    background: var(--bs-gray-900);
    border-bottom: 1px solid var(--bs-gray-700);
    padding: 0.75rem 1rem;
}

.card-header h2 {
    display: flex;
    align-items: center;
    margin: 0;
    color: var(--bs-light);
}

.card-body {
    padding: 1rem;
}

/* League section styles */
.league-section {
    border-bottom: 1px solid var(--bs-gray-700);
    padding-bottom: 1rem;
}

.league-section:last-child {
    border-bottom: none;
}

.league-header {
    color: var(--bs-primary);
    font-weight: 600;
    padding: 0.5rem 0;
}

/* Match display styles */
.team-home, .team-away {
    font-weight: 500;
    flex: 1;
}

.team-home {
    text-align: right;
    padding-right: 1rem;
}

.team-away {
    text-align: left;
    padding-left: 1rem;
}

.score {
    font-weight: bold;
    min-width: 100px;
    text-align: center;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .team-home, .team-away {
        font-size: 0.9rem;
    }

    .score {
        min-width: 80px;
        font-size: 0.9rem;
    }
}
```

## 5. static/js/main.js
```javascript
document.addEventListener('DOMContentLoaded', function() {
    // Handle navigation active states
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });

    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});
```
