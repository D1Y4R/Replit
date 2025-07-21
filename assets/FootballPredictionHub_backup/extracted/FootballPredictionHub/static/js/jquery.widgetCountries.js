(function($) {
    $.fn.widgetCountries = function(options) {
        var settings = $.extend({
            widgetLeagueLocation: '#widgetLeague',
            widgetLiveScoreLocation: '#widgetLiveScore',
            widgetWidth: '100%',
            preferentialLeagues: ['8634', '590']
        }, options);

        return this.each(function() {
            var $widget = $(this);
            
            // Widget initialization
            function initWidget() {
                $widget.addClass('football-widget');
                $widget.css('width', settings.widgetWidth);
                loadMatches();
            }

            // Load matches data
            function loadMatches() {
                $.ajax({
                    url: '/api/v3/fixtures',
                    method: 'GET',
                    success: function(data) {
                        displayMatches(data);
                    },
                    error: function(err) {
                        console.error('Widget error:', err);
                    }
                });
            }

            // Display matches in widget
            function displayMatches(data) {
                var matches = data.response || [];
                var html = '<div class="widget-container">';
                
                // Sort leagues based on preference
                matches.sort(function(a, b) {
                    var aIndex = settings.preferentialLeagues.indexOf(a.league.id);
                    var bIndex = settings.preferentialLeagues.indexOf(b.league.id);
                    
                    if (aIndex === -1) aIndex = 999;
                    if (bIndex === -1) bIndex = 999;
                    
                    return aIndex - bIndex;
                });

                // Group matches by league
                var leagues = {};
                matches.forEach(function(match) {
                    if (!leagues[match.league.id]) {
                        leagues[match.league.id] = {
                            league: match.league,
                            matches: []
                        };
                    }
                    leagues[match.league.id].matches.push(match);
                });

                // Create HTML for each league
                Object.values(leagues).forEach(function(league) {
                    html += createLeagueSection(league);
                });

                html += '</div>';
                $widget.html(html);
            }

            // Create league section HTML
            function createLeagueSection(league) {
                var html = `
                    <div class="league-section">
                        <div class="league-header">
                            <img src="${league.league.logo}" alt="${league.league.name}" class="league-logo">
                            <span class="league-name">${league.league.name}</span>
                        </div>
                        <div class="matches-container">
                `;

                league.matches.forEach(function(match) {
                    // Takım bilgilerini konsola yazdır
                    console.log("Match data for debug:", {
                        matchId: match.fixture?.id,
                        homeTeam: {
                            id: match.teams?.home?.id || 0,
                            name: match.teams?.home?.name || 'Bilinmiyor'
                        },
                        awayTeam: {
                            id: match.teams?.away?.id || 0, 
                            name: match.teams?.away?.name || 'Bilinmiyor'
                        }
                    });
                    html += createMatchRow(match);
                });

                html += '</div></div>';
                return html;
            }

            // Create match row HTML
            function createMatchRow(match) {
                var status = match.fixture.status.short;
                var isLive = ['1H', '2H', 'HT', 'ET', 'P', 'BT'].includes(status);
                var time = isLive ? status : match.fixture.date.split('T')[1].substring(0, 5);

                return `
                    <div class="match-row ${isLive ? 'live' : ''}">
                        <div class="match-time">${time}</div>
                        <div class="match-teams">
                            <div class="team home">
                                <img src="${match.teams.home.logo}" alt="${match.teams.home.name}" class="team-logo">
                                <span class="team-name">${match.teams.home.name}</span>
                            </div>
                            <div class="match-score">
                                ${match.goals.home !== null ? match.goals.home : '-'} - ${match.goals.away !== null ? match.goals.away : '-'}
                            </div>
                            <div class="team away">
                                <img src="${match.teams.away.logo}" alt="${match.teams.away.name}" class="team-logo">
                                <span class="team-name">${match.teams.away.name}</span>
                            </div>
                        </div>
                        <div class="match-actions">
                            <button class="btn btn-sm btn-primary predict-match-btn" 
                                data-home-id="${match.teams.home.id || 0}" 
                                data-away-id="${match.teams.away.id || 0}"
                                data-home-name="${match.teams.home.name || 'Ev Sahibi'}"
                                data-away-name="${match.teams.away.name || 'Deplasman'}">
                                Tahmin
                            </button>
                            <button class="btn btn-sm btn-warning team-stats-btn-v2" 
                                data-home-id="${match.teams.home?.id || 0}" 
                                data-away-id="${match.teams.away?.id || 0}"
                                data-home-name="${match.teams.home?.name || 'Ev Sahibi'}"
                                data-away-name="${match.teams.away?.name || 'Deplasman'}">
                                Sürpriz
                            </button>
                        </div>
                    </div>
                `;
            }

            // Initialize widget
            initWidget();
            
            // Auto-refresh for live matches
            setInterval(loadMatches, 60000);
        });
    };
}(jQuery));
