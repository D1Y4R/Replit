(function($) {
    $.fn.widgetLeague = function(options) {
        const settings = $.extend({
            widgetLiveScoreLocation: '#widgetLiveScore',
            apiKey: '013856bafc4f8aa6387fceb53d7a9c91ea1d575f10c32865d9f8a75f60dac3bc',
            refreshInterval: 60000, // 1 minute
        }, options);

        return this.each(function() {
            const $widget = $(this);
            let currentLeagueId = null;

            function loadLeagueMatches(leagueId) {
                currentLeagueId = leagueId;
                $widget.html('<div class="loading">Loading matches...</div>');

                $.ajax({
                    url: 'https://v3.football.api-sports.io/fixtures',
                    headers: {
                        'x-rapidapi-key': settings.apiKey
                    },
                    data: {
                        league: leagueId,
                        season: new Date().getFullYear()
                    },
                    success: function(response) {
                        if (response.response) {
                            displayMatches(response.response);
                        }
                    },
                    error: function(err) {
                        $widget.html('<div class="error">Error loading matches</div>');
                        console.error('API Error:', err);
                    }
                });
            }

            function displayMatches(matches) {
                const $container = $('<div class="widget-league"></div>');
                
                matches.forEach(match => {
                    const $matchItem = $(`
                        <div class="match-item">
                            <div class="match-time">${formatMatchTime(match.fixture.date)}</div>
                            <div class="match-teams">
                                <div class="team">
                                    <img class="team-logo" src="${match.teams.home.logo}" alt="${match.teams.home.name}">
                                    <span class="team-name">${match.teams.home.name}</span>
                                </div>
                                <div class="match-score">
                                    ${getMatchScore(match)}
                                </div>
                                <div class="team">
                                    <img class="team-logo" src="${match.teams.away.logo}" alt="${match.teams.away.name}">
                                    <span class="team-name">${match.teams.away.name}</span>
                                </div>
                            </div>
                            <div class="match-status ${getStatusClass(match.fixture.status.short)}">
                                ${getStatusText(match.fixture.status)}
                            </div>
                        </div>
                    `);

                    $container.append($matchItem);
                });

                $widget.html($container);
            }

            function formatMatchTime(dateStr) {
                const date = new Date(dateStr);
                return date.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });
            }

            function getMatchScore(match) {
                if (match.fixture.status.short === 'NS') {
                    return 'vs';
                }
                return `${match.goals.home} - ${match.goals.away}`;
            }

            function getStatusClass(status) {
                switch (status) {
                    case 'LIVE': return 'status-live';
                    case 'FT': return 'status-finished';
                    default: return 'status-scheduled';
                }
            }

            function getStatusText(status) {
                switch (status.short) {
                    case 'LIVE': return 'CANLI';
                    case 'FT': return 'TAMAMLANDI';
                    case 'NS': return 'BAŞLAMAYI BEKLİYOR';
                    default: return status.long;
                }
            }

            // Event listeners for league selection
            $(document).on('leagueSelected', function(e, leagueId) {
                loadLeagueMatches(leagueId);
            });

            // Auto-refresh for live matches
            setInterval(function() {
                if (currentLeagueId) {
                    loadLeagueMatches(currentLeagueId);
                }
            }, settings.refreshInterval);
        });
    };
}(jQuery));
