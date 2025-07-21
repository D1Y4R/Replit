
// API-Football Widget Debug Script
document.addEventListener('DOMContentLoaded', function() {
    console.log('API-Football Debug: Page loaded');
    
    // Monitor widget element
    const apiFootballWidget = document.getElementById('wg-api-football-games');
    if (apiFootballWidget) {
        console.log('API-Football Debug: Widget found', apiFootballWidget.dataset);
        
        // API key check
        const apiKey = apiFootballWidget.getAttribute('data-key');
        console.log('API-Football Debug: API Key:', apiKey ? 'Present (' + apiKey.substring(0, 4) + '...)' : 'Missing');
        
        // Date check
        const widgetDate = apiFootballWidget.getAttribute('data-date');
        console.log('API-Football Debug: Widget Date:', widgetDate || 'Not specified');
        
        // Monitor widget changes
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'childList') {
                    console.log('API-Football Debug: Widget content changed', apiFootballWidget.children.length);
                    
                    // Check for error messages
                    const errorElements = apiFootballWidget.querySelectorAll('.error-message');
                    if (errorElements.length > 0) {
                        console.error('API-Football Debug: Widget error detected', 
                            Array.from(errorElements).map(el => el.textContent).join(', '));
                    }
                    
                    // Check if API data is loaded
                    const dataElements = apiFootballWidget.querySelectorAll('.match');
                    if (dataElements.length > 0) {
                        console.log('API-Football Debug: Successfully loaded matches', dataElements.length);
                    } else if (apiFootballWidget.querySelector('.no-match')) {
                        console.log('API-Football Debug: No matches found for the selected date');
                    }
                }
            });
        });
        
        // Start observing
        observer.observe(apiFootballWidget, { childList: true, subtree: true });
        
        // Check API availability
        fetch('https://v3.football.api-sports.io/status', {
            method: 'GET',
            headers: {
                'x-rapidapi-key': apiKey,
                'x-rapidapi-host': 'v3.football.api-sports.io'
            }
        })
        .then(response => response.json())
        .then(data => {
            console.log('API-Football Debug: API Status Check', data);
        })
        .catch(error => {
            console.error('API-Football Debug: API Status Check Failed', error);
        });
    } else {
        console.warn('API-Football Debug: Widget not found on page');
    }
    
    // Monitor global widget object
    const checkInterval = setInterval(function() {
        if (window.WgGames) {
            console.log('API-Football Debug: Widget global object found');
            clearInterval(checkInterval);
        }
    }, 1000);
    
    setTimeout(function() {
        clearInterval(checkInterval);
    }, 10000);
});
console.log("API-Football Debug: Page loaded");

document.addEventListener('DOMContentLoaded', function() {
    const widget = document.getElementById('wg-api-football-games');
    
    if (widget) {
        console.log("API-Football Debug: Widget found", {
            host: widget.getAttribute('data-host'),
            key: widget.getAttribute('data-key')?.substring(0, 5) + '...',
            date: widget.getAttribute('data-date'),
            league: widget.getAttribute('data-league'),
            season: widget.getAttribute('data-season'),
            timezone: widget.getAttribute('data-timezone'),
            theme: widget.getAttribute('data-theme'),
            refresh: widget.getAttribute('data-refresh'),
            showToolbar: widget.getAttribute('data-show-toolbar'),
            showErrors: widget.getAttribute('data-show-errors'),
            showLogos: widget.getAttribute('data-show-logos'),
            modalGame: widget.getAttribute('data-modal-game'),
            modalStandings: widget.getAttribute('data-modal-standings'),
            modalShowLogos: widget.getAttribute('data-modal-show-logos')
        });

        // Check API key
        const apiKey = widget.getAttribute('data-key');
        console.log("API-Football Debug: API Key:", apiKey ? "Present (" + apiKey.substring(0, 5) + "...)" : "Missing");
        
        // Check date
        const date = widget.getAttribute('data-date');
        console.log("API-Football Debug: Widget Date:", date);
        
        // Verify API status
        fetch('https://v3.football.api-sports.io/status', {
            method: 'GET',
            headers: {
                'x-rapidapi-host': 'v3.football.api-sports.io',
                'x-rapidapi-key': apiKey
            }
        })
        .then(response => response.json())
        .then(data => {
            console.log("API-Football Debug: API Status Check", data);
        })
        .catch(error => {
            console.error("API-Football Debug: API Status Check Error", error);
        });
    }
    
    // Check if widget script loaded
    const widgetScript = document.querySelector('script[src*="widgets.api-sports.io"]');
    if (widgetScript) {
        console.log("Widget script başarıyla yüklendi");
    } else {
        console.warn("Widget script yüklenemedi!");
    }
});
