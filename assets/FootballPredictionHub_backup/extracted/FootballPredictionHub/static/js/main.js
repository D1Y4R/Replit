// Tahmin sonuçlarını özetleyen fonksiyon
function displayPredictionSummary(predictions) {
    if (!predictions || !predictions.most_confident_bet) {
        return;
    }

    const mostConfidentMarket = predictions.most_confident_bet.market;
    const mostConfidentPrediction = predictions.most_confident_bet.prediction;
    const mostConfidentProbability = predictions.most_confident_bet.probability;

    // Tahmin marketini ve sonucunu Türkçe olarak formatla
    let formattedMarket = getMarketNameTurkish(mostConfidentMarket);
    let formattedPrediction = formatPrediction(mostConfidentMarket, mostConfidentPrediction);


    // En yüksek olasılıklı tahmini ekle
    $('#predictionSummary').html(`<p class="mt-3">En yüksek olasılıklı tahmin: ${formattedMarket} - ${formattedPrediction} (${mostConfidentProbability}%)</p>`);
}

// İlk Yarı/Maç Sonu formatı için yardımcı fonksiyon
function formatHalfTimeFullTime(htft) {
    if (!htft) return '';
    const parts = htft.split('/');
    if (parts.length !== 2) return htft;

    let ilkYari = '';
    let macSonu = '';

    // İlk yarı
    if (parts[0] === 'HOME_WIN' || parts[0] === 'MS1') {
        ilkYari = 'Ev';
    } else if (parts[0] === 'DRAW' || parts[0] === 'X') {
        ilkYari = 'Beraberlik';  
    } else if (parts[0] === 'AWAY_WIN' || parts[0] === 'MS2') {
        ilkYari = 'Deplasman';
    } else {
        ilkYari = parts[0];
    }

    // Maç sonu
    if (parts[1] === 'HOME_WIN' || parts[1] === 'MS1') {
        macSonu = 'Ev';
    } else if (parts[1] === 'DRAW' || parts[1] === 'X') {
        macSonu = 'Beraberlik';
    } else if (parts[1] === 'AWAY_WIN' || parts[1] === 'MS2') {
        macSonu = 'Deplasman';
    } else {
        macSonu = parts[1];
    }

    return `${ilkYari}/${macSonu}`;
}

document.addEventListener('DOMContentLoaded', function() {
    // Handle navigation active states
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
    
    // V2 - Tamamen yeni sürüm - Takım istatistikleri butonu (Sürpriz butonu)
    $(document).on('click', '.team-stats-btn-v2', function() {
        const homeTeamId = $(this).data('home-id');
        const awayTeamId = $(this).data('away-id');
        const homeTeamName = $(this).data('home-name') || 'Ev Sahibi';
        const awayTeamName = $(this).data('away-name') || 'Deplasman';
        
        console.log("V2 - Sürpriz butonu tıklandı:", 
            "home_id=", homeTeamId, "away_id=", awayTeamId);
        
        // Yeni fonksiyonu çağır - team-halfTime-stats.js dosyasından
        if (typeof window.showTeamHalfTimeStats === 'function') {
            window.showTeamHalfTimeStats(homeTeamId, awayTeamId, homeTeamName, awayTeamName);
        } else {
            console.error("showTeamHalfTimeStats fonksiyonu bulunamadı!");
            alert("Sürpriz fonksiyonu yüklenemedi. Lütfen sayfayı yenileyip tekrar deneyin.");
        }
    });

    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Widget error handling
    function handleWidgetError(widget, error) {
        console.error('Widget yüklenirken hata:', widget.id, error);
        widget.innerHTML = '<div class="alert alert-danger" role="alert">' +
            '<i class="fas fa-exclamation-triangle me-2"></i>' +
            'Veriler yüklenirken bir hata oluştu. Lütfen sayfayı yenileyiniz.<br>' +
            '<small class="text-muted">Hata detayı: ' + (error.message || 'Bilinmeyen hata') + '</small>' +
            '</div>';
    }

    // Initialize widgets
    document.querySelectorAll('.api_football_loader').forEach(widget => {
        console.log('Widget başlatılıyor:', widget.id);

        // Add loading indicator
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'text-center my-3';
        loadingDiv.innerHTML = '<div class="spinner-border text-primary" role="status">' +
            '<span class="visually-hidden">Yükleniyor...</span></div>' +
            '<div class="mt-2 small text-muted">Veriler yükleniyor...</div>';
        widget.appendChild(loadingDiv);

        // Handle widget errors
        widget.addEventListener('error', function(e) {
            console.error('Widget yükleme hatası:', e);
            handleWidgetError(this, e);
        });
    });

    // Global error handler for widgets
    window.addEventListener('error', function(e) {
        if (e.target && e.target.classList && e.target.classList.contains('api_football_loader')) {
            console.error('Global widget hatası:', e);
            handleWidgetError(e.target, e);
        }
    }, true);

    // Log widget script loading
    const widgetScript = document.querySelector('script[src*="widgets.js"]');
    if (widgetScript) {
        widgetScript.addEventListener('load', function() {
            console.log('Widget script başarıyla yüklendi');
        });
        widgetScript.addEventListener('error', function(e) {
            console.error('Widget script yüklenirken hata oluştu:', e);
            document.querySelectorAll('.api_football_loader').forEach(widget => {
                handleWidgetError(widget, e);
            });
        });
    }

    // Search functionality
    const searchInput = document.getElementById('searchInput');

    function searchFixtures() {
        const searchValue = searchInput ? searchInput.value.toLowerCase() : '';

        document.querySelectorAll('.league-section').forEach(section => {
            const leagueName = section.querySelector('.league-header').textContent.toLowerCase();
            const matches = section.querySelectorAll('.match-item');
            let sectionVisible = false;

            matches.forEach(match => {
                const homeTeam = match.getAttribute('data-home-name').toLowerCase();
                const awayTeam = match.getAttribute('data-away-name').toLowerCase();

                const matchesSearch = leagueName.includes(searchValue) || 
                                    homeTeam.includes(searchValue) || 
                                    awayTeam.includes(searchValue);

                if (matchesSearch || !searchValue) {
                    match.style.display = '';
                    sectionVisible = true;
                } else {
                    match.style.display = 'none';
                }
            });

            section.style.display = sectionVisible ? '' : 'none';
        });
    }

    // Add event listeners for input changes
    if (searchInput) {
        searchInput.addEventListener('input', searchFixtures);
        searchInput.addEventListener('keyup', function(event) {
            if (event.key === 'Enter') {
                searchFixtures();
            }
        });
    }
});

function formatPrediction(betType, prediction) {
    switch (betType) {
        case 'over_2_5_goals':
        case '2.5_üst_alt':  // Alternatif anahtar ekledim
            if (prediction === 'YES' || prediction === '2.5 ÜST') return '2.5 ÜST';
            if (prediction === 'NO' || prediction === '2.5 ALT') return '2.5 ALT';
            return prediction;
        case 'over_3_5_goals':
        case '3.5_üst_alt':  // Alternatif anahtar ekledim
            if (prediction === 'YES' || prediction === '3.5 ÜST') return '3.5 ÜST';
            if (prediction === 'NO' || prediction === '3.5 ALT') return '3.5 ALT';
            return prediction;
        case 'both_teams_to_score':
        case 'kg_var_yok':  // Alternatif anahtar ekledim
            if (prediction === 'YES' || prediction === 'KG VAR') return 'KG VAR';
            if (prediction === 'NO' || prediction === 'KG YOK') return 'KG YOK';
            return prediction;
        case 'match_result':
        case 'maç_sonucu':  // Alternatif anahtar ekledim
            switch(prediction) {
                case 'HOME_WIN': return 'MS1';
                case 'DRAW': return 'X';
                case 'AWAY_WIN': return 'MS2';
                case 'MS1': return 'MS1';
                case 'X': return 'X';
                case 'MS2': return 'MS2';
                default: return prediction;
            }
        case 'first_goal':
        case 'ilk_gol':  // Alternatif anahtar ekledim
            if (prediction === 'HOME') return 'EV';
            if (prediction === 'AWAY') return 'DEP';
            if (prediction === 'NO_GOAL') return 'GOL YOK';
            if (prediction === undefined) return 'Belirsiz';
            if (typeof prediction === 'object' && prediction !== null) {
                if (prediction.team) {
                    if (prediction.team === 'HOME') return 'EV';
                    if (prediction.team === 'AWAY') return 'DEP';
                    return 'GOL YOK';
                }
            }
            return 'İlk Gol';
        case 'half_time_full_time':
        case 'ilk_yarı_maç_sonu':  // Alternatif anahtar ekledim
            return prediction.split('/').map(result => {
                switch(result) {
                    case 'HOME_WIN': case 'MS1': return 'MS1';
                    case 'DRAW': case 'X': return 'X';
                    case 'AWAY_WIN': case 'MS2': return 'MS2';
                    default: return result;
                }
            }).join('/');

        default:
            // Genel YES/NO değerlerini çevir
            if (prediction === 'YES') return 'VAR';
            if (prediction === 'NO') return 'YOK';
            return prediction;
    }
}
// En yüksek olasılıklı tahmini belirle (3.5 Alt/Üst ve first_goal hariç)
function findMostConfidentBet(predictions) {
    const excludedMarkets = ['first_goal', 'first_goal_team', 'over_3_5_goals']; // İlk gol bahsini hariç tut
    if (!predictions || !predictions.betting_predictions) return null;

    const allBets = predictions.betting_predictions;
    let highest = { market: '', prediction: '', probability: 0 };

    // Tüm bahis tiplerini kontrol et (ilk gol hariç)
    for (const market in allBets) {
        // İlk gol ile ilgili bahisleri atla
        if (excludedMarkets.includes(market)) continue;

        if (allBets[market] && allBets[market].probability > highest.probability) {
            highest = {
                market: market,
                prediction: allBets[market].prediction,
                probability: allBets[market].probability
            };
        }
    }

    return highest;
}

// Piyasa adını Türkçe'ye çevir
function getMarketNameTurkish(market) {
    switch(market) {
        case 'match_result': return 'Maç Sonucu';
        case 'both_teams_to_score': return 'KG VAR/YOK';
        case 'over_2_5_goals': return '2.5 Alt/Üst';
        case 'over_3_5_goals': return '3.5 Alt/Üst';
        case 'exact_score': return 'Kesin Skor';
        case 'half_time_full_time': return 'İlk Yarı/Maç Sonu';
        case 'first_goal': return 'İlk Gol'; //Bu satırı silmeyi düşünebilirsiniz.
        case 'first_goal_team': return 'İlk Golü Atacak Takım'; //Bu satırı silmeyi düşünebilirsiniz.
        case 'cards_over_3_5': return 'Kart 3.5 Alt/Üst';
        case 'corners_over_9_5': return 'Korner 9.5 Alt/Üst';
        default: return market;
    }
}

// AI İçgörüleri kaldırıldı