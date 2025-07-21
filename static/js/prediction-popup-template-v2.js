// Tahmin popup'ı için tam HTML şablonu - Basitleştirilmiş versiyon
function generatePredictionHTML(data) {
    // Veri kontrolü
    if (!data || !data.predictions) {
        return '<div class="alert alert-warning">Tahmin verisi yükleniyor...</div>';
    }
    
    const predictions = data.predictions;
    const homeTeam = data.match_info?.home_team || { name: 'Ev Sahibi', id: '' };
    const awayTeam = data.match_info?.away_team || { name: 'Deplasman', id: '' };
    const teamData = data.team_data || {};
    const homeData = teamData.home || {};
    const awayData = teamData.away || {};
    const confidence = Math.round((data.confidence || 0.75) * 100);
    
    let html = '';
    
    // Başlık - Mobil uyumlu alt alta dizilim
    html += `
        <div class="text-center mb-4">
            <div class="mt-3">
                <div class="mb-2">
                    <h3 class="mb-0">${homeTeam.name}</h3>
                </div>
                <div class="mb-2">
                    <h4 class="text-primary mb-0">VS</h4>
                </div>
                <div>
                    <h3 class="mb-0">${awayTeam.name}</h3>
                </div>
            </div>
        </div>
    `;
    
    // Modern Sayfa Seçici
    html += '<div class="page-selector mb-4">' +
        '<style>' +
        '.page-selector {' +
        '    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);' +
        '    padding: 12px;' +
        '    border-radius: 12px;' +
        '    box-shadow: 0 4px 15px rgba(0,0,0,0.3);' +
        '}' +
        '.page-tab {' +
        '    background: rgba(255,255,255,0.05);' +
        '    border: none;' +
        '    color: #9ca3af;' +
        '    padding: 10px 16px;' +
        '    margin: 4px;' +
        '    border-radius: 8px;' +
        '    font-size: 13px;' +
        '    font-weight: 500;' +
        '    transition: all 0.3s ease;' +
        '    cursor: pointer;' +
        '    position: relative;' +
        '    overflow: hidden;' +
        '}' +
        '.page-tab:hover {' +
        '    background: rgba(255,255,255,0.1);' +
        '    color: #fff;' +
        '    transform: translateY(-2px);' +
        '}' +
        '.page-tab.active {' +
        '    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);' +
        '    color: #fff;' +
        '    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);' +
        '}' +
        '.page-tab i {' +
        '    margin-right: 6px;' +
        '}' +
        '@media (max-width: 576px) {' +
        '    .page-tab {' +
        '        padding: 8px 12px;' +
        '        font-size: 12px;' +
        '    }' +
        '}' +
        '</style>' +
            '<div class="d-flex flex-wrap justify-content-center">' +
                '<button class="page-tab active" onclick="showPredictionPage(1)">' +
                    '<i class="fas fa-chart-line"></i> Temel' +
                '</button>' +
                '<button class="page-tab" onclick="showPredictionPage(2)">' +
                    '<i class="fas fa-clock"></i> İY/MS' +
                '</button>' +
                '<button class="page-tab" onclick="showPredictionPage(3)">' +
                    '<i class="fas fa-balance-scale"></i> Handikap' +
                '</button>' +
                '<button class="page-tab" onclick="showPredictionPage(4)">' +
                    '<i class="fas fa-futbol"></i> Goller' +
                '</button>' +
                '<button class="page-tab" onclick="showPredictionPage(5)">' +
                    '<i class="fas fa-users"></i> Takım/Çifte' +
                '</button>' +
                '<button class="page-tab" onclick="showPredictionPage(6)">' +
                    '<i class="fas fa-brain"></i> Açıklama' +
                '</button>' +
            '</div>' +
        '</div>';
    
    // Sayfa 1 - Temel Tahminler
    html += `<div id="predictionPage1">`;
    
    // Tahmin Özeti
    html += `
        <div class="card mb-3">
            <div class="card-header bg-dark text-white">
                <h5 class="mb-0">Tahmin Özeti</h5>
            </div>
            <div class="card-body">
                <p class="lead text-center mb-3">
                    ${(function() {
                        if (predictions.most_likely_outcome === 'HOME_WIN') {
                            return homeTeam.name + ' kazanması bekleniyor';
                        } else if (predictions.most_likely_outcome === 'AWAY_WIN') {
                            return awayTeam.name + ' kazanması bekleniyor';
                        } else {
                            return 'Beraberlik bekleniyor';
                        }
                    })()}
                </p>
                <div class="text-center mb-3">
                    <strong>En yüksek olasılıklı tahmin: </strong>
                    ${formatMostConfidentBet(predictions)}
                </div>
                <div class="progress" style="height: 25px;">
                    <div class="progress-bar bg-success" role="progressbar" style="width: ${confidence}%">
                        Güven: %${confidence}
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Takım Formları
    html += `
        <div class="row mb-3">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-dark text-white text-center">
                        <h6 class="mb-0">${homeTeam.name}</h6>
                    </div>
                    <div class="card-body">
                        <div class="text-center mb-2">
                            <strong>Son Maçlar:</strong>
                        </div>
                        <div class="d-flex justify-content-center mb-3">
                            ${generateFormBadges(homeData.recent_form || 'WWDLW')}
                        </div>
                        <div class="small">
                            <table class="table table-sm">
                                <tr>
                                    <td>Ortalama Atılan Gol:</td>
                                    <td class="text-end">${homeData.avg_goals_scored || '1.50'}</td>
                                </tr>
                                <tr>
                                    <td>Ortalama Yenilen Gol:</td>
                                    <td class="text-end">${homeData.avg_goals_conceded || '1.20'}</td>
                                </tr>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-dark text-white text-center">
                        <h6 class="mb-0">${awayTeam.name}</h6>
                    </div>
                    <div class="card-body">
                        <div class="text-center mb-2">
                            <strong>Son Maçlar:</strong>
                        </div>
                        <div class="d-flex justify-content-center mb-3">
                            ${generateFormBadges(awayData.recent_form || 'LWDWL')}
                        </div>
                        <div class="small">
                            <table class="table table-sm">
                                <tr>
                                    <td>Ortalama Atılan Gol:</td>
                                    <td class="text-end">${awayData.avg_goals_scored || '1.30'}</td>
                                </tr>
                                <tr>
                                    <td>Ortalama Yenilen Gol:</td>
                                    <td class="text-end">${awayData.avg_goals_conceded || '1.40'}</td>
                                </tr>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Kazanma Olasılıkları
    html += `
        <div class="card mb-3">
            <div class="card-header bg-dark text-white">
                <h5 class="mb-0">Kazanma Olasılıkları</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-4 text-center">
                        <h6>Ev:</h6>
                        <h4 class="text-success">${predictions.home_win_probability}%</h4>
                        <div class="progress" style="height: 10px;">
                            <div class="progress-bar bg-success" style="width: ${predictions.home_win_probability}%"></div>
                        </div>
                    </div>
                    <div class="col-4 text-center">
                        <h6>Beraberlik:</h6>
                        <h4 class="text-warning">${predictions.draw_probability}%</h4>
                        <div class="progress" style="height: 10px;">
                            <div class="progress-bar bg-warning" style="width: ${predictions.draw_probability}%"></div>
                        </div>
                    </div>
                    <div class="col-4 text-center">
                        <h6>Deplasman:</h6>
                        <h4 class="text-danger">${predictions.away_win_probability}%</h4>
                        <div class="progress" style="height: 10px;">
                            <div class="progress-bar bg-danger" style="width: ${predictions.away_win_probability}%"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Beklenen Gol Sayısı
    html += `
        <div class="card mb-3">
            <div class="card-header bg-dark text-white">
                <h5 class="mb-0">Beklenen Gol Sayısı</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-6 text-center">
                        <h3 class="text-primary">${predictions.expected_goals.home}</h3>
                        <p class="mb-0">Ev Sahibi</p>
                    </div>
                    <div class="col-6 text-center">
                        <h3 class="text-primary">${predictions.expected_goals.away}</h3>
                        <p class="mb-0">Deplasman</p>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Bahis Tahminleri (Mevcut sistem)
    html += generateBettingSection(predictions.betting_predictions);
    
    // Sayfa 1'i kapat
    html += '</div>';
    
    // Sayfa 2 - İY/MS Tahminleri
    html += `<div id="predictionPage2" style="display: none;">`;
    if (predictions.advanced_predictions) {
        // HT/FT Tahminleri
        if (predictions.advanced_predictions.htft) {
            html += generateHTFTSection(predictions.advanced_predictions.htft);
        }
        
        // İlk Yarı Gol Tahminleri
        if (predictions.advanced_predictions.halftime_goals) {
            html += generateHalfTimeGoalsSection(predictions.advanced_predictions.halftime_goals);
        }
    } else {
        html += '<div class="alert alert-info">İY/MS tahminleri yükleniyor...</div>';
    }
    html += '</div>';
    
    // Sayfa 3 - Handikap Tahminleri
    html += `<div id="predictionPage3" style="display: none;">`;
    if (predictions.advanced_predictions) {
        // Asya Handikapı
        if (predictions.advanced_predictions.asian_handicap) {
            html += generateAsianHandicapSection(predictions.advanced_predictions.asian_handicap);
        }
        
        // Avrupa Handikapı
        if (predictions.advanced_predictions.european_handicap) {
            html += generateEuropeanHandicapSection(predictions.advanced_predictions.european_handicap);
        }
    } else {
        html += '<div class="alert alert-info">Handikap tahminleri yükleniyor...</div>';
    }
    html += '</div>';
    
    // Sayfa 4 - Gol Tahminleri
    html += `<div id="predictionPage4" style="display: none;">`;
    if (predictions.advanced_predictions) {
        // Gol Aralıkları
        if (predictions.advanced_predictions.goal_ranges) {
            html += generateGoalRangesSection(predictions.advanced_predictions.goal_ranges);
        }
        
        // Toplam Gol Marketleri
        if (predictions.advanced_predictions.total_goals_markets) {
            html += generateTotalGoalsSection(predictions.advanced_predictions.total_goals_markets);
        }
    } else {
        html += '<div class="alert alert-info">Gol tahminleri yükleniyor...</div>';
    }
    html += '</div>';
    
    // Sayfa 5 - Takım Gol ve Çifte Şans
    html += `<div id="predictionPage5" style="display: none;">`;
    if (predictions.advanced_predictions) {
        // Takım Gol Tahminleri
        if (predictions.advanced_predictions.team_goals) {
            html += generateTeamGoalsSection(predictions.advanced_predictions.team_goals);
        }
        
        // Çifte Şans
        if (predictions.advanced_predictions.double_chance) {
            html += generateDoubleChanceSection(predictions.advanced_predictions.double_chance);
        }
    } else {
        html += '<div class="alert alert-info">Takım tahminleri yükleniyor...</div>';
    }
    html += '</div>';
    
    // Sayfa 6 - Açıklamalar (Explainable AI)
    html += `<div id="predictionPage6" style="display: none;">`;
    if (data.explanation) {
        html += generateExplanationSection(data.explanation);
    } else {
        html += '<div class="alert alert-info">Açıklama verisi bulunmuyor.</div>';
    }
    
    // H2H (Karşılıklı Maç Geçmişi)
    if (data.h2h_data || predictions.h2h_history) {
        html += generateH2HSection(data.h2h_data || predictions.h2h_history, homeTeam.name, awayTeam.name);
    }
    
    html += '</div>';
    
    return html;
}

// Form badge'lerini oluştur
function generateFormBadges(formString) {
    if (!formString) return '<span class="text-muted">Veri yok</span>';
    
    return formString.split('').map(result => {
        let bgClass = '';
        switch(result) {
            case 'W': bgClass = 'bg-success'; break;
            case 'D': bgClass = 'bg-warning'; break;
            case 'L': bgClass = 'bg-danger'; break;
            default: bgClass = 'bg-secondary';
        }
        return `<span class="badge ${bgClass} mx-1 p-2" style="width: 35px; height: 35px; display: inline-flex; align-items: center; justify-content: center; border-radius: 50%;">${result}</span>`;
    }).join('');
}

// En yüksek olasılıklı tahmin
function formatMostConfidentBet(predictions) {
    const mostConfident = predictions.most_confident_bet;
    if (!mostConfident || !mostConfident.market) {
        return 'Tahmin hesaplanıyor...';
    }
    
    const marketNames = {
        'both_teams_to_score': 'KG Var/Yok',
        'over_2_5_goals': '2.5 Üst/Alt',
        'over_3_5_goals': '3.5 Üst/Alt',
        'match_result': 'Maç Sonucu',
        'exact_score': 'Kesin Skor'
    };
    
    const marketName = marketNames[mostConfident.market] || mostConfident.market;
    let predictionText = mostConfident.prediction;
    
    // Tahmin değerini formatla
    if (mostConfident.market === 'both_teams_to_score') {
        predictionText = mostConfident.prediction === 'YES' ? 'KG VAR' : 'KG YOK';
    } else if (mostConfident.market === 'over_2_5_goals') {
        predictionText = mostConfident.prediction === 'YES' ? '2.5 ÜST' : '2.5 ALT';
    } else if (mostConfident.market === 'over_3_5_goals') {
        predictionText = mostConfident.prediction === 'YES' ? '3.5 ÜST' : '3.5 ALT';
    } else if (mostConfident.market === 'match_result') {
        if (mostConfident.prediction === 'HOME_WIN') predictionText = 'Ev Sahibi';
        else if (mostConfident.prediction === 'AWAY_WIN') predictionText = 'Deplasman';
        else if (mostConfident.prediction === 'DRAW') predictionText = 'Beraberlik';
    }
    
    return `${marketName} - ${predictionText} (%${Math.round(mostConfident.probability)})`;
}

// Bahis tahminleri bölümü
function generateBettingSection(betting) {
    if (!betting) return '';
    
    let html = `
        <div class="card mb-3">
            <div class="card-header bg-dark text-white">
                <h5 class="mb-0">Bahis Tahminleri</h5>
            </div>
            <div class="card-body">
                <div class="row">
    `;
    
    // KG Var/Yok
    if (betting.both_teams_to_score) {
        const btts = betting.both_teams_to_score;
        const bttsValue = btts.prediction === 'YES' ? 'KG VAR' : 'KG YOK';
        const bttsProb = Math.round(btts.probability);
        
        html += `
            <div class="col-md-6 mb-3">
                <div class="card h-100">
                    <div class="card-header text-center bg-secondary text-white">
                        <h6 class="mb-0">KG Var/Yok</h6>
                    </div>
                    <div class="card-body text-center">
                        <h4 class="mb-2">${bttsValue}</h4>
                        <div class="progress" style="height: 20px;">
                            <div class="progress-bar bg-info" style="width: ${bttsProb}%">
                                %${bttsProb}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    // 2.5 Üst/Alt
    if (betting.over_2_5_goals) {
        const over25 = betting.over_2_5_goals;
        const over25Value = over25.prediction === 'YES' ? '2.5 ÜST' : '2.5 ALT';
        const over25Prob = Math.round(over25.probability);
        
        html += `
            <div class="col-md-6 mb-3">
                <div class="card h-100">
                    <div class="card-header text-center bg-secondary text-white">
                        <h6 class="mb-0">2.5 Üst/Alt</h6>
                    </div>
                    <div class="card-body text-center">
                        <h4 class="mb-2">${over25Value}</h4>
                        <div class="progress" style="height: 20px;">
                            <div class="progress-bar bg-info" style="width: ${over25Prob}%">
                                %${over25Prob}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    // 3.5 Üst/Alt
    if (betting.over_3_5_goals) {
        const over35 = betting.over_3_5_goals;
        const over35Value = over35.prediction === 'YES' ? '3.5 ÜST' : '3.5 ALT';
        const over35Prob = Math.round(over35.probability);
        
        html += `
            <div class="col-md-6 mb-3">
                <div class="card h-100">
                    <div class="card-header text-center bg-secondary text-white">
                        <h6 class="mb-0">3.5 Üst/Alt</h6>
                    </div>
                    <div class="card-body text-center">
                        <h4 class="mb-2">${over35Value}</h4>
                        <div class="progress" style="height: 20px;">
                            <div class="progress-bar bg-info" style="width: ${over35Prob}%">
                                %${over35Prob}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Kesin Skor
    if (betting.exact_score) {
        const exactScore = betting.exact_score;
        const scoreProb = Math.round(exactScore.probability);
        
        html += `
            <div class="col-md-6 mb-3">
                <div class="card h-100">
                    <div class="card-header text-center bg-secondary text-white">
                        <h6 class="mb-0">Kesin Skor</h6>
                    </div>
                    <div class="card-body text-center">
                        <h4 class="mb-2">${exactScore.prediction}</h4>
                        <div class="progress" style="height: 20px;">
                            <div class="progress-bar bg-info" style="width: ${scoreProb}%">
                                %${scoreProb}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    html += `
                </div>
            </div>
        </div>
    `;
    
    return html;
}

// Global fonksiyon olarak tanımla
window.generatePredictionHTML = generatePredictionHTML;

// Sayfa değiştirme fonksiyonu
window.showPredictionPage = function(pageNumber) {
    // Tüm sayfaları gizle
    for (let i = 1; i <= 5; i++) {
        document.getElementById('predictionPage' + i).style.display = 'none';
    }
    
    // Tüm butonlardan active class'ı kaldır
    const allTabs = document.querySelectorAll('.page-tab');
    allTabs.forEach(tab => tab.classList.remove('active'));
    
    // Tıklanan butona active class ekle
    event.target.closest('.page-tab').classList.add('active');
    
    // Seçilen sayfayı göster
    document.getElementById('predictionPage' + pageNumber).style.display = 'block';
};

// HT/FT tahminleri bölümü - Modern tasarım
function generateHTFTSection(htft) {
    if (!htft || !htft.predictions) return '<div class="text-center text-muted">HT/FT tahminleri yükleniyor...</div>';
    
    let html = `
        <div class="modern-prediction-section">
            <style>
                .modern-prediction-section {
                    background: rgba(255, 255, 255, 0.03);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 16px;
                    padding: 20px;
                    margin-bottom: 20px;
                    backdrop-filter: blur(10px);
                }
                .section-header {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    margin-bottom: 20px;
                }
                .section-icon {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    width: 40px;
                    height: 40px;
                    border-radius: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
                }
                .most-likely-banner {
                    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
                    border: 1px solid rgba(16, 185, 129, 0.3);
                    border-radius: 12px;
                    padding: 15px;
                    margin-bottom: 20px;
                    text-align: center;
                }
                .htft-grid {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 12px;
                }
                .htft-card {
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 10px;
                    padding: 12px;
                    text-align: center;
                    transition: all 0.3s ease;
                    border: 1px solid transparent;
                    cursor: pointer;
                }
                .htft-card:hover {
                    background: rgba(255, 255, 255, 0.08);
                    transform: translateY(-2px);
                    border-color: rgba(255, 255, 255, 0.2);
                }
                .htft-card.highlighted {
                    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%);
                    border-color: rgba(59, 130, 246, 0.5);
                }
                .htft-label {
                    font-size: 13px;
                    color: #9ca3af;
                    margin-bottom: 5px;
                }
                .htft-percentage {
                    font-size: 18px;
                    font-weight: 700;
                    color: #fff;
                }
            </style>
            
            <div class="section-header">
                <div class="section-icon">
                    <i class="fas fa-clock" style="color: #fff; font-size: 18px;"></i>
                </div>
                <h5 style="margin: 0; color: #fff;">İlk Yarı / Maç Sonu Tahminleri</h5>
            </div>
            
            <div class="most-likely-banner">
                <div style="color: #10b981; font-weight: 600;">
                    <i class="fas fa-star"></i> En Olası Sonuç
                </div>
                <div style="font-size: 20px; color: #fff; margin-top: 5px;">
                    ${formatHTFTResult(htft.most_likely)}
                </div>
                <div style="font-size: 24px; font-weight: 700; color: #10b981;">
                    %${Math.round(htft.most_likely_prob)}
                </div>
            </div>
            
            <div class="htft-grid">
    `;
    
    const htftCombinations = [
        { key: 'HOME_HOME', label: 'Ev-Ev', icon: '🏠→🏠' },
        { key: 'HOME_DRAW', label: 'Ev-Ber', icon: '🏠→🤝' },
        { key: 'HOME_AWAY', label: 'Ev-Dep', icon: '🏠→✈️' },
        { key: 'DRAW_HOME', label: 'Ber-Ev', icon: '🤝→🏠' },
        { key: 'DRAW_DRAW', label: 'Ber-Ber', icon: '🤝→🤝' },
        { key: 'DRAW_AWAY', label: 'Ber-Dep', icon: '🤝→✈️' },
        { key: 'AWAY_HOME', label: 'Dep-Ev', icon: '✈️→🏠' },
        { key: 'AWAY_DRAW', label: 'Dep-Ber', icon: '✈️→🤝' },
        { key: 'AWAY_AWAY', label: 'Dep-Dep', icon: '✈️→✈️' }
    ];
    
    // Tahminleri olasılığa göre sırala (büyükten küçüğe)
    const sortedCombinations = htftCombinations
        .map(combo => ({
            ...combo,
            probability: htft.predictions[combo.key] || 0
        }))
        .sort((a, b) => b.probability - a.probability);
    
    sortedCombinations.forEach((combo, index) => {
        const isHighest = combo.key === htft.most_likely;
        const isTop3 = index < 3; // İlk 3 en yüksek olasılık
        html += `
            <div class="htft-card ${isHighest ? 'highlighted' : ''} ${isTop3 ? 'top-probability' : ''}" style="${isTop3 ? 'border-color: rgba(251, 191, 36, 0.3);' : ''}">
                <div style="font-size: 16px; margin-bottom: 5px;">${combo.icon}</div>
                <div class="htft-label">${combo.label}</div>
                <div class="htft-percentage" style="${isTop3 ? 'color: #fbbf24;' : ''}">${Math.round(combo.probability)}%</div>
                ${index === 0 ? '<div style="font-size: 10px; color: #10b981; margin-top: 5px;">EN YÜKSEK</div>' : ''}
            </div>
        `;
    });
    
    html += '</div></div>';
    return html;
}

// İlk yarı gol tahminleri - Modern tasarım
function generateHalfTimeGoalsSection(htGoals) {
    if (!htGoals) return '';
    
    return `
        <div class="modern-prediction-section">
            <style>
                .goal-card {
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 12px;
                    padding: 20px;
                    margin-bottom: 15px;
                    transition: all 0.3s ease;
                }
                .goal-card:hover {
                    background: rgba(255, 255, 255, 0.08);
                    transform: translateY(-2px);
                    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
                }
                .goal-line {
                    font-size: 24px;
                    font-weight: 700;
                    color: #fbbf24;
                    margin-bottom: 15px;
                    text-align: center;
                }
                .goal-percentage {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin: 10px 0;
                }
                .percentage-label {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    font-size: 14px;
                    color: #9ca3af;
                }
                .percentage-value {
                    font-size: 18px;
                    font-weight: 600;
                    color: #fff;
                }
                .progress-bar-container {
                    height: 8px;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 4px;
                    overflow: hidden;
                    margin: 8px 0;
                }
                .progress-bar-fill {
                    height: 100%;
                    background: linear-gradient(90deg, #fbbf24, #f59e0b);
                    transition: width 0.3s ease;
                }
                .expected-goals-banner {
                    background: linear-gradient(135deg, rgba(251, 191, 36, 0.1) 0%, rgba(251, 191, 36, 0.05) 100%);
                    border: 1px solid rgba(251, 191, 36, 0.3);
                    border-radius: 12px;
                    padding: 15px;
                    text-align: center;
                    margin-top: 15px;
                }
            </style>
            
            <div class="section-header">
                <div class="section-icon" style="background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);">
                    <i class="fas fa-stopwatch" style="color: #fff; font-size: 18px;"></i>
                </div>
                <h5 style="margin: 0; color: #fff;">İlk Yarı Gol Tahminleri</h5>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="goal-card">
                        <div class="goal-line">0.5</div>
                        <div class="goal-percentage">
                            <div class="percentage-label">
                                <i class="fas fa-arrow-up" style="color: #10b981;"></i>
                                ÜST
                            </div>
                            <div class="percentage-value" style="color: #10b981;">
                                ${Math.round(htGoals.over_0_5)}%
                            </div>
                        </div>
                        <div class="progress-bar-container">
                            <div class="progress-bar-fill" style="width: ${htGoals.over_0_5}%; background: #10b981;"></div>
                        </div>
                        <div class="goal-percentage">
                            <div class="percentage-label">
                                <i class="fas fa-arrow-down" style="color: #ef4444;"></i>
                                ALT
                            </div>
                            <div class="percentage-value" style="color: #ef4444;">
                                ${Math.round(htGoals.under_0_5)}%
                            </div>
                        </div>
                        <div class="progress-bar-container">
                            <div class="progress-bar-fill" style="width: ${htGoals.under_0_5}%; background: #ef4444;"></div>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <div class="goal-card">
                        <div class="goal-line">1.5</div>
                        <div class="goal-percentage">
                            <div class="percentage-label">
                                <i class="fas fa-arrow-up" style="color: #10b981;"></i>
                                ÜST
                            </div>
                            <div class="percentage-value" style="color: #10b981;">
                                ${Math.round(htGoals.over_1_5)}%
                            </div>
                        </div>
                        <div class="progress-bar-container">
                            <div class="progress-bar-fill" style="width: ${htGoals.over_1_5}%; background: #10b981;"></div>
                        </div>
                        <div class="goal-percentage">
                            <div class="percentage-label">
                                <i class="fas fa-arrow-down" style="color: #ef4444;"></i>
                                ALT
                            </div>
                            <div class="percentage-value" style="color: #ef4444;">
                                ${Math.round(htGoals.under_1_5)}%
                            </div>
                        </div>
                        <div class="progress-bar-container">
                            <div class="progress-bar-fill" style="width: ${htGoals.under_1_5}%; background: #ef4444;"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="expected-goals-banner">
                <div style="color: #fbbf24; font-weight: 600;">
                    <i class="fas fa-chart-line"></i> Beklenen İlk Yarı Gol Sayısı
                </div>
                <div style="font-size: 28px; font-weight: 700; color: #fff; margin-top: 5px;">
                    ${htGoals.expected_ht_goals}
                </div>
            </div>
        </div>
    `;
}

// Asya handikapı bölümü - Modern tasarım
function generateAsianHandicapSection(asianHandicap) {
    if (!asianHandicap || !asianHandicap.predictions) return '<div class="text-center text-muted">Asya handikapı tahminleri yükleniyor...</div>';
    
    let html = `
        <div class="modern-prediction-section">
            <style>
                .handicap-container {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px;
                }
                .handicap-card {
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 12px;
                    padding: 15px;
                    transition: all 0.3s ease;
                }
                .handicap-card:hover {
                    background: rgba(255, 255, 255, 0.08);
                    transform: translateY(-2px);
                    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
                }
                .handicap-card.recommended {
                    background: linear-gradient(135deg, rgba(236, 72, 153, 0.1) 0%, rgba(236, 72, 153, 0.05) 100%);
                    border-color: rgba(236, 72, 153, 0.5);
                }
                .handicap-value {
                    font-size: 22px;
                    font-weight: 700;
                    color: #ec4899;
                    text-align: center;
                    margin-bottom: 10px;
                }
                .handicap-odds {
                    display: flex;
                    justify-content: space-between;
                    margin-top: 10px;
                }
                .odds-item {
                    text-align: center;
                    flex: 1;
                }
                .odds-label {
                    font-size: 12px;
                    color: #6b7280;
                    margin-bottom: 5px;
                }
                .odds-value {
                    font-size: 18px;
                    font-weight: 600;
                    color: #fff;
                }
                .recommended-banner {
                    background: linear-gradient(135deg, rgba(236, 72, 153, 0.1) 0%, rgba(236, 72, 153, 0.05) 100%);
                    border: 1px solid rgba(236, 72, 153, 0.3);
                    border-radius: 12px;
                    padding: 15px;
                    margin-bottom: 20px;
                    text-align: center;
                }
                .goal-diff-badge {
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 8px;
                    padding: 10px;
                    margin-top: 15px;
                    text-align: center;
                }
            </style>
            
            <div class="section-header">
                <div class="section-icon" style="background: linear-gradient(135deg, #ec4899 0%, #be185d 100%);">
                    <i class="fas fa-balance-scale" style="color: #fff; font-size: 18px;"></i>
                </div>
                <h5 style="margin: 0; color: #fff;">Asya Handikapı Tahminleri</h5>
            </div>
            
            <div class="recommended-banner">
                <div style="color: #ec4899; font-weight: 600;">
                    <i class="fas fa-trophy"></i> Önerilen Handikap
                </div>
                <div style="font-size: 24px; color: #fff; margin: 10px 0;">
                    ${asianHandicap.best_handicap.handicap > 0 ? '+' : ''}${asianHandicap.best_handicap.handicap}
                </div>
                <div style="color: #10b981; font-weight: 600;">
                    Güven: %${Math.round(asianHandicap.best_handicap.confidence)}
                </div>
            </div>
            
            <div class="handicap-container">
    `;
    
    // Handikapları sırala
    const sortedHandicaps = Object.entries(asianHandicap.predictions)
        .sort((a, b) => b[1].handicap - a[1].handicap)
        .slice(0, 6); // En önemli 6 handikap
    
    sortedHandicaps.forEach(([key, pred]) => {
        const isRecommended = pred.recommended;
        html += `
            <div class="handicap-card ${isRecommended ? 'recommended' : ''}">
                <div class="handicap-value">
                    ${pred.handicap > 0 ? '+' : ''}${pred.handicap}
                </div>
                <div class="handicap-odds">
                    <div class="odds-item">
                        <div class="odds-label">Ev Sahibi</div>
                        <div class="odds-value" style="color: ${pred.home_win > 50 ? '#10b981' : '#fff'};">
                            ${Math.round(pred.home_win)}%
                        </div>
                    </div>
                    <div class="odds-item">
                        <div class="odds-label">Deplasman</div>
                        <div class="odds-value" style="color: ${pred.away_win > 50 ? '#10b981' : '#fff'};">
                            ${Math.round(pred.away_win)}%
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
    
    html += `
            </div>
            <div class="goal-diff-badge">
                <i class="fas fa-chart-bar"></i> Beklenen Gol Farkı: <strong>${asianHandicap.expected_goal_diff}</strong>
            </div>
        </div>
    `;
    
    return html;
}

// Avrupa handikapı bölümü - Modern tasarım
function generateEuropeanHandicapSection(europeanHandicap) {
    if (!europeanHandicap || !europeanHandicap.predictions) return '';
    
    let html = `
        <div class="modern-prediction-section">
            <style>
                .euro-handicap-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                    gap: 15px;
                }
                .euro-handicap-card {
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 12px;
                    padding: 15px;
                    transition: all 0.3s ease;
                }
                .euro-handicap-card:hover {
                    background: rgba(255, 255, 255, 0.08);
                    transform: translateY(-2px);
                    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
                }
                .euro-handicap-value {
                    font-size: 20px;
                    font-weight: 700;
                    color: #8b5cf6;
                    text-align: center;
                    margin-bottom: 15px;
                    padding: 10px;
                    background: rgba(139, 92, 246, 0.1);
                    border-radius: 8px;
                }
                .euro-odds-container {
                    display: flex;
                    justify-content: space-between;
                    gap: 10px;
                }
                .euro-odds-item {
                    flex: 1;
                    text-align: center;
                    padding: 10px;
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 8px;
                    border: 1px solid rgba(255, 255, 255, 0.05);
                }
                .euro-odds-label {
                    font-size: 11px;
                    color: #6b7280;
                    margin-bottom: 5px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                .euro-odds-value {
                    font-size: 16px;
                    font-weight: 600;
                    color: #fff;
                }
                .euro-odds-item.highest {
                    background: rgba(16, 185, 129, 0.1);
                    border-color: rgba(16, 185, 129, 0.3);
                }
                .euro-odds-item.highest .euro-odds-value {
                    color: #10b981;
                }
            </style>
            
            <div class="section-header">
                <div class="section-icon" style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);">
                    <i class="fas fa-flag-checkered" style="color: #fff; font-size: 18px;"></i>
                </div>
                <h5 style="margin: 0; color: #fff;">Avrupa Handikapı Tahminleri</h5>
            </div>
            
            <div class="euro-handicap-grid">
    `;
    
    const sortedHandicaps = Object.entries(europeanHandicap.predictions)
        .sort((a, b) => b[1].handicap - a[1].handicap)
        .slice(0, 6); // En önemli 6 handikap
    
    sortedHandicaps.forEach(([key, pred]) => {
        const highest = Math.max(pred.home_win, pred.draw, pred.away_win);
        html += `
            <div class="euro-handicap-card">
                <div class="euro-handicap-value">
                    ${pred.handicap > 0 ? '+' : ''}${pred.handicap}
                </div>
                <div class="euro-odds-container">
                    <div class="euro-odds-item ${pred.home_win === highest ? 'highest' : ''}">
                        <div class="euro-odds-label">Ev</div>
                        <div class="euro-odds-value">${Math.round(pred.home_win)}%</div>
                    </div>
                    <div class="euro-odds-item ${pred.draw === highest ? 'highest' : ''}">
                        <div class="euro-odds-label">Ber</div>
                        <div class="euro-odds-value">${Math.round(pred.draw)}%</div>
                    </div>
                    <div class="euro-odds-item ${pred.away_win === highest ? 'highest' : ''}">
                        <div class="euro-odds-label">Dep</div>
                        <div class="euro-odds-value">${Math.round(pred.away_win)}%</div>
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div></div>';
    return html;
}

// Gol aralıkları bölümü - Modern tasarım
function generateGoalRangesSection(goalRanges) {
    if (!goalRanges || !goalRanges.predictions) return '<div class="text-center text-muted">Gol aralığı tahminleri yükleniyor...</div>';
    
    let html = `
        <div class="modern-prediction-section">
            <style>
                .goal-ranges-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 15px;
                    margin-bottom: 20px;
                }
                .goal-range-card {
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 12px;
                    padding: 20px;
                    text-align: center;
                    transition: all 0.3s ease;
                    position: relative;
                    overflow: hidden;
                }
                .goal-range-card:hover {
                    background: rgba(255, 255, 255, 0.08);
                    transform: translateY(-3px);
                    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
                }
                .goal-range-card.highest {
                    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%);
                    border-color: rgba(34, 197, 94, 0.5);
                }
                .goal-range-card.highest::before {
                    content: '⭐';
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    font-size: 16px;
                }
                .goal-range-label {
                    font-size: 18px;
                    font-weight: 700;
                    color: #22c55e;
                    margin-bottom: 10px;
                }
                .goal-range-percentage {
                    font-size: 32px;
                    font-weight: 800;
                    color: #fff;
                    margin-bottom: 5px;
                }
                .goal-range-expected {
                    font-size: 11px;
                    color: #6b7280;
                }
                .match-info-banner {
                    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%);
                    border: 1px solid rgba(34, 197, 94, 0.3);
                    border-radius: 12px;
                    padding: 20px;
                    display: flex;
                    justify-content: space-around;
                    align-items: center;
                    text-align: center;
                }
                .match-info-item {
                    flex: 1;
                }
                .match-info-label {
                    font-size: 12px;
                    color: #6b7280;
                    margin-bottom: 5px;
                }
                .match-info-value {
                    font-size: 20px;
                    font-weight: 700;
                    color: #22c55e;
                }
            </style>
            
            <div class="section-header">
                <div class="section-icon" style="background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);">
                    <i class="fas fa-chart-bar" style="color: #fff; font-size: 18px;"></i>
                </div>
                <h5 style="margin: 0; color: #fff;">Gol Aralığı Tahminleri</h5>
            </div>
            
            <div class="goal-ranges-grid">
    `;
    
    Object.entries(goalRanges.predictions).forEach(([range, data]) => {
        const isHighest = range === goalRanges.most_likely_range;
        html += `
            <div class="goal-range-card ${isHighest ? 'highest' : ''}">
                <div class="goal-range-label">${range}</div>
                <div class="goal-range-percentage">${Math.round(data.probability)}%</div>
                <div class="goal-range-expected">Beklenen: ${data.expected_in_range}</div>
            </div>
        `;
    });
    
    html += `
            </div>
            
            <div class="match-info-banner">
                <div class="match-info-item">
                    <div class="match-info-label">Toplam Beklenen Gol</div>
                    <div class="match-info-value">${goalRanges.total_expected_goals}</div>
                </div>
                <div class="match-info-item">
                    <div class="match-info-label">Maç Tipi</div>
                    <div class="match-info-value">${goalRanges.match_type}</div>
                </div>
                <div class="match-info-item">
                    <div class="match-info-label">En Olası Aralık</div>
                    <div class="match-info-value">${goalRanges.most_likely_range}</div>
                </div>
            </div>
        </div>
    `;
    
    return html;
}

// Toplam gol marketleri - Modern tasarım
function generateTotalGoalsSection(totalGoals) {
    if (!totalGoals) return '';
    
    let html = `
        <div class="modern-prediction-section">
            <style>
                .total-goals-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                }
                .total-goals-card {
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 12px;
                    padding: 15px;
                    transition: all 0.3s ease;
                }
                .total-goals-card:hover {
                    background: rgba(255, 255, 255, 0.08);
                    transform: translateY(-2px);
                    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
                }
                .goals-threshold {
                    font-size: 22px;
                    font-weight: 700;
                    color: #3b82f6;
                    text-align: center;
                    margin-bottom: 15px;
                }
                .goals-prediction {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 10px;
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 8px;
                    margin-bottom: 10px;
                    transition: all 0.2s ease;
                }
                .goals-prediction:hover {
                    background: rgba(255, 255, 255, 0.05);
                }
                .goals-label {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    font-size: 14px;
                    color: #9ca3af;
                }
                .goals-value {
                    font-size: 18px;
                    font-weight: 600;
                }
                .over-value {
                    color: #10b981;
                }
                .under-value {
                    color: #ef4444;
                }
                .recommended-tag {
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    color: #fff;
                    font-size: 10px;
                    padding: 2px 8px;
                    border-radius: 12px;
                    font-weight: 600;
                    margin-left: 8px;
                }
            </style>
            
            <div class="section-header">
                <div class="section-icon" style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);">
                    <i class="fas fa-sort-numeric-up" style="color: #fff; font-size: 18px;"></i>
                </div>
                <h5 style="margin: 0; color: #fff;">Toplam Gol Marketleri</h5>
            </div>
            
            <div class="total-goals-grid">
    `;
    
    Object.entries(totalGoals).forEach(([threshold, data]) => {
        const isOverFavorite = data.over > 50;
        const strongestSide = isOverFavorite ? data.over : data.under;
        
        html += `
            <div class="total-goals-card">
                <div class="goals-threshold">${threshold} Gol</div>
                
                <div class="goals-prediction">
                    <div class="goals-label">
                        <i class="fas fa-arrow-up" style="color: #10b981;"></i>
                        ÜST
                    </div>
                    <div class="goals-value over-value">
                        ${Math.round(data.over)}%
                        ${data.over > 65 ? '<span class="recommended-tag">ÖNERİ</span>' : ''}
                    </div>
                </div>
                
                <div class="goals-prediction">
                    <div class="goals-label">
                        <i class="fas fa-arrow-down" style="color: #ef4444;"></i>
                        ALT
                    </div>
                    <div class="goals-value under-value">
                        ${Math.round(data.under)}%
                        ${data.under > 65 ? '<span class="recommended-tag">ÖNERİ</span>' : ''}
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div></div>';
    return html;
}

// Takım gol tahminleri - Modern tasarım
function generateTeamGoalsSection(teamGoals) {
    if (!teamGoals || !teamGoals.home_team) return '<div class="text-center text-muted">Takım gol tahminleri yükleniyor...</div>';
    
    let html = `
        <div class="modern-prediction-section">
            <style>
                .team-goals-container {
                    display: grid;
                    grid-template-columns: 1fr;
                    gap: 20px;
                    margin-bottom: 20px;
                }
                @media (min-width: 768px) {
                    .team-goals-container {
                        grid-template-columns: repeat(2, 1fr);
                    }
                }
                .team-card {
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 16px;
                    overflow: hidden;
                    transition: all 0.3s ease;
                    max-width: 100%;
                }
                .team-card:hover {
                    transform: translateY(-3px);
                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
                }
                .team-header {
                    padding: 15px;
                    text-align: center;
                    color: #fff;
                    font-weight: 700;
                    font-size: 16px;
                }
                @media (min-width: 768px) {
                    .team-header {
                        padding: 20px;
                        font-size: 18px;
                    }
                }
                .home-team-header {
                    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
                }
                .away-team-header {
                    background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
                }
                .most-likely-goals {
                    text-align: center;
                    padding: 15px;
                    background: rgba(255, 255, 255, 0.03);
                }
                .most-likely-number {
                    font-size: 36px;
                    font-weight: 800;
                    color: #fff;
                    line-height: 1;
                }
                .most-likely-label {
                    font-size: 12px;
                    color: #9ca3af;
                    margin-top: 5px;
                }
                .most-likely-prob {
                    font-size: 18px;
                    color: #10b981;
                    font-weight: 600;
                }
                .team-stats {
                    padding: 15px;
                }
                @media (min-width: 768px) {
                    .most-likely-goals {
                        padding: 20px;
                    }
                    .most-likely-number {
                        font-size: 48px;
                    }
                    .most-likely-label {
                        font-size: 14px;
                    }
                    .most-likely-prob {
                        font-size: 20px;
                    }
                    .team-stats {
                        padding: 20px;
                    }
                }
                .stat-item {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 12px;
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 8px;
                    margin-bottom: 10px;
                }
                .stat-label {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    color: #9ca3af;
                    font-size: 14px;
                }
                .stat-value {
                    font-size: 16px;
                    font-weight: 600;
                    color: #fff;
                }
                .ou-predictions {
                    padding: 0 15px 15px;
                }
                .ou-title {
                    font-size: 11px;
                    color: #6b7280;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    margin-bottom: 10px;
                }
                .ou-item {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 6px 10px;
                    background: rgba(255, 255, 255, 0.02);
                    border-radius: 6px;
                    margin-bottom: 6px;
                }
                @media (min-width: 768px) {
                    .ou-predictions {
                        padding: 0 20px 20px;
                    }
                    .ou-title {
                        font-size: 12px;
                    }
                    .ou-item {
                        padding: 8px 12px;
                        margin-bottom: 8px;
                    }
                }
                .ou-threshold {
                    font-size: 13px;
                    color: #9ca3af;
                }
                .ou-values {
                    display: flex;
                    gap: 15px;
                }
                .ou-over {
                    color: #10b981;
                    font-weight: 600;
                }
                .ou-under {
                    color: #ef4444;
                    font-weight: 600;
                }
                .btts-banner {
                    background: linear-gradient(135deg, rgba(249, 115, 22, 0.1) 0%, rgba(249, 115, 22, 0.05) 100%);
                    border: 1px solid rgba(249, 115, 22, 0.3);
                    border-radius: 12px;
                    padding: 20px;
                    text-align: center;
                }
            </style>
            
            <div class="section-header">
                <div class="section-icon" style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);">
                    <i class="fas fa-users" style="color: #fff; font-size: 18px;"></i>
                </div>
                <h5 style="margin: 0; color: #fff;">Takım Gol Tahminleri</h5>
            </div>
            
            <div class="team-goals-container">
    `;
    
    // Ev sahibi takım
    html += `
        <div class="team-card">
            <div class="team-header home-team-header">
                <i class="fas fa-home"></i> ${teamGoals.home_team.team_name}
            </div>
            
            <div class="most-likely-goals">
                <div class="most-likely-number">${teamGoals.home_team.most_likely_goals}</div>
                <div class="most-likely-label">En Olası Gol Sayısı</div>
                <div class="most-likely-prob">%${Math.round(teamGoals.home_team.most_likely_prob)}</div>
            </div>
            
            <div class="team-stats">
                <div class="stat-item">
                    <div class="stat-label">
                        <i class="fas fa-calculator"></i> Beklenen Gol
                    </div>
                    <div class="stat-value">${teamGoals.home_team.expected_goals}</div>
                </div>
                
                <div class="stat-item">
                    <div class="stat-label">
                        <i class="fas fa-futbol"></i> Gol Atma
                    </div>
                    <div class="stat-value" style="color: #10b981;">%${Math.round(teamGoals.home_team.score_probability)}</div>
                </div>
                
                <div class="stat-item">
                    <div class="stat-label">
                        <i class="fas fa-shield-alt"></i> Gol Yememe
                    </div>
                    <div class="stat-value" style="color: #3b82f6;">%${Math.round(teamGoals.home_team.clean_sheet_prob)}</div>
                </div>
            </div>
            
            <div class="ou-predictions">
                <div class="ou-title">Alt/Üst Tahminleri</div>
    `;
    
    Object.entries(teamGoals.home_team.over_under).forEach(([threshold, data]) => {
        html += `
            <div class="ou-item">
                <div class="ou-threshold">${threshold} Gol</div>
                <div class="ou-values">
                    <span class="ou-over">Ü: ${Math.round(data.over)}%</span>
                    <span class="ou-under">A: ${Math.round(data.under)}%</span>
                </div>
            </div>
        `;
    });
    
    html += '</div></div>';
    
    // Deplasman takım
    html += `
        <div class="team-card">
            <div class="team-header away-team-header">
                <i class="fas fa-plane"></i> ${teamGoals.away_team.team_name}
            </div>
            
            <div class="most-likely-goals">
                <div class="most-likely-number">${teamGoals.away_team.most_likely_goals}</div>
                <div class="most-likely-label">En Olası Gol Sayısı</div>
                <div class="most-likely-prob">%${Math.round(teamGoals.away_team.most_likely_prob)}</div>
            </div>
            
            <div class="team-stats">
                <div class="stat-item">
                    <div class="stat-label">
                        <i class="fas fa-calculator"></i> Beklenen Gol
                    </div>
                    <div class="stat-value">${teamGoals.away_team.expected_goals}</div>
                </div>
                
                <div class="stat-item">
                    <div class="stat-label">
                        <i class="fas fa-futbol"></i> Gol Atma
                    </div>
                    <div class="stat-value" style="color: #10b981;">%${Math.round(teamGoals.away_team.score_probability)}</div>
                </div>
                
                <div class="stat-item">
                    <div class="stat-label">
                        <i class="fas fa-shield-alt"></i> Gol Yememe
                    </div>
                    <div class="stat-value" style="color: #3b82f6;">%${Math.round(teamGoals.away_team.clean_sheet_prob)}</div>
                </div>
            </div>
            
            <div class="ou-predictions">
                <div class="ou-title">Alt/Üst Tahminleri</div>
    `;
    
    Object.entries(teamGoals.away_team.over_under).forEach(([threshold, data]) => {
        html += `
            <div class="ou-item">
                <div class="ou-threshold">${threshold} Gol</div>
                <div class="ou-values">
                    <span class="ou-over">Ü: ${Math.round(data.over)}%</span>
                    <span class="ou-under">A: ${Math.round(data.under)}%</span>
                </div>
            </div>
        `;
    });
    
    html += '</div></div></div>';
    
    // Her iki takım da gol atar
    if (teamGoals.both_teams_score) {
        html += `
            <div class="btts-banner">
                <div style="color: #f97316; font-weight: 600; margin-bottom: 10px;">
                    <i class="fas fa-users"></i> Her İki Takım Gol Atar
                </div>
                <div style="font-size: 36px; font-weight: 800; color: #fff;">
                    ${Math.round(teamGoals.both_teams_score.probability)}%
                </div>
            </div>
        `;
    }
    
    html += '</div>';
    return html;
}

// H2H (Karşılıklı Maç Geçmişi) Bölümü - Gelişmiş Mobil Uyumlu Versiyon
function generateH2HSection(h2hData, homeTeamName, awayTeamName) {
    let html = `
        <div class="modern-h2h-section">
            <style>
                .modern-h2h-section {
                    background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.5) 100%);
                    backdrop-filter: blur(20px);
                    border: 1px solid rgba(148, 163, 184, 0.1);
                    border-radius: 20px;
                    padding: 16px;
                    margin-top: 20px;
                    position: relative;
                    overflow: hidden;
                    animation: slideInUp 0.5s ease-out;
                }
                
                @keyframes slideInUp {
                    from {
                        transform: translateY(20px);
                        opacity: 0;
                    }
                    to {
                        transform: translateY(0);
                        opacity: 1;
                    }
                }
                
                .modern-h2h-section::before {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 2px;
                    background: linear-gradient(90deg, transparent, #fbbf24, transparent);
                    animation: shimmer 3s infinite;
                }
                
                @keyframes shimmer {
                    0% { transform: translateX(-100%); }
                    100% { transform: translateX(100%); }
                }
                
                @media (min-width: 768px) {
                    .modern-h2h-section {
                        padding: 28px;
                        margin-top: 24px;
                    }
                }
                
                .h2h-header {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    margin-bottom: 24px;
                    font-size: 16px;
                    font-weight: 700;
                    color: #f3f4f6;
                }
                
                @media (min-width: 768px) {
                    .h2h-header {
                        font-size: 20px;
                        gap: 16px;
                    }
                }
                
                .h2h-icon {
                    width: 44px;
                    height: 44px;
                    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
                    border-radius: 14px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #fff;
                    font-size: 20px;
                    box-shadow: 0 4px 20px rgba(245, 158, 11, 0.3);
                    animation: pulse 2s infinite;
                }
                
                @keyframes pulse {
                    0%, 100% { transform: scale(1); }
                    50% { transform: scale(1.05); }
                }
                
                @media (min-width: 768px) {
                    .h2h-icon {
                        width: 48px;
                        height: 48px;
                        font-size: 22px;
                    }
                }
                
                .h2h-stats-grid {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 12px;
                    margin-bottom: 24px;
                }
                
                @media (min-width: 768px) {
                    .h2h-stats-grid {
                        gap: 16px;
                        margin-bottom: 28px;
                    }
                }
                
                .h2h-stat-card {
                    background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 16px;
                    padding: 16px 8px;
                    text-align: center;
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                    position: relative;
                    overflow: hidden;
                }
                
                .h2h-stat-card::after {
                    content: '';
                    position: absolute;
                    top: -50%;
                    left: -50%;
                    width: 200%;
                    height: 200%;
                    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
                    opacity: 0;
                    transition: opacity 0.3s ease;
                    pointer-events: none;
                }
                
                @media (min-width: 768px) {
                    .h2h-stat-card {
                        padding: 24px 12px;
                    }
                    
                    .h2h-stat-card:hover {
                        transform: translateY(-4px) scale(1.02);
                        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
                    }
                    
                    .h2h-stat-card:hover::after {
                        opacity: 1;
                    }
                }
                
                .h2h-stat-value {
                    font-size: 28px;
                    font-weight: 900;
                    margin-bottom: 6px;
                    line-height: 1;
                    letter-spacing: -0.5px;
                    animation: fadeIn 0.6s ease-out;
                }
                
                @keyframes fadeIn {
                    from { opacity: 0; transform: scale(0.8); }
                    to { opacity: 1; transform: scale(1); }
                }
                
                @media (min-width: 768px) {
                    .h2h-stat-value {
                        font-size: 36px;
                        margin-bottom: 10px;
                    }
                }
                
                .h2h-stat-label {
                    font-size: 12px;
                    color: #cbd5e1;
                    line-height: 1.4;
                    font-weight: 500;
                }
                
                @media (min-width: 768px) {
                    .h2h-stat-label {
                        font-size: 14px;
                    }
                }
                
                .home-win .h2h-stat-value { 
                    color: #10b981;
                    text-shadow: 0 0 20px rgba(16, 185, 129, 0.5);
                }
                
                .draw .h2h-stat-value { 
                    color: #f59e0b;
                    text-shadow: 0 0 20px rgba(245, 158, 11, 0.5);
                }
                
                .away-win .h2h-stat-value { 
                    color: #ef4444;
                    text-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
                }
                
                .h2h-goals-row {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 12px;
                    margin-bottom: 24px;
                }
                
                @media (min-width: 768px) {
                    .h2h-goals-row {
                        gap: 16px;
                        margin-bottom: 28px;
                    }
                }
                
                .h2h-matches-section {
                    margin-top: 24px;
                }
                
                .h2h-matches-header {
                    font-size: 15px;
                    font-weight: 600;
                    margin-bottom: 16px;
                    color: #f3f4f6;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                
                .h2h-matches-header::before {
                    content: '⚽';
                    font-size: 16px;
                }
                
                @media (min-width: 768px) {
                    .h2h-matches-header {
                        font-size: 17px;
                        margin-bottom: 20px;
                    }
                }
                
                .h2h-match-row {
                    background: rgba(255, 255, 255, 0.04);
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    border-radius: 14px;
                    padding: 14px;
                    margin-bottom: 10px;
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                    position: relative;
                    overflow: hidden;
                }
                
                .h2h-match-row::before {
                    content: '';
                    position: absolute;
                    left: 0;
                    top: 0;
                    bottom: 0;
                    width: 3px;
                    background: transparent;
                    transition: background 0.3s ease;
                }
                
                @media (min-width: 768px) {
                    .h2h-match-row {
                        padding: 16px 20px;
                        margin-bottom: 12px;
                    }
                    
                    .h2h-match-row:hover {
                        background: rgba(255, 255, 255, 0.08);
                        transform: translateX(6px);
                        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
                    }
                }
                
                .h2h-match-row.home-win-row::before { background: #10b981; }
                .h2h-match-row.draw-row::before { background: #f59e0b; }
                .h2h-match-row.away-win-row::before { background: #ef4444; }
                
                .h2h-match-content {
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                }
                
                @media (min-width: 768px) {
                    .h2h-match-content {
                        flex-direction: row;
                        align-items: center;
                        justify-content: space-between;
                        gap: 16px;
                    }
                }
                
                .h2h-match-main {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    flex: 1;
                    min-width: 0;
                }
                
                @media (min-width: 768px) {
                    .h2h-match-main {
                        gap: 20px;
                    }
                }
                
                .h2h-match-date {
                    font-size: 12px;
                    color: #94a3b8;
                    min-width: 65px;
                    font-weight: 500;
                    white-space: nowrap;
                }
                
                @media (min-width: 768px) {
                    .h2h-match-date {
                        font-size: 14px;
                        min-width: 85px;
                    }
                }
                
                .h2h-match-score {
                    font-size: 13px;
                    font-weight: 700;
                    text-align: center;
                    letter-spacing: 0.3px;
                    white-space: nowrap;
                    flex: 1;
                }
                
                @media (min-width: 768px) {
                    .h2h-match-score {
                        font-size: 15px;
                        letter-spacing: 0.5px;
                    }
                }
                
                .h2h-match-icon {
                    font-size: 14px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: 24px;
                    height: 24px;
                    border-radius: 6px;
                    background: rgba(255, 255, 255, 0.1);
                    flex-shrink: 0;
                }
                
                @media (min-width: 768px) {
                    .h2h-match-icon {
                        font-size: 18px;
                        width: 32px;
                        height: 32px;
                        border-radius: 8px;
                    }
                }
                
                .h2h-match-league {
                    font-size: 11px;
                    color: #cbd5e1;
                    font-weight: 500;
                    background: rgba(255, 255, 255, 0.05);
                    padding: 4px 10px;
                    border-radius: 8px;
                    white-space: nowrap;
                    text-align: left;
                    align-self: flex-start;
                }
                
                @media (min-width: 768px) {
                    .h2h-match-league {
                        font-size: 13px;
                        padding: 5px 12px;
                        text-align: right;
                        align-self: center;
                    }
                }
                
                .h2h-btts-info {
                    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(59, 130, 246, 0.08) 100%);
                    border: 1px solid rgba(59, 130, 246, 0.3);
                    border-radius: 16px;
                    padding: 16px;
                    margin-top: 20px;
                    text-align: center;
                    font-size: 13px;
                    font-weight: 500;
                    color: #dbeafe;
                    animation: slideInUp 0.6s ease-out 0.3s backwards;
                }
                
                .h2h-btts-info i {
                    color: #60a5fa;
                    margin-right: 6px;
                }
                
                @media (min-width: 768px) {
                    .h2h-btts-info {
                        padding: 20px;
                        margin-top: 24px;
                        font-size: 14px;
                    }
                }
                
                .h2h-empty-state {
                    text-align: center;
                    padding: 40px 20px;
                    color: #94a3b8;
                    font-size: 14px;
                }
                
                .h2h-empty-state i {
                    font-size: 48px;
                    color: #475569;
                    margin-bottom: 16px;
                }
                
                @media (min-width: 768px) {
                    .h2h-empty-state {
                        padding: 60px 30px;
                        font-size: 16px;
                    }
                    
                    .h2h-empty-state i {
                        font-size: 56px;
                    }
                }
            </style>
            
            <div class="h2h-header">
                <div class="h2h-icon">
                    <i class="fas fa-history"></i>
                </div>
                <span>Karşılıklı Maç Geçmişi (H2H)</span>
            </div>
    `;
    
    // H2H verisi yoksa
    if (!h2hData || !h2hData.matches || h2hData.matches.length === 0) {
        html += `
            <div class="h2h-empty-state">
                <i class="fas fa-history"></i>
                <div>Son 10 yılda karşılıklı maç bulunmuyor.</div>
            </div>
        `;
    } else {
        // H2H istatistikleri
        const stats = calculateH2HStats(h2hData.matches, homeTeamName, awayTeamName);
        
        // Özet istatistikler
        html += `
            <div class="h2h-stats-grid">
                <div class="h2h-stat-card home-win">
                    <div class="h2h-stat-value">${stats.homeWins}</div>
                    <div class="h2h-stat-label">${homeTeamName}<br>Galibiyeti</div>
                </div>
                <div class="h2h-stat-card draw">
                    <div class="h2h-stat-value">${stats.draws}</div>
                    <div class="h2h-stat-label">Beraberlik</div>
                </div>
                <div class="h2h-stat-card away-win">
                    <div class="h2h-stat-value">${stats.awayWins}</div>
                    <div class="h2h-stat-label">${awayTeamName}<br>Galibiyeti</div>
                </div>
            </div>
            
            <div class="h2h-goals-row">
                <div class="h2h-stat-card">
                    <div class="h2h-stat-value text-info">${stats.totalGoals}</div>
                    <div class="h2h-stat-label">Toplam Gol</div>
                </div>
                <div class="h2h-stat-card">
                    <div class="h2h-stat-value text-primary">${stats.avgGoals.toFixed(1)}</div>
                    <div class="h2h-stat-label">Maç Başı Ort. Gol</div>
                </div>
            </div>
        `;
        
        // Son maçlar listesi
        html += `
            <div class="h2h-matches-section">
                <h6 class="h2h-matches-header">Son ${Math.min(10, h2hData.matches.length)} Karşılaşma</h6>
        `;
        
        // Her maç için satır oluştur (son 10 maç)
        h2hData.matches.slice(0, 10).forEach(match => {
            const homeScore = match.home_score || match.match_hometeam_score || 0;
            const awayScore = match.away_score || match.match_awayteam_score || 0;
            const date = formatH2HDate(match.date || match.match_date);
            const league = match.league_name || match.competition || 'Lig';
            
            // Takım isimlerini al
            const homeTeam = match.home_team || match.match_hometeam_name || homeTeamName;
            const awayTeam = match.away_team || match.match_awayteam_name || awayTeamName;
            
            // Kazananı belirle
            let resultClass = '';
            let rowClass = '';
            if (homeScore > awayScore) {
                resultClass = 'text-success';
                rowClass = 'home-win-row';
            } else if (awayScore > homeScore) {
                resultClass = 'text-danger';
                rowClass = 'away-win-row';
            } else {
                resultClass = 'text-warning';
                rowClass = 'draw-row';
            }
            
            html += `
                <div class="h2h-match-row ${rowClass}">
                    <div class="h2h-match-content">
                        <div class="h2h-match-main">
                            <span class="h2h-match-date">${date}</span>
                            <span class="h2h-match-score ${resultClass}">${homeTeam} ${homeScore} - ${awayScore} ${awayTeam}</span>
                        </div>
                        <div class="h2h-match-league">${league}</div>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        
        // Ek bilgiler
        if (stats.bttsCount > 0) {
            html += `
                <div class="h2h-btts-info">
                    <i class="fas fa-info-circle"></i> 
                    Son ${h2hData.matches.length} maçın ${stats.bttsCount} tanesinde (%${stats.bttsPercentage}) her iki takım da gol attı.
                </div>
            `;
        }
    }
    
    html += `
        </div>
    `;
    
    return html;
}

// H2H istatistiklerini hesapla
function calculateH2HStats(matches, homeTeamName, awayTeamName) {
    let homeWins = 0;
    let awayWins = 0;
    let draws = 0;
    let totalGoals = 0;
    let bttsCount = 0;
    
    matches.forEach(match => {
        const homeScore = parseInt(match.home_score || match.match_hometeam_score || 0);
        const awayScore = parseInt(match.away_score || match.match_awayteam_score || 0);
        
        totalGoals += homeScore + awayScore;
        
        if (homeScore > awayScore) {
            homeWins++;
        } else if (awayScore > homeScore) {
            awayWins++;
        } else {
            draws++;
        }
        
        if (homeScore > 0 && awayScore > 0) {
            bttsCount++;
        }
    });
    
    return {
        homeWins,
        awayWins,
        draws,
        totalGoals,
        avgGoals: matches.length > 0 ? totalGoals / matches.length : 0,
        bttsCount,
        bttsPercentage: matches.length > 0 ? Math.round((bttsCount / matches.length) * 100) : 0
    };
}

// H2H tarihini formatla
function formatH2HDate(dateStr) {
    if (!dateStr) return 'Tarih yok';
    
    try {
        const date = new Date(dateStr);
        const day = date.getDate().toString().padStart(2, '0');
        const month = (date.getMonth() + 1).toString().padStart(2, '0');
        const year = date.getFullYear();
        return `${day}.${month}.${year}`;
    } catch (e) {
        return dateStr;
    }
}

// Açıklama bölümü - Explainable AI
function generateExplanationSection(explanation) {
    if (!explanation) return '<div class="alert alert-info">Açıklama verisi bulunmuyor.</div>';
    
    let html = `
        <div class="modern-prediction-section">
            <style>
                .explanation-container {
                    padding: 10px;
                }
                @media (min-width: 768px) {
                    .explanation-container {
                        padding: 20px;
                    }
                }
                
                /* Güven Seviyesi - Mobil Optimize */
                .confidence-meter {
                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
                    border: 1px solid rgba(102, 126, 234, 0.2);
                    border-radius: 16px;
                    padding: 15px 12px;
                    margin-bottom: 15px;
                    text-align: center;
                    position: relative;
                    overflow: hidden;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                }
                .confidence-meter::before {
                    content: '';
                    position: absolute;
                    top: -50%;
                    right: -50%;
                    width: 200%;
                    height: 200%;
                    background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
                    animation: pulse 4s ease-in-out infinite;
                }
                @keyframes pulse {
                    0%, 100% { opacity: 0.5; transform: scale(0.8); }
                    50% { opacity: 1; transform: scale(1); }
                }
                .confidence-level {
                    font-size: 22px;
                    font-weight: 700;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    text-fill-color: transparent;
                    margin: 2px 0;
                    position: relative;
                    z-index: 1;
                    line-height: 1;
                }
                @media (min-width: 768px) {
                    .confidence-meter {
                        padding: 20px 18px;
                    }
                    .confidence-level {
                        font-size: 32px;
                        margin: 6px 0;
                    }
                }
                .confidence-category {
                    font-size: 10px;
                    font-weight: 600;
                    color: #a78bfa;
                    text-transform: uppercase;
                    letter-spacing: 0.2px;
                    margin-bottom: 4px;
                }
                @media (min-width: 768px) {
                    .confidence-category {
                        font-size: 13px;
                        letter-spacing: 0.4px;
                    }
                }
                .confidence-meter h5 {
                    font-size: 11px;
                    font-weight: 600;
                    margin-bottom: 6px;
                    color: #e9d5ff;
                }
                @media (min-width: 768px) {
                    .confidence-meter h5 {
                        font-size: 14px;
                        margin-bottom: 8px;
                    }
                }
                .confidence-meter p {
                    font-size: 10px;
                    line-height: 1.3;
                    margin-bottom: 0;
                    position: relative;
                    z-index: 1;
                    color: rgba(255,255,255,0.7);
                }
                @media (min-width: 768px) {
                    .confidence-meter p {
                        font-size: 12px;
                        line-height: 1.4;
                    }
                }
                
                /* Anahtar Faktörler - Mobil Optimize */
                .key-factors {
                    background: rgba(255, 255, 255, 0.03);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 16px;
                    padding: 15px;
                    margin-bottom: 20px;
                }
                @media (min-width: 768px) {
                    .key-factors {
                        padding: 25px;
                    }
                }
                .key-factors > h5 {
                    font-size: 16px;
                    margin-bottom: 15px;
                }
                @media (min-width: 768px) {
                    .key-factors > h5 {
                        font-size: 18px;
                        margin-bottom: 20px;
                    }
                }
                .factor-item {
                    display: flex;
                    align-items: flex-start;
                    gap: 12px;
                    padding: 10px;
                    margin-bottom: 8px;
                    background: rgba(255, 255, 255, 0.02);
                    border-radius: 10px;
                    transition: all 0.3s ease;
                }
                @media (min-width: 768px) {
                    .factor-item {
                        align-items: center;
                        gap: 15px;
                        padding: 14px;
                        margin-bottom: 12px;
                    }
                    .factor-item:hover {
                        background: rgba(255, 255, 255, 0.05);
                        transform: translateX(5px);
                    }
                }
                .factor-icon {
                    width: 36px;
                    height: 36px;
                    min-width: 36px;
                    border-radius: 10px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 16px;
                }
                @media (min-width: 768px) {
                    .factor-icon {
                        width: 44px;
                        height: 44px;
                        min-width: 44px;
                        font-size: 20px;
                    }
                }
                .factor-content {
                    flex: 1;
                    min-width: 0;
                }
                .factor-content strong {
                    font-size: 14px;
                    display: block;
                    margin-bottom: 2px;
                }
                .factor-content .text-muted {
                    font-size: 12px;
                    line-height: 1.4;
                }
                @media (min-width: 768px) {
                    .factor-content strong {
                        font-size: 15px;
                        margin-bottom: 4px;
                    }
                    .factor-content .text-muted {
                        font-size: 13px;
                        line-height: 1.5;
                    }
                }
                .factor-positive {
                    background: rgba(34, 197, 94, 0.2);
                    color: #22c55e;
                }
                .factor-negative {
                    background: rgba(239, 68, 68, 0.2);
                    color: #ef4444;
                }
                .factor-neutral {
                    background: rgba(59, 130, 246, 0.2);
                    color: #3b82f6;
                }
                
                /* SWOT Analizi - Mobil Optimize */
                .swot-section {
                    margin-top: 25px;
                }
                .swot-section > h5 {
                    font-size: 16px;
                    margin-bottom: 15px;
                }
                @media (min-width: 768px) {
                    .swot-section > h5 {
                        font-size: 18px;
                        margin-bottom: 20px;
                    }
                }
                .swot-grid {
                    display: grid;
                    grid-template-columns: 1fr;
                    gap: 12px;
                }
                @media (min-width: 768px) {
                    .swot-grid {
                        grid-template-columns: repeat(2, 1fr);
                        gap: 15px;
                    }
                }
                @media (min-width: 1024px) {
                    .swot-grid {
                        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                    }
                }
                .swot-card {
                    background: rgba(255, 255, 255, 0.03);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 12px;
                    padding: 15px;
                    transition: all 0.3s ease;
                }
                @media (min-width: 768px) {
                    .swot-card {
                        padding: 20px;
                    }
                    .swot-card:hover {
                        background: rgba(255, 255, 255, 0.05);
                        transform: translateY(-2px);
                        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
                    }
                }
                .swot-card h6 {
                    font-size: 14px;
                    font-weight: 700;
                    margin-bottom: 12px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                @media (min-width: 768px) {
                    .swot-card h6 {
                        font-size: 16px;
                        margin-bottom: 15px;
                        gap: 10px;
                    }
                }
                .swot-card ul {
                    list-style: none;
                    padding: 0;
                    margin: 0;
                }
                .swot-card li {
                    padding: 6px 0;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                    font-size: 12px;
                    line-height: 1.5;
                }
                @media (min-width: 768px) {
                    .swot-card li {
                        padding: 8px 0;
                        font-size: 14px;
                    }
                }
                .swot-card li:last-child {
                    border-bottom: none;
                }
                
                /* Doğal Dil Açıklaması - Mobil Optimize */
                .natural-explanation {
                    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%);
                    border: 1px solid rgba(59, 130, 246, 0.2);
                    border-radius: 16px;
                    padding: 15px;
                    margin-top: 20px;
                }
                @media (min-width: 768px) {
                    .natural-explanation {
                        padding: 25px;
                        margin-top: 25px;
                    }
                }
                .natural-explanation h5 {
                    font-size: 16px;
                    margin-bottom: 12px;
                }
                @media (min-width: 768px) {
                    .natural-explanation h5 {
                        font-size: 18px;
                        margin-bottom: 15px;
                    }
                }
                .natural-explanation p {
                    margin: 0;
                    line-height: 1.7;
                    font-size: 13px;
                    color: rgba(255, 255, 255, 0.9);
                }
                @media (min-width: 768px) {
                    .natural-explanation p {
                        line-height: 1.8;
                        font-size: 15px;
                    }
                }
                
                /* Bölümler Arası Boşluk */
                .explanation-section {
                    margin-bottom: 20px;
                }
                .explanation-section:last-child {
                    margin-bottom: 0;
                }
            </style>
            
            <div class="explanation-container">
    `;
    
    // Güven Seviyesi - Kaldırıldı
    
    // Anahtar Faktörler
    if (explanation.key_factors && explanation.key_factors.length > 0) {
        html += `
            <div class="explanation-section">
                <div class="key-factors">
                    <h5>
                        <i class="fas fa-key"></i> Anahtar Faktörler
                    </h5>
        `;
        
        explanation.key_factors.forEach(factor => {
            let iconClass = 'factor-neutral';
            let icon = 'fa-minus';
            
            if (factor.impact === 'positive') {
                iconClass = 'factor-positive';
                icon = 'fa-arrow-up';
            } else if (factor.impact === 'negative') {
                iconClass = 'factor-negative';
                icon = 'fa-arrow-down';
            }
            
            html += `
                <div class="factor-item">
                    <div class="factor-icon ${iconClass}">
                        <i class="fas ${icon}"></i>
                    </div>
                    <div class="factor-content">
                        <strong>${factor.name || factor.factor || ''}</strong>
                        <div class="text-muted">${factor.description || ''}</div>
                    </div>
                </div>
            `;
        });
        
        html += `</div></div>`;
    }
    
    // SWOT Analizi
    if (explanation.detailed_analysis) {
        const analysis = explanation.detailed_analysis;
        html += `
            <div class="explanation-section swot-section">
                <h5>
                    <i class="fas fa-chess"></i> Detaylı Analiz
                </h5>
                <div class="swot-grid">
        `;
        
        // Güçlü Yönler
        if (analysis.strengths && analysis.strengths.length > 0) {
            html += `
                <div class="swot-card">
                    <h6><i class="fas fa-shield-alt text-success"></i> Güçlü Yönler</h6>
                    <ul>
                        ${analysis.strengths.map(item => `<li>${item}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        // Zayıf Yönler
        if (analysis.weaknesses && analysis.weaknesses.length > 0) {
            html += `
                <div class="swot-card">
                    <h6><i class="fas fa-exclamation-triangle text-warning"></i> Zayıf Yönler</h6>
                    <ul>
                        ${analysis.weaknesses.map(item => `<li>${item}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        // Fırsatlar
        if (analysis.opportunities && analysis.opportunities.length > 0) {
            html += `
                <div class="swot-card">
                    <h6><i class="fas fa-lightbulb text-info"></i> Fırsatlar</h6>
                    <ul>
                        ${analysis.opportunities.map(item => `<li>${item}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        // Tehditler
        if (analysis.threats && analysis.threats.length > 0) {
            html += `
                <div class="swot-card">
                    <h6><i class="fas fa-bolt text-danger"></i> Tehditler</h6>
                    <ul>
                        ${analysis.threats.map(item => `<li>${item}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        html += `</div></div>`;
    }
    
    // Doğal Dil Açıklaması
    if (explanation.natural_language_explanation) {
        html += `
            <div class="explanation-section">
                <div class="natural-explanation">
                    <h5>
                        <i class="fas fa-comment-dots"></i> Tahmin Açıklaması
                    </h5>
                    <p>${explanation.natural_language_explanation}</p>
                </div>
            </div>
        `;
    }
    
    html += `
            </div>
        </div>
    `;
    
    return html;
}

// Çifte şans bölümü - Modern tasarım
function generateDoubleChanceSection(doubleChance) {
    if (!doubleChance || !doubleChance.predictions) return '';
    
    let html = `
        <div class="modern-prediction-section">
            <style>
                .double-chance-container {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px;
                }
                .double-chance-card {
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 12px;
                    padding: 20px;
                    text-align: center;
                    transition: all 0.3s ease;
                    position: relative;
                    overflow: hidden;
                }
                .double-chance-card:hover {
                    background: rgba(255, 255, 255, 0.08);
                    transform: translateY(-3px);
                    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
                }
                .double-chance-card.safest {
                    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%);
                    border-color: rgba(34, 197, 94, 0.5);
                }
                .double-chance-card.safest::before {
                    content: '✓ EN GÜVENLİ';
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
                    color: #fff;
                    font-size: 10px;
                    padding: 4px 10px;
                    border-radius: 12px;
                    font-weight: 600;
                }
                .double-chance-label {
                    font-size: 24px;
                    font-weight: 700;
                    color: #10b981;
                    margin-bottom: 15px;
                }
                .double-chance-percentage {
                    font-size: 48px;
                    font-weight: 800;
                    color: #fff;
                    margin-bottom: 10px;
                    line-height: 1;
                }
                .double-chance-desc {
                    font-size: 13px;
                    color: #9ca3af;
                    line-height: 1.4;
                }
                .safest-banner {
                    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%);
                    border: 1px solid rgba(34, 197, 94, 0.3);
                    border-radius: 12px;
                    padding: 15px;
                    margin-bottom: 20px;
                    text-align: center;
                }
            </style>
            
            <div class="section-header">
                <div class="section-icon" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
                    <i class="fas fa-shield-alt" style="color: #fff; font-size: 18px;"></i>
                </div>
                <h5 style="margin: 0; color: #fff;">Çifte Şans Tahminleri</h5>
            </div>
            
            <div class="safest-banner">
                <div style="color: #22c55e; font-weight: 600;">
                    <i class="fas fa-trophy"></i> En Güvenli Seçenek
                </div>
                <div style="font-size: 20px; color: #fff; margin-top: 5px;">
                    ${doubleChance.safest_option} - %${Math.round(doubleChance.predictions[doubleChance.safest_option].probability)}
                </div>
            </div>
            
            <div class="double-chance-container">
    `;
    
    Object.entries(doubleChance.predictions).forEach(([key, data]) => {
        const isHighest = key === doubleChance.safest_option;
        html += `
            <div class="double-chance-card ${isHighest ? 'safest' : ''}">
                <div class="double-chance-label">${key}</div>
                <div class="double-chance-percentage">${Math.round(data.probability)}%</div>
                <div class="double-chance-desc">${data.description}</div>
            </div>
        `;
    });
    
    html += '</div></div>';
    return html;
}

// HT/FT sonuç formatla
function formatHTFTResult(result) {
    const mapping = {
        'HOME_HOME': 'Ev/Ev',
        'HOME_DRAW': 'Ev/Ber',
        'HOME_AWAY': 'Ev/Dep',
        'DRAW_HOME': 'Ber/Ev',
        'DRAW_DRAW': 'Ber/Ber',
        'DRAW_AWAY': 'Ber/Dep',
        'AWAY_HOME': 'Dep/Ev',
        'AWAY_DRAW': 'Dep/Ber',
        'AWAY_AWAY': 'Dep/Dep'
    };
    return mapping[result] || result;
}

// Sayfa değiştirme fonksiyonu - Sekmeleri düzgün değiştirmek için
window.showPredictionPage = function(pageNumber) {
    // Tüm sayfaları gizle
    for (let i = 1; i <= 6; i++) {
        const page = document.getElementById('predictionPage' + i);
        if (page) {
            page.style.display = 'none';
        }
    }
    
    // Tüm tab butonlarından active sınıfını kaldır
    document.querySelectorAll('.page-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Seçili sayfayı göster
    const selectedPage = document.getElementById('predictionPage' + pageNumber);
    if (selectedPage) {
        selectedPage.style.display = 'block';
    }
    
    // Aktif tab butonunu işaretle
    const tabs = document.querySelectorAll('.page-tab');
    if (tabs[pageNumber - 1]) {
        tabs[pageNumber - 1].classList.add('active');
    }
}