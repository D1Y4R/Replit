// Tahmin verilerini işleme ve görüntüleme işlevleri

function updatePredictionUI(data) {
    console.log("updatePredictionUI çağrıldı");
    
    // Global tahmin verilerini kaydet (form ve motivasyon için)
    window.predictionData = data;

    // Takım bilgilerini güncelle
    $('#homeTeamName').text(data.home_team.name);
    $('#awayTeamName').text(data.away_team.name);
    
    // ********** DOĞRUDAN BACKEND VERİLERİNİ KULLAN **********
    console.log("Backend'den gelen tahmin verilerini doğrudan kullanıyorum...");
    
    // Olasılık çubuklarını güncelle
    if (data.predictions && data.predictions.match_winner) {
        const homeProb = Math.round(data.predictions.match_winner.home_win * 100);
        const drawProb = Math.round(data.predictions.match_winner.draw * 100);
        const awayProb = Math.round(data.predictions.match_winner.away_win * 100);
        
        updateProbabilityBars(homeProb, drawProb, awayProb);
    }
    
    // Skor tahmini ve bahis tahminlerini güncelle
    if (data.predictions) {
        // Kesin skor tahmini - Backend'den doğrudan alınır
        if (data.predictions.exact_score) {
            $('#predictedScore').text(data.predictions.exact_score);
        }
        
        // Bahis tahminlerini doğrudan backend verilerinden al
        if (data.predictions.betting_predictions) {
            const betting = data.predictions.betting_predictions;
            
            // KG VAR/YOK tahmini
            if (betting.both_teams_to_score) {
                let bttsValue = betting.both_teams_to_score.prediction;
                const bttsProb = betting.both_teams_to_score.probability || 0;
                
                // Backend artık doğrudan KG VAR/KG YOK değerlerini gönderiyor
                // Eski sürüm uyumluluğu için kontrol 
                if (bttsValue === 'YES') {
                    bttsValue = 'KG VAR';
                } else if (bttsValue === 'NO') {
                    bttsValue = 'KG YOK';
                }
                
                $('#bttsValue').text(bttsValue);
                $('#bttsProb').text(bttsProb.toFixed(2) + '%');
            }
            
            // Toplam gol tahmini (ÜST/ALT 2.5)
            if (betting.over_2_5_goals) {
                const overValue = betting.over_2_5_goals.prediction;
                const overProb = betting.over_2_5_goals.probability || 0;
                
                $('#overValue').text(overValue);
                $('#overProb').text(overProb.toFixed(2) + '%');
            }
            
            // Toplam gol tahmini (ÜST/ALT 3.5)
            if (betting.over_3_5_goals) {
                const over35Value = betting.over_3_5_goals.prediction;
                const over35Prob = betting.over_3_5_goals.probability || 0;
                
                $('#over35Value').text(over35Value);
                $('#over35Prob').text(over35Prob.toFixed(2) + '%');
            }
        }
    }
    
    // Form ve motivasyon alanlarını güncelle
    updateMotivationUI(data);
}

function updateProbabilityBars(homeProb, drawProb, awayProb) {
    // Olasılık değerlerini göster
    $('#homeWinProb').text(homeProb + '%');
    $('#drawProb').text(drawProb + '%');
    $('#awayWinProb').text(awayProb + '%');
    
    // Olasılık çubuklarını ayarla
    $('#homeWinBar').css('width', homeProb + '%');
    $('#drawBar').css('width', drawProb + '%');
    $('#awayWinBar').css('width', awayProb + '%');
    
    // En yüksek olasılığa sahip sonucu vurgula
    $('.prob-bar').removeClass('highest-prob');
    
    if (homeProb >= drawProb && homeProb >= awayProb) {
        $('#homeWinBar').addClass('highest-prob');
    } else if (drawProb >= homeProb && drawProb >= awayProb) {
        $('#drawBar').addClass('highest-prob');
    } else {
        $('#awayWinBar').addClass('highest-prob');
    }
}

function updateMotivationUI(data) {
    if (!data.home_team || !data.away_team) return;
    
    // Form faktörleri
    try {
        const homeForm = data.home_team.form.detailed_data.all.slice(0, 5);
        const awayForm = data.away_team.form.detailed_data.all.slice(0, 5);
        
        // Form ikonlarını oluştur
        const homeFormHTML = createFormIcons(homeForm);
        const awayFormHTML = createFormIcons(awayForm);
        
        // Form ikonlarını göster
        $('#homeTeamForm').html(homeFormHTML);
        $('#awayTeamForm').html(awayFormHTML);
        
    } catch (e) {
        console.error("Form verileri gösterilirken hata:", e);
    }
}

function createFormIcons(matches) {
    if (!matches || !matches.length) return '<span class="form-icon">-</span>';
    
    let formHTML = '';
    
    // Son maçları yeniden eskiye doğru göster
    for (let i = 0; i < matches.length; i++) {
        const match = matches[i];
        let result = match.result || '-';
        
        if (result === 'W') {
            formHTML += '<span class="form-icon win">W</span>';
        } else if (result === 'D') {
            formHTML += '<span class="form-icon draw">D</span>';
        } else if (result === 'L') {
            formHTML += '<span class="form-icon loss">L</span>';
        } else {
            formHTML += '<span class="form-icon">-</span>';
        }
    }
    
    return formHTML;
}

// Taraftar faktörleri, motivasyon ve takım dışı etkileri daha sonra eklenecek
