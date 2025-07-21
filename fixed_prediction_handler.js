// Tahmin verilerini işleme ve görüntüleme işlevleri

function updatePredictionUI(data) {
    console.log("updatePredictionUI çağrıldı");
    console.log("Tahmin ham verisi:", data);
    
    // Global tahmin verilerini kaydet (form ve motivasyon için)
    window.predictionData = data;

    // Takım bilgilerini güncelle
    $('#homeTeamName').text(data.home_team.name);
    $('#awayTeamName').text(data.away_team.name);
    
    // ********** TUTARSIZLIK DÜZELTME **********
    console.log("Tahmin verilerini tutarlılık için kontrol ediyorum...");
    
    if (data.predictions) {
        let backendCorrectedScore = null;
        
        // 1. Tutarlılık düzeltme mesajını kontrol et
        if (data.predictions.debug_final_check && 
            data.predictions.debug_final_check.includes('Tutarsızlık giderildi')) {
            
            console.warn("BACKEND TUTARSIZLIK DÜZELTMESİ BULUNDU:", data.predictions.debug_final_check);
            
            // "Tutarsızlık giderildi: Skor 1-1 -> 1-2 olarak güncellendi" formatından skoru çıkar
            const scoreMatch = data.predictions.debug_final_check.match(/Skor\s+([0-9]+-[0-9]+)\s+->\s+([0-9]+-[0-9]+)/);
            if (scoreMatch && scoreMatch[2]) {
                const oldScore = scoreMatch[1];
                backendCorrectedScore = scoreMatch[2];
                
                console.warn(`SKOR DEĞİŞİKLİĞİ TESPİT EDİLDİ: ${oldScore} -> ${backendCorrectedScore}`);
                
                // Yeni skorun toplam gol sayısını hesapla
                const [homeGoals, awayGoals] = backendCorrectedScore.split('-').map(Number);
                const totalGoals = homeGoals + awayGoals;
                
                // Toplam gol sayısı 3 veya daha fazlaysa, betting_predictions'daki over_2_5_goals değerini kontrol et ve gerekirse düzelt
                if (totalGoals >= 3) {
                    console.log("Toplam gol sayısı 3 veya daha fazla, 2.5 ÜST olmalı");
                    
                    // Backend'den gelen over_2_5_goals değerini kontrol et
                    const over25Prediction = data.predictions.betting_predictions?.over_2_5_goals?.prediction;
                    
                    // Eğer over_2_5_goals değeri 'YES' veya '2.5 ÜST' değilse, bu bir tutarsızlıktır
                    if (over25Prediction !== 'YES' && over25Prediction !== '2.5 ÜST') {
                        console.error("KRİTİK TUTARSIZLIK: Backend skoru değiştirmiş ama over_2_5_goals verisi güncellenmemiş!");
                        console.warn("Skor toplamı:", totalGoals, "ama over_2_5_goals:", over25Prediction);
                        
                        // Tutarsızlığı düzelt: API yanıtında over_2_5_goals değerini güncelle
                        if (data.predictions.betting_predictions && data.predictions.betting_predictions.over_2_5_goals) {
                            data.predictions.betting_predictions.over_2_5_goals.prediction = 'YES';
                            data.predictions.betting_predictions.over_2_5_goals.display_value = '2.5 ÜST';
                            data.predictions.betting_predictions.over_2_5_goals.probability = 0.9;
                            
                            console.warn("DÜZELTME YAPILDI: over_2_5_goals değeri '2.5 ÜST' olarak güncellendi.");
                        }
                    }
                }
            }
        }
    }
    
    // Olasılık çubuklarını güncelle
    if (data.predictions && data.predictions.match_winner) {
        const homeProb = Math.round(data.predictions.match_winner.home_win * 100);
        const drawProb = Math.round(data.predictions.match_winner.draw * 100);
        const awayProb = Math.round(data.predictions.match_winner.away_win * 100);
        
        updateProbabilityBars(homeProb, drawProb, awayProb);
    }
    
    // Skor tahmini ve bahis tahminlerini güncelle
    if (data.predictions) {
        let scoreValue = "";
        
        // Kesin skor tahmini
        if (data.predictions.exact_score && data.predictions.exact_score.score) {
            scoreValue = data.predictions.exact_score.score;
            
            // Eğer tutarsızlık düzeltmesi yapıldıysa, doğru skoru kullan
            if (data.predictions.debug_final_check && 
                data.predictions.debug_final_check.includes('Tutarsızlık giderildi')) {
                
                const scoreMatch = data.predictions.debug_final_check.match(/Skor\s+([0-9]+-[0-9]+)\s+->\s+([0-9]+-[0-9]+)/);
                if (scoreMatch && scoreMatch[2]) {
                    scoreValue = scoreMatch[2];
                    console.warn("DÜZELTME YAPILDI: Tutarsızlık giderilmiş skor kullanılıyor:", scoreValue);
                }
            }
            
            $('#predictedScore').text(scoreValue);
        }
        
        // Bahis tahminlerini güncelle
        if (data.predictions.betting_predictions) {
            updateBettingPredictions(data.predictions.betting_predictions);
        }
    }
    
    // Takım formlarını güncelle
    populateTeamForms(data);
    
    // Motive/Form değerlerini güncelle
    updateMotivationTable(data);
    
    // İlk Yarı/Maç Sonu tahminlerini güncelle
    updateHTFTPredictions(data);
    
    // İstatistik butonlarını etkinleştir
    enableTeamStatButtons(data);
    
    // Tahminin ne zaman yapıldığını göster
    const timestamp = data.timestamp ? new Date(data.timestamp) : new Date();
    const formattedDate = timestamp.toLocaleString();
    $('#predictionTimestamp').text(formattedDate);
    
    // Tahmin modalını göster
    $('#predictionModal').modal('show');
    
    // Global değişken ile UI'ın başlatıldığını işaretle
    window.predictionModalInitialized = true;
    
    // console.log("Form/motivasyon verileri güncellendi:", {
    //     home_team_form: data.home_team.form,
    //     away_team_form: data.away_team.form,
    //     data.home_team.name,
    //     data.away_team.name
    // });

    // Eğer İY/MS bölümü zaten oluşturulmuşsa kaldır
    if ($('#htftPredictionSection').length > 0) {
        $('#htftPredictionSection').remove();
    }

    // Sayfa yüklendiğinde otomatik olarak aşağı kaydır
    $('#predictionModal').animate({ scrollTop: 0 }, 'slow');
}

function updateProbabilityBars(homeProb, drawProb, awayProb) {
    // Olasılık barlarını güncelle
    $('#homeWinBar').css('width', homeProb + '%').text(homeProb + '%');
    $('#drawBar').css('width', drawProb + '%').text(drawProb + '%');
    $('#awayWinBar').css('width', awayProb + '%').text(awayProb + '%');
}

// Skor tahminine göre bahis değerlerini hesapla
function calculateBettingValuesFromScore(homeGoals, awayGoals) {
    const totalGoals = homeGoals + awayGoals;
    
    // Tahmin sonuçlarını içeren nesne
    const predictions = {
        over_2_5: {
            prediction: totalGoals > 2 ? 'YES' : 'NO',
            display_value: totalGoals > 2 ? '2.5 ÜST' : '2.5 ALT',
            probability: totalGoals > 2 ? 0.9 : 0.85
        },
        btts: {
            prediction: (homeGoals > 0 && awayGoals > 0) ? 'YES' : 'NO',
            display_value: (homeGoals > 0 && awayGoals > 0) ? 'KG VAR' : 'KG YOK',
            probability: (homeGoals > 0 && awayGoals > 0) ? 0.85 : 0.8
        }
    };
    
    console.log("Skor üzerinden hesaplanan bahis tahminleri:", predictions);
    return predictions;
}

function updateBettingPredictions(bettingPredictions) {
    console.log("Bahis tahminleri güncelleniyor - YENİ TUTARLI ALGORİTMA");
    console.log("API'den gelen bahis tahminleri:", bettingPredictions);
    
    /* --- ADIM 1: KESİN SKOR TAHMİNİNİ BELİRLE --- */
    
    // DOM'daki mevcut skoru kontrol et (eğer zaten varsa)
    const currentDomScore = $('#predictedScore').text().trim();
    console.log("DOM'daki mevcut skor:", currentDomScore);
    
    // Skor değişkenlerini başlat
    let predictedScore = "";
    let scoreHomeGoals = 0;
    let scoreAwayGoals = 0;
    let validScore = false;
    
    // Öncelik 1: API yanıtında exact_score varsa kullan
    if (bettingPredictions.exact_score && bettingPredictions.exact_score.prediction) {
        try {
            const scoreFromAPI = bettingPredictions.exact_score.prediction;
            const [scoreHome, scoreAway] = scoreFromAPI.split("-").map(Number);
            
            if (!isNaN(scoreHome) && !isNaN(scoreAway)) {
                scoreHomeGoals = scoreHome;
                scoreAwayGoals = scoreAway;
                validScore = true;
                predictedScore = scoreFromAPI;
                console.log("Skor API'den başarıyla alındı:", predictedScore);
            }
        } catch(e) {
            console.error("API skor ayrıştırma hatası:", e);
        }
    } 
    // Öncelik 2: DOM'da bir skor varsa kullan
    else if (currentDomScore && currentDomScore.includes("-")) {
        try {
            const [scoreHome, scoreAway] = currentDomScore.split("-").map(Number);
            if (!isNaN(scoreHome) && !isNaN(scoreAway)) {
                scoreHomeGoals = scoreHome;
                scoreAwayGoals = scoreAway;
                validScore = true;
                predictedScore = currentDomScore;
                console.log("Skor DOM'dan alındı:", predictedScore);
            }
        } catch(e) {
            console.error("DOM skor ayrıştırma hatası:", e);
        }
    }
    
    // Geçerli bir skor bulunamadıysa, varsayılan 1-1 tahminini kullan
    if (!validScore) {
        console.warn("Geçerli bir skor tahmini bulunamadı, varsayılan 1-1 kullanılıyor");
        scoreHomeGoals = 1;
        scoreAwayGoals = 1;
        predictedScore = "1-1";
        validScore = true;
    }
    
    /* --- ADIM 2: KESİN SKOR ÜZERİNDEN TÜM BAHİS TAHMİNLERİNİ HESAPLA --- */
    
    // Toplam gol sayısını hesapla
    const totalGoals = scoreHomeGoals + scoreAwayGoals;
    
    // Skor üzerinden tüm bahis tahminlerini hesapla
    let calculatedPredictions = {
        // 2.5 ÜST/ALT tahmini
        over_2_5: {
            prediction: totalGoals > 2 ? 'YES' : 'NO',
            display_value: totalGoals > 2 ? '2.5 ÜST' : '2.5 ALT',
            probability: totalGoals > 2 ? 0.9 : 0.85
        },
        
        // KG VAR/YOK tahmini
        btts: {
            prediction: (scoreHomeGoals > 0 && scoreAwayGoals > 0) ? 'YES' : 'NO',
            display_value: (scoreHomeGoals > 0 && scoreAwayGoals > 0) ? 'KG VAR' : 'KG YOK',
            probability: (scoreHomeGoals > 0 && scoreAwayGoals > 0) ? 0.85 : 0.8
        },
        
        // Maç sonucu tahmini
        match_winner: {
            prediction: scoreHomeGoals > scoreAwayGoals ? 'HOME' : (scoreAwayGoals > scoreHomeGoals ? 'AWAY' : 'DRAW'),
            display_value: scoreHomeGoals > scoreAwayGoals ? 'EV SAHİBİ KAZANIR' : (scoreAwayGoals > scoreHomeGoals ? 'DEPLASMAN KAZANIR' : 'BERABERLIK'),
            probability: 0.85
        }
    };
    
    console.log("Skor üzerinden hesaplanan bahis tahminleri:", calculatedPredictions);
    
    // API'den gelen tahminleri, SKOR'dan hesaplanan değerlerle değiştir
    // Bu, tutarsızlığı önlemek için çok önemli!
    
    // 2.5 ÜST/ALT tahminini güncelle
    if (bettingPredictions.over_2_5_goals) {
        bettingPredictions.over_2_5_goals.prediction = calculatedPredictions.over_2_5.prediction;
        bettingPredictions.over_2_5_goals.display_value = calculatedPredictions.over_2_5.display_value;
        // Olasılık değerini tut ama aşırı düşükse düzelt
        if (totalGoals > 2 && bettingPredictions.over_2_5_goals.probability < 0.6) {
            bettingPredictions.over_2_5_goals.probability = 0.75;
            console.warn("ÜST 2.5 olasılığı düşüktü, 0.75 olarak düzeltildi");
        } else if (totalGoals <= 2 && bettingPredictions.over_2_5_goals.probability > 0.4) {
            bettingPredictions.over_2_5_goals.probability = 0.25;
            console.warn("ALT 2.5 olasılığı yüksekti, 0.25 olarak düzeltildi");
        }
    } else {
        // Eğer API'de yoksa, hesaplanan değeri ekle
        bettingPredictions.over_2_5_goals = calculatedPredictions.over_2_5;
    }
        
    // KG VAR/YOK tahminini güncelle
    if (bettingPredictions.btts) {
        bettingPredictions.btts.prediction = calculatedPredictions.btts.prediction;
        bettingPredictions.btts.display_value = calculatedPredictions.btts.display_value;
        // Olasılık değerini tut ama aşırı düşükse düzelt
        const bttsActual = (scoreHomeGoals > 0 && scoreAwayGoals > 0);
        if (bttsActual && bettingPredictions.btts.probability < 0.6) {
            bettingPredictions.btts.probability = 0.75;
            console.warn("KG VAR olasılığı düşüktü, 0.75 olarak düzeltildi");
        } else if (!bttsActual && bettingPredictions.btts.probability > 0.4) {
            bettingPredictions.btts.probability = 0.25;
            console.warn("KG YOK olasılığı yüksekti, 0.25 olarak düzeltildi");
        }
    } else {
        // Eğer API'de yoksa, hesaplanan değeri ekle
        bettingPredictions.btts = calculatedPredictions.btts;
    }
    
    /* --- ADIM 3: UI'YI GÜNCELLE --- */
    
    // Toplam gol sayısı göstergesini güncelle
    $('#totalGoals').text(totalGoals.toFixed(1));
    
    // ÜST/ALT değerlerini alıp UI'yi güncelle
    const over25Prediction = bettingPredictions.over_2_5_goals.prediction;
    const over25Value = bettingPredictions.over_2_5_goals.display_value;
    const over25Prob = bettingPredictions.over_2_5_goals.probability;
    const isOver25 = over25Prediction === 'YES';
    
    // KG VAR/YOK değerlerini alıp UI'yi güncelle
    const bttsPrediction = bettingPredictions.btts.prediction;
    const bttsValue = bettingPredictions.btts.display_value;
    const bttsProbValue = bettingPredictions.btts.probability;
    const isBtts = bttsPrediction === 'YES';
    
    console.log("Hesaplanan bahis değerleri:", {
        score: predictedScore,
        totalGoals,
        over_2_5: {value: over25Value, probability: over25Prob},
        btts: {value: bttsValue, probability: bttsProbValue}
    });
    
    // ÜST/ALT göstergesini güncelle
    if (isOver25) {
        $('#over25Value').text('ÜST');
        $('#over25Prob').text(Math.round(over25Prob * 100) + '%');
        $('#over25Icon').html('<i class="fas fa-arrow-up text-primary"></i>');
    } else {
        $('#over25Value').text('ALT');
        $('#over25Prob').text(Math.round((1 - over25Prob) * 100) + '%');
        $('#over25Icon').html('<i class="fas fa-arrow-down text-danger"></i>');
    }
    
    // KG VAR/YOK göstergesini güncelle
    if (isBtts) {
        $('#bttsValue').text('KG VAR');
        $('#bttsProb').text(Math.round(bttsProbValue * 100) + '%');
        $('#bttsIcon').html('<i class="fas fa-check text-success"></i>');
    } else {
        $('#bttsValue').text('KG YOK');
        $('#bttsProb').text(Math.round((1 - bttsProbValue) * 100) + '%');
        $('#bttsIcon').html('<i class="fas fa-times text-danger"></i>');
    }
    
    // Tüm bahis kartlarını görünür yap
    $('.betting-card').removeClass('d-none');
    
    // 2.5 ÜST/ALT bahis kartını güncelle
    updateBettingCard(
        '#over25Card',
        isOver25 ? 'ÜST' : 'ALT',
        Math.round(isOver25 ? over25Prob * 100 : (1 - over25Prob) * 100) + '%',
        isOver25 ? 'text-primary' : 'text-danger',
        isOver25 ? 'fa-arrow-up' : 'fa-arrow-down'
    );
    
    // KG VAR/YOK bahis kartını güncelle
    updateBettingCard(
        '#bttsCard',
        isBtts ? 'KG VAR' : 'KG YOK',
        Math.round(isBtts ? bttsProbValue * 100 : (1 - bttsProbValue) * 100) + '%',
        isBtts ? 'text-success' : 'text-danger',
        isBtts ? 'fa-check' : 'fa-times'
    );

    // Güncellenmiş tahmin verisini debugging için logla
    console.log("GÜNCELLENMIŞ VE TUTARLI BAHİS TAHMİNLERİ:", {
        exact_score: predictedScore,
        over_2_5: bettingPredictions.over_2_5_goals,
        btts: bettingPredictions.btts
    });
}

function updateBettingCard(cardSelector, value, probability, colorClass, iconClass) {
    const card = $(cardSelector);
    
    if (card.length) {
        const valueElement = card.find('.betting-value');
        const probabilityElement = card.find('.betting-probability');
        const iconElement = card.find('.betting-icon i');
        
        valueElement.text(value);
        probabilityElement.text(probability);
        
        // İkon rengini ve sınıfını güncelle
        iconElement.removeClass();
        iconElement.addClass('fas ' + iconClass + ' ' + colorClass);
    }
}
