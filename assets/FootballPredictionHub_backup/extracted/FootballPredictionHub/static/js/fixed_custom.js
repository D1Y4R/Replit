// Tahmin verilerini işleme ve görüntüleme işlevleri - Tutarlılık düzeltmesi ile

function updatePredictionUI(data) {
    console.log("updatePredictionUI çağrıldı");
    
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

function populateTeamForms(data) {
    // Eğer veriler varsa göster
    if (data.home_team && data.home_team.form && data.away_team && data.away_team.form) {
        displayTeamForm('#homeTeamForm', data.home_team.form);
        displayTeamForm('#awayTeamForm', data.away_team.form);
    }
}

function displayTeamForm(selector, formData) {
    if (formData) {
        // Form verilerini göster
        const formContainer = $(selector);
        formContainer.empty();
        
        // İstatistikleri göster
        const formTable = $('<table class="form-stats-table"></table>');
        
        // İstatistik satırları
        const rows = [
            { label: 'Maç', value: formData.matches_played || 0 },
            { label: 'Galibiyet', value: formData.wins || 0 },
            { label: 'Beraberlik', value: formData.draws || 0 },
            { label: 'Mağlubiyet', value: formData.losses || 0 },
            { label: 'Attığı Gol', value: formData.goals_scored || 0 },
            { label: 'Yediği Gol', value: formData.goals_conceded || 0 },
            { label: 'Maç Başı Gol', value: formData.avg_goals_scored ? formData.avg_goals_scored.toFixed(2) : '0.00' },
            { label: 'İlk Yarı Gol', value: formData.first_half_goals ? formData.first_half_goals.toFixed(2) : '0.00' }
        ];
        
        rows.forEach(row => {
            formTable.append(`<tr><td>${row.label}</td><td class="text-right">${row.value}</td></tr>`);
        });
        
        formContainer.append(formTable);
    }
}

function updateMotivationTable(data) {
    // Form ve motivasyon değerlendirmesi
    if (data.home_team && data.away_team) {
        const homeForm = data.home_team.form || {};
        const awayForm = data.away_team.form || {};
        
        // Form puanlarını hesapla (maksimum 5 üzerinden)
        const calculateFormScore = (form) => {
            if (!form) return 0;
            const matches = form.matches_played || 0;
            if (matches === 0) return 0;
            
            // Son 5 maçtaki performans
            const wins = form.wins || 0;
            const draws = form.draws || 0;
            
            // Basit form puanı: (galibiyet * 3 + beraberlik) / (olası maksimum puan)
            return Math.min(5, Math.round(((wins * 3 + draws) / (matches * 3)) * 5));
        };
        
        // Gol formu (0-5 arası)
        const calculateGoalForm = (form) => {
            if (!form) return 0;
            const avgGoals = form.avg_goals_scored || 0;
            // 0-2.5+ gol aralığını 0-5 puana dönüştür
            return Math.min(5, Math.round(avgGoals * 2));
        };
        
        // Ev/Deplasman avantajı (0-5 arası)
        const homeAdvantage = 4; // Ev sahibi avantajı
        const awayDisadvantage = 2; // Deplasman dezavantajı
        
        // Yakın zamanda oynanan maç yoğunluğu (0-5 arası, 5 en yorgun)
        const calculateFatigue = (form) => {
            if (!form || !form.recent_matches) return 0;
            const recentMatches = form.recent_matches.length;
            // Son 14 günde 4+ maç oynamak yorgunluk yaratır
            return Math.min(5, Math.round(recentMatches / 2));
        };
        
        // Puan farkı - sıralamadaki fark veya son maçlardaki fark
        const leaguePositionDiff = () => {
            const homePos = data.home_team.league_position || 10;
            const awayPos = data.away_team.league_position || 10;
            // Pozisyon farkını 0-5 aralığına normalleştir
            return Math.min(5, Math.max(0, Math.round(Math.abs(homePos - awayPos) / 4)));
        };
        
        // Motivasyon takımları son maçlarındaki trend ve yaklaşan önemli maçlara göre değerlendirir
        const calculateMotivation = (team) => {
            // Varsayılan olarak orta düzeyde motivasyon
            return 3;
        };
        
        // Hesaplamaları yap
        const homeFormScore = calculateFormScore(homeForm);
        const awayFormScore = calculateFormScore(awayForm);
        const homeGoalForm = calculateGoalForm(homeForm);
        const awayGoalForm = calculateGoalForm(awayForm);
        const homeFatigue = calculateFatigue(homeForm);
        const awayFatigue = calculateFatigue(awayForm);
        const positionDiff = leaguePositionDiff();
        const homeMotivation = calculateMotivation(data.home_team);
        const awayMotivation = calculateMotivation(data.away_team);
        
        // Form ve motivasyon tablosunu güncelle
        updateFormMotivationUI(
            homeFormScore, awayFormScore,
            homeGoalForm, awayGoalForm,
            homeAdvantage, awayDisadvantage,
            homeFatigue, awayFatigue,
            positionDiff,
            homeMotivation, awayMotivation
        );
    }
}

function updateFormMotivationUI(
    homeForm, awayForm,
    homeGoalForm, awayGoalForm,
    homeAdvantage, awayDisadvantage,
    homeFatigue, awayFatigue,
    positionDiff,
    homeMotivation, awayMotivation
) {
    // Motivasyon tablosu
    const motivationTable = $('#motivationTable');
    motivationTable.empty();
    
    // Faktörleri listeye ekle
    const factors = [
        { name: 'Form', home: homeForm, away: awayForm, higher_better: true },
        { name: 'Gol Formu', home: homeGoalForm, away: awayGoalForm, higher_better: true },
        { name: 'Ev/Dep. Faktörü', home: homeAdvantage, away: awayDisadvantage, higher_better: true },
        { name: 'Yorgunluk', home: homeFatigue, away: awayFatigue, higher_better: false },
        { name: 'Motivasyon', home: homeMotivation, away: awayMotivation, higher_better: true }
    ];
    
    // Tablo başlığı
    const headerRow = $('<tr></tr>');
    headerRow.append('<th>Faktör</th>');
    headerRow.append('<th class="text-center">Ev</th>');
    headerRow.append('<th class="text-center">Deplasman</th>');
    motivationTable.append(headerRow);
    
    // Faktörleri tabloya ekle
    factors.forEach(factor => {
        const row = $('<tr></tr>');
        
        // Faktör adı
        row.append(`<td>${factor.name}</td>`);
        
        // Ev sahibi değeri
        const homeClass = getFavorableClass(factor.home, factor.away, factor.higher_better);
        row.append(`<td class="text-center ${homeClass}">${getStarRating(factor.home)}</td>`);
        
        // Deplasman değeri
        const awayClass = getFavorableClass(factor.away, factor.home, factor.higher_better);
        row.append(`<td class="text-center ${awayClass}">${getStarRating(factor.away)}</td>`);
        
        motivationTable.append(row);
    });
}

function getFavorableClass(value1, value2, higher_better) {
    if (higher_better) {
        if (value1 > value2) return 'text-success';
        if (value1 < value2) return 'text-danger';
    } else {
        if (value1 < value2) return 'text-success';
        if (value1 > value2) return 'text-danger';
    }
    return '';
}

function getStarRating(value) {
    // 0-5 arası değeri yıldız olarak göster
    const fullStars = Math.floor(value);
    const halfStar = value - fullStars >= 0.5;
    const emptyStars = 5 - fullStars - (halfStar ? 1 : 0);
    
    let stars = '';
    
    // Dolu yıldızlar
    for (let i = 0; i < fullStars; i++) {
        stars += '<i class="fas fa-star"></i>';
    }
    
    // Yarım yıldız
    if (halfStar) {
        stars += '<i class="fas fa-star-half-alt"></i>';
    }
    
    // Boş yıldızlar
    for (let i = 0; i < emptyStars; i++) {
        stars += '<i class="far fa-star"></i>';
    }
    
    return stars;
}

function updateHTFTPredictions(data) {
    // IY/MS tahminleri için UI oluştur
    if (data.predictions && data.predictions.betting_predictions && data.predictions.betting_predictions.half_time_full_time) {
        createHTFTPredictionUI(data.predictions.betting_predictions.half_time_full_time);
    }
}

function createHTFTPredictionUI(htftData) {
    // IY/MS tahmin alanı
    const htftSection = $('<div id="htftPredictionSection" class="mt-4"></div>');
    
    // Başlık
    htftSection.append('<h5 class="text-center mb-3">İlk Yarı / Maç Sonu Tahminleri</h5>');
    
    // 3x3 tablo oluştur
    const htftTable = $('<table class="table table-bordered htft-table"></table>');
    const outcomes = ['1', 'X', '2']; // İlk Yarı: 1=Ev Sahibi Önde, X=Berabere, 2=Deplasman Önde
    
    // Tablo başlığı
    const headerRow = $('<tr></tr>');
    headerRow.append('<th class="text-center">İY / MS</th>');
    outcomes.forEach(outcome => {
        headerRow.append(`<th class="text-center">MS: ${outcome}</th>`);
    });
    htftTable.append(headerRow);
    
    // Tablo gövdesi
    outcomes.forEach(htOutcome => {
        const row = $('<tr></tr>');
        row.append(`<th class="text-center">İY: ${htOutcome}</th>`);
        
        outcomes.forEach(ftOutcome => {
            const combination = `${htOutcome}/${ftOutcome}`;
            const cellClass = getCombinationClass(combination, htftData);
            const probability = getCombinationProbability(combination, htftData);
            
            row.append(`<td class="text-center ${cellClass}">${combination}<br><small>${probability}%</small></td>`);
        });
        
        htftTable.append(row);
    });
    
    htftSection.append(htftTable);
    
    // En olası İY/MS kombinasyonunu göster
    if (htftData.prediction) {
        const mostLikely = htftData.prediction;
        const mostLikelyProb = htftData.probability ? Math.round(htftData.probability * 100) : "?";
        
        htftSection.append(`
            <div class="alert alert-info mt-3">
                <strong>En Olası İY/MS:</strong> ${mostLikely} (${mostLikelyProb}%)
            </div>
        `);
    }
    
    // Prediction modal'a ekle
    $('#predictionModal .modal-body').append(htftSection);
}

function getCombinationClass(combination, htftData) {
    // Eğer bu kombinasyon en olası olanıysa vurgula
    if (htftData.prediction === combination) {
        return 'bg-info text-white font-weight-bold';
    }
    
    // Diğer durumlarda, muhtemel kombinasyonları vurgula
    // Örneğin, ev sahibi galibiyet kombinasyonları için yeşil ton
    if (combination.endsWith('/1')) {
        return 'bg-success-light';
    }
    // Beraberlik kombinasyonları için gri ton
    else if (combination.endsWith('/X')) {
        return 'bg-secondary-light';
    }
    // Deplasman galibiyet kombinasyonları için mavi ton
    else if (combination.endsWith('/2')) {
        return 'bg-primary-light';
    }
    
    return '';
}

function getCombinationProbability(combination, htftData) {
    // API'den gelen olasılıkları kullan
    if (htftData.probabilities && htftData.probabilities[combination]) {
        return Math.round(htftData.probabilities[combination] * 100);
    }
    
    // API'de yoksa, varsayılan değerler kullan
    // En muhtemel kombinasyon için daha yüksek olasılık
    if (htftData.prediction === combination) {
        return Math.round(htftData.probability * 100) || 30;
    }
    
    // Diğer kombinasyonlar için düşük olasılıklar
    const defaultValues = {
        '1/1': 15, '1/X': 8, '1/2': 4,
        'X/1': 10, 'X/X': 12, 'X/2': 8,
        '2/1': 5, '2/X': 8, '2/2': 15
    };
    
    return defaultValues[combination] || 5;
}

function enableTeamStatButtons(data) {
    // Takım istatistik butonlarını etkinleştir
    if (data.home_team && data.home_team.id) {
        $('#homeTeamStatBtn').attr('data-team-id', data.home_team.id);
        $('#homeTeamStatBtn').removeClass('disabled');
    }
    
    if (data.away_team && data.away_team.id) {
        $('#awayTeamStatBtn').attr('data-team-id', data.away_team.id);
        $('#awayTeamStatBtn').removeClass('disabled');
    }
}

// Prediction Modal Kapatma
$(document).on('click', '#closePredictionModal', function() {
    $('#predictionModal').modal('hide');
    
    // Arka plan filtresini kaldır
    $('.filter-blur').removeClass('filter-blur');
    console.log("Modal kapandı, arka plan filtreleri temizlendi");
});

// Takım İstatistik Butonu Click Yönetimi
$(document).on('click', '.team-stat-btn', function() {
    const teamId = $(this).attr('data-team-id');
    if (teamId) {
        showTeamStats(teamId);
    }
});

// Tahmin yenileme
function refreshPrediction() {
    console.log("refreshPrediction fonksiyonu çalıştırıldı");
    
    // Global değişkenden tahmin verilerini al
    if (window.predictionData) {
        // Tahmin verilerini yenile
        console.log("Tahmin verileri yenilendi:", window.predictionData);
        updatePredictionUI(window.predictionData);
    } else {
        console.error("Tahmin güncellenirken hata:", window.predictionData);
    }
}

// Takım istatistikleri popup'ı
function showTeamStats(teamId) {
    // API'den takım maçlarını al
    $.ajax({
        url: `/api/v3/fixtures/team/${teamId}`,
        method: 'GET',
        success: function(data) {
            // Takım istatistikleri modalını oluştur ve göster
            createTeamStatsModal(data, teamId);
        },
        error: function(err) {
            console.error("Takım istatistikleri alınırken hata:", err);
            showErrorModal("Takım istatistikleri alınamadı. Lütfen daha sonra tekrar deneyin.");
        }
    });
}

function createTeamStatsModal(matches, teamId) {
    // Takım adını al
    const teamName = window.predictionData ? 
                    (window.predictionData.home_team.id == teamId ? 
                     window.predictionData.home_team.name : window.predictionData.away_team.name) : 
                    "Takım";
    
    // Modal oluştur
    const modal = $(`
        <div class="modal fade" id="teamStatsModal" tabindex="-1" role="dialog" aria-labelledby="teamStatsModalLabel" aria-hidden="true">
            <div class="modal-dialog modal-lg" role="document">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="teamStatsModalLabel">${teamName} - Son Maçlar</h5>
                        <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                            <span aria-hidden="true">&times;</span>
                        </button>
                    </div>
                    <div class="modal-body">
                        <div class="matches-list">
                            <table class="table">
                                <thead>
                                    <tr>
                                        <th>Tarih</th>
                                        <th>Maç</th>
                                        <th>Skor</th>
                                    </tr>
                                </thead>
                                <tbody id="teamMatches">
                                </tbody>
                            </table>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-dismiss="modal">Kapat</button>
                    </div>
                </div>
            </div>
        </div>
    `);
    
    // Maçları ekle
    const matchesList = modal.find('#teamMatches');
    if (matches && matches.length > 0) {
        matches.forEach(match => {
            matchesList.append(`
                <tr>
                    <td>${match.date}</td>
                    <td>${match.match}</td>
                    <td>${match.score}</td>
                </tr>
            `);
        });
    } else {
        matchesList.append(`
            <tr>
                <td colspan="3" class="text-center">Bu takım için maç verisi bulunamadı.</td>
            </tr>
        `);
    }
    
    // Modalı ekle ve göster
    $('body').append(modal);
    $('#teamStatsModal').modal('show');
    
    // Modal kapandığında DOM'dan kaldır
    $('#teamStatsModal').on('hidden.bs.modal', function () {
        $(this).remove();
    });
}

function showErrorModal(message) {
    // Basit hata modalı
    const modal = $(`
        <div class="modal fade" id="errorModal" tabindex="-1" role="dialog" aria-hidden="true">
            <div class="modal-dialog" role="document">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Hata</h5>
                        <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                            <span aria-hidden="true">&times;</span>
                        </button>
                    </div>
                    <div class="modal-body">
                        <p>${message}</p>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-dismiss="modal">Tamam</button>
                    </div>
                </div>
            </div>
        </div>
    `);
    
    // Modalı ekle ve göster
    $('body').append(modal);
    $('#errorModal').modal('show');
    
    // Modal kapandığında DOM'dan kaldır
    $('#errorModal').on('hidden.bs.modal', function () {
        $(this).remove();
    });
}