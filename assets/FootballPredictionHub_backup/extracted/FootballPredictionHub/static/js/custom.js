function updatePredictionUI(data) {
    // Ana tahmin detaylarını güncelle
    if (data && data.predictions) {
        // Gol beklentileri
        $('#homeExpectedGoals').text(data.predictions.expected_goals.home.toFixed(2));
        $('#awayExpectedGoals').text(data.predictions.expected_goals.away.toFixed(2));
        $('#totalExpectedGoals').text((data.predictions.expected_goals.home + data.predictions.expected_goals.away).toFixed(2));

        // Maç sonucu olasılıkları
        $('#homeWinProb').text(data.predictions.home_win_probability.toFixed(2) + '%');
        $('#drawProb').text(data.predictions.draw_probability.toFixed(2) + '%');
        $('#awayWinProb').text(data.predictions.away_win_probability.toFixed(2) + '%');

        // En olası sonuç
        let outcomeText = '';
        switch (data.predictions.most_likely_outcome) {
            case 'HOME_WIN': outcomeText = 'Ev Sahibi Kazanır'; break;
            case 'DRAW': outcomeText = 'Beraberlik'; break;
            case 'AWAY_WIN': outcomeText = 'Deplasman Kazanır'; break;
        }
        $('#mostLikelyOutcome').text(outcomeText);

        // Bahis tahminleri
        if (data.predictions.betting_predictions) {
            const betting = data.predictions.betting_predictions;

            // KG Var/Yok
            let formattedBothTeamsToScore = betting.both_teams_to_score.prediction;
            // YES/NO gibi değerleri Türkçe formata dönüştür
            if (formattedBothTeamsToScore === 'YES' || formattedBothTeamsToScore.toLowerCase() === 'yes') {
                formattedBothTeamsToScore = 'KG VAR';
            } else if (formattedBothTeamsToScore === 'NO' || formattedBothTeamsToScore.toLowerCase() === 'no') {
                formattedBothTeamsToScore = 'KG YOK';
            }
            $('#bttsValue').text(formattedBothTeamsToScore);
            $('#bttsProb').text(betting.both_teams_to_score.probability.toFixed(2) + '%');

            // 2.5 Üst/Alt
            let formattedOver25 = betting.over_2_5_goals.prediction;
            // YES/NO gibi değerleri Türkçe formata dönüştür
            if (formattedOver25 === 'YES' || formattedOver25.toLowerCase() === 'yes') {
                formattedOver25 = '2.5 ÜST';
            } else if (formattedOver25 === 'NO' || formattedOver25.toLowerCase() === 'no') {
                formattedOver25 = '2.5 ALT';
            }
            $('#over25Value').text(formattedOver25);
            $('#over25Prob').text(betting.over_2_5_goals.probability.toFixed(2) + '%');

            // 3.5 Üst/Alt
            let formattedOver35 = betting.over_3_5_goals.prediction;
            // YES/NO gibi değerleri Türkçe formata dönüştür
            if (formattedOver35 === 'YES' || formattedOver35.toLowerCase() === 'yes') {
                formattedOver35 = '3.5 ÜST';
            } else if (formattedOver35 === 'NO' || formattedOver35.toLowerCase() === 'no') {
                formattedOver35 = '3.5 ALT';
            }
            $('#over35Value').text(formattedOver35);
            $('#over35Prob').text(betting.over_3_5_goals.probability.toFixed(2) + '%');

            // Kesin skor
            $('#exactScoreValue').text(betting.exact_score.prediction);
            $('#exactScoreProb').text(betting.exact_score.probability.toFixed(2) + '%');
        }

        // Gelişmiş model sonuçlarını güncelle
        updateAdvancedModelsTable(data);
    }
}

function updateAdvancedModelsTable(data) {
    if (!data || !data.predictions) return;

    // Standart tahmin modeli (Monte Carlo)
    const standardHomeGoals = data.predictions.raw_metrics.expected_home_goals;
    const standardAwayGoals = data.predictions.raw_metrics.expected_away_goals;
    $('#standardHomeGoals').text(standardHomeGoals.toFixed(2));
    $('#standardAwayGoals').text(standardAwayGoals.toFixed(2));
    $('#standardPrediction').text(getPredictionText(standardHomeGoals, standardAwayGoals));

    // Sinir ağı tahminleri
    if (data.predictions.neural_predictions) {
        const neuralHomeGoals = data.predictions.neural_predictions.home_goals;
        const neuralAwayGoals = data.predictions.neural_predictions.away_goals;
        $('#neuralHomeGoals').text(neuralHomeGoals.toFixed(2));
        $('#neuralAwayGoals').text(neuralAwayGoals.toFixed(2));
        $('#neuralPrediction').text(getPredictionText(neuralHomeGoals, neuralAwayGoals));
    }

    // Zero-Inflated Poisson ve Ensemble modeli
    if (data.predictions.advanced_models && data.predictions.advanced_models.zero_inflated_poisson) {
        const zipHomeGoals = data.predictions.advanced_models.zero_inflated_poisson.expected_goals.home;
        const zipAwayGoals = data.predictions.advanced_models.zero_inflated_poisson.expected_goals.away;
        $('#zipHomeGoals').text(zipHomeGoals.toFixed(2));
        $('#zipAwayGoals').text(zipAwayGoals.toFixed(2));
        $('#zipPrediction').text(getPredictionText(zipHomeGoals, zipAwayGoals));

        // Kombinasyon sonucu
        if (data.predictions.advanced_models.final_combined_prediction) {
            const combinedHomeGoals = data.predictions.advanced_models.final_combined_prediction.home_goals;
            const combinedAwayGoals = data.predictions.advanced_models.final_combined_prediction.away_goals;
            $('#combinedHomeGoals').text(combinedHomeGoals.toFixed(2));
            $('#combinedAwayGoals').text(combinedAwayGoals.toFixed(2));
            $('#combinedPrediction').text(getPredictionText(combinedHomeGoals, combinedAwayGoals));
        }
    } else {
        // Gelişmiş modeller yoksa bu satırı gizle
        $('tr:contains("Zero-Inflated Poisson/Ensemble")').hide();
        // Kombinasyon satırını da güncelle
        const combinedHomeGoals = data.predictions.expected_goals.home;
        const combinedAwayGoals = data.predictions.expected_goals.away;
        $('#combinedHomeGoals').text(combinedHomeGoals.toFixed(2));
        $('#combinedAwayGoals').text(combinedAwayGoals.toFixed(2));
        $('#combinedPrediction').text(getPredictionText(combinedHomeGoals, combinedAwayGoals));
    }
}

function getPredictionText(homeGoals, awayGoals) {
    const diff = homeGoals - awayGoals;
    if (diff > 0.5) return 'Ev Sahibi Kazanır';
    if (diff < -0.5) return 'Deplasman Kazanır';
    return 'Beraberlik';
}

// Tahmin detayları için modal göster
$(document).on('click', '.predict-match-btn', function() {
    const homeTeamId = $(this).data('home-id');
    const awayTeamId = $(this).data('away-id');
    const homeTeamName = $(this).data('home-name');
    const awayTeamName = $(this).data('away-name');

    // Global window.showPrediction fonksiyonunu kullan (index.html içinde tanımlı)
    if (typeof window.showPrediction === 'function') {
        window.showPrediction(homeTeamId, awayTeamId, homeTeamName, awayTeamName, false);
    } else {
        console.error("showPrediction fonksiyonu bulunamadı!");
    }
});

// Tahmin göster (belirli bir maç için) - KULLANILMIYOR, sadece index.html içindeki sürüm kullanılacak
// Bu fonksiyon artık kullanılmamaktadır. 
// index.html'deki showPrediction fonksiyonu ile çakışma önlemek için kaldırıldı
// İlgili kod için templates/index.html dosyasına bakınız
function showPredictionCustomJs(homeTeamId, awayTeamId, homeTeamName, awayTeamName) {
    console.warn("custom.js içindeki showPrediction fonksiyonu artık kullanılmıyor!");
    console.warn("Lütfen index.html içindeki showPrediction fonksiyonunu kullanın");
    
    // Çakışmaları önlemek için, doğrudan index.html'deki versiyonu çağır
    if (typeof window.showPrediction === 'function') {
        window.showPrediction(homeTeamId, awayTeamId, homeTeamName, awayTeamName, false);
    } else {
        console.error("Global showPrediction fonksiyonu bulunamadı");
    }
}

function formatPrediction(type, prediction) {
    // main.js'deki formatPrediction fonksiyonuna göre güncellendi
    switch (type) {
        case 'over_2_5_goals':
        case '2.5_üst_alt':
            if (prediction === 'YES' || prediction === '2.5 ÜST') return '2.5 ÜST';
            if (prediction === 'NO' || prediction === '2.5 ALT') return '2.5 ALT';
            return prediction;
        case 'over_3_5_goals':
        case '3.5_üst_alt':
            if (prediction === 'YES' || prediction === '3.5 ÜST') return '3.5 ÜST';
            if (prediction === 'NO' || prediction === '3.5 ALT') return '3.5 ALT';
            return prediction;
        case 'both_teams_to_score':
        case 'kg_var_yok':
            if (prediction === 'YES' || prediction === 'KG VAR') return 'KG VAR';
            if (prediction === 'NO' || prediction === 'KG YOK') return 'KG YOK';
            return prediction;
        case 'match_result':
        case 'maç_sonucu':
            switch(prediction) {
                case 'HOME_WIN': return 'MS1';
                case 'DRAW': return 'X';
                case 'AWAY_WIN': return 'MS2';
                case 'MS1': return 'MS1';
                case 'X': return 'X';
                case 'MS2': return 'MS2';
                default: return prediction;
            }
        default:
            // Genel YES/NO değerlerini çevir
            if (prediction === 'YES') return 'VAR';
            if (prediction === 'NO') return 'YOK';
            return prediction;
    }
}

function refreshPrediction(homeTeamId, awayTeamId, homeTeamName, awayTeamName) {
    // Yükleniyor göster, içeriği gizle
    $('#predictionLoading').show();
    $('#predictionContent').hide();
    $('#predictionError').hide();
    
    // API isteği yap - force_update parametresi true olarak gönder
    const url = `/api/predict-match/${homeTeamId}/${awayTeamId}?home_name=${encodeURIComponent(homeTeamName)}&away_name=${encodeURIComponent(awayTeamName)}&force_update=true`;
    
    $.ajax({
        url: url,
        type: 'GET',
        dataType: 'json',
        success: function(data) {
            $('#predictionLoading').hide();
            $('#predictionContent').show();
            // Tahmin sonuçlarını güncelle
            updatePredictionUI(data);
        },
        error: function(error) {
            $('#predictionLoading').hide();
            $('#predictionError').text('Tahmin güncellenirken hata oluştu: ' + error.responseJSON?.error || 'Sunucu hatası').show();
        }
    });
}

// Sürpriz butonu işlevi: İlk Yarı/İkinci Yarı gol istatistiklerini göster
window.showTeamHalfTimeStats = function(homeId, awayId, homeName, awayName) {
    console.log("Sürpriz butonu işlevi çağrıldı:", {
        homeTeam: {id: homeId, name: homeName},
        awayTeam: {id: awayId, name: awayName}
    });
    
    // Takım ID'lerini sayısal değere dönüştür
    homeId = parseInt(homeId, 10) || 0;
    awayId = parseInt(awayId, 10) || 0;
    
    // Modal başlığını güncelle ve göster
    $('#predictionModalLabel').text(`${homeName} vs ${awayName} - İlk Yarı Performans İstatistikleri`);
    $('#predictionModal').modal('show');
    
    // Modal içeriğindeki başlığı da güncelle
    $('#matchTitle').text(`${homeName} vs ${awayName}`);
    
    // Yükleniyor göster, içeriği gizle
    $('#predictionLoading').show();
    $('#predictionContent').hide();
    $('#predictionError').hide();
    
    // Eğer takım ID'leri yoksa kullanıcıya bildir
    if (homeId === 0 || awayId === 0) {
        console.warn("TAKİM ID'LERİ BULUNAMADI:", {homeId, awayId});
        
        $('#predictionLoading').hide();
        $('#predictionContent').html(`
            <div class="alert alert-warning">
                <h4>Takım ID'leri bulunamadı</h4>
                <p>Bu maç için yarı istatistikleri gösterilemiyor çünkü takım ID'leri API'den alınamadı.</p>
                <p>Lütfen başka bir maç seçin veya daha sonra tekrar deneyin.</p>
            </div>
        `).show();
        return;
    }
    
    // Önce normal tahmin fonksiyonundan veriyi çekelim - ilk yarı skorları için
    const url = `/api/predict-match/${homeId}/${awayId}?home_name=${encodeURIComponent(homeName)}&away_name=${encodeURIComponent(awayName)}&force_update=false`;

    $.ajax({
        url: url,
        type: 'GET',
        dataType: 'json',
        success: function(data) {
            console.log("Tahmin verileri alındı - Yarı skorları analizi için:", data);
            
            // Eğer veri yoksa ya da bozuk veri dönerse
            if (!data || !data.home_team || !data.away_team) {
                $('#predictionLoading').hide();
                $('#predictionContent').html(`
                    <div class="alert alert-warning">
                        <h4>${homeName} vs ${awayName} - Veri Bulunamadı</h4>
                        <p>Bu maç için yarı istatistikleri alınamadı.</p>
                        <p>Lütfen daha sonra tekrar deneyin veya başka bir maç seçin.</p>
                    </div>
                `).show();
                return;
            }
            
            // Takımların son maçlarındaki ilk yarı/ikinci yarı gol verilerini hazırla
            const homeStats = processTeamHalfTimeStatsFromPrediction(data.home_team, homeName);
            const awayStats = processTeamHalfTimeStatsFromPrediction(data.away_team, awayName);
            
            // Bu fonksiyon tahmin verilerinden ilk yarı ve ikinci yarı istatistikleri çıkarır
            function processTeamHalfTimeStatsFromPrediction(teamData, teamName) {
                console.log("İşlenen takım verisi:", teamData);
                
                // Eğer takım verisi yoksa veya eksikse
                if (!teamData || !teamData.form || !teamData.form.detailed_data || !teamData.form.detailed_data.all) {
                    return {
                        status: "Veri bulunamadı",
                        matches: [],
                        message: "Bu takım için maç verisi bulunamadı",
                        team_id: teamData?.id || 0
                    };
                }
                
                // Dış JS dosyasındaki fonksiyonu kullan - versiyon ekliyoruz önbellek sorununu çözmek için
                return processTeamHalfTimeStats(teamData, teamName, "v2");
                
                // Maç sayaçları
                let homeMatchCount = 0;
                let awayMatchCount = 0;
                
                // Her maç için istatistikleri topla
                for (const match of matches) {
                    // İlk yarı gollerini al
                    const htGoalsFor = Number(match.ht_goals_scored || 0);
                    const htGoalsAgainst = Number(match.ht_goals_conceded || 0);
                    
                    // Toplam gollerini al
                    const ftGoalsFor = Number(match.goals_scored || 0); 
                    const ftGoalsAgainst = Number(match.goals_conceded || 0);
                    
                    // API'den alınan verilerde çoğunlukla ilk yarı ve tam maç skorları aynı
                    // Bu problemi çözmek için ikinci yarı gol sayılarını düzeltmemiz gerekiyor
                    
                    console.log("Maç verileri:", {
                        htGoalsFor: htGoalsFor,
                        htGoalsAgainst: htGoalsAgainst,
                        ftGoalsFor: ftGoalsFor,
                        ftGoalsAgainst: ftGoalsAgainst
                    });
                    
                    // İkinci yarı gollerini hesapla (tam maç - ilk yarı)
                    // NOT: Farklı liglerde veriler farklı gelebiliyor, en iyi yaklaşım aşağıdaki
                    let secondHalfGoalsFor, secondHalfGoalsAgainst;
                    
                    // İlk yarı ve tam maç skorları aynıysa, futbol gerçekliğinde bu neredeyse imkansız
                    // İstatistiksel olarak maçların %75'inden fazlasında ikinci yarıda en az 1 gol olur
                    if (htGoalsFor === ftGoalsFor && htGoalsAgainst === ftGoalsAgainst) {
                        // İstatistiksel olarak, genellikle ikinci yarıda ilk yarıya göre %10 daha fazla gol atılır
                        // Eğer ilk yarıda 0 gol varsa, ikinci yarıda 1 gol beklenir
                        
                        if (htGoalsFor === 0 && htGoalsAgainst === 0) {
                            // 0-0 ilk yarıdan sonra genellikle ikinci yarıda 1-2 gol olur
                            secondHalfGoalsFor = 1;
                            secondHalfGoalsAgainst = 0;
                        } else {
                            // İlk yarıda gol varsa, ikinci yarıda da benzer oranda gol beklenir
                            secondHalfGoalsFor = Math.max(1, Math.round(htGoalsFor * 1.1));
                            secondHalfGoalsAgainst = Math.max(0, Math.round(htGoalsAgainst * 0.9));
                        }
                    } else {
                        // Eğer farklıysa, gerçek farkı hesapla
                        secondHalfGoalsFor = Math.max(0, ftGoalsFor - htGoalsFor);
                        secondHalfGoalsAgainst = Math.max(0, ftGoalsAgainst - htGoalsAgainst);
                    }
                    
                    console.log("İkinci yarı goller (düzeltilmiş):", {
                        secondHalfGoalsFor: secondHalfGoalsFor,
                        secondHalfGoalsAgainst: secondHalfGoalsAgainst
                    });
                    
                    // Ev sahibi/deplasman ayrımına göre istatistikleri topla
                    if (match.is_home) {
                        homeFirstHalfGoalsFor += htGoalsFor;
                        homeFirstHalfGoalsAgainst += htGoalsAgainst;
                        homeSecondHalfGoalsFor += secondHalfGoalsFor;
                        homeSecondHalfGoalsAgainst += secondHalfGoalsAgainst;
                        homeMatchCount++;
                    } else {
                        awayFirstHalfGoalsFor += htGoalsFor;
                        awayFirstHalfGoalsAgainst += htGoalsAgainst;
                        awaySecondHalfGoalsFor += secondHalfGoalsFor;
                        awaySecondHalfGoalsAgainst += secondHalfGoalsAgainst;
                        awayMatchCount++;
                    }
                    
                    // İşlenmiş maç verisi
                    processedMatches.push({
                        match_id: match.match_id || 'unknown',
                        date: match.date || 'unknown',
                        opponent: match.opponent || 'unknown',
                        is_home: match.is_home,
                        first_half: {
                            goals_scored: htGoalsFor,
                            goals_conceded: htGoalsAgainst
                        },
                        second_half: {
                            goals_scored: secondHalfGoalsFor,
                            goals_conceded: secondHalfGoalsAgainst
                        },
                        full_time: {
                            goals_scored: ftGoalsFor,
                            goals_conceded: ftGoalsAgainst
                        }
                    });
                }
                
                // Toplam maç sayısı
                const totalMatches = homeMatchCount + awayMatchCount;
                
                // İlk ve ikinci yarı toplam golleri
                const totalFirstHalfGoalsFor = homeFirstHalfGoalsFor + awayFirstHalfGoalsFor;
                const totalFirstHalfGoalsAgainst = homeFirstHalfGoalsAgainst + awayFirstHalfGoalsAgainst;
                const totalSecondHalfGoalsFor = homeSecondHalfGoalsFor + awaySecondHalfGoalsFor;
                const totalSecondHalfGoalsAgainst = homeSecondHalfGoalsAgainst + awaySecondHalfGoalsAgainst;
                
                // NOT: İkinci yarı skorlarını düzgün hesaplamak için algoritma kullanıyoruz
                // Eğer ilk yarı skoru ile tam maç skoru aynıysa, istatistiklere dayalı hesaplama yapılıyor
                
                // Ortalamalar (maç sayısı sıfır değilse)
                const avgFirstHalfGoalsFor = totalMatches > 0 ? totalFirstHalfGoalsFor / totalMatches : 0;
                const avgFirstHalfGoalsAgainst = totalMatches > 0 ? totalFirstHalfGoalsAgainst / totalMatches : 0;
                const avgSecondHalfGoalsFor = totalMatches > 0 ? totalSecondHalfGoalsFor / totalMatches : 0;
                const avgSecondHalfGoalsAgainst = totalMatches > 0 ? totalSecondHalfGoalsAgainst / totalMatches : 0;
                
                // Sonuç formatını hazırla
                return {
                    team_id: teamData.id || 0,
                    team_name: teamName,
                    total_matches_analyzed: totalMatches,
                    status: "OK",
                    matches: processedMatches,
                    statistics: {
                        first_half: {
                            total_goals: totalFirstHalfGoalsFor,
                            avg_goals_per_match: parseFloat(avgFirstHalfGoalsFor.toFixed(2)),
                            home_goals: homeFirstHalfGoalsFor,
                            away_goals: awayFirstHalfGoalsFor
                        },
                        second_half: {
                            total_goals: totalSecondHalfGoalsFor,
                            avg_goals_per_match: parseFloat(avgSecondHalfGoalsFor.toFixed(2)),
                            home_goals: homeSecondHalfGoalsFor,
                            away_goals: awaySecondHalfGoalsFor
                        },
                        full_time: {
                            total_goals: totalFirstHalfGoalsFor + totalSecondHalfGoalsFor,
                            avg_goals_per_match: parseFloat((avgFirstHalfGoalsFor + avgSecondHalfGoalsFor).toFixed(2))
                        }
                    }
                };
            }
            
            console.log("İşlenmiş yarı istatistikleri:", {homeStats, awayStats});
        
            $('#predictionLoading').hide();
            $('#predictionContent').show();
            
            // Her iki takım için veri bulunamadıysa uyarı göster
            if ((homeStats.status === "Veri bulunamadı" || homeStats.matches?.length === 0) && 
                (awayStats.status === "Veri bulunamadı" || awayStats.matches?.length === 0)) {
                $('#predictionContent').html(`
                    <div class="row">
                        <div class="col-md-12">
                            <div class="alert alert-warning">
                                <h4>${homeName} vs ${awayName} - Veri Bulunamadı</h4>
                                <p>Her iki takım için ilk yarı/ikinci yarı istatistikleri bulunamadı. Bu durum şu nedenlerden kaynaklanabilir:</p>
                                <ul>
                                    <li>Takımların son dönemdeki maç verileri eksik olabilir</li>
                                    <li>Veri sağlayıcılarında bu takımlar için detaylı istatistikler mevcut değil</li>
                                    <li>Geçici bir bağlantı sorunu olabilir</li>
                                </ul>
                                <p>Lütfen daha sonra tekrar deneyin veya başka takımlar seçin.</p>
                            </div>
                        </div>
                    </div>
                `);
                return;
            }
            
            // Takım istatistiklerini göster
            let content = `
                <div class="row">
                    <div class="col-md-12">
                        <div class="alert alert-info">
                            <h4 id="matchTitle">${homeName} vs ${awayName} - İlk Yarı Performans İstatistikleri</h4>
                            <p>Son 21 maçtaki ilk ve ikinci yarı gol istatistikleri</p>
                        </div>
                    </div>
                </div>
            `;
            
            // Ev sahibi takım istatistikleri
            if (homeStats.status === "OK" && homeStats.total_matches_analyzed > 0) {
                content += `
                <div class="row mb-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header bg-primary text-white">
                                <h5>${homeName} - ${homeStats.total_matches_analyzed} maç analizi</h5>
                            </div>
                            <div class="card-body">
                                <table class="table table-striped">
                                    <thead>
                                        <tr>
                                            <th>İstatistik</th>
                                            <th>İlk Yarı (0-45 dk)</th>
                                            <th>İkinci Yarı (46-90 dk)</th>
                                            <th>Toplam</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td>Atılan Gol</td>
                                            <td>${homeStats.statistics.first_half.total_goals} (${homeStats.statistics.first_half.avg_goals_per_match} maç başına)</td>
                                            <td>${homeStats.statistics.second_half.total_goals} (${homeStats.statistics.second_half.avg_goals_per_match} maç başına)</td>
                                            <td>${homeStats.statistics.full_time.total_goals} (${homeStats.statistics.full_time.avg_goals_per_match} maç başına)</td>
                                        </tr>
                                        <tr>
                                            <td>Ev Sahibi / Deplasman</td>
                                            <td>Ev: ${homeStats.statistics.first_half.home_goals} / Dep: ${homeStats.statistics.first_half.away_goals}</td>
                                            <td>Ev: ${homeStats.statistics.second_half.home_goals} / Dep: ${homeStats.statistics.second_half.away_goals}</td>
                                            <td>Ev: ${homeStats.statistics.first_half.home_goals + homeStats.statistics.second_half.home_goals} / Dep: ${homeStats.statistics.first_half.away_goals + homeStats.statistics.second_half.away_goals}</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
                `;
            } else {
                content += `
                <div class="row">
                    <div class="col-md-12">
                        <div class="alert alert-warning">
                            <h5>${homeName} için yarı istatistikleri bulunamadı</h5>
                            <p>Bu takımın son dönemdeki maç verileri eksik olabilir.</p>
                        </div>
                    </div>
                </div>
                `;
            }
            
            // Deplasman takımı istatistikleri
            if (awayStats.status === "OK" && awayStats.total_matches_analyzed > 0) {
                content += `
                <div class="row">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header bg-danger text-white">
                                <h5>${awayName} - ${awayStats.total_matches_analyzed} maç analizi</h5>
                            </div>
                            <div class="card-body">
                                <table class="table table-striped">
                                    <thead>
                                        <tr>
                                            <th>İstatistik</th>
                                            <th>İlk Yarı (0-45 dk)</th>
                                            <th>İkinci Yarı (46-90 dk)</th>
                                            <th>Toplam</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td>Atılan Gol</td>
                                            <td>${awayStats.statistics.first_half.total_goals} (${awayStats.statistics.first_half.avg_goals_per_match} maç başına)</td>
                                            <td>${awayStats.statistics.second_half.total_goals} (${awayStats.statistics.second_half.avg_goals_per_match} maç başına)</td>
                                            <td>${awayStats.statistics.full_time.total_goals} (${awayStats.statistics.full_time.avg_goals_per_match} maç başına)</td>
                                        </tr>
                                        <tr>
                                            <td>Ev Sahibi / Deplasman</td>
                                            <td>Ev: ${awayStats.statistics.first_half.home_goals} / Dep: ${awayStats.statistics.first_half.away_goals}</td>
                                            <td>Ev: ${awayStats.statistics.second_half.home_goals} / Dep: ${awayStats.statistics.second_half.away_goals}</td>
                                            <td>Ev: ${awayStats.statistics.first_half.home_goals + awayStats.statistics.second_half.home_goals} / Dep: ${awayStats.statistics.first_half.away_goals + awayStats.statistics.second_half.away_goals}</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
                `;
            } else {
                content += `
                <div class="row">
                    <div class="col-md-12">
                        <div class="alert alert-warning">
                            <h5>${awayName} için yarı istatistikleri bulunamadı</h5>
                            <p>Bu takımın son dönemdeki maç verileri eksik olabilir.</p>
                        </div>
                    </div>
                </div>
                `;
            }
            
            // İstatistik karşılaştırması ve özet
            if (homeStats.status === "OK" && awayStats.status === "OK") {
                content += `
                <div class="row mt-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header bg-dark text-white">
                                <h5>${homeName} vs ${awayName} - Karşılaştırma</h5>
                            </div>
                            <div class="card-body">
                                <table class="table">
                                    <thead>
                                        <tr>
                                            <th>Dönem</th>
                                            <th>${homeName} (Maç başı gol)</th>
                                            <th>${awayName} (Maç başı gol)</th>
                                            <th>Fark</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td>İlk Yarı (0-45 dk)</td>
                                            <td>${homeStats.statistics.first_half.avg_goals_per_match}</td>
                                            <td>${awayStats.statistics.first_half.avg_goals_per_match}</td>
                                            <td>${(homeStats.statistics.first_half.avg_goals_per_match - awayStats.statistics.first_half.avg_goals_per_match).toFixed(2)}</td>
                                        </tr>
                                        <tr>
                                            <td>İkinci Yarı (46-90 dk)</td>
                                            <td>${homeStats.statistics.second_half.avg_goals_per_match}</td>
                                            <td>${awayStats.statistics.second_half.avg_goals_per_match}</td>
                                            <td>${(homeStats.statistics.second_half.avg_goals_per_match - awayStats.statistics.second_half.avg_goals_per_match).toFixed(2)}</td>
                                        </tr>
                                        <tr>
                                            <td>Toplam</td>
                                            <td>${homeStats.statistics.full_time.avg_goals_per_match}</td>
                                            <td>${awayStats.statistics.full_time.avg_goals_per_match}</td>
                                            <td>${(homeStats.statistics.full_time.avg_goals_per_match - awayStats.statistics.full_time.avg_goals_per_match).toFixed(2)}</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
                `;
            }
            
            // İlk yarı/maç sonu tahminlerini hesapla (eğer her iki takım için de veri varsa)
            let htftHtml = '';
            if (homeStats.status === "OK" && awayStats.status === "OK") {
                try {
                    // Takım ID'leri için global değişken kontrolü yap
                    var homeTeamId = data.home_team.id || ""; 
                    var awayTeamId = data.away_team.id || "";
                    
                    // Tahmin butonundan maç sonucu verilerini al
                    $.ajax({
                        url: `/api/predict-match/${homeTeamId}/${awayTeamId}?home_name=${encodeURIComponent(homeName)}&away_name=${encodeURIComponent(awayName)}&force_update=false`,
                        method: 'GET',
                        async: false, // Senkron çalıştır ki diğer kodlardan önce hazır olsun
                        success: function(predictionData) {
                            console.log("Tahmin butonu verileri:", predictionData);
                            
                            // İY/MS tahminlerini hesapla - tahmin butonu verileriyle uyumlu hale getir
                            const htftData = predictHalfTimeFullTime(homeStats, awayStats);
                            
                            // Tahmin butonu ile uyumluluğu sağla
                            // Eğer tahmin butonu ev sahibi kazanır diyorsa, İY/MS'de maç sonu 1 olanları artır
                            if (predictionData.outcome === "HOME_WIN") {
                                htftData.all_probabilities['1/1'] = Math.min(65, htftData.all_probabilities['1/1'] * 1.4);
                                htftData.all_probabilities['X/1'] = Math.min(45, htftData.all_probabilities['X/1'] * 1.2);
                                htftData.all_probabilities['2/1'] = Math.min(25, htftData.all_probabilities['2/1'] * 1.5);
                            } 
                            // Eğer tahmin butonu berabere diyorsa, İY/MS'de maç sonu X olanları artır
                            else if (predictionData.outcome === "DRAW") {
                                htftData.all_probabilities['1/X'] = Math.min(50, htftData.all_probabilities['1/X'] * 1.3);
                                htftData.all_probabilities['X/X'] = Math.min(60, htftData.all_probabilities['X/X'] * 1.5);
                                htftData.all_probabilities['2/X'] = Math.min(40, htftData.all_probabilities['2/X'] * 1.3);
                            }
                            // Eğer tahmin butonu deplasman kazanır diyorsa, İY/MS'de maç sonu 2 olanları artır
                            else if (predictionData.outcome === "AWAY_WIN") {
                                htftData.all_probabilities['1/2'] = Math.min(40, htftData.all_probabilities['1/2'] * 1.3);
                                htftData.all_probabilities['X/2'] = Math.min(45, htftData.all_probabilities['X/2'] * 1.2);
                                htftData.all_probabilities['2/2'] = Math.min(65, htftData.all_probabilities['2/2'] * 1.4);
                            }
                            
                            // Toplam 100'e normalizasyon
                            let total = 0;
                            for (const key in htftData.all_probabilities) {
                                total += htftData.all_probabilities[key];
                            }
                            
                            const factor = 100 / total;
                            for (const key in htftData.all_probabilities) {
                                htftData.all_probabilities[key] = Math.round(htftData.all_probabilities[key] * factor);
                                // Minimum değeri 3 olsun
                                if (htftData.all_probabilities[key] < 3) {
                                    htftData.all_probabilities[key] = 3;
                                }
                            }
                            
                            // En yüksek olasılıklı 3 tahmini bul
                            htftData.top_predictions = findTopPredictions(htftData.all_probabilities, 3);
                            
                            // En olası tahmin
                            htftData.prediction = htftData.top_predictions[0].prediction;
                            
                            // İlk yarı sonuç dağılımlarını alalım (mevcut istatistiklerden)
                            const homeFirstHalfResults = homeStats && homeStats.statistics && homeStats.statistics.first_half && homeStats.statistics.first_half.results 
                                ? homeStats.statistics.first_half.results 
                                : { total: { "1": 0, "X": 0, "2": 0 } };
                                
                            const awayFirstHalfResults = awayStats && awayStats.statistics && awayStats.statistics.first_half && awayStats.statistics.first_half.results 
                                ? awayStats.statistics.first_half.results 
                                : { total: { "1": 0, "X": 0, "2": 0 } };
                                
                            // Takımların maç sayılarını alalım
                            const homeMatches = homeStats && homeStats.total_matches_analyzed || 0;
                            const awayMatches = awayStats && awayStats.total_matches_analyzed || 0;
                            
                            // İY/MS tahminleri için HTML oluştur
                            htftHtml = generateHtFtPredictionHTML(htftData, homeName, awayName, homeFirstHalfResults, awayFirstHalfResults, homeMatches, awayMatches);
                            
                            console.log("İY/MS tahminleri hesaplandı (tahmin butonu ile uyumlu):", htftData);
                        },
                        error: function(err) {
                            console.error("Tahmin butonu verileri alınamadı:", err);
                            
                            // Hata durumunda normal hesaplamayı yap
                            const htftData = predictHalfTimeFullTime(homeStats, awayStats);
                            
                            // İlk yarı sonuç dağılımlarını alalım (mevcut istatistiklerden)
                            const homeFirstHalfResults = homeStats && homeStats.statistics && homeStats.statistics.first_half && homeStats.statistics.first_half.results 
                                ? homeStats.statistics.first_half.results 
                                : { total: { "1": 0, "X": 0, "2": 0 } };
                                
                            const awayFirstHalfResults = awayStats && awayStats.statistics && awayStats.statistics.first_half && awayStats.statistics.first_half.results 
                                ? awayStats.statistics.first_half.results 
                                : { total: { "1": 0, "X": 0, "2": 0 } };
                                
                            // Takımların maç sayılarını alalım
                            const homeMatches = homeStats && homeStats.total_matches_analyzed || 0;
                            const awayMatches = awayStats && awayStats.total_matches_analyzed || 0;
                            
                            htftHtml = generateHtFtPredictionHTML(htftData, homeName, awayName, homeFirstHalfResults, awayFirstHalfResults, homeMatches, awayMatches);
                            
                            console.log("İY/MS tahminleri hesaplandı (normal):", htftData);
                        }
                    });
                } catch (err) {
                    console.error("İY/MS tahminleri hesaplanırken hata oluştu:", err);
                    htftHtml = `
                        <div class="alert alert-warning mt-4">
                            <h5>İY/MS Tahmini Yapılamadı</h5>
                            <p>Tahmin hesaplaması sırasında bir hata oluştu: ${err.message}</p>
                        </div>
                    `;
                }
            }
            
            // İçeriği göster (istatistikler + İY/MS tahminleri)
            $('#predictionContent').html(content + htftHtml);
        },
        error: function(error) {
            console.error("Yarı istatistikleri alınırken hata:", error);
            $('#predictionLoading').hide();
            $('#predictionContent').html(`
                <div class="row">
                    <div class="col-md-12">
                        <div class="alert alert-warning">
                            <h4>${homeName} vs ${awayName} - Veri Alınamadı</h4>
                            <p>Takımların ilk yarı/ikinci yarı istatistikleri alınamadı. Bu durum şu nedenlerden kaynaklanabilir:</p>
                            <ul>
                                <li>Veri sağlayıcısına bağlantı sırasında sorun oluştu</li>
                                <li>Takımların son dönemdeki maç verileri eksik olabilir</li>
                                <li>Veri sağlayıcısı geçici olarak kullanılamıyor olabilir</li>
                            </ul>
                            <p>Lütfen daha sonra tekrar deneyin.</p>
                        </div>
                    </div>
                </div>
            `);
            $('#predictionContent').show();
        }
    });
}