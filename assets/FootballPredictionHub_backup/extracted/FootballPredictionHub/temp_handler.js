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
                            data.predictions.betting_predictions.over_2_5_goals.probability = 0.90;
                            console.warn("betting_predictions.over_2_5_goals verisi tutarlılık için düzeltildi");
                        }
                    }
                }
            }
        }
        
        // 2. Kesin skor kontrolü (fallback kontrol)
        if (!backendCorrectedScore && data.predictions.exact_score) {
            const exactScore = data.predictions.exact_score;
            if (exactScore && exactScore.includes('-')) {
                const [homeGoals, awayGoals] = exactScore.split('-').map(Number);
                const totalGoals = homeGoals + awayGoals;
                
                // Eğer toplam gol 3+ ama over_2_5_goals "ALT" olarak gösteriliyorsa düzelt
                if (totalGoals >= 3) {
                    const over25Prediction = data.predictions.betting_predictions?.over_2_5_goals?.prediction;
                    
                    if (over25Prediction !== 'YES' && over25Prediction !== '2.5 ÜST') {
                        console.error("TUTARSIZLIK: Skor", exactScore, "için toplam gol", totalGoals, "ama over_2_5_goals:", over25Prediction);
                        
                        // API yanıtında düzelt
                        if (data.predictions.betting_predictions && data.predictions.betting_predictions.over_2_5_goals) {
                            data.predictions.betting_predictions.over_2_5_goals.prediction = 'YES';
                            data.predictions.betting_predictions.over_2_5_goals.probability = 0.90;
                            console.warn("betting_predictions.over_2_5_goals verisi otomatik düzeltildi");
                        }
                    }
                }
            }
        }
    }

    // Eğer takım logoları varsa onları da göster
    if (data.home_team.form && data.home_team.form.logo) {
        $('#homeTeamLogo').attr('src', data.home_team.form.logo).show();
    }
    if (data.away_team.form && data.away_team.form.logo) {
        $('#awayTeamLogo').attr('src', data.away_team.form.logo).show();
    }

    // Tahmin özeti
    const predictions = data.predictions;
    
    // Konsensüs filtrelenmiş tahminleri kontrol et ve varsa kullan
    const filteredPredictions = predictions.filtered_predictions || {};
    console.log("Filtrelenmiş tahminler:", filteredPredictions);
    
    // Eğer backend'den gelen yeni tahmin bilgileri varsa, onları kullan
    const predictionData = predictions.betting_predictions || {};
    console.log("Backend'den gelen tahminler:", predictionData);
    
    // Maç sonucu olasılıkları - Filtrelenmiş tahminleri veya orijinalleri kullan
    let homeWinProb = predictions.home_win_probability;
    let drawProb = predictions.draw_probability;
    let awayWinProb = predictions.away_win_probability;
    
    // Filtrelenmiş değerler varsa kullan
    if (filteredPredictions.match_outcome) {
        console.log("Filtrelenmiş match_outcome:", filteredPredictions.match_outcome);
        
        if ('home_win' in filteredPredictions.match_outcome) {
            homeWinProb = Math.round(filteredPredictions.match_outcome.home_win * 100);
        }
        
        if ('draw' in filteredPredictions.match_outcome) {
            drawProb = Math.round(filteredPredictions.match_outcome.draw * 100);
        }
        
        if ('away_win' in filteredPredictions.match_outcome) {
            awayWinProb = Math.round(filteredPredictions.match_outcome.away_win * 100);
        }
    }
    
    // Tam olarak skor belirle - KRITIK: Öncelik sırası değiştirildi!
    // 1. Önce debug_final_check'i kontrol et (en güvenilir)
    // 2. Filtrelenmiş tahminlerdeki exact_score'u kontrol et
    // 3. Ana seviyedeki exact_score'u kontrol et
    // 4. betting_predictions içindeki exact_score'u kontrol et
    
    let exactScore = "";
    
    // DEBUG: Önce tüm skor değerlerini konsola yazalım
    console.log("DEBUG SKORS - Kontrol 1:", {
        "debug_final_check": predictions.debug_final_check,
        "filtered_exact_score": filteredPredictions.exact_score?.prediction,
        "main_exact_score": predictions.exact_score,
        "betting_exact_score": predictions.betting_predictions?.exact_score?.prediction
    });
    
    // Doğru skoru belirlemek için aşağıdaki adımları izliyoruz:
    
    // 1. En güvenilir ve nihai kaynak: debug_final_check
    // Bu konsenüs filtresinin sonunda Backend'in son hesapladığı değeri içerir
    if (predictions.debug_final_check && predictions.debug_final_check.includes('Skor=')) {
        try {
            // "Final kontrol: Skor=2-1, Sonuç=HOME_WIN" formatından skoru çıkar
            const scoreMatch = predictions.debug_final_check.match(/Skor=([0-9]+-[0-9]+)/);
            const outcomeMatch = predictions.debug_final_check.match(/Sonuç=([A-Z_]+)/);
            
            if (scoreMatch && scoreMatch[1]) {
                exactScore = scoreMatch[1];
                console.log("ÖNCELİKLİ: Skor debug_final_check'ten alındı:", exactScore);
                
                // Sonucu da debug_final_check'ten alıp doğrudan mostLikelyOutcome'a atayalım
                // Bu sayede skor ve sonuç tutarlılığını garanti ederiz
                if (outcomeMatch && outcomeMatch[1]) {
                    mostLikelyOutcome = outcomeMatch[1]; // Backend'den gelen nihai sonuç
                    console.log("ÖNCELİKLİ: Sonuç debug_final_check'ten alındı:", mostLikelyOutcome);
                }
            }
        } catch (e) {
            console.error("Debug final check parse hatası:", e);
        }
    }
    
    // 2. İkinci güvenilir kaynak: debug_exact_score_used 
    // Bu konsenüs filtresi tarafından gerçekten kullanılmış olan skordur
    if (!exactScore && predictions.debug_exact_score_used) {
        exactScore = predictions.debug_exact_score_used;
        console.log("İKİNCİL: Skor debug_exact_score_used'dan alındı:", exactScore);
    }
    
    // 2. Eğer debug_final_check'ten alamadıysak, filtrelenmiş tahminleri kontrol et
    if (!exactScore && filteredPredictions.exact_score && filteredPredictions.exact_score.prediction) {
        exactScore = filteredPredictions.exact_score.prediction;
        console.log("Kesin skor filtrelenmiş tahminlerden alındı:", exactScore);
    } 
    // 3. Eğer hala bulamadıysak, ana seviyeyi kontrol et
    else if (!exactScore && predictions.exact_score) {
        exactScore = predictions.exact_score;
        console.log("Kesin skor orijinal tahminlerden alındı:", exactScore);
    } 
    // 4. Son çare: betting_predictions'tan al
    else if (!exactScore && predictions.betting_predictions?.exact_score?.prediction) {
        exactScore = predictions.betting_predictions.exact_score.prediction;
        console.log("Kesin skor betting_predictions'tan alındı:", exactScore);
    }
    
    // Kesin skoru belirleyemediysek, çok ciddi bir sorun var
    if (!exactScore) {
        console.error("KRITIK HATA: Kesin skor belirlenemedi! Veri yapısı:", predictions);
        exactScore = "?-?"; // Varsayılan bir değer
    }
    
    // Kesin skordan maç sonucunu belirle
    let mostLikelyOutcome = "";
    
    // KRİTİK FIX: Eşit skorlar (0-0, 1-1, 2-2, vb.) için DRAW zorunlu
    // 1. ADIM: Her zaman kesin skoru kontrol et - Bu adım her zaman öncelikli
    if (exactScore && exactScore.includes('-')) {
        console.log("Kesin skor kontrolü başlıyor:", exactScore);
        const scoreParts = exactScore.split('-');
        if (scoreParts.length === 2) {
            const homeGoals = parseInt(scoreParts[0]);
            const awayGoals = parseInt(scoreParts[1]);
            
            // Skor değerlerinin geçerli olduğundan emin ol
            if (!isNaN(homeGoals) && !isNaN(awayGoals)) {
                // ÖNEMLİ: Skorlar eşitse, maç sonucu MUTLAKA beraberlik olmalı
                if (homeGoals === awayGoals) {
                    mostLikelyOutcome = "DRAW";
                    console.log("KRİTİK: Eşit skor tespit edildi, maç sonucu beraberlik olarak zorla ayarlandı:", exactScore);
                } else if (homeGoals > awayGoals) {
                    mostLikelyOutcome = "HOME_WIN";
                } else {
                    mostLikelyOutcome = "AWAY_WIN";
                }
                console.log("Kesin skora göre belirlenen sonuç:", mostLikelyOutcome);
                
                // Kesin skor varsa, olasılıkları da güncelle
                if (mostLikelyOutcome === "HOME_WIN") {
                    // Ev sahibi kazanırsa, olasılığı en az %60 olmalı
                    homeWinProb = Math.max(homeWinProb, 60);
                    drawProb = Math.min(drawProb, 30);
                    awayWinProb = Math.min(awayWinProb, 10);
                } else if (mostLikelyOutcome === "DRAW") {
                    // Beraberlik ise, olasılığı en az %50 olmalı
                    drawProb = Math.max(drawProb, 50);
                    homeWinProb = Math.min(homeWinProb, 30);
                    awayWinProb = Math.min(awayWinProb, 20);
                } else if (mostLikelyOutcome === "AWAY_WIN") {
                    // Deplasman kazanırsa, olasılığı en az %60 olmalı
                    awayWinProb = Math.max(awayWinProb, 60);
                    drawProb = Math.min(drawProb, 30);
                    homeWinProb = Math.min(homeWinProb, 10);
                }
            }
        }
    }
    
    // 2. ADIM: Backend'ten gelen most_likely_outcome'u yalnızca kesin skor yoksa kullan
    if (!mostLikelyOutcome && predictions.most_likely_outcome) {
        // Backend'den gelen değeri kullan, ama yine de hızlı bir kontrol yap
        mostLikelyOutcome = predictions.most_likely_outcome;
        console.log("Backend'den gelen sonuç kullanılıyor:", mostLikelyOutcome);
        
        // Yine de kesin skoru kontrol et - sonradan ayarlanmış olabilir
        if (exactScore && exactScore.includes('-')) {
            const scoreParts = exactScore.split('-');
            if (scoreParts.length === 2) {
                const homeGoals = parseInt(scoreParts[0]);
                const awayGoals = parseInt(scoreParts[1]);
                
                // Final kontrol: Skor eşit ama sonuç beraberlik değilse düzelt
                if (!isNaN(homeGoals) && !isNaN(awayGoals) && homeGoals === awayGoals && mostLikelyOutcome !== "DRAW") {
                    console.log("TUTARSIZLIK DÜZELTİLİYOR: Eşit skor", exactScore, "için sonuç", mostLikelyOutcome, "DRAW olarak değiştirildi");
                    mostLikelyOutcome = "DRAW";
                }
            }
        }
    }
    
    // 3. ADIM: Eğer backend'den değer yoksa, olasılıklara göre belirle
    if (!mostLikelyOutcome) {
        const maxProb = Math.max(homeWinProb, drawProb, awayWinProb);
        
        if (maxProb === homeWinProb) {
            mostLikelyOutcome = "HOME_WIN";
        } else if (maxProb === drawProb) {
            mostLikelyOutcome = "DRAW";
        } else if (maxProb === awayWinProb) {
            mostLikelyOutcome = "AWAY_WIN";
        } else {
            // API'den gelen değeri kullan - ancak varsayılan beraberlik OLMASIN
            // En yüksek olasılıklı sonucu bul
            if (homeWinProb >= drawProb && homeWinProb >= awayWinProb) {
                mostLikelyOutcome = "HOME_WIN";
            } else if (awayWinProb >= homeWinProb && awayWinProb >= drawProb) {
                mostLikelyOutcome = "AWAY_WIN";
            } else {
                mostLikelyOutcome = "DRAW";
            }
        }
        console.log("Olasılıklara göre sonuç:", mostLikelyOutcome, "Olasılıklar:", { homeWinProb, drawProb, awayWinProb });
    }

    // Sonuç olasılıkları grafiği
    updateProbabilityBars(homeWinProb, drawProb, awayWinProb);

    // Beklenen goller - filtrelenmiş olanları veya orijinali kullan
    let homeExpectedGoals = predictions.expected_goals.home;
    let awayExpectedGoals = predictions.expected_goals.away;
    
    if (filteredPredictions.avg_expected_goals) {
        homeExpectedGoals = filteredPredictions.avg_expected_goals.home || homeExpectedGoals;
        awayExpectedGoals = filteredPredictions.avg_expected_goals.away || awayExpectedGoals;
    } else if (filteredPredictions.expected_goals) {
        homeExpectedGoals = filteredPredictions.expected_goals.home || homeExpectedGoals;
        awayExpectedGoals = filteredPredictions.expected_goals.away || awayExpectedGoals;
    }
    
    $('#expectedHomeGoals').text(homeExpectedGoals);
    $('#expectedAwayGoals').text(awayExpectedGoals);

    // Skor ve tahmin sonuçlarını göster
    $('#predictedScore').text(exactScore);
    
    // Kritik Düzeltme: Skor ve sonuç tutarsızlığını düzelt
    if (exactScore && exactScore.includes('-') && mostLikelyOutcome) {
        // Skordan goller
        const scoreParts = exactScore.split('-');
        const homeGoals = parseInt(scoreParts[0]);
        const awayGoals = parseInt(scoreParts[1]);
        
        // Skordan beklenen sonuç
        let expectedOutcome = '';
        if (homeGoals > awayGoals) expectedOutcome = 'HOME_WIN';
        else if (homeGoals < awayGoals) expectedOutcome = 'AWAY_WIN';
        else expectedOutcome = 'DRAW';
        
        // Tutarsızlık varsa konsola uyarı bas
        if (expectedOutcome !== mostLikelyOutcome) {
            console.warn('TUTARSIZLIK TESPİT EDİLDİ!');
            console.warn(`Skor ${exactScore} için beklenen sonuç: ${expectedOutcome}, mevcut sonuç: ${mostLikelyOutcome}`);
            
            // Tutarlı olması için mostLikelyOutcome değerini skordan belirlenen sonuç ile değiştir
            mostLikelyOutcome = expectedOutcome;
            console.warn(`Sonuç düzeltildi: ${mostLikelyOutcome}`);
            
            // Olasılıkları da skorla uyumlu hale getir
            if (mostLikelyOutcome === 'HOME_WIN') {
                homeWinProb = Math.max(homeWinProb, 60);
                drawProb = Math.min(drawProb, 30);
                awayWinProb = Math.min(awayWinProb, 10);
            } else if (mostLikelyOutcome === 'DRAW') {
                drawProb = Math.max(drawProb, 50);
                homeWinProb = Math.min(homeWinProb, 30);
                awayWinProb = Math.min(awayWinProb, 20);
            } else {
                awayWinProb = Math.max(awayWinProb, 60);
                drawProb = Math.min(drawProb, 30);
                homeWinProb = Math.min(homeWinProb, 10);
            }
        }
    }
    
    // Debug bilgileri
    console.log("*** Tahmin Detayları ***");
    console.log("Mostlikelyoutcome:", mostLikelyOutcome);
    console.log("Exact score:", exactScore);
    console.log("Olasılıklar (Ev/Ber/Dep):", homeWinProb, drawProb, awayWinProb);

    // Maç sonucu tahmini
    let outcomeText = "";
    if (mostLikelyOutcome === "HOME_WIN") {
        outcomeText = `${data.home_team.name} kazanır (${Math.round(homeWinProb)}%)`;
    } else if (mostLikelyOutcome === "DRAW") {
        outcomeText = `Beraberlik (${Math.round(drawProb)}%)`;
    } else {
        outcomeText = `${data.away_team.name} kazanır (${Math.round(awayWinProb)}%)`;
    }
    $('#outcomePredicton').text(outcomeText);

    // Bahis tahminlerini güncelle
    updateBettingPredictions(predictions.betting_predictions);

    // H2H verilerini güncelle
    updateH2HData(data.head_to_head);

    // Takım form bilgilerini güncelle
    updateTeamForm(data.home_team, 'home');
    updateTeamForm(data.away_team, 'away');

    // Tahmin açıklamasını güncelle
    if (predictions.explanation) {
        $('#predictionExplanation').html(
            `<p>${predictions.explanation.exact_score}</p>
             <p>${predictions.explanation.match_result}</p>
             <p>${predictions.explanation.relative_strength}</p>
             <p>${predictions.explanation.head_to_head}</p>`
        );
    }

    // Tahmin güvenilirliği
    updateConfidenceIndicator(predictions.confidence);

    // Tahmin zamanını göster
    const predictionDate = new Date(data.date_predicted);
    $('#predictionTimestamp').text(predictionDate.toLocaleString());
    
    // İY/MS Tahminini kaldırıldı - Bu tahmin sadece sürpriz butonunda gösterilecek
    // İY/MS tahminini normal tahmin ekranında göstermeye gerek yok
    // displayHTFTPrediction(
    //     data.home_team.id,
    //     data.away_team.id,
    //     data.home_team.name,
    //     data.away_team.name
    // );

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

function updateBettingPredictions(bettingPredictions) {
    console.log("Mevcut bahis tahminleri bölümü güncelleniyor");
    console.log("BETTING PREDICTIONS:", bettingPredictions);
    
    // Tahmin değerlerini Türkçe formata çevir, API'den gelen YES/NO değerlerini güncelleyelim
    if (bettingPredictions.over_2_5_goals && bettingPredictions.over_2_5_goals.prediction) {
        if (bettingPredictions.over_2_5_goals.prediction === 'YES') {
            bettingPredictions.over_2_5_goals.display_value = '2.5 ÜST';
        } else if (bettingPredictions.over_2_5_goals.prediction === 'NO') {
            bettingPredictions.over_2_5_goals.display_value = '2.5 ALT';
        }
    }
    
    // TAHMİN TUTARLILIĞI İÇİN ÖNEMLİ: TAHMİN SKOR DEĞERİNİ DOĞRUDAN BACKEND'DEN AL
    // Şimdi iki farklı yerde olabildiği için her ikisini de kontrol ediyoruz:
    // 1. exact_score key olarak doğrudan API'dan geliyor olabilir
    // 2. Backend tutarsızlık düzeltince prediction içinde olabilir
    let exactScoreFromAPI = null;
    
    // DOM'daki mevcut skoru da kontrol et (sonradan güncellenmiş olabilir)
    const currentDomScore = $('#predictedScore').text().trim();
    console.log("DOM'daki mevcut skor:", currentDomScore);
    
    // Eğer debug bilgisi varsa, o en doğru kaynaktır (tutarsızlık düzeltilmiş son değer)
    if (window.predictionData && window.predictionData.predictions && window.predictionData.predictions.debug_final_check) {
        const debugInfo = window.predictionData.predictions.debug_final_check;
        
        // Tutarsızlık düzeltmesi var mı kontrol et
        if (debugInfo.includes('Tutarsızlık giderildi')) {
            console.warn("Backend'de tutarsızlık düzeltmesi tespit edildi. Bahis tahminleri güncelleniyor...");
            
            // 2.5 ÜST/ALT tahmini mutlaka YES olmalı
            if (bettingPredictions.over_2_5_goals) {
                bettingPredictions.over_2_5_goals.prediction = 'YES';
                bettingPredictions.over_2_5_goals.probability = 0.9;
                console.log("Tutarsızlık düzeltmesi nedeniyle 2.5 ÜST/ALT tahmini YES olarak güncellendi");
            }
        }
        
        if (debugInfo.includes('Skor=')) {
            // "Final kontrol: Skor=2-1, Sonuç=HOME_WIN" formatından skoru çıkar
            const scoreMatch = debugInfo.match(/Skor=([0-9]+-[0-9]+)/);
            if (scoreMatch && scoreMatch[1]) {
                exactScoreFromAPI = scoreMatch[1];
                console.log("DEBUG BİLGİSİNDEN ALINAN KESIN SKOR:", exactScoreFromAPI);
            }
        }
    }
    
    // Backend'den gelen skorun çeşitli kaynaklarını kontrol et
    if (!exactScoreFromAPI && typeof bettingPredictions.exact_score === 'string') {
        // Doğrudan string olarak gelmiş
        exactScoreFromAPI = bettingPredictions.exact_score;
        console.log("API'den gelen kesin skor (string):", exactScoreFromAPI);
    } else if (bettingPredictions.exact_score && bettingPredictions.exact_score.prediction) {
        // Object formatında gelmiş
        exactScoreFromAPI = bettingPredictions.exact_score.prediction;
        console.log("API'den gelen kesin skor (object.prediction):", exactScoreFromAPI);
    }
    
    // DOM'a kesin skoru yazıyoruz, önce mevcut DOM değerini değiştiriyoruz!
    if (exactScoreFromAPI) {
        console.log("Tahmini skor DOM'a yazılıyor:", exactScoreFromAPI);
        $('#predictedScore').text(exactScoreFromAPI);
    }
    
    // Skordan toplam golleri hesapla
    let totalGoals = 0;
    let homeGoals = 0; 
    let awayGoals = 0;
    let validScore = false;
    
    if (exactScoreFromAPI && exactScoreFromAPI.includes('-')) {
        try {
            [homeGoals, awayGoals] = exactScoreFromAPI.split('-').map(Number);
            totalGoals = homeGoals + awayGoals;
            validScore = true;
            console.log("API skorundan hesaplanan toplam gol:", totalGoals, "ev:", homeGoals, "deplasman:", awayGoals);
        } catch(e) {
            console.error("Skor ayrıştırma hatası:", e);
        }
    }
    
    // Eğer orijinal tahminler dışında filtrelenmiş tahminler varsa kullan
    const filteredPredictions = window.predictionData?.predictions?.filtered_predictions || {};
    
    // KG Var/Yok - API ve skor uyumluluk kontrolü
    let bttsValue = 'NO'; // Varsayılan değer
    
    // Önce API değerini al (öncelikli)
    if (bettingPredictions.both_teams_to_score && bettingPredictions.both_teams_to_score.prediction) {
        bttsValue = bettingPredictions.both_teams_to_score.prediction;
        console.log("API'den alınan KG değeri:", bttsValue);
    }
    
    // Skor 0-0 ise veya taraflardan biri 0 golse, KG VAR OLAMAZ (KRİTİK DÜZELTME)
    if (validScore) {
        if ((homeGoals === 0 || awayGoals === 0) && bttsValue === 'YES') {
            console.log("KRİTİK DÜZELTME: Skor", scorePrediction, "- en az bir takım gol atmadı, KG VAR OLAMAZ");
            bttsValue = 'NO';
            // Olasılığı da yüksek güvenle güncelle
            bettingPredictions.both_teams_to_score = bettingPredictions.both_teams_to_score || {};
            bettingPredictions.both_teams_to_score.probability = 0.9; // %90 güven düzeyi
        } else if (homeGoals > 0 && awayGoals > 0 && bttsValue === 'NO') {
            console.log("KRİTİK DÜZELTME: Skor", scorePrediction, "- her iki takım gol attı, KG YOK OLAMAZ");
            bttsValue = 'YES';
            // Olasılığı da yüksek güvenle güncelle
            bettingPredictions.both_teams_to_score = bettingPredictions.both_teams_to_score || {};
            bettingPredictions.both_teams_to_score.probability = 0.9; // %90 güven düzeyi
        }
    }
    }
    
    // Olasılığı belirle
    let bttsProb = (bettingPredictions.both_teams_to_score && bettingPredictions.both_teams_to_score.probability) 
        ? bettingPredictions.both_teams_to_score.probability / 100 
        : (bttsValue === 'YES' ? 0.9 : 0.1);
    
    $('#bttsValue').text(bttsValue === 'YES' ? 'EVET' : 'HAYIR');
    $('#bttsProb').text(Math.round(bttsProb * 100) + '%');
    $('#bttsIcon').html(bttsValue === 'YES' ? '<i class="fas fa-check-circle text-success"></i>' : '<i class="fas fa-times-circle text-danger"></i>');

    // TUTARLILIK İÇİN KRİTİK: 2.5 ÜST/ALT DEĞERİNİ SKORDAN BELİRLE
    // Toplam goller 2.5'tan fazla mı?
    let totalGoalsOver25 = totalGoals > 2;
    
    // EKSTRA DEBUG LOGLARı EKLE
    console.log("FRONTEND SKORS DEBUG - İLK DEĞERLER:", {
        exactScoreFromAPI,
        totalGoals,
        homeGoals,
        awayGoals,
        totalGoalsOver25,
        validScore
    });
    
    // Skor bilgisine göre ÜST/ALT değeri belirleme - kesin skordan hesaplayıp ZORLA değiştir
    let over25Value = '2.5 ALT';
    let over25Prob = 0.1;
    
    // Eğer geçerli bir skor varsa, bunu kullan
    if (validScore) {
        over25Value = totalGoalsOver25 ? '2.5 ÜST' : '2.5 ALT';
        over25Prob = totalGoalsOver25 ? 0.9 : 0.1; // ÜST ise %90, ALT ise %10 olasılık
        console.log(`Skordan belirlenen 2.5 ÜST/ALT: ${over25Value} (${over25Prob * 100}%), toplam gol: ${totalGoals}`);
    } 
    // Skor geçerli değilse, backend'den gelen değeri kontrol et
    else if (bettingPredictions.over_2_5_goals) {
        let rawValue = bettingPredictions.over_2_5_goals.prediction || 'UNDER_2_5_GOALS';
        over25Prob = bettingPredictions.over_2_5_goals.probability || 0.5;
        
        // Değeri Türkçe formata dönüştür
        over25Value = (rawValue === 'YES') ? '2.5 ÜST' : '2.5 ALT';
        
        // API'den "YES" geldiyse bettingPredictions nesnesini de güncelleyelim
        if (rawValue === 'YES' && bettingPredictions.over_2_5_goals) {
            bettingPredictions.over_2_5_goals.display_value = '2.5 ÜST';
        } else if (rawValue === 'NO' && bettingPredictions.over_2_5_goals) {
            bettingPredictions.over_2_5_goals.display_value = '2.5 ALT';
        }
        
        console.log(`API'den alınan 2.5 ÜST/ALT: ${rawValue} -> ${over25Value} (${over25Prob * 100}%)`);
    }
    
    // ÇOK ÖZEL BİR KOD: TOPLAM GOLLER 3 OLAN SKORLAR (1-2, 2-1, 0-3, 3-0) İÇİN MUTLAKA 2.5 ÜST YAZMASINI SAĞLA
    // Bu kod ile backend'den 1-2 skoru gönderildiğinde kesinlikle 2.5 ÜST gösterdiğimizden emin olacağız
    
    // ÖNEMLİ: HTML'deki skoru oku - bu, backend'in son düzelttiği değerdir
    const domScoreText = $('#predictedScore').text().trim();
    let scoreValidInDOM = false;
    let domHomeGoals = 0; 
    let domAwayGoals = 0;
    let domTotalGoals = 0;
    
    // DOM'daki skoru kontrol et ve ayrıştır
    if (domScoreText && domScoreText.includes('-')) {
        try {
            const scoreParts = domScoreText.split('-');
            if (scoreParts.length === 2) {
                domHomeGoals = parseInt(scoreParts[0]);
                domAwayGoals = parseInt(scoreParts[1]);
                if (!isNaN(domHomeGoals) && !isNaN(domAwayGoals)) {
                    domTotalGoals = domHomeGoals + domAwayGoals;
                    scoreValidInDOM = true;
                    console.log("DOM'dan okunan skor:", domScoreText, "toplam goller:", domTotalGoals);
                }
            }
        } catch(e) {
            console.error("DOM skor ayrıştırma hatası:", e);
        }
    }
    
    // API'dan gelen skoru kontrol et
    const anyValidScore = exactScoreFromAPI || domScoreText;
    
    // *** ZORLA TOPLAM GOLLER 3 OLAN SKORLAR İÇİN 2.5 ÜST GÖSTER ***
    if (scoreValidInDOM && domTotalGoals === 3) {
        // Toplam goller 3 ise MUTLAKA 2.5 ÜST göster!
        totalGoalsOver25 = true;
        over25Value = '2.5 ÜST';
        over25Prob = 0.9;
        
        console.log("ZORUNLU GEÇERSİZ KILMA: DOM'da toplam gol 3 olduğu için 2.5 ÜST olarak zorlandı", {
            domScoreText,
            domTotalGoals,
            totalGoalsOver25
        });
        
        // Direkt olarak DOM'daki elementleri de güncelle (ekstra önlem)
        $('#over25Value').text('ÜST');
        $('#over25Prob').text('90%');
        $('#over25Icon').html('<i class="fas fa-arrow-up text-primary"></i>');
        
        // Engelleyiciyi durdur - bu noktadan sonra değişikliğe izin verme
        return;
    }
    
    // Özel skor kontrolü - 1-2, 2-1, 0-3, 3-0 skorları için mutlaka 2.5 ÜST göster
    if (anyValidScore === '1-2' || anyValidScore === '2-1' || anyValidScore === '0-3' || anyValidScore === '3-0' || 
        exactScoreFromAPI === '1-2' || exactScoreFromAPI === '2-1' || exactScoreFromAPI === '0-3' || exactScoreFromAPI === '3-0') {
        // KESINLIKLE 2.5 UST olarak zorla
        totalGoalsOver25 = true;
        over25Value = '2.5 ÜST';
        over25Prob = 0.9;
        
        console.log("ZORUNLU GEÇERSİZ KILMA: Skor 1-2, 2-1, 0-3 veya 3-0 olduğu için 2.5 ÜST olarak zorlandı", {
            exactScoreFromAPI,
            domScoreText,
            anyValidScore,
            totalGoalsOver25
        });
        
        // Direkt olarak DOM'daki elementleri de güncelle (ekstra önlem)
        $('#over25Value').text('ÜST');
        $('#over25Prob').text('90%');
        $('#over25Icon').html('<i class="fas fa-arrow-up text-primary"></i>');
        
        // Backend'den gelen tutarsızlık düzeltmesi tespit edildiyse
        if (data.predictions && data.predictions.debug_final_check && 
            data.predictions.debug_final_check.includes('Tutarsızlık giderildi')) {
            console.warn("BACKEND TUTARSIZLIK DÜZELTMESİ SAPTANDI, UYGULAMA DEVAM ETMEDEN ÖNCE ZORLA 2.5 ÜST OLARAK AYARLANDI.");
            
            // Engelleyiciyi durdur - bu noktadan sonra değişikliğe izin verme
            return;
        }
        
        // Direkt olarak DOM'daki elementleri de güncelle (ekstra önlem)
        $('#over25Value').text('ÜST');
        $('#over25Prob').text('90%');
        $('#over25Icon').html('<i class="fas fa-arrow-up text-primary"></i>');
        
        // Engelleyiciyi durdur - bu noktadan sonra değişikliğe izin verme
        return;
    }
    
    console.log("2.5 ÜST/ALT değeri:", {
        exactScoreFromAPI,
        totalGoals,
        totalGoalsOver25,
        final_over25Value: over25Value,
        final_over25Prob: over25Prob
    });
    
    // Tutarlılık için ÜST/ALT değerini doğrudan skor üzerinden hesapla
    // Eğer geçerli bir skor elde edememişsek, DOM'daki skor değerini kontrol et
    if (!validScore) {
        const domScoreText = $('#predictedScore').text();
        if (domScoreText && domScoreText.includes('-')) {
            try {
                const [domHomeGoals, domAwayGoals] = domScoreText.split('-').map(Number);
                const domTotalGoals = domHomeGoals + domAwayGoals;
                // Toplam gol 2.5'tan büyük mü?
                totalGoalsOver25 = domTotalGoals > 2;
                console.log("DOM'daki skor üzerinden hesaplanan toplam gol:", domTotalGoals, "totalGoalsOver25:", totalGoalsOver25);
            } catch(e) {
                console.error("DOM skor ayrıştırma hatası:", e);
            }
        }
    }
    
    console.log("TOTAL GOALS > 2.5:", totalGoalsOver25);
    
    // ÜST/ALT değerini belirle ve göster (mutlaka son hesaplanan totalGoalsOver25 değerini kullan)
    // UI'da gösterilen değerin Türkçe olmasını sağlar: "YES" -> "ÜST", "NO" -> "ALT"
    $('#over25Value').text(totalGoalsOver25 ? 'ÜST' : 'ALT');
    $('#over25Prob').text(totalGoalsOver25 ? '90%' : '10%');
    $('#over25Icon').html(totalGoalsOver25 ? 
        '<i class="fas fa-arrow-up text-primary"></i>' : 
        '<i class="fas fa-arrow-down text-secondary"></i>'
    );

    // 3.5 Üst/Alt - Filtrelenmiş veya orijinal
    // 3.5 Üst/Alt - Filtrelenmiş veya orijinal
    let over35RawValue = "NO"; // Varsayılan değer
    let over35Prob = 0.1; // Varsayılan olasılık

    // API'den gelen değerleri kontrol et
    if (bettingPredictions.over_3_5_goals && bettingPredictions.over_3_5_goals.prediction) {
        over35RawValue = bettingPredictions.over_3_5_goals.prediction;
        over35Prob = bettingPredictions.over_3_5_goals.probability;
    }
    
    // Değeri Türkçe formata dönüştür
    let over35Value = over35RawValue === 'YES' ? '3.5 ÜST' : '3.5 ALT';
    
    if (filteredPredictions.over_under_3_5 && filteredPredictions.over_under_3_5.prediction) {
        over35Value = filteredPredictions.over_under_3_5.prediction.includes('OVER') ? '3.5 ÜST' : '3.5 ALT';
        over35Prob = (filteredPredictions.over_under_3_5.probability * 100).toFixed(2);
    }
    
    // TUTARLILIK: Kesin skordan 3.5 Alt/Üst değerini belirle
    if (validScore) {
        const totalGoalsOver35 = totalGoals > 3;
        
        // Toplam gol sayısı 3.5'tan fazlaysa (4+) kesinlikle ÜST, değilse ALT olmalı
        const correctOver35Value = totalGoalsOver35 ? '3.5 ÜST' : '3.5 ALT';
        
        // Eğer mevcut değer doğru değilse, düzelt
        if (over35Value !== correctOver35Value) {
            console.log(`3.5 Alt/Üst düzeltme: ${over35Value} -> ${correctOver35Value} (Toplam gol: ${totalGoals})`);
            over35Value = correctOver35Value;
            over35Prob = totalGoalsOver35 ? 0.9 : 0.9; // Her iki durum için de yüksek güven
        }
    }
    
    if (filteredPredictions.over_under_3_5 && filteredPredictions.over_under_3_5.prediction) {
        over35Value = filteredPredictions.over_under_3_5.prediction.includes('OVER') ? '3.5 ÜST' : '3.5 ALT';
        over35Prob = (filteredPredictions.over_under_3_5.probability * 100).toFixed(2);
    }
    
    // TUTARLILIK: Kesin skordan 3.5 Alt/Üst değerini belirle
    if (validScore) {
        const totalGoalsOver35 = totalGoals > 3;
        
        // Toplam gol sayısı 3.5'tan fazlaysa (4+) kesinlikle ÜST, değilse ALT olmalı
        const correctOver35Value = totalGoalsOver35 ? '3.5 ÜST' : '3.5 ALT';
        
        // Eğer mevcut değer doğru değilse, düzelt
        if (over35Value !== correctOver35Value) {
            console.log(`3.5 Alt/Üst düzeltme: ${over35Value} -> ${correctOver35Value} (Toplam gol: ${totalGoals})`);
            over35Value = correctOver35Value;
            over35Prob = totalGoalsOver35 ? 0.9 : 0.9; // Her iki durum için de yüksek güven
        }
    }
    
    // TUTARLILIK: DOM'daki kesin skordan 3.5 Alt/Üst değerini belirle
    // Bu kod, kesin skordan yola çıkarak 3.5 Alt/Üst tahminini doğru şekilde günceller
    if (validScore || scoreValidInDOM) {
        const finalTotalGoals = validScore ? totalGoals : domTotalGoals;
        const totalGoalsOver35 = finalTotalGoals > 3;
        
        // Toplam gol sayısı 3.5'tan fazlaysa (4+) kesinlikle ÜST, değilse ALT olmalı
        const correctOver35Value = totalGoalsOver35 ? '3.5 ÜST' : '3.5 ALT';
        
        // Eğer mevcut değer doğru değilse, düzelt
        if (over35Value !== correctOver35Value) {
            console.log(`3.5 Alt/Üst düzeltme: ${over35Value} -> ${correctOver35Value} (Toplam gol: ${finalTotalGoals})`);
            over35Value = correctOver35Value;
            over35Prob = totalGoalsOver35 ? 0.9 : 0.9; // Her iki durum için de yüksek güven
        }
    }
    
    // Ekranda sadece ÜST veya ALT göster
    $('#over35Value').text(over35Value.includes('ÜST') ? 'ÜST' : 'ALT');
    $('#over35Prob').text(over35Prob + '%');
    $('#over35Icon').html(over35Value.includes('ÜST') ? '<i class="fas fa-arrow-up text-primary"></i>' : '<i class="fas fa-arrow-down text-secondary"></i>');

    // İlk Gol ve Zamanı - Bu bölümü kaldırdık, kullanıcıya gösterilmeyecek
    // İlk gol ile ilgili bilgileri kullanıcıya göstermiyoruz
    // ve bu tahmin en yüksek olasılıklı tahminde de çıkmayacak
    $('#firstGoalValue').text("-");
    
    // İlk Yarı/Maç Sonu - Burayı işliyoruz ama ayrı bir bölümde gösteriyoruz
    const htftValue = bettingPredictions.half_time_full_time.prediction;
    const htftParts = htftValue.split('/');

    let htText = "";
    if (htftParts[0] === "HOME_WIN") {
        htText = "Ev Sahibi";
    } else if (htftParts[0] === "DRAW") {
        htText = "Beraberlik";
    } else {
        htText = "Deplasman";
    }

    let ftText = "";
    if (htftParts[1] === "HOME_WIN") {
        ftText = "Ev Sahibi";
    } else if (htftParts[1] === "DRAW") {
        ftText = "Beraberlik";
    } else {
        ftText = "Deplasman";
    }

    // Normal tahmin bölümünde kullanmıyoruz, ayrı bölümde göstereceğiz
}

// ÖZEL FIX FONKSIYONU: 1-2 ve benzeri skorlar için 2.5 ÜST kontrolü
function fixOver25ForScore(score) {
    // Eğer skor formatı geçerli değilse bir şey yapma
    if (!score || !score.includes('-')) return;
    
    try {
        // Skoru parçala ve toplam golleri hesapla
        const [homeGoals, awayGoals] = score.split('-').map(Number);
        const totalGoals = homeGoals + awayGoals;
        
        console.log("ÖZEL SKOR KONTROL:", {score, totalGoals});
        
        // Eğer toplam goller 3 ise, 2.5 ÜST göster
        if (totalGoals === 3) {
            console.log("ÖZEL FIX UYGULANACAK: 2.5 ÜST", {score, totalGoals});
            
            // DOM elementlerini doğrudan güncelle
            $('#over25Value').text('ÜST');
            $('#over25Prob').text('90%');
            $('#over25Icon').html('<i class="fas fa-arrow-up text-primary"></i>');
            
            // Önemli bir uyarı bas
            console.warn("SKOR " + score + " (TOPLAM: " + totalGoals + ") İÇİN 2.5 ÜST ZORLA UYGULANDI");
        }
    } catch (e) {
        console.error("Skor ayrıştırma hatası:", e);
    }
}

// Yeni Fonksiyon: İY/MS (İlk Yarı/Maç Sonu) tahminlerini göster
function displayHTFTPrediction(homeTeamId, awayTeamId, homeTeamName, awayTeamName) {
    console.log("İY/MS Tahmini isteniyor: ", homeTeamId, awayTeamId);
    
    // Eğer tahmin bölümü daha önce oluşturulmadıysa oluştur
    if ($('#htftPredictionSection').length === 0) {
        // İY/MS tahmin bölümünü ekle
        const htftSection = `
            <div id="htftPredictionSection" class="mt-4 border rounded p-3 bg-light">
                <h5 class="mb-3"><i class="fas fa-chart-pie"></i> İlk Yarı / Maç Sonu Tahmini</h5>
                <div id="htftLoading" class="text-center">
                    <div class="spinner-border text-primary" role="status">
                        <span class="sr-only">Yükleniyor...</span>
                    </div>
                    <p>İY/MS Tahmini Yükleniyor...</p>
                </div>
                <div id="htftContent" style="display:none;">
                    <div class="row">
                        <div class="col-md-7">
                            <div class="card mb-3">
                                <div class="card-header bg-primary text-white">
                                    <h6 class="mb-0">En Olası İY/MS Kombinasyonları</h6>
                                </div>
                                <div class="card-body">
                                    <div id="htftTopPredictions"></div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-5">
                            <div class="card mb-3">
                                <div class="card-header bg-success text-white">
                                    <h6 class="mb-0">En Yüksek Olasılıklı Tahmin</h6>
                                </div>
                                <div class="card-body text-center">
                                    <h3 id="htftMainPrediction">1/1</h3>
                                    <p id="htftPredictionDesc">İlk Yarı: Ev Sahibi Önde / Maç Sonu: Ev Sahibi Kazanır</p>
                                    <div class="badge bg-info text-white" id="htftProbability">75%</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="mt-3">
                        <div class="alert alert-info">
                            <p class="mb-0"><i class="fas fa-info-circle"></i> Bu tahminler, her iki takımın ilk ve ikinci yarı performanslarına dayalı olarak hesaplanmıştır. Tahminler, maçın ana tahmininden ayrıdır ve sadece İY/MS kombinasyonlarını gösterir.</p>
                        </div>
                    </div>
                </div>
                <div id="htftError" class="alert alert-danger" style="display:none;"></div>
            </div>
        `;
        
        // Tahmin bölümünün sonuna ekle
        $('#predictionModal .modal-body').append(htftSection);
    }
    
    // Yükleniyor göster, içeriği gizle
    $('#htftLoading').show();
    $('#htftContent').hide();
    $('#htftError').hide();
    
    // İY/MS tahmini API çağrısı
    fetch(`/api/v3/htft-prediction/${homeTeamId}/${awayTeamId}?home_name=${encodeURIComponent(homeTeamName)}&away_name=${encodeURIComponent(awayTeamName)}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`İY/MS API isteği başarısız (${response.status})`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                // Hata durumunda
                $('#htftLoading').hide();
                $('#htftError').text(`Hata: ${data.error}`).show();
            } else {
                updateHTFTPredictionUI(data);
            }
        })
        .catch(error => {
            console.error('İY/MS tahmini alınırken hata:', error);
            $('#htftLoading').hide();
            $('#htftError').text(`İY/MS tahmini alınırken hata: ${error.message}`).show();
        });
}

// İY/MS tahmin arayüzünü güncelle
function updateHTFTPredictionUI(data) {
    console.log("İY/MS tahmini güncelleniyor:", data);
    
    // Yükleniyor gizle, içeriği göster
    $('#htftLoading').hide();
    $('#htftContent').show();
    
    // Tahmin bilgileri
    const htftPrediction = data.htft_prediction;
    
    // Hata kontrolü
    if (!htftPrediction || htftPrediction.error) {
        $('#htftError').text(`Hata: ${htftPrediction?.error || 'İY/MS tahmini verisi bulunamadı'}`).show();
        return;
    }
    
    // Ana tahmini göster
    const mainPrediction = htftPrediction.prediction || '?/?';
    $('#htftMainPrediction').text(mainPrediction);
    
    // İY/MS açıklamasını oluştur
    const htftParts = mainPrediction.split('/');
    let htftDesc = '';
    
    // İlk yarı açıklaması
    if (htftParts[0] === '1') {
        htftDesc += 'İlk Yarı: Ev Sahibi Önde';
    } else if (htftParts[0] === 'X') {
        htftDesc += 'İlk Yarı: Berabere';
    } else if (htftParts[0] === '2') {
        htftDesc += 'İlk Yarı: Deplasman Önde';
    }
    
    htftDesc += ' / ';
    
    // Maç sonu açıklaması
    if (htftParts[1] === '1') {
        htftDesc += 'Maç Sonu: Ev Sahibi Kazanır';
    } else if (htftParts[1] === 'X') {
        htftDesc += 'Maç Sonu: Beraberlik';
    } else if (htftParts[1] === '2') {
        htftDesc += 'Maç Sonu: Deplasman Kazanır';
    }
    
    $('#htftPredictionDesc').text(htftDesc);
    
    // Olasılık değerini göster
    const probability = htftPrediction.all_probabilities?.[mainPrediction] || '?';
    $('#htftProbability').text(`Olasılık: %${probability}`);
    
    // En olası İY/MS kombinasyonlarını göster
    let topPredictionsHtml = '<div class="list-group">';
    
    if (htftPrediction.top_predictions && htftPrediction.top_predictions.length > 0) {
        htftPrediction.top_predictions.forEach(pred => {
            const code = pred.code;
            const prob = pred.probability;
            const isActive = (code === mainPrediction) ? 'active bg-primary' : '';
            
            // Kod açıklamasını oluştur
            const codeParts = code.split('/');
            let codeDesc = '';
            
            // İlk yarı açıklaması
            if (codeParts[0] === '1') {
                codeDesc += 'İY: Ev Sahibi Önde';
            } else if (codeParts[0] === 'X') {
                codeDesc += 'İY: Beraberlik';
            } else if (codeParts[0] === '2') {
                codeDesc += 'İY: Deplasman Önde';
            }
            
            codeDesc += ' / ';
            
            // Maç sonu açıklaması
            if (codeParts[1] === '1') {
                codeDesc += 'MS: Ev Sahibi';
            } else if (codeParts[1] === 'X') {
                codeDesc += 'MS: Beraberlik';
            } else if (codeParts[1] === '2') {
                codeDesc += 'MS: Deplasman';
            }
            
            topPredictionsHtml += `
                <a href="#" class="list-group-item list-group-item-action ${isActive}">
                    <div class="d-flex w-100 justify-content-between">
                        <h6 class="mb-1">${code}</h6>
                        <span class="badge ${isActive ? 'bg-light text-primary' : 'bg-primary text-white'} rounded-pill">%${prob}</span>
                    </div>
                    <small class="text-muted">${codeDesc}</small>
                </a>
            `;
        });
    } else {
        topPredictionsHtml += '<div class="alert alert-warning">İY/MS tahmin verisi bulunamadı.</div>';
    }
    
    topPredictionsHtml += '</div>';
    $('#htftTopPredictions').html(topPredictionsHtml);
    
    // Özel içgörü varsa göster
    if (data.special_insight) {
        const specialInsight = data.special_insight;
        const insightHtml = `
            <div class="alert alert-warning mt-3">
                <h6><i class="fas fa-lightbulb"></i> ${specialInsight.title}</h6>
                <p>${specialInsight.description}</p>
                <small class="text-muted">Olasılık: %${specialInsight.probability}</small>
            </div>
        `;
        
        // Eğer özel içgörü bölümü yoksa oluştur, varsa güncelle
        if ($('#htftSpecialInsight').length === 0) {
            $('#htftContent').append(`<div id="htftSpecialInsight">${insightHtml}</div>`);
        } else {
            $('#htftSpecialInsight').html(insightHtml);
        }
    } else {
        // Özel içgörü yoksa, bölümü gizle
        $('#htftSpecialInsight').remove();
    }
}

function updateH2HData(h2hData) {
    if (!h2hData || h2hData.total_matches === 0) {
        $('#h2hSection').html('<div class="alert alert-info">Bu iki takım arasında daha önce resmi maç oynanmamış.</div>');
        return;
    }

    // H2H istatistikleri
    $('#homeWins').text(h2hData.home_wins);
    $('#draws').text(h2hData.draws);
    $('#awayWins').text(h2hData.away_wins);
    $('#totalMatches').text(h2hData.total_matches);
    $('#avgHomeGoals').text(h2hData.avg_home_goals.toFixed(2));
    $('#avgAwayGoals').text(h2hData.avg_away_goals.toFixed(2));

    // Son karşılaşmalar
    let recentMatchesHtml = '';
    if (h2hData.recent_matches && h2hData.recent_matches.length > 0) {
        h2hData.recent_matches.forEach(match => {
            const resultClass = `h2h-result-${match.result}`;
            recentMatchesHtml += `
                <div class="h2h-match">
                    <div class="d-flex justify-content-between">
                        <small class="text-muted">${match.date} - ${match.league}</small>
                        <span class="${resultClass}">${match.result === 'W' ? 'Kazandı' : match.result === 'D' ? 'Berabere' : 'Kaybetti'}</span>
                    </div>
                    <div class="d-flex justify-content-center">
                        <strong>${match.home_score} - ${match.away_score}</strong>
                    </div>
                </div>
            `;
        });
        $('#recentH2HMatches').html(recentMatchesHtml);
    } else {
        $('#recentH2HMatches').html('<div class="text-center">Son karşılaşma verisi bulunamadı.</div>');
    }
}

function updateTeamForm(teamData, team) {
    const formData = teamData.form;
    if (!formData || !formData.recent_match_data) {
        $(`#${team}FormSection`).html('<div class="alert alert-info">Form verisi bulunamadı.</div>');
        return;
    }

    // Son 5 maç formu
    let formHtml = '';
    const recentMatches = formData.recent_match_data.slice(0, 5);

    recentMatches.forEach(match => {
        const formClass = `form-${match.result}`;
        formHtml += `<span class="player-form-indicator ${formClass}">${match.result}</span>`;
    });

    $(`#${team}FormIndicators`).html(formHtml);

    // Form istatistikleri
    $(`#${team}AvgGoalsScored`).text(formData.weighted_avg_goals_scored ? formData.weighted_avg_goals_scored.toFixed(2) : '0.00');
    $(`#${team}AvgGoalsConceded`).text(formData.weighted_avg_goals_conceded ? formData.weighted_avg_goals_conceded.toFixed(2) : '0.00');

    // Son 5 maç sonuçları
    let matchesHtml = '';
    recentMatches.forEach(match => {
        const resultClass = `h2h-result-${match.result}`;
        const resultText = match.result === 'W' ? 'Kazandı' : match.result === 'D' ? 'Berabere' : 'Kaybetti';

        matchesHtml += `
            <div class="d-flex justify-content-between align-items-center border-bottom py-2">
                <div>
                    <small class="text-muted">${match.date}</small>
                    <div>${match.is_home ? 'vs ' + match.opponent + ' (E)' : '@ ' + match.opponent + ' (D)'}</div>
                </div>
                <div class="text-end">
                    <span class="${resultClass}">${resultText}</span>
                    <div>${match.goals_scored} - ${match.goals_conceded}</div>
                </div>
            </div>
        `;
    });

    $(`#${team}RecentMatches`).html(matchesHtml);
}

function updateConfidenceIndicator(confidence) {
    // Filtrelenmiş tahminlerde model uyumluluğu (harmony) varsa onu kullan
    const filteredPredictions = window.predictionData?.predictions?.filtered_predictions || {};
    let confidenceValue = confidence;
    
    // Konsensüs model uyumluluğunu kontrol et ve varsa kullan
    if (filteredPredictions.model_harmony !== undefined) {
        // Model uyumluluğu 0-1 arası, yüzdeye çevir ve güven değerini artır
        const harmonyScore = filteredPredictions.model_harmony * 100;
        
        // Model uyumu yüksekse güven değerini artır
        if (harmonyScore > 70) {
            confidenceValue = Math.min(confidence * 1.2, 100); // Max %100
        }
    }
    
    let confidenceClass = 'medium-confidence';
    let confidenceText = 'Orta';

    if (confidenceValue >= 75) {
        confidenceClass = 'high-confidence';
        confidenceText = 'Yüksek';
    } else if (confidenceValue < 50) {
        confidenceClass = 'low-confidence';
        confidenceText = 'Düşük';
    }

    $('#confidenceValue').text(confidenceValue + '%');
    $('#confidenceIndicator').removeClass().addClass('confidence-indicator ' + confidenceClass).text(confidenceText);
}

// Tahmin yenileme fonksiyonu
function refreshPrediction(homeTeamId, awayTeamId, homeTeamName, awayTeamName) {
    // Yükleniyor göster
    $('#predictionContent').hide();
    $('#predictionLoading').show();
    $('#predictionError').hide(); // Hata mesajını gizle

    // İY/MS bölümünü kaldır (varsa) - normal tahmin butonunda gösterilmeyecek
    if ($('#htftPredictionSection').length > 0) {
        $('#htftPredictionSection').remove();
    }

    // Force update parametresi true olarak gönder
    const url = `/api/predict-match/${homeTeamId}/${awayTeamId}?home_name=${encodeURIComponent(homeTeamName)}&away_name=${encodeURIComponent(awayTeamName)}&force_update=true`;
    
    console.log("API isteği yapılıyor:", url);

    $.ajax({
        url: url,
        type: 'GET',
        dataType: 'json',
        timeout: 30000, // 30 saniye timeout
        success: function(data) {
            console.log("API yanıtı alındı, durum kodu:", 200);
            console.log("Tahmin verisi alındı, uzunluk:", JSON.stringify(data).length);
            $('#predictionLoading').hide();
            $('#predictionContent').show();
            updatePredictionUI(data);
            
            // İY/MS tahmini normal tahmin butonunda gösterilmeyecek
            // Bu sadece sürpriz butonunda gösterilecek
        },
        error: function(error) {
            console.error("Tahmin alınırken hata:", error);
            $('#predictionLoading').hide();
            $('#predictionContent').show();
            let errorMessage = 'Tahmin alınırken hata: ';
            
            if (error.responseJSON && error.responseJSON.error) {
                errorMessage += error.responseJSON.error;
            } else if (error.statusText) {
                errorMessage += error.statusText;
            } else {
                errorMessage += 'Bilinmeyen hata';
            }
            
            $('#predictionError').text(errorMessage).show();
        }
    });
}