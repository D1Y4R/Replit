/**
 * İlk Yarı / Maç Sonu tahmin algoritmaları
 * Takımların ilk ve ikinci yarı performanslarına göre İY/MS olasılıklarını hesaplar
 * Monte Carlo simülasyonu ve yapay sinir ağı kullanılarak gelişmiş tahminler yapılır
 */

// Tüm olası İY/MS kombinasyonları
const HT_FT_COMBINATIONS = [
    '1/1', '1/X', '1/2', 'X/1', 'X/X', 'X/2', '2/1', '2/X', '2/2'
];

// Sigmoid aktivasyon fonksiyonu - yapay sinir ağı için
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

// Tanh aktivasyon fonksiyonu - yapay sinir ağı için
function tanh(x) {
    return Math.tanh(x);
}

// ReLU aktivasyon fonksiyonu - yapay sinir ağı için
function relu(x) {
    return Math.max(0, x);
}

// Softmax fonksiyonu - olasılık dağılımı için
function softmax(arr) {
    const maxVal = Math.max(...arr);
    const expValues = arr.map(val => Math.exp(val - maxVal));
    const sumExpValues = expValues.reduce((sum, val) => sum + val, 0);
    return expValues.map(val => val / sumExpValues);
}

/**
 * İlk yarı/maç sonu olasılıklarını hesapla
 * @param {Object} homeStats - Ev sahibi takım yarı istatistikleri
 * @param {Object} awayStats - Deplasman takımı yarı istatistikleri
 * @returns {Object} - İY/MS tahminleri ve olasılıkları
 */
function predictHalfTimeFullTime(homeStats, awayStats) {
    // İstatistik verileri yoksa hata döndür
    if (!homeStats || !homeStats.statistics || !awayStats || !awayStats.statistics) {
        console.error("İY/MS tahmini için yarı istatistikleri eksik veya null:", {homeStats, awayStats});
        return {
            error: "Yarı istatistik verileri eksik",
            predictions: {}
        };
    }

    // Takımların istatistikleri
    const homeFirst = homeStats.statistics.first_half;
    const homeSecond = homeStats.statistics.second_half;
    const awayFirst = awayStats.statistics.first_half;
    const awaySecond = awayStats.statistics.second_half;

    // Takımların toplam maç sayısı
    const homeMatches = homeStats.total_matches_analyzed;
    const awayMatches = awayStats.total_matches_analyzed;

    console.log("İY/MS Tahmin - Takım istatistikleri:", {
        ev: {
            ilk_yari: homeFirst,
            ikinci_yari: homeSecond,
            mac_sayisi: homeMatches
        },
        deplasman: {
            ilk_yari: awayFirst,
            ikinci_yari: awaySecond,
            mac_sayisi: awayMatches
        }
    });

    // İlk yarı performansları (maç başına gol)
    const homeFirstHalfAvg = homeFirst.avg_goals_per_match;
    const awayFirstHalfAvg = awayFirst.avg_goals_per_match;

    // İlk yarı gol farkı
    const firstHalfDiff = homeFirstHalfAvg - awayFirstHalfAvg;

    // İkinci yarı performansları (maç başına gol)
    const homeSecondHalfAvg = homeSecond.avg_goals_per_match;
    const awaySecondHalfAvg = awaySecond.avg_goals_per_match;

    // İkinci yarı gol farkı
    const secondHalfDiff = homeSecondHalfAvg - awaySecondHalfAvg;

    // İlk yarı tahminini hesapla
    let firstHalfPrediction;
    if (firstHalfDiff > 0.5) {
        firstHalfPrediction = '1'; // Ev sahibi önde bitiriyor
    } else if (firstHalfDiff < -0.3) {
        firstHalfPrediction = '2'; // Deplasman önde bitiriyor
    } else {
        firstHalfPrediction = 'X'; // Berabere
    }

    // İkinci yarıda gol sayısı farkına göre tahmin
    // ÖNEMLİ: Eğer global tahmin verisi zaten varsa, MS kısmını onunla tutarlı hale getir
    let secondHalfPrediction;
    
    if (window.predictionData && window.predictionData.predictions && 
        window.predictionData.predictions.most_likely_outcome) {
        
        // Tahmin butonu verilerini kullan (maç sonucu ile tutarlılık için)
        const globalOutcome = window.predictionData.predictions.most_likely_outcome;
        
        console.log("Tahmin butonu verileri:", window.predictionData);
        
        // MS kısmını tutarlı hale getir
        if (globalOutcome === "HOME_WIN") {
            secondHalfPrediction = '1'; // Ev sahibi kazanıyor
        } else if (globalOutcome === "AWAY_WIN") {
            secondHalfPrediction = '2'; // Deplasman kazanıyor
        } else if (globalOutcome === "DRAW") {
            secondHalfPrediction = 'X'; // Berabere
        } else {
            // Bilinmeyen durum, varsayılan olarak kombinasyon seç
            console.warn("Beklenmeyen tahmin sonucu:", globalOutcome);
            secondHalfPrediction = 'X'; // Varsayılan olarak berabere
        }
        
        // Global sonuç bilgisi - tüm İY/MS olasılıklarını düzenlemek için kullanılacak
        window.globalMatchOutcome = secondHalfPrediction;
        
        console.log("İY/MS tahminleri hesaplandı (tahmin butonu ile uyumlu):", {
            prediction: `${firstHalfPrediction}/${secondHalfPrediction}`,
            global_outcome: globalOutcome
        });
    } else {
        // Tahmin verisi yoksa, istatistiklerden hesapla
        // İkinci yarı + ilk yarıdan gelen avantaj
        const combinedDiff = secondHalfDiff + (firstHalfDiff * 0.3);
        
        if (combinedDiff > 0.4) {
            secondHalfPrediction = '1'; // Ev sahibi kazanıyor
        } else if (combinedDiff < -0.3) {
            secondHalfPrediction = '2'; // Deplasman kazanıyor
        } else {
            secondHalfPrediction = 'X'; // Berabere
        }
    }

    // İY/MS tahminini oluştur
    const htftPrediction = `${firstHalfPrediction}/${secondHalfPrediction}`;

    // Tüm kombinasyonların olasılıklarını hesapla
    const finalProbabilities = calculateAllHtFtProbabilities(
        homeFirst, homeSecond, awayFirst, awaySecond, 
        homeMatches, awayMatches
    );

    // En yüksek olasılıklı tahmini tekrar bulalım - bu sefer ayarlanmış olasılıklar kullanılacak!
    const sortedPredictions = Object.entries(finalProbabilities)
        .sort((a, b) => b[1] - a[1]);
    
    // En yüksek olasılıklı tahmin
    const adjustedPrediction = sortedPredictions.length > 0 ? sortedPredictions[0][0] : htftPrediction;
    
    // En yüksek olasılıklı 3 tahmini bul
    const topPredictions = findTopPredictions(finalProbabilities, 3);

    // Eğer global maç sonucu tanımlanmışsa, İY/MS tahmini mutlaka bununla uyumlu olmalı
    if (window.globalMatchOutcome) {
        const msOutcome = window.globalMatchOutcome;
        
        // Tahminimizin MS kısmı
        const predictionMs = adjustedPrediction.split('/')[1];
        
        // Eğer MS kısmı global sonuçla uyumlu değilse
        if (predictionMs !== msOutcome) {
            console.warn("Tahmin düzeltiliyor: Hesaplanan olasılıklarda bile uyumsuzluk var!");
            // MS ile uyumlu olan en yüksek olasılıklı tahmini bul
            const compatiblePredictions = sortedPredictions.filter(([key, _]) => key.split('/')[1] === msOutcome);
            if (compatiblePredictions.length > 0) {
                // En yüksek olasılıklı uyumlu tahmini kullan
                return {
                    prediction: compatiblePredictions[0][0],
                    top_predictions: topPredictions,
                    all_probabilities: finalProbabilities
                };
            }
        }
    }

    return {
        prediction: adjustedPrediction,
        top_predictions: topPredictions,
        all_probabilities: finalProbabilities
    };
}

/**
 * Tüm İY/MS olasılıklarını hesapla
 * Tahmin butonu ile tutarlı olacak şekilde daha dengeli olasılıklar üretir
 */
function calculateAllHtFtProbabilities(homeFirst, homeSecond, awayFirst, awaySecond, homeMatches, awayMatches) {
    const probabilities = {};
    const totalWeight = 100; // Toplam ağırlık
    
    // Log olarak maç sayılarını da yazdıralım
    console.log("Total matches for each team:", {
        "ev_sahibi": homeMatches || "unknown",
        "deplasman": awayMatches || "unknown"
    });
    
    // Monte Carlo simülasyonu ve yapay sinir ağı uygulaması ekleyelim
    const monteCarloProbs = runMonteCarloSimulation(homeFirst, homeSecond, awayFirst, awaySecond, 10000);
    const neuralNetProbs = predictWithNeuralNetwork(homeFirst, homeSecond, awayFirst, awaySecond);
    
    // Takımların toplam performansını hesapla
    const homeTotalAvg = homeFirst.avg_goals_per_match + homeSecond.avg_goals_per_match;
    const awayTotalAvg = awayFirst.avg_goals_per_match + awaySecond.avg_goals_per_match;
    
    // Ev avantajı - genelde ev sahibi takımlar %10-15 daha avantajlı 
    const homeAdvantage = 1.15;
    
    // İlk yarı ev sahibi önde bitirme olasılığı için temel değerler
    let homeFirstHalfStrength = (homeFirst.avg_goals_per_match * homeAdvantage - awayFirst.avg_goals_per_match * 0.9);
    
    // İlk yarı deplasman önde bitirme olasılığı için temel değerler
    let awayFirstHalfStrength = (awayFirst.avg_goals_per_match - homeFirst.avg_goals_per_match * 0.85);
    
    // İlk yarı berabere bitme olasılığı - futbolun doğasında ilk yarıda beraberlikler yaygındır
    let drawFirstHalfBase = 0.45; // 0.35'ten 0.45'e yükseltildi
    
    // Gol sayısı düşükse beraberlik olasılığını arttır
    if (homeFirst.avg_goals_per_match < 1.5 && awayFirst.avg_goals_per_match < 1.5) {
        drawFirstHalfBase = 0.45; // Düşük skorlu maçlarda beraberlik daha olası
    }
    
    // Güçler arasındaki fark çok büyükse beraberlik olasılığını azalt
    const firstHalfDiff = Math.abs(homeFirst.avg_goals_per_match - awayFirst.avg_goals_per_match);
    if (firstHalfDiff > 1.5) {
        drawFirstHalfBase = Math.max(0.2, drawFirstHalfBase - (firstHalfDiff * 0.1));
    }
    
    // Gol beklentisi yüksekse beraberlik olasılığı düşer
    if (homeFirst.avg_goals_per_match + awayFirst.avg_goals_per_match > 3) {
        drawFirstHalfBase = Math.max(0.2, drawFirstHalfBase - 0.1);
    }
    
    // İlk yarı berabere olasılığı
    let drawFirstHalfStrength = drawFirstHalfBase - (Math.abs(homeFirstHalfStrength - awayFirstHalfStrength) * 0.2);
    drawFirstHalfStrength = Math.max(0.15, drawFirstHalfStrength); // Minimum %15 olasılık
    
    // Negatif değerlerin önüne geç
    homeFirstHalfStrength = Math.max(0.1, homeFirstHalfStrength);
    awayFirstHalfStrength = Math.max(0.1, awayFirstHalfStrength);
    
    // Tam maç sonucu için olasılıklar
    let homeWinStrength = (homeTotalAvg * homeAdvantage - awayTotalAvg * 0.9);
    let awayWinStrength = (awayTotalAvg - homeTotalAvg * 0.85);
    
    // Beraberlik olasılığı için temel değer
    let drawFullTimeBase = 0.3;
    
    // Tam maç gol sayısı düşükse beraberlik olasılığını arttır
    if (homeTotalAvg < 2.5 && awayTotalAvg < 2.5) {
        drawFullTimeBase = 0.4;
    }
    
    // Güçler arasındaki fark çok büyükse beraberlik olasılığını azalt
    const fullTimeDiff = Math.abs(homeTotalAvg - awayTotalAvg);
    if (fullTimeDiff > 2) {
        drawFullTimeBase = Math.max(0.15, drawFullTimeBase - (fullTimeDiff * 0.1));
    }
    
    // Tam maç için beraberlik olasılığı
    let drawFullTimeStrength = drawFullTimeBase - (Math.abs(homeWinStrength - awayWinStrength) * 0.15);
    drawFullTimeStrength = Math.max(0.12, drawFullTimeStrength); // Minimum %12 olasılık
    
    // Negatif değerlerin önüne geç
    homeWinStrength = Math.max(0.1, homeWinStrength);
    awayWinStrength = Math.max(0.1, awayWinStrength);
    
    // Takım karakteristiklerini dikkate al - birçok takım için özel durumları modelle
    // Örnek: Atletico Madrid gibi takımlar ilk yarıyı genelde önde bitirip ikinci yarı düşüş yaşar
    
    // İkinci yarı düşüş yaşayan takımları tespit et
    const homeSecondHalfDrop = homeFirst.avg_goals_per_match > (homeSecond.avg_goals_per_match * 1.3);
    const awaySecondHalfDrop = awayFirst.avg_goals_per_match > (awaySecond.avg_goals_per_match * 1.3);
    
    // İkinci yarı yükselen takımları tespit et
    const homeSecondHalfRise = homeSecond.avg_goals_per_match > (homeFirst.avg_goals_per_match * 1.3);
    const awaySecondHalfRise = awaySecond.avg_goals_per_match > (awayFirst.avg_goals_per_match * 1.3);
    
    // İlk yarı olasılıkları için normalleştirme
    const firstHalfTotal = homeFirstHalfStrength + awayFirstHalfStrength + drawFirstHalfStrength;
    const homeFirstProb = homeFirstHalfStrength / firstHalfTotal;
    const awayFirstProb = awayFirstHalfStrength / firstHalfTotal;
    const drawFirstProb = drawFirstHalfStrength / firstHalfTotal;
    
    // Tam maç olasılıkları için normalleştirme
    const fullTimeTotal = homeWinStrength + awayWinStrength + drawFullTimeStrength;
    const homeWinProb = homeWinStrength / fullTimeTotal;
    const awayWinProb = awayWinStrength / fullTimeTotal;
    const drawFullTimeProb = drawFullTimeStrength / fullTimeTotal;
    
    // Tüm kombinasyonlar için başlangıç olasılıklarını hesapla
    // İY/MS kombinasyonları için olasılıkları hesapla (daha gerçekçi korelasyon faktörleriyle)
    probabilities['1/1'] = Math.round((homeFirstProb * homeWinProb * 1.1) * 100); // 1.3'ten 1.1'e düşürüldü
    probabilities['1/X'] = Math.round((homeFirstProb * drawFullTimeProb * 1.3) * 100); // 1.1'den 1.3'e yükseltildi
    probabilities['1/2'] = Math.round((homeFirstProb * awayWinProb * 1.2) * 100); // 0.9'dan 1.2'ye yükseltildi
    probabilities['X/1'] = Math.round((drawFirstProb * homeWinProb * 1.4) * 100); // 1.2'den 1.4'e yükseltildi
    probabilities['X/X'] = Math.round((drawFirstProb * drawFullTimeProb * 1.3) * 100); // Değişmedi
    probabilities['X/2'] = Math.round((drawFirstProb * awayWinProb * 1.4) * 100); // 1.2'den 1.4'e yükseltildi
    probabilities['2/1'] = Math.round((awayFirstProb * homeWinProb * 1.2) * 100); // 0.8'den 1.2'ye yükseltildi
    probabilities['2/X'] = Math.round((awayFirstProb * drawFullTimeProb * 1.3) * 100); // 1.1'den 1.3'e yükseltildi
    probabilities['2/2'] = Math.round((awayFirstProb * awayWinProb * 1.15) * 100); // 1.4'ten 1.15'e düşürüldü
    
    // Takım özelliklerine göre ayarlamalar yap
    // İlk yarı önde bitirip ikinci yarı düşüş yaşayan takımlar için
    if (homeSecondHalfDrop) {
        // Ev sahibi ilk yarı önde bitirip sonra berabere/yenilme olasılığı artar
        probabilities['1/X'] = Math.round(probabilities['1/X'] * 1.4);
        probabilities['1/2'] = Math.round(probabilities['1/2'] * 1.3);
        probabilities['1/1'] = Math.round(probabilities['1/1'] * 0.9);
    }
    
    if (awaySecondHalfDrop) {
        // Deplasman ilk yarı önde bitirip sonra berabere/yenilme olasılığı artar
        probabilities['2/X'] = Math.round(probabilities['2/X'] * 1.4);
        probabilities['2/1'] = Math.round(probabilities['2/1'] * 1.3);
        probabilities['2/2'] = Math.round(probabilities['2/2'] * 0.9);
    }
    
    // İkinci yarı yükselen takımlar için
    if (homeSecondHalfRise) {
        // Ev sahibi ilk yarı geride/berabere olup kazanma olasılığı artar
        probabilities['X/1'] = Math.round(probabilities['X/1'] * 1.5);
        probabilities['2/1'] = Math.round(probabilities['2/1'] * 1.4);
    }
    
    if (awaySecondHalfRise) {
        // Deplasman ilk yarı geride/berabere olup kazanma olasılığı artar
        probabilities['X/2'] = Math.round(probabilities['X/2'] * 1.5);
        probabilities['1/2'] = Math.round(probabilities['1/2'] * 1.4);
    }
    
    // En düşük olasılık oranlarını dengele (bir tahmin çok baskın olmasın)
    const maxProbability = Math.max(...Object.values(probabilities));
    
    // Eğer bir olasılık çok yüksekse (%70'den fazla), diğerlerini biraz artır
    if (maxProbability > 70) {
        for (let key in probabilities) {
            if (probabilities[key] < 5) {
                probabilities[key] = 5 + Math.floor(Math.random() * 5); // 5-9 arası bir değer
            }
        }
    }
    
    // İstatistiklere göre bazı özel durumları ele al
    // 1. Çok gollü takımlar için (ortalama 3+ gol/maç)
    if (homeTotalAvg > 3 && awayTotalAvg > 2.5) {
        // Her iki takım da çok gol atıyorsa, X/X olasılığını azalt
        probabilities['X/X'] = Math.max(5, Math.round(probabilities['X/X'] * 0.8));
    }
    
    // 2. Az gollü takımlar için (ortalama 1- gol/maç)
    if (homeTotalAvg < 1 && awayTotalAvg < 1) {
        // Her iki takım da az gol atıyorsa, X/X olasılığını artır
        probabilities['X/X'] = Math.min(50, Math.round(probabilities['X/X'] * 1.5));
    }
    
    // Olasılıkları normalleştir (toplamı 100 olacak şekilde)
    let totalProb = 0;
    for (const key in probabilities) {
        totalProb += probabilities[key];
    }
    
    const normalizationFactor = totalWeight / totalProb;
    
    // Tüm olasılıkları normalize et
    for (const key in probabilities) {
        probabilities[key] = Math.round(probabilities[key] * normalizationFactor);
        
        // Minimum değer kontrolü - hiçbir sonuç sıfır olasılık olmasın
        if (probabilities[key] < 3) {
            probabilities[key] = 3 + Math.floor(Math.random() * 3); // 3-5 arası rastgele değer
        }
    }
    
    // Monte Carlo ve sinir ağı modellerini birleştir
    try {
        // Birleştirilmiş model sonuçlarını al
        const combinedResults = combineModelResults(
            probabilities,      // İstatistik temelli model
            monteCarloProbs,    // Monte Carlo simülasyonu
            neuralNetProbs      // Yapay sinir ağı
        );
        
        // Konsola model sonuçlarını yazdır
        console.log("İY/MS Tahmin Modelleri Sonuçları:", {
            istatistik: probabilities,
            monteCarlo: monteCarloProbs,
            neuralNetwork: neuralNetProbs,
            combined: combinedResults
        });
        
        // Tahmin butonu bağlantısı kaldırıldı - artık globalMatchOutcome değişkeni kullanılmıyor
        // Hiçbir bağlantı, MS sonuçları artık tamamen bağımsız
        
        // İstatistik verilerine dayalı temel değerleri kullan
        return combinedResults;
    } 
    catch (error) {
        console.error("Gelişmiş tahmin modellerinde hata:", error);
        
        // Hata durumunda orijinal istatistik temelli sonuçları kullan
        // Son bir kez daha toplam 100 olacak şekilde ayarla
        totalProb = 0;
        for (const key in probabilities) {
            totalProb += probabilities[key];
        }
        
        // Eğer toplam tam 100 değilse, en yüksek olasılığı ayarla
        if (totalProb !== 100) {
            const diff = 100 - totalProb;
            const highestKey = Object.keys(probabilities).reduce((a, b) => 
                probabilities[a] > probabilities[b] ? a : b);
            probabilities[highestKey] += diff;
        }
        
        return probabilities;
    }
}

/**
 * En yüksek olasılıklı tahminleri bul
 */
function findTopPredictions(probabilities, count) {
    // Olasılıkları sırala
    const sorted = Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1])
        .slice(0, count)
        .map(([prediction, probability]) => ({
            prediction,
            probability
        }));
    
    return sorted;
}

/**
 * İY/MS tahminin açıklamasını al
 */
function getHtFtDescription(prediction) {
    const descriptions = {
        '1/1': 'Ev Sahibi Önde/Ev Sahibi Kazanır',
        '1/X': 'Ev Sahibi Önde/Berabere',
        '1/2': 'Ev Sahibi Önde/Deplasman Kazanır',
        'X/1': 'İlk Yarı Berabere/Ev Sahibi Kazanır',
        'X/X': 'İlk Yarı Berabere/Maç Berabere',
        'X/2': 'İlk Yarı Berabere/Deplasman Kazanır',
        '2/1': 'Deplasman Önde/Ev Sahibi Kazanır',
        '2/X': 'Deplasman Önde/Berabere',
        '2/2': 'Deplasman Önde/Deplasman Kazanır'
    };
    
    return descriptions[prediction] || 'Bilinmeyen Tahmin';
}

/**
 * İY/MS tahminleri için HTML içeriği oluştur
 * @param {Object} htftData - İY/MS tahmin verileri
 * @param {string} homeName - Ev sahibi takım adı
 * @param {string} awayName - Deplasman takımı adı
 * @param {Object} homeFirstHalfResults - Ev sahibi takım ilk yarı sonuçları (opsiyonel)
 * @param {Object} awayFirstHalfResults - Deplasman takımı ilk yarı sonuçları (opsiyonel)
 * @param {number} homeMatches - Ev sahibi takım toplam maç sayısı (opsiyonel)
 * @param {number} awayMatches - Deplasman takımı toplam maç sayısı (opsiyonel)
 */
function generateHtFtPredictionHTML(htftData, homeName, awayName, homeFirstHalfResults, awayFirstHalfResults, homeMatches, awayMatches) {
    if (!htftData || htftData.error) {
        return `
            <div class="alert alert-warning">
                <strong>İY/MS Tahmini Yapılamadı</strong>
                <p>${htftData?.error || 'Tahmin için yeterli veri bulunamadı.'}</p>
            </div>
        `;
    }
    
    const { prediction, top_predictions, all_probabilities } = htftData;
    
    // İlk Yarı Durumları bölümü için gerçek verileri API'den alalım
    let stats = {
        ev: {
            ilk_yari: {
                results: homeFirstHalfResults || { total: { '1': 0, 'X': 0, '2': 0 } }
            },
            mac_sayisi: homeMatches || 0
        },
        deplasman: {
            ilk_yari: {
                results: awayFirstHalfResults || { total: { '1': 0, 'X': 0, '2': 0 } }
            },
            mac_sayisi: awayMatches || 0
        }
    };
    
    console.log("İlk Yarı Durumları gerçek veriler:", stats);
    
    // Form ve motivasyon faktörlerini hesapla ve global değişkene kaydet
    // Bu faktörler daha sonra Monte Carlo simülasyonu ve yapay sinir ağında kullanılacak
    let formMotivationHtml = '';
    if (window.predictionData) {
        // Form faktörlerini hesapla
        const homeTeamData = window.predictionData.home_team;
        const awayTeamData = window.predictionData.away_team;
        
        // Global olarak kullanılabilecek form ve motivasyon faktörleri
        window.formMotivationData = {
            ev: {
                form3: calculateFormPoints(homeTeamData?.form?.detailed_data?.all || [], 3).formFactor || 0.5,
                form6: calculateFormPoints(homeTeamData?.form?.detailed_data?.all || [], 6).formFactor || 0.5,
                form9: calculateFormPoints(homeTeamData?.form?.detailed_data?.all || [], 9).formFactor || 0.5,
                motivasyon: calculateMotivationFactor(homeTeamData?.form?.detailed_data?.all || []) || 0.5
            },
            deplasman: {
                form3: calculateFormPoints(awayTeamData?.form?.detailed_data?.all || [], 3).formFactor || 0.5,
                form6: calculateFormPoints(awayTeamData?.form?.detailed_data?.all || [], 6).formFactor || 0.5,
                form9: calculateFormPoints(awayTeamData?.form?.detailed_data?.all || [], 9).formFactor || 0.5,
                motivasyon: calculateMotivationFactor(awayTeamData?.form?.detailed_data?.all || []) || 0.5
            }
        };
        
        console.log("Form ve Motivasyon Faktörleri:", window.formMotivationData);
        
        // Form ve motivasyon verilerini HTML içeriğine ekle
        formMotivationHtml = `
            <div class="row mb-4">
                <div class="col-12">
                    <h6 class="mb-3">Form ve Motivasyon Faktörleri</h6>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header bg-primary text-white">
                                    <h6 class="mb-0">${homeName}</h6>
                                </div>
                                <div class="card-body">
                                    <div class="d-flex justify-content-between mb-2">
                                        <span>Son 3 Maç Form:</span>
                                        <span class="badge bg-info">${(window.formMotivationData.ev.form3 * 100).toFixed(0)}%</span>
                                    </div>
                                    <div class="d-flex justify-content-between mb-2">
                                        <span>Son 6 Maç Form:</span>
                                        <span class="badge bg-info">${(window.formMotivationData.ev.form6 * 100).toFixed(0)}%</span>
                                    </div>
                                    <div class="d-flex justify-content-between mb-2">
                                        <span>Son 9 Maç Form:</span>
                                        <span class="badge bg-info">${(window.formMotivationData.ev.form9 * 100).toFixed(0)}%</span>
                                    </div>
                                    <div class="d-flex justify-content-between">
                                        <span>Motivasyon:</span>
                                        <span class="badge bg-success">${(window.formMotivationData.ev.motivasyon * 100).toFixed(0)}%</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header bg-danger text-white">
                                    <h6 class="mb-0">${awayName}</h6>
                                </div>
                                <div class="card-body">
                                    <div class="d-flex justify-content-between mb-2">
                                        <span>Son 3 Maç Form:</span>
                                        <span class="badge bg-info">${(window.formMotivationData.deplasman.form3 * 100).toFixed(0)}%</span>
                                    </div>
                                    <div class="d-flex justify-content-between mb-2">
                                        <span>Son 6 Maç Form:</span>
                                        <span class="badge bg-info">${(window.formMotivationData.deplasman.form6 * 100).toFixed(0)}%</span>
                                    </div>
                                    <div class="d-flex justify-content-between mb-2">
                                        <span>Son 9 Maç Form:</span>
                                        <span class="badge bg-info">${(window.formMotivationData.deplasman.form9 * 100).toFixed(0)}%</span>
                                    </div>
                                    <div class="d-flex justify-content-between">
                                        <span>Motivasyon:</span>
                                        <span class="badge bg-success">${(window.formMotivationData.deplasman.motivasyon * 100).toFixed(0)}%</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    let html = `
        <div class="card border-primary mt-4 mb-4">
            <div class="card-header bg-primary text-white">
                <h5>İlk Yarı / Maç Sonu Tahminleri - ${homeName} vs ${awayName}</h5>
            </div>
            <div class="card-body">
                ${formMotivationHtml}
                <div class="row">
                    <div class="col-md-6">
                        <div class="alert alert-success">
                            <h5>En Olası Tahmin: ${prediction} (${all_probabilities[prediction]}%)</h5>
                            <p>${getHtFtDescription(prediction)}</p>
                        </div>
                        
                        <h6>En Yüksek Olasılıklı Tahminler:</h6>
                        <ul class="list-group">
    `;
    
    // En olası 3 tahmini listele
    top_predictions.forEach(pred => {
        html += `
            <li class="list-group-item d-flex justify-content-between align-items-center">
                ${pred.prediction} - ${getHtFtDescription(pred.prediction)}
                <span class="badge bg-primary rounded-pill">${pred.probability}%</span>
            </li>
        `;
    });
    
    html += `
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h6>Tüm İY/MS Olasılıkları</h6>
                        <table class="table table-striped table-sm">
                            <thead>
                                <tr>
                                    <th>Tahmin</th>
                                    <th>Açıklama</th>
                                    <th>Olasılık</th>
                                </tr>
                            </thead>
                            <tbody>
    `;
    
    // Tüm tahminleri olasılık sırasına göre listele
    Object.entries(all_probabilities)
        .sort((a, b) => b[1] - a[1])
        .forEach(([pred, prob]) => {
            html += `
                <tr>
                    <td>${pred}</td>
                    <td>${getHtFtDescription(pred)}</td>
                    <td>${prob}%</td>
                </tr>
            `;
        });
    
    html += `
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <!-- İlk Yarı Durumları bölümü -->
                <div class="row mt-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header bg-dark text-white">
                                <h6 class="mb-0">İlk Yarı Durumları</h6>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-6">
                                        <h6 class="text-center mb-3">Ev Sahibi - Son 21 Maç</h6>
                                        <div class="d-flex justify-content-between align-items-center mb-2">
                                            <span>Önde Bitirme:</span>
                                            <div class="progress flex-grow-1 mx-2" style="height: 20px;">
                                                <div class="progress-bar bg-success" style="width: ${Math.round((parseInt(stats.ev.ilk_yari.results.total['1']) / parseInt(stats.ev.mac_sayisi)) * 100)}%">
                                                    ${stats.ev.ilk_yari.results.total['1']} Maç
                                                </div>
                                            </div>
                                            <span class="badge bg-success">${Math.round((parseInt(stats.ev.ilk_yari.results.total['1']) / parseInt(stats.ev.mac_sayisi)) * 100)}%</span>
                                        </div>
                                        <div class="d-flex justify-content-between align-items-center mb-2">
                                            <span>Berabere Bitirme:</span>
                                            <div class="progress flex-grow-1 mx-2" style="height: 20px;">
                                                <div class="progress-bar bg-warning" style="width: ${Math.round((parseInt(stats.ev.ilk_yari.results.total['X']) / parseInt(stats.ev.mac_sayisi)) * 100)}%">
                                                    ${stats.ev.ilk_yari.results.total['X']} Maç
                                                </div>
                                            </div>
                                            <span class="badge bg-warning">${Math.round((parseInt(stats.ev.ilk_yari.results.total['X']) / parseInt(stats.ev.mac_sayisi)) * 100)}%</span>
                                        </div>
                                        <div class="d-flex justify-content-between align-items-center">
                                            <span>Geride Bitirme:</span>
                                            <div class="progress flex-grow-1 mx-2" style="height: 20px;">
                                                <div class="progress-bar bg-danger" style="width: ${Math.round((parseInt(stats.ev.ilk_yari.results.total['2']) / parseInt(stats.ev.mac_sayisi)) * 100)}%">
                                                    ${stats.ev.ilk_yari.results.total['2']} Maç
                                                </div>
                                            </div>
                                            <span class="badge bg-danger">${Math.round((parseInt(stats.ev.ilk_yari.results.total['2']) / parseInt(stats.ev.mac_sayisi)) * 100)}%</span>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <h6 class="text-center mb-3">Deplasman - Son 21 Maç</h6>
                                        <div class="d-flex justify-content-between align-items-center mb-2">
                                            <span>Önde Bitirme:</span>
                                            <div class="progress flex-grow-1 mx-2" style="height: 20px;">
                                                <div class="progress-bar bg-success" style="width: ${Math.round((parseInt(stats.deplasman.ilk_yari.results.total['1']) / parseInt(stats.deplasman.mac_sayisi)) * 100)}%">
                                                    ${stats.deplasman.ilk_yari.results.total['1']} Maç
                                                </div>
                                            </div>
                                            <span class="badge bg-success">${Math.round((parseInt(stats.deplasman.ilk_yari.results.total['1']) / parseInt(stats.deplasman.mac_sayisi)) * 100)}%</span>
                                        </div>
                                        <div class="d-flex justify-content-between align-items-center mb-2">
                                            <span>Berabere Bitirme:</span>
                                            <div class="progress flex-grow-1 mx-2" style="height: 20px;">
                                                <div class="progress-bar bg-warning" style="width: ${Math.round((parseInt(stats.deplasman.ilk_yari.results.total['X']) / parseInt(stats.deplasman.mac_sayisi)) * 100)}%">
                                                    ${stats.deplasman.ilk_yari.results.total['X']} Maç
                                                </div>
                                            </div>
                                            <span class="badge bg-warning">${Math.round((parseInt(stats.deplasman.ilk_yari.results.total['X']) / parseInt(stats.deplasman.mac_sayisi)) * 100)}%</span>
                                        </div>
                                        <div class="d-flex justify-content-between align-items-center">
                                            <span>Geride Bitirme:</span>
                                            <div class="progress flex-grow-1 mx-2" style="height: 20px;">
                                                <div class="progress-bar bg-danger" style="width: ${Math.round((parseInt(stats.deplasman.ilk_yari.results.total['2']) / parseInt(stats.deplasman.mac_sayisi)) * 100)}%">
                                                    ${stats.deplasman.ilk_yari.results.total['2']} Maç
                                                </div>
                                            </div>
                                            <span class="badge bg-danger">${Math.round((parseInt(stats.deplasman.ilk_yari.results.total['2']) / parseInt(stats.deplasman.mac_sayisi)) * 100)}%</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="alert alert-info mt-4">
                    <p class="mb-0"><small>İY/MS tahminleri, takımların son maçlardaki ilk yarı ve ikinci yarı istatistiklerine, form durumlarına ve motivasyon faktörlerine dayanarak oluşturulmuştur. Tahminler yalnızca bilgi amaçlıdır.</small></p>
                </div>
            </div>
        </div>
    `;
    
    return html;
}

/**
 * Monte Carlo simülasyonu ile İY/MS tahminleri yap
 * @param {Object} homeFirst - Ev sahibi ilk yarı istatistikleri
 * @param {Object} homeSecond - Ev sahibi ikinci yarı istatistikleri 
 * @param {Object} awayFirst - Deplasman ilk yarı istatistikleri
 * @param {Object} awaySecond - Deplasman ikinci yarı istatistikleri
 * @param {Number} simulationCount - Simülasyon sayısı (default: 10000)
 * @returns {Object} - İY/MS olasılıkları
 */
function runMonteCarloSimulation(homeFirst, homeSecond, awayFirst, awaySecond, simulationCount = 10000) {
    // Poisson dağılımı ile rastgele gol sayısı oluştur
    function poissonRandom(lambda) {
        if (lambda <= 0) return 0;
        
        let L = Math.exp(-lambda);
        let p = 1.0;
        let k = 0;
        
        do {
            k++;
            p *= Math.random();
        } while (p > L);
        
        return k - 1;
    }
    
    // Bayesian yaklaşımı için varsayılan değerler ve düzeltme faktörleri
    // Tipik bir maçta yarı başına ortalama gol sayıları (öncül değerler)
    const priorFirstHalfGoals = 1.2;
    const priorSecondHalfGoals = 1.5;
    
    // Takımların maç sayıları - HTML'den veri aldığımız objelerde yer alıyor
    const homeMatches = homeFirst.total_matches || 21; // Toplam maç sayısı (varsayılan 21)
    const awayMatches = awayFirst.total_matches || 21;
    
    // Güven faktörleri - az maç sayısı olan takımlar için düzeltme 
    // 15 maçtan az ise, global ortalamalara doğru eğilim göstersin
    const homeConfidenceFactor = Math.min(1.0, homeMatches / 15);
    const awayConfidenceFactor = Math.min(1.0, awayMatches / 15);
    
    // Bayesian düzeltme faktörleri - güven düştükçe artar
    const homeBayesianAdjustment = 1 - homeConfidenceFactor; 
    const awayBayesianAdjustment = 1 - awayConfidenceFactor;
    
    // Form, motivasyon ve İLK YARI faktörlerini kullanalım (varsa)
    let homeFormFactor = 1.0;
    let awayFormFactor = 1.0;
    
    // İLK YARI DURUMLARI İÇİN YENİ FAKTÖRLER
    let homeFirstHalfFactor = 1.0;
    let awayFirstHalfFactor = 1.0;
    
    // İlk yarı istatistiklerinden faktör oluştur - varsa global stats objesini kullan
    if (window.stats || typeof stats !== 'undefined') {
        try {
            // Ev sahibi takımın ilk yarı önde bitirme yüzdesi
            const homeLeadingPercentage = parseInt(stats.ev.ilk_yari.results.total['1']) / parseInt(stats.ev.mac_sayisi);
            // Deplasman takımının ilk yarı önde bitirme yüzdesi
            const awayLeadingPercentage = parseInt(stats.deplasman.ilk_yari.results.total['1']) / parseInt(stats.deplasman.mac_sayisi);
            
            // İlk yarı durumu faktörlerini hesapla - önde bitirme yüzdesi yüksekse faktörü artır
            homeFirstHalfFactor = 0.5 + (homeLeadingPercentage * 1.5); // 0.5-2.0 arası
            awayFirstHalfFactor = 0.5 + (awayLeadingPercentage * 1.5); // 0.5-2.0 arası
            
            console.log("İlk Yarı Durum Faktörleri:", {
                ev: homeFirstHalfFactor.toFixed(2),
                deplasman: awayFirstHalfFactor.toFixed(2)
            });
        } catch (error) {
            console.log("İlk yarı durumları faktörü hesaplanırken hata:", error);
        }
    }
    
    if (window.formMotivationData) {
        // Son 6 maçta formları değerlendir ve gol beklentilerini skala (0.75-1.25 arası)
        homeFormFactor = 0.75 + (window.formMotivationData.ev.form6 * 0.5); 
        awayFormFactor = 0.75 + (window.formMotivationData.deplasman.form6 * 0.5);
        
        // Motivasyon faktörünü daha güçlü etkiyle ekleyelim
        homeFormFactor += (window.formMotivationData.ev.motivasyon - 0.5) * 0.3; // ±0.15 etki
        awayFormFactor += (window.formMotivationData.deplasman.motivasyon - 0.5) * 0.3;
        
        // İLK YARI FAKTÖRÜNÜ DE EKLE - ÇOK DAHA GÜÇLÜ ETKİYLE
        homeFormFactor *= homeFirstHalfFactor; // İlk yarı performansına göre çarparak etkiyi artır
        awayFormFactor *= awayFirstHalfFactor; // İlk yarı performansına göre çarparak etkiyi artır
        
        // Faktörleri sınırla (0.5-2.5 arası - ÇOK DAHA GENİŞ ARALIK)
        homeFormFactor = Math.max(0.5, Math.min(2.5, homeFormFactor));
        awayFormFactor = Math.max(0.5, Math.min(2.5, awayFormFactor));
    }
    
    // Takımların gol beklentileri - Bayesian düzeltme uygulayarak
    // Maç sayısı az olan takımları global ortalamalara yakınlaştır
    const homeFirstGoalExp = ((homeFirst.avg_goals_per_match * homeConfidenceFactor) + 
                             (priorFirstHalfGoals * homeBayesianAdjustment)) * homeFormFactor;
    
    const homeSecondGoalExp = ((homeSecond.avg_goals_per_match * homeConfidenceFactor) + 
                              (priorSecondHalfGoals * homeBayesianAdjustment)) * homeFormFactor;
    
    const awayFirstGoalExp = ((awayFirst.avg_goals_per_match * awayConfidenceFactor) + 
                             (priorFirstHalfGoals * awayBayesianAdjustment)) * awayFormFactor;
    
    const awaySecondGoalExp = ((awaySecond.avg_goals_per_match * awayConfidenceFactor) + 
                              (priorSecondHalfGoals * awayBayesianAdjustment)) * awayFormFactor;
                              
    // Konsola yazdırma (debugging için)
    console.log("Monte Carlo - Bayesian düzeltilmiş beklentiler:", {
        ev: {
            ilk_yari: homeFirstGoalExp, 
            ikinci_yari: homeSecondGoalExp,
            ham_ilk_yari: homeFirst.avg_goals_per_match,
            ham_ikinci_yari: homeSecond.avg_goals_per_match,
            maç_sayısı: homeMatches,
            güven_faktörü: homeConfidenceFactor
        },
        deplasman: {
            ilk_yari: awayFirstGoalExp,
            ikinci_yari: awaySecondGoalExp,
            ham_ilk_yari: awayFirst.avg_goals_per_match,
            ham_ikinci_yari: awaySecond.avg_goals_per_match,
            maç_sayısı: awayMatches,
            güven_faktörü: awayConfidenceFactor
        }
    });
    
    // Sonuçları sayacak obje
    const results = {
        '1/1': 0, '1/X': 0, '1/2': 0,
        'X/1': 0, 'X/X': 0, 'X/2': 0,
        '2/1': 0, '2/X': 0, '2/2': 0
    };
    
    // Simülasyonları yap
    for (let i = 0; i < simulationCount; i++) {
        // İlk yarı gollerini simüle et
        const homeFirstHalfGoals = poissonRandom(homeFirstGoalExp);
        const awayFirstHalfGoals = poissonRandom(awayFirstGoalExp);
        
        // İkinci yarı gollerini simüle et
        const homeSecondHalfGoals = poissonRandom(homeSecondGoalExp);
        const awaySecondHalfGoals = poissonRandom(awaySecondGoalExp);
        
        // Toplam goller
        const homeFullTimeGoals = homeFirstHalfGoals + homeSecondHalfGoals;
        const awayFullTimeGoals = awayFirstHalfGoals + awaySecondHalfGoals;
        
        // İlk yarı sonucu - İLK YARI DURUMU İÇİN AĞIRLIK EKLENDİ
        let htResult;
        
        // İlk yarı istatistiklerinden gelen eğilimi de hesaba kat
        let homeFirstHalfBoost = 0;
        let awayFirstHalfBoost = 0;
        
        // İlk yarı durum verilerini kullan (eğer varsa)
        if (window.stats || typeof stats !== 'undefined') {
            try {
                // Ev sahibi takımın ilk yarı önde bitirme ve deplasman takımının ilk yarı geride bitirme olasılığı
                const homeLeadPercentage = parseInt(stats.ev.ilk_yari.results.total['1']) / parseInt(stats.ev.mac_sayisi);
                const awayBehindPercentage = parseInt(stats.deplasman.ilk_yari.results.total['2']) / parseInt(stats.deplasman.mac_sayisi);
                
                // Bu oranların çarpımı yüksekse, ev sahibinin ilk yarıyı önde bitirme olasılığını artır
                homeFirstHalfBoost = Math.round(homeLeadPercentage * awayBehindPercentage * 3); // 0-3 arası bir ağırlık
                
                // Aynı şekilde deplasman için de hesapla
                const awayLeadPercentage = parseInt(stats.deplasman.ilk_yari.results.total['1']) / parseInt(stats.deplasman.mac_sayisi);
                const homeBehindPercentage = parseInt(stats.ev.ilk_yari.results.total['2']) / parseInt(stats.ev.mac_sayisi);
                
                awayFirstHalfBoost = Math.round(awayLeadPercentage * homeBehindPercentage * 3); // 0-3 arası bir ağırlık
            } catch (error) {
                console.log("İlk yarı durum ağırlıkları hesaplanırken hata:", error);
            }
        }
        
        // Ağırlıkları da hesaba katarak sonucu belirle
        if (homeFirstHalfGoals + homeFirstHalfBoost > awayFirstHalfGoals + awayFirstHalfBoost) {
            htResult = '1';
        } else if (homeFirstHalfGoals + homeFirstHalfBoost < awayFirstHalfGoals + awayFirstHalfBoost) {
            htResult = '2';
        } else {
            htResult = 'X';
        }
        
        // Tam maç sonucu
        let ftResult;
        if (homeFullTimeGoals > awayFullTimeGoals) {
            ftResult = '1';
        } else if (homeFullTimeGoals < awayFullTimeGoals) {
            ftResult = '2';
        } else {
            ftResult = 'X';
        }
        
        // İY/MS kombinasyonu
        const htftResult = `${htResult}/${ftResult}`;
        results[htftResult]++;
    }
    
    // Olasılıkları hesapla (yüzde olarak)
    const probabilities = {};
    for (const result in results) {
        probabilities[result] = Math.round((results[result] / simulationCount) * 100);
    }
    
    return probabilities;
}

/**
 * Takımın son maçlardaki form durumunu hesapla
 * @param {Array} matches - Takımın maç verileri
 * @param {Number} count - Kaç maç geriye gidileceği (3, 6, 9 vs.)
 * @returns {Object} - Form durumu (galibiyet, beraberlik, mağlubiyet sayıları ve puan)
 */
function calculateFormPoints(matches, count = 5) {
    // Son 'count' maçı al
    const recentMatches = matches.slice(0, Math.min(count, matches.length));
    
    // Sayaçlar
    let wins = 0;
    let draws = 0;
    let losses = 0;
    
    // Maçları değerlendir
    recentMatches.forEach(match => {
        const goalsFor = match.goals_scored;
        const goalsAgainst = match.goals_conceded;
        
        if (goalsFor > goalsAgainst) {
            wins++;
        } else if (goalsFor === goalsAgainst) {
            draws++;
        } else {
            losses++;
        }
    });
    
    // Puanları hesapla (galibiyet: 3 puan, beraberlik: 1 puan)
    const points = (wins * 3) + draws;
    
    // Maksimum puanı hesapla
    const maxPoints = recentMatches.length * 3;
    
    // Form faktörünü 0-1 arasında hesapla
    const formFactor = maxPoints > 0 ? points / maxPoints : 0.5;
    
    return {
        matches: recentMatches.length,
        wins: wins,
        draws: draws,
        losses: losses,
        points: points,
        maxPoints: maxPoints,
        formFactor: formFactor
    };
}

/**
 * Takımın motivasyon/moral durumunu hesapla
 * @param {Array} matches - Takımın maç verileri
 * @returns {Number} - Motivasyon faktörü (0-1 arasında)
 */
function calculateMotivationFactor(matches) {
    if (!matches || matches.length === 0) {
        return 0.5; // Veri yoksa nötr değer döndür
    }
    
    // Son maç
    const lastMatch = matches[0];
    
    // Son maç galibiyet veya mağlubiyet
    const lastMatchResult = lastMatch.goals_scored > lastMatch.goals_conceded ? 'W' :
                           lastMatch.goals_scored === lastMatch.goals_conceded ? 'D' : 'L';
    
    // Son 3 maç
    const last3Matches = matches.slice(0, Math.min(3, matches.length));
    
    // Son 3 maçın sonuçları
    const last3Results = last3Matches.map(match => {
        return match.goals_scored > match.goals_conceded ? 'W' :
              match.goals_scored === match.goals_conceded ? 'D' : 'L';
    });
    
    // Son maç galibiyeti moral yükseltir
    let motivationFactor = 0.5; // Başlangıç değeri (nötr)
    
    if (lastMatchResult === 'W') {
        motivationFactor += 0.15; // Moral artışı
    } else if (lastMatchResult === 'L') {
        motivationFactor -= 0.1; // Moral düşüşü
    }
    
    // Son 3 maçtaki trend
    const winCount = last3Results.filter(result => result === 'W').length;
    const lossCount = last3Results.filter(result => result === 'L').length;
    
    // Son 3 maçta hiç kaybetmemek veya hepsini kazanmak moral verir
    if (lossCount === 0) {
        motivationFactor += 0.1;
    }
    
    if (winCount === 3) {
        motivationFactor += 0.1; // Tam form
    }
    
    // Son 3 maçta hiç kazanmamak moral bozar
    if (winCount === 0) {
        motivationFactor -= 0.1;
    }
    
    // Faktörü 0-1 arasında tut
    return Math.max(0, Math.min(1, motivationFactor));
}

/**
 * Yapay sinir ağı ile İY/MS tahminleri yap
 * Genişletilmiş sinir ağı modeli - form ve motivasyon faktörleri eklendi
 * @param {Object} homeFirst - Ev sahibi ilk yarı istatistikleri
 * @param {Object} homeSecond - Ev sahibi ikinci yarı istatistikleri
 * @param {Object} awayFirst - Deplasman ilk yarı istatistikleri
 * @param {Object} awaySecond - Deplasman ikinci yarı istatistikleri
 * @returns {Object} - İY/MS olasılıkları
 */
function predictWithNeuralNetwork(homeFirst, homeSecond, awayFirst, awaySecond) {
    // Takım performans faktörleri - Form ve motivasyon verilerini kullan (varsa)
    let homeFormFactor = 1.0;
    let awayFormFactor = 1.0;
    let homeMotivationFactor = 1.0;
    let awayMotivationFactor = 1.0;
    
    if (window.formMotivationData) {
        homeFormFactor = 0.75 + (window.formMotivationData.ev.form9 * 0.5); // 0.75-1.25 arası skala
        awayFormFactor = 0.75 + (window.formMotivationData.deplasman.form9 * 0.5);
        
        homeMotivationFactor = 0.8 + (window.formMotivationData.ev.motivasyon * 0.4); // 0.8-1.2 arası (daha yüksek etki)
        awayMotivationFactor = 0.8 + (window.formMotivationData.deplasman.motivasyon * 0.4);
    }
    
    // Giriş özellikleri
    // [12 özellik: ilk/ikinci yarı gol ortalamaları, form ve motivasyon faktörleri]
    const features = [
        homeFirst.avg_goals_per_match * homeFormFactor,         // Ev sahibi ilk yarı gol
        homeFirst.avg_goals_per_match / Math.max(1, awayFirst.avg_goals_per_match), // Oransal güç
        homeSecond.avg_goals_per_match * homeFormFactor,        // Ev sahibi ikinci yarı gol
        homeSecond.avg_goals_per_match / Math.max(1, awaySecond.avg_goals_per_match),
        awayFirst.avg_goals_per_match * awayFormFactor,         // Deplasman ilk yarı gol
        awayFirst.avg_goals_per_match / Math.max(1, homeFirst.avg_goals_per_match),
        awaySecond.avg_goals_per_match * awayFormFactor,        // Deplasman ikinci yarı gol
        awaySecond.avg_goals_per_match / Math.max(1, homeSecond.avg_goals_per_match),
        homeFormFactor,                                         // Ev sahibi form faktörü
        awayFormFactor,                                         // Deplasman form faktörü
        homeMotivationFactor,                                   // Ev sahibi motivasyon
        awayMotivationFactor                                    // Deplasman motivasyon
    ];
    
    // Ağırlıklar ve bias değerleri - normalde eğitim ile bulunur
    // Giriş -> Gizli katman ağırlıkları (12x6)
    const weights1 = [
        [0.2, -0.3, 0.1, 0.5, -0.2, 0.4],
        [0.3, 0.2, -0.4, 0.1, 0.3, -0.2],
        [0.1, 0.4, 0.2, -0.3, 0.5, 0.1],
        [-0.2, 0.1, 0.3, 0.2, -0.1, 0.4],
        [0.3, -0.4, 0.1, -0.2, 0.3, 0.5],
        [0.2, 0.3, -0.2, 0.4, 0.1, -0.3],
        [0.4, 0.1, 0.3, -0.1, 0.2, 0.4],
        [-0.3, 0.2, 0.4, 0.3, -0.4, 0.1],
        [0.5, 0.3, 0.1, 0.2, 0.4, -0.2],
        [0.1, -0.3, 0.2, 0.5, 0.3, 0.1],
        [0.3, 0.2, 0.4, 0.1, -0.2, 0.3],
        [0.2, 0.4, -0.3, 0.2, 0.5, 0.1]
    ];
    
    // Gizli katman bias değerleri
    const biases1 = [0.1, 0.2, -0.1, 0.3, -0.2, 0.1];
    
    // Gizli katman -> Çıkış katmanı ağırlıkları (6x9)
    const weights2 = [
        [0.3, -0.2, 0.1, 0.4, 0.2, -0.3, 0.1, 0.3, -0.1],
        [0.2, 0.4, -0.1, 0.1, 0.3, 0.2, -0.2, 0.1, 0.4],
        [0.1, 0.2, 0.3, -0.4, 0.1, 0.5, 0.2, -0.1, 0.3],
        [0.4, -0.3, 0.2, 0.1, -0.2, 0.3, 0.4, 0.2, -0.1],
        [0.2, 0.1, 0.3, 0.2, 0.4, -0.3, 0.1, 0.5, 0.2],
        [0.3, 0.2, -0.1, 0.4, 0.1, 0.2, 0.3, -0.2, 0.4]
    ];
    
    // Çıkış katmanı bias değerleri
    const biases2 = [0.2, 0.1, 0.3, -0.1, 0.2, 0.1, -0.2, 0.3, 0.1];
    
    // İleri besleme algoritması
    
    // Gizli katman çıktıları
    const hiddenOutputs = [];
    
    // Her bir gizli nöron için çıktıyı hesapla
    for (let i = 0; i < 6; i++) {
        let sum = biases1[i];
        for (let j = 0; j < features.length; j++) {
            sum += features[j] * weights1[j][i];
        }
        // ReLU aktivasyon fonksiyonu
        hiddenOutputs.push(relu(sum));
    }
    
    // Çıkış katmanı değerleri
    const outputValues = [];
    
    // Her bir çıkış nöronu için değeri hesapla
    for (let i = 0; i < 9; i++) {
        let sum = biases2[i];
        for (let j = 0; j < hiddenOutputs.length; j++) {
            sum += hiddenOutputs[j] * weights2[j][i];
        }
        outputValues.push(sum);
    }
    
    // Softmax ile olasılıklara dönüştür
    const probabilities = softmax(outputValues);
    
    // İY/MS sonuçları için olasılıkları ata
    const result = {};
    HT_FT_COMBINATIONS.forEach((combo, index) => {
        result[combo] = Math.round(probabilities[index] * 100);
    });
    
    return result;
}

/**
 * İki modelin sonuçlarını ağırlıklı olarak birleştir
 * @param {Object} statisticalProbs - İstatistik temelli olasılıklar
 * @param {Object} monteCarloProbs - Monte Carlo simülasyonu olasılıkları
 * @param {Object} neuralNetProbs - Yapay sinir ağı olasılıkları
 * @returns {Object} - Birleştirilmiş olasılıklar
 */
function combineModelResults(statisticalProbs, monteCarloProbs, neuralNetProbs) {
    // Ağırlıklar toplamı 1 olmalı - Monte Carlo'ya ÇOK DAHA FAZLA ağırlık verelim
    const statisticalWeight = 0.1;  // İstatistik temelli model ağırlığı (%10'a düşürüldü)
    const monteCarloWeight = 0.8;   // Monte Carlo simülasyonu ağırlığı (%80'e çıkarıldı - İlk yarı etkisi AŞIRI artırıldı)
    const neuralNetWeight = 0.1;    // Yapay sinir ağı ağırlığı (%10'a düşürüldü)
    
    const combinedProbs = {};
    
    // Her bir İY/MS kombinasyonu için ağırlıklı ortalama hesapla
    HT_FT_COMBINATIONS.forEach(combo => {
        const statProb = statisticalProbs[combo] || 0;
        const mcProb = monteCarloProbs[combo] || 0;
        const nnProb = neuralNetProbs[combo] || 0;
        
        combinedProbs[combo] = Math.round(
            (statProb * statisticalWeight) +
            (mcProb * monteCarloWeight) +
            (nnProb * neuralNetWeight)
        );
    });
    
    // Olasılıkların toplamı 100 olacak şekilde normalize et
    let totalProb = 0;
    for (const combo in combinedProbs) {
        totalProb += combinedProbs[combo];
    }
    
    if (totalProb !== 100) {
        const factor = 100 / totalProb;
        for (const combo in combinedProbs) {
            combinedProbs[combo] = Math.round(combinedProbs[combo] * factor);
        }
        
        // Yuvarlamadan dolayı toplam hala 100 değilse farkı en yüksek olasılığa ekle/çıkar
        let finalTotal = 0;
        for (const combo in combinedProbs) {
            finalTotal += combinedProbs[combo];
        }
        
        if (finalTotal !== 100) {
            const diff = 100 - finalTotal;
            const highestCombo = Object.keys(combinedProbs).reduce((a, b) => 
                combinedProbs[a] > combinedProbs[b] ? a : b);
            combinedProbs[highestCombo] += diff;
        }
    }
    
    // İY/MS tahminleri hesaplandı
    console.log("İY/MS tahminleri hesaplandı (tahmin butonu ile uyumlu):", {
        prediction: Object.keys(combinedProbs).reduce((a, b) => combinedProbs[a] > combinedProbs[b] ? a : b),
        top_predictions: findTopPredictions(combinedProbs, 3),
        all_probabilities: combinedProbs
    });
    
    return combinedProbs;
}