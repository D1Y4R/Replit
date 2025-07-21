// Bu script takımların ilk yarı ve ikinci yarı istatistiklerini gösterir
// Global window nesnesi için görünür olmalı
window.showTeamHalfTimeStats = function(homeId, awayId, homeName, awayName) {
    console.log("showTeamHalfTimeStats fonksiyonu çalıştırıldı: ", {homeId, awayId, homeName, awayName});
    
    // Tahmin butonu ile ilişkiyi kaldır - Sürpriz butonu artık bağımsız çalışacak
    window.globalMatchOutcome = null;
    
    // Modal'ı göster
    $('#predictionModal').modal('show');
    $('#predictionModalLabel').text(`${homeName} vs ${awayName} - İlk Yarı Performans İstatistikleri`);
    
    // Yükleniyor göster, içeriği gizle
    $('#predictionLoading').show();
    $('#predictionContent').empty().hide();  // İçeriği temizleyip gizle
    $('#predictionError').hide();
    
    if (!homeId || !awayId) {
        console.error("Takım ID'leri eksik:", {homeId, awayId});
        $('#predictionLoading').hide();
        $('#predictionContent').html(`
            <div class="alert alert-danger">
                <h4>Hata: Takım bilgileri eksik</h4>
                <p>Takım ID'leri bulunamadı. Lütfen sayfayı yenileyip tekrar deneyin.</p>
            </div>
        `).show();
        return;
    }
    
    // Tahmin verilerini kontrol et
    if (!window.predictionData || !window.predictionData.predictions) {
        console.log("Tahmin butonu verisi bulunamadı, otomatik olarak çağrılıyor...");
        // Önce normal tahmin verilerini al (arka planda)
        getBackgroundPrediction(homeId, awayId, homeName, awayName);
    } else {
        // Normal şekilde yarı istatistiklerini getir
        fetchHalfTimeStats(homeId, awayId, homeName, awayName);
    }
};

// Arka planda tahmin butonu işlevini çağır
function getBackgroundPrediction(homeId, awayId, homeName, awayName) {
    $.ajax({
        url: `/api/predict-match/${homeId}/${awayId}?home_name=${encodeURIComponent(homeName)}&away_name=${encodeURIComponent(awayName)}&force_update=false`,
        type: 'GET',
        dataType: 'json'
    })
    .done(function(data) {
        console.log("Arka planda tahmin verileri alındı:", data);
        // Global değişkene atama
        window.predictionData = data;
        // Şimdi yarı istatistiklerini getir
        fetchHalfTimeStats(homeId, awayId, homeName, awayName);
    })
    .fail(function(error) {
        console.error("Arka planda tahmin verisi alınamadı:", error);
        // Hata olsa bile yarı istatistiklerini getir
        fetchHalfTimeStats(homeId, awayId, homeName, awayName);
    });
}

// Yarı istatistiklerini çeken yardımcı fonksiyon
function fetchHalfTimeStats(homeId, awayId, homeName, awayName) {
    // Yeni Backend API'sinden İY/MS tahminlerini çekelim
    let url = `/api/v3/htft-prediction/${homeId}/${awayId}?home_name=${encodeURIComponent(homeName)}&away_name=${encodeURIComponent(awayName)}&force_update=true`;
    
    // Tahmin butonu verileri ve sürpriz butonu parametresi
    let global_outcome = {};
    
    // Konsola bilgi yazdıralım
    if (window.predictionData && window.predictionData.predictions) {
        const predictions = window.predictionData.predictions;
        
        // Maç sonucu tahmini
        const outcome = predictions.most_likely_outcome;
        
        // Beklenen gol değerleri (daha tutarlı tahminler için)
        const expectedHomeGoals = predictions.score_prediction?.beklenen_home_gol;
        const expectedAwayGoals = predictions.score_prediction?.beklenen_away_gol;
        
        console.log("Tahmin butonu sonucu:", outcome);
        console.log("Beklenen goller:", {ev: expectedHomeGoals, deplasman: expectedAwayGoals});
        
        // Tam global_outcome nesnesi oluştur
        global_outcome = {
            match_outcome: outcome,
            expected_home_goals: expectedHomeGoals,
            expected_away_goals: expectedAwayGoals,
            is_surprise_button: true  // Sürpriz butonu için özel flag
        };
        
        // Nesneyi JSON formatına çevir ve URL'ye ekle
        url += `&global_outcome=${encodeURIComponent(JSON.stringify(global_outcome))}`;
    } else {
        // Tahmin butonu verisi yoksa sadece sürpriz butonu parametresi gönder
        url += '&surprise_button=true';
    }
    
    $.ajax({
        url: url,
        type: 'GET',
        dataType: 'json'
    })
    .done(function(data) {
        console.log("İlk yarı/ikinci yarı verileri:", data);
        
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
        
        // Takımların yarı skorlarını analiz et
        const homeStats = data.home_team.stats;
        const awayStats = data.away_team.stats;
        
        // İstatistik nesnelerini güvenli şekilde al
        const hasHomeStats = homeStats && homeStats.statistics;
        const hasAwayStats = awayStats && awayStats.statistics;
        
        // Tahmin algoritması için maç sayılarını alalım
        const homeMatches = homeStats.total_matches_analyzed;
        const awayMatches = awayStats.total_matches_analyzed;
        
        console.log("İşlenmiş yarı istatistikleri:", {homeStats, awayStats});
        
        // İlk yarı sonuç dağılımlarını al - varsayılan değerleri güvenlik için ayarla
        const homeFirstHalfResults = (homeStats && homeStats.statistics && homeStats.statistics.first_half && homeStats.statistics.first_half.results) 
            ? homeStats.statistics.first_half.results 
            : { total: { "1": 0, "X": 0, "2": 0 } };
            
        const awayFirstHalfResults = (awayStats && awayStats.statistics && awayStats.statistics.first_half && awayStats.statistics.first_half.results) 
            ? awayStats.statistics.first_half.results 
            : { total: { "1": 0, "X": 0, "2": 0 } };
        
        // İY/MS kombinasyon dağılımlarını al - varsayılan değerleri güvenlik için ayarla
        const homeHtFtCombos = (homeStats && homeStats.statistics && homeStats.statistics.ht_ft_combinations) 
            ? homeStats.statistics.ht_ft_combinations 
            : { total: { "1/1": 0, "1/X": 0, "1/2": 0, "X/1": 0, "X/X": 0, "X/2": 0, "2/1": 0, "2/X": 0, "2/2": 0 } };
            
        const awayHtFtCombos = (awayStats && awayStats.statistics && awayStats.statistics.ht_ft_combinations) 
            ? awayStats.statistics.ht_ft_combinations 
            : { total: { "1/1": 0, "1/X": 0, "1/2": 0, "X/1": 0, "X/X": 0, "X/2": 0, "2/1": 0, "2/X": 0, "2/2": 0 } };
        
        // Yükleniyor göstergesini kapat, içeriği göster
        $('#predictionLoading').hide();
        $('#predictionContent').show();
        
        // Her iki takım için veri bulunamadıysa uyarı göster
        if ((homeStats.status === "Veri bulunamadı" || homeStats.matches.length === 0) && 
            (awayStats.status === "Veri bulunamadı" || awayStats.matches.length === 0)) {
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
        
        // Takımlar için sonuçları göster
        let content = `
            <div class="htft-stats-container">
                <div class="row">
                    <div class="col-md-12">
                        <div class="alert alert-info">
                            <h4>${homeName} vs ${awayName} - İlk Yarı Performans İstatistikleri</h4>
                            <p>Son 21 maçtaki ilk yarı performans analizleri ve İY/MS tahmini</p>
                        </div>
                    </div>
                </div>

                <!-- Barcelona İlk Yarı Performansı -->
                <div class="row mb-4">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header bg-primary text-white">
                                <h5>${homeName} İlk Yarı Performansı</h5>
                            </div>
                            <div class="card-body">
                                <h6>Son ${homeStats.total_matches_analyzed} maçta ilk yarı sonuçları:</h6>
                                
                                <div class="mb-3">
                                    <p><strong>Önde bitirilen ilk yarı sayısı:</strong> ${homeFirstHalfResults.total["1"]} maç (%${Math.round(homeFirstHalfResults.total["1"] * 100 / homeStats.total_matches_analyzed)})</p>
                                    <p><strong>Berabere bitirilen ilk yarı sayısı:</strong> ${homeFirstHalfResults.total["X"]} maç (%${Math.round(homeFirstHalfResults.total["X"] * 100 / homeStats.total_matches_analyzed)})</p>
                                    <p><strong>Geride bitirilen ilk yarı sayısı:</strong> ${homeFirstHalfResults.total["2"]} maç (%${Math.round(homeFirstHalfResults.total["2"] * 100 / homeStats.total_matches_analyzed)})</p>
                                </div>
                                
                                <h6>Ev sahibi ve deplasmanda ilk yarı performansı:</h6>
                                <ul>
                                    <li>Ev sahibiyken önde: ${homeFirstHalfResults.home["1"] || 0} maç</li>
                                    <li>Ev sahibiyken berabere: ${homeFirstHalfResults.home["X"] || 0} maç</li>
                                    <li>Ev sahibiyken geride: ${homeFirstHalfResults.home["2"] || 0} maç</li>
                                    <li>Deplasmanken önde: ${homeFirstHalfResults.away["1"] || 0} maç</li>
                                    <li>Deplasmanken berabere: ${homeFirstHalfResults.away["X"] || 0} maç</li>
                                    <li>Deplasmanken geride: ${homeFirstHalfResults.away["2"] || 0} maç</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header bg-danger text-white">
                                <h5>${awayName} İlk Yarı Performansı</h5>
                            </div>
                            <div class="card-body">
                                <h6>Son ${awayStats.total_matches_analyzed} maçta ilk yarı sonuçları:</h6>
                                
                                <div class="mb-3">
                                    <p><strong>Önde bitirilen ilk yarı sayısı:</strong> ${awayFirstHalfResults.total["1"]} maç (%${Math.round(awayFirstHalfResults.total["1"] * 100 / awayStats.total_matches_analyzed)})</p>
                                    <p><strong>Berabere bitirilen ilk yarı sayısı:</strong> ${awayFirstHalfResults.total["X"]} maç (%${Math.round(awayFirstHalfResults.total["X"] * 100 / awayStats.total_matches_analyzed)})</p>
                                    <p><strong>Geride bitirilen ilk yarı sayısı:</strong> ${awayFirstHalfResults.total["2"]} maç (%${Math.round(awayFirstHalfResults.total["2"] * 100 / awayStats.total_matches_analyzed)})</p>
                                </div>
                                
                                <h6>Ev sahibi ve deplasmanda ilk yarı performansı:</h6>
                                <ul>
                                    <li>Ev sahibiyken önde: ${awayFirstHalfResults.home["1"] || 0} maç</li>
                                    <li>Ev sahibiyken berabere: ${awayFirstHalfResults.home["X"] || 0} maç</li>
                                    <li>Ev sahibiyken geride: ${awayFirstHalfResults.home["2"] || 0} maç</li>
                                    <li>Deplasmanken önde: ${awayFirstHalfResults.away["1"] || 0} maç</li>
                                    <li>Deplasmanken berabere: ${awayFirstHalfResults.away["X"] || 0} maç</li>
                                    <li>Deplasmanken geride: ${awayFirstHalfResults.away["2"] || 0} maç</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
        `;
        
        // İY/MS tahminlerini göster
        if (data.htft_prediction && !data.htft_prediction.error) {
            const htftData = data.htft_prediction;
            const topPredictions = htftData.top_predictions || [];
            
            content += `
                <div class="row mb-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header bg-success text-white">
                                <h5>İlk Yarı / Maç Sonu Tahmini</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-6">
                                        <h6 class="mb-3">En Olası İY/MS Tahmini</h6>
                                        <div class="alert alert-success">
                                            <h3 class="text-center">${htftData.prediction}</h3>
                                            <p class="text-center mb-0">${getHtFtDescription(htftData.prediction)}</p>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <h6 class="mb-3">En Olası Tahminler</h6>
                                        <ul class="list-group">
                                            ${topPredictions.map(p => `
                                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                                    ${p.prediction}: ${getHtFtDescription(p.prediction)}
                                                    <span class="badge bg-primary rounded-pill">${p.probability}%</span>
                                                </li>
                                            `).join('')}
                                        </ul>
                                    </div>
                                </div>
                                
                                <div class="row mt-4">
                                    <div class="col-md-12">
                                        <h6 class="mb-3">Takımların Gerçekleşen İY/MS Kombinasyonları</h6>
                                        <div class="row">
                                            <div class="col-md-6">
                                                <div class="card">
                                                    <div class="card-header bg-primary text-white">
                                                        <h5 class="mb-0">${homeName}</h5>
                                                    </div>
                                                    <div class="card-body">
                                                        <div class="table-responsive">
                                                            <table class="table table-sm">
                                                                <thead>
                                                                    <tr>
                                                                        <th>İY/MS</th>
                                                                        <th>Maç Sayısı</th>
                                                                        <th>Yüzde</th>
                                                                    </tr>
                                                                </thead>
                                                                <tbody>
                                                                    ${Object.entries(homeHtFtCombos.total)
                                                                        .filter(([_, count]) => count > 0)
                                                                        .sort((a, b) => b[1] - a[1])
                                                                        .map(([combo, count]) => `
                                                                            <tr>
                                                                                <td><strong>${combo}</strong> - ${getHtFtDescription(combo)}</td>
                                                                                <td>${count}</td>
                                                                                <td>${homeStats.total_matches_analyzed ? Math.round(count * 100 / homeStats.total_matches_analyzed) : 0}%</td>
                                                                            </tr>
                                                                        `).join('')}
                                                                </tbody>
                                                            </table>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                            <div class="col-md-6">
                                                <div class="card">
                                                    <div class="card-header bg-danger text-white">
                                                        <h5 class="mb-0">${awayName}</h5>
                                                    </div>
                                                    <div class="card-body">
                                                        <div class="table-responsive">
                                                            <table class="table table-sm">
                                                                <thead>
                                                                    <tr>
                                                                        <th>İY/MS</th>
                                                                        <th>Maç Sayısı</th>
                                                                        <th>Yüzde</th>
                                                                    </tr>
                                                                </thead>
                                                                <tbody>
                                                                    ${Object.entries(awayHtFtCombos.total)
                                                                        .filter(([_, count]) => count > 0)
                                                                        .sort((a, b) => b[1] - a[1])
                                                                        .map(([combo, count]) => `
                                                                            <tr>
                                                                                <td><strong>${combo}</strong> - ${getHtFtDescription(combo)}</td>
                                                                                <td>${count}</td>
                                                                                <td>${awayStats.total_matches_analyzed ? Math.round(count * 100 / awayStats.total_matches_analyzed) : 0}%</td>
                                                                            </tr>
                                                                        `).join('')}
                                                                </tbody>
                                                            </table>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="row mt-3">
                                    <div class="col-md-12">
                                        <h6 class="mb-3">Tüm İY/MS Olasılıkları</h6>
                                        <div class="table-responsive">
                                            <table class="table table-striped table-bordered">
                                                <thead>
                                                    <tr>
                                                        <th>İY/MS</th>
                                                        <th>Açıklama</th>
                                                        <th>Olasılık</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    ${Object.entries(htftData.all_probabilities).map(([key, value]) => `
                                                        <tr>
                                                            <td><strong>${key}</strong></td>
                                                            <td>${getHtFtDescription(key)}</td>
                                                            <td>${value}%</td>
                                                        </tr>
                                                    `).join('')}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        } else if (data.htft_prediction && data.htft_prediction.error) {
            content += `
                <div class="row mb-4">
                    <div class="col-md-12">
                        <div class="alert alert-warning">
                            <h5>İY/MS Tahmini Yapılamadı</h5>
                            <p>${data.htft_prediction.error}</p>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Dışarı kapatma divini ekle - bu eksik div etiketini kapatır
        content += `</div>`;

        // Kodu doğru yerinde tanımla - içeriği oluşturmadan önce homeFirstHalf ve diğer değişkenleri tanımla
        
        // İlk yarı ve ikinci yarı istatistiklerini güvenli alım
        const homeFirstHalf = hasHomeStats && homeStats.statistics && homeStats.statistics.first_half ? homeStats.statistics.first_half : {};
        const homeSecondHalf = hasHomeStats && homeStats.statistics && homeStats.statistics.second_half ? homeStats.statistics.second_half : {};
        const awayFirstHalf = hasAwayStats && awayStats.statistics && awayStats.statistics.first_half ? awayStats.statistics.first_half : {};
        const awaySecondHalf = hasAwayStats && awayStats.statistics && awayStats.statistics.second_half ? awayStats.statistics.second_half : {};

        // Gol Dağılımı Kartlarını Ekle - Barcelona için
        if (hasHomeStats) {
            const totalHomeGoals = (homeFirstHalf.total_goals || 0) + (homeSecondHalf.total_goals || 0);
            const avgTotalGoals = (homeStats.total_matches_analyzed > 0) ? (totalHomeGoals / homeStats.total_matches_analyzed).toFixed(2) : 0;
            content += `
                <div class="row mb-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header bg-primary text-white">
                                <h5>${homeName} - ${homeStats.total_matches_analyzed || 0} Maç Gol Dağılımı Analizi</h5>
                            </div>
                            <div class="card-body">
                                <div class="table-responsive">
                                    <table class="table table-striped table-bordered">
                                        <thead class="thead-dark">
                                            <tr>
                                                <th>İstatistik</th>
                                                <th>İlk Yarı (0-45 dk)</th>
                                                <th>İkinci Yarı (46-90 dk)</th>
                                                <th>Toplam</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td><strong>Atılan Gol</strong></td>
                                                <td>${homeFirstHalf.total_goals || 0} (${homeFirstHalf.avg_goals_per_match || 0} maç başına)</td>
                                                <td>${homeSecondHalf.total_goals || 0} (${homeSecondHalf.avg_goals_per_match || 0} maç başına)</td>
                                                <td>${totalHomeGoals} (${avgTotalGoals} maç başına)</td>
                                            </tr>
                                            <tr>
                                                <td><strong>Ev Sahibi / Deplasman</strong></td>
                                                <td>Ev: ${homeFirstHalf.home_goals || 0} / Dep: ${homeFirstHalf.away_goals || 0}</td>
                                                <td>Ev: ${homeSecondHalf.home_goals || 0} / Dep: ${homeSecondHalf.away_goals || 0}</td>
                                                <td>Ev: ${(homeFirstHalf.home_goals || 0) + (homeSecondHalf.home_goals || 0)} / Dep: ${(homeFirstHalf.away_goals || 0) + (homeSecondHalf.away_goals || 0)}</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Ev sahibi takım istatistikleri
        if (homeStats.status !== "Veri bulunamadı" && homeStats.matches && homeStats.matches.length > 0) {
            content += `
                <div class="row mb-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header bg-primary text-white">
                                <h5>${homeName} - Son ${homeStats.total_matches_analyzed || 0} Maç İlk Yarı Performansı</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-5">
                                        <div class="alert alert-info">
                                            <h6 class="mb-3 text-center">İlk Yarı Performans Özeti</h6>
                                            <div class="d-flex justify-content-between mb-3">
                                                <div class="text-center p-2 rounded border bg-success text-white">
                                                    <strong class="d-block">Önde</strong>
                                                    <span class="d-block fs-4">${homeFirstHalfResults.total && homeFirstHalfResults.total["1"] ? homeFirstHalfResults.total["1"] : 0}</span>
                                                    <small class="text-white">${homeStats.total_matches_analyzed && homeFirstHalfResults.total && homeFirstHalfResults.total["1"] ? Math.round(homeFirstHalfResults.total["1"] * 100 / homeStats.total_matches_analyzed) : 0}%</small>
                                                </div>
                                                <div class="text-center p-2 rounded border bg-warning">
                                                    <strong class="d-block">Berabere</strong>
                                                    <span class="d-block fs-4">${homeFirstHalfResults.total && homeFirstHalfResults.total["X"] ? homeFirstHalfResults.total["X"] : 0}</span>
                                                    <small class="text-dark">${homeStats.total_matches_analyzed && homeFirstHalfResults.total && homeFirstHalfResults.total["X"] ? Math.round(homeFirstHalfResults.total["X"] * 100 / homeStats.total_matches_analyzed) : 0}%</small>
                                                </div>
                                                <div class="text-center p-2 rounded border bg-danger text-white">
                                                    <strong class="d-block">Geride</strong>
                                                    <span class="d-block fs-4">${homeFirstHalfResults.total && homeFirstHalfResults.total["2"] ? homeFirstHalfResults.total["2"] : 0}</span>
                                                    <small class="text-white">${homeStats.total_matches_analyzed && homeFirstHalfResults.total && homeFirstHalfResults.total["2"] ? Math.round(homeFirstHalfResults.total["2"] * 100 / homeStats.total_matches_analyzed) : 0}%</small>
                                                </div>
                                            </div>
                                            <div class="progress mb-2" style="height: 20px;">
                                                <div class="progress-bar bg-success" role="progressbar" 
                                                    style="width: ${homeStats.total_matches_analyzed && homeFirstHalfResults.total && homeFirstHalfResults.total["1"] ? Math.round(homeFirstHalfResults.total["1"] * 100 / homeStats.total_matches_analyzed) : 0}%" 
                                                    aria-valuenow="${homeFirstHalfResults.total && homeFirstHalfResults.total["1"] ? homeFirstHalfResults.total["1"] : 0}" 
                                                    aria-valuemin="0" 
                                                    aria-valuemax="${homeStats.total_matches_analyzed || 0}">
                                                    ${homeFirstHalfResults.total && homeFirstHalfResults.total["1"] ? homeFirstHalfResults.total["1"] : 0}
                                                </div>
                                                <div class="progress-bar bg-warning" role="progressbar" 
                                                    style="width: ${homeStats.total_matches_analyzed && homeFirstHalfResults.total && homeFirstHalfResults.total["X"] ? Math.round(homeFirstHalfResults.total["X"] * 100 / homeStats.total_matches_analyzed) : 0}%" 
                                                    aria-valuenow="${homeFirstHalfResults.total && homeFirstHalfResults.total["X"] ? homeFirstHalfResults.total["X"] : 0}" 
                                                    aria-valuemin="0" 
                                                    aria-valuemax="${homeStats.total_matches_analyzed || 0}">
                                                    ${homeFirstHalfResults.total && homeFirstHalfResults.total["X"] ? homeFirstHalfResults.total["X"] : 0}
                                                </div>
                                                <div class="progress-bar bg-danger" role="progressbar" 
                                                    style="width: ${homeStats.total_matches_analyzed && homeFirstHalfResults.total && homeFirstHalfResults.total["2"] ? Math.round(homeFirstHalfResults.total["2"] * 100 / homeStats.total_matches_analyzed) : 0}%" 
                                                    aria-valuenow="${homeFirstHalfResults.total && homeFirstHalfResults.total["2"] ? homeFirstHalfResults.total["2"] : 0}" 
                                                    aria-valuemin="0" 
                                                    aria-valuemax="${homeStats.total_matches_analyzed || 0}">
                                                    ${homeFirstHalfResults.total && homeFirstHalfResults.total["2"] ? homeFirstHalfResults.total["2"] : 0}
                                                </div>
                                            </div>
                                            <small class="d-block text-center text-muted mt-2">1 = Önde, X = Berabere, 2 = Geride</small>
                                        </div>
                                    </div>
                                    <div class="col-md-7">
                                        <table class="table table-striped table-bordered">
                                            <thead class="table-dark">
                                                <tr>
                                                    <th colspan="2" class="text-center">İlk Yarı İstatistikleri</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                <tr>
                                                    <td><strong>Toplam Gol:</strong></td>
                                                    <td>${homeFirstHalf.total_goals || 0}</td>
                                                </tr>
                                                <tr>
                                                    <td><strong>Maç Başına Gol:</strong></td>
                                                    <td>${homeFirstHalf.avg_goals_per_match || 0}</td>
                                                </tr>
                                                <tr>
                                                    <td><strong>Ev Sahibiyken Gol:</strong></td>
                                                    <td>${homeFirstHalf.home_goals || 0}</td>
                                                </tr>
                                                <tr>
                                                    <td><strong>Deplasmanken Gol:</strong></td>
                                                    <td>${homeFirstHalf.away_goals || 0}</td>
                                                </tr>
                                                <tr>
                                                    <td><strong>İlk Yarı Önde Olduğu Maçlar:</strong></td>
                                                    <td>${homeFirstHalfResults.total && homeFirstHalfResults.total["1"] ? homeFirstHalfResults.total["1"] : 0} (${homeStats.total_matches_analyzed && homeFirstHalfResults.total && homeFirstHalfResults.total["1"] ? Math.round(homeFirstHalfResults.total["1"] * 100 / homeStats.total_matches_analyzed) : 0}%)</td>
                                                </tr>
                                                <tr>
                                                    <td><strong>İlk Yarı Berabere Olduğu Maçlar:</strong></td>
                                                    <td>${homeFirstHalfResults.total && homeFirstHalfResults.total["X"] ? homeFirstHalfResults.total["X"] : 0} (${homeStats.total_matches_analyzed && homeFirstHalfResults.total && homeFirstHalfResults.total["X"] ? Math.round(homeFirstHalfResults.total["X"] * 100 / homeStats.total_matches_analyzed) : 0}%)</td>
                                                </tr>
                                                <tr>
                                                    <td><strong>İlk Yarı Geride Olduğu Maçlar:</strong></td>
                                                    <td>${homeFirstHalfResults.total && homeFirstHalfResults.total["2"] ? homeFirstHalfResults.total["2"] : 0} (${homeStats.total_matches_analyzed && homeFirstHalfResults.total && homeFirstHalfResults.total["2"] ? Math.round(homeFirstHalfResults.total["2"] * 100 / homeStats.total_matches_analyzed) : 0}%)</td>
                                                </tr>
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Deplasman takımı istatistikleri
        if (awayStats.status !== "Veri bulunamadı" && awayStats.matches && awayStats.matches.length > 0) {
            content += `
                <div class="row mb-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header bg-danger text-white">
                                <h5>${awayName} - Son ${awayStats.total_matches_analyzed || 0} Maç İlk Yarı Performansı</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-5">
                                        <div class="alert alert-info">
                                            <h6 class="mb-3 text-center">İlk Yarı Performans Özeti</h6>
                                            <div class="d-flex justify-content-between mb-3">
                                                <div class="text-center p-2 rounded border bg-success text-white">
                                                    <strong class="d-block">Önde</strong>
                                                    <span class="d-block fs-4">${awayFirstHalfResults.total && awayFirstHalfResults.total["1"] ? awayFirstHalfResults.total["1"] : 0}</span>
                                                    <small class="text-white">${awayStats.total_matches_analyzed && awayFirstHalfResults.total && awayFirstHalfResults.total["1"] ? Math.round(awayFirstHalfResults.total["1"] * 100 / awayStats.total_matches_analyzed) : 0}%</small>
                                                </div>
                                                <div class="text-center p-2 rounded border bg-warning">
                                                    <strong class="d-block">Berabere</strong>
                                                    <span class="d-block fs-4">${awayFirstHalfResults.total && awayFirstHalfResults.total["X"] ? awayFirstHalfResults.total["X"] : 0}</span>
                                                    <small class="text-dark">${awayStats.total_matches_analyzed && awayFirstHalfResults.total && awayFirstHalfResults.total["X"] ? Math.round(awayFirstHalfResults.total["X"] * 100 / awayStats.total_matches_analyzed) : 0}%</small>
                                                </div>
                                                <div class="text-center p-2 rounded border bg-danger text-white">
                                                    <strong class="d-block">Geride</strong>
                                                    <span class="d-block fs-4">${awayFirstHalfResults.total && awayFirstHalfResults.total["2"] ? awayFirstHalfResults.total["2"] : 0}</span>
                                                    <small class="text-white">${awayStats.total_matches_analyzed && awayFirstHalfResults.total && awayFirstHalfResults.total["2"] ? Math.round(awayFirstHalfResults.total["2"] * 100 / awayStats.total_matches_analyzed) : 0}%</small>
                                                </div>
                                            </div>
                                            <div class="progress mb-2" style="height: 20px;">
                                                <div class="progress-bar bg-success" role="progressbar" 
                                                    style="width: ${awayStats.total_matches_analyzed && awayFirstHalfResults.total && awayFirstHalfResults.total["1"] ? Math.round(awayFirstHalfResults.total["1"] * 100 / awayStats.total_matches_analyzed) : 0}%" 
                                                    aria-valuenow="${awayFirstHalfResults.total && awayFirstHalfResults.total["1"] ? awayFirstHalfResults.total["1"] : 0}" 
                                                    aria-valuemin="0" 
                                                    aria-valuemax="${awayStats.total_matches_analyzed || 0}">
                                                    ${awayFirstHalfResults.total && awayFirstHalfResults.total["1"] ? awayFirstHalfResults.total["1"] : 0}
                                                </div>
                                                <div class="progress-bar bg-warning" role="progressbar" 
                                                    style="width: ${awayStats.total_matches_analyzed && awayFirstHalfResults.total && awayFirstHalfResults.total["X"] ? Math.round(awayFirstHalfResults.total["X"] * 100 / awayStats.total_matches_analyzed) : 0}%" 
                                                    aria-valuenow="${awayFirstHalfResults.total && awayFirstHalfResults.total["X"] ? awayFirstHalfResults.total["X"] : 0}" 
                                                    aria-valuemin="0" 
                                                    aria-valuemax="${awayStats.total_matches_analyzed || 0}">
                                                    ${awayFirstHalfResults.total && awayFirstHalfResults.total["X"] ? awayFirstHalfResults.total["X"] : 0}
                                                </div>
                                                <div class="progress-bar bg-danger" role="progressbar" 
                                                    style="width: ${awayStats.total_matches_analyzed && awayFirstHalfResults.total && awayFirstHalfResults.total["2"] ? Math.round(awayFirstHalfResults.total["2"] * 100 / awayStats.total_matches_analyzed) : 0}%" 
                                                    aria-valuenow="${awayFirstHalfResults.total && awayFirstHalfResults.total["2"] ? awayFirstHalfResults.total["2"] : 0}" 
                                                    aria-valuemin="0" 
                                                    aria-valuemax="${awayStats.total_matches_analyzed || 0}">
                                                    ${awayFirstHalfResults.total && awayFirstHalfResults.total["2"] ? awayFirstHalfResults.total["2"] : 0}
                                                </div>
                                            </div>
                                            <small class="d-block text-center text-muted mt-2">1 = Önde, X = Berabere, 2 = Geride</small>
                                        </div>
                                    </div>
                                    <div class="col-md-7">
                                        <table class="table table-striped table-bordered">
                                            <thead class="table-dark">
                                                <tr>
                                                    <th colspan="2" class="text-center">İlk Yarı İstatistikleri</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                <tr>
                                                    <td><strong>Toplam Gol:</strong></td>
                                                    <td>${awayFirstHalf.total_goals || 0}</td>
                                                </tr>
                                                <tr>
                                                    <td><strong>Maç Başına Gol:</strong></td>
                                                    <td>${awayFirstHalf.avg_goals_per_match || 0}</td>
                                                </tr>
                                                <tr>
                                                    <td><strong>Ev Sahibiyken Gol:</strong></td>
                                                    <td>${awayFirstHalf.home_goals || 0}</td>
                                                </tr>
                                                <tr>
                                                    <td><strong>Deplasmanken Gol:</strong></td>
                                                    <td>${awayFirstHalf.away_goals || 0}</td>
                                                </tr>
                                                <tr>
                                                    <td><strong>İlk Yarı Önde Olduğu Maçlar:</strong></td>
                                                    <td>${awayFirstHalfResults.total && awayFirstHalfResults.total["1"] ? awayFirstHalfResults.total["1"] : 0} (${awayStats.total_matches_analyzed && awayFirstHalfResults.total && awayFirstHalfResults.total["1"] ? Math.round(awayFirstHalfResults.total["1"] * 100 / awayStats.total_matches_analyzed) : 0}%)</td>
                                                </tr>
                                                <tr>
                                                    <td><strong>İlk Yarı Berabere Olduğu Maçlar:</strong></td>
                                                    <td>${awayFirstHalfResults.total && awayFirstHalfResults.total["X"] ? awayFirstHalfResults.total["X"] : 0} (${awayStats.total_matches_analyzed && awayFirstHalfResults.total && awayFirstHalfResults.total["X"] ? Math.round(awayFirstHalfResults.total["X"] * 100 / awayStats.total_matches_analyzed) : 0}%)</td>
                                                </tr>
                                                <tr>
                                                    <td><strong>İlk Yarı Geride Olduğu Maçlar:</strong></td>
                                                    <td>${awayFirstHalfResults.total && awayFirstHalfResults.total["2"] ? awayFirstHalfResults.total["2"] : 0} (${awayStats.total_matches_analyzed && awayFirstHalfResults.total && awayFirstHalfResults.total["2"] ? Math.round(awayFirstHalfResults.total["2"] * 100 / awayStats.total_matches_analyzed) : 0}%)</td>
                                                </tr>
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // İstatistiklerden çıkarılabilecek analizler
        const homeFirstHalfAvg = homeFirstHalf.avg_goals_per_match || 0;
        const homeSecondHalfAvg = homeSecondHalf.avg_goals_per_match || 0;
        const awayFirstHalfAvg = awayFirstHalf.avg_goals_per_match || 0;
        const awaySecondHalfAvg = awaySecondHalf.avg_goals_per_match || 0;
        
        let analysisText = "<p>İstatistiklere dayanarak:</p><ul>";
        
        // Takımların ilk yarı durumu analizleri
        if (homeFirstHalfResults && homeFirstHalfResults.total) {
            if (homeFirstHalfResults.total["1"] > homeFirstHalfResults.total["2"]) {
                analysisText += `<li><strong>${homeName}</strong> son ${homeStats.total_matches_analyzed || 0} maçın ${homeFirstHalfResults.total["1"]} tanesinde ilk yarıyı önde tamamladı (${homeStats.total_matches_analyzed ? Math.round(homeFirstHalfResults.total["1"] * 100 / homeStats.total_matches_analyzed) : 0}%)</li>`;
            } else if (homeFirstHalfResults.total["2"] > homeFirstHalfResults.total["1"]) {
                analysisText += `<li><strong>${homeName}</strong> son ${homeStats.total_matches_analyzed || 0} maçın ${homeFirstHalfResults.total["2"]} tanesinde ilk yarıyı geride tamamladı (${homeStats.total_matches_analyzed ? Math.round(homeFirstHalfResults.total["2"] * 100 / homeStats.total_matches_analyzed) : 0}%)</li>`;
            }
            
            if (homeFirstHalfResults.total["X"] > Math.max(homeFirstHalfResults.total["1"], homeFirstHalfResults.total["2"])) {
                analysisText += `<li><strong>${homeName}</strong> son ${homeStats.total_matches_analyzed || 0} maçın ${homeFirstHalfResults.total["X"]} tanesinde ilk yarıyı berabere tamamladı (${homeStats.total_matches_analyzed ? Math.round(homeFirstHalfResults.total["X"] * 100 / homeStats.total_matches_analyzed) : 0}%)</li>`;
            }
        }
        
        if (awayFirstHalfResults && awayFirstHalfResults.total) {
            if (awayFirstHalfResults.total["1"] > awayFirstHalfResults.total["2"]) {
                analysisText += `<li><strong>${awayName}</strong> son ${awayStats.total_matches_analyzed || 0} maçın ${awayFirstHalfResults.total["1"]} tanesinde ilk yarıyı önde tamamladı (${awayStats.total_matches_analyzed ? Math.round(awayFirstHalfResults.total["1"] * 100 / awayStats.total_matches_analyzed) : 0}%)</li>`;
            } else if (awayFirstHalfResults.total["2"] > awayFirstHalfResults.total["1"]) {
                analysisText += `<li><strong>${awayName}</strong> son ${awayStats.total_matches_analyzed || 0} maçın ${awayFirstHalfResults.total["2"]} tanesinde ilk yarıyı geride tamamladı (${awayStats.total_matches_analyzed ? Math.round(awayFirstHalfResults.total["2"] * 100 / awayStats.total_matches_analyzed) : 0}%)</li>`;
            }
            
            if (awayFirstHalfResults.total["X"] > Math.max(awayFirstHalfResults.total["1"], awayFirstHalfResults.total["2"])) {
                analysisText += `<li><strong>${awayName}</strong> son ${awayStats.total_matches_analyzed || 0} maçın ${awayFirstHalfResults.total["X"]} tanesinde ilk yarıyı berabere tamamladı (${awayStats.total_matches_analyzed ? Math.round(awayFirstHalfResults.total["X"] * 100 / awayStats.total_matches_analyzed) : 0}%)</li>`;
            }
        }
        
        // Gol atma eğilimleri
        if (homeFirstHalfAvg > homeSecondHalfAvg) {
            analysisText += `<li><strong>${homeName}</strong> ilk yarıda daha fazla gol atma eğiliminde (${homeFirstHalfAvg.toFixed(2)} vs ${homeSecondHalfAvg.toFixed(2)})</li>`;
        } else if (homeFirstHalfAvg < homeSecondHalfAvg) {
            analysisText += `<li><strong>${homeName}</strong> ikinci yarıda daha fazla gol atma eğiliminde (${homeSecondHalfAvg.toFixed(2)} vs ${homeFirstHalfAvg.toFixed(2)})</li>`;
        }
        
        if (awayFirstHalfAvg > awaySecondHalfAvg) {
            analysisText += `<li><strong>${awayName}</strong> ilk yarıda daha fazla gol atma eğiliminde (${awayFirstHalfAvg.toFixed(2)} vs ${awaySecondHalfAvg.toFixed(2)})</li>`;
        } else if (awayFirstHalfAvg < awaySecondHalfAvg) {
            analysisText += `<li><strong>${awayName}</strong> ikinci yarıda daha fazla gol atma eğiliminde (${awaySecondHalfAvg.toFixed(2)} vs ${awayFirstHalfAvg.toFixed(2)})</li>`;
        }
        
        // İlk / ikinci yarılar karşılaştırma
        const totalFirstHalfAvg = (homeFirstHalfAvg + awayFirstHalfAvg) / 2;
        const totalSecondHalfAvg = (homeSecondHalfAvg + awaySecondHalfAvg) / 2;
        
        if (totalFirstHalfAvg > totalSecondHalfAvg) {
            analysisText += `<li>Bu maçta ilk yarıda daha fazla gol olma olasılığı yüksek görünüyor</li>`;
        } else if (totalFirstHalfAvg < totalSecondHalfAvg) {
            analysisText += `<li>Bu maçta ikinci yarıda daha fazla gol olma olasılığı yüksek görünüyor</li>`;
        }
        
        analysisText += "</ul>";
        
        // Analizleri ekle
        content += `
            <div class="row mt-3">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header bg-dark text-white">
                            <h5>Analiz ve Öngörüler</h5>
                        </div>
                        <div class="card-body">
                            ${analysisText}
                            <div class="alert alert-secondary mt-3">
                                <p class="mb-0"><strong>Not:</strong> Bu analizler sadece son 21 maçtaki ilk yarı (0-45 dk) ve ikinci yarı (46-90 dk) gol istatistiklerine dayanmaktadır ve garanti içermez.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // İçeriği güncelle
        $('#predictionContent').html(content);
    })
    .fail(function(error) {
        console.error("Takım yarı istatistikleri alınırken hata:", error);
        $('#predictionLoading').hide();
        $('#predictionError').hide();
        
        // Hata ne olursa olsun kullanıcı dostu bir mesaj göster
        $('#predictionContent').html(`
            <div class="htft-stats-container">
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
    });
};

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
    
    return descriptions[prediction] || formatSpecialMarket(prediction);
}

/**
 * Ek pazarlar için formatlanmış açıklama döndürür
 * Bu fonksiyon API'den gelen raw değerleri (YES, NO) 
 * kullanıcı arayüzü için formatlı değerlere (KG VAR, 2.5 ÜST vb.) dönüştürür
 */
function formatSpecialMarket(rawValue) {
    if (!rawValue) return 'Bilinmeyen Tahmin';
    
    // Eğer zaten formatlanmış bir değer ise (örn: KG VAR), direkt döndür
    if (rawValue.includes('KG VAR') || rawValue.includes('KG YOK') || 
        rawValue.includes('ÜST') || rawValue.includes('ALT')) {
        return rawValue;
    }
    
    // BTTS (Both Teams To Score) - Karşılıklı Gol
    if (rawValue === 'YES' || rawValue.toLowerCase() === 'yes') return 'KG VAR';
    if (rawValue === 'NO' || rawValue.toLowerCase() === 'no') return 'KG YOK';
    
    // Over/Under 2.5 formatları
    if (rawValue === 'OVER_2_5' || rawValue === 'OVER 2.5') return '2.5 ÜST';
    if (rawValue === 'UNDER_2_5' || rawValue === 'UNDER 2.5') return '2.5 ALT';
    
    // Over/Under 3.5 formatları
    if (rawValue === 'OVER_3_5' || rawValue === 'OVER 3.5') return '3.5 ÜST';
    if (rawValue === 'UNDER_3_5' || rawValue === 'UNDER 3.5') return '3.5 ALT';
    
    // Tahmin türlerini kontrol et
    if (rawValue.includes('OVER') && rawValue.includes('2.5')) return '2.5 ÜST';
    if (rawValue.includes('UNDER') && rawValue.includes('2.5')) return '2.5 ALT';
    if (rawValue.includes('OVER') && rawValue.includes('3.5')) return '3.5 ÜST';
    if (rawValue.includes('UNDER') && rawValue.includes('3.5')) return '3.5 ALT';
    
    // Eğer hiçbir formatlama kuralına uymuyorsa, olduğu gibi döndür
    return rawValue;
}

// Bu yardımcı fonksiyon, takım verilerinden ilk yarı ve ikinci yarı istatistiklerini işler
function processTeamHalfTimeStats(teamData, teamName, version = "v1") {
    console.log(`processTeamHalfTimeStats çağrıldı (${version}):`, teamData?.id || "bilinmeyen id");
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
    
    // Son maçların verileri
    const matches = teamData.form.detailed_data.all;
    const processedMatches = [];
    
    // İstatistik sayaçları
    let homeFirstHalfGoalsFor = 0;
    let homeFirstHalfGoalsAgainst = 0;
    let awayFirstHalfGoalsFor = 0;
    let awayFirstHalfGoalsAgainst = 0;
    
    let homeSecondHalfGoalsFor = 0;
    let homeSecondHalfGoalsAgainst = 0;
    let awaySecondHalfGoalsFor = 0;
    let awaySecondHalfGoalsAgainst = 0;
    
    // Maç sayaçları
    let homeMatchCount = 0;
    let awayMatchCount = 0;
    
    // İlk yarı sonuç sayaçları (1=önde, X=berabere, 2=geride)
    let firstHalfResults = {
        home: { "1": 0, "X": 0, "2": 0 },  // Ev sahibiyken ilk yarı sonuçları
        away: { "1": 0, "X": 0, "2": 0 },  // Deplasmanken ilk yarı sonuçları
        total: { "1": 0, "X": 0, "2": 0 }  // Toplam ilk yarı sonuçları
    };
    
    // İY/MS kombinasyon sayaçları
    let iymsCombo = {
        home: { "1/1": 0, "1/X": 0, "1/2": 0, "X/1": 0, "X/X": 0, "X/2": 0, "2/1": 0, "2/X": 0, "2/2": 0 },
        away: { "1/1": 0, "1/X": 0, "1/2": 0, "X/1": 0, "X/X": 0, "X/2": 0, "2/1": 0, "2/X": 0, "2/2": 0 },
        total: { "1/1": 0, "1/X": 0, "1/2": 0, "X/1": 0, "X/X": 0, "X/2": 0, "2/1": 0, "2/X": 0, "2/2": 0 }
    };
    
    // Her maç için istatistikleri topla
    for (const match of matches) {
        // İlk yarı gollerini al
        const htGoalsFor = match.ht_goals_scored || 0;
        const htGoalsAgainst = match.ht_goals_conceded || 0;
        
        // Toplam gollerini al
        const ftGoalsFor = match.goals_scored || 0; 
        const ftGoalsAgainst = match.goals_conceded || 0;
        
        // İkinci yarı gollerini hesapla (toplam - ilk yarı)
        const secondHalfGoalsFor = ftGoalsFor - htGoalsFor;
        const secondHalfGoalsAgainst = ftGoalsAgainst - htGoalsAgainst;
        
        // İlk yarı sonucunu belirle (1=önde, X=berabere, 2=geride)
        let htResult = "X";  // Beraberlik varsayılan değer
        if (htGoalsFor > htGoalsAgainst) {
            htResult = "1";  // Öndeyse
        } else if (htGoalsFor < htGoalsAgainst) {
            htResult = "2";  // Gerideyse
        }
        
        // Maç sonu sonucunu belirle (1=galibiyet, X=beraberlik, 2=mağlubiyet)
        let ftResult = "X";  // Beraberlik varsayılan değer
        if (ftGoalsFor > ftGoalsAgainst) {
            ftResult = "1";  // Galibiyetse
        } else if (ftGoalsFor < ftGoalsAgainst) {
            ftResult = "2";  // Mağlubiyetse
        }
        
        // İY/MS kombinasyonunu belirle
        const htftCombo = `${htResult}/${ftResult}`;
        
        // Ev sahibi/deplasman ayrımına göre istatistikleri topla
        if (match.is_home) {
            homeFirstHalfGoalsFor += htGoalsFor;
            homeFirstHalfGoalsAgainst += htGoalsAgainst;
            homeSecondHalfGoalsFor += secondHalfGoalsFor;
            homeSecondHalfGoalsAgainst += secondHalfGoalsAgainst;
            homeMatchCount++;
            
            // İlk yarı sonuç sayaçlarını güncelle
            firstHalfResults.home[htResult]++;
            firstHalfResults.total[htResult]++;
            
            // İY/MS kombinasyon sayaçlarını güncelle
            if (iymsCombo.home[htftCombo] !== undefined) {
                iymsCombo.home[htftCombo]++;
                iymsCombo.total[htftCombo]++;
            }
        } else {
            awayFirstHalfGoalsFor += htGoalsFor;
            awayFirstHalfGoalsAgainst += htGoalsAgainst;
            awaySecondHalfGoalsFor += secondHalfGoalsFor;
            awaySecondHalfGoalsAgainst += secondHalfGoalsAgainst;
            awayMatchCount++;
            
            // İlk yarı sonuç sayaçlarını güncelle
            firstHalfResults.away[htResult]++;
            firstHalfResults.total[htResult]++;
            
            // İY/MS kombinasyon sayaçlarını güncelle
            if (iymsCombo.away[htftCombo] !== undefined) {
                iymsCombo.away[htftCombo]++;
                iymsCombo.total[htftCombo]++;
            }
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
    
    // Bayesian yaklaşımı için varsayılan değerler
    // Tipik bir maçta yarı başına ortalama gol sayıları
    const priorFirstHalfGoals = 1.2; // İlk yarı için genel ortalama
    const priorSecondHalfGoals = 1.5; // İkinci yarı için genel ortalama
    
    // Dinamik ağırlıklandırma faktörü
    // Maç sayısı azaldıkça global ortalamalara daha fazla eğilim gösterir
    // 15 maçtan az verisi olan takımlar için düzeltme uygula
    const confidenceFactor = Math.min(1.0, totalMatches / 15);
    const bayesianAdjustmentFactor = 1 - confidenceFactor;
    
    // Ham ortalamalar (düzeltilmemiş)
    const rawFirstHalfGoalsFor = totalMatches > 0 ? totalFirstHalfGoalsFor / totalMatches : 0;
    const rawFirstHalfGoalsAgainst = totalMatches > 0 ? totalFirstHalfGoalsAgainst / totalMatches : 0;
    const rawSecondHalfGoalsFor = totalMatches > 0 ? totalSecondHalfGoalsFor / totalMatches : 0;
    const rawSecondHalfGoalsAgainst = totalMatches > 0 ? totalSecondHalfGoalsAgainst / totalMatches : 0;
    
    // Bayesian düzeltme uygulanmış ortalamalar
    // Maç sayısı azsa, genel futbol ortalamalarına doğru eğilimli hale getir
    const avgFirstHalfGoalsFor = totalMatches > 0 
        ? (rawFirstHalfGoalsFor * confidenceFactor + priorFirstHalfGoals * bayesianAdjustmentFactor) 
        : priorFirstHalfGoals;
    
    const avgFirstHalfGoalsAgainst = totalMatches > 0 
        ? (rawFirstHalfGoalsAgainst * confidenceFactor + priorFirstHalfGoals * bayesianAdjustmentFactor) 
        : priorFirstHalfGoals;
    
    const avgSecondHalfGoalsFor = totalMatches > 0 
        ? (rawSecondHalfGoalsFor * confidenceFactor + priorSecondHalfGoals * bayesianAdjustmentFactor) 
        : priorSecondHalfGoals;
    
    const avgSecondHalfGoalsAgainst = totalMatches > 0 
        ? (rawSecondHalfGoalsAgainst * confidenceFactor + priorSecondHalfGoals * bayesianAdjustmentFactor) 
        : priorSecondHalfGoals;
    
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
                away_goals: awayFirstHalfGoalsFor,
                // YENİ: İlk yarı sonuç dağılımı
                results: firstHalfResults
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
            },
            // YENİ: İY/MS kombinasyon dağılımı
            ht_ft_combinations: iymsCombo
        }
    };
}