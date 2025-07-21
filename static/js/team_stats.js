// Takım İstatistikleri Popup Fonksiyonları

// Takım istatistiklerini gösteren popup
function showTeamStats(homeTeamId, awayTeamId, homeTeamName, awayTeamName) {
    // Event propagation'ı durdur
    event.stopPropagation();
    
    // Popup modalı için HTML hazırla ve ekle (eğer yoksa)
    if (!document.getElementById('teamStatsModal')) {
        const modalHTML = `
        <div class="modal fade team-stats-modal" id="teamStatsModal" tabindex="-1" aria-labelledby="teamStatsModalLabel" aria-hidden="true">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="teamStatsModalLabel">Takım Detaylı İstatistikleri</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <div class="container-fluid">
                            <div class="row">
                                <div class="col-md-6">
                                    <h4 id="homeTeamName" class="text-center mb-3 text-info"></h4>
                                    <div id="homeTeamStatsLoading" class="text-center">
                                        <div class="spinner-border text-primary" role="status">
                                            <span class="visually-hidden">Yükleniyor...</span>
                                        </div>
                                        <p class="mt-2">İstatistikler yükleniyor...</p>
                                    </div>
                                    <div id="homeTeamStats" class="stats-container" style="display: none;"></div>
                                </div>
                                <div class="col-md-6">
                                    <h4 id="awayTeamName" class="text-center mb-3 text-info"></h4>
                                    <div id="awayTeamStatsLoading" class="text-center">
                                        <div class="spinner-border text-primary" role="status">
                                            <span class="visually-hidden">Yükleniyor...</span>
                                        </div>
                                        <p class="mt-2">İstatistikler yükleniyor...</p>
                                    </div>
                                    <div id="awayTeamStats" class="stats-container" style="display: none;"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>`;
        
        document.body.insertAdjacentHTML('beforeend', modalHTML);
    }
    
    // Modal elementlerini al
    const modal = new bootstrap.Modal(document.getElementById('teamStatsModal'));
    const homeTeamNameElement = document.getElementById('homeTeamName');
    const awayTeamNameElement = document.getElementById('awayTeamName');
    const homeTeamStatsLoading = document.getElementById('homeTeamStatsLoading');
    const awayTeamStatsLoading = document.getElementById('awayTeamStatsLoading');
    const homeTeamStats = document.getElementById('homeTeamStats');
    const awayTeamStats = document.getElementById('awayTeamStats');
    
    // Takım isimlerini ayarla
    homeTeamNameElement.textContent = homeTeamName;
    awayTeamNameElement.textContent = awayTeamName;
    
    // Yükleme göstergelerini göster, içerik alanlarını gizle
    homeTeamStatsLoading.style.display = 'block';
    awayTeamStatsLoading.style.display = 'block';
    homeTeamStats.style.display = 'none';
    awayTeamStats.style.display = 'none';
    
    // Modalı göster
    modal.show();
    
    // Her iki takımın istatistiklerini asenkron şekilde yükle
    Promise.all([
        fetchTeamStats(homeTeamId, homeTeamId, awayTeamId, homeTeamName, awayTeamName),
        fetchTeamStats(awayTeamId, homeTeamId, awayTeamId, homeTeamName, awayTeamName)
    ]).then(([homeStats, awayStats]) => {
        // Ev sahibi takım istatistiklerini göster
        displayTeamStats(homeStats, homeTeamStats, homeTeamStatsLoading);
        
        // Deplasman takımı istatistiklerini göster
        displayTeamStats(awayStats, awayTeamStats, awayTeamStatsLoading);
    }).catch(error => {
        console.error('Takım istatistikleri yüklenirken hata:', error);
        homeTeamStatsLoading.innerHTML = `<div class="alert alert-danger">İstatistikler yüklenemedi: ${error.message}</div>`;
        awayTeamStatsLoading.innerHTML = `<div class="alert alert-danger">İstatistikler yüklenemedi: ${error.message}</div>`;
    });
}

// Takım istatistiklerini API'den çek
async function fetchTeamStats(teamId, homeTeamId, awayTeamId, homeTeamName, awayTeamName) {
    try {
        // 1. YÖNTEM: Doğrudan tahmin API'sinden veri çekmek (çalıştığını biliyoruz)
        try {
            console.log(`Takım ${teamId} için tahmin API'sinden bilgileri alıyoruz...`);
            // Takım adını parametre olarak alıyoruz
            const teamNameParam = teamId === homeTeamId ? homeTeamName : awayTeamName;
            const predictionResponse = await fetch(`/api/predict-match/${teamId}/${teamId}?home_name=${encodeURIComponent(teamNameParam)}&away_name=${encodeURIComponent(teamNameParam)}`);
            
            if (predictionResponse.ok) {
                const predictionData = await predictionResponse.json();
                console.log("Tahmin API'sinden veri başarıyla alındı:", predictionData);
                
                // Ev sahibi takım formu verileri
                if (predictionData && predictionData.home_team && predictionData.home_team.form) {
                    // Takım adını API'den gelen veriden veya modal başlığından al
                    const teamName = predictionData.home_team.name || teamNameParam || `Takım ${teamId}`;
                    const formData = predictionData.home_team.form;
                    
                    // Takımın detaylı form verilerinden maç bilgilerini çıkar
                    if (formData.detailed_data && formData.detailed_data.all) {
                        const matches = formData.detailed_data.all;
                        const formattedMatches = [];
                        
                        matches.forEach(match => {
                            // Takım adını ve rakip takım adını birlikte göster
                            formattedMatches.push({
                                date: match.date || "",
                                match: `${match.is_home ? teamName : (match.opponent || "Rakip")} vs ${match.is_home ? (match.opponent || "Rakip") : teamName}`,
                                score: `${match.goals_scored} - ${match.goals_conceded}`
                            });
                        });
                        
                        if (formattedMatches.length > 0) {
                            console.log(`Tahmin API'sinden ${formattedMatches.length} maç verisi alındı`);
                            return formattedMatches;
                        }
                    }
                }
            }
        } catch (predictionError) {
            console.error("Tahmin API'sinden veri çekerken hata:", predictionError);
        }
        
        // 2. YÖNTEM: Takım istatistikleri API'sini kullanmak
        try {
            console.log(`Takım ${teamId} için istatistik API'sinden bilgileri alıyoruz...`);
            const statsResponse = await fetch(`/api/v3/team-stats/${teamId}`);
            
            if (statsResponse.ok) {
                const statsData = await statsResponse.json();
                if (statsData && statsData.length > 0) {
                    console.log(`İstatistik API'sinden ${statsData.length} maç verisi alındı`);
                    return statsData;
                }
            }
        } catch (statsError) {
            console.error("İstatistik API'sinden veri çekerken hata:", statsError);
        }
        
        // 3. YÖNTEM: Tahmin sayfasını yüklemeyi dene ve HTML'den veriler çıkar
        try {
            console.log(`Takım ${teamId} için HTML sayfasından bilgileri çıkarmayı deniyoruz...`);
            const response = await fetch(`/predict-match/${teamId}/9999`);
            
            if (response.ok) {
                const html = await response.text();
                const teamName = extractTeamNameFromHTML(html, teamId);
                console.log(`HTML'den takım adı: ${teamName}`);
                
                // Takım adını bulduysak minimum veriler oluştur
                const teamMatches = [];
                const today = new Date();
                
                for (let i = 0; i < 5; i++) {
                    const pastDate = new Date(today);
                    pastDate.setDate(today.getDate() - (i * 7));
                    const dateStr = pastDate.toISOString().split('T')[0];
                    
                    teamMatches.push({
                        date: dateStr,
                        match: `${teamName} - Son ${i+1} Maç`,
                        score: "Sonuç bilgisi bulunamadı"
                    });
                }
                
                console.log(`HTML sayfasından ${teamMatches.length} geçici maç verisi oluşturuldu`);
                return teamMatches;
            }
        } catch (htmlError) {
            console.error("HTML sayfasından veri çıkarmada hata:", htmlError);
        }
        
        // HTML'den takım adını çıkarma fonksiyonu
        function extractTeamNameFromHTML(html, teamId) {
            try {
                const parser = new DOMParser();
                const doc = parser.parseFromString(html, "text/html");
                
                // Başlıktan takım adını çıkarmaya çalış
                const titleElement = doc.querySelector('title');
                if (titleElement && titleElement.textContent) {
                    const titleText = titleElement.textContent;
                    
                    // Başlık genellikle "Ev Sahibi vs Deplasman" formatındadır
                    const vsIndex = titleText.indexOf(' vs ');
                    if (vsIndex > 0) {
                        return titleText.substring(0, vsIndex).trim();
                    }
                }
                
                // H4 elementlerinden takım adını çıkarmaya çalış
                const h4Elements = doc.querySelectorAll('h4');
                for (let h4 of h4Elements) {
                    if (h4.textContent && h4.textContent.trim() !== '') {
                        return h4.textContent.trim();
                    }
                }
                
                // Varsayılan takım adı
                return `Takım ${teamId}`;
            } catch (e) {
                console.error("HTML'den takım adı çıkarmada hata:", e);
                return `Takım ${teamId}`;
            }
        }
        
        // Orijinal API ile deneyelim
        try {
            const response = await fetch(`/api/v3/fixtures/team/${teamId}`);
            if (response.ok) {
                const data = await response.json();
                if (data && data.length > 0) {
                    return data;
                }
            }
        } catch (apiError) {
            console.error("Birincil API ile veri alınamadı:", apiError);
        }
        
        // Yedek API ile deneyelim
        try {
            // Takım adını parametre olarak gönderelim
            const teamNameParam = teamId === homeTeamId ? homeTeamName : awayTeamName;
            const backupResponse = await fetch(`/api/team-matches/${teamId}?team_name=${encodeURIComponent(teamNameParam)}&stats=true`);
            if (backupResponse.ok) {
                const backupData = await backupResponse.json();
                if (backupData && backupData.matches && Array.isArray(backupData.matches)) {
                    const teamMatches = [];
                    backupData.matches.forEach(match => {
                        teamMatches.push({
                            date: match.date || '',
                            match: match.match || '',
                            score: match.score || ''
                        });
                    });
                    return teamMatches;
                }
            }
        } catch (backupError) {
            console.error(`Yedek API ile takım verileri alınamadı:`, backupError);
        }
        
        // Tüm API'ler başarısız olursa minimal veri döndür
        const teamMatches = [];
        const today = new Date();
        
        for (let i = 0; i < 3; i++) {
            const pastDate = new Date(today);
            pastDate.setDate(today.getDate() - (i * 7));
            const dateStr = pastDate.toISOString().split('T')[0];
            
            teamMatches.push({
                date: dateStr,
                match: `Takım ${teamId} - Maç bilgisi`,
                score: "API'den veri alınamadı"
            });
        }
        
        return teamMatches;
    } catch (error) {
        console.error(`Takım (${teamId}) istatistikleri alınırken hata:`, error);
        
        // Minimal veri döndür
        return [
            {
                date: new Date().toISOString().split('T')[0],
                match: `Takım ${teamId} - Veri bulunamadı`,
                score: "Hata oluştu"
            }
        ];
    }
}

// Takım istatistiklerini göster
function displayTeamStats(stats, container, loadingElement) {
    // Her durumda bir şeyler göster
    loadingElement.style.display = 'none';
    container.style.display = 'block';
    
    if (!stats || !stats.length) {
        // Veri yoksa bilgi ver
        container.innerHTML = `
            <div class="alert alert-warning bg-dark text-warning border-dark">
                <p>Bu takım için istatistik bulunamadı.</p>
                <p>Olası nedenler:</p>
                <ul>
                    <li>Takım son dönemde maç oynamamış olabilir</li>
                    <li>API veritabanında takım bilgisi eksik olabilir</li>
                    <li>Takım ID'si API ile uyumlu değil</li>
                </ul>
                <p>Farklı bir takım seçmeyi deneyin.</p>
            </div>`;
        return;
    }
    
    // En az bazı maçlar gösterilebiliyorsa, HTML oluştur
    let html = '<div class="list-group bg-dark">';
    stats.forEach(match => {
        html += `
            <div class="list-group-item bg-dark text-light border-secondary">
                <div class="d-flex justify-content-between align-items-center mb-1">
                    <div><small class="text-info">${match.date || ''}</small></div>
                </div>
                <div class="text-center">
                    <span class="match-teams text-light">${match.match || ''}</span>
                    <br>
                    <strong class="match-score text-warning">${match.score || ''}</strong>
                </div>
            </div>
        `;
    });
    html += '</div>';
    
    container.innerHTML = html;
}

// Global scope'a ekle
window.showTeamStats = showTeamStats;