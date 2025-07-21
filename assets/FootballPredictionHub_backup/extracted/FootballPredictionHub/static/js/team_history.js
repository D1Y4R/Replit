// Takım geçmiş maç verilerini göstermek için JavaScript fonksiyonları

// Format date function
function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString('tr-TR');
}

// Fetch team history
async function fetchTeamHistory(teamId) {
    try {
        const response = await fetch(`/api/team-matches/${teamId}`);
        const data = await response.json();
        return data.matches || [];
    } catch (error) {
        console.error('Error fetching team history:', error);
        return [];
    }
}

// Display team history
function displayTeamHistory(matches, containerSelector) {
    const container = document.querySelector(containerSelector);
    if (matches.length === 0) {
        container.innerHTML = '<div class="alert alert-info">Geçmiş maç bulunamadı.</div>';
        return;
    }

    let html = '<div class="list-group">';
    matches.forEach(match => {
        html += `
            <div class="list-group-item">
                <div class="d-flex justify-content-between align-items-start mb-1">
                    <div><small>${match.date || ''}</small></div>
                    <div><strong>${match.match || ''}</strong></div>
                </div>
                <div class="text-center">
                    <span style="font-size: 1.2rem; font-weight: bold;">${match.score || ''}</span>
                    <br>
                    <span style="font-size: 0.9rem; color: #6c757d;">(İY: ${match.half_time_score || '? - ?'})</span>
                </div>
            </div>
        `;
    });
    html += '</div>';
    
    container.innerHTML = html;
}

// Show team history modal
async function showTeamHistory(homeTeamId, awayTeamId, homeTeamName, awayTeamName) {
    const modal = new bootstrap.Modal(document.getElementById('teamHistoryModal'));
    
    // Set modal title
    document.querySelector('#teamHistoryModalLabel').textContent = `${homeTeamName} vs ${awayTeamName} - Son 5 Maç`;

    // Reset history containers
    document.querySelector('.home-team-history').innerHTML = `
        <div class="spinner-border text-primary" role="status">
            <span class="visually-hidden">Yükleniyor...</span>
        </div>`;
    document.querySelector('.away-team-history').innerHTML = `
        <div class="spinner-border text-primary" role="status">
            <span class="visually-hidden">Yükleniyor...</span>
        </div>`;

    // Show modal
    modal.show();

    // Fetch and display team histories
    const [homeTeamMatches, awayTeamMatches] = await Promise.all([
        fetchTeamHistory(homeTeamId),
        fetchTeamHistory(awayTeamId)
    ]);

    displayTeamHistory(homeTeamMatches, '.home-team-history');
    displayTeamHistory(awayTeamMatches, '.away-team-history');
}

// Document'e showTeamHistory fonksiyonunu ekle
document.showTeamHistory = showTeamHistory;
