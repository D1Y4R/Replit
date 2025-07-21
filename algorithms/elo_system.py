"""
Elo Rating Sistemi
Takım güçlerini dinamik olarak hesaplar
"""
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EloSystem:
    """
    Futbol için özelleştirilmiş Elo rating sistemi
    """
    
    def __init__(self, initial_rating=1500, k_factor=30):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.decay_factor = 0.95  # Eski maçlar için
        self.ratings = {}  # Takım ID -> Elo rating
        
    def get_expected_score(self, rating_a, rating_b):
        """
        Beklenen skor hesapla (0-1 arası)
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
        
    def update_rating(self, team_id, opponent_rating, actual_score, match_age_days=0):
        """
        Maç sonucuna göre Elo rating güncelle
        
        Args:
            team_id: Takım ID
            opponent_rating: Rakip Elo puanı
            actual_score: Gerçek sonuç (1=galibiyet, 0.5=beraberlik, 0=mağlubiyet)
            match_age_days: Maçın kaç gün önce olduğu
            
        Returns:
            float: Yeni Elo rating
        """
        current_rating = self.ratings.get(team_id, self.initial_rating)
        expected_score = self.get_expected_score(current_rating, opponent_rating)
        
        # Zaman bazlı decay
        time_weight = self.decay_factor ** (match_age_days / 30)  # Ayda %5 azalma
        
        # Elo güncelleme formülü
        rating_change = self.k_factor * (actual_score - expected_score) * time_weight
        new_rating = current_rating + rating_change
        
        self.ratings[team_id] = new_rating
        logger.debug(f"Elo güncellendi - Takım: {team_id}, Eski: {current_rating:.0f}, Yeni: {new_rating:.0f}")
        
        return new_rating
        
    def calculate_team_elo(self, team_id, matches):
        """
        Takımın son maçlarına göre Elo hesapla
        
        Args:
            team_id: Takım ID
            matches: Maç listesi (en yeni önce)
            
        Returns:
            float: Güncel Elo rating
        """
        if not matches:
            return self.initial_rating
            
        # Son 120 gündeki maçları filtrele
        today = datetime.now()
        cutoff = today - timedelta(days=120)
        filtered_matches = []
        
        for match in matches:
            try:
                match_date = datetime.strptime(match.get('date', ''), '%Y-%m-%d')
                if match_date >= cutoff:
                    filtered_matches.append(match)
            except:
                # Tarih parse edilemezse dahil et
                filtered_matches.append(match)
                
        if not filtered_matches:
            logger.warning(f"Takım {team_id} için son 120 günde maç bulunamadı")
            return self.initial_rating
            
        # Başlangıç Elo
        self.ratings[team_id] = self.initial_rating
        
        # Maçları tersten işle (eskiden yeniye)
        for i, match in enumerate(reversed(filtered_matches)):
            # Maç sonucu
            goals_for = match.get('goals_scored', 0)
            goals_against = match.get('goals_conceded', 0)
            
            if goals_for > goals_against:
                actual_score = 1.0  # Galibiyet
            elif goals_for == goals_against:
                actual_score = 0.5  # Beraberlik
            else:
                actual_score = 0.0  # Mağlubiyet
                
            # Rakip Elo - gol farkına göre tahmin et
            goal_diff = abs(goals_for - goals_against)
            if goal_diff >= 3:
                # Büyük farkla yenilgi/galibiyet - güçlü/zayıf rakip
                opponent_rating = self.initial_rating + (150 if goals_against > goals_for else -150)
            elif goal_diff == 2:
                opponent_rating = self.initial_rating + (100 if goals_against > goals_for else -100)
            elif goal_diff == 1:
                opponent_rating = self.initial_rating + (50 if goals_against > goals_for else -50)
            else:  # Beraberlik
                opponent_rating = self.initial_rating
            
            # Maç yaşı (varsayılan 0)
            match_age_days = 0
            if 'date' in match:
                try:
                    match_date = datetime.strptime(match['date'], '%Y-%m-%d')
                    match_age_days = (datetime.now() - match_date).days
                except:
                    pass
                    
            # Elo güncelle
            self.update_rating(team_id, opponent_rating, actual_score, match_age_days)
            
        logger.info(f"Takım {team_id} için Elo hesaplandı: {self.ratings[team_id]:.0f} ({len(filtered_matches)} maç, son 120 gün)")
        return self.ratings[team_id]
        
    def get_elo_difference(self, home_id, away_id):
        """
        İki takım arasındaki Elo farkını hesapla
        
        Returns:
            float: home_elo - away_elo
        """
        home_elo = self.ratings.get(home_id, self.initial_rating)
        away_elo = self.ratings.get(away_id, self.initial_rating)
        return home_elo - away_elo