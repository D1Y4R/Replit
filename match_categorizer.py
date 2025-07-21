"""
Maç Kategorilendirme Modülü
Maçları lig, takım profili ve maç tipine göre kategorilere ayırır
"""
import logging

logger = logging.getLogger(__name__)

class MatchCategorizer:
    """
    Maçları kategorilere ayıran sınıf
    """
    
    def __init__(self):
        # Lig kategorileri
        self.league_categories = {
            "high_scoring": [
                "Bundesliga", "Eredivisie", "MLS", "Championship", 
                "2. Bundesliga", "Jupiler Pro League", "Eliteserien",
                "Allsvenskan", "J1 League", "K League 1"
            ],
            "low_scoring": [
                "Serie A", "Ligue 1", "LaLiga", "Liga Portugal",
                "Serie B", "Ligue 2", "LaLiga 2", "Greek Super League",
                "Russian Premier League", "Ukrainian Premier League"
            ],
            "medium_scoring": [
                "Premier League", "Süper Lig", "Brasileirão", "Liga MX",
                "Primera División", "Ekstraklasa", "Czech Liga",
                "Austrian Bundesliga", "Scottish Premiership", "A-League"
            ]
        }
        
        # Sezon dönemleri (ay bazlı)
        self.season_periods = {
            "season_start": [8, 9],      # Ağustos, Eylül
            "mid_season": [10, 11, 12, 1, 2, 3],  # Ekim-Mart
            "season_end": [4, 5, 6],     # Nisan, Mayıs, Haziran
            "summer_break": [7]          # Temmuz
        }
        
    def categorize_match(self, match_info):
        """
        Maçı kategorilere ayır
        
        Args:
            match_info: Maç bilgileri
            
        Returns:
            dict: Kategori bilgileri
        """
        categories = {
            "league_category": self._get_league_category(match_info),
            "match_type": self._get_match_type(match_info),
            "team_profiles": self._get_team_profiles(match_info),
            "season_period": self._get_season_period(match_info),
            "special_conditions": self._get_special_conditions(match_info)
        }
        
        return categories
        
    def _get_league_category(self, match_info):
        """Lig kategorisini belirle"""
        league = match_info.get("league", "")
        
        # Tam eşleşme kontrolü
        for category, leagues in self.league_categories.items():
            if any(league_name.lower() in league.lower() for league_name in leagues):
                return category
                
        # Varsayılan
        return "medium_scoring"
        
    def _get_match_type(self, match_info):
        """Maç tipini belirle"""
        elo_diff = abs(match_info.get("elo_diff", 0))
        home_position = match_info.get("home_position", 10)
        away_position = match_info.get("away_position", 10)
        
        # Derbi kontrolü
        if self._is_derby(match_info):
            return "derby"
            
        # Küme düşme mücadelesi
        if home_position > 15 and away_position > 15:
            return "relegation_battle"
            
        # Şampiyonluk yarışı
        if home_position <= 3 and away_position <= 3:
            return "title_race"
            
        # Elo farkına göre
        if elo_diff > 300:
            return "heavy_favorite"
        elif elo_diff > 150:
            return "moderate_favorite"
        elif elo_diff > 50:
            return "slight_favorite"
        else:
            return "balanced"
            
    def _get_team_profiles(self, match_info):
        """Takım profillerini belirle"""
        home_stats = match_info.get("home_stats", {})
        away_stats = match_info.get("away_stats", {})
        
        profiles = {
            "home": self._determine_team_profile(home_stats),
            "away": self._determine_team_profile(away_stats)
        }
        
        return profiles
        
    def _determine_team_profile(self, team_stats):
        """Tek bir takımın profilini belirle"""
        avg_goals_scored = team_stats.get("avg_goals_scored", 1.5)
        avg_goals_conceded = team_stats.get("avg_goals_conceded", 1.5)
        
        # Ofansif takım
        if avg_goals_scored > 2.0:
            if avg_goals_conceded > 1.5:
                return "attacking_open"  # Açık oyun
            else:
                return "attacking_solid"  # Sağlam savunmalı ofansif
                
        # Defansif takım
        elif avg_goals_conceded < 1.0:
            if avg_goals_scored < 1.0:
                return "defensive_tight"  # Çok kapalı oyun
            else:
                return "defensive_counter"  # Kontra ağırlıklı
                
        # Dengeli takım
        else:
            return "balanced"
            
    def _get_season_period(self, match_info):
        """Sezon dönemini belirle"""
        import datetime
        
        match_date = match_info.get("date", datetime.datetime.now())
        if isinstance(match_date, str):
            try:
                match_date = datetime.datetime.strptime(match_date, "%Y-%m-%d")
            except:
                match_date = datetime.datetime.now()
                
        month = match_date.month
        
        for period, months in self.season_periods.items():
            if month in months:
                return period
                
        return "mid_season"
        
    def _get_special_conditions(self, match_info):
        """Özel durumları belirle"""
        conditions = []
        
        # Son hafta
        if match_info.get("is_last_week", False):
            conditions.append("last_week")
            
        # Kupa maçı
        if "cup" in match_info.get("league", "").lower():
            conditions.append("cup_match")
            
        # Avrupa kupası haftası
        if match_info.get("european_week", False):
            conditions.append("european_week")
            
        # Hava durumu
        weather = match_info.get("weather", {})
        if weather.get("rain", False):
            conditions.append("rainy")
        if weather.get("temperature", 20) < 5:
            conditions.append("cold")
        if weather.get("temperature", 20) > 30:
            conditions.append("hot")
            
        return conditions
        
    def _is_derby(self, match_info):
        """Derbi maçı kontrolü"""
        # Basit derbi kontrolü - aynı şehir veya rival takımlar
        home_team = match_info.get("home_team", "").lower()
        away_team = match_info.get("away_team", "").lower()
        
        # Bilinen derbiler
        derbies = [
            ("galatasaray", "fenerbahce"),
            ("besiktas", "galatasaray"),
            ("besiktas", "fenerbahce"),
            ("real madrid", "barcelona"),
            ("real madrid", "atletico"),
            ("manchester united", "manchester city"),
            ("liverpool", "everton"),
            ("milan", "inter"),
            ("juventus", "torino"),
            ("roma", "lazio"),
            ("boca", "river"),
            ("arsenal", "tottenham"),
            ("ajax", "feyenoord"),
            ("benfica", "sporting"),
            ("porto", "benfica"),
            ("bayern", "dortmund"),
            ("schalke", "dortmund")
        ]
        
        for team1, team2 in derbies:
            if (team1 in home_team and team2 in away_team) or \
               (team2 in home_team and team1 in away_team):
                return True
                
        return False
        
    def get_category_weights(self, categories):
        """
        Kategorilere göre önerilen model ağırlıkları
        
        Returns:
            dict: Model ağırlıkları
        """
        # Temel ağırlıklar
        weights = {
            'poisson': 0.25,
            'dixon_coles': 0.18,
            'xgboost': 0.12,
            'monte_carlo': 0.15,
            'crf': 0.15,
            'neural_network': 0.15
        }
        
        # Lig kategorisine göre ayarla
        league_cat = categories.get("league_category", "medium_scoring")
        
        if league_cat == "high_scoring":
            # Yüksek skorlu ligler için
            weights['poisson'] = 0.30
            weights['monte_carlo'] = 0.25
            weights['dixon_coles'] = 0.10
            weights['xgboost'] = 0.15
            weights['neural_network'] = 0.15
            weights['crf'] = 0.05
            
        elif league_cat == "low_scoring":
            # Düşük skorlu ligler için
            weights['dixon_coles'] = 0.35
            weights['poisson'] = 0.20
            weights['crf'] = 0.15
            weights['xgboost'] = 0.15
            weights['neural_network'] = 0.10
            weights['monte_carlo'] = 0.05
            
        # Maç tipine göre ince ayar
        match_type = categories.get("match_type", "balanced")
        
        if match_type == "derby":
            # Derbilerde belirsizlik artar
            weights['monte_carlo'] += 0.05
            weights['neural_network'] += 0.05
            weights['poisson'] -= 0.05
            weights['dixon_coles'] -= 0.05
            
        elif match_type == "relegation_battle":
            # Küme düşme mücadelesi - savunma ön planda
            weights['dixon_coles'] += 0.10
            weights['crf'] += 0.05
            weights['poisson'] -= 0.10
            weights['monte_carlo'] -= 0.05
            
        elif match_type == "heavy_favorite":
            # Ezici favori durumu
            weights['poisson'] += 0.10
            weights['xgboost'] += 0.05
            weights['monte_carlo'] -= 0.10
            weights['neural_network'] -= 0.05
            
        # Takım profillerine göre ayarla
        team_profiles = categories.get("team_profiles", {})
        home_profile = team_profiles.get("home", "balanced")
        away_profile = team_profiles.get("away", "balanced")
        
        if home_profile == "attacking_open" and away_profile == "attacking_open":
            # İki ofansif takım
            weights['poisson'] += 0.10
            weights['monte_carlo'] += 0.05
            weights['dixon_coles'] -= 0.15
            
        elif "defensive" in home_profile and "defensive" in away_profile:
            # İki defansif takım
            weights['dixon_coles'] += 0.15
            weights['crf'] += 0.05
            weights['poisson'] -= 0.10
            weights['monte_carlo'] -= 0.10
            
        # Ağırlıkları normalize et
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights