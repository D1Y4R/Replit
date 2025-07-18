"""
Dinamik Ağırlık Hesaplama Motoru
Model performansı ve maç kategorilerine göre dinamik ağırlıklar hesaplar
"""
import logging
from model_performance_tracker import ModelPerformanceTracker
from match_categorizer import MatchCategorizer

logger = logging.getLogger(__name__)

class DynamicWeightCalculator:
    """
    Dinamik model ağırlıklarını hesaplayan sınıf
    """
    
    def __init__(self):
        self.performance_tracker = ModelPerformanceTracker()
        self.match_categorizer = MatchCategorizer()
        
        # Temel ağırlıklar (güvenli minimum)
        self.base_weights = {
            'poisson': 0.25,
            'dixon_coles': 0.18,
            'xgboost': 0.12,
            'monte_carlo': 0.15,
            'crf': 0.15,
            'neural_network': 0.15
        }
        
        # Maksimum sapma limiti
        self.max_deviation = 0.30  # %30
        
    def calculate_weights(self, match_info):
        """
        Maç için dinamik ağırlıkları hesapla
        
        Args:
            match_info: Maç bilgileri
            
        Returns:
            dict: Model ağırlıkları
        """
        # Maçı kategorize et
        categories = self.match_categorizer.categorize_match(match_info)
        
        # Kategori bazlı önerilen ağırlıkları al
        category_weights = self.match_categorizer.get_category_weights(categories)
        
        # Her model için performans faktörlerini hesapla
        performance_adjusted_weights = self._apply_performance_factors(
            category_weights, 
            match_info, 
            categories
        )
        
        # Bağlam faktörlerini uygula
        context_adjusted_weights = self._apply_context_factors(
            performance_adjusted_weights,
            match_info,
            categories
        )
        
        # Maksimum sapma kontrolü
        final_weights = self._apply_deviation_limits(context_adjusted_weights)
        
        # Normalize et
        total_weight = sum(final_weights.values())
        final_weights = {k: v/total_weight for k, v in final_weights.items()}
        
        # Log
        logger.info(f"Dinamik ağırlıklar hesaplandı:")
        logger.info(f"  Lig: {match_info.get('league', 'Unknown')} ({categories['league_category']})")
        logger.info(f"  Maç tipi: {categories['match_type']}")
        logger.info(f"  Final ağırlıklar: {final_weights}")
        
        return final_weights
        
    def _apply_performance_factors(self, weights, match_info, categories):
        """
        Model performans faktörlerini uygula
        
        Returns:
            dict: Performans ayarlı ağırlıklar
        """
        adjusted_weights = weights.copy()
        
        league = match_info.get("league", None)
        match_type = categories.get("match_type", None)
        
        for model_name in adjusted_weights:
            # Performans faktörünü al (0.7 - 1.3 arası)
            perf_factor = self.performance_tracker.get_performance_factors(
                model_name,
                league=league,
                match_type=match_type
            )
            
            # Ağırlığı ayarla
            adjusted_weights[model_name] *= perf_factor
            
            logger.debug(f"{model_name} performans faktörü: {perf_factor:.2f}")
            
        return adjusted_weights
        
    def _apply_context_factors(self, weights, match_info, categories):
        """
        Bağlamsal faktörleri uygula
        
        Returns:
            dict: Bağlam ayarlı ağırlıklar
        """
        adjusted_weights = weights.copy()
        
        # Sezon dönemi faktörü
        season_period = categories.get("season_period", "mid_season")
        
        if season_period == "season_start":
            # Sezon başında veri az, ML modelleri zayıf
            adjusted_weights['poisson'] *= 1.15
            adjusted_weights['dixon_coles'] *= 1.10
            adjusted_weights['xgboost'] *= 0.85
            adjusted_weights['neural_network'] *= 0.90
            
        elif season_period == "season_end":
            # Sezon sonu, motivasyon faktörleri
            adjusted_weights['monte_carlo'] *= 1.20
            adjusted_weights['neural_network'] *= 1.10
            adjusted_weights['poisson'] *= 0.90
            
        # Özel durumlar
        special_conditions = categories.get("special_conditions", [])
        
        if "cup_match" in special_conditions:
            # Kupa maçları daha belirsiz
            adjusted_weights['monte_carlo'] *= 1.15
            adjusted_weights['neural_network'] *= 1.10
            adjusted_weights['poisson'] *= 0.90
            adjusted_weights['dixon_coles'] *= 0.85
            
        if "rainy" in special_conditions:
            # Yağmurlu havada düşük skor eğilimi
            adjusted_weights['dixon_coles'] *= 1.20
            adjusted_weights['crf'] *= 1.10
            adjusted_weights['poisson'] *= 0.90
            adjusted_weights['monte_carlo'] *= 0.80
            
        # Son N maç formu
        home_form = self._calculate_recent_form(match_info.get("home_stats", {}))
        away_form = self._calculate_recent_form(match_info.get("away_stats", {}))
        
        if abs(home_form - away_form) > 0.5:
            # Form farkı yüksek
            adjusted_weights['xgboost'] *= 1.15
            adjusted_weights['neural_network'] *= 1.10
            
        return adjusted_weights
        
    def _apply_deviation_limits(self, weights):
        """
        Maksimum sapma limitlerini uygula
        
        Returns:
            dict: Limitli ağırlıklar
        """
        limited_weights = {}
        
        for model_name, weight in weights.items():
            base_weight = self.base_weights.get(model_name, 0.15)
            
            # Maksimum ve minimum limitleri hesapla
            max_weight = base_weight * (1 + self.max_deviation)
            min_weight = base_weight * (1 - self.max_deviation)
            
            # Limitle
            limited_weights[model_name] = max(min_weight, min(max_weight, weight))
            
            if weight != limited_weights[model_name]:
                logger.debug(f"{model_name} ağırlığı limitlendi: {weight:.3f} -> {limited_weights[model_name]:.3f}")
                
        return limited_weights
        
    def _calculate_recent_form(self, team_stats):
        """
        Son maçlardan form değeri hesapla
        
        Returns:
            float: 0-1 arası form değeri
        """
        recent_matches = team_stats.get("recent_matches", [])
        if not recent_matches:
            return 0.5
            
        # Son 5 maçı al
        last_5 = recent_matches[:5]
        
        points = 0
        for match in last_5:
            result = match.get("result", "D")
            if result == "W":
                points += 3
            elif result == "D":
                points += 1
                
        # 0-1 arası normalize et (15 puan maksimum)
        return points / 15.0
        
    def get_weight_explanation(self, final_weights, categories):
        """
        Ağırlık dağılımının açıklamasını oluştur
        
        Returns:
            str: Açıklama metni
        """
        explanation = []
        
        # En yüksek ağırlığa sahip model
        top_model = max(final_weights, key=final_weights.get)
        top_weight = final_weights[top_model]
        
        explanation.append(f"Bu maç için {top_model.upper()} modeli öne çıkıyor (%{top_weight*100:.0f}).")
        
        # Lig kategorisi açıklaması
        league_cat = categories.get("league_category", "medium_scoring")
        if league_cat == "high_scoring":
            explanation.append("Yüksek skorlu bir ligde oynandığı için gol odaklı modeller ağırlıkta.")
        elif league_cat == "low_scoring":
            explanation.append("Düşük skorlu bir ligde oynandığı için savunma odaklı modeller tercih edildi.")
            
        # Maç tipi açıklaması
        match_type = categories.get("match_type", "balanced")
        if match_type == "derby":
            explanation.append("Derbi maçı olduğu için belirsizlik modelleri güçlendirildi.")
        elif match_type == "heavy_favorite":
            explanation.append("Ezici favori durumu olduğu için istatistiksel modeller öne çıktı.")
            
        return " ".join(explanation)