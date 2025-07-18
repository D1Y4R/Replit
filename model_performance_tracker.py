"""
Model Performans Takip Sistemi
Her modelin başarı oranlarını takip eder ve raporlar
"""
import json
import os
from datetime import datetime
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class ModelPerformanceTracker:
    """
    Model performanslarını takip eden sınıf
    """
    
    def __init__(self):
        self.performance_file = "performance_metrics.json"
        self.performance_data = self._load_performance_data()
        
    def _load_performance_data(self):
        """Performans verilerini yükle"""
        if os.path.exists(self.performance_file):
            try:
                with open(self.performance_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                logger.warning("Performans verileri yüklenemedi, yeni dosya oluşturuluyor")
        
        # Varsayılan yapı
        return {
            "models": {
                "poisson": self._create_model_metrics(),
                "dixon_coles": self._create_model_metrics(),
                "xgboost": self._create_model_metrics(),
                "monte_carlo": self._create_model_metrics(),
                "crf": self._create_model_metrics(),
                "neural_network": self._create_model_metrics()
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def _create_model_metrics(self):
        """Boş model metriklerini oluştur"""
        return {
            "overall": {
                "predictions": 0,
                "correct": 0,
                "accuracy": 0.0
            },
            "by_league": defaultdict(lambda: {"predictions": 0, "correct": 0, "accuracy": 0.0}),
            "by_match_type": defaultdict(lambda: {"predictions": 0, "correct": 0, "accuracy": 0.0}),
            "by_prediction_type": {
                "match_result": {"predictions": 0, "correct": 0, "accuracy": 0.0},
                "btts": {"predictions": 0, "correct": 0, "accuracy": 0.0},
                "over_under": {"predictions": 0, "correct": 0, "accuracy": 0.0}
            }
        }
    
    def track_prediction(self, model_name, prediction_data, actual_result, match_info):
        """
        Tahmin sonucunu kaydet
        
        Args:
            model_name: Model adı
            prediction_data: Tahmin verileri
            actual_result: Gerçek sonuç
            match_info: Maç bilgileri (lig, takımlar vb.)
        """
        if model_name not in self.performance_data["models"]:
            self.performance_data["models"][model_name] = self._create_model_metrics()
        
        model_metrics = self.performance_data["models"][model_name]
        
        # Genel performans
        model_metrics["overall"]["predictions"] += 1
        
        # Maç sonucu doğruluğu
        if self._check_match_result_accuracy(prediction_data, actual_result):
            model_metrics["overall"]["correct"] += 1
            
        # Lig bazlı performans
        league = match_info.get("league", "unknown")
        if league not in model_metrics["by_league"]:
            model_metrics["by_league"][league] = {"predictions": 0, "correct": 0, "accuracy": 0.0}
            
        model_metrics["by_league"][league]["predictions"] += 1
        
        # Maç tipi bazlı performans
        match_type = self._determine_match_type(match_info)
        if match_type not in model_metrics["by_match_type"]:
            model_metrics["by_match_type"][match_type] = {"predictions": 0, "correct": 0, "accuracy": 0.0}
            
        model_metrics["by_match_type"][match_type]["predictions"] += 1
        
        # Doğruluk oranlarını güncelle
        self._update_accuracy_rates(model_name)
        
        # Kaydet
        self._save_performance_data()
        
    def _check_match_result_accuracy(self, prediction, actual):
        """Maç sonucu tahmininin doğruluğunu kontrol et"""
        predicted_outcome = prediction.get("most_likely_outcome", "")
        actual_outcome = self._determine_actual_outcome(actual)
        
        return predicted_outcome == actual_outcome
        
    def _determine_actual_outcome(self, result):
        """Gerçek maç sonucunu belirle"""
        home_goals = result.get("home_goals", 0)
        away_goals = result.get("away_goals", 0)
        
        if home_goals > away_goals:
            return "HOME_WIN"
        elif home_goals < away_goals:
            return "AWAY_WIN"
        else:
            return "DRAW"
            
    def _determine_match_type(self, match_info):
        """Maç tipini belirle"""
        # Elo farkına göre
        elo_diff = abs(match_info.get("elo_diff", 0))
        
        if elo_diff > 300:
            return "heavy_favorite"
        elif elo_diff > 150:
            return "favorite"
        elif elo_diff > 50:
            return "slight_favorite"
        else:
            return "balanced"
            
    def _update_accuracy_rates(self, model_name):
        """Doğruluk oranlarını güncelle"""
        model_metrics = self.performance_data["models"][model_name]
        
        # Genel doğruluk
        if model_metrics["overall"]["predictions"] > 0:
            model_metrics["overall"]["accuracy"] = (
                model_metrics["overall"]["correct"] / 
                model_metrics["overall"]["predictions"]
            ) * 100
            
        # Lig bazlı doğruluk
        for league, stats in model_metrics["by_league"].items():
            if stats["predictions"] > 0:
                stats["accuracy"] = (stats["correct"] / stats["predictions"]) * 100
                
        # Maç tipi bazlı doğruluk
        for match_type, stats in model_metrics["by_match_type"].items():
            if stats["predictions"] > 0:
                stats["accuracy"] = (stats["correct"] / stats["predictions"]) * 100
    
    def get_model_performance(self, model_name):
        """Belirli bir modelin performansını getir"""
        return self.performance_data["models"].get(model_name, None)
        
    def get_best_model_for_league(self, league):
        """Belirli bir lig için en iyi modeli bul"""
        best_model = None
        best_accuracy = 0
        
        for model_name, metrics in self.performance_data["models"].items():
            if league in metrics["by_league"]:
                accuracy = metrics["by_league"][league]["accuracy"]
                if accuracy > best_accuracy and metrics["by_league"][league]["predictions"] > 10:
                    best_accuracy = accuracy
                    best_model = model_name
                    
        return best_model, best_accuracy
        
    def get_best_model_for_match_type(self, match_type):
        """Belirli bir maç tipi için en iyi modeli bul"""
        best_model = None
        best_accuracy = 0
        
        for model_name, metrics in self.performance_data["models"].items():
            if match_type in metrics["by_match_type"]:
                accuracy = metrics["by_match_type"][match_type]["accuracy"]
                if accuracy > best_accuracy and metrics["by_match_type"][match_type]["predictions"] > 10:
                    best_accuracy = accuracy
                    best_model = model_name
                    
        return best_model, best_accuracy
        
    def get_performance_factors(self, model_name, league=None, match_type=None):
        """
        Model için performans faktörlerini hesapla
        
        Returns:
            float: 0.7 - 1.3 arası performans faktörü
        """
        model_metrics = self.performance_data["models"].get(model_name, None)
        if not model_metrics:
            return 1.0
            
        factors = []
        
        # Genel performans faktörü
        overall_accuracy = model_metrics["overall"]["accuracy"]
        if model_metrics["overall"]["predictions"] > 20:
            # 50% doğruluk = 1.0 faktör, 70% = 1.3, 30% = 0.7
            overall_factor = 0.7 + (overall_accuracy - 30) * 0.0075
            factors.append(overall_factor)
            
        # Lig bazlı faktör
        if league and league in model_metrics["by_league"]:
            league_stats = model_metrics["by_league"][league]
            if league_stats["predictions"] > 5:
                league_factor = 0.7 + (league_stats["accuracy"] - 30) * 0.0075
                factors.append(league_factor)
                
        # Maç tipi faktörü
        if match_type and match_type in model_metrics["by_match_type"]:
            type_stats = model_metrics["by_match_type"][match_type]
            if type_stats["predictions"] > 5:
                type_factor = 0.7 + (type_stats["accuracy"] - 30) * 0.0075
                factors.append(type_factor)
                
        # Faktörlerin ortalamasını al
        if factors:
            avg_factor = sum(factors) / len(factors)
            # 0.7 - 1.3 arasında sınırla
            return max(0.7, min(1.3, avg_factor))
        else:
            return 1.0
            
    def _save_performance_data(self):
        """Performans verilerini kaydet"""
        self.performance_data["last_updated"] = datetime.now().isoformat()
        
        try:
            with open(self.performance_file, 'w', encoding='utf-8') as f:
                json.dump(self.performance_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Performans verileri kaydedilemedi: {e}")
            
    def generate_performance_report(self):
        """Performans raporu oluştur"""
        report = []
        report.append("=== MODEL PERFORMANS RAPORU ===\n")
        
        for model_name, metrics in self.performance_data["models"].items():
            report.append(f"\n{model_name.upper()} Modeli:")
            report.append(f"  Genel Doğruluk: %{metrics['overall']['accuracy']:.1f}")
            report.append(f"  Toplam Tahmin: {metrics['overall']['predictions']}")
            
            # En iyi performans gösterdiği lig
            best_league = max(
                metrics["by_league"].items(), 
                key=lambda x: x[1]["accuracy"] if x[1]["predictions"] > 5 else 0,
                default=(None, None)
            )
            if best_league[0]:
                report.append(f"  En İyi Lig: {best_league[0]} (%{best_league[1]['accuracy']:.1f})")
                
        return "\n".join(report)