"""
Açıklanabilir AI (XAI) Modülü
SHAP değerleri, özellik önemi ve tahmin açıklamaları
"""
import numpy as np
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# SHAP kütüphanesi opsiyonel
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    logger.warning("SHAP kütüphanesi bulunamadı, basit açıklamalar kullanılacak")
    SHAP_AVAILABLE = False

class PredictionExplainer:
    """
    Tahmin açıklama ve yorumlama sistemi
    """
    
    def __init__(self):
        self.feature_names = [
            'home_xg', 'away_xg', 'home_xga', 'away_xga', 
            'elo_diff', 'home_form', 'away_form',
            'home_avg_goals', 'away_avg_goals',
            'home_avg_conceded', 'away_avg_conceded',
            'h2h_home_wins', 'h2h_draws', 'h2h_away_wins',
            'home_advantage', 'form_momentum', 'goal_trend',
            'defensive_stability', 'match_importance', 'team_confidence'
        ]
        
        self.explainers = {}
        self.explanation_templates = self._load_explanation_templates()
        
    def _load_explanation_templates(self):
        """Açıklama şablonlarını yükle"""
        return {
            'high_home_xg': "Ev sahibi takım son maçlarda yüksek gol beklentisi ({value:.2f} xG) gösteriyor.",
            'low_away_xga': "Deplasman takımının defansı zayıf ({value:.2f} xGA), bu ev sahibine avantaj sağlıyor.",
            'positive_elo_diff': "Ev sahibi takım {value:.0f} puan daha yüksek Elo reytingine sahip, bu güç farkını gösteriyor.",
            'negative_elo_diff': "Deplasman takımı {value:.0f} puan daha yüksek Elo reytingine sahip, favori durumunda.",
            'high_home_form': "Ev sahibi mükemmel formda ({value:.2f}/3.0), son maçlarda istikrarlı performans.",
            'low_away_form': "Deplasman takımı kötü formda ({value:.2f}/3.0), son maçlarda düşük performans.",
            'home_advantage': "Ev sahibi avantajı bu maçta {value:.1%} ek kazanma şansı sağlıyor.",
            'h2h_dominance': "Tarihsel üstünlük: {team} takımı son karşılaşmaların {value:.0%}'ini kazanmış.",
            'high_scoring_expected': "Her iki takım da yüksek gol ortalamasına sahip, {value:.1f} toplam gol bekleniyor.",
            'defensive_match': "Her iki takım da defansif oynuyor, düşük skorlu maç bekleniyor ({value:.1f} gol).",
            'momentum_shift': "{team} takımı pozitif momentum yakalamış, form grafiği yükselişte.",
            'pressure_situation': "Kritik maç! {reason} nedeniyle yüksek baskı altında oynanacak.",
            'confidence_high': "{team} takımı yüksek özgüvenle ({value:.0%}) sahaya çıkacak.",
            'tactical_advantage': "{formation1} vs {formation2} eşleşmesinde {team} taktiksel avantaja sahip."
        }
        
    def explain_prediction(self, prediction_data, model=None, features=None):
        """
        Tahmin açıklaması oluştur
        
        Args:
            prediction_data: Tahmin sonuçları
            model: Kullanılan model (SHAP için)
            features: Özellik vektörü (SHAP için)
            
        Returns:
            dict: Açıklama detayları
        """
        explanation = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction_data.get('most_likely_outcome'),
            'confidence': prediction_data.get('confidence', 0),
            'key_factors': [],
            'detailed_analysis': {},
            'natural_language_explanation': "",
            'visualizations': []
        }
        
        # SHAP analizi (eğer mümkünse)
        if SHAP_AVAILABLE and model is not None and features is not None:
            shap_analysis = self._calculate_shap_values(model, features)
            explanation['shap_analysis'] = shap_analysis
            explanation['key_factors'].extend(self._extract_top_shap_factors(shap_analysis))
        
        # Kural tabanlı analiz
        rule_based_factors = self._analyze_rule_based_factors(prediction_data)
        explanation['key_factors'].extend(rule_based_factors)
        
        # Detaylı analiz
        explanation['detailed_analysis'] = self._perform_detailed_analysis(prediction_data)
        
        # Doğal dil açıklaması
        explanation['natural_language_explanation'] = self._generate_natural_language_explanation(
            prediction_data, explanation['key_factors']
        )
        
        # Güven açıklaması
        explanation['confidence_reasoning'] = self._explain_confidence(prediction_data)
        
        # Risk faktörleri
        explanation['risk_factors'] = self._identify_risk_factors(prediction_data)
        
        return explanation
        
    def _calculate_shap_values(self, model, features):
        """SHAP değerlerini hesapla"""
        try:
            # Model tipine göre explainer seç
            model_type = type(model).__name__
            
            if model_type not in self.explainers:
                if 'XGB' in model_type or 'xgboost' in model_type.lower():
                    self.explainers[model_type] = shap.TreeExplainer(model)
                elif 'RandomForest' in model_type:
                    self.explainers[model_type] = shap.TreeExplainer(model)
                elif 'Neural' in model_type or 'Sequential' in model_type:
                    self.explainers[model_type] = shap.DeepExplainer(model, features)
                else:
                    # Genel explainer
                    self.explainers[model_type] = shap.KernelExplainer(
                        model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                        features
                    )
            
            explainer = self.explainers[model_type]
            shap_values = explainer.shap_values(features)
            
            # Multi-class için ilk sınıfı al (HOME_WIN)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            return {
                'shap_values': shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                'base_value': float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0,
                'feature_importance': self._calculate_feature_importance(shap_values)
            }
            
        except Exception as e:
            logger.error(f"SHAP hesaplama hatası: {e}")
            return None
            
    def _extract_top_shap_factors(self, shap_analysis, top_n=5):
        """En önemli SHAP faktörlerini çıkar"""
        if not shap_analysis or 'feature_importance' not in shap_analysis:
            return []
            
        factors = []
        feature_importance = shap_analysis['feature_importance']
        
        # En yüksek önem derecesine sahip özellikleri sırala
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for feature_name, importance in sorted_features[:top_n]:
            # Özellik indeksini bul
            if feature_name in self.feature_names:
                feature_idx = self.feature_names.index(feature_name)
                shap_value = shap_analysis['shap_values'][0][feature_idx] if len(shap_analysis['shap_values']) > 0 else 0
                
                factors.append({
                    'feature': feature_name,
                    'importance': abs(importance),
                    'impact': 'positive' if shap_value > 0 else 'negative',
                    'shap_value': shap_value,
                    'description': self._get_feature_description(feature_name, shap_value)
                })
                
        return factors
        
    def _calculate_feature_importance(self, shap_values):
        """Özellik önem derecelerini hesapla"""
        if not isinstance(shap_values, np.ndarray):
            shap_values = np.array(shap_values)
            
        # Ortalama mutlak SHAP değerleri
        if shap_values.ndim == 1:
            importance_values = np.abs(shap_values)
        else:
            importance_values = np.mean(np.abs(shap_values), axis=0)
            
        # Feature isimlerine eşle
        importance_dict = {}
        for i, importance in enumerate(importance_values):
            if i < len(self.feature_names):
                importance_dict[self.feature_names[i]] = float(importance)
                
        return importance_dict
        
    def _analyze_rule_based_factors(self, prediction_data):
        """Kural tabanlı faktör analizi"""
        factors = []
        
        # xG analizi
        home_xg = prediction_data.get('expected_goals', {}).get('home', 0)
        away_xg = prediction_data.get('expected_goals', {}).get('away', 0)
        
        if home_xg > away_xg * 1.5:
            factors.append({
                'feature': 'goal_expectation',
                'importance': 0.8,
                'impact': 'positive',
                'value': home_xg,
                'description': f"Ev sahibi yüksek gol beklentisi ({home_xg:.2f} xG)"
            })
        elif away_xg > home_xg * 1.5:
            factors.append({
                'feature': 'goal_expectation',
                'importance': 0.8,
                'impact': 'negative',
                'value': away_xg,
                'description': f"Deplasman yüksek gol beklentisi ({away_xg:.2f} xG)"
            })
            
        # Form analizi
        if 'form_analysis' in prediction_data:
            home_form = prediction_data['form_analysis'].get('home_form', {})
            away_form = prediction_data['form_analysis'].get('away_form', {})
            
            if home_form.get('points_per_game', 0) > 2.5:
                factors.append({
                    'feature': 'form',
                    'importance': 0.7,
                    'impact': 'positive',
                    'value': home_form['points_per_game'],
                    'description': "Ev sahibi mükemmel formda"
                })
                
        # H2H analizi
        if 'h2h_analysis' in prediction_data:
            h2h = prediction_data['h2h_analysis']
            if h2h.get('home_dominance', 0) > 0.6:
                factors.append({
                    'feature': 'head_to_head',
                    'importance': 0.6,
                    'impact': 'positive',
                    'value': h2h['home_dominance'],
                    'description': "Ev sahibinin tarihsel üstünlüğü var"
                })
                
        return factors
        
    def _perform_detailed_analysis(self, prediction_data):
        """Detaylı analiz yap"""
        analysis = {
            'strengths': [],
            'weaknesses': [],
            'opportunities': [],
            'threats': []
        }
        
        # Güçlü yönler
        if prediction_data.get('home_win_probability', 0) > 50:
            analysis['strengths'].extend([
                "Ev sahibi avantajı",
                "Yüksek kazanma olasılığı",
                "İstatistiksel üstünlük"
            ])
            
        # Zayıf yönler
        if prediction_data.get('confidence', 0) < 0.7:
            analysis['weaknesses'].extend([
                "Düşük tahmin güveni",
                "Belirsiz sonuç",
                "Yakın istatistikler"
            ])
            
        # Fırsatlar
        if prediction_data.get('over_under', {}).get('over_2_5', 0) > 70:
            analysis['opportunities'].extend([
                "Yüksek skorlu maç potansiyeli",
                "Her iki takım gol atabilir",
                "Hücum ağırlıklı oyun beklentisi"
            ])
            
        # Tehditler
        if prediction_data.get('advanced_predictions', {}).get('team_goals', {}).get('away_over_1_5', 0) > 60:
            analysis['threats'].extend([
                "Deplasman takımı gol tehdidi",
                "Savunma zafiyeti riski",
                "Kontra atak tehlikesi"
            ])
            
        return analysis
        
    def _generate_natural_language_explanation(self, prediction_data, key_factors):
        """Doğal dil açıklaması oluştur"""
        outcome = prediction_data.get('most_likely_outcome', 'UNKNOWN')
        confidence = prediction_data.get('confidence', 0)
        home_prob = prediction_data.get('home_win_probability', 0)
        draw_prob = prediction_data.get('draw_probability', 0)
        away_prob = prediction_data.get('away_win_probability', 0)
        
        # Temel açıklama
        if outcome == 'HOME_WIN':
            base_explanation = f"Ev sahibi takımın kazanması bekleniyor (%{home_prob:.0f} olasılık)."
        elif outcome == 'AWAY_WIN':
            base_explanation = f"Deplasman takımının kazanması bekleniyor (%{away_prob:.0f} olasılık)."
        else:
            base_explanation = f"Beraberlik en olası sonuç (%{draw_prob:.0f} olasılık)."
            
        # Güven açıklaması
        if confidence > 0.8:
            confidence_text = "Bu tahmin yüksek güvenilirliğe sahip."
        elif confidence > 0.6:
            confidence_text = "Bu tahmin orta düzeyde güvenilir."
        else:
            confidence_text = "Bu tahmin düşük güvenilirliğe sahip, sonuç belirsiz."
            
        # Ana faktörleri ekle
        factor_explanations = []
        for factor in key_factors[:3]:  # En önemli 3 faktör
            factor_explanations.append(factor.get('description', ''))
            
        # Birleştir
        full_explanation = f"{base_explanation} {confidence_text}\n\n"
        full_explanation += "Ana faktörler:\n"
        for i, explanation in enumerate(factor_explanations, 1):
            full_explanation += f"{i}. {explanation}\n"
            
        # Ek bilgiler
        expected_goals = prediction_data.get('expected_goals', {})
        total_expected = expected_goals.get('home', 0) + expected_goals.get('away', 0)
        
        full_explanation += f"\nBeklenen toplam gol: {total_expected:.1f}"
        
        if prediction_data.get('over_under', {}).get('over_2_5', 0) > 60:
            full_explanation += " (Yüksek skorlu maç bekleniyor)"
        else:
            full_explanation += " (Düşük skorlu maç bekleniyor)"
            
        return full_explanation
        
    def _explain_confidence(self, prediction_data):
        """Güven seviyesini açıkla"""
        confidence = prediction_data.get('confidence', 0)
        factors_affecting_confidence = []
        
        # Olasılık dağılımı
        home_prob = prediction_data.get('home_win_probability', 0)
        draw_prob = prediction_data.get('draw_probability', 0)
        away_prob = prediction_data.get('away_win_probability', 0)
        
        max_prob = max(home_prob, draw_prob, away_prob)
        prob_variance = np.var([home_prob, draw_prob, away_prob])
        
        if max_prob > 50:
            factors_affecting_confidence.append({
                'factor': 'clear_favorite',
                'impact': 'positive',
                'description': f"Net favori var (%{max_prob:.0f} olasılık)"
            })
        else:
            factors_affecting_confidence.append({
                'factor': 'close_probabilities',
                'impact': 'negative',
                'description': "Yakın olasılıklar, belirsiz sonuç"
            })
            
        # Model uyumu
        if 'model_agreement' in prediction_data:
            agreement = prediction_data['model_agreement']
            if agreement > 0.8:
                factors_affecting_confidence.append({
                    'factor': 'high_model_agreement',
                    'impact': 'positive',
                    'description': "Tüm modeller hemfikir"
                })
            elif agreement < 0.5:
                factors_affecting_confidence.append({
                    'factor': 'low_model_agreement',
                    'impact': 'negative',
                    'description': "Modeller arasında fikir ayrılığı"
                })
                
        # Veri kalitesi
        if prediction_data.get('data_quality', {}).get('completeness', 1) < 0.8:
            factors_affecting_confidence.append({
                'factor': 'incomplete_data',
                'impact': 'negative',
                'description': "Eksik veri nedeniyle belirsizlik"
            })
            
        return {
            'confidence_level': confidence,
            'confidence_category': self._categorize_confidence(confidence),
            'factors': factors_affecting_confidence,
            'recommendation': self._get_confidence_recommendation(confidence)
        }
        
    def _categorize_confidence(self, confidence):
        """Güven seviyesini kategorize et"""
        if confidence >= 0.85:
            return 'very_high'
        elif confidence >= 0.75:
            return 'high'
        elif confidence >= 0.65:
            return 'moderate'
        elif confidence >= 0.55:
            return 'low'
        else:
            return 'very_low'
            
    def _get_confidence_recommendation(self, confidence):
        """Güven seviyesine göre öneri"""
        if confidence >= 0.75:
            return "Bu tahmin güvenilir, kararlarınızda kullanabilirsiniz."
        elif confidence >= 0.65:
            return "Orta düzey güven, ek faktörleri de değerlendirin."
        else:
            return "Düşük güven seviyesi, bu tahmini dikkatli kullanın."
            
    def _identify_risk_factors(self, prediction_data):
        """Risk faktörlerini belirle"""
        risks = []
        
        # Düşük güven riski
        if prediction_data.get('confidence', 0) < 0.6:
            risks.append({
                'type': 'low_confidence',
                'severity': 'high',
                'description': 'Tahmin güvenilirliği düşük',
                'mitigation': 'Ek analiz yapın veya tahmini kullanmayın'
            })
            
        # Veri eksikliği riski
        if prediction_data.get('data_quality', {}).get('missing_features', 0) > 3:
            risks.append({
                'type': 'data_quality',
                'severity': 'medium',
                'description': 'Önemli veri eksiklikleri var',
                'mitigation': 'Eksik verileri tamamlamaya çalışın'
            })
            
        # Ekstrem tahmin riski
        expected_total = prediction_data.get('expected_goals', {}).get('home', 0) + \
                        prediction_data.get('expected_goals', {}).get('away', 0)
        
        if expected_total > 5.0:
            risks.append({
                'type': 'extreme_prediction',
                'severity': 'medium',
                'description': 'Anormal yüksek gol beklentisi',
                'mitigation': 'Ekstrem maç olabilir, dikkatli değerlendirin'
            })
        elif expected_total < 1.0:
            risks.append({
                'type': 'extreme_prediction',
                'severity': 'medium',
                'description': 'Anormal düşük gol beklentisi',
                'mitigation': 'Defansif maç bekleniyor, skorlar düşük olabilir'
            })
            
        # Model uyumsuzluğu riski
        if prediction_data.get('model_disagreement', 0) > 0.3:
            risks.append({
                'type': 'model_disagreement',
                'severity': 'medium',
                'description': 'Modeller arasında ciddi fikir ayrılığı',
                'mitigation': 'Farklı model tahminlerini ayrı ayrı inceleyin'
            })
            
        return risks
        
    def _get_feature_description(self, feature_name, value):
        """Özellik açıklaması oluştur"""
        descriptions = {
            'home_xg': f"Ev sahibi gol beklentisi {value:.2f} birim etki ediyor",
            'away_xg': f"Deplasman gol beklentisi {value:.2f} birim etki ediyor",
            'elo_diff': f"Güç farkı {value:.0f} puan tahmine etki ediyor",
            'home_form': f"Ev sahibi formu {value:.2f} birim katkı sağlıyor",
            'away_form': f"Deplasman formu {value:.2f} birim etki ediyor",
            'home_advantage': "Ev sahibi avantajı tahmine olumlu etki ediyor" if value > 0 else "Ev avantajı etkisi düşük",
            'form_momentum': f"Form momentumu {value:.2f} birim etki gösteriyor",
            'match_importance': f"Maç önemi {value:.2f} seviyesinde tahmine etki ediyor"
        }
        
        return descriptions.get(feature_name, f"{feature_name} özelliği {value:.2f} etki gösteriyor")
        
    def create_visual_explanation(self, prediction_data, explanation, save_path='explanations/'):
        """Görsel açıklama oluştur"""
        import os
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Özellik önem grafiği
        if 'key_factors' in explanation and explanation['key_factors']:
            self._plot_feature_importance(explanation['key_factors'], 
                                        f"{save_path}feature_importance_{timestamp}.png")
            
        # 2. Olasılık dağılımı
        self._plot_probability_distribution(prediction_data,
                                          f"{save_path}probability_dist_{timestamp}.png")
        
        # 3. SHAP değerleri (eğer varsa)
        if 'shap_analysis' in explanation and explanation['shap_analysis']:
            self._plot_shap_values(explanation['shap_analysis'],
                                 f"{save_path}shap_values_{timestamp}.png")
            
        # 4. Risk matrisi
        if 'risk_factors' in explanation:
            self._plot_risk_matrix(explanation['risk_factors'],
                                 f"{save_path}risk_matrix_{timestamp}.png")
            
        return {
            'feature_importance': f"feature_importance_{timestamp}.png",
            'probability_distribution': f"probability_dist_{timestamp}.png",
            'shap_values': f"shap_values_{timestamp}.png" if 'shap_analysis' in explanation else None,
            'risk_matrix': f"risk_matrix_{timestamp}.png" if 'risk_factors' in explanation else None
        }
        
    def _plot_feature_importance(self, key_factors, filename):
        """Özellik önem grafiği"""
        features = [f['feature'] for f in key_factors[:10]]
        importances = [f['importance'] for f in key_factors[:10]]
        impacts = [f['impact'] for f in key_factors[:10]]
        
        # Renkleri belirle
        colors = ['green' if impact == 'positive' else 'red' for impact in impacts]
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(features, importances, color=colors)
        
        # Değerleri bar üzerine yaz
        for bar, importance in zip(bars, importances):
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                    f'{importance:.2f}', ha='left', va='center')
            
        plt.xlabel('Önem Derecesi')
        plt.title('Tahmine En Çok Etki Eden Faktörler')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
    def _plot_probability_distribution(self, prediction_data, filename):
        """Olasılık dağılım grafiği"""
        outcomes = ['Ev Sahibi', 'Beraberlik', 'Deplasman']
        probabilities = [
            prediction_data.get('home_win_probability', 0),
            prediction_data.get('draw_probability', 0),
            prediction_data.get('away_win_probability', 0)
        ]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(outcomes, probabilities, color=['blue', 'gray', 'red'])
        
        # Değerleri bar üzerine yaz
        for bar, prob in zip(bars, probabilities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{prob:.1f}%', ha='center', va='bottom')
            
        plt.ylabel('Olasılık (%)')
        plt.title('Maç Sonucu Olasılık Dağılımı')
        plt.ylim(0, 100)
        
        # En olası sonucu vurgula
        max_idx = probabilities.index(max(probabilities))
        bars[max_idx].set_edgecolor('black')
        bars[max_idx].set_linewidth(3)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
    def _plot_shap_values(self, shap_analysis, filename):
        """SHAP değerleri grafiği"""
        if not shap_analysis or 'shap_values' not in shap_analysis:
            return
            
        shap_values = shap_analysis['shap_values'][0] if shap_analysis['shap_values'] else []
        feature_names_subset = self.feature_names[:len(shap_values)]
        
        # En önemli 10 özelliği al
        abs_shap = [abs(v) for v in shap_values]
        top_indices = sorted(range(len(abs_shap)), key=lambda i: abs_shap[i], reverse=True)[:10]
        
        top_features = [feature_names_subset[i] for i in top_indices]
        top_values = [shap_values[i] for i in top_indices]
        
        plt.figure(figsize=(10, 6))
        colors = ['green' if v > 0 else 'red' for v in top_values]
        bars = plt.barh(top_features, top_values, color=colors)
        
        plt.xlabel('SHAP Değeri')
        plt.title('Özellik Etki Analizi (SHAP)')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
    def _plot_risk_matrix(self, risk_factors, filename):
        """Risk matrisi görselleştirmesi"""
        if not risk_factors:
            return
            
        # Risk seviyelerini sayısal değerlere çevir
        severity_map = {'low': 1, 'medium': 2, 'high': 3}
        
        risk_types = [r['type'] for r in risk_factors]
        severities = [severity_map.get(r['severity'], 2) for r in risk_factors]
        
        plt.figure(figsize=(10, 6))
        
        # Risk matrisini oluştur
        colors = ['yellow', 'orange', 'red']
        for i, (risk_type, severity) in enumerate(zip(risk_types, severities)):
            plt.scatter(i, severity, s=500, c=colors[severity-1], edgecolors='black', linewidth=2)
            plt.text(i, severity, risk_type.replace('_', '\n'), ha='center', va='center', fontsize=8)
            
        plt.ylim(0.5, 3.5)
        plt.xlim(-0.5, len(risk_types) - 0.5)
        plt.yticks([1, 2, 3], ['Düşük', 'Orta', 'Yüksek'])
        plt.xticks([])
        plt.ylabel('Risk Seviyesi')
        plt.title('Tahmin Risk Faktörleri')
        plt.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
    def generate_explanation_report(self, prediction_data, explanation, output_file='explanation_report.html'):
        """HTML formatında açıklama raporu oluştur"""
        # Replace işlemini f-string dışında yap
        nl_explanation = explanation.get('natural_language_explanation', '').replace('\n', '<br>')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tahmin Açıklama Raporu</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .prediction-box {{ 
                    background-color: #f0f0f0; 
                    padding: 15px; 
                    border-radius: 5px; 
                    margin: 10px 0;
                }}
                .factor {{ 
                    margin: 10px 0; 
                    padding: 10px; 
                    border-left: 3px solid #007bff;
                    background-color: #f8f9fa;
                }}
                .positive {{ border-left-color: #28a745; }}
                .negative {{ border-left-color: #dc3545; }}
                .risk {{ 
                    margin: 10px 0; 
                    padding: 10px; 
                    background-color: #fff3cd;
                    border: 1px solid #ffeaa7;
                }}
                .confidence-bar {{
                    width: 100%;
                    background-color: #e0e0e0;
                    border-radius: 5px;
                    overflow: hidden;
                }}
                .confidence-fill {{
                    height: 30px;
                    background-color: #007bff;
                    text-align: center;
                    line-height: 30px;
                    color: white;
                }}
            </style>
        </head>
        <body>
            <h1>Maç Tahmini Açıklama Raporu</h1>
            
            <div class="prediction-box">
                <h2>Tahmin Özeti</h2>
                <p><strong>Sonuç:</strong> {prediction_data.get('most_likely_outcome', 'Bilinmiyor')}</p>
                <p><strong>Güven:</strong></p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {prediction_data.get('confidence', 0)*100}%">
                        {prediction_data.get('confidence', 0)*100:.1f}%
                    </div>
                </div>
            </div>
            
            <h2>Doğal Dil Açıklaması</h2>
            <p>{nl_explanation}</p>
            
            <h2>Ana Faktörler</h2>
        """
        
        # Faktörleri ekle
        for factor in explanation.get('key_factors', [])[:5]:
            impact_class = 'positive' if factor.get('impact') == 'positive' else 'negative'
            html_content += f"""
            <div class="factor {impact_class}">
                <strong>{factor.get('feature', 'Unknown')}:</strong> {factor.get('description', '')}
                <br>Önem: {factor.get('importance', 0):.2f}
            </div>
            """
            
        # Risk faktörleri
        html_content += "<h2>Risk Faktörleri</h2>"
        for risk in explanation.get('risk_factors', []):
            html_content += f"""
            <div class="risk">
                <strong>{risk.get('type', '').replace('_', ' ').title()}:</strong> {risk.get('description', '')}
                <br>Seviye: {risk.get('severity', 'unknown')}
                <br>Öneri: {risk.get('mitigation', '')}
            </div>
            """
            
        html_content += """
        </body>
        </html>
        """
        
        # Dosyaya yaz
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"Açıklama raporu oluşturuldu: {output_file}")
        return output_file