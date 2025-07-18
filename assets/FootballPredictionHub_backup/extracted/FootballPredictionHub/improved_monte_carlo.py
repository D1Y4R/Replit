"""
Geliştirilmiş Monte Carlo Simülasyonu Modülü

Bu modül, futbol maç tahminlerinde kullanılan gelişmiş Monte Carlo simülasyonlarını içerir.
Standart Poisson temelli Monte Carlo simülasyonlarının ötesinde, aşağıdaki özellikleri sunar:

1. Negatif Binomial dağılımı desteği - Poisson'a göre daha fazla varyasyon
2. Dixon-Coles düzeltmesi - Düşük skorlu maçlarda daha doğru tahminler
3. Takım güçlerine göre dinamik gol dağılımı ayarlaması
4. Savunma zayıflığı analizi - Zayıf savunmalı takımlara karşı gol dağılımı güncelleme
5. İlk yarı/maç sonu dinamikleri - 45-90 dakika arası momentum değişimlerini modellemek

Kullanım:
    from improved_monte_carlo import EnhancedMonteCarlo
    
    simulator = EnhancedMonteCarlo()
    results = simulator.run_simulation(home_stats, away_stats, simulations=10000)
"""

import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class EnhancedMonteCarlo:
    """
    Gelişmiş Monte Carlo simülasyonu sınıfı.
    Futbol maç tahminleri için çeşitli olasılık dağılımları ve
    düzeltme faktörleri ile zenginleştirilmiş simülasyonlar sunar.
    """
    
    def __init__(self):
        """Monte Carlo simülatörünü başlat"""
        # Dixon-Coles düzeltme parametreleri
        self.dixon_coles_tau = 0.15  # Düşük skorlu maçlarda düzeltme faktörü
        
        # Negatif Binomial parametreleri
        self.neg_binomial_dispersion = 1.2  # Varsayılan dağılım faktörü
        
    def run_simulation(self, home_stats, away_stats, simulations=10000, 
                       home_defense_weakness=1.0, away_defense_weakness=1.0):
        """
        Monte Carlo simülasyonu çalıştır
        
        Args:
            home_stats: Ev sahibi takım istatistikleri
            away_stats: Deplasman takımı istatistikleri
            simulations: Simülasyon sayısı
            home_defense_weakness: Ev sahibi takımın savunma zayıflık faktörü (1.0 = normal)
            away_defense_weakness: Deplasman takımının savunma zayıflık faktörü (1.0 = normal)
            
        Returns:
            dict: Simülasyon sonuçları
        """
        # Gol beklentileri
        home_lambda = home_stats.get('expected_goals', 1.5)
        away_lambda = away_stats.get('expected_goals', 1.2) 
        
        # Savunma zayıflıklarını modele dahil et - "Zehirli Savunma Analizi"
        # Savunma zayıflığı faktörü, rakip takımın gol beklentisini artırır
        adjusted_home_lambda = home_lambda * away_defense_weakness
        adjusted_away_lambda = away_lambda * home_defense_weakness
        
        # Form ve savunma zayıflığı arasındaki ilişkiyi değerlendir
        # Eğer form verileri mevcutsa, savunma zayıflığını ince ayar yap
        if 'form' in home_stats and 'form' in away_stats:
            home_form = home_stats['form']
            away_form = away_stats['form']
            
            # Son maç verilerini kontrol et
            if home_form and 'recent_match_data' in home_form:
                home_matches = home_form['recent_match_data'][:5]  # Son 5 maç
                if home_matches:
                    # Son 5 maçta konsantre savunma problemi var mı?
                    late_goals_conceded = 0
                    for match in home_matches:
                        if 'late_goals_conceded' in match and match['late_goals_conceded'] > 0:
                            late_goals_conceded += 1
                    
                    # Eğer son 5 maçın 2+ tanesinde son dakika golleri yendiyse
                    # deplasman gol beklentisini maçın son bölümünde artır
                    if late_goals_conceded >= 2:
                        # Zayıf konsantrasyon faktörü: İkinci yarıda daha fazla gol yeme eğilimi
                        adjusted_away_lambda *= 1.1  # %10 daha fazla gol beklentisi
                        logger.info(f"Ev sahibi takım son dakika zayıflığı tespit edildi: {late_goals_conceded} maçta geç gol yedi")
            
            # Deplasman takımı için de benzer analizi yap
            if away_form and 'recent_match_data' in away_form:
                away_matches = away_form['recent_match_data'][:5]  # Son 5 maç
                if away_matches:
                    # Son 5 maçta konsantre savunma problemi var mı?
                    late_goals_conceded = 0
                    for match in away_matches:
                        if 'late_goals_conceded' in match and match['late_goals_conceded'] > 0:
                            late_goals_conceded += 1
                    
                    # Eğer son 5 maçın 2+ tanesinde son dakika golleri yendiyse
                    # ev sahibi gol beklentisini maçın son bölümünde artır
                    if late_goals_conceded >= 2:
                        # Zayıf konsantrasyon faktörü
                        adjusted_home_lambda *= 1.12  # %12 daha fazla gol beklentisi
                        logger.info(f"Deplasman takımı son dakika zayıflığı tespit edildi: {late_goals_conceded} maçta geç gol yedi")
        
        logger.info(f"Monte Carlo simülasyonu: Ev={adjusted_home_lambda:.2f}, Deplasman={adjusted_away_lambda:.2f} gol beklentisi")
        logger.info(f"Savunma zayıflık faktörleri: Ev={home_defense_weakness:.2f}, Deplasman={away_defense_weakness:.2f}")
        
        # Sonuç sayaçları
        home_wins = 0
        draws = 0
        away_wins = 0
        
        # Detaylı istatistik sayaçları
        btts_count = 0  # İki takım da gol attı
        over_2_5_count = 0  # 2.5 üstü gol
        over_3_5_count = 0  # 3.5 üstü gol
        
        # Skor dağılımı
        score_distribution = {}
        
        # İY/MS dağılımları
        half_time_distribution = {}
        half_time_full_time_distribution = {}
        
        # İlk gol zamanlaması
        first_goal_timing = {
            "1-15": 0, "16-30": 0, "31-45": 0,
            "46-60": 0, "61-75": 0, "76-90": 0, "No Goal": 0
        }
        
        # Simülasyonları çalıştır
        for _ in range(simulations):
            # İlk yarı
            home_goals_ht, away_goals_ht = self._simulate_half(adjusted_home_lambda, adjusted_away_lambda, is_first_half=True)
            
            # İkinci yarı - ilk yarı skoruna göre dinamik ayarla
            home_goals_2h, away_goals_2h = self._simulate_half(
                adjusted_home_lambda, 
                adjusted_away_lambda,
                is_first_half=False,
                ht_score=(home_goals_ht, away_goals_ht)
            )
            
            # Toplam goller
            home_goals_ft = home_goals_ht + home_goals_2h
            away_goals_ft = away_goals_ht + away_goals_2h
            
            # Maç sonucu
            if home_goals_ft > away_goals_ft:
                home_wins += 1
            elif home_goals_ft == away_goals_ft:
                draws += 1
            else:
                away_wins += 1
                
            # Her iki takım da gol attı mı
            if home_goals_ft > 0 and away_goals_ft > 0:
                btts_count += 1
                
            # 2.5 üstü mü
            total_goals = home_goals_ft + away_goals_ft
            if total_goals > 2.5:
                over_2_5_count += 1
                
            # 3.5 üstü mü
            if total_goals > 3.5:
                over_3_5_count += 1
                
            # Skor dağılımı güncelle
            ft_score = f"{home_goals_ft}-{away_goals_ft}"
            score_distribution[ft_score] = score_distribution.get(ft_score, 0) + 1
            
            # İlk yarı skor dağılımı
            ht_score = f"{home_goals_ht}-{away_goals_ht}"
            half_time_distribution[ht_score] = half_time_distribution.get(ht_score, 0) + 1
            
            # İY/MS dağılımı güncelle
            ht_result = self._get_result_code(home_goals_ht, away_goals_ht)
            ft_result = self._get_result_code(home_goals_ft, away_goals_ft)
            htft_result = f"{ht_result}/{ft_result}"
            half_time_full_time_distribution[htft_result] = half_time_full_time_distribution.get(htft_result, 0) + 1
            
            # İlk gol zamanlaması
            if home_goals_ft + away_goals_ft == 0:
                first_goal_timing["No Goal"] += 1
            else:
                goal_minute = self._simulate_first_goal_time(home_lambda, away_lambda)
                if goal_minute <= 15:
                    first_goal_timing["1-15"] += 1
                elif goal_minute <= 30:
                    first_goal_timing["16-30"] += 1
                elif goal_minute <= 45:
                    first_goal_timing["31-45"] += 1
                elif goal_minute <= 60:
                    first_goal_timing["46-60"] += 1
                elif goal_minute <= 75:
                    first_goal_timing["61-75"] += 1
                else:
                    first_goal_timing["76-90"] += 1
            
        # Sonuçları hesapla
        home_win_prob = home_wins / simulations * 100
        draw_prob = draws / simulations * 100
        away_win_prob = away_wins / simulations * 100
        
        # Diğer bahis olasılıkları
        btts_prob = btts_count / simulations * 100
        over_2_5_prob = over_2_5_count / simulations * 100
        over_3_5_prob = over_3_5_count / simulations * 100
        
        # En olası skorları bul
        most_likely_scores = sorted(score_distribution.items(), key=lambda x: x[1], reverse=True)[:5]
        most_likely_scores = [(score, count / simulations * 100) for score, count in most_likely_scores]
        
        # En olası İY/MS kombinasyonlarını bul
        most_likely_htft = sorted(half_time_full_time_distribution.items(), key=lambda x: x[1], reverse=True)
        most_likely_htft = [(result, count / simulations * 100) for result, count in most_likely_htft]
        
        # İlk gol zamanlamasını hesapla
        first_goal_probs = {time: count / simulations * 100 for time, count in first_goal_timing.items()}
        
        # Sonuçları döndür
        return {
            'match_outcome': {
                'home_win': home_win_prob,
                'draw': draw_prob,
                'away_win': away_win_prob
            },
            'score_probabilities': {score: prob for score, prob in most_likely_scores},
            'half_time_full_time': {result: prob for result, prob in most_likely_htft},
            'other_markets': {
                'btts': btts_prob,
                'over_2_5': over_2_5_prob,
                'over_3_5': over_3_5_prob
            },
            'first_goal_timing': first_goal_probs,
            'simulations': simulations,
            'expected_goals': {
                'home': adjusted_home_lambda,
                'away': adjusted_away_lambda
            }
        }
    
    def _simulate_half(self, home_lambda, away_lambda, is_first_half=True, ht_score=None):
        """
        Bir yarıyı simüle et
        
        Args:
            home_lambda: Ev sahibi takımın gol beklentisi (tüm maç için)
            away_lambda: Deplasman takımının gol beklentisi (tüm maç için)
            is_first_half: İlk yarı mı?
            ht_score: İlk yarı skoru (ikinci yarı simülasyonu için)
            
        Returns:
            tuple: (ev_goller, deplasman_goller)
        """
        # İlk/ikinci yarı gol dağılımı
        if is_first_half:
            # İlk yarıda tüm gollerin yaklaşık %40'ı atılır
            half_home_lambda = home_lambda * 0.4
            half_away_lambda = away_lambda * 0.4
        else:
            # İkinci yarıda tüm gollerin yaklaşık %60'ı atılır
            half_home_lambda = home_lambda * 0.6
            half_away_lambda = away_lambda * 0.6
            
            # İlk yarı skor farkına göre motivasyon/taktik ayarlaması
            if ht_score:
                home_ht, away_ht = ht_score
                score_diff = home_ht - away_ht
                
                # Geride olan takım daha agresif oynar
                if score_diff > 0:  # Ev sahibi önde
                    half_away_lambda *= min(1.4, 1.0 + abs(score_diff) * 0.15)  # Deplasman daha atak
                    if score_diff >= 2:  # Ev sahibi rahat önde, tempo düşebilir
                        half_home_lambda *= 0.9
                elif score_diff < 0:  # Deplasman önde
                    half_home_lambda *= min(1.4, 1.0 + abs(score_diff) * 0.15)  # Ev sahibi daha atak
                    if score_diff <= -2:  # Deplasman rahat önde, tempo düşebilir
                        half_away_lambda *= 0.9
                # Beraberlikte hafif değişiklik yok
        
        # ÖNEMLİ: Düşük beklenen gol değerleri için 0-0, 1-0, 0-1 skorlarının olasılığını artır
        # İki takım da düşük beklenen gol değerlerine sahipse özel bir yaklaşım uygula
        if home_lambda < 1.0 and away_lambda < 1.0:
            # Yüksek olasılıkla düşük skorlar üret - 0-0 veya 1-1 skorları daha olasıdır
            method = np.random.choice(['low_scoring', 'poisson', 'dixon_coles'], 
                                     p=[0.6, 0.25, 0.15])  # 0-0 eğilimini artırmak için düşük skorlara daha fazla ağırlık
            
            if method == 'low_scoring':
                # Beklenen gol değerlerine göre düşük skorlu sonuçları modelle
                low_score_probs = {}
                
                # Tüm goller (maç başına)
                total_lambda = home_lambda + away_lambda
                
                # Düşük skorlara daha yüksek olasılık ver
                if total_lambda < 1.5:
                    # Çok düşük toplam gol beklentisi (1.5 altı)
                    low_score_probs = {
                        '0-0': 0.50,  # 0-0 için %50 olasılık (10% artırıldı)
                        '1-0': 0.20 * (home_lambda / total_lambda) if total_lambda > 0 else 0.15,
                        '0-1': 0.20 * (away_lambda / total_lambda) if total_lambda > 0 else 0.15,
                        '1-1': 0.08,  # 1-1 için %8 olasılık (azaltıldı)
                        'other': 0.07  # Diğer skorlar için %7 (azaltıldı)
                    }
                else:
                    # Düşük toplam gol beklentisi (1.5-2.0 arası)
                    low_score_probs = {
                        '0-0': 0.35,  # 0-0 için %35 olasılık (artırıldı)
                        '1-0': 0.20 * (home_lambda / total_lambda) if total_lambda > 0 else 0.15,
                        '0-1': 0.20 * (away_lambda / total_lambda) if total_lambda > 0 else 0.15,
                        '1-1': 0.12,  # 1-1 için %12 olasılık (azaltıldı)
                        'other': 0.13  # Diğer skorlar için %13 (azaltıldı)
                    }
                
                # Düşük skor olasılıklarına göre skor seç
                rand = np.random.random()
                cum_prob = 0.0
                selected_score = None
                
                for score, prob in low_score_probs.items():
                    cum_prob += prob
                    if rand < cum_prob:
                        selected_score = score
                        break
                
                if selected_score == '0-0':
                    home_goals, away_goals = 0, 0
                elif selected_score == '1-0':
                    home_goals, away_goals = 1, 0
                elif selected_score == '0-1':
                    home_goals, away_goals = 0, 1
                elif selected_score == '1-1':
                    home_goals, away_goals = 1, 1
                else:  # 'other' durumu - standart Poisson kullan
                    home_goals = np.random.poisson(half_home_lambda)
                    away_goals = np.random.poisson(half_away_lambda)
            
            elif method == 'poisson':
                # Standart Poisson dağılımı - düşük lambdalar zaten düşük skor üretecek
                home_goals = np.random.poisson(half_home_lambda)
                away_goals = np.random.poisson(half_away_lambda)
            
            elif method == 'dixon_coles':
                # Dixon-Coles düzeltmesi ile düşük skorlu maçlar için özel ayarlanmış Poisson
                home_goals = np.random.poisson(half_home_lambda)
                away_goals = np.random.poisson(half_away_lambda)
                
                # Düşük beklenen gol değerleri için 0-0 olasılığını artır
                if home_goals == 0 and away_goals == 0:
                    # Zaten 0-0, yüksek olasılıkla sakla
                    if np.random.random() < 0.8:  # %80 olasılıkla 0-0'ı koru
                        pass
                elif home_goals == 0 and away_goals == 1:
                    # 0-1, yüksek olasılıkla sakla
                    if np.random.random() < 0.7:  # %70 olasılıkla 0-1'i koru
                        pass
                    else:
                        # Bazen 0-0'a dönüştür
                        if np.random.random() < 0.5:
                            away_goals = 0
                elif home_goals == 1 and away_goals == 0:
                    # 1-0, yüksek olasılıkla sakla
                    if np.random.random() < 0.7:  # %70 olasılıkla 1-0'ı koru
                        pass
                    else:
                        # Bazen 0-0'a dönüştür
                        if np.random.random() < 0.5:
                            home_goals = 0
                elif home_goals == 1 and away_goals == 1:
                    # 1-1, daha düşük olasılıkla sakla, düşük beklenen gol için nadir
                    if np.random.random() < 0.3:  # %30 olasılıkla 1-1'i koru
                        pass
                    else:
                        # Genellikle 0-0, 1-0 veya 0-1'e dönüştür
                        rand = np.random.random()
                        if rand < 0.5:  # %50 olasılıkla 0-0
                            home_goals, away_goals = 0, 0
                        elif rand < 0.75:  # %25 olasılıkla 1-0
                            away_goals = 0
                        else:  # %25 olasılıkla 0-1
                            home_goals = 0
                else:
                    # Yüksek skorları düşürme eğilimi
                    if np.random.random() < 0.7:  # %70 olasılıkla düşük skor tercih et
                        rand = np.random.random()
                        if rand < 0.4:  # %40 olasılıkla 0-0
                            home_goals, away_goals = 0, 0
                        elif rand < 0.65:  # %25 olasılıkla 1-0
                            home_goals, away_goals = 1, 0
                        elif rand < 0.9:  # %25 olasılıkla 0-1
                            home_goals, away_goals = 0, 1
                        else:  # %10 olasılıkla 1-1
                            home_goals, away_goals = 1, 1
        else:
            # Normal beklenen gol değerleri için standart yaklaşım
            # Farklı dağılımlar arasında rastgele seçim
            method = np.random.choice(['poisson', 'neg_binomial', 'dixon_coles'], 
                                     p=[0.6, 0.3, 0.1])  # Dağılım yöntemlerinin ağırlıkları
            
            if method == 'poisson':
                # Standart Poisson dağılımı
                home_goals = np.random.poisson(half_home_lambda)
                away_goals = np.random.poisson(half_away_lambda)
            
            elif method == 'neg_binomial':
                # Negatif Binomial dağılımı - Poisson'dan daha fazla varyasyon
                # Poisson mu ve var parametrelerinden r ve p hesapla
                try:
                    # r ve p parametrelerini hesapla
                    home_r = half_home_lambda / (self.neg_binomial_dispersion - 1) if self.neg_binomial_dispersion > 1 else 1.0
                    away_r = half_away_lambda / (self.neg_binomial_dispersion - 1) if self.neg_binomial_dispersion > 1 else 1.0
                    
                    home_p = home_r / (home_r + half_home_lambda)
                    away_p = away_r / (away_r + half_away_lambda)
                    
                    # Negatif binomial dağılımından örnekle
                    home_goals = np.random.negative_binomial(home_r, home_p)
                    away_goals = np.random.negative_binomial(away_r, away_p)
                except Exception:
                    # Hata durumunda Poisson'a geri dön
                    home_goals = np.random.poisson(half_home_lambda)
                    away_goals = np.random.poisson(half_away_lambda)
                    
            elif method == 'dixon_coles':
                # Dixon-Coles düzeltmesi ile Poisson
                # Önce bağımsız Poisson dağılımlarından gol sayılarını örnekle
                home_goals = np.random.poisson(half_home_lambda)
                away_goals = np.random.poisson(half_away_lambda)
                
                # Düşük skorlu maçlar için Dixon-Coles düzeltmesi uygula
                if home_goals <= 1 and away_goals <= 1:
                    # 0-0, 1-0, 0-1 ve 1-1 durumları için düzeltme
                    # Bir kez daha örnek alma ile basit bir simülasyon
                    adjustment = 0.0
                    
                    if home_goals == 0 and away_goals == 0:
                        # 0-0 durumu için düzeltme
                        adjustment = self.dixon_coles_tau  # Pozitif değer, olasılığı artırır
                    elif home_goals == 1 and away_goals == 1:
                        # 1-1 durumu için düzeltme
                        adjustment = self.dixon_coles_tau * 0.5  # Daha az pozitif değer
                    elif (home_goals == 1 and away_goals == 0) or (home_goals == 0 and away_goals == 1):
                        # 1-0 veya 0-1 durumları için düzeltme
                        adjustment = -self.dixon_coles_tau * 0.3  # Negatif değer, olasılığı azaltır
                    
                    # Düzeltme uygula (basit bir yaklaşım olarak, yeniden örnek alma)
                    if np.random.random() < abs(adjustment):
                        if adjustment > 0:  # Olasılığı artır - bu skor daha olası
                            pass  # Mevcut skoru koru
                        else:  # Olasılığı azalt - yeniden örnek al
                            # Tekrar örnek al ama aynı dağılımdan
                            home_goals = np.random.poisson(half_home_lambda)
                            away_goals = np.random.poisson(half_away_lambda)
        
        # Makul bir aralık içinde sınırla
        home_goals = min(5, max(0, home_goals))  # 0-5 arası
        away_goals = min(5, max(0, away_goals))  # 0-5 arası
        
        return home_goals, away_goals
    
    def _get_result_code(self, home_goals, away_goals):
        """
        Skor durumuna göre sonuç kodunu döndür (1, X, 2)
        
        Args:
            home_goals: Ev sahibi takımın gol sayısı
            away_goals: Deplasman takımının gol sayısı
            
        Returns:
            str: Sonuç kodu ("1", "X" veya "2")
        """
        if home_goals > away_goals:
            return "1"
        elif home_goals == away_goals:
            return "X"
        else:
            return "2"
    
    def _simulate_first_goal_time(self, home_lambda, away_lambda):
        """
        İlk gol süresini simüle et
        
        Args:
            home_lambda: Ev sahibi gol beklentisi
            away_lambda: Deplasman gol beklentisi
            
        Returns:
            int: İlk gol dakikası (1-90)
        """
        total_lambda = home_lambda + away_lambda
        
        # Golsüz maç olasılığını hesapla
        p_no_goal = np.exp(-total_lambda)
        
        if np.random.random() < p_no_goal:
            return 0  # Golsüz maç
        
        # İlk golün hangi dakikada atılacağını simüle et
        # Üstel dağılım kullanarak
        goal_rate = total_lambda / 90.0  # Dakika başına gol beklentisi
        goal_time = np.random.exponential(1.0 / goal_rate)
        
        # 1-90 dakika aralığında sınırla
        return max(1, min(90, int(goal_time)))