"""
Match Insights Generator
Bu modül, maç tahminlerine dayalı olarak doğal dil açıklamaları ve içgörüler üretir.

Özellikler:
1. Tahmin verilerinden anlamlı açıklamalar üretme
2. Takımların formuna dayalı analizler
3. Geçmiş karşılaşma sonuçlarından içgörüler
4. İstatistiksel modellerden çıkarımlar
5. İY/MS tahminlerinin nedenleri

Kullanım:
    from match_insights import MatchInsightsGenerator
    
    insights = MatchInsightsGenerator()
    result = insights.generate_match_insights(home_team_id, away_team_id, predictions_data)
"""

import os
import json
import random
import logging
import math
from datetime import datetime

# Logging
logger = logging.getLogger(__name__)

class MatchInsightsGenerator:
    """
    Maç içgörüleri ve doğal dil açıklamaları üreten sınıf
    """
    def __init__(self):
        """
        Sınıfı başlat ve gerekli kaynakları yükle
        """
        # İçgörü şablonları ve parametreleri
        self.templates = {
            "intro": [
                "{home_team} ile {away_team} arasındaki karşılaşmada {favorite_team} %{win_prob} olasılıkla favori görünüyor.",
                "Analizimize göre, {home_team} - {away_team} maçında {outcome_text} bekliyoruz.",
                "{home_team} - {away_team} karşılaşması için modelimiz {score_prediction} şeklinde bir skor tahmin ediyor."
            ],
            "form": [
                "{team} son {form_matches} maçında {wins} galibiyet, {draws} beraberlik ve {losses} mağlubiyet aldı.",
                "{team} şu anda {form_description} durumunda bulunuyor ve son maçlarında {recent_performance}.",
                "{team} son dönemdeki performansıyla {form_momentum} bir görüntü sergiliyor."
            ],
            "h2h": [
                "İki takım arasındaki son {h2h_matches} karşılaşmada {h2h_home_wins} kez ev sahibi, {h2h_away_wins} kez deplasman galip geldi, {h2h_draws} maç ise berabere sonuçlandı.",
                "Geçmiş karşılaşmalar incelendiğinde {h2h_dominant_team} bu rakibine karşı üstünlük kurmayı başardı.",
                "İki takım arasındaki son maçlarda {avg_goals:.1f} gol ortalaması dikkat çekiyor."
            ],
            "htft": [
                "İY/MS analizi, en olası senaryonun %{htft_highest_prob} olasılıkla {htft_highest} olduğunu gösteriyor.",
                "İlk yarı/maç sonu tahminlerimize göre {htft_insight}.",
                "Maçın ilk yarısında {first_half_prediction}, maç sonunda ise {full_time_prediction} bekliyoruz."
            ],
            "stats": [
                "{home_team}, kendi sahasında maç başına ortalama {home_goals_scored:.1f} gol atarken, {away_team} deplasmanda maç başına ortalama {away_goals_scored:.1f} gol kaydediyor.",
                "{home_team} savunması kendi evinde maç başına {home_goals_conceded:.1f} gol yerken, {away_team} deplasmanda {away_goals_conceded:.1f} gol yiyor.",
                "İstatistiklere göre {home_team} evindeki maçlarının %{home_clean_sheet_pct} kadarca kısmında kalesini gole kapattı, {away_team} ise deplasmanda %{away_clean_sheet_pct} oranında bu başarıyı gösterebildi."
            ],
            "key_insights": [
                "{key_insight_1}",
                "{key_insight_2}",
                "{key_insight_3}"
            ],
            "conclusion": [
                "Tüm faktörler değerlendirildiğinde {match_conclusion}.",
                "Sonuç olarak, bu maçta {conclusion_outcome} ihtimali yüksek görünüyor.",
                "Tahmin modelimiz bu karşılaşma için {conclusion_bet_type} öneriyor."
            ]
        }
        
        # Form tanımlamaları
        self.form_descriptions = {
            "excellent": ["mükemmel", "çok iyi", "üst düzey", "harika"],
            "good": ["iyi", "olumlu", "güçlü", "etkili"],
            "average": ["ortalama", "dengeli", "istikrarlı", "standart"],
            "poor": ["kötü", "zayıf", "düşük", "etkisiz"],
            "terrible": ["çok kötü", "berbat", "kriz", "felaket"]
        }
        
        # Momentum tanımlamaları
        self.momentum_descriptions = {
            "positive": ["yükselen", "gelişen", "iyileşen", "güçlenen"],
            "neutral": ["dengeli", "istikrarlı", "tutarlı", "sabit"],
            "negative": ["düşen", "kötüleşen", "gerileyen", "zayıflayan"]
        }
        
        # İY/MS açıklamaları
        self.htft_descriptions = {
            "1/1": "Ev sahibi takımın ilk yarıda ve maç sonunda öne geçeceği",
            "1/X": "Ev sahibi takımın ilk yarıda önde olacağı ancak maçın berabere biteceği",
            "1/2": "Ev sahibi takımın ilk yarıda önde olacağı ancak maçı deplasman takımının kazanacağı",
            "X/1": "İlk yarının berabere biteceği, ancak maçı ev sahibi takımın kazanacağı",
            "X/X": "Hem ilk yarının hem de maçın berabere biteceği",
            "X/2": "İlk yarının berabere biteceği, ancak maçı deplasman takımının kazanacağı",
            "2/1": "Deplasman takımının ilk yarıda önde olacağı ancak maçı ev sahibi takımın kazanacağı",
            "2/X": "Deplasman takımının ilk yarıda önde olacağı ancak maçın berabere biteceği",
            "2/2": "Deplasman takımının ilk yarıda ve maç sonunda öne geçeceği"
        }
        
        # Skor tahmini açıklamaları
        self.score_insights = {
            "low_scoring": [
                "düşük skorlu bir maç",
                "az gollü bir karşılaşma",
                "defansif bir mücadele",
                "gol bulmakta zorlanan takımlar"
            ],
            "high_scoring": [
                "bol gollü bir maç",
                "ofansif bir karşılaşma",
                "savunmaların zorlanacağı bir mücadele",
                "karşılıklı gol beklentisi yüksek bir maç"
            ],
            "balanced": [
                "dengeli bir karşılaşma",
                "çekişmeli bir mücadele",
                "kritik anlara sahne olabilecek bir maç",
                "tarafların birbirine üstünlük kurmakta zorlanacağı"
            ]
        }
        
    def generate_match_insights(self, home_team_id, away_team_id, predictions_data=None, additional_data=None):
        """
        Verilen maç verilerine göre doğal dil açıklamaları ve içgörüler üret
        
        Args:
            home_team_id: Ev sahibi takım ID'si
            away_team_id: Deplasman takımı ID'si
            predictions_data: Maç tahmin verileri (opsiyonel, verilmezse API'den alınır)
            additional_data: Ek maç verileri (form, h2h vs., home_team_name, away_team_name)
            
        Returns:
            dict: İçgörü ve açıklamalar içeren veri yapısı
        """
        # Log ekleyelim - takım isimlerine ilk elden bak
        if additional_data:
            home_team_name = additional_data.get('home_team_name', f"Takım {home_team_id}")
            away_team_name = additional_data.get('away_team_name', f"Takım {away_team_id}")
            logger.info(f"İçgörü oluşturma başlıyor (Takımlar: {home_team_name} vs {away_team_name})")
        
        # Takım ID'lerinin sayı türünde olmasını sağla
        try:
            if isinstance(home_team_id, str) and home_team_id.isdigit():
                home_team_id = int(home_team_id)
            if isinstance(away_team_id, str) and away_team_id.isdigit():
                away_team_id = int(away_team_id)
        except (ValueError, TypeError):
            pass  # Dönüştürme başarısız olduysa orijinal değeri kullan
            
        if additional_data is None:
            additional_data = {}
            
        # Tahmin verileri yok veya yetersizse, tahmin önbelleğinden alma deneyelim
        if not predictions_data or not isinstance(predictions_data, dict) or len(predictions_data) <= 1:
            logger.info(f"Tahmin önbelleğinden veriler alınıyor: {home_team_id}-{away_team_id}")
            try:
                from main import predictor
                # Tahmin önbelleğini kontrol et
                cache_data = predictor.load_cache()
                match_key = f"{home_team_id}-{away_team_id}"
                
                # Önbellekte bu maç var mı?
                if cache_data and match_key in cache_data:
                    logger.info(f"Önbellekte eşleşen maç bulundu: {match_key}")
                    match_data = cache_data[match_key]
                    
                    # Temel tahmin varsa kullan
                    if 'predictions' in match_data:
                        # Takım adlarını güncelle
                        if 'home_team_name' in match_data:
                            home_team_name = match_data.get('home_team_name', f"Takım {home_team_id}")
                            if additional_data:
                                additional_data['home_team_name'] = home_team_name
                        
                        if 'away_team_name' in match_data:
                            away_team_name = match_data.get('away_team_name', f"Takım {away_team_id}")
                            if additional_data:
                                additional_data['away_team_name'] = away_team_name
                        
                        # Tahmin verilerini hazırla
                        predictions_data = {
                            'base_prediction': match_data,
                            'htft_prediction': match_data.get('htft_prediction', {}),
                            'home_team_matches': {
                                'stats': match_data.get('home_form', {}),
                                'team_name': home_team_name,
                                'matches': []
                            },
                            'away_team_matches': {
                                'stats': match_data.get('away_form', {}),
                                'team_name': away_team_name,
                                'matches': []
                            },
                            'home_stats': match_data.get('home_stats', {}),
                            'away_stats': match_data.get('away_stats', {})
                        }
                        
                        # H2H verileri varsa ekle
                        if 'h2h' in match_data:
                            predictions_data['h2h'] = match_data['h2h']
                        
                        logger.info(f"Önbellekten tahmin ve form verileri başarıyla alındı: {home_team_id} vs {away_team_id}")
                else:
                    logger.warning(f"Önbellekte eşleşen maç bulunamadı: {match_key}")
            except Exception as e:
                logger.error(f"Önbellekten veriler alınırken hata: {str(e)}")
        try:
            # Tahmin verileri yoksa, gerekli bilgileri getir
            if not predictions_data:
                try:
                    from match_prediction import MatchPredictor
                    from halfTime_fullTime_predictor import predict_half_time_full_time
                    from api_routes import get_team_half_time_stats, get_team_matches
                    
                    # Temel tahmin verilerini al
                    predictor = MatchPredictor()
                    if additional_data and 'home_team_name' in additional_data and 'away_team_name' in additional_data:
                        home_team_name = additional_data['home_team_name']
                        away_team_name = additional_data['away_team_name']
                        prediction_result = predictor.predict_match(home_team_id, away_team_id, home_team_name, away_team_name, force_update=False)
                    else:
                        # Takım isimleri bilinmediğinden, varsayılan isimler kullanarak tahmin yap
                        home_team_name = f"Takım {home_team_id}"
                        away_team_name = f"Takım {away_team_id}"
                        prediction_result = predictor.predict_match(home_team_id, away_team_id, home_team_name, away_team_name, force_update=False)
                    
                    # İY/MS tahminleri ve H2H verilerini almaya çalış
                    try:
                        # Takım istatistiklerini güvenli bir şekilde al
                        home_stats_response = get_team_half_time_stats(int(home_team_id))
                        away_stats_response = get_team_half_time_stats(int(away_team_id))
                        
                        # Response tipindeki verileri işle
                        home_stats = {}
                        away_stats = {}
                        
                        # İlk olarak tuple kontrolü yap
                        if isinstance(home_stats_response, tuple) and len(home_stats_response) > 0:
                            home_stats_response = home_stats_response[0]  # İlk eleman veriyi içerir
                        if isinstance(away_stats_response, tuple) and len(away_stats_response) > 0:
                            away_stats_response = away_stats_response[0]  # İlk eleman veriyi içerir
                        
                        # Sonra Response nesnesi kontrolü yap
                        if hasattr(home_stats_response, 'json'):
                            home_stats = home_stats_response.json()
                        elif isinstance(home_stats_response, dict):
                            home_stats = home_stats_response
                            
                        if hasattr(away_stats_response, 'json'):
                            away_stats = away_stats_response.json()
                        elif isinstance(away_stats_response, dict):
                            away_stats = away_stats_response
                    except Exception as e:
                        logger.error(f"Takım istatistikleri alınırken hata: {str(e)}")
                        home_stats = {}
                        away_stats = {}
                    
                    # H2H verilerini al - prediction_result içinden
                    h2h_data = prediction_result.get('h2h', {})
                    # H2H verisi gerçekten mevcut mu kontrol et
                    logger.info(f"H2H verileri mevcut mu: {bool(h2h_data)} (Uzunluk: {len(h2h_data) if isinstance(h2h_data, dict) else 'bilinmiyor'})")
                    logger.info(f"H2H verileri mevcut mu: {bool(h2h_data)}")
                    
                    if home_stats and away_stats:
                        htft_prediction = predict_half_time_full_time(
                            home_stats, away_stats, 
                            prediction_result.get('prediction', '1-1'),
                            prediction_result.get('home_form'),
                            prediction_result.get('away_form'),
                            home_team_id, away_team_id
                        )
                    else:
                        htft_prediction = {}
                    
                    # Form bilgilerini al
                    home_team_matches_response = get_team_matches(home_team_id)
                    away_team_matches_response = get_team_matches(away_team_id)
                    
                    # Yanıtları işle
                    home_team_matches = {}
                    away_team_matches = {}
                    
                    if hasattr(home_team_matches_response, 'json'):
                        home_team_matches = home_team_matches_response.json()
                    elif isinstance(home_team_matches_response, dict):
                        home_team_matches = home_team_matches_response
                        
                    if hasattr(away_team_matches_response, 'json'):
                        away_team_matches = away_team_matches_response.json()
                    elif isinstance(away_team_matches_response, dict):
                        away_team_matches = away_team_matches_response
                    
                    # H2H (karşılıklı maç) verilerini almaya çalış
                    try:
                        import show_h2h_data
                        h2h_data = show_h2h_data.get_h2h_data(home_team_id, away_team_id, home_team_name, away_team_name)
                        logger.info(f"H2H verisi alındı: {len(h2h_data.get('h2h_matches', []))} maç bulundu")
                    except Exception as e:
                        logger.error(f"H2H verisi alınırken hata: {str(e)}")
                        h2h_data = {"h2h_matches": []}
                    
                    # Tahmin verilerini birleştir
                    predictions_data = {
                        'base_prediction': prediction_result,
                        'htft_prediction': htft_prediction,
                        'home_team_matches': home_team_matches, 
                        'away_team_matches': away_team_matches,
                        'home_stats': home_stats,
                        'away_stats': away_stats,
                        'h2h_data': h2h_data  # H2H verilerini tahmin verilerine ekle
                    }
                except Exception as e:
                    logger.error(f"Tahmin verileri oluşturulurken hata: {str(e)}")
                    # Tahmin oluşturulamadıysa boş bir sözlük kullan
                    predictions_data = {
                        'base_prediction': {},
                        'htft_prediction': {},
                        'home_team_matches': {},
                        'away_team_matches': {},
                        'home_stats': {},
                        'away_stats': {}
                    }
            
            # Temel bilgileri hazırla
            match_data = self._prepare_match_data(home_team_id, away_team_id, predictions_data, additional_data)
            
            # Maç verisi doğrulaması
            if not match_data:
                logger.warning("Hazırlanan maç verisi boş veya geçersiz")
                raise ValueError("Geçerli maç verisi oluşturulamadı")
                
            # İçgörü ve açıklamaları oluştur
            insights = self._generate_insights(match_data)
            
            # Insights doğrulaması
            if not insights or not isinstance(insights, dict):
                logger.warning(f"Oluşturulan içgörüler geçersiz: {insights}")
                raise ValueError("Geçerli içgörü oluşturulamadı")
            
            # Özet ve anahtar noktaları oluştur
            summary = self._generate_summary(insights)
            key_points = self._extract_key_points(insights, match_data)
            
            # Sonuçları döndür
            return {
                'match_data': match_data,
                'insights': insights,
                'summary': summary if summary else "Bu maç için henüz özet oluşturulmadı.",
                'key_points': key_points if key_points else []
            }
            
        except Exception as e:
            logger.error(f"İçgörü oluşturulurken hata: {str(e)}")
            # Takım isimlerini additional_data'dan al
            home_team_name = None
            away_team_name = None
            
            if additional_data:
                home_team_name = additional_data.get('home_team_name')
                away_team_name = additional_data.get('away_team_name')
            
            if not home_team_name:
                home_team_name = f"Takım {home_team_id}"
            
            if not away_team_name:
                away_team_name = f"Takım {away_team_id}"
                
            # Basit bir fallback açıklama döndür
            return {
                'match_data': {
                    'home_team': home_team_name,
                    'away_team': away_team_name,
                    'date': datetime.now().strftime("%Y-%m-%d")
                },
                'insights': {
                    'general': f"Bu maç için tahmin analizleri oluşturulurken bir sorun oluştu: {str(e)}"
                },
                'summary': "Maç analizi şu anda kullanılamıyor.",
                'key_points': []
            }
    
    def _prepare_match_data(self, home_team_id, away_team_id, predictions_data, additional_data=None):
        """
        Maç verilerini hazırla
        
        Args:
            home_team_id: Ev sahibi takım ID'si
            away_team_id: Deplasman takımı ID'si
            predictions_data: Maç tahmin verileri
            additional_data: Ek maç verileri (opsiyonel)
            
        Returns:
            dict: Hazırlanmış maç verileri
        """
        logger.info(f"Tahmin verilerini hazırlama: Ev sahibi ID {home_team_id}, Deplasman ID {away_team_id}")
        
        # Gelen veri hiç yoksa veya boş bir sözlük ise
        if not predictions_data or not isinstance(predictions_data, dict):
            logger.warning(f"Geçerli tahmin verisi bulunamadı: {predictions_data}")
            predictions_data = {
                'base_prediction': {},
                'htft_prediction': {},
                'home_team_matches': {},
                'away_team_matches': {},
                'home_stats': {},
                'away_stats': {}
            }
        
        # Tahmin verilerinden temel bilgileri çıkar - güvenli erişim
        base_prediction = predictions_data.get('base_prediction', {})
        if not isinstance(base_prediction, dict):
            logger.warning(f"base_prediction geçerli bir sözlük değil: {base_prediction}")
            base_prediction = {}
            
        htft_prediction = predictions_data.get('htft_prediction', {})
        if not isinstance(htft_prediction, dict):
            logger.warning(f"htft_prediction geçerli bir sözlük değil: {htft_prediction}")
            htft_prediction = {}
            
        home_team_matches = predictions_data.get('home_team_matches', {})
        if not isinstance(home_team_matches, dict):
            logger.warning(f"home_team_matches geçerli bir sözlük değil: {home_team_matches}")
            home_team_matches = {}
            
        away_team_matches = predictions_data.get('away_team_matches', {})
        if not isinstance(away_team_matches, dict):
            logger.warning(f"away_team_matches geçerli bir sözlük değil: {away_team_matches}")
            away_team_matches = {}
            
        home_stats = predictions_data.get('home_stats', {})
        if not isinstance(home_stats, dict):
            logger.warning(f"home_stats geçerli bir sözlük değil: {home_stats}")
            home_stats = {}
            
        away_stats = predictions_data.get('away_stats', {})
        if not isinstance(away_stats, dict):
            logger.warning(f"away_stats geçerli bir sözlük değil: {away_stats}")
            away_stats = {}
        
        # Takım isimleri - öncelikle additional_data'dan kontrol et, sonra base_prediction'dan
        home_team_name = None
        away_team_name = None
        
        # Önce ek verilere bakalım (main.py'den gelen)
        if additional_data:
            home_team_name = additional_data.get('home_team_name')
            away_team_name = additional_data.get('away_team_name')
        
        # Eğer ek verilerde yoksa tahmin verisinden bakalım    
        if not home_team_name:
            home_team_name = base_prediction.get('home_team_name', f"Takım {home_team_id}")
        
        if not away_team_name:
            away_team_name = base_prediction.get('away_team_name', f"Takım {away_team_id}")
        
        # Temel tahminler - hepsi için string veya sayı olabilir, güvenle sayıya çevirelim
        try:
            home_win_prob = float(base_prediction.get('home_win_probability', 33))
        except (ValueError, TypeError):
            home_win_prob = 33
            
        try:
            draw_prob = float(base_prediction.get('draw_probability', 34))
        except (ValueError, TypeError):
            draw_prob = 34
            
        try:
            away_win_prob = float(base_prediction.get('away_win_probability', 33)) 
        except (ValueError, TypeError):
            away_win_prob = 33
        
        # Favorinin belirlenmesi
        favorite_team = home_team_name
        favorite_win_prob = home_win_prob
        
        # Sayısal karşılaştırma yapalım (artık hepsi float)
        if away_win_prob > home_win_prob and away_win_prob > draw_prob:
            favorite_team = away_team_name
            favorite_win_prob = away_win_prob
        elif draw_prob > home_win_prob and draw_prob > away_win_prob:
            favorite_team = "Beraberlik"
            favorite_win_prob = draw_prob
        
        # En olası skor - olası string değerlerden korunalım
        try:
            home_goals = float(base_prediction.get('expected_home_goals', 1.2))
        except (ValueError, TypeError):
            home_goals = 1.2
            
        try:
            away_goals = float(base_prediction.get('expected_away_goals', 1.0))
        except (ValueError, TypeError):
            away_goals = 1.0
        
        # En olası İY/MS - string veya sayı olabilir
        htft_highest_prob = 0.0
        htft_highest = "1/1"
        if htft_prediction:
            for key, value in htft_prediction.items():
                try:
                    # String olarak gelmişse sayıya çevir
                    if isinstance(value, str):
                        value = float(value)
                    # Int veya float ise doğrudan karşılaştır
                    elif isinstance(value, (int, float)):
                        value = float(value)
                    else:
                        continue
                        
                    if value > htft_highest_prob:
                        htft_highest_prob = value
                        htft_highest = key
                except (ValueError, TypeError):
                    # Sayıya çevrilemezse geç
                    continue
        
        # Form analizi
        home_form = self._analyze_team_form(home_team_matches)
        away_form = self._analyze_team_form(away_team_matches)
        
        # H2H analizi (karşılıklı maçlar)
        # predictions_data'dan h2h verilerini kontrol et
        h2h_data = predictions_data.get('h2h_data', {})
        
        # Eğer predictions_data içinde h2h verisi yoksa, base_prediction içindeki h2h verilerine bak
        if not h2h_data or not isinstance(h2h_data, dict) or not h2h_data.get('h2h_matches'):
            logger.info("H2H verisi predictions_data içinde bulunamadı, base_prediction içinde aranıyor")
            h2h_data = base_prediction.get('h2h', {})
        
        # H2H analizini gerçekleştir
        h2h_analysis = self._analyze_h2h(h2h_data, home_team_name, away_team_name)
        
        # IY/MS detaylı analiz
        htft_analysis = self._analyze_htft(htft_prediction, home_team_name, away_team_name)
        
        # Skor tahmini
        score_prediction = self._generate_score_prediction(home_goals, away_goals)
        
        # Maç verilerini hazırla
        match_data = {
            'date': datetime.now().strftime("%Y-%m-%d"),
            'home_team': home_team_name,
            'away_team': away_team_name,
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'favorite_team': favorite_team,
            'favorite_win_prob': favorite_win_prob,
            'outcome_probs': {
                'home_win': home_win_prob,
                'draw': draw_prob,
                'away_win': away_win_prob
            },
            'expected_goals': {
                'home': home_goals,
                'away': away_goals,
                'total': home_goals + away_goals
            },
            'score_prediction': score_prediction,
            'htft': htft_prediction,
            'htft_highest': htft_highest,
            'htft_highest_prob': htft_highest_prob,
            'htft_analysis': htft_analysis,
            'home_form': home_form,
            'away_form': away_form,
            'h2h': h2h_analysis,
            'home_stats': home_stats,
            'away_stats': away_stats
        }
        
        # Ek veriler varsa ekle
        if additional_data:
            match_data.update(additional_data)
        
        return match_data
    
    def _analyze_team_form(self, team_matches):
        """
        Takımın form durumunu analiz et
        
        Args:
            team_matches: Takım maç verileri
            
        Returns:
            dict: Form analizi
        """
        # Varsayılan değerler
        form_data = {
            'matches': 5,
            'wins': 2,
            'draws': 1,
            'losses': 2,
            'goals_scored': 7,
            'goals_conceded': 6,
            'form_score': 0.5,  # 0-1 arası, 1 en iyi form
            'form_description': 'iyi',
            'form_trend': 'kararlı',
            'form_momentum': 'orta',
            'recent_performance': 'istikrarlı bir performans sergiliyor',
            'form_text': 'Takım ortalama bir form durumunda.'
        }
        
        # Takım maç verileri yoksa varsayılan değerleri döndür
        if not team_matches or not isinstance(team_matches, dict) or 'matches' not in team_matches:
            logger.warning(f"Geçerli takım maç verisi bulunamadı: {team_matches}")
            return form_data
        
        matches = team_matches.get('matches', [])
        if not matches or not isinstance(matches, list):
            return form_data
        
        # Son 5 maç analizi
        recent_matches = matches[:5] if len(matches) >= 5 else matches
        
        wins = 0
        draws = 0
        losses = 0
        goals_scored = 0
        goals_conceded = 0
        form_trend = []  # 3=galibiyet, 1=beraberlik, 0=mağlubiyet
        
        # Son maçları analiz et
        for match in recent_matches:
            goals_for = match.get('goals_scored', 0)
            goals_against = match.get('goals_conceded', 0)
            
            goals_scored += goals_for
            goals_conceded += goals_against
            
            if goals_for > goals_against:
                wins += 1
                form_trend.append(3)
            elif goals_for == goals_against:
                draws += 1
                form_trend.append(1)
            else:
                losses += 1
                form_trend.append(0)
                
        # Toplam maç sayısı
        total_matches = len(recent_matches)
        
        # Form puanı hesapla (0-1 arası)
        if total_matches > 0:
            form_score = (wins * 3 + draws) / (total_matches * 3)
        else:
            form_score = 0.5  # Varsayılan
        
        # Form tanımı belirle
        form_description = 'average'
        if form_score >= 0.8:
            form_description = 'excellent'
        elif form_score >= 0.6:
            form_description = 'good'
        elif form_score >= 0.4:
            form_description = 'average'
        elif form_score >= 0.2:
            form_description = 'poor'
        else:
            form_description = 'terrible'
            
        # Form momentum belirle
        form_momentum = 'neutral'
        if len(form_trend) >= 3:
            if form_trend[0] > form_trend[-1]:  # Son maç ilk maçtan kötüyse
                form_momentum = 'negative'
            elif form_trend[0] < form_trend[-1]:  # Son maç ilk maçtan iyiyse
                form_momentum = 'positive'
                
        # Son performans tanımı
        recent_performance_text = self._generate_recent_performance_text(form_trend)
                
        # Form metin açıklaması
        form_text = f"Takım {random.choice(self.form_descriptions[form_description])} bir form durumunda bulunuyor."
        form_text += f" Son {total_matches} maçında {wins} galibiyet, {draws} beraberlik ve {losses} mağlubiyet aldı."
        
        if form_momentum != 'neutral':
            form_text += f" Form trendi {random.choice(self.momentum_descriptions[form_momentum])} bir grafik çiziyor."
        
        # Form verilerini güncelle
        form_data.update({
            'matches': total_matches,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'goals_scored': goals_scored,
            'goals_conceded': goals_conceded,
            'form_score': form_score,
            'form_description': form_description,
            'form_momentum': form_momentum,
            'recent_performance': recent_performance_text,
            'form_text': form_text
        })
        
        return form_data
    
    def _generate_recent_performance_text(self, form_trend):
        """
        Son performans açıklaması oluştur
        
        Args:
            form_trend: Form trend bilgisi (3=galibiyet, 1=beraberlik, 0=mağlubiyet)
            
        Returns:
            str: Performans açıklaması
        """
        if not form_trend:
            return "istikrarlı bir performans sergiliyor"
            
        # Son 3 maç kazanılmış mı
        if len(form_trend) >= 3 and all(result == 3 for result in form_trend[:3]):
            return "son 3 maçını kazanarak yüksek bir form yakaladı"
            
        # Son 3 maç kaybedilmiş mi
        if len(form_trend) >= 3 and all(result == 0 for result in form_trend[:3]):
            return "son 3 maçını kaybederek kötü bir form sergiliyor"
            
        # Son maç kazanılmış mı
        if form_trend[0] == 3:
            return "son maçını kazanarak moralli bir şekilde sahaya çıkıyor"
            
        # Son maç kaybedilmiş mi
        if form_trend[0] == 0:
            return "son maçını kaybederek moral bozukluğu yaşıyor"
            
        # Beraberlikler fazla mı
        if form_trend.count(1) > len(form_trend) / 2:
            return "beraberliklerle puan kaybediyor"
            
        # Varsayılan
        return "değişken bir performans gösteriyor"
    
    def _analyze_h2h(self, h2h_data, home_team_name, away_team_name):
        """
        H2H (karşılıklı maç) verilerini analiz et
        
        Args:
            h2h_data: Karşılıklı maç verileri
            home_team_name: Ev sahibi takım adı
            away_team_name: Deplasman takımı adı
            
        Returns:
            dict: H2H analizi
        """
        # Varsayılan değerler
        h2h_analysis = {
            'total_matches': 0,
            'home_wins': 0,
            'draws': 0,
            'away_wins': 0,
            'home_goals': 0,
            'away_goals': 0,
            'avg_goals': 0,
            'dominant_team': None,
            'h2h_text': 'İki takım arasındaki geçmiş maçlar için yeterli veri bulunmuyor.'
        }
        
        # H2H verileri yoksa varsayılan değerleri döndür
        if not h2h_data or not isinstance(h2h_data, dict):
            return h2h_analysis
        
        matches = h2h_data.get('matches', [])
        if not matches or not isinstance(matches, list):
            return h2h_analysis
        
        # Toplam maç sayısı
        total_matches = len(matches)
        if total_matches == 0:
            return h2h_analysis
        
        # İstatistikleri hesapla
        home_wins = h2h_data.get('home_wins', 0)
        away_wins = h2h_data.get('away_wins', 0)
        draws = h2h_data.get('draws', 0)
        home_goals = h2h_data.get('home_goals', 0)
        away_goals = h2h_data.get('away_goals', 0)
        
        # Ortalama gol sayısı
        avg_goals = (home_goals + away_goals) / total_matches if total_matches > 0 else 0
        
        # Baskın takımı belirle
        dominant_team = None
        if home_wins > away_wins + draws:
            dominant_team = home_team_name
        elif away_wins > home_wins + draws:
            dominant_team = away_team_name
        else:
            dominant_team = "Beraberlik"
            
        # H2H metin açıklaması
        h2h_text = ""
        if total_matches > 0:
            h2h_text = f"İki takım arasındaki son {total_matches} karşılaşmada "
            h2h_text += f"{home_wins} kez {home_team_name}, {away_wins} kez {away_team_name} galip geldi, "
            h2h_text += f"{draws} maç ise berabere sonuçlandı. "
            
            if dominant_team != "Beraberlik":
                h2h_text += f"Karşılıklı maçlarda {dominant_team} üstünlük kurmayı başarıyor. "
                
            h2h_text += f"İki takım arasındaki maçlarda maç başına ortalama {avg_goals:.1f} gol atılıyor."
        else:
            h2h_text = "İki takım arasındaki geçmiş maçlar için yeterli veri bulunmuyor."
        
        # H2H verilerini güncelle
        h2h_analysis.update({
            'total_matches': total_matches,
            'home_wins': home_wins,
            'draws': draws,
            'away_wins': away_wins,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'avg_goals': avg_goals,
            'dominant_team': dominant_team,
            'h2h_text': h2h_text
        })
        
        return h2h_analysis
    
    def _analyze_htft(self, htft_prediction, home_team_name, away_team_name):
        """
        İY/MS tahminlerini analiz et
        
        Args:
            htft_prediction: İY/MS tahmin verileri
            home_team_name: Ev sahibi takım adı
            away_team_name: Deplasman takımı adı
            
        Returns:
            dict: İY/MS analizi
        """
        # Varsayılan değerler
        htft_analysis = {
            'highest': '1/1',
            'highest_prob': 0,
            'htft_text': 'İY/MS analizi için yeterli veri bulunmuyor.',
            'first_half': {
                '1': 0,  # Ev sahibi önde
                'X': 0,  # Beraberlik
                '2': 0   # Deplasman önde
            },
            'full_time': {
                '1': 0,  # Ev sahibi kazanır
                'X': 0,  # Beraberlik
                '2': 0   # Deplasman kazanır
            }
        }
        
        # İY/MS verileri yoksa varsayılan değerleri döndür
        if not htft_prediction or not isinstance(htft_prediction, dict):
            return htft_analysis
        
        try:
            # En yüksek olasılıklı İY/MS kombinasyonunu bul
            highest_prob = 0
            highest_htft = '1/1'
            
            for htft, prob in htft_prediction.items():
                if isinstance(prob, (int, float)) and prob > highest_prob:
                    highest_prob = prob
                    highest_htft = htft
                    
            # İlk yarı ve tam maç olasılıklarını hesapla
            first_half = {'1': 0.0, 'X': 0.0, '2': 0.0}
            full_time = {'1': 0.0, 'X': 0.0, '2': 0.0}
            
            for htft, prob in htft_prediction.items():
                try:
                    # String olarak gelmişse sayıya çevir
                    if isinstance(prob, str):
                        prob = float(prob)
                    # Int veya float ise doğrudan kullan
                    elif isinstance(prob, (int, float)):
                        prob = float(prob)
                    else:
                        continue
                        
                    parts = htft.split('/')
                    if len(parts) == 2:
                        first_half_result = parts[0]
                        full_time_result = parts[1]
                        
                        if first_half_result in first_half:
                            first_half[first_half_result] += prob
                            
                        if full_time_result in full_time:
                            full_time[full_time_result] += prob
                except (ValueError, TypeError):
                    # Sayıya çevrilemezse geç
                    continue
            
            # İY/MS metin açıklaması
            htft_text = ""
            if hasattr(self, 'htft_descriptions') and isinstance(self.htft_descriptions, dict) and highest_htft in self.htft_descriptions:
                description = self.htft_descriptions[highest_htft]
                htft_text = f"İY/MS analizi, en olası senaryonun %{highest_prob} olasılıkla {description} olduğunu gösteriyor. "
                
                # İlk yarı analizi
                first_half_highest = '1'
                first_half_prob = 0.0
                
                # Sözlük üzerinde manuel döngü ile en yüksek değeri bulma
                for result, prob in first_half.items():
                    try:
                        prob_float = float(prob)
                        if prob_float > first_half_prob:
                            first_half_prob = prob_float
                            first_half_highest = result
                    except (ValueError, TypeError):
                        continue
                
                if first_half_highest == '1':
                    htft_text += f"İlk yarıda %{first_half_prob:.0f} olasılıkla {home_team_name} önde olabilir. "
                elif first_half_highest == '2':
                    htft_text += f"İlk yarıda %{first_half_prob:.0f} olasılıkla {away_team_name} önde olabilir. "
                else:
                    htft_text += f"İlk yarının %{first_half_prob:.0f} olasılıkla berabere bitmesi bekleniyor. "
                    
                # Tam maç analizi
                full_time_highest = '1'
                full_time_prob = 0.0
                
                # Sözlük üzerinde manuel döngü ile en yüksek değeri bulma
                for result, prob in full_time.items():
                    try:
                        prob_float = float(prob)
                        if prob_float > full_time_prob:
                            full_time_prob = prob_float
                            full_time_highest = result
                    except (ValueError, TypeError):
                        continue
                
                if full_time_highest == '1':
                    htft_text += f"Maç sonunda %{full_time_prob:.0f} olasılıkla {home_team_name} galip gelebilir."
                elif full_time_highest == '2':
                    htft_text += f"Maç sonunda %{full_time_prob:.0f} olasılıkla {away_team_name} galip gelebilir."
                else:
                    htft_text += f"Maçın %{full_time_prob:.0f} olasılıkla berabere bitmesi bekleniyor."
            else:
                htft_text = "İY/MS analizi için yeterli veri bulunmuyor."
            
            # İY/MS verilerini güncelle
            htft_analysis.update({
                'highest': highest_htft,
                'highest_prob': highest_prob,
                'htft_text': htft_text,
                'first_half': first_half,
                'full_time': full_time
            })
            
        except Exception as e:
            logger.error(f"İY/MS analizi sırasında hata: {str(e)}")
            # Hata olursa varsayılan değerleri döndür
            
        return htft_analysis
    
    def _generate_score_prediction(self, home_goals, away_goals):
        """
        Skor tahmini oluştur - geliştirilmiş çeşitlilik sağlayan algoritma
        
        Args:
            home_goals: Ev sahibi beklenen gol sayısı
            away_goals: Deplasman beklenen gol sayısı
            
        Returns:
            str: Skor tahmini
        """
        # Gol beklentileriyle başlayalım
        try:
            home_goals_float = float(home_goals) if home_goals is not None else 1.0
            away_goals_float = float(away_goals) if away_goals is not None else 0.8
        except (ValueError, TypeError):
            home_goals_float = 1.0
            away_goals_float = 0.8
        
        # Her seferinde aynı skoru döndürmemek için random seed belirleme
        random.seed(datetime.now().timestamp())
        
        # AI içgörülerinin 1-1 ve beraberlik takıntısını önlemek için ek bir randomizasyon
        # ekleyen algoritma. Bu sayede sadece matematiksel beklentilere değil, farklı 
        # skorlar üretmeyi de hedefler
        
        # 1. Poisson dağılımı kullanarak gol olasılıklarını hesapla
        from math import exp, factorial
        
        # Monte Carlo simulasyonu için ev sahibi ve deplasman skorlarını üret
        def poisson_random(lambda_val):
            L = exp(-lambda_val)
            k = 0
            p = 1.0
            
            while p > L:
                k += 1
                p *= random.random()
                
            return k - 1 if k > 0 else 0
            
        # 2. Olası skorlar üretme
        home_scores = []
        away_scores = []
        
        # Randomize factor - modele çeşitlilik katar (0.75-1.25 arası çarpan)
        home_factor = 0.75 + (random.random() * 0.5)
        away_factor = 0.75 + (random.random() * 0.5)
        
        # Gol sayılarını ayarlayalım
        home_goals_adj = home_goals_float * home_factor
        away_goals_adj = away_goals_float * away_factor
        
        # Simülasyon bazlı çeşitli skorlar üret
        for _ in range(5):
            h_score = poisson_random(home_goals_adj)
            a_score = poisson_random(away_goals_adj)
            home_scores.append(h_score)
            away_scores.append(a_score)
        
        # Tam muhtemel skorları hesapla - 1-1 skorunu azaltmak için
        # rastgele belirlenen bir skoru döndür
        if random.random() < 0.7:
            # %70 ihtimalle beklenti değerlerine göre yuvarlama
            home_score = round(home_goals_adj)
            away_score = round(away_goals_adj)
            
            # 1-1 skoru oluşursa, rastgele başka bir skor seç
            if home_score == 1 and away_score == 1:
                if random.random() < 0.6:
                    # %60 ihtimalle simülasyon skorlarından birini kullan
                    idx = random.randint(0, len(home_scores) - 1)
                    home_score = home_scores[idx]
                    away_score = away_scores[idx]
                else:
                    # %40 ihtimalle hafifçe düzenlenmiş bir skor üret
                    options = [(2, 1), (1, 2), (1, 0), (0, 1), (2, 0), (0, 2)]
                    home_score, away_score = random.choice(options)
        else:
            # %30 ihtimalle simülasyon skorlarından birini kullan
            idx = random.randint(0, len(home_scores) - 1)
            home_score = home_scores[idx]
            away_score = away_scores[idx]
        
        # Yüksek skorları sınırla
        if home_score > 3:
            home_score = min(home_score, 3 + (0 if random.random() < 0.8 else 1))
            
        if away_score > 3:
            away_score = min(away_score, 3 + (0 if random.random() < 0.8 else 1))
        
        # Skor tahmini döndür
        return f"{home_score}-{away_score}"
    
    def _generate_insights(self, match_data):
        """
        Maç verilerine göre içgörüler oluştur
        
        Args:
            match_data: Maç verileri
            
        Returns:
            dict: İçgörüler
        """
        # Temel verileri çıkar (hata önleyici kontroller ile)
        home_team = match_data.get('home_team', 'Ev Sahibi')
        away_team = match_data.get('away_team', 'Deplasman')
        favorite_team = match_data.get('favorite_team', home_team)
        win_prob = match_data.get('favorite_win_prob', 50)
        
        # String olarak gelen değerleri sayıya çevirmeyi dene (hata olduysa varsayılan değeri kullan)
        if isinstance(win_prob, str):
            try:
                win_prob = float(win_prob)
            except (ValueError, TypeError):
                win_prob = 50
        
        # Form verilerini kontrol et - form_score değeri olup olmadığına bak
        # Yoksa varsayılan değerlerle oluştur
        home_form = match_data.get('home_form', {})
        if not isinstance(home_form, dict) or 'form_score' not in home_form:
            home_form = {'form_score': 0.5, 'form_trend': 'kararlı', 'form_momentum': 'orta'}
            
        away_form = match_data.get('away_form', {})
        if not isinstance(away_form, dict) or 'form_score' not in away_form:
            away_form = {'form_score': 0.5, 'form_trend': 'kararlı', 'form_momentum': 'orta'}
        h2h = match_data.get('h2h', {})
        htft_analysis = match_data.get('htft_analysis', {})
        
        # Sonuç tahmini (1X2) - güvenli erişim ve iyileştirilmiş algoritmalarla dengeli dağılım
        outcome_text = "beraberlik"
        outcome_probs = match_data.get('outcome_probs', {'home_win': 35, 'draw': 30, 'away_win': 35})
        
        # Sayısal değerleri veri tipi güvenli şekilde al
        try:
            home_win = float(outcome_probs.get('home_win', 35))
            draw = float(outcome_probs.get('draw', 30))
            away_win = float(outcome_probs.get('away_win', 35))
            
            # Her durumda tahmin yapabilmesi için 33-34-33 dağılımından uzaklaşalım
            # Rastgele değerler ekleyerek beraberlik tahmini olasılığını azaltalım
            form_difference = abs(home_form.get('form_score', 0.5) - away_form.get('form_score', 0.5))
            random_factor = random.uniform(0, 10)  # 0-10 arası rastgele değer
            
            # Form farkı büyükse ev sahibi veya deplasman galibiyeti olasılığını artır
            if form_difference > 0.2:
                if home_form.get('form_score', 0.5) > away_form.get('form_score', 0.5):
                    home_win += random_factor
                    draw -= (random_factor / 2)
                    away_win -= (random_factor / 2)
                else:
                    away_win += random_factor
                    draw -= (random_factor / 2)
                    home_win -= (random_factor / 2)
            else:
                # Form farkı küçükse rastgele ev sahibi veya deplasman galibiyeti olasılığını artır
                if random.random() > 0.5:
                    home_win += random_factor
                    draw -= (random_factor / 2)
                    away_win -= (random_factor / 2)
                else:
                    away_win += random_factor
                    draw -= (random_factor / 2)
                    home_win -= (random_factor / 2)
            
            # Negatif olasılıkları engelle
            home_win = max(home_win, 10)
            draw = max(draw, 10)
            away_win = max(away_win, 10)
            
            # Toplam olasılık 100 olacak şekilde normalize et
            total = home_win + draw + away_win
            home_win = (home_win / total) * 100
            draw = (draw / total) * 100
            away_win = (away_win / total) * 100
            
        except (ValueError, TypeError):
            home_win, draw, away_win = 35, 30, 35
        
        # En yüksek olasılığa göre sonuç metnini belirle
        if home_win > draw and home_win > away_win:
            outcome_text = f"{home_team} galibiyeti"
        elif away_win > draw and away_win > home_win:
            outcome_text = f"{away_team} galibiyeti"
        else:
            outcome_text = "beraberlik"
            
        # Toplam gol analizi - güvenli erişim
        expected_goals = match_data.get('expected_goals', {'home': 1.0, 'away': 1.0, 'total': 2.0})
        
        # Sayısal değerleri veri tipi güvenli şekilde al
        try:
            total_expected_goals = float(expected_goals.get('total', 2.0))
        except (ValueError, TypeError):
            total_expected_goals = 2.0
            
        score_category = 'balanced'
        if total_expected_goals < 2.0:
            score_category = 'low_scoring'
        elif total_expected_goals > 3.0:
            score_category = 'high_scoring'
            
        # İçgörüleri oluştur
        insights = {
            'general': self._format_template('intro', match_data, {
                'home_team': home_team,
                'away_team': away_team,
                'favorite_team': favorite_team,
                'win_prob': round(win_prob),
                'outcome_text': outcome_text,
                'score_prediction': match_data['score_prediction']
            }),
            
            'home_form': self._format_template('form', match_data, {
                'team': home_team,
                'form_matches': home_form.get('matches', 5),
                'wins': home_form.get('wins', 2),
                'draws': home_form.get('draws', 1),
                'losses': home_form.get('losses', 2),
                'form_description': random.choice(self.form_descriptions.get(home_form.get('form_description', 'iyi'), ["iyi"])),
                'recent_performance': home_form.get('recent_performance', 'istikrarlı bir performans sergiliyor'),
                'form_momentum': random.choice(self.momentum_descriptions.get(home_form.get('form_momentum', 'orta'), ["orta"]))
            }),
            
            'away_form': self._format_template('form', match_data, {
                'team': away_team,
                'form_matches': away_form.get('matches', 5),
                'wins': away_form.get('wins', 2),
                'draws': away_form.get('draws', 1),
                'losses': away_form.get('losses', 2),
                'form_description': random.choice(self.form_descriptions.get(away_form.get('form_description', 'iyi'), ["iyi"])),
                'recent_performance': away_form.get('recent_performance', 'istikrarlı bir performans sergiliyor'),
                'form_momentum': random.choice(self.momentum_descriptions.get(away_form.get('form_momentum', 'orta'), ["orta"]))
            }),
            
            'h2h': self._format_template('h2h', match_data, {
                'h2h_matches': h2h.get('total_matches', 0),
                'h2h_home_wins': h2h.get('home_wins', 0),
                'h2h_away_wins': h2h.get('away_wins', 0),
                'h2h_draws': h2h.get('draws', 0),
                'h2h_dominant_team': h2h.get('dominant_team', "Beraberlik") if h2h.get('dominant_team', "Beraberlik") != "Beraberlik" else "genellikle beraberlik",
                'avg_goals': h2h.get('avg_goals', 0)
            }),
            
            'htft': self._format_template('htft', match_data, {
                'htft_highest': self.htft_descriptions.get(htft_analysis.get('highest', '1/1'), htft_analysis.get('highest', '1/1')),
                'htft_highest_prob': htft_analysis.get('highest_prob', 25),
                'htft_insight': self._generate_htft_insight(htft_analysis, home_team, away_team),
                'first_half_prediction': self._generate_half_prediction(htft_analysis.get('first_half', {'1': 33, 'X': 34, '2': 33}), home_team, away_team, 'first_half'),
                'full_time_prediction': self._generate_half_prediction(htft_analysis.get('full_time', {'1': 33, 'X': 34, '2': 33}), home_team, away_team, 'full_time')
            }),
            
            'stats': self._format_template('stats', match_data, {
                'home_team': home_team,
                'away_team': away_team,
                'home_goals_scored': expected_goals.get('home', 1.0),
                'away_goals_scored': expected_goals.get('away', 1.0),
                'home_goals_conceded': 1.0,  # Varsayılan değer
                'away_goals_conceded': 1.2,  # Varsayılan değer
                'home_clean_sheet_pct': 30,  # Varsayılan değer
                'away_clean_sheet_pct': 20   # Varsayılan değer
            }),
            
            'score_insight': random.choice(self.score_insights[score_category]),
            
            'key_insights': self._format_template('key_insights', match_data, {
                'key_insight_1': self._generate_key_insight(match_data, 1),
                'key_insight_2': self._generate_key_insight(match_data, 2),
                'key_insight_3': self._generate_key_insight(match_data, 3)
            }),
            
            'conclusion': self._format_template('conclusion', match_data, {
                'match_conclusion': self._generate_match_conclusion(match_data),
                'conclusion_outcome': outcome_text,
                'conclusion_bet_type': self._generate_conclusion_bet_type(match_data)
            })
        }
        
        return insights
    
    def _format_template(self, template_name, match_data, params):
        """
        Belirtilen şablonu formatla
        
        Args:
            template_name: Şablon adı
            match_data: Maç verileri
            params: Şablon parametreleri
            
        Returns:
            str: Formatlanmış şablon
        """
        if template_name not in self.templates:
            return ""
            
        # Şablonu rastgele seç
        template = random.choice(self.templates[template_name])
        
        # Şablonu formatla
        try:
            return template.format(**params)
        except Exception as e:
            logger.error(f"Şablon formatlanırken hata: {str(e)}")
            return template
    
    def _generate_htft_insight(self, htft_analysis, home_team, away_team):
        """
        İY/MS içgörüsü oluştur
        
        Args:
            htft_analysis: İY/MS analizi
            home_team: Ev sahibi takım adı
            away_team: Deplasman takımı adı
            
        Returns:
            str: İY/MS içgörüsü
        """
        if not htft_analysis or not isinstance(htft_analysis, dict) or 'highest' not in htft_analysis:
            return "yeterli veri bulunmuyor"
            
        # En olası İY/MS'nin açıklaması
        highest_htft = htft_analysis.get('highest', '1/1')
        if not highest_htft or not isinstance(highest_htft, str):
            return "belirsiz bir sonuç bekleniyor"
            
        # Şablonu bul veya varsayılan bir metin döndür
        if highest_htft in self.htft_descriptions:
            return self.htft_descriptions[highest_htft]
        else:
            # İY/MS kodunu parçala ve her biri için açıklama oluştur
            parts = highest_htft.split('/')
            if len(parts) == 2:
                first_half, full_time = parts
                first_desc = "Ev sahibi önde" if first_half == "1" else "Deplasman önde" if first_half == "2" else "Beraberlik"
                full_desc = "Ev sahibi galip" if full_time == "1" else "Deplasman galip" if full_time == "2" else "Beraberlik"
                return f"İlk yarı {first_desc}, maç sonu {full_desc} olarak sonuçlanabilir"
            else:
                return "belirsiz bir sonuç bekleniyor"
    
    def _generate_half_prediction(self, half_probs, home_team, away_team, half_type):
        """
        Yarı tahmini oluştur
        
        Args:
            half_probs: Yarı olasılıkları
            home_team: Ev sahibi takım adı
            away_team: Deplasman takımı adı
            half_type: Yarı tipi ('first_half' veya 'full_time')
            
        Returns:
            str: Yarı tahmini
        """
        if not half_probs:
            return "belirsiz bir sonuç"
            
        # En olası sonucu bul
        highest_result = '1'
        highest_prob = 0.0
        
        # Sözlük üzerinde manuel döngü ile en yüksek değeri bulma
        for result, prob in half_probs.items():
            try:
                prob_float = float(prob)
                if prob_float > highest_prob:
                    highest_prob = prob_float
                    highest_result = result
            except (ValueError, TypeError):
                continue
        
        # Yarı tipi
        half_name = "ilk yarıda" if half_type == 'first_half' else "maç sonunda"
        
        # Sonuç açıklaması
        if highest_result == '1':
            return f"{half_name} %{highest_prob:.0f} olasılıkla {home_team} önde olacak"
        elif highest_result == '2':
            return f"{half_name} %{highest_prob:.0f} olasılıkla {away_team} önde olacak"
        else:
            return f"{half_name} %{highest_prob:.0f} olasılıkla beraberlik olacak"
    
    def _generate_key_insight(self, match_data, insight_number):
        """
        Anahtar içgörü oluştur
        
        Args:
            match_data: Maç verileri
            insight_number: İçgörü numarası
            
        Returns:
            str: Anahtar içgörü
        """
        home_team = match_data['home_team']
        away_team = match_data['away_team']
        
        # Form karşılaştırması (insight 1)
        if insight_number == 1:
            home_form_score = match_data.get('home_form', {}).get('form_score', 0.5)
            away_form_score = match_data.get('away_form', {}).get('form_score', 0.5)
            
            if abs(home_form_score - away_form_score) > 0.3:
                better_team = home_team if home_form_score > away_form_score else away_team
                worse_team = away_team if home_form_score > away_form_score else home_team
                return f"{better_team}, {worse_team}'e göre belirgin şekilde daha iyi form sergiliyor."
            else:
                return "İki takım da yakın form durumunda bulunuyor."
                
        # Gol beklentisi (insight 2)
        elif insight_number == 2:
            expected_goals = match_data.get('expected_goals', {'home': 1.0, 'away': 1.0, 'total': 2.0})
            total_goals = expected_goals.get('total', 2.0)
            score_prediction = match_data.get('score_prediction', '1-1')
            
            if total_goals < 2.0:
                return f"Maçta {score_prediction} gibi düşük bir skor bekleniyor, toplam gol sayısı muhtemelen az olacak."
            elif total_goals > 3.0:
                return f"Maçta {score_prediction} gibi yüksek bir skor bekleniyor, karşılıklı gol ihtimali yüksek."
            else:
                return f"Maçta {score_prediction} gibi dengeli bir skor bekleniyor."
                
        # İY/MS özel durumlar (insight 3)
        else:
            htft_highest = match_data.get('htft_highest', 'X/X')
            
            # Tersine dönüş senaryoları
            if htft_highest == '1/2':
                return f"{home_team} ilk yarıda önde olsa bile, {away_team} ikinci yarıda toparlanıp maçı kazanabilir."
            elif htft_highest == '2/1':
                return f"{away_team} ilk yarıda önde olsa bile, {home_team} ikinci yarıda toparlanıp maçı kazanabilir."
            elif htft_highest == 'X/1':
                return f"İlk yarı berabere bittikten sonra {home_team} ikinci yarıda üstünlük kurabilir."
            elif htft_highest == 'X/2':
                return f"İlk yarı berabere bittikten sonra {away_team} ikinci yarıda üstünlük kurabilir."
            else:
                h2h = match_data.get('h2h', {'total_matches': 0, 'avg_goals': 0})
                h2h_total = h2h.get('total_matches', 0)
                if h2h_total > 3:
                    h2h_avg_goals = h2h.get('avg_goals', 0)
                    if h2h_avg_goals > 2.5:
                        return f"İki takım arasındaki son maçlarda maç başına {h2h_avg_goals:.1f} gol atıldı, bu maçta da gol beklentisi yüksek."
                    else:
                        return f"İki takım arasındaki son maçlarda maç başına {h2h_avg_goals:.1f} gol atıldı, bu maçta da az gol olabilir."
                else:
                    return "İki takımın form durumları ve istatistikleri dikkate alındığında çekişmeli bir maç bekleniyor."
    
    def _generate_match_conclusion(self, match_data):
        """
        Maç sonucu hakkında genel bir sonuç oluştur
        
        Args:
            match_data: Maç verileri
            
        Returns:
            str: Maç sonucu
        """
        home_team = match_data.get('home_team', 'Ev Sahibi')
        away_team = match_data.get('away_team', 'Deplasman')
        
        # Güvenli erişim
        outcome_probs = match_data.get('outcome_probs', {'home_win': 33, 'draw': 34, 'away_win': 33})
        home_win_prob = outcome_probs.get('home_win', 33)
        draw_prob = outcome_probs.get('draw', 34)
        away_win_prob = outcome_probs.get('away_win', 33)
        
        # En olası sonuç
        if home_win_prob > draw_prob and home_win_prob > away_win_prob:
            if home_win_prob > 60:
                return f"{home_team}'in kazanma olasılığı yüksek"
            else:
                return f"{home_team}'in hafif favori olduğu bir maç bekleniyor"
        elif away_win_prob > draw_prob and away_win_prob > home_win_prob:
            if away_win_prob > 60:
                return f"{away_team}'in kazanma olasılığı yüksek"
            else:
                return f"{away_team}'in hafif favori olduğu bir maç bekleniyor"
        else:
            return "beraberlik ihtimali yüksek olan dengeli bir maç bekleniyor"
    
    def _generate_conclusion_bet_type(self, match_data):
        """
        Sonuç bahis tipi önerisi oluştur
        
        Args:
            match_data: Maç verileri
            
        Returns:
            str: Bahis tipi önerisi
        """
        # Güvenli erişim
        outcome_probs = match_data.get('outcome_probs', {'home_win': 33, 'draw': 34, 'away_win': 33})
        home_win_prob = outcome_probs.get('home_win', 33)
        draw_prob = outcome_probs.get('draw', 34)
        away_win_prob = outcome_probs.get('away_win', 33)
        
        # Beklenen gol değerlerini güvenli al
        expected_goals = match_data.get('expected_goals', {'home': 1.0, 'away': 1.0, 'total': 2.0})
        total_goals = expected_goals.get('total', 2.0)
        home_expected = expected_goals.get('home', 1.0)
        away_expected = expected_goals.get('away', 1.0)
        
        # Bahis önerisi - En yüksek olasılığı bul
        max_prob = home_win_prob
        if draw_prob > max_prob:
            max_prob = draw_prob
        if away_win_prob > max_prob:
            max_prob = away_win_prob
            
        if max_prob > 50:
            if home_win_prob > draw_prob and home_win_prob > away_win_prob:
                return "MS 1"
            elif away_win_prob > draw_prob and away_win_prob > home_win_prob:
                return "MS 2"
            else:
                return "MS X"
        elif total_goals > 2.8:
            return "2.5 Üst"
        elif total_goals < 2.0:
            return "2.5 Alt"
        else:
            # Karşılıklı gol durumu
            if home_expected > 0.8 and away_expected > 0.8:
                return "KG Var"
            else:
                return "KG Yok"
    
    def _generate_summary(self, insights):
        """
        İçgörülerden özet oluştur
        
        Args:
            insights: İçgörüler
            
        Returns:
            str: Özet
        """
        summary = insights.get('general', '')
        
        # Ev sahibi ve deplasman form bilgilerini ekle
        home_form = insights.get('home_form', '')
        away_form = insights.get('away_form', '')
        
        if home_form and away_form:
            summary += f" {home_form} {away_form}"
            
        # H2H bilgilerini ekle
        h2h = insights.get('h2h', '')
        if h2h:
            summary += f" {h2h}"
            
        # Skor içgörüsünü ekle
        score_insight = insights.get('score_insight', '')
        if score_insight:
            summary += f" Bu maçta {score_insight} bekliyoruz."
            
        # Sonuç bilgilerini ekle
        conclusion = insights.get('conclusion', '')
        if conclusion:
            summary += f" {conclusion}"
            
        return summary
    
    def _extract_key_points(self, insights, match_data):
        """
        İçgörülerden anahtar noktaları çıkar
        
        Args:
            insights: İçgörüler
            match_data: Maç verileri
            
        Returns:
            list: Anahtar noktalar
        """
        key_points = []
        
        # Genel tahmin bilgisi
        if 'general' in insights:
            key_points.append(insights['general'])
            
        # İY/MS analizi
        if 'htft' in insights:
            key_points.append(insights['htft'])
            
        # Anahtar içgörüler
        if 'key_insights' in insights:
            key_insights = insights['key_insights'].split('.')
            for insight in key_insights:
                if insight and len(insight.strip()) > 10:
                    key_points.append(f"{insight.strip()}.")
                    
        # Skor içgörüsü
        if 'score_insight' in insights:
            key_points.append(f"Bu maçta {insights['score_insight']} bekliyoruz.")
            
        # Sonuç
        if 'conclusion' in insights:
            key_points.append(insights['conclusion'])
            
        return key_points