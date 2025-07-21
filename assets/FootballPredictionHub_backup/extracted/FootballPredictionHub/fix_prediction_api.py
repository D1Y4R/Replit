import json
import logging
import requests
from flask import jsonify

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_factors():
    """Gelişmiş faktörlerin API yanıtında bulunup bulunmadığını test et"""
    try:
        # Test tahmin verisini yükle
        with open('./data/test_prediction.json', 'r', encoding='utf-8') as f:
            test_prediction = json.load(f)
            
        logger.info("Test tahmini yüklendi")
        
        # Gelişmiş faktörleri kontrol et
        if 'enhanced_factors' in test_prediction:
            logger.info("✅ Test tahmini gelişmiş faktörleri içeriyor")
            
            # Faktör detaylarını logla
            if 'match_importance' in test_prediction['enhanced_factors']:
                logger.info(f"Maç önemi: {test_prediction['enhanced_factors']['match_importance']['description']}")
                
            if 'historical_pattern' in test_prediction['enhanced_factors']:
                logger.info(f"Tarihsel örüntü: {test_prediction['enhanced_factors']['historical_pattern']['description']}")
                
            if 'momentum' in test_prediction['enhanced_factors']:
                logger.info(f"Momentum: {test_prediction['enhanced_factors']['momentum']['description']}")
            
            return True
        else:
            logger.warning("❌ Test tahmini gelişmiş faktörleri içermiyor")
            return False
            
    except Exception as e:
        logger.error(f"Test sırasında hata: {str(e)}")
        return False
        
def check_prediction_structure(prediction):
    """Tahmin yapısını kontrol et ve eksik alanları raporla"""
    required_fields = [
        'match', 
        'home_team', 
        'away_team', 
        'predictions', 
        'enhanced_factors'
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in prediction:
            missing_fields.append(field)
            
    if missing_fields:
        logger.warning(f"❌ Eksik alanlar: {', '.join(missing_fields)}")
        return False
    else:
        logger.info("✅ Tüm gerekli alanlar mevcut")
        return True
        
def update_predictions_cache():
    """predictions_cache.json dosyasını güncelle - test için"""
    try:
        # Mevcut önbelleği yükle
        try:
            with open('predictions_cache.json', 'r', encoding='utf-8') as f:
                cache = json.load(f)
                logger.info(f"Mevcut önbellek yüklendi: {len(cache)} tahmin")
        except (FileNotFoundError, json.JSONDecodeError):
            cache = {}
            logger.warning("Mevcut önbellek yüklenemedi, yeni önbellek oluşturuluyor")
            
        # Test tahminini yükle
        with open('./data/test_prediction.json', 'r', encoding='utf-8') as f:
            test_prediction = json.load(f)
            
        # Önbelleğe ekle
        cache_key = f"{test_prediction['home_team']['id']}_{test_prediction['away_team']['id']}"
        cache[cache_key] = test_prediction
        logger.info(f"Test tahmini önbelleğe eklendi: {cache_key}")
        
        # Güncellenmiş önbelleği kaydet
        with open('predictions_cache.json', 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
            logger.info("Önbellek güncellendi")
            
        return True
    except Exception as e:
        logger.error(f"Önbellek güncellenirken hata: {str(e)}")
        return False
        
def test_api_response():
    """API yanıtını test et"""
    try:
        # API'ye istek gönder
        url = "http://localhost:80/api/predict-match/610/1005?home_name=Galatasaray&away_name=Fenerbahce"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            prediction = response.json()
            logger.info("API yanıtı alındı")
            
            # Yanıt yapısını kontrol et
            if check_prediction_structure(prediction):
                logger.info("✅ API yanıtı doğru yapıda")
                return True
            else:
                logger.warning("❌ API yanıtı hatalı yapıda")
                return False
        else:
            logger.error(f"API isteği başarısız: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"API testi sırasında hata: {str(e)}")
        return False
        
if __name__ == "__main__":
    logger.info("Gelişmiş tahmin faktörleri testi başlatılıyor...")
    
    # Test tahminini kontrol et
    if test_enhanced_factors():
        logger.info("✅ Test tahmini kontrol edildi")
        
        # Önbelleği güncelle
        if update_predictions_cache():
            logger.info("✅ Önbellek güncellendi")
            
            # API yanıtını test et
            if test_api_response():
                logger.info("✅ Tüm testler başarılı!")
            else:
                logger.warning("❌ API testi başarısız")
        else:
            logger.warning("❌ Önbellek güncellemesi başarısız")
    else:
        logger.warning("❌ Test tahmini kontrolü başarısız")