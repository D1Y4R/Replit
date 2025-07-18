import pickle
import json
import os

print("ML Model Dosyaları Analizi\n" + "="*50)

# PKL dosyalarını kontrol et
pkl_files = [
    "assets/FootballPredictionHub_backup/extracted/FootballPredictionHub/away_ensemble_model.pkl",
    "assets/FootballPredictionHub_backup/extracted/FootballPredictionHub/away_poisson_model.pkl", 
    "assets/FootballPredictionHub_backup/extracted/FootballPredictionHub/home_ensemble_model.pkl",
    "assets/FootballPredictionHub_backup/extracted/FootballPredictionHub/home_poisson_model.pkl",
    "assets/FootballPredictionHub_backup/extracted/FootballPredictionHub/models/crf_model.pkl"
]

for pkl_file in pkl_files:
    if os.path.exists(pkl_file):
        try:
            with open(pkl_file, 'rb') as f:
                model = pickle.load(f)
                print(f"\n📁 {os.path.basename(pkl_file)}")
                print(f"   Tip: {type(model).__name__}")
                print(f"   Boyut: {os.path.getsize(pkl_file)} bytes")
                
                # Model tipine göre detayları yazdır
                if hasattr(model, 'get_params'):
                    print(f"   Sklearn Model: {model.__class__.__name__}")
                if hasattr(model, 'feature_importances_'):
                    print(f"   Feature sayısı: {len(model.feature_importances_)}")
                if hasattr(model, 'classes_'):
                    print(f"   Sınıflar: {model.classes_}")
                    
        except Exception as e:
            print(f"\n❌ {os.path.basename(pkl_file)} - Hata: {str(e)}")

# JSON model dosyasını kontrol et  
json_file = "assets/FootballPredictionHub_backup/extracted/FootballPredictionHub/self_learning_model.json"
if os.path.exists(json_file):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            print(f"\n📄 self_learning_model.json")
            print(f"   İçerik: {list(data.keys()) if isinstance(data, dict) else 'List'}")
            if isinstance(data, dict):
                for key in list(data.keys())[:3]:
                    print(f"   - {key}: {type(data[key]).__name__}")
    except Exception as e:
        print(f"\n❌ self_learning_model.json - Hata: {str(e)}")

print("\n\nŞimdi hangi modellerin sistemimize entegre edilebileceğini kontrol ediyorum...")

# XGBoost model dosyalarını ara
print("\n🔍 XGBoost model dosyaları aranıyor...")
for root, dirs, files in os.walk("assets/FootballPredictionHub_backup/extracted"):
    for file in files:
        if 'xgb' in file.lower() or 'xgboost' in file.lower():
            print(f"   ✓ Bulundu: {os.path.join(root, file)}")