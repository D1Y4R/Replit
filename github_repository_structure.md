# FootballPredictionHub - GitHub Repository Structure

## 📁 Ana Klasörler (Main Folders)

### 🧠 **algorithms/** - Tahmin Algoritmaları
- `__init__.py` - Python paket başlatıcı
- `ensemble.py` - Ensemble tahmin modeli
- `team_goals_predictor.py` - Takım gol tahmincisi
- `self_learning.py` - Kendi kendini öğrenen model
- `halftime_predictor.py` - İlk yarı tahmincisi
- `xg_calculator.py` - Expected Goals hesaplayıcı
- `dixon_coles.py` - Dixon-Coles modeli
- `poisson_model.py` - Poisson dağılım modeli
- `htft_surprise_detector.py` - HT/FT sürpriz dedektörü
- `crf_predictor.py` - Conditional Random Field tahmincisi
- `neural_network.py` - Yapay sinir ağı modeli
- `goal_range_predictor.py` - Gol aralığı tahmincisi
- `xgboost_model.py` - XGBoost makine öğrenmesi modeli
- `extreme_detector.py` - Ekstrem durum dedektörü
- `double_chance_predictor.py` - Çifte şans tahmincisi
- `handicap_predictor.py` - Handikap tahmincisi
- `monte_carlo.py` - Monte Carlo simülasyonu
- `elo_system.py` - ELO rating sistemi

### 🎨 **templates/** - HTML Şablonları
- `base.html` - Ana şablon
- `index.html` - Ana sayfa
- `fixtures.html` - Fikstür sayfası
- `api_v3.html` - API v3 dokümantasyonu
- `cache_table.html` - Önbellek tablosu
- `football_data.html` - Futbol verisi sayfası
- `leagues.html` - Ligler sayfası
- `match_insights.html` - Maç içgörüleri
- `model_validation.html` - Model doğrulama
- `predictions.html` - Tahminler sayfası

### 🎨 **static/** - Statik Dosyalar
#### **css/** - Stil Dosyaları
- `custom.css` - Özel stiller
- `match-actions.css` - Maç aksiyonları
- `match-insights.css` - Maç içgörüleri
- `prediction-modal.css` - Tahmin modalı
- `widget-style.css` - Widget stilleri
- `widgetCountries.css` - Ülke widget'ı
- `widgetLeague.css` - Lig widget'ı

#### **js/** - JavaScript Dosyaları
- `api_football_debug.js` - API debug
- `custom.js` - Özel JavaScript
- `fixed_custom.js` - Düzeltilmiş özel JS
- `insights-popup.js` - İçgörü popup'ı
- `jquery.widgetCountries.js` - Ülke widget'ı
- `jquery.widgetLeague.js` - Lig widget'ı
- `main.js` - Ana JavaScript
- `prediction-handler.js` - Tahmin işleyici
- `prediction-handler.js.bak` - Tahmin işleyici yedek
- `prediction-popup-template-v2.js` - Tahmin popup şablonu
- `team_history.js` - Takım geçmişi
- `team_stats.js` - Takım istatistikleri

#### **img/** - Resimler
- `default-league.png` - Varsayılan lig resmi
- `default-team.png` - Varsayılan takım resmi

#### **images/** - Resimler
- `team-placeholder.svg` - Takım placeholder

### 🤖 **models/** - Eğitilmiş Modeller
- `xgb_1x2.json` - XGBoost 1X2 modeli
- `self_learning_model.json` - Kendi kendini öğrenen model
- `crf_model.pkl` - CRF modeli

### 📊 **data/** - Veri Dosyaları
- `test_prediction.json` - Test tahmin verisi

### 💾 **backups/** - Yedekler
- `custom.js.bak` - Özel JS yedeği

### 📁 **attached_assets/** - Ekli Varlıklar
- Çeşitli raporlar ve dokümantasyon dosyaları
- API dokümantasyonu
- Ekran görüntüleri

### 📁 **assets/** - Varlıklar
- Yedek dosyalar ve arşivler

### 📁 **backtest/** - Geri Test
- Geri test dosyaları

## 📄 Ana Dosyalar (Main Files)

### 🚀 **Core Application Files**
- `main.py` - Ana uygulama dosyası (43KB, 1052 satır)
- `api_routes.py` - API rotaları (61KB, 1414 satır)
- `match_prediction.py` - Maç tahmin sistemi (52KB, 1223 satır)

### 🧠 **AI/ML Files**
- `dynamic_team_analyzer.py` - Dinamik takım analizörü (18KB, 493 satır)
- `adaptation_tracker.py` - Adaptasyon takipçisi (18KB, 506 satır)
- `situational_analyzer.py` - Durumsal analizör (14KB, 415 satır)
- `tactical_profiler.py` - Taktik profilleyici (11KB, 310 satır)
- `momentum_analyzer.py` - Momentum analizörü (8.7KB, 271 satır)
- `match_categorizer.py` - Maç kategorize edici (10KB, 301 satır)
- `model_performance_tracker.py` - Model performans takipçisi (10KB, 260 satır)
- `performance_optimizer.py` - Performans optimizörü (19KB, 593 satır)
- `dynamic_weight_calculator.py` - Dinamik ağırlık hesaplayıcı (8.3KB, 236 satır)

### 📊 **Advanced Features**
- `advanced_features.py` - Gelişmiş özellikler (31KB, 801 satır)
- `continuous_learner.py` - Sürekli öğrenen sistem (13KB, 298 satır)
- `distributed_trainer.py` - Dağıtık eğitici (29KB, 807 satır)
- `explainable_ai.py` - Açıklanabilir AI (30KB, 725 satır)
- `model_evaluator.py` - Model değerlendirici (12KB, 283 satır)
- `model_validator.py` - Model doğrulayıcı (27KB, 719 satır)
- `train_models.py` - Model eğitici (19KB, 536 satır)

### 🔧 **Configuration & Setup**
- `api_config.py` - API konfigürasyonu (5.6KB)
- `api_config.json` - API ayarları (129B)
- `pyproject.toml` - Python proje ayarları (761B)
- `uv.lock` - Bağımlılık kilidi (416KB, 2599 satır)
- `.replit` - Replit konfigürasyonu (797B)

### 📊 **Data & Cache**
- `predictions_cache.json` - Tahmin önbelleği (516KB, 16436 satır)
- `prediction.json` - Tahmin verisi (68KB)
- `performance_metrics.json` - Performans metrikleri (4.0KB, 197 satır)
- `team_performance.db` - Takım performans veritabanı (48KB)
- `team_specific_config.json` - Takım özel ayarları (86B)

### 📝 **Documentation**
- `replit.md` - Replit dokümantasyonu (19KB, 307 satır)
- `DEVELOPMENT_ROADMAP.md` - Geliştirme yol haritası (2.9KB, 101 satır)
- `DYNAMIC_TEAM_ANALYZER_ROADMAP.md` - Dinamik takım analizörü yol haritası (3.8KB, 132 satır)
- `SMART_ENSEMBLE_ROADMAP.md` - Akıllı ensemble yol haritası (3.2KB, 103 satır)
- `ML_LEARNING_STATUS.md` - ML öğrenme durumu (503B, 18 satır)
- `project_files.md` - Proje dosyaları listesi (14KB)

### 🧪 **Testing Files**
- `test_dynamic_team_analyzer.py` - Dinamik takım analizörü testi (5.6KB, 163 satır)
- `test_dynamic_ensemble.py` - Dinamik ensemble testi (6.0KB, 181 satır)
- `test_dynamic_confidence.py` - Dinamik güven testi (2.1KB, 64 satır)
- `test_full_prediction.py` - Tam tahmin testi (2.3KB, 64 satır)
- `test_h2h_fix.py` - H2H düzeltme testi (2.4KB, 64 satır)
- `test_app.py` - Uygulama testi (2.6KB)
- `test_app_log.txt` - Test log dosyası (792B)
- `test_prediction.json` - Test tahmin verisi (1.3KB)

### 🔍 **Debug & Analysis**
- `async_data_fetcher.py` - Asenkron veri getirici (20KB, 544 satır)
- `debug_h2h_api.py` - H2H API debug (2.8KB, 89 satır)
- `quick_h2h_test.py` - Hızlı H2H testi (1.7KB, 49 satır)
- `check_h2h_data.py` - H2H veri kontrolü (2.6KB, 56 satır)
- `check_team_ids.py` - Takım ID kontrolü (2.0KB, 46 satır)
- `check_models.py` - Model kontrolü (2.6KB, 57 satır)
- `check_js_syntax.py` - JS syntax kontrolü (1.6KB, 49 satır)
- `fix_team_ids.py` - Takım ID düzeltme (4.6KB, 136 satır)

### 📊 **Reports & Analysis**
- `h2h_usage_report.md` - H2H kullanım raporu (2.9KB, 62 satır)
- `HTFT_rapor.md` - HT/FT raporu (4.3KB)
- `ekstrem_maclar_roadmap.md` - Ekstrem maçlar yol haritası (846B)
- `ekstrem_maclar_oneriler.md` - Ekstrem maçlar önerileri (3.6KB)
- `buriram_sabah_skor_analizi.md` - Buriram sabah skor analizi (2.3KB)

### 🛠️ **Utility Files**
- `create_project_zip.py` - Proje zip oluşturucu (1.5KB)
- `display_cache.py` - Önbellek görüntüleyici (6.6KB)
- `football_api_config.py` - Futbol API konfigürasyonu (957B)
- `grok_helper.py` - Grok yardımcısı (1.5KB)
- `add_validation_data.py` - Doğrulama verisi ekleyici (3.0KB)
- `fixed_prediction_handler.js` - Düzeltilmiş tahmin işleyici (16KB)
- `fixed_prediction_handler_part2.js` - Düzeltilmiş tahmin işleyici 2 (18KB)

### 📄 **HTML Files (Root)**
- `api_v3.html` - API v3 dokümantasyonu (12KB, 253 satır)
- `base.html` - Ana şablon (4.3KB, 85 satır)
- `cache_table.html` - Önbellek tablosu (5.6KB, 107 satır)
- `fixtures.html` - Fikstür sayfası (72KB, 1534 satır)
- `football_data.html` - Futbol verisi sayfası (4.9KB, 148 satır)
- `leagues.html` - Ligler sayfası (3.3KB, 75 satır)
- `match_insights.html` - Maç içgörüleri (25KB)
- `model_validation.html` - Model doğrulama (101KB)
- `predictions.html` - Tahminler sayfası (43KB)

### 📄 **Other Files**
- `algorithm_code.html` - Algoritma kodu HTML (225KB)
- `generated-icon.png` - Oluşturulan ikon (235KB)
- `leicester_newcastle_prediction.json` - Leicester-Newcastle tahmini (71KB)
- `nohup.out` - Nohup çıktısı (7.3KB)

## 📊 Repository İstatistikleri

- **Toplam Dosya Sayısı:** 2364 dosya
- **Toplam Boyut:** ~34.5 MB
- **Ana Proje Klasörü:** FootballPredictionHub
- **Teknoloji Stack:** Python, Flask, JavaScript, HTML/CSS
- **AI/ML Modelleri:** XGBoost, Neural Networks, CRF, Ensemble Models
- **Veri Kaynakları:** API-Football, Özel veri setleri
- **Özellikler:** Maç tahmini, Takım analizi, Performans takibi, Gerçek zamanlı veri

Bu repository, gelişmiş futbol tahmin algoritmaları ve makine öğrenmesi modelleri içeren kapsamlı bir futbol analiz platformudur.