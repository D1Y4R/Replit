# Football Prediction Hub

## Overview

This Football Prediction Hub is a comprehensive football match prediction system that combines advanced machine learning models, statistical analysis, and real-time data to provide accurate match predictions. The system specializes in various prediction types including exact scores, half-time/full-time outcomes, betting predictions, and performance analytics.

## User Preferences

Preferred communication style: Simple, everyday language.
Interface language: Turkish
Priority: Automatic fixture data refresh when API keys are updated

## System Architecture

The application follows a modular Flask-based architecture with multiple specialized prediction models and data processing components:

### Core Architecture
- **Backend Framework**: Flask web application with RESTful API endpoints
- **Data Processing**: Real-time football data fetching from multiple APIs
- **Prediction Engine**: Multi-model ensemble approach with specialized algorithms
- **Caching Layer**: JSON-based prediction caching for performance optimization
- **Model Validation**: Cross-validation and backtesting capabilities

## Key Components

### 1. Main Application (`main.py`)
- Flask web server with routing and request handling
- Integration point for all prediction models
- API endpoint management for match data and predictions
- Template rendering for web interface

### 2. Match Prediction Engine (`match_prediction.py`)
- Core prediction logic using multiple algorithms
- Poisson distribution modeling for goal predictions
- Monte Carlo simulations for match outcomes
- Integration with advanced ML models when available
- Betting predictions (over/under, both teams to score, exact scores)

### 3. Advanced ML Models (`advanced_ml_models.py`)
- XGBoost gradient boosting models
- LSTM neural networks for time series analysis
- Bayesian networks for probabilistic predictions
- TensorFlow/Keras deep learning implementations

### 4. Specialized Prediction Models
- **Half-Time/Full-Time Predictor**: Specialized algorithms for HT/FT outcomes
- **Team-Specific Models**: Customized predictions based on team characteristics
- **Enhanced Monte Carlo**: Improved simulation with Dixon-Coles corrections
- **CRF Predictor**: Pre-trained Conditional Random Fields model for structured 1X2 predictions
- **Self-Learning Model**: Adaptive model with dynamic weight adjustment based on match context
- **Extreme Match Detector**: Intelligent detection and handling of high-scoring matches

### 5. Data Management
- **API Integration**: Multiple football data APIs (Football-Data.org, API-Football)
- **Dynamic Team Analyzer**: Real-time team performance analysis
- **Team Performance Updater**: Automated data refresh mechanisms
- **Prediction Caching**: JSON-based caching system for performance

### 6. Model Validation and Learning
- **Model Validator**: Cross-validation and backtesting capabilities
- **Self-Learning Predictor**: Adaptive model that learns from prediction accuracy with dynamic weight adjustment
- **Performance Monitoring**: Continuous model performance tracking
- **Pre-trained Models**: Integrated CRF and self-learning models from backup system with proven accuracy

### 7. Enhanced Analysis Features
- **Goal Trend Analyzer**: Momentum and scoring pattern analysis
- **Enhanced Prediction Factors**: Match importance and historical pattern analysis
- **Match Insights Generator**: Natural language explanations of predictions

## Data Flow

1. **Data Acquisition**: Real-time match data fetched from external APIs
2. **Data Processing**: Team statistics, form analysis, and historical data compilation
3. **Feature Engineering**: Creation of prediction features from raw data
4. **Model Execution**: Multiple prediction models run in parallel
5. **Result Aggregation**: Ensemble methods combine individual model predictions
6. **Consistency Checking**: Validation of prediction logic and internal consistency
7. **Caching**: Results stored for performance and historical analysis
8. **API Response**: JSON-formatted predictions returned to frontend

## External Dependencies

### APIs
- **Football-Data.org API**: Primary source for match fixtures and results
- **API-Football**: Secondary data source for comprehensive coverage
- **XAI API**: Integration with Grok AI for advanced analysis (optional)

### Python Libraries
- **Flask**: Web framework and API development
- **TensorFlow/Keras**: Deep learning models
- **XGBoost**: Gradient boosting algorithms
- **scikit-learn**: Traditional ML algorithms and validation
- **NumPy/Pandas**: Data manipulation and analysis
- **SciPy**: Statistical distributions and analysis

### Frontend Technologies
- **Bootstrap**: UI framework with dark theme support
- **jQuery**: DOM manipulation and AJAX requests
- **Chart.js**: Data visualization (when available)

## Deployment Strategy

### Development Environment
- **Replit**: Primary development and hosting platform
- **SQLite**: Local database for team performance data
- **JSON Files**: Configuration and caching storage
- **Environment Variables**: API keys and sensitive configuration

### Production Considerations
- **Database Migration**: Ready for PostgreSQL integration via Drizzle ORM
- **Scalability**: Modular architecture supports horizontal scaling
- **API Rate Limiting**: Built-in handling for external API constraints
- **Error Handling**: Comprehensive error management and fallback mechanisms

### Key Features
- **Real-time Predictions**: Live match prediction generation
- **Multiple Prediction Types**: Exact scores, match outcomes, betting predictions
- **Model Ensemble**: Combination of multiple prediction algorithms
- **Performance Tracking**: Continuous model validation and improvement
- **Responsive Design**: Mobile-friendly web interface
- **API Documentation**: RESTful API for external integration
- **Centralized API Key Management**: Dynamic API key configuration with automatic system updates
- **Auto-refresh Fixtures**: Automatic fixture data reload when API keys are updated

### Recent Changes (July 18, 2025 - Latest Update)
- **Dynamic Team Analyzer Sistemi**: Kapsamlı takım analiz sistemi implementasyonu
  - **4 Çekirdek Modül**: Momentum Analyzer, Tactical Profiler, Situational Analyzer, Adaptation Tracker
  - **Momentum Analizi**: 
    - Son 10 maç form trendi ve psikolojik momentum (0-100 skor)
    - Ev/deplasman performans ayrımı
    - Gol atma/yeme trend analizi
    - Mevcut seri takibi (galibiyet/beraberlik/yenilgi serileri)
  - **Taktiksel Profil**:
    - Takım oyun stili belirleme (attacking_high_press, defensive_counter, balanced vb.)
    - Tempo analizi (very_fast, fast, medium, slow, very_slow)
    - Savunma sağlamlığı değerlendirmesi
    - Set parça etkinliği ve yarı performansları
  - **Durumsal Faktörler**:
    - Rakip gücüne göre performans analizi (üst/alt sıra takımlara karşı)
    - Büyük maç performansı ve kritik maç yönetimi
    - Motivasyon seviyesi hesaplama (0-100)
    - Özel durumlar: title_race, relegation_battle, european_race
  - **Adaptasyon Takibi**:
    - Form evrimi ve tutarlılık trendi
    - Taktiksel değişim tespiti ve başarı oranı
    - Gelişim hızı hesaplama (atak/savunma)
    - Rakip uyum matrisi
  - **Tahmin Ayarlamaları**: Her analiz sonucu otomatik tahmin düzeltmeleri
    - goals_expectation: ±0.5 gol ayarlaması
    - btts_probability: ±15% KG olasılık ayarı
    - over_2_5_probability: ±10% 2.5 üst/alt ayarı
    - confidence_modifier: ±10% güven düzeltmesi
  - **Takım Karşılaştırma**: İki takım arasında momentum, taktik, motivasyon analizi
  - **Maç Dinamikleri**: one_sided, tactical_battle, balanced paternleri
  - **Sürpriz Potansiyeli**: low, medium, high, very_high seviyeleri

### Recent Changes (July 18, 2025 - Earlier)
- **Akıllı Dinamik Ensemble Sistemi**: Sabit model ağırlıkları yerine dinamik ağırlık hesaplama sistemi implementasyonu
  - **Model Performans Takibi**: Her modelin lig ve maç tipi bazlı başarı oranlarını takip eden sistem
  - **Maç Kategorilendirme**: Maçları lig tipi (yüksek/düşük/orta skorlu), takım profili ve özel durumlara göre kategorize eden modül
  - **Dinamik Ağırlık Hesaplama**: Maç özelliklerine ve model performanslarına göre optimal ağırlıkları hesaplayan motor
  - **Lig Kategorileri**:
    - Yüksek skorlu: Bundesliga, Eredivisie, MLS (Poisson ve Monte Carlo ağırlığı artar)
    - Düşük skorlu: Serie A, Ligue 1, La Liga (Dixon-Coles ve CRF ağırlığı artar)
    - Orta skorlu: Premier League, Süper Lig (Dengeli dağılım)
  - **Özel Durumlar**: Derbi, kupa maçı, sezon başı/sonu, hava durumu faktörleri
  - **Performans Faktörleri**: 0.7-1.3 arası dinamik çarpanlar
  - **Maksimum Sapma**: Temel ağırlıklardan %30'dan fazla sapma engellenmiş
  - **Geriye Dönük Uyumluluk**: Eski sistem fallback olarak korunmuş
  - **KG (BTTS) Tahminleri**: Tüm modellerin ağırlıklı ortalaması ile hesaplanıyor

### Recent Changes (July 15, 2025 - Latest Update #3)
- **Logarithmic Lambda Cross Calculation**: Implemented logarithmic correction for lambda cross calculation
  - Replaced static home/away advantage (1.1/0.9) with dynamic logarithmic adjustment
  - Formula: lambda = xG * xGA * (1 ± 0.1 * log(home_xG/away_xG + 1))
  - More sensitive to team strength differences
  - Better balance between favorites and underdogs
  - Reduces extreme predictions while maintaining realistic goal expectations
  - Elo-based adjustment only applies in extreme cases (50+ Elo difference)

### Recent Changes (July 15, 2025 - Latest Update #2)
- **Advanced HT/FT Surprise Detection System**: Implemented sophisticated surprise detection exclusively for HT/FT predictions
  - Created dedicated `HTFTSurpriseDetector` module with 10 different surprise factors
  - Analyzes momentum reversals, fatigue impact, psychological pressure, and tactical adaptations
  - Detects second-half specialist teams and comeback potential
  - Evaluates historical H2H surprises and pressure handling abilities
  - Dynamically adjusts HT/FT probabilities when high surprise potential is detected
  - Increases accuracy for surprise results like HOME_AWAY, AWAY_HOME, HOME_DRAW patterns
  - Machine learning features: LSTM pattern recognition for second-half changes
  - Gradient Boosting with specialized features for first-half vs second-half performance
  - Ensemble weight adjustment based on surprise indicators
  - Completely isolated from other prediction types - only affects HT/FT predictions

### Recent Changes (July 15, 2025 - Latest Update)
- **Enhanced Data Collection for HT/FT Predictions**: Removed all data limitations to use maximum available data
  - HalfTimeFullTimePredictor now uses all available matches instead of last 20
  - Momentum calculation uses all available matches instead of last 10
  - Match prediction API fetches all matches from last 120 days (no 30-match limit)
  - XGCalculator increased from 30-match limit to unlimited (999)
  - Neural Network, CRF Predictor algorithms now use all available data
  - This ensures more accurate predictions with larger data samples
  - Better statistical significance for HT/FT and other predictions

### Recent Changes (July 14, 2025 - Latest Update #5)
- **Mobile-Responsive Design Optimization**: Enhanced mobile experience for Explanation and H2H sections
  - Completely redesigned Explanation tab with responsive CSS grid and mobile-first approach
  - Added animated confidence meter with pulse effect
  - Improved touch targets and spacing for mobile users
  - Redesigned H2H section with modern stat cards and better mobile layout
  - Added media queries for all components (768px and 1024px breakpoints)
  - Reduced font sizes and padding for mobile devices
  - Enhanced visual hierarchy with better color gradients
  - Optimized factor items with flex layout for better mobile readability
  - SWOT analysis now uses responsive grid (1 column mobile, 2 columns tablet, auto-fit desktop)

### Recent Changes (July 14, 2025 - Latest Update #4)
- **H2H (Head-to-Head) Card Added**: Added historical matchup data to prediction popup
  - New H2H card displays last 10 years of matches between teams
  - Shows win/draw/loss statistics and average goals per match
  - Integrated async data fetching for H2H data from API
  - Added visual match history with dates, scores, and league information
  - Displays BTTS percentage for historical matches
  - H2H data automatically included in all new predictions

### Recent Changes (July 14, 2025 - Latest Update #3)
- **Critical Bug Fix**: Fixed match outcome probabilities not summing to 100%
  - Added normalization for 1X2 (home/draw/away) probabilities in ensemble system
  - Ensured all match outcomes always total exactly 100%
  - Applied same normalization pattern already used for BTTS and Over/Under markets
  - Cleared prediction cache to regenerate all predictions with correct probabilities

### Recent Changes (July 14, 2025 - Latest Update #2)
- **Explainable AI UI Enhancement**: Added new "Açıklama" (Explanation) tab as 6th tab in prediction popup
  - Confidence level meter with visual percentage display
  - Key factors with positive/negative/neutral impact indicators
  - SWOT analysis display (Strengths, Weaknesses, Opportunities, Threats)
  - Natural language explanation section with formatted text
  - Removed redundant "Tahmin Analizi" card from bottom of all tabs
  - Modern gradient design matching existing dark theme

### Recent Changes (July 14, 2025 - Latest Update)
- **Comprehensive System Enhancement**: Implemented all 8 core improvement modules as specified in development roadmap
- **Model Evaluator Integration**: Added automated model performance tracking and real-time accuracy analysis
- **Continuous Learning System**: Implemented adaptive learning system that updates models based on prediction accuracy
- **Advanced Feature Engineering**: Integrated sophisticated feature extraction including form momentum, goal trends, and team confidence metrics
- **Distributed Training Support**: Added parallel model training capabilities for improved performance
- **Model Validation Framework**: Comprehensive validation system with k-fold cross-validation, temporal validation, and liga-based testing
- **Explainable AI (XAI)**: Full integration of prediction explanations with natural language descriptions and SHAP value analysis
- **Performance Optimization Suite**: 
  - Redis-like in-memory caching system for predictions
  - Asynchronous data fetching with aiohttp for parallel API calls
  - Batch processing capabilities for multiple predictions
  - Query optimization for team data fetching
  - Response compression for improved API performance
- **Async Prediction Support**: New `get_async_predictions()` method for handling multiple match predictions simultaneously
- **Enhanced Caching**: Migrated from basic file cache to sophisticated two-tier caching (memory + file)
- **Prediction Explanation**: Each prediction now includes detailed explanations of key factors and confidence reasoning

### Recent Changes (July 14, 2025 - Earlier)
- **ML Model Integration from Backup**: Successfully integrated pre-trained ML models from backup system
- **CRF Model Implementation**: Added Conditional Random Fields model (crf_model.pkl) for enhanced 1X2 predictions
- **Self-Learning Model**: Integrated adaptive self-learning model with dynamic weight adjustment based on match context
- **Neural Network Implementation**: Added new Neural Network model (algorithms/neural_network.py) with TensorFlow support
- **Enhanced Ensemble System**: Updated ensemble algorithm to support 7 models (Poisson, Dixon-Coles, XGBoost, Monte Carlo, CRF, Neural Network + dynamic weighting)
- **XGBoost Training Fix**: Fixed class labeling issues in XGBoost training with proper 0,1,2 class distribution
- **Dynamic Weight System**: Self-learning model adjusts algorithm weights based on match characteristics (extreme/low-scoring/normal)
- **Model Files Organization**: Created dedicated models/ directory for storing ML model files (.pkl and .json)
- **Real Data Integration**: All ML models now train on real cached match data from predictions_cache.json
- **Continuous Learning Implementation**: CRF and Neural Network models now continuously learn from new data instead of using static pre-trained models
- **Dynamic Model Training**: 
  - XGBoost: Retrains when 10+ matches available in cache
  - CRF: Retrains when 15+ matches available, uses RandomForest for flexibility
  - Neural Network: Retrains when 20+ matches available, uses TensorFlow Sequential model
- **Algorithm Weight Distribution**: 
  - Normal matches: Poisson 25%, CRF 15%, Dixon-Coles 18%, XGBoost 12%, Monte Carlo 15%, Neural Network 15%
  - Extreme matches: Poisson 22%, Dixon-Coles 25%, Monte Carlo 22%, XGBoost 10%, CRF 10%, Neural Network 11%
  - Low-scoring matches: Dixon-Coles 30%, Poisson 22%, CRF 13%, Monte Carlo 13%, XGBoost 10%, Neural Network 12%

### Previous Changes (July 13, 2025)
- **Extreme Match Detection System**: Added intelligent detection for high-scoring matches
- **Dynamic Lambda Limits**: Implemented adaptive goal expectation limits (normal: 4.0, extreme: 8.0)
- **Enhanced Poisson Model**: Extended matrix support to 15x15 for extreme matches
- **Algorithm Weight Adjustment**: Different ensemble weights for extreme vs normal matches
- **Betting Prediction Fix**: Fixed to always show higher probability option (e.g., NO 58% instead of YES 42%)
- **Validation Logic**: Added extreme match validation to prevent unrealistic low predictions
- **Mobile-Responsive Prediction Popup**: Team names now display vertically (A/VS/B format) for better mobile experience
- **Consolidated Cache System**: Migrated from individual JSON files per match to single predictions_cache.json file
- **File Cleanup**: Removed duplicate prediction popup file (prediction-popup-template.js)
- **Monte Carlo Consistency Fix**: Added seed-based random number generation for consistent predictions
- **XGBoost Model Enhancement**: Added missing over/under and BTTS predictions for proper ensemble calculations

#### Algorithm Improvements (Report Implementation - July 13, 2025)
- **120-Day Data Filtering**: Both xG calculator and Elo system now filter matches from last 120 days
- **Elo-Integrated xG Calculation**: New `calculate_xg_xga_with_elo` method that adjusts xG/xGA based on team strength
- **Favorite Team Correction**: Implemented threshold-based xG adjustment when favorite team has lower xG than opponent
- **xGA Correction for Favorites**: Favorite teams maintain lower xGA with threshold protection
- **Enhanced Cross-Lambda Calculation**: Lambda values now incorporate Elo-based corrections with 1.1x home advantage
- **Improved Elo Opponent Estimation**: Opponent strength estimated based on goal difference patterns

### Previous Changes (July 12, 2025)
- **API Key Management System**: Implemented centralized API key configuration system
- **Dynamic Module Reloading**: Added automatic module reload functionality when API keys are updated
- **Fixture Data Auto-refresh**: System now automatically refreshes fixture data after API key updates
- **Navbar API Settings**: Added API settings modal accessible from navbar
- **Turkish Interface**: Maintained Turkish language interface throughout the system

The system is designed to be robust, scalable, and continuously improving through self-learning capabilities and model validation feedback loops.