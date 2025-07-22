#!/bin/bash
# CodeSandbox Başlatma Scripti
# FootballHub için optimize edilmiş

echo "🚀 FootballHub CodeSandbox Başlatılıyor..."

# Python versiyonunu kontrol et
echo "📊 Python versiyonu kontrol ediliyor..."
python3 --version

# Pip'i güncelle
echo "📦 Pip güncelleniyor..."
python3 -m pip install --upgrade pip

# Core dependencies yükle (hızlı kurulum)
echo "📦 Core dependencies yükleniyor..."
python3 -m pip install Flask==3.1.1 requests==2.32.4 pytz==2025.2 numpy==2.3.1 pandas==2.3.1 scikit-learn==1.7.0 --no-cache-dir

# Flask kurulumunu kontrol et
echo "✅ Flask kurulumu kontrol ediliyor..."
python3 -c 'import flask; print("✅ Flask başarıyla yüklendi")'

# Requests kurulumunu kontrol et
echo "✅ Requests kurulumu kontrol ediliyor..."
python3 -c 'import requests; print("✅ Requests başarıyla yüklendi")'

# Data processing libraries kontrol et
echo "✅ Data processing libraries kontrol ediliyor..."
python3 -c 'import numpy, pandas; print("✅ NumPy ve Pandas başarıyla yüklendi")'

# Machine learning library kontrol et
echo "✅ Machine learning library kontrol ediliyor..."
python3 -c 'import sklearn; print("✅ Scikit-learn başarıyla yüklendi")'

# Sistem testi
echo "🧪 Sistem testi yapılıyor..."
python3 -c 'import main; print("✅ Sistem testi başarılı")'

# Sunucuyu başlat
echo "🌐 Sunucu başlatılıyor..."
echo "📱 CodeSandbox Preview: http://localhost:5000"
echo "🔗 API Endpoint: http://localhost:5000/api/health"
python3 main.py