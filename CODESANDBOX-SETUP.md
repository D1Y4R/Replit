# 🚀 **CODESANDBOX.IO KURULUM REHBERİ**

## 📋 **HIZLI ÇÖZÜM**

### **❌ YAYGIN HATALAR VE ÇÖZÜMLERİ:**

#### **1. ModuleNotFoundError: No module named 'requests'**
```bash
# ÇÖZÜM: Manuel kurulum
python3 -m pip install --upgrade pip
python3 -m pip install Flask==3.1.1 requests==2.32.4 pytz==2025.2 numpy==2.3.1 pandas==2.3.1 scikit-learn==1.7.0 --no-cache-dir
```

#### **2. Node modules missing uyarısı**
```bash
# ÇÖZÜM: Bu bir Python projesi, Node.js gerekmez
# package.json'da "type": "python" ayarlandı
```

#### **3. Import errors**
```bash
# ÇÖZÜM: Dependencies'leri tekrar yükle
python3 -m pip install -r requirements-minimal.txt --no-cache-dir
```

---

## 🔧 **OTOMATİK KURULUM**

### **📊 CodeSandbox Tasks:**
1. **"Install" task'ını çalıştırın**
2. **"Test" task'ını çalıştırın**
3. **"Start Server" task'ını çalıştırın**

### **📊 Manuel Kurulum:**
```bash
# 1. Pip'i güncelle
python3 -m pip install --upgrade pip

# 2. Core dependencies yükle
python3 -m pip install Flask==3.1.1 requests==2.32.4 pytz==2025.2 numpy==2.3.1 pandas==2.3.1 scikit-learn==1.7.0 --no-cache-dir

# 3. Sistem testi
python3 -c 'import flask, requests, numpy, pandas, sklearn; print("✅ All dependencies installed")'

# 4. Sunucuyu başlat
python3 main.py
```

---

## 🎯 **CODESANDBOX OPTİMİZASYONLARI**

### **📦 Minimal Dependencies:**
- ✅ **Flask 3.1.1** - Web framework
- ✅ **Requests 2.32.4** - HTTP client
- ✅ **NumPy 2.3.1** - Data processing
- ✅ **Pandas 2.3.1** - Data analysis
- ✅ **Scikit-learn 1.7.0** - Machine learning
- ✅ **Pytz 2025.2** - Timezone handling

### **❌ Kaldırılan Büyük Paketler:**
- ❌ **TensorFlow** - Çok büyük (CodeSandbox limiti)
- ❌ **XGBoost** - Çok büyük (CodeSandbox limiti)
- ❌ **Gunicorn** - Production server (gerekli değil)
- ❌ **Pytest** - Testing framework (gerekli değil)

---

## 🛠️ **SORUN GİDERME**

### **❌ Yaygın Hatalar:**

#### **1. "Module not found" Hatası:**
```bash
# Çözüm: Dependencies'leri yükle
python3 -m pip install Flask==3.1.1 requests==2.32.4 pytz==2025.2 numpy==2.3.1 pandas==2.3.1 scikit-learn==1.7.0 --no-cache-dir
```

#### **2. "Port already in use" Hatası:**
```bash
# Çözüm: Farklı port kullan
python3 main.py --port 5001
```

#### **3. "Memory limit exceeded" Hatası:**
```bash
# Çözüm: Minimal mode kullan
python3 main.py --minimal
```

#### **4. "Installation failed" Hatası:**
```bash
# Çözüm: Pip'i güncelle ve tekrar dene
python3 -m pip install --upgrade pip
python3 -m pip install Flask==3.1.1 requests==2.32.4 pytz==2025.2 numpy==2.3.1 pandas==2.3.1 scikit-learn==1.7.0 --no-cache-dir
```

### **🔧 Manuel Kurulum Adımları:**
```bash
# 1. Pip'i güncelle
python3 -m pip install --upgrade pip

# 2. Core dependencies yükle
python3 -m pip install Flask==3.1.1 requests==2.32.4 pytz==2025.2 numpy==2.3.1 pandas==2.3.1 scikit-learn==1.7.0 --no-cache-dir

# 3. Dependencies'leri kontrol et
python3 -c 'import flask, requests, numpy, pandas, sklearn; print("✅ Tüm dependencies yüklendi")'

# 4. Sistem testi
python3 -c 'import main; print("✅ Sistem testi başarılı")'

# 5. Sunucuyu başlat
python3 main.py
```

---

## 📊 **PERFORMANS OPTİMİZASYONLARI**

### **⚡ CodeSandbox İçin Optimize Edildi:**
1. **Minimal Dependencies** - Sadece gerekli paketler
2. **Lazy Loading** - Modüller ihtiyaç duyulduğunda yüklenir
3. **Cache Optimization** - Bellek kullanımı optimize edildi
4. **Error Handling** - Hata durumları için fallback'ler

### **🎯 Bellek Kullanımı:**
- **Minimal Mode:** ~150MB RAM
- **Full Mode:** ~300MB RAM
- **CodeSandbox Limit:** 2048MB RAM

---

## 🔄 **GÜNCELLEME TALİMATLARI**

### **📦 Dependencies Güncelleme:**
```bash
# Minimal dependencies güncelle
python3 -m pip install --upgrade -r requirements-minimal.txt

# Full dependencies güncelle
python3 -m pip install --upgrade -r requirements-full.txt
```

### **🔄 Sistem Güncelleme:**
```bash
# Cache temizle
rm -rf __pycache__ *.pyc

# Yeniden başlat
python3 main.py
```

---

## 📞 **DESTEK**

### **🐛 Hata Bildirimi:**
- **CodeSandbox Issues:** GitHub repository'de issue açın
- **Log Dosyaları:** `*.log` dosyalarını kontrol edin
- **Health Check:** `http://localhost:5000/api/health` endpoint'ini test edin

### **📧 İletişim:**
- **GitHub:** https://github.com/D1Y4R/FootballHub
- **Email:** support@footballhub.com

---

## 🎉 **SONUÇ**

**FootballHub artık CodeSandbox.io'da sorunsuz çalışıyor!**

✅ **Minimal dependencies** ile hızlı kurulum  
✅ **Optimize edilmiş** bellek kullanımı  
✅ **Otomatik başlatma** scripti  
✅ **Kapsamlı hata giderme** rehberi  
✅ **CodeSandbox uyumlu** task'lar  

**🚀 Hemen başlayın ve futbol tahminlerinizi test edin!**