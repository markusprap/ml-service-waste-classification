# Railway Deployment Guide - ML Service

## Repository GitHub
🔗 **Repository:** https://github.com/markusprap/ml-service-waste-classification.git

## Pre-Deployment Checklist ✅

### 1. Persiapan Repository
- [x] Model file `model-update.h5` sudah ada
- [x] Class names `class_names.json` sudah ada  
- [x] Dockerfile sudah dikonfigurasi
- [x] Procfile untuk Railway sudah ada
- [x] Requirements.txt sudah lengkap

### 2. Environment Variables untuk Railway
Setelah deploy ke Railway, set environment variables berikut:

```bash
# Required Environment Variables
CORS_ORIGINS=https://your-frontend-domain.vercel.app,http://localhost:3000
LOG_LEVEL=INFO
ML_SERVICE_DEBUG=false

# Optional (sudah ada default values)
MAX_CONTENT_LENGTH=16777216
TARGET_IMAGE_SIZE=224
```

### 3. Deployment Steps

#### Option 1: Deploy via Railway CLI
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize and deploy
railway init
railway up
```

#### Option 2: Deploy via GitHub Integration
1. Push code ke GitHub repository
2. Connect repository di Railway dashboard  
3. Set environment variables
4. Railway akan auto-deploy

### 4. Testing Endpoints

#### Health Check
```bash
curl https://your-railway-domain.railway.app/health
```

#### Classification Test
```bash
curl -X POST https://your-railway-domain.railway.app/classify \
  -F "image=@test_image.jpg"
```

### 5. Expected Response Format

#### Successful Classification
```json
{
  "success": true,
  "data": {
    "subcategory": "Plastik",
    "main_category": "Anorganik", 
    "confidence": 0.95
  }
}
```

#### Error Response
```json
{
  "success": false,
  "error": "Error message here"
}
```

### 6. Integration dengan Frontend

Update CORS_ORIGINS di Railway dengan domain frontend Vercel:
```
CORS_ORIGINS=https://your-frontend.vercel.app
```

Update frontend untuk call ML service:
```javascript
const response = await fetch('https://your-ml-service.railway.app/classify', {
  method: 'POST',
  body: formData
});
```

### 7. Monitoring & Logs

- Railway dashboard untuk monitoring
- Logs tersedia di Railway console
- Health check endpoint: `/health`

### 8. Troubleshooting

#### Common Issues:
1. **CORS Error**: Pastikan frontend domain ada di CORS_ORIGINS
2. **Model Loading Error**: Cek apakah model file ter-upload dengan benar
3. **Memory Limit**: Railway free tier memiliki limitasi memory
4. **Cold Start**: First request mungkin lambat karena loading model

#### Performance Optimization:
- Model loading di initialize sekali saja
- Gunicorn dengan 2 workers untuk handle concurrent requests
- Request timeout 120 detik untuk processing gambar besar
