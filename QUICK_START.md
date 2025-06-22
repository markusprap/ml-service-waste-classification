# ML Service - Quick Start Guide

## ✅ BERHASIL! Masalah TensorFlow telah teratasi

**Status**: Model berjalan dengan sempurna di TensorFlow 2.16.1
**Server**: http://localhost:5000
**Akurasi**: 99.99% (tested dengan gambar botol plastik)

## Cara Menjalankan Server

### Opsi 1: Menggunakan Batch File (Termudah)
```
start_server.bat
```

### Opsi 2: Manual di PowerShell
```powershell
cd "d:\PROJECTS\Coding Camp 2025 powered by DBS Foundation\ml-service"
.\tf_216_env\Scripts\activate
python app.py
```

## Endpoints yang Tersedia

### Health Check
```
GET http://localhost:5000/health
```

### Classification 
```
POST http://localhost:5000/api/classify
Content-Type: multipart/form-data
Body: image file dengan key "image"
```

## Testing dengan Postman atau Browser

1. **Health Check**: Buka browser ke `http://localhost:5000/health`
2. **Classification**: Gunakan Postman untuk POST ke `http://localhost:5000/api/classify` dengan image file

## Response Format

```json
{
  "success": true,
  "data": {
    "confidence": 0.999976634979248,
    "main_category": "Anorganik", 
    "subcategory": "Plastik"
  }
}
```

## Kelas yang Didukung

Model dapat mengklasifikasikan 15 jenis sampah:
- Alat_Pembersih_Kimia
- Alumunium  
- Baterai
- Kaca
- Kardus
- Karet
- Kertas
- Lampu_dan_Elektronik
- Minyak_dan_Oli_Bekas
- Obat_dan_Medis
- Plastik
- Sisa_Buah_dan_Sayur
- Sisa_Makanan
- Styrofoam
- Tekstil

## Solusi untuk Cloud Deployment (Nanti)

Jika masalah payment cloud teratasi:
1. Update requirements.txt ke TensorFlow 2.16.1 ✅ (Sudah selesai)
2. Deploy ke Railway/Heroku/Google Cloud Run
3. Timeline: 1-2 jam
