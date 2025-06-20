# ML Service - Waste Classification

Machine Learning microservice untuk klasifikasi sampah menggunakan TensorFlow/Keras.

## Features
- REST API untuk klasifikasi gambar sampah
- Support untuk berbagai format gambar (JPEG, PNG, WebP)
- Health check endpoint
- CORS enabled untuk integrasi frontend

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy environment file:
```bash
cp .env.example .env
```

3. Run the application:
```bash
python app.py
```

API akan tersedia di `http://localhost:5000`

## Deployment

### Railway Deployment

1. Connect repository ke Railway
2. Set environment variables:
   - `CORS_ORIGINS`: Domain frontend Anda (contoh: https://your-app.vercel.app)
   - `LOG_LEVEL`: INFO (untuk production)
   - Variabel lain sesuai kebutuhan

3. Railway akan otomatis mendeteksi Procfile dan melakukan deployment

### Docker Deployment

```bash
docker build -t ml-service .
docker run -p 5000:5000 ml-service
```

## API Endpoints

### Health Check
- **GET** `/health`
- Response: Status service dan versi

### Classification
- **POST** `/classify`
- Body: multipart/form-data dengan field `image`
- Response: Hasil klasifikasi dengan confidence score

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| ML_SERVICE_HOST | Host binding | 0.0.0.0 |
| ML_SERVICE_PORT | Port number | 5000 |
| CORS_ORIGINS | Allowed origins | localhost:3000,localhost:3001 |
| MAX_CONTENT_LENGTH | Max file size (bytes) | 16777216 (16MB) |
| TARGET_IMAGE_SIZE | Image resize target | 224 |
| LOG_LEVEL | Logging level | INFO |

## Model Information

Model: TensorFlow/Keras H5 format
- Input: 224x224 RGB images
- Output: Multi-class classification untuk jenis sampah
- Classes: Sesuai dengan file `class_names.json`
