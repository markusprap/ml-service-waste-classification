# 🚨 EMERGENCY ROLLBACK GUIDE

## Quick Rollback Instructions (If Deployment Fails)

### Immediate Actions:
1. **Check Railway logs** first for specific error
2. **Revert to previous commit** if needed
3. **Use backup configurations** below

### Common Railway Deployment Issues & Fixes:

#### Issue 1: Memory Limit Exceeded
**Symptoms**: Container killed, OOM errors
**Fix**: 
```
# In Railway dashboard:
- Go to Variables
- Add: RAILWAY_MEMORY_LIMIT=1024
- Redeploy
```

#### Issue 2: Build Timeout
**Symptoms**: Build process stops, timeout errors
**Fix**:
```
# Reduce build complexity temporarily
# Remove from Dockerfile (line 12-15):
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
```

#### Issue 3: Port Binding Error
**Symptoms**: Health check fails, port errors
**Fix**:
```
# In railway.json, change:
"startCommand": "python app.py"
# Instead of gunicorn
```

#### Issue 4: Model Loading Error
**Symptoms**: TensorFlow errors, model not found
**Fix**: Check model file path in Railway logs

### Backup Configurations:

#### Minimal Dockerfile (Emergency):
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV PORT=8080
EXPOSE $PORT
CMD ["python", "app.py"]
```

#### Minimal railway.json (Emergency):
```json
{
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "startCommand": "python app.py"
  }
}
```

### Contact Information:
- **Technical Support**: Check Railway Discord/Documentation
- **Project Lead**: Pak Markus
- **ML Engineer**: Bu Ayu (GitHub Copilot)

### Recovery Time Estimate:
- **Minor config fix**: 5-10 minutes
- **Code rollback**: 10-15 minutes  
- **Full reconfiguration**: 30-45 minutes

## 📞 Emergency Contacts:
1. Railway Support: https://railway.app/help
2. TensorFlow Issues: https://github.com/tensorflow/tensorflow/issues
3. GitHub Repository: [Your repo URL]

**Remember: Don't panic! Most issues are configuration-related and can be fixed quickly.**
