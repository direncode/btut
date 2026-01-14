# BTUT Complete Deployment Guide

**Production deployment instructions for all platforms**

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Building for Production](#building-for-production)
4. [Vercel Deployment (Frontend)](#vercel-deployment)
5. [Fly.io Deployment (Backend)](#flyio-deployment)
6. [Railway Deployment (Alternative)](#railway-deployment)
7. [Docker Deployment](#docker-deployment)
8. [AWS Lambda (Serverless)](#aws-lambda)
9. [Environment Configuration](#environment-configuration)
10. [Post-Deployment Testing](#post-deployment-testing)

---

## 🔧 Prerequisites

### Required Tools

```bash
# Node.js 20+
node --version  # Should be v20+

# Rust (latest stable)
rustc --version  # Should be 1.75+
cargo --version

# wasm-pack (for WASM builds)
cargo install wasm-pack

# Python 3.11+
python --version  # Should be 3.11+

# Docker (optional, recommended)
docker --version
docker-compose --version

# Vercel CLI
npm i -g vercel

# Fly.io CLI
curl -L https://fly.io/install.sh | sh

# Railway CLI (optional)
npm i -g @railway/cli
```

---

## 💻 Local Development

### 1. Clone and Install

```bash
# Clone repository
git clone https://github.com/yourusername/btut.git
cd btut

# Install Node.js dependencies
npm install

# Install Python dependencies
cd api
pip install -r requirements.txt
cd ..

# Build Rust engine
cd rust-engine
cargo build --release
cd ..
```

### 2. Build WASM Module

```bash
cd rust-engine

# Build for web target
wasm-pack build --target web --out-dir pkg --release

# Verify output
ls pkg/  # Should see btut_wasm.js, btut_wasm_bg.wasm, etc.

cd ..
```

### 3. Start Development Servers

**Terminal 1 - Frontend:**
```bash
npm run dev
# Open http://localhost:3000
```

**Terminal 2 - Backend:**
```bash
cd api
python main.py
# API running on http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 4. Test Locally

```bash
# Test frontend
open http://localhost:3000

# Test API
curl http://localhost:8000/health

# Run simulation
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"config": {"N": 10000, "gamma": 1.45, "tau": 0.30, "iterations": 20}}'
```

---

## 🏗️ Building for Production

### 1. Build Frontend

```bash
# Set production API URL
export NEXT_PUBLIC_API_URL=https://btut-api.fly.dev

# Build Next.js
npm run build

# Test production build locally
npm start
```

### 2. Build Rust (Native + WASM)

```bash
cd rust-engine

# Native release build
cargo build --release

# WASM release build
wasm-pack build --target web --release --out-dir pkg

cd ..
```

### 3. Prepare Backend

```bash
cd api

# Install production dependencies
pip install -r requirements.txt

# Test
python -m pytest tests/  # If you have tests

cd ..
```

---

## 🌐 Vercel Deployment (Frontend)

### Initial Setup

```bash
# Login to Vercel
vercel login

# Link project (first time)
vercel link

# Set environment variables
vercel env add NEXT_PUBLIC_API_URL production
# Enter: https://btut-api.fly.dev
```

### Deploy

```bash
# Deploy to production
vercel --prod

# Your site will be live at:
# https://btut-platform.vercel.app (or your custom domain)
```

### Custom Domain (Optional)

```bash
# Add domain via Vercel CLI
vercel domains add btut.ai

# Or via Vercel dashboard:
# 1. Go to project settings
# 2. Domains → Add
# 3. Follow DNS instructions
```

### Automatic Deployments

```bash
# Link GitHub repository
# 1. Go to Vercel dashboard
# 2. Import Git Repository
# 3. Select your BTUT repo
# 4. Configure build settings:
#    - Framework: Next.js
#    - Root Directory: ./
#    - Build Command: npm run build
#    - Output Directory: .next

# Every push to main will auto-deploy
```

---

## 🚀 Fly.io Deployment (Backend)

### Initial Setup

```bash
# Login
fly auth login

# Create app (first time)
fly launch

# Answer prompts:
# - App name: btut-api
# - Region: Choose closest to your users
# - PostgreSQL: No (we use in-memory for now)
# - Redis: No (optional, can add later)
```

### Deploy

```bash
# Deploy
fly deploy

# Your API will be live at:
# https://btut-api.fly.dev
```

### Set Environment Variables

```bash
# Set secrets
fly secrets set API_ENV=production
fly secrets set SECRET_KEY=your-secret-key-here

# List secrets
fly secrets list
```

### Scale Application

```bash
# Scale to 2 instances
fly scale count 2

# Increase memory
fly scale memory 2048

# Change VM size
fly scale vm shared-cpu-2x
```

### Monitor

```bash
# View logs
fly logs

# Check status
fly status

# Open dashboard
fly dashboard
```

### Custom Domain (Optional)

```bash
# Add certificate
fly certs add api.btut.ai

# Add CNAME record in your DNS:
# CNAME api btut-api.fly.dev

# Verify
fly certs show api.btut.ai
```

---

## 🚂 Railway Deployment (Alternative Backend)

### Initial Setup

```bash
# Login
railway login

# Initialize
railway init

# Select "Empty Project"
# Name: btut-backend
```

### Deploy

```bash
# Deploy from local directory
railway up

# Or link GitHub repo for auto-deploy
railway link
```

### Configure

```bash
# Set environment variables
railway variables set API_ENV=production
railway variables set PORT=8000

# Add Dockerfile
# Railway will auto-detect Dockerfile.backend
```

### Domain

```bash
# Railway provides automatic domain
railway domain

# For custom domain, add in dashboard:
# Settings → Domains → Add Domain
```

---

## 🐳 Docker Deployment

### Docker Compose (Recommended for Self-Hosting)

```bash
# Create .env file
cp .env.example .env
# Edit .env with your values

# Build and start all services
docker-compose up -d

# Services:
# - Frontend: http://localhost:3000
# - Backend: http://localhost:8000
# - Redis: localhost:6379

# Scale backend
docker-compose up -d --scale backend=4

# View logs
docker-compose logs -f

# Stop all
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Individual Docker Builds

**Backend:**
```bash
# Build
docker build -f Dockerfile.backend -t btut-backend .

# Run
docker run -d \
  -p 8000:8000 \
  -e API_ENV=production \
  --name btut-backend \
  btut-backend

# Check logs
docker logs btut-backend
```

**Frontend:**
```bash
# Build
docker build -f Dockerfile.frontend -t btut-frontend .

# Run
docker run -d \
  -p 3000:3000 \
  -e NEXT_PUBLIC_API_URL=http://localhost:8000 \
  --name btut-frontend \
  btut-frontend

# Check logs
docker logs btut-frontend
```

### Docker Hub (Share Images)

```bash
# Login
docker login

# Tag images
docker tag btut-backend yourusername/btut-backend:latest
docker tag btut-frontend yourusername/btut-frontend:latest

# Push
docker push yourusername/btut-backend:latest
docker push yourusername/btut-frontend:latest
```

---

## ☁️ AWS Lambda (Serverless Backend)

### Setup

```bash
# Install AWS CLI
pip install awscli

# Configure
aws configure

# Install SAM CLI
pip install aws-sam-cli
```

### Deploy Lambda Function

```bash
cd cloud/lambda

# Package dependencies
pip install -r requirements.txt -t package/

# Create deployment package
cd package
zip -r ../lambda_function.zip .
cd ..
zip -g lambda_function.zip lambda_function.py

# Create Lambda function
aws lambda create-function \
  --function-name btut-simulator \
  --runtime python3.11 \
  --role arn:aws:iam::YOUR_ACCOUNT:role/lambda-execution-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://lambda_function.zip \
  --timeout 300 \
  --memory-size 3008

# Create API Gateway (for HTTP access)
aws apigatewayv2 create-api \
  --name btut-api \
  --protocol-type HTTP \
  --target arn:aws:lambda:REGION:ACCOUNT:function:btut-simulator
```

### Update Function

```bash
# Update code
zip -g lambda_function.zip lambda_function.py

# Update Lambda
aws lambda update-function-code \
  --function-name btut-simulator \
  --zip-file fileb://lambda_function.zip
```

---

## ⚙️ Environment Configuration

### Production Environment Variables

**Frontend (.env.production):**
```bash
NEXT_PUBLIC_API_URL=https://btut-api.fly.dev
NEXT_PUBLIC_WS_URL=wss://btut-api.fly.dev
NEXT_TELEMETRY_DISABLED=1
```

**Backend (.env):**
```bash
API_ENV=production
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
CORS_ORIGINS=https://btut.ai,https://btut-platform.vercel.app
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
LOG_LEVEL=info
```

### Secrets Management

**Vercel:**
```bash
vercel env add SECRET_KEY production
vercel env add DATABASE_URL production
```

**Fly.io:**
```bash
fly secrets set SECRET_KEY=xxx
fly secrets set DATABASE_URL=xxx
```

**Railway:**
```bash
railway variables set SECRET_KEY=xxx
railway variables set DATABASE_URL=xxx
```

---

## 🧪 Post-Deployment Testing

### Health Checks

```bash
# Frontend
curl https://btut-platform.vercel.app/

# Backend
curl https://btut-api.fly.dev/health

# Expected response:
# {"status":"healthy","timestamp":"2025-01-13T...","active_simulations":0}
```

### API Tests

```bash
# Get presets
curl https://btut-api.fly.dev/api/presets

# Run simulation
curl -X POST https://btut-api.fly.dev/api/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "N": 10000,
      "gamma": 1.45,
      "tau": 0.30,
      "iterations": 20
    }
  }'

# Run benchmark
curl -X POST https://btut-api.fly.dev/api/benchmark
```

### Load Testing

```bash
# Install Apache Bench
apt-get install apache2-utils  # Ubuntu/Debian
brew install ab  # macOS

# Test API endpoint
ab -n 1000 -c 10 https://btut-api.fly.dev/health

# Test simulation endpoint
ab -n 100 -c 5 -p simulation_payload.json -T application/json \
  https://btut-api.fly.dev/api/simulate
```

### Monitoring

**Vercel:**
- Dashboard: https://vercel.com/dashboard
- Analytics: Automatic
- Error tracking: Integrated

**Fly.io:**
```bash
# Metrics
fly dashboard

# Logs
fly logs --app btut-api

# Scale based on load
fly autoscale set min=1 max=10 --app btut-api
```

---

## 🔐 Security Checklist

- [ ] HTTPS enabled (Vercel/Fly.io auto)
- [ ] API rate limiting configured
- [ ] CORS origins restricted
- [ ] Environment secrets not in code
- [ ] Input validation on all endpoints
- [ ] Health checks enabled
- [ ] Logging configured
- [ ] Error handling implemented
- [ ] No sensitive data in logs
- [ ] Regular dependency updates

---

## 📊 Performance Optimization

### Frontend

```bash
# Enable Next.js optimizations in next.config.js
module.exports = {
  compress: true,
  poweredByHeader: false,
  generateEtags: true,
  images: {
    formats: ['image/avif', 'image/webp'],
  },
}
```

### Backend

```bash
# Use gunicorn with uvicorn workers
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000

# Enable Redis caching (add to docker-compose.yml)
```

### Database (If Needed)

```bash
# Add PostgreSQL on Fly.io
fly postgres create --name btut-db

# Attach to app
fly postgres attach btut-db --app btut-api
```

---

## 🎉 Deployment Complete!

Your BTUT platform is now live:

- **Frontend**: https://btut-platform.vercel.app
- **Backend API**: https://btut-api.fly.dev
- **API Docs**: https://btut-api.fly.dev/docs
- **Health**: https://btut-api.fly.dev/health

### Next Steps

1. ✅ Set up custom domain (optional)
2. ✅ Configure monitoring/alerts
3. ✅ Set up CI/CD pipelines
4. ✅ Add SSL certificates (if custom domain)
5. ✅ Configure backups (if using database)
6. ✅ Share with community!

---

## 🆘 Troubleshooting

### Build Failures

**WASM build fails:**
```bash
# Ensure wasm-pack is installed
cargo install wasm-pack

# Clear cache
cargo clean
wasm-pack build --target web
```

**Next.js build fails:**
```bash
# Clear cache
rm -rf .next node_modules
npm install
npm run build
```

### Deployment Issues

**Vercel deployment fails:**
```bash
# Check logs
vercel logs

# Redeploy
vercel --prod --force
```

**Fly.io deployment fails:**
```bash
# Check logs
fly logs

# Restart app
fly apps restart btut-api

# Rebuild
fly deploy --force
```

### Runtime Errors

**API not responding:**
```bash
# Check Fly.io status
fly status

# Scale up
fly scale count 2

# Check logs
fly logs --app btut-api
```

**CORS errors:**
- Add frontend URL to `CORS_ORIGINS` in backend `.env`
- Restart backend

---

## 📞 Support

- **GitHub Issues**: https://github.com/yourusername/btut/issues
- **Email**: your.email@example.com
- **Documentation**: https://btut.ai/docs

---

**Deployment guide maintained by the BTUT team. Last updated: 2025-01-13**
