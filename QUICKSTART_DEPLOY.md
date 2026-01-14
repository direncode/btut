# ⚡ BTUT Quick Start - Deploy in 10 Minutes

**Get your production BTUT platform live in 10 minutes**

---

## 🚀 Prerequisites (2 minutes)

```bash
# Check you have these installed
node --version    # Need 20+
cargo --version   # Need 1.75+
python --version  # Need 3.11+

# Install if missing:
# Node: https://nodejs.org/
# Rust: https://rustup.rs/
# Python: https://python.org/

# Install deployment CLIs
npm i -g vercel
curl -L https://fly.io/install.sh | sh
```

---

## 🏗️ Build Locally (3 minutes)

```bash
# 1. Navigate to project
cd /path/to/btut

# 2. Install dependencies
npm install
cd api && pip install -r requirements.txt && cd ..

# 3. Build WASM (most important!)
cd rust-engine
wasm-pack build --target web --release --out-dir pkg
cd ..

# 4. Test locally
npm run dev &    # Start frontend on :3000
cd api && python main.py &  # Start backend on :8000

# 5. Open browser
open http://localhost:3000

# If everything works, press Ctrl+C to stop servers
```

---

## 🌐 Deploy (5 minutes)

### Step 1: Deploy Backend to Fly.io (2 mins)

```bash
# Login
fly auth login

# Deploy (one command!)
fly launch --name btut-api --region sea --now

# Test it
curl https://btut-api.fly.dev/health
```

### Step 2: Deploy Frontend to Vercel (2 mins)

```bash
# Login
vercel login

# Set API URL
export NEXT_PUBLIC_API_URL=https://btut-api.fly.dev

# Deploy (one command!)
vercel --prod

# Done! Your site is live at https://btut-platform.vercel.app
```

### Step 3: Connect Them (1 min)

```bash
# Add env var to Vercel
vercel env add NEXT_PUBLIC_API_URL production
# Enter: https://btut-api.fly.dev

# Redeploy with new env
vercel --prod
```

---

## ✅ Verify Deployment (30 seconds)

```bash
# Test frontend (open in browser)
open https://btut-platform.vercel.app

# Test backend
curl https://btut-api.fly.dev/health

# Test simulation
curl -X POST https://btut-api.fly.dev/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"config":{"N":10000,"gamma":1.45,"tau":0.30,"iterations":20}}'

# Should return: {"simulation_id":"...","status":"completed",...}
```

---

## 🎉 Success!

Your BTUT platform is now live:

- **Frontend**: https://btut-platform.vercel.app
- **Backend**: https://btut-api.fly.dev
- **Docs**: https://btut-api.fly.dev/docs

---

## 🐛 Quick Troubleshooting

**WASM build fails:**
```bash
cargo install wasm-pack
cd rust-engine
wasm-pack build --target web --release
```

**Vercel deploy fails:**
```bash
vercel --prod --force
```

**Fly.io deploy fails:**
```bash
fly deploy --force
```

**Need help?** See `DEPLOYMENT_COMPLETE.md` for detailed troubleshooting.

---

## 📋 What to Do Next

1. ✅ Set up custom domain (optional)
2. ✅ Configure monitoring
3. ✅ Set up auto-deploy from GitHub
4. ✅ Share your platform!

See `FINAL_DEPLOYMENT_INSTRUCTIONS.md` for complete guide.

---

**Built in 10 minutes. Production-ready. Deploy it! 🚀**
