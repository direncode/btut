# 🔧 Windows Quick Fix Guide

## Critical Issues Found

### 1. ✅ Fix Rust WASM Build Error

The WASM build failed because `getrandom` needs the "js" feature enabled.

**Fix Cargo.toml:**

```powershell
# Edit rust-engine/Cargo.toml and add this line to [dependencies]:
# getrandom = { version = "0.2", features = ["js"] }
```

### 2. ✅ Fix Vercel Config Error

Remove the `routes` field from `vercel.json`:

```powershell
# Edit vercel.json and remove the "routes" section entirely
```

### 3. ✅ Install Fly.io CLI (Windows)

```powershell
# Download Fly.io for Windows
iwr https://fly.io/install.ps1 -useb | iex

# Or use scoop
scoop install flyctl
```

##  Quick Commands (Copy These!)

### Fix Cargo.toml
```powershell
cd rust-engine
notepad Cargo.toml
# Add this line after line 16:
# getrandom = { version = "0.2", features = ["js"] }
```

### Build WASM
```powershell
cd rust-engine
wasm-pack build --target web --out-dir pkg --release
cd ..
```

### Fix vercel.json
```powershell
notepad vercel.json
# Delete lines 10-14 (the "routes" section)
```

### Test Backend Works
```powershell
cd api
python main.py
# Press Ctrl+C after you see "BTUT API Server Started"
```

### For now: Skip frontend build (has missing files)
The frontend needs additional simulation wrapper files that weren't created yet.

## What Works Now

✅ Backend API (FastAPI) - Fully functional
✅ Python SDK - Ready to use
✅ Rust core modules - Complete
⚠️ WASM build - Needs getrandom fix
⚠️ Frontend - Needs simulation wrapper files
⚠️ Fly.io CLI - Not installed yet

## Next Steps

1. Apply the Cargo.toml fix above
2. Rebuild WASM
3. I'll create the missing simulation wrapper files
4. Then you can deploy

**Want me to create a PowerShell script to auto-fix these issues?**
