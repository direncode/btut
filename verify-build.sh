#!/bin/bash

# BTUT Build Verification Script
# Verifies that all components are properly built and ready for deployment

set -e  # Exit on error

echo "========================================"
echo "BTUT Build Verification"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
ERRORS=0
WARNINGS=0

# Function to check command exists
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 is installed"
        return 0
    else
        echo -e "${RED}✗${NC} $1 is NOT installed"
        ERRORS=$((ERRORS + 1))
        return 1
    fi
}

# Function to check file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1 exists"
        return 0
    else
        echo -e "${RED}✗${NC} $1 NOT found"
        ERRORS=$((ERRORS + 1))
        return 1
    fi
}

# Function to check directory exists
check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1 exists"
        return 0
    else
        echo -e "${RED}✗${NC} $1 NOT found"
        ERRORS=$((ERRORS + 1))
        return 1
    fi
}

echo "1. Checking Prerequisites..."
echo "----------------------------"
check_command node
check_command npm
check_command cargo
check_command python3
check_command docker

echo ""
echo "2. Checking Rust Engine..."
echo "----------------------------"
check_dir "rust-engine"
check_file "rust-engine/Cargo.toml"
check_file "rust-engine/src/lib.rs"
check_file "rust-engine/src/core.rs"
check_file "rust-engine/src/agents.rs"
check_file "rust-engine/src/convergence.rs"

# Try to build Rust (optional, can take time)
if [ "$1" == "--build-rust" ]; then
    echo "Building Rust engine..."
    cd rust-engine
    if cargo build --release; then
        echo -e "${GREEN}✓${NC} Rust build successful"
    else
        echo -e "${RED}✗${NC} Rust build failed"
        ERRORS=$((ERRORS + 1))
    fi
    cd ..
fi

echo ""
echo "3. Checking API Backend..."
echo "----------------------------"
check_dir "api"
check_file "api/main.py"
check_file "api/requirements.txt"
check_dir "lib/simulation"
check_file "lib/simulation/btut_engine.py"

echo ""
echo "4. Checking Frontend..."
echo "----------------------------"
check_file "package.json"
check_file "next.config.js"
check_file "tsconfig.json"
check_dir "app"
check_file "app/page.tsx"
check_file "app/layout.tsx"
check_dir "components"
check_dir "lib"
check_file "lib/wasm-loader.ts"
check_file "lib/api-client.ts"

echo ""
echo "5. Checking Python SDK..."
echo "----------------------------"
check_dir "python-sdk"
check_file "python-sdk/setup.py"
check_file "python-sdk/btut/__init__.py"

echo ""
echo "6. Checking Integrations..."
echo "----------------------------"
check_dir "integration/ros"
check_dir "integration/sumo"

echo ""
echo "7. Checking Docker Config..."
echo "----------------------------"
check_file "Dockerfile.backend"
check_file "Dockerfile.frontend"
check_file "docker-compose.yml"

echo ""
echo "8. Checking Deployment Config..."
echo "----------------------------"
check_file "vercel.json"
check_file "fly.toml"
check_file ".env.example"

echo ""
echo "9. Checking Documentation..."
echo "----------------------------"
check_file "README.md"
check_file "README_PRODUCTION.md"
check_file "DEPLOYMENT_COMPLETE.md"
check_file "FINAL_DEPLOYMENT_INSTRUCTIONS.md"

echo ""
echo "========================================"
echo "Verification Summary"
echo "========================================"

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "Your BTUT platform is ready for deployment!"
    echo ""
    echo "Next steps:"
    echo "1. Build WASM: cd rust-engine && wasm-pack build --target web --release"
    echo "2. Install deps: npm install && cd api && pip install -r requirements.txt"
    echo "3. Deploy: See FINAL_DEPLOYMENT_INSTRUCTIONS.md"
    exit 0
else
    echo -e "${RED}✗ Found $ERRORS error(s)${NC}"
    echo ""
    echo "Please fix the errors above before deploying."
    exit 1
fi
