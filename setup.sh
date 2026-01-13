#!/bin/bash
# BTUT Platform - Production Setup Script

set -e

echo ""
echo "╔════════════════════════════════════════╗"
echo "║   BTUT Platform - Production Setup    ║"
echo "║   DARPA Challenge 13 Solution         ║"
echo "╚════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js not found${NC}"
    echo "Install from: https://nodejs.org/"
    exit 1
fi
echo -e "${GREEN}✅ Node.js $(node --version)${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    echo "Install from: https://python.org/"
    exit 1
fi
echo -e "${GREEN}✅ Python $(python3 --version)${NC}"

if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git not found${NC}"
    echo "Install from: https://git-scm.com/"
    exit 1
fi
echo -e "${GREEN}✅ Git $(git --version | cut -d' ' -f3)${NC}"

# Install Web Platform
echo ""
echo "📦 Installing Web Platform..."
npm install --silent

# Install API Server
echo ""
echo "📦 Installing API Server..."
cd api
pip3 install -r requirements.txt --quiet
cd ..

# Install Python SDK
echo ""
echo "📦 Installing Python SDK..."
cd python-sdk
pip3 install -e . --quiet
cd ..

# Install Benchmarks
echo ""
echo "📦 Installing Benchmark Suite..."
cd benchmarks
pip3 install -r requirements.txt --quiet
cd ..

# Build web app
echo ""
echo "🏗️  Building Web Application..."
npm run build --silent

echo ""
echo "╔════════════════════════════════════════╗"
echo "║        ✅ Setup Complete!              ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo -e "${BLUE}🚀 Quick Start Commands:${NC}"
echo ""
echo "  Start Web App (http://localhost:3000):"
echo -e "    ${GREEN}npm run dev${NC}"
echo ""
echo "  Start API Server (http://localhost:8000):"
echo -e "    ${GREEN}cd api && python server.py${NC}"
echo ""
echo "  Run Benchmarks:"
echo -e "    ${GREEN}cd benchmarks && python benchmark_suite.py${NC}"
echo ""
echo "  Test Python SDK:"
echo -e "    ${GREEN}python -c 'from btut import Simulator; print(\"SDK Ready!\")'${NC}"
echo ""
echo -e "${BLUE}📚 Documentation:${NC}"
echo "  - README.md - Main documentation"
echo "  - docs/ONE_PAGE_PROOF.md - Research summary"
echo "  - DEPLOYMENT.md - Deployment guide"
echo ""
echo -e "${BLUE}🌐 Deploy to Production:${NC}"
echo ""
echo "  Vercel (Web App):"
echo -e "    ${GREEN}vercel${NC}"
echo ""
echo "  AWS Lambda (API):"
echo -e "    ${GREEN}cd cloud/lambda && ./deploy.sh${NC}"
echo ""
echo "  Docker:"
echo -e "    ${GREEN}docker-compose up${NC}"
echo ""
echo "🎉 You're ready to ship! Run 'npm run dev' to start."
echo ""
