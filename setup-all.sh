#!/bin/bash
# BTUT Platform - Complete Setup Script
# Installs all dependencies and components

set -e

echo "========================================="
echo "BTUT Platform - Complete Setup"
echo "========================================="
echo ""

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Install from: https://nodejs.org/"
    exit 1
fi
echo "✅ Node.js $(node --version)"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install from: https://python.org/"
    exit 1
fi
echo "✅ Python $(python3 --version)"

# Install Web App
echo ""
echo "📦 Installing Web Application..."
npm install

# Install API Server
echo ""
echo "📦 Installing API Server..."
cd api
pip3 install -r requirements.txt
cd ..

# Install Python SDK
echo ""
echo "📦 Installing Python SDK..."
cd python-sdk
pip3 install -e .
cd ..

# Install Benchmarks
echo ""
echo "📦 Installing Benchmark Suite..."
cd benchmarks
pip3 install -r requirements.txt
cd ..

echo ""
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Start Web App:"
echo "   npm run dev"
echo ""
echo "2. Start API Server:"
echo "   cd api && python server.py"
echo ""
echo "3. Run Benchmarks:"
echo "   cd benchmarks && python benchmark_suite.py"
echo ""
echo "4. Use Python SDK:"
echo "   python -c 'from btut import Simulator; print(\"SDK Ready!\")'"
echo ""
echo "========================================="
