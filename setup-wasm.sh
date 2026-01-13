#!/bin/bash
# BTUT Platform - WASM Setup Script

echo "🦀 Setting up Rust/WASM for BTUT Platform..."
echo ""

# Check for Rust
if ! command -v rustc &> /dev/null; then
    echo "❌ Rust not found. Installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
else
    echo "✅ Rust already installed: $(rustc --version)"
fi

# Check for wasm-pack
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ wasm-pack not found. Installing..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
else
    echo "✅ wasm-pack already installed: $(wasm-pack --version)"
fi

# Add wasm target
echo "🎯 Adding wasm32-unknown-unknown target..."
rustup target add wasm32-unknown-unknown

echo ""
echo "✨ Setup complete!"
echo ""
echo "To build WASM module:"
echo "  npm run build:wasm"
echo ""
echo "To develop without WASM:"
echo "  npm run dev"
echo "  (Will auto-fallback to TypeScript engine)"
echo ""
echo "To build for production:"
echo "  npm run build"
echo "  (Includes WASM compilation)"
