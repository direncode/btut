#!/bin/bash
set -e

echo "Building Rust WASM module..."

cd rust-engine

# Build with wasm-pack
wasm-pack build --target web --out-dir ../public/wasm --release

echo "WASM build complete!"
echo "Output: public/wasm/"

cd ..

# Copy types to lib
if [ -f "public/wasm/btut_wasm.d.ts" ]; then
    cp public/wasm/btut_wasm.d.ts lib/simulation/btut-wasm.d.ts
    echo "TypeScript definitions copied to lib/simulation/"
fi

echo "Done! WASM module ready for import."
