#!/bin/bash
# Developer setup script for Brainsmith with Poetry
# Sets up editable installs from deps/ directory

set -e  # Exit on error

echo "🔧 Setting up Brainsmith developer environment with Poetry..."

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    exit 1
fi

# Check for Poetry
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is required but not found"
    echo "Install with: curl -sSL https://install.python-poetry.org | sh"
    exit 1
fi

# Check for existing Poetry environment
if ! poetry env info --path &> /dev/null; then
    echo "❌ No Poetry environment found. Please run 'poetry install' first."
    exit 1
fi

echo "✅ Poetry environment found at: $(poetry env info --path)"

# Define our Git dependencies - URLs
declare -A GIT_URLS=(
    ["qonnx"]="https://github.com/fastmachinelearning/qonnx.git"
    ["finn"]="https://github.com/Xilinx/finn.git"
    ["finn-experimental"]="https://github.com/Xilinx/finn-experimental.git"
    ["brevitas"]="https://github.com/Xilinx/brevitas.git"
    ["dataset-loading"]="https://github.com/fbcotter/dataset_loading.git"
    ["onnxscript"]="https://github.com/jsmonson/onnxscript.git"
)

# Define our Git dependencies - Revisions (commit hash, tag, or branch)
declare -A GIT_REVS=(
    ["qonnx"]="9153395712b5617d38b058900c873c6fc522b343"
    ["finn"]="bd9baeb7ddad0f613689f3be81df28067f8c1d9b"
    ["finn-experimental"]="0724be21111a21f0d81a072fccc1c446e053f851"
    ["brevitas"]="95edaa0bdc8e639e39b1164466278c59df4877be"
    ["dataset-loading"]="0.0.4"
    ["onnxscript"]="62c7110aba46554432ce8e82ba2d8a086bd6227c"
)

# Clone or update Git repositories
echo "📥 Cloning/updating Git repositories to deps/..."
mkdir -p deps
cd deps

for pkg in "${!GIT_URLS[@]}"; do
    url="${GIT_URLS[$pkg]}"
    rev="${GIT_REVS[$pkg]}"
    
    if [ -d "$pkg" ]; then
        echo "  Updating $pkg..."
        cd "$pkg"
        git fetch --all
        git checkout "$rev"
        cd ..
    else
        echo "  Cloning $pkg..."
        git clone "$url" "$pkg"
        cd "$pkg"
        git checkout "$rev"
        cd ..
    fi
done

cd ..

# Now add the editable versions
echo ""
echo "📦 Installing Git packages as editable..."

for pkg in "${!GIT_URLS[@]}"; do
    echo "  Adding $pkg as editable..."
    poetry add --editable "./deps/$pkg" --no-interaction
done

# Install brainsmith itself as editable
echo ""
echo "📦 Installing brainsmith as editable..."
poetry add --editable . --no-interaction

# Optional: Setup simulation dependencies
echo ""
echo "🔧 Setting up simulation dependencies (optional)..."
if poetry run python -m brainsmith.core.plugins.simulation setup_cppsim 2>/dev/null; then
    echo "  ✅ C++ simulation dependencies installed"
else
    echo "  ⚠️  C++ simulation setup skipped (optional)"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "The following packages are installed as editable:"
echo "  - brainsmith -> . (main project)"
for pkg in "${!GIT_URLS[@]}"; do
    echo "  - $pkg -> ./deps/$pkg"
done