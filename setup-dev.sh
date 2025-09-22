#!/bin/bash
# Developer setup script for Brainsmith with Poetry
# Sets up complete development environment

set -e  # Exit on error

echo "🔧 Setting up Brainsmith developer environment..."

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

# Check for Git
if ! command -v git &> /dev/null; then
    echo "❌ Git is required but not found"
    exit 1
fi

echo "✅ Prerequisites satisfied"

# Step 1: Fetch Git dependencies
echo ""
echo "📥 Step 1: Fetching Git dependencies..."
./fetch-repos.sh

# Step 2: Install all dependencies via Poetry
echo ""
echo "📦 Step 2: Installing Python dependencies..."
poetry install

# Step 3: Optional simulation dependencies
echo ""
echo "🔧 Step 3: Setting up optional simulation dependencies..."
if poetry run python -m brainsmith.core.plugins.dependencies setup_cppsim 2>/dev/null; then
    echo "  ✅ C++ simulation dependencies installed"
else
    echo "  ⚠️  C++ simulation setup skipped (optional)"
fi

echo ""
echo "🎉 Development environment setup complete!"
echo ""

# Activate the virtual environment
VENV_PATH=$(poetry env info --path)
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

echo ""
echo "✅ Virtual environment activated!"
echo ""
echo "Brainsmith is ready for development. Available commands:"
echo "  • smith --help  # Direct access to smith CLI"
echo "  • deactivate    # Exit virtual environment"
echo ""
echo "To update dependencies in the future:"
echo "  • ./fetch-repos.sh  # Update Git repositories"
echo "  • poetry install   # Sync Python dependencies"