#!/bin/bash
# Developer setup script for Brainsmith with Poetry
# Sets up complete development environment

set -e  # Exit on error

# Parse command line arguments
FORCE_MODE=""
if [[ "$1" == "--force" ]] || [[ "$1" == "-f" ]]; then
    FORCE_MODE="--force"
fi

echo "🔧 Setting up Brainsmith developer environment..."

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    exit 1
fi

# Ensure python symlink exists (for compatibility)
if [ ! -e /usr/bin/python ] && [ -e /usr/bin/python3 ]; then
    if [ -w /usr/bin ]; then
        ln -sf /usr/bin/python3 /usr/bin/python
    else
        mkdir -p "$HOME/.local/bin"
        ln -sf /usr/bin/python3 "$HOME/.local/bin/python"
        export PATH="$HOME/.local/bin:$PATH"
    fi
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
if [ -n "$FORCE_MODE" ]; then
    echo "  (Force mode: re-downloading all dependencies)"
fi
./fetch-repos.sh $FORCE_MODE

# Step 2: Configure Poetry for project-local virtual environment
echo ""
echo "⚙️  Configuring Poetry for project-local installation..."

# Use project-local virtual environments for simplicity
poetry config virtualenvs.in-project true
poetry config virtualenvs.create true

# Prevent keyring-related hangs
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

# Only clean environments in force mode
if [ -n "$FORCE_MODE" ]; then
    # Clean existing .venv for fresh install
    if [ -d ".venv" ]; then
        echo "📋 Force mode: Removing existing .venv directory..."
        rm -rf .venv
    fi
    
    # Remove any Poetry environments in cache
    poetry env remove --all -n 2>/dev/null || true
fi

echo ""
echo "📦 Step 2: Installing Python dependencies..."
poetry install

# Simple verification - trust Poetry's exit code
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Installation failed"
    exit 1
fi

# Step 3: Setup all dependencies
echo ""
echo "🔧 Step 3: Setting up all dependencies..."

# # Force Poetry to use the .venv we just created
# export VIRTUAL_ENV="$PWD/.venv"
# export PATH="$VIRTUAL_ENV/bin:$PATH"

# Now run smith setup
poetry run smith setup all $FORCE_MODE

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Or prefix commands with poetry:"
echo "  poetry run smith --help"