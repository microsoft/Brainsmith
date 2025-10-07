#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# One-stop setup script for Brainsmith development
# Performs all setup actions: fetch repos, install poetry, install dependencies, run brainsmith setup
# Works for both host and container environments

set -e  # Exit on error

# Parse command line arguments
FORCE_MODE=""
SKIP_REPOS=""
SKIP_POETRY=""
SKIP_BOARDS=""
SKIP_SIM=""
DOCKER_MODE=""
QUIET_MODE=""

# Function to check if a component should be skipped
should_skip() {
    local component=$1
    for skip in "${SKIP_COMPONENTS[@]}"; do
        if [ "$skip" == "$component" ]; then
            return 0
        fi
    done
    return 1
}

# Array to store components to skip
SKIP_COMPONENTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force|-f)
            FORCE_MODE="--force"
            shift
            ;;
        --skip)
            shift
            if [[ $# -eq 0 ]] || [[ "$1" == --* ]]; then
                echo "Error: --skip requires at least one component"
                echo "Valid components: repos, poetry, boards, sim"
                exit 1
            fi
            # Collect all components to skip until next flag
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                case "$1" in
                    repos|poetry|boards|sim)
                        SKIP_COMPONENTS+=("$1")
                        ;;
                    *)
                        echo "Error: Unknown component '$1'"
                        echo "Valid components: repos, poetry, boards, sim"
                        exit 1
                        ;;
                esac
                shift
            done
            ;;
        --docker)
            DOCKER_MODE="1"
            shift
            ;;
        --quiet|-q)
            QUIET_MODE="1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--force|-f] [--skip <component>...] [--docker] [--quiet|-q]"
            echo "  Components: repos, poetry, boards, sim"
            echo "  Example: $0 --skip boards sim --docker --quiet"
            exit 1
            ;;
    esac
done

# Set legacy skip variables based on new array
should_skip "repos" && SKIP_REPOS="1"
should_skip "poetry" && SKIP_POETRY="1"
should_skip "boards" && SKIP_BOARDS="1"
should_skip "sim" && SKIP_SIM="1"

# Export quiet mode for child processes
if [ "$QUIET_MODE" == "1" ]; then
    export BSMITH_QUIET="1"
fi

echo "🔧 Setting up Brainsmith developer environment..."

# Docker-specific configuration
if [ "$DOCKER_MODE" == "1" ]; then
    if [ -n "$BSMITH_CONTAINER_DIR" ]; then
        echo "📦 Configuring Poetry for Docker container..."
        export POETRY_CONFIG_DIR="$BSMITH_CONTAINER_DIR/poetry-config"
        export POETRY_CACHE_DIR="$BSMITH_CONTAINER_DIR/poetry-cache"
        mkdir -p "$POETRY_CONFIG_DIR" "$POETRY_CACHE_DIR"
    else
        echo "⚠️  Warning: BSMITH_CONTAINER_DIR not set in Docker mode"
    fi
else
    # Run environment validation for local installations
    if [ -f "docker/validate-env.sh" ]; then
        echo "🔍 Validating system requirements..."
        if ! bash docker/validate-env.sh; then
            echo ""
            echo "⚠️  System validation failed. Some features may not work properly."
            echo ""
            read -p "Continue anyway? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
        echo ""
    fi
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    exit 1
fi

# Check for Git (needed for fetch-repos.sh)
if ! command -v git &> /dev/null; then
    echo "❌ Git is required but not found"
    exit 1
fi

echo "✅ Basic prerequisites satisfied"

# Step 1: Fetch Git repositories (unless skipped)
if [ "$SKIP_REPOS" != "1" ]; then
    echo ""
    echo "📥 Step 1: Fetching Git dependencies..."
    if [ -n "$FORCE_MODE" ]; then
        echo "  (Force mode: re-downloading all dependencies)"
    fi
    
    # Only fetch if needed or forced
    if [ -n "$FORCE_MODE" ] || [ ! -d "${BSMITH_DEPS_DIR:-deps}" ] || [ -z "$(ls -A ${BSMITH_DEPS_DIR:-deps} 2>/dev/null)" ]; then
        if [ -x "./docker/fetch-repos.sh" ]; then
            ./docker/fetch-repos.sh $FORCE_MODE
        else
            echo "⚠️  fetch-repos.sh not found or not executable"
        fi
    else
        echo "  Dependencies already fetched (use --force to re-fetch)"
    fi
else
    echo ""
    echo "⏩ Skipping repository fetch (--skip-repos)"
fi

# Step 2: Install Poetry if not available (unless skipped)
if ! command -v poetry &> /dev/null && [ "$SKIP_POETRY" != "1" ]; then
    echo ""
    echo "📦 Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    # Add Poetry to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
fi

# Verify Poetry is available
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is required but not found"
    echo "Install with: curl -sSL https://install.python-poetry.org | sh"
    exit 1
fi

# Step 3: Configure Poetry for project-local virtual environment
echo ""
echo "⚙️  Configuring Poetry..."
poetry config virtualenvs.in-project true
poetry config virtualenvs.create true

# Prevent keyring-related hangs
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

# Clean environments if forced
if [ -n "$FORCE_MODE" ] && [ -d ".venv" ]; then
    echo "📋 Force mode: Removing existing .venv directory..."
    if [ "$DOCKER_MODE" == "1" ]; then
        # In Docker, the .venv might be a mounted volume - just clean its contents
        echo "  Docker mode: Cleaning .venv contents instead of removing directory..."
        find .venv -mindepth 1 -delete 2>/dev/null || true
    else
        # In local mode, we can remove the entire directory
        rm -rf .venv
    fi
    poetry env remove --all -n 2>/dev/null || true
fi

# Step 4: Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
if [ "$QUIET_MODE" == "1" ]; then
    poetry install -q
else
    poetry install
fi

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Poetry install failed"
    exit 1
fi

# Step 5: Run brainsmith setup based on skip flags
echo ""
echo "🔧 Running brainsmith setup..."

# Determine which setup commands to run
if [ "$SKIP_BOARDS" == "1" ] && [ "$SKIP_SIM" == "1" ]; then
    # Skip all setup commands
    echo "  Skipping all brainsmith setup (boards and sim skipped)..."
elif [ "$SKIP_BOARDS" == "1" ]; then
    # Run only simulation setup (cppsim and xsim)
    echo "  Setting up simulation only (boards skipped)..."
    poetry run brainsmith setup cppsim $FORCE_MODE
    if [ $? -eq 0 ]; then
        poetry run brainsmith setup xsim $FORCE_MODE
    fi
elif [ "$SKIP_SIM" == "1" ]; then
    # Run only boards
    echo "  Setting up board files only (simulation skipped)..."
    poetry run brainsmith setup boards $FORCE_MODE
else
    # Run all
    echo "  Setting up all components..."
    poetry run brainsmith setup all $FORCE_MODE
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Development environment setup complete!"
    echo ""
    echo "To activate the virtual environment, run:"
    echo "  source .venv/bin/activate"
    echo ""
    echo "Or prefix commands with poetry:"
    echo "  poetry run smith --help"
else
    echo ""
    echo "⚠️  Setup completed with warnings. Check the output above."
fi