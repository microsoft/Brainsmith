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
DOCS_MODE=""

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
        --docs)
            DOCS_MODE="1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--force|-f] [--skip <component>...] [--docker] [--quiet|-q] [--docs]"
            echo "  Components: repos, poetry, boards, sim"
            echo "  Example: $0 --skip boards sim --docker --quiet --docs"
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

echo -e "\033[36mSetting up Brainsmith developer environment...\033[0m"

# Docker-specific configuration
if [ "$DOCKER_MODE" == "1" ]; then
    if [ -n "$BSMITH_CONTAINER_DIR" ]; then
        # Remove stale completion marker from previous runs
        if [ -f "$BSMITH_CONTAINER_DIR/setup_complete" ]; then
            rm -f "$BSMITH_CONTAINER_DIR/setup_complete"
            echo -e "\033[33mRemoved stale setup completion marker\033[0m"
        fi

        echo -e "\033[36mConfiguring Poetry for Docker container...\033[0m"
        export POETRY_CONFIG_DIR="$BSMITH_CONTAINER_DIR/poetry-config"
        export POETRY_CACHE_DIR="$BSMITH_CONTAINER_DIR/poetry-cache"
        mkdir -p "$POETRY_CONFIG_DIR" "$POETRY_CACHE_DIR"
    else
        echo "⚠️ Warning: BSMITH_CONTAINER_DIR not set in Docker mode"
    fi
else
    # Run environment validation for local installations
    if [ -f "docker/validate-env.sh" ]; then
        echo -e "\033[36mValidating system requirements...\033[0m"
        if ! bash docker/validate-env.sh; then
            echo ""
            echo "⚠️ System validation failed. Some features may not work properly."
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
    echo "✗ Python 3 is required but not found"
    exit 1
fi

# Check for Git (needed for fetch-repos.sh)
if ! command -v git &> /dev/null; then
    echo "✗ Git is required but not found"
    exit 1
fi

echo -e "\033[32m✓\033[0m Basic prerequisites satisfied"

# Step 1: Fetch Git repositories (unless skipped)
if [ "$SKIP_REPOS" != "1" ]; then
    echo ""
    echo -e "\033[36mStep 1: Fetching Git dependencies...\033[0m"
    if [ -n "$FORCE_MODE" ]; then
        echo "  (Force mode: re-downloading all dependencies)"
    fi
    ./docker/fetch-repos.sh $FORCE_MODE
else
    echo ""
    echo -e "\033[33mSkipping repository fetch (--skip-repos)\033[0m"
fi

# Step 2: Install Poetry if not available (unless skipped)
if ! command -v poetry &> /dev/null && [ "$SKIP_POETRY" != "1" ]; then
    echo ""
    echo -e "\033[36mInstalling Poetry...\033[0m"
    curl -sSL https://install.python-poetry.org | python3 -
    # Add Poetry to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
fi

# Verify Poetry is available
if ! command -v poetry &> /dev/null; then
    echo "✗ Poetry is required but not found"
    echo "Install with: curl -sSL https://install.python-poetry.org | sh"
    exit 1
fi

# Step 3: Configure Poetry for project-local virtual environment
echo ""
echo -e "\033[36mConfiguring Poetry...\033[0m"
poetry config virtualenvs.in-project true
poetry config virtualenvs.create true

# Prevent keyring-related hangs
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

# Clean environments if forced
if [ -n "$FORCE_MODE" ] && [ -d ".venv" ]; then
    echo -e "\033[33mForce mode: Removing existing .venv directory...\033[0m"
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
echo -e "\033[36mInstalling Python dependencies...\033[0m"
if [ "$DOCS_MODE" == "1" ]; then
    echo "  Including documentation dependencies (mkdocs, mike, etc.)..."
fi
if [ "$QUIET_MODE" == "1" ]; then
    if [ "$DOCS_MODE" == "1" ]; then
        poetry install -q --with docs
    else
        poetry install -q
    fi
else
    if [ "$DOCS_MODE" == "1" ]; then
        poetry install --with docs
    else
        poetry install
    fi
fi

if [ $? -eq 0 ]; then
    echo -e "\033[32m✓\033[0m Dependencies installed successfully"
else
    echo "✗ Poetry install failed"
    exit 1
fi

# Step 4.5: Initialize default project config if needed
if [ ! -f "brainsmith.yaml" ]; then
    echo ""
    echo "No config found at brainsmith.yaml, generating default"
    poetry run brainsmith project init
    if [ $? -eq 0 ]; then
        echo -e "\033[32m✓\033[0m Default config created in .brainsmith/"
        echo "   Edit brainsmith.yaml to set your Xilinx paths"
    else
        echo "⚠️  Could not create default config (non-fatal)"
    fi
else
    echo ""
    echo -e "\033[32m✓\033[0m Using existing brainsmith.yaml"
fi

# Step 5.5: Activate environment and optionally configure direnv
echo ""
echo -e "\033[36mActivating brainsmith environment...\033[0m"

# Check if direnv is installed and allow .envrc if present
if command -v direnv &> /dev/null && [ -f ".envrc" ]; then
    echo "  Configuring direnv for automatic environment loading..."
    direnv allow
    echo -e "\033[32m✓\033[0m direnv configured (environment will auto-load on cd)"
fi

# Source environment for this script's execution
if [ -f ".brainsmith/env.sh" ]; then
    source .brainsmith/env.sh
    echo -e "\033[32m✓\033[0m Environment activated for setup"
else
    echo "✗ Environment file not found at .brainsmith/env.sh"
    echo "  This should have been created during config initialization"
    exit 1
fi

# Step 6: Run brainsmith setup based on skip flags
echo ""
echo -e "\033[36mRunning brainsmith setup...\033[0m"

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
    echo -e "\033[1;32mDevelopment environment setup complete!\033[0m"

    # Write completion marker for container readiness check (Docker mode only)
    if [ "$DOCKER_MODE" == "1" ] && [ -n "$BSMITH_CONTAINER_DIR" ]; then
        touch "$BSMITH_CONTAINER_DIR/setup_complete"
        echo -e "\033[32m✓\033[0m Setup completion marker written to $BSMITH_CONTAINER_DIR/setup_complete"
    fi

    echo ""
    echo -e "\033[1;32mSetup complete!\033[0m"
    echo ""
    if command -v direnv &> /dev/null; then
        echo "To activate the environment (direnv users):"
        echo "  cd .  # Reload directory to activate"
    else
        echo "To activate the environment:"
        echo "  source .venv/bin/activate && source .brainsmith/env.sh"
    fi
    echo ""
    echo "Then verify setup:"
    echo "  brainsmith project info"

else
    echo ""
    echo "⚠️ Setup completed with warnings. Check the output above."
fi
