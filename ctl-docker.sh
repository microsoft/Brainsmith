#!/bin/bash
# Brainsmith Container Management Script
# Poetry-based orchestration for development environments
#
# Key commands:
#   start  - Start container and run complete setup automatically
#   shell  - Open interactive shell in running container  
#   clean  - Stop container and clean artifacts (use --deep for full reset)
#   [cmd]  - Execute any command in container (e.g., "smith setup all")
#
# Typical workflow:
#   poetry install                         # Set up Poetry environment (host)
#   ./fetch-repos.sh                       # Fetch Git repositories (host)
#   ./ctl-docker.sh start                  # Start container + automatic setup (all-in-one)
#   ./ctl-docker.sh shell                  # Interactive development
#   ./ctl-docker.sh "smith build model.py" # Run commands
#   ./ctl-docker.sh clean --deep           # Full cleanup when needed

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

gecho () { echo -e "${GREEN}$1${NC}"; }
recho () { echo -e "${RED}$1${NC}"; }
yecho () { echo -e "${YELLOW}$1${NC}"; }
becho () { echo -e "${BLUE}$1${NC}"; }

# Auto-detect brainsmith directory (where this script lives)
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
export BSMITH_DIR=$(readlink -f "$SCRIPT_DIR")

# Generate container name based on brainsmith directory (no timestamp for persistence)
BSMITH_DIR_HASH=$(echo "$BSMITH_DIR" | md5sum | cut -d' ' -f1 | head -c 8)
DOCKER_UNAME=$(id -un)
DOCKER_INST_NAME="brainsmith_dev_${DOCKER_UNAME}_${BSMITH_DIR_HASH}"
BSMITH_CONTAINER_DIR="/tmp/${DOCKER_INST_NAME}"
DOCKER_INST_NAME="${DOCKER_INST_NAME,,}"

# ===== ENVIRONMENT DEFAULTS (Developers: override these as needed) =====

# Essential paths - default to container-specific tmp directory for isolation
export BSMITH_BUILD_DIR="${BSMITH_BUILD_DIR:-$BSMITH_CONTAINER_DIR/build}"
export BSMITH_DEPS_DIR="${BSMITH_DEPS_DIR:-$BSMITH_DIR/deps}"

# Docker-specific control variables
export BSMITH_SKIP_DEP_REPOS="${BSMITH_SKIP_DEP_REPOS:-0}"
export BSMITH_DOCKER_RUN_AS_ROOT="${BSMITH_DOCKER_RUN_AS_ROOT:-0}"
export BSMITH_DOCKER_PREBUILT="${BSMITH_DOCKER_PREBUILT:-0}"
export BSMITH_DOCKER_NO_CACHE="${BSMITH_DOCKER_NO_CACHE:-0}"
export BSMITH_DOCKER_GPU="${BSMITH_DOCKER_GPU:-0}"

# Xilinx/FINN variables - Overrides values from brainsmith CLI/yaml settings
# These provide defaults that can be overridden by environment variables
export BSMITH_XILINX_PATH="${BSMITH_XILINX_PATH:-/tools/Xilinx}"
export BSMITH_XILINX_VERSION="${BSMITH_XILINX_VERSION:-2024.2}"
export VIVADO_IP_CACHE="${VIVADO_IP_CACHE:-}"
export PLATFORM_REPO_PATHS="${PLATFORM_REPO_PATHS:-}"

# Docker runtime detection
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

# Debug settings
export BSMITH_DEBUG="${BSMITH_DEBUG:-0}"

# ===== END OF ENVIRONMENT DEFAULTS =====

# Show configuration in debug mode
if [ "$BSMITH_DEBUG" = "1" ]; then
    echo "[DEBUG] Docker environment:"
    echo "[DEBUG]   DOCKER_INST_NAME=$DOCKER_INST_NAME"
    echo "[DEBUG]   BSMITH_DIR=$BSMITH_DIR"
    echo "[DEBUG]   BSMITH_BUILD_DIR=$BSMITH_BUILD_DIR"
    echo "[DEBUG]   BSMITH_DEPS_DIR=$BSMITH_DEPS_DIR"
    [ -n "$BSMITH_XILINX_PATH" ] && echo "[DEBUG]   BSMITH_XILINX_PATH=$BSMITH_XILINX_PATH"
fi

# Debug output for container name generation (only if BSMITH_DEBUG is set)
debug() {
    [ "${BSMITH_DEBUG:-0}" = "1" ] && echo "DEBUG: $1" >&2
}

# Docker-specific configuration (computed values, not from config file)
# Handle Docker tag with CI fallback
if [ -z "$BSMITH_DOCKER_TAG" ]; then
    # Try git describe first, fallback to commit hash for CI
    GIT_TAG=$(cd $BSMITH_DIR; git describe --always --tags --dirty 2>/dev/null)
    if [ -n "$GIT_TAG" ] && [ "$GIT_TAG" != "$(cd $BSMITH_DIR; git rev-parse --short HEAD 2>/dev/null)" ]; then
        BSMITH_DOCKER_TAG="microsoft/brainsmith:$GIT_TAG"
    else
        # Fallback for CI or when no tags
        COMMIT_HASH=$(cd $BSMITH_DIR; git rev-parse --short HEAD 2>/dev/null || echo "latest")
        BSMITH_DOCKER_TAG="microsoft/brainsmith:ci-$COMMIT_HASH"
    fi
fi

# GPU support is controlled by BSMITH_DOCKER_GPU environment variable

# Validate Docker flags for security
validate_docker_flags() {
    if [[ "$BSMITH_DOCKER_FLAGS" =~ "/var/run/docker.sock" ]] || \
       [[ "$BSMITH_DOCKER_EXTRA" =~ "/var/run/docker.sock" ]]; then
        recho "ERROR: Mounting Docker socket is not allowed for security reasons"
        recho "This would give the container full control over the host"
        exit 1
    fi
}

show_help() {
    cat << EOF
Brainsmith Container Management (Poetry-based)

Usage: ./ctl-docker.sh COMMAND [OPTIONS]

Container Commands:
    start          Start container and run complete setup automatically
    shell          Interactive shell in running container
    build          Build Docker image
    stop           Stop container
    restart        Stop and start container
    status         Show container status
    clean          Stop and clean container/artifacts (use --deep for full reset)
                   Options: --deep, --deep deps, --deep venv, --deep cache
    
Other Commands:
    logs           Show container logs
    
Note: Complete workflow is now streamlined:
    poetry install           # Install dependencies from pyproject.toml (host)
    ./fetch-repos.sh         # Fetch Git repositories (host)
    ./ctl-docker.sh start    # Start container + automatic setup (all-in-one)

Examples:
    ./ctl-docker.sh start                                   # Start and set up everything automatically
    ./ctl-docker.sh shell                                   # Interactive development
    ./ctl-docker.sh "smith build model.py"                  # Run smith commands
    ./ctl-docker.sh "smith setup cppsim"                    # Install additional components
    ./ctl-docker.sh clean                                   # Stop container and clean build files
    ./ctl-docker.sh clean --deep                            # Full reset (removes everything)
    ./ctl-docker.sh clean --deep venv                       # Clean only virtual environment
EOF
}

# Check available disk space before operations
check_disk_space() {
    local required_gb="${1:-10}"  # Default 10GB
    local available_kb=$(df "$BSMITH_DIR" | tail -1 | awk '{print $4}')
    local available_gb=$((available_kb / 1024 / 1024))

    if [ $available_gb -lt $required_gb ]; then
        recho "ERROR: Insufficient disk space"
        recho "Required: ${required_gb}GB, Available: ${available_gb}GB"
        recho "Location: $BSMITH_DIR"
        exit 1
    else
        gecho "Disk space check passed: ${available_gb}GB available"
    fi
}

# Monitor container setup process with live output
check_container_ready() {
    local container_name="$1"</    
    
    gecho "Container initializing..."
    
    # Give container a moment to start
    sleep 2
    
    # Simply follow the logs - Docker handles the streaming
    docker logs -f "$container_name" 2>&1 &
    
    local LOG_PID=$!
    
    # Monitor readiness in a loop
    while true; do
        # Check if container is still running
        if ! docker inspect "$container_name" --format='{{.State.Running}}' 2>/dev/null | grep -q "true"; then
            kill $LOG_PID 2>/dev/null || true
            recho "Container stopped unexpectedly"
            return 1
        fi
        
        # Check if Poetry dependencies are installed and smith is available
        if docker exec "$container_name" bash -c "cd $BSMITH_DIR && [ -f .venv/bin/smith ]" >/dev/null 2>&1; then
            kill $LOG_PID 2>/dev/null || true
            return 0
        fi
        
        sleep 2
    done
}

get_container_status() {
    docker inspect "$DOCKER_INST_NAME" >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        STATUS=$(docker inspect --format='{{.State.Status}}' "$DOCKER_INST_NAME")
        echo "$STATUS"
    else
        echo "not_found"
    fi
}

is_container_running() {
    STATUS=$(get_container_status)
    [ "$STATUS" = "running" ]
}

build_image() {
    # Check disk space before building (requires 15GB for builds)
    check_disk_space 15

    gecho "Building Docker image $BSMITH_DOCKER_TAG with Poetry support"

    OLD_PWD=$(pwd)
    cd $BSMITH_DIR

    [ "$BSMITH_DOCKER_NO_CACHE" = "1" ] && BSMITH_DOCKER_BUILD_FLAGS+="--no-cache "

    docker build -f docker/Dockerfile \
        --build-arg BACKEND=finn \
        --tag=$BSMITH_DOCKER_TAG \
        $BSMITH_DOCKER_BUILD_FLAGS .

    cd $OLD_PWD
}

# Check for existing containers with the same name
check_container_conflicts() {
    local status=$(get_container_status)
    if [ "$status" != "not_found" ]; then
        becho "Found existing container $DOCKER_INST_NAME (status: $status)"
        if [ "$status" = "exited" ]; then
            becho "Will reuse existing container"
        fi
    fi
}

# Common container setup logic
setup_container_if_needed() {
    STATUS=$(get_container_status)

    if [ "$STATUS" = "running" ]; then
        return 0
    elif [ "$STATUS" = "exited" ]; then
        gecho "Starting existing container $DOCKER_INST_NAME"
        docker start "$DOCKER_INST_NAME"
        return $?
    fi

    # Build image if it doesn't exist or if not using prebuilt
    if [ "$BSMITH_DOCKER_PREBUILT" = "0" ]; then
        build_image
    fi

    # Create the container
    create_container
    return $?
}

# Create background container
create_container() {
    # Validate Docker flags for security before proceeding
    validate_docker_flags

    # Check for container name conflicts
    check_container_conflicts

    gecho "Creating new container $DOCKER_INST_NAME"

    # Create necessary directories on host
    mkdir -p $BSMITH_CONTAINER_DIR
    mkdir -p $BSMITH_BUILD_DIR

    # Build Docker command with daemon options
    DOCKER_CMD="docker run"
    DOCKER_CMD+=" -d -t"

    DOCKER_CMD+=" --name $DOCKER_INST_NAME"
    DOCKER_CMD+=" --init --hostname $DOCKER_INST_NAME"
    DOCKER_CMD+=" -e SHELL=/bin/bash"
    DOCKER_CMD+=" -w $BSMITH_DIR"

    # Essential volume mounts
    DOCKER_CMD+=" -v $BSMITH_DIR:$BSMITH_DIR"
    DOCKER_CMD+=" -v $BSMITH_CONTAINER_DIR:$BSMITH_CONTAINER_DIR"
    DOCKER_CMD+=" -v $BSMITH_BUILD_DIR:$BSMITH_BUILD_DIR"
    
    # Mount project-local .venv if it exists
    if [ -d "$BSMITH_DIR/.venv" ]; then
        DOCKER_CMD+=" -v $BSMITH_DIR/.venv:$BSMITH_DIR/.venv"
    fi

    # Essential environment variables
    DOCKER_CMD+=" -e BSMITH_BUILD_DIR=$BSMITH_BUILD_DIR"
    DOCKER_CMD+=" -e BSMITH_DIR=$BSMITH_DIR"
    DOCKER_CMD+=" -e BSMITH_DEPS_DIR=$BSMITH_DEPS_DIR"
    DOCKER_CMD+=" -e BSMITH_CONTAINER_DIR=$BSMITH_CONTAINER_DIR"
    DOCKER_CMD+=" -e BSMITH_SKIP_DEP_REPOS=$BSMITH_SKIP_DEP_REPOS"
    DOCKER_CMD+=" -e PYTHONUNBUFFERED=1"
    DOCKER_CMD+=" -e BSMITH_PLUGINS_STRICT=${BSMITH_PLUGINS_STRICT:-true}"
    DOCKER_CMD+=" -e DOCKER_INST_NAME=$DOCKER_INST_NAME"
    
    # Set Poetry to use project-local virtual environments
    DOCKER_CMD+=" -e POETRY_VIRTUALENVS_IN_PROJECT=true"
    
    # FINN-specific environment variables
    DOCKER_CMD+=" -e FINN_BUILD_DIR=$BSMITH_BUILD_DIR"
    DOCKER_CMD+=" -e FINN_ROOT=$BSMITH_DIR"

    # User/permission setup (preserved from original)
    if [ "$BSMITH_DOCKER_RUN_AS_ROOT" = "0" ]; then
        # Only mount system files if they exist and are readable
        [ -r /etc/group ] && DOCKER_CMD+=" -v /etc/group:/etc/group:ro"
        [ -r /etc/passwd ] && DOCKER_CMD+=" -v /etc/passwd:/etc/passwd:ro"
        # Security: /etc/shadow contains password hashes and should not be mounted
        [ -d /etc/sudoers.d ] && DOCKER_CMD+=" -v /etc/sudoers.d:/etc/sudoers.d:ro"

        DOCKER_CMD+=" --user $(id -u):$(id -g)"
    fi

    # Xilinx setup - if Xilinx path is provided and exists
    if [ ! -z "$BSMITH_XILINX_PATH" ] && [ -d "$BSMITH_XILINX_PATH" ]; then
        # Mount Xilinx installation
        DOCKER_CMD+=" -v $BSMITH_XILINX_PATH:$BSMITH_XILINX_PATH"
        DOCKER_CMD+=" -e BSMITH_XILINX_PATH=$BSMITH_XILINX_PATH"
        
        # Pass through Xilinx-related variables if they are defined
        [ -n "$BSMITH_XILINX_VERSION" ] && DOCKER_CMD+=" -e BSMITH_XILINX_VERSION=$BSMITH_XILINX_VERSION"
        [ -n "$VIVADO_IP_CACHE" ] && DOCKER_CMD+=" -e VIVADO_IP_CACHE=$VIVADO_IP_CACHE"

        # Xilinx workarounds
        DOCKER_CMD+=" -e LD_PRELOAD=/lib/x86_64-linux-gnu/libudev.so.1"
        DOCKER_CMD+=" -e XILINX_LOCAL_USER_DATA=no"
        
        # Mount platform repo if it exists
        if [ ! -z "$PLATFORM_REPO_PATHS" ] && [ -d "$PLATFORM_REPO_PATHS" ]; then
            DOCKER_CMD+=" -v $PLATFORM_REPO_PATHS:$PLATFORM_REPO_PATHS"
            DOCKER_CMD+=" -e PLATFORM_REPO_PATHS=$PLATFORM_REPO_PATHS"
        fi
        
        # Handle Xilinx license file/server
        if [ ! -z "$XILINXD_LICENSE_FILE" ]; then
            # Check if it's a file path (not a server address like 2100@server)
            if [[ "$XILINXD_LICENSE_FILE" != *"@"* ]]; then
                # It might be a file path or multiple paths separated by colons
                IFS=':' read -ra LICENSE_PATHS <<< "$XILINXD_LICENSE_FILE"
                for LICENSE_PATH in "${LICENSE_PATHS[@]}"; do
                    if [ -f "$LICENSE_PATH" ]; then
                        # Mount the directory containing the license file
                        LICENSE_DIR=$(dirname "$LICENSE_PATH")
                        DOCKER_CMD+=" -v $LICENSE_DIR:$LICENSE_DIR:ro"
                    fi
                done
            fi
            # Always pass the environment variable
            DOCKER_CMD+=" -e XILINXD_LICENSE_FILE=$XILINXD_LICENSE_FILE"
        fi
    fi

    # GPU support (preserved from original)
    if [ "$BSMITH_DOCKER_GPU" = "1" ]; then
        gecho "GPU support enabled (BSMITH_DOCKER_GPU=1)"
        if [ ! -z "$NVIDIA_VISIBLE_DEVICES" ]; then
            DOCKER_CMD+=" --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"
        else
            DOCKER_CMD+=" --gpus all"
        fi
    fi

    # Additional flags from BSMITH_DOCKER_EXTRA and other sources
    DOCKER_CMD+=" $BSMITH_DOCKER_EXTRA $BSMITH_DOCKER_FLAGS"

    # Run container with the image tag
    DOCKER_CMD+=" $BSMITH_DOCKER_TAG"
    
    gecho "Starting container..."
    # Execute without command to trigger daemon mode
    RESULT=$($DOCKER_CMD)
    DOCKER_EXIT_CODE=$?

    # Simplified readiness check
    sleep 2
    FINAL_STATUS=$(get_container_status)

    if [ "$FINAL_STATUS" != "running" ]; then
        recho "Container failed to start properly. Status: $FINAL_STATUS"
        echo "=== Container logs ===" >&2
        docker logs "$DOCKER_INST_NAME" 2>&1 || echo "No logs available" >&2
        return 1
    else
        # Wait for full container setup
        if check_container_ready "$DOCKER_INST_NAME"; then
            return 0
        else
            recho "Container failed to initialize properly"
            echo "=== Container logs (last 50 lines) ===" >&2
            docker logs --tail 50 "$DOCKER_INST_NAME" 2>&1 || echo "No logs available" >&2
            echo "=== Stopping and removing failed container ===" >&2
            docker stop "$DOCKER_INST_NAME" 2>/dev/null || true
            docker rm "$DOCKER_INST_NAME" 2>/dev/null || true
            return 1
        fi
    fi
}

# Start container and run automatic setup
start_daemon() {
    setup_container_if_needed
    local result=$?
    
    # If container started successfully, run automatic setup
    if [ $result -eq 0 ]; then
        gecho ""
        gecho "✓ Container is ready and initialized!"
        gecho ""
        gecho "Running automatic setup..."
        
        # Run setup automatically
        exec_in_container "smith setup all"
        local setup_result=$?
        
        if [ $setup_result -eq 0 ]; then
            gecho ""
            gecho "✓ Complete setup finished successfully!"
            gecho ""
            gecho "Container is ready for development:"
            gecho "  ./ctl-docker.sh shell                # Open interactive shell"
            gecho "  ./ctl-docker.sh \"smith --help\"       # Explore smith CLI"
            gecho "  ./ctl-docker.sh \"smith setup check\"  # Check setup status"
        elif [ $setup_result -eq 130 ]; then
            yecho ""
            yecho "Setup interrupted by user"
            yecho "Container is running. You can:"
            yecho "  ./ctl-docker.sh \"smith setup all\"    # Resume setup"
            yecho "  ./ctl-docker.sh shell                # Use current environment"
        else
            yecho ""
            yecho "⚠️  Automatic setup encountered issues"
            yecho "Container is running but setup may be incomplete"
            yecho ""
            yecho "You can:"
            yecho "  ./ctl-docker.sh shell                # Use basic environment"
            yecho "  ./ctl-docker.sh \"smith setup all\"    # Retry setup manually"
            yecho "  ./ctl-docker.sh \"smith setup check\"  # Check what's missing"
        fi
    fi
    
    return $result
}

stop_container() {
    if is_container_running; then
        gecho "Stopping container $DOCKER_INST_NAME"
        docker stop "$DOCKER_INST_NAME"
    else
        yecho "Container $DOCKER_INST_NAME is not running"
    fi
}

restart_container() {
    stop_container
    sleep 2
    start_daemon
}

show_status() {
    STATUS=$(get_container_status)
    case "$STATUS" in
        "running")
            gecho "Container $DOCKER_INST_NAME is running"
            docker ps --filter "name=$DOCKER_INST_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
            ;;
        "exited")
            yecho "Container $DOCKER_INST_NAME exists but is stopped"
            ;;
        "not_found")
            recho "Container $DOCKER_INST_NAME does not exist"
            ;;
        *)
            yecho "Container $DOCKER_INST_NAME status: $STATUS"
            ;;
    esac
}

exec_in_container() {
    if ! is_container_running; then
        recho "Container $DOCKER_INST_NAME is not running. Start it first with: ./ctl-docker.sh start"
        return 1
    fi

    if [ $# -eq 0 ]; then
        recho "No command specified for exec"
        return 1
    fi

    # Build command with proper quoting
    CMD=""
    for arg in "$@"; do
        if [ -z "$CMD" ]; then
            CMD="$arg"
        else
            CMD="$CMD $arg"
        fi
    done

    # Use the environment setup script for proper Poetry activation
    # Debug: show the command being executed
    debug "Executing in container: source /usr/local/bin/setup-env-streamlined.sh && cd $BSMITH_DIR && $CMD"
    docker exec -e PYTHONUNBUFFERED=1 -e BSMITH_PLUGINS_STRICT=${BSMITH_PLUGINS_STRICT:-true} "$DOCKER_INST_NAME" /usr/local/bin/entrypoint-exec.sh bash -c "$CMD"
    return $?
}

open_shell() {
    if ! is_container_running; then
        recho "Container $DOCKER_INST_NAME is not running. Start it first with: ./ctl-docker.sh start"
        return 1
    fi

    gecho "Opening shell in container $DOCKER_INST_NAME"
    # Use exec entrypoint for interactive shell with welcome message
    docker exec -it -e BSMITH_PLUGINS_STRICT=${BSMITH_PLUGINS_STRICT:-true} "$DOCKER_INST_NAME" /usr/local/bin/entrypoint-exec.sh
}

show_logs() {
    docker logs "$DOCKER_INST_NAME" "$@"
}

# Clean build artifacts only
clean_build_artifacts() {
    gecho "Cleaning build artifacts..."
    
    # Clean build directory
    if [ -d "$BSMITH_BUILD_DIR" ]; then
        gecho "Removing build directory: $BSMITH_BUILD_DIR"
        rm -rf "$BSMITH_BUILD_DIR"
    fi
    
    # Clean Vivado IP cache
    if [ -d "$VIVADO_IP_CACHE" ]; then
        gecho "Removing Vivado IP cache: $VIVADO_IP_CACHE"
        rm -rf "$VIVADO_IP_CACHE"
    fi
    
    # Clean temporary markers
    local temp_files=(
        "/tmp/.brainsmith_packages_installed"
        "/tmp/.brainsmith_deps_ready"
        "/tmp/.monitor_${DOCKER_INST_NAME}_*"
    )
    
    for file in "${temp_files[@]}"; do
        if ls $file 2>/dev/null 1>&2; then
            gecho "Removing temporary files: $file"
            rm -f $file
        fi
    done
    
    gecho "Build artifacts cleaned"
}

# Unified clean command with optional deep clean targets
clean_all() {
    local deep_targets=()
    
    # Parse arguments for --deep and specific targets
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --deep|-d)
                shift
                if [[ $# -eq 0 ]] || [[ "$1" == -* ]]; then
                    # Just --deep with no target means all
                    deep_targets=("all")
                else
                    # Collect specific targets
                    while [[ $# -gt 0 ]] && [[ "$1" != -* ]]; do
                        deep_targets+=("$1")
                        shift
                    done
                fi
                ;;
            *)
                shift
                ;;
        esac
    done
    
    yecho "Starting clean..."
    
    # 1. Always stop container if running
    if is_container_running; then
        gecho "Stopping running container..."
        stop_container
    fi
    
    # 2. Always remove container
    STATUS=$(get_container_status)
    if [ "$STATUS" != "not_found" ]; then
        gecho "Removing container $DOCKER_INST_NAME"
        docker rm -f "$DOCKER_INST_NAME"
    fi
    
    # 3. Always clean build artifacts
    clean_build_artifacts
    
    # 4. Deep clean specific targets if requested
    if [ ${#deep_targets[@]} -gt 0 ]; then
        yecho "Performing deep clean of: ${deep_targets[*]}"
        
        for target in "${deep_targets[@]}"; do
            case "$target" in
                "all")
                    # Clean everything
                    deep_targets=("cache" "deps" "venv")
                    break
                    ;;
            esac
        done
        
        for target in "${deep_targets[@]}"; do
            case "$target" in
                "cache")
                    # Remove Docker image and cache
                    if docker images | grep -q "$BSMITH_DOCKER_TAG"; then
                        gecho "Removing Docker image: $BSMITH_DOCKER_TAG"
                        docker rmi -f "$BSMITH_DOCKER_TAG"
                    fi
                    
                    # Remove dangling images
                    local dangling_images=$(docker images -f "dangling=true" -q)
                    if [ -n "$dangling_images" ]; then
                        gecho "Removing dangling Docker images..."
                        docker rmi $dangling_images
                    fi
                    
                    # Prune Docker build cache
                    gecho "Pruning Docker build cache..."
                    docker builder prune -f
                    ;;
                    
                "deps")
                    # Clean dependency repos
                    if [ -d "$BSMITH_DEPS_DIR" ]; then
                        gecho "Removing dependency repositories..."
                        rm -rf "$BSMITH_DEPS_DIR"
                    fi
                    ;;
                    
                "venv")
                    # Clean project-local .venv
                    if [ -d "$BSMITH_DIR/.venv" ]; then
                        gecho "Removing project-local .venv..."
                        rm -rf "$BSMITH_DIR/.venv"
                    fi
                    ;;
                    
                *)
                    yecho "Unknown deep clean target: $target"
                    ;;
            esac
        done
    fi
    
    gecho "Clean complete!"
    
    # Show disk space recovered
    if command -v du >/dev/null 2>&1; then
        gecho "Disk space available: $(df -h "$BSMITH_DIR" | tail -1 | awk '{print $4}')"
    fi
}


# Main command handling
case "${1:-help}" in
    "build")
        build_image
        ;;
    "start")
        start_daemon
        ;;
    "shell")
        open_shell
        ;;
    "stop")
        stop_container
        ;;
    "restart")
        restart_container
        ;;
    "status")
        show_status
        ;;
    "logs")
        shift
        show_logs "$@"
        ;;
    "clean")
        shift
        clean_all "$@"
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        # Default to exec if no recognized command
        exec_in_container "$@"
        ;;
esac