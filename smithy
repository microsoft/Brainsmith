#!/bin/bash
# Brainsmith Container Management Script
# Provides utilities for managing persistent Brainsmith containers

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
DOCKER_INST_NAME="${DOCKER_INST_NAME,,}"

# Debug output for container name generation (only if BSMITH_DEBUG is set)
debug() {
    [ "${BSMITH_DEBUG:-0}" = "1" ] && echo "DEBUG: $1" >&2
}

# Set defaults (same as run-docker.sh)
: ${BSMITH_HW_COMPILER="finn"}

# Set Xilinx environment defaults with bashrc fallback
if [ -z "$BSMITH_XILINX_PATH" ] && [ -f "$HOME/.bashrc" ]; then
    BSMITH_XILINX_PATH=$(grep -E "^export BSMITH_XILINX_PATH=" "$HOME/.bashrc" 2>/dev/null | cut -d'=' -f2- | tr -d '"' | head -1)
fi
: ${BSMITH_XILINX_PATH="/opt/Xilinx"}

if [ -z "$BSMITH_XILINX_VERSION" ] && [ -f "$HOME/.bashrc" ]; then
    BSMITH_XILINX_VERSION=$(grep -E "^export BSMITH_XILINX_VERSION=" "$HOME/.bashrc" 2>/dev/null | cut -d'=' -f2- | tr -d '"' | head -1)
fi
: ${BSMITH_XILINX_VERSION="2024.2"}

if [ -z "$BSMITH_DOCKER_EXTRA" ] && [ -f "$HOME/.bashrc" ]; then
    BSMITH_DOCKER_EXTRA=$(grep -E "^export BSMITH_DOCKER_EXTRA=" "$HOME/.bashrc" 2>/dev/null | cut -d'=' -f2- | tr -d '"' | head -1)
fi
: ${BSMITH_DOCKER_EXTRA=""}

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
: ${LOCALHOST_URL="localhost"}
: ${NETRON_PORT=8080}
: ${NUM_DEFAULT_WORKERS=4}
: ${NVIDIA_VISIBLE_DEVICES=""}

: ${BSMITH_BUILD_DIR="/tmp/$DOCKER_INST_NAME"}
: ${BSMITH_SSH_KEY_DIR="$BSMITH_DIR/ssh_keys"}
: ${PLATFORM_REPO_PATHS="/opt/xilinx/platforms"}
: ${OHMYXILINX="${BSMITH_DIR}/deps/oh-my-xilinx"}
: ${VIVADO_HLS_LOCAL=$VIVADO_PATH}
: ${VIVADO_IP_CACHE=$BSMITH_BUILD_DIR/vivado_ip_cache}
: ${DOCKER_BUILDKIT="1"}
: ${BSMITH_DOCKER_PREBUILT="0"}
: ${BSMITH_DOCKER_NO_CACHE="0"}
: ${BSMITH_SKIP_DEP_REPOS="0"}
: ${BSMITH_DOCKER_RUN_AS_ROOT="0"}
: ${BSMITH_DOCKER_GPU="$(docker info | grep nvidia | wc -m)"}
: ${BSMITH_DOCKER_BUILD_FLAGS=""}
: ${BSMITH_DOCKER_FLAGS=""}

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
Brainsmith Container Management

Usage: $0 COMMAND [OPTIONS]

Commands:
    daemon         Start persistent container in background
    exec CMD       Execute command in running container
    shell          Interactive shell in running container
    build          Build Docker image
    start          Interactive shell (one-time container)
    stop           Stop container
    status         Show container status
    cleanup        Remove container
    logs           Show container logs

Examples:
    $0 daemon && $0 exec "python script.py"    # Typical workflow
    $0 shell                                         # Interactive development
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

# Monitor container startup via log streaming
monitor_container_startup() {
    local container_name="$1"
    local timeout="${2:-300}"  # Default 5 minutes
    local start_time=$(date +%s)
    
    gecho "Starting container and monitoring initialization..."
    
    # Create a temporary file to track completion
    local completion_file="/tmp/.monitor_${container_name}_$$"
    
    # Start log monitoring in background
    {
        docker logs -f "$container_name" 2>&1 | while IFS= read -r line; do
            # Check for status messages (with more flexible matching)
            if [[ "$line" =~ BRAINSMITH_STATUS:(.+)$ ]]; then
                local status="${BASH_REMATCH[1]}"
                local status_parts=(${status//:/ })
                local status_type="${status_parts[0]}"
                local status_detail="${status_parts[1]:-}"
                
                case "$status_type" in
                    "INITIALIZING")
                        gecho "→ Container initializing..."
                        ;;
                    "FETCHING_DEPENDENCIES")
                        gecho "→ Fetching dependency repositories..."
                        ;;
                    "INSTALLING_PACKAGES")
                        if [ -n "$status_detail" ]; then
                            if [ "$status_detail" = "starting" ]; then
                                gecho "→ Starting package installation..."
                            else
                                gecho "→ Installing package: $status_detail"
                            fi
                        else
                            gecho "→ Installing packages..."
                        fi
                        ;;
                    "BUILDING_FINNXSI")
                        gecho "→ Building finnxsi extension..."
                        ;;
                    "READY")
                        gecho "✓ Container is ready!"
                        echo "SUCCESS" > "$completion_file"
                        break
                        ;;
                    "ERROR")
                        recho "✗ Container initialization failed: $status_detail"
                        echo "ERROR:$status_detail" > "$completion_file"
                        break
                        ;;
                esac
            elif [[ "$line" =~ "All setup complete - container is now fully ready" ]]; then
                # Fallback detection for ready state
                gecho "✓ Container is ready!"
                echo "SUCCESS" > "$completion_file"
                break
            elif [ "${BSMITH_SHOW_INIT_LOGS:-false}" = "true" ]; then
                # Show all logs if requested
                echo "  $line"
            fi
        done
    } &
    
    local monitor_pid=$!
    
    # Wait for completion or timeout
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if [ -f "$completion_file" ]; then
            local result=$(cat "$completion_file")
            kill $monitor_pid 2>/dev/null || true
            wait $monitor_pid 2>/dev/null || true
            rm -f "$completion_file"
            
            if [ "$result" = "SUCCESS" ]; then
                return 0
            else
                recho "Container initialization failed: ${result#ERROR:}"
                return 1
            fi
        fi
        
        # Check if container is still running
        if ! docker inspect "$container_name" --format='{{.State.Running}}' 2>/dev/null | grep -q "true"; then
            kill $monitor_pid 2>/dev/null || true
            wait $monitor_pid 2>/dev/null || true
            rm -f "$completion_file"
            recho "Container stopped unexpectedly during initialization"
            return 1
        fi
        
        sleep 2
        elapsed=$((elapsed + 2))
    done
    
    # Timeout reached
    kill $monitor_pid 2>/dev/null || true
    wait $monitor_pid 2>/dev/null || true
    rm -f "$completion_file"
    recho "Container initialization timeout after ${timeout}s"
    return 1
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
    
    gecho "Building Docker image $BSMITH_DOCKER_TAG"
    
    OLD_PWD=$(pwd)
    cd $BSMITH_DIR
    
    [ "$BSMITH_DOCKER_NO_CACHE" = "1" ] && BSMITH_DOCKER_BUILD_FLAGS+="--no-cache "
    
    docker build -f docker/Dockerfile \
        --build-arg BACKEND=$BSMITH_HW_COMPILER \
        --build-arg ENTRYPOINT=docker/entrypoint.sh \
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
    
    # Create the container but don't start it yet
    create_container "$1"
    return $?
}

# Create container with the specified mode
create_container() {
    MODE="$1"
    
    # Validate Docker flags for security before proceeding
    validate_docker_flags
    
    # Check for container name conflicts
    check_container_conflicts
    
    gecho "Creating new container $DOCKER_INST_NAME"
    
    # Create necessary directories
    mkdir -p $BSMITH_BUILD_DIR
    mkdir -p $BSMITH_SSH_KEY_DIR
    
    # Build Docker command with all required options
    DOCKER_CMD="docker run"
    
    if [ "$MODE" = "daemon" ]; then
        DOCKER_CMD+=" -d -t"
    else
        DOCKER_CMD+=" -it --rm"
    fi
    
    DOCKER_CMD+=" --name $DOCKER_INST_NAME"
    DOCKER_CMD+=" --init --hostname $DOCKER_INST_NAME"
    DOCKER_CMD+=" -e SHELL=/bin/bash"
    DOCKER_CMD+=" -w $BSMITH_DIR"
    
    # Essential volume mounts
    DOCKER_CMD+=" -v $BSMITH_DIR:$BSMITH_DIR"
    DOCKER_CMD+=" -v $BSMITH_BUILD_DIR:$BSMITH_BUILD_DIR"
    
    # Essential environment variables
    DOCKER_CMD+=" -e BSMITH_BUILD_DIR=$BSMITH_BUILD_DIR"
    DOCKER_CMD+=" -e BSMITH_DIR=$BSMITH_DIR"
    DOCKER_CMD+=" -e BSMITH_SKIP_DEP_REPOS=$BSMITH_SKIP_DEP_REPOS"
    DOCKER_CMD+=" -e LOCALHOST_URL=$LOCALHOST_URL"
    DOCKER_CMD+=" -e NUM_DEFAULT_WORKERS=$NUM_DEFAULT_WORKERS"
    
    # User/permission setup (unless running as root)
    if [ "$BSMITH_DOCKER_RUN_AS_ROOT" = "0" ]; then
        # Only mount system files if they exist and are readable
        [ -r /etc/group ] && DOCKER_CMD+=" -v /etc/group:/etc/group:ro"
        [ -r /etc/passwd ] && DOCKER_CMD+=" -v /etc/passwd:/etc/passwd:ro"
        # REMOVED: [ -r /etc/shadow ] && DOCKER_CMD+=" -v /etc/shadow:/etc/shadow:ro"
        # Security: /etc/shadow contains password hashes and should not be mounted
        [ -d /etc/sudoers.d ] && DOCKER_CMD+=" -v /etc/sudoers.d:/etc/sudoers.d:ro"
        
        # SSH key directory mount for non-root user
        if [ -d "$BSMITH_SSH_KEY_DIR" ]; then
            DOCKER_CMD+=" -v $BSMITH_SSH_KEY_DIR:$HOME/.ssh"
        fi
        DOCKER_CMD+=" --user $(id -u):$(id -g)"
    else
        # SSH key directory mount for root user
        if [ -d "$BSMITH_SSH_KEY_DIR" ]; then
            DOCKER_CMD+=" -v $BSMITH_SSH_KEY_DIR:/root/.ssh"
        fi
    fi
    
    # Dependency and Xilinx setup (if not skipping deps)
    if [ "$BSMITH_SKIP_DEP_REPOS" = "0" ]; then
        DOCKER_CMD+=" -e VIVADO_IP_CACHE=$VIVADO_IP_CACHE"
        DOCKER_CMD+=" -e OHMYXILINX=$OHMYXILINX"
        
        # Xilinx workarounds
        DOCKER_CMD+=" -e LD_PRELOAD=/lib/x86_64-linux-gnu/libudev.so.1"
        DOCKER_CMD+=" -e XILINX_LOCAL_USER_DATA=no"
        
        # Xilinx tools (if available)
        if [ ! -z "$BSMITH_XILINX_PATH" ]; then
            VIVADO_PATH="$BSMITH_XILINX_PATH/Vivado/$BSMITH_XILINX_VERSION"
            VITIS_PATH="$BSMITH_XILINX_PATH/Vitis/$BSMITH_XILINX_VERSION"
            HLS_PATH="$BSMITH_XILINX_PATH/Vitis_HLS/$BSMITH_XILINX_VERSION"
            
            DOCKER_CMD+=" -v $BSMITH_XILINX_PATH:$BSMITH_XILINX_PATH"
            [ -d "$VIVADO_PATH" ] && DOCKER_CMD+=" -e XILINX_VIVADO=$VIVADO_PATH -e VIVADO_PATH=$VIVADO_PATH"
            [ -d "$HLS_PATH" ] && DOCKER_CMD+=" -e HLS_PATH=$HLS_PATH"
            [ -d "$VITIS_PATH" ] && DOCKER_CMD+=" -e VITIS_PATH=$VITIS_PATH"
            [ -d "$PLATFORM_REPO_PATHS" ] && DOCKER_CMD+=" -v $PLATFORM_REPO_PATHS:$PLATFORM_REPO_PATHS -e PLATFORM_REPO_PATHS=$PLATFORM_REPO_PATHS"
        fi
    fi
    
    # GPU support
    if [ "$BSMITH_DOCKER_GPU" != 0 ]; then
        gecho "nvidia-docker detected, enabling GPUs"
        if [ ! -z "$NVIDIA_VISIBLE_DEVICES" ]; then
            DOCKER_CMD+=" --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"
        else
            DOCKER_CMD+=" --gpus all"
        fi
    fi
    
    # Additional flags from BSMITH_DOCKER_EXTRA and other sources
    DOCKER_CMD+=" $BSMITH_DOCKER_EXTRA $BSMITH_DOCKER_FLAGS"
    
    # Image and command
    if [ "$MODE" = "daemon" ]; then
        # Use proper entrypoint with daemon mode - industry standard approach
        DOCKER_CMD+=" -e BSMITH_CONTAINER_MODE=daemon"
        DOCKER_CMD+=" $BSMITH_DOCKER_TAG"
        gecho "Starting daemon container..."
        # Execute with explicit empty command to trigger daemon mode
        RESULT=$($DOCKER_CMD "")
        DOCKER_EXIT_CODE=$?
        
        # Wait a moment and check if container actually started
        sleep 2
        FINAL_STATUS=$(get_container_status)
        
        if [ "$FINAL_STATUS" != "running" ]; then
            recho "Container failed to start properly. Status: $FINAL_STATUS"
            echo "=== Container logs ===" >&2
            docker logs "$DOCKER_INST_NAME" 2>&1 || echo "No logs available" >&2
            echo "=== Docker inspect ===" >&2
            docker inspect "$DOCKER_INST_NAME" --format='{{.State}}' 2>&1 || echo "Inspect failed" >&2
            return 1
        else
            # Use new log monitoring system instead of polling
            local init_timeout="${BSMITH_INIT_TIMEOUT:-300}"
            
            if monitor_container_startup "$DOCKER_INST_NAME" "$init_timeout"; then
                gecho "Container started successfully in daemon mode"
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
    else
        DOCKER_CMD+=" $BSMITH_DOCKER_TAG bash"
        gecho "Starting interactive container..."
        exec $DOCKER_CMD
    fi
}

# Build image if needed, create container if needed, open interactive shell
start_interactive() {
    # Build image if it doesn't exist or if not using prebuilt
    if [ "$BSMITH_DOCKER_PREBUILT" = "0" ]; then
        build_image
    fi
    
    # For interactive mode, always create new container with --rm
    create_container "interactive"
}

# Build image if needed, create container if needed, start daemon in background
start_daemon() {
    setup_container_if_needed "daemon"
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
        recho "Container $DOCKER_INST_NAME is not running. Start it first with: $0 daemon"
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
    
    # Use the fast exec entrypoint for optimized performance
    docker exec "$DOCKER_INST_NAME" /usr/local/bin/entrypoint_exec.sh bash -c "$CMD"
    return $?
}

open_shell() {
    if ! is_container_running; then
        recho "Container $DOCKER_INST_NAME is not running. Start it first with: $0 start daemon"
        return 1
    fi
    
    gecho "Opening shell in container $DOCKER_INST_NAME"
    docker exec -it "$DOCKER_INST_NAME" bash
}

show_logs() {
    docker logs "$DOCKER_INST_NAME" "$@"
}

cleanup_container() {
    STATUS=$(get_container_status)
    if [ "$STATUS" != "not_found" ]; then
        gecho "Removing container $DOCKER_INST_NAME"
        docker rm -f "$DOCKER_INST_NAME"
    else
        yecho "Container $DOCKER_INST_NAME does not exist"
    fi
}

# Main command handling
case "${1:-help}" in
    "build")
        build_image
        ;;
    "start")
        start_interactive
        ;;
    "daemon")
        start_daemon
        ;;
    "exec")
        shift
        exec_in_container "$@"
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
    "cleanup")
        cleanup_container
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        recho "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
