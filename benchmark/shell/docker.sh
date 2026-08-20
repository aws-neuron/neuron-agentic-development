#!/usr/bin/env bash
# Docker utilities for the NAD skill benchmark.
# Builds the run image (cached by Dockerfile hash) and runs kiro-cli in a
# container. Called from run_skill_tests.py as:
#   ./docker.sh build <directory>
#   ./docker.sh run-kiro <directory> <prompt> [--model MODEL] [--timeout SECONDS]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Image prefix for all benchmark images
IMAGE_PREFIX="skillbench"

# Cross-platform timeout command (macOS uses gtimeout from coreutils)
TIMEOUT_CMD=""
if command -v gtimeout &> /dev/null; then
    TIMEOUT_CMD="gtimeout"
elif command -v timeout &> /dev/null; then
    TIMEOUT_CMD="timeout"
fi

# Env vars passed through to the container. kiro-cli authenticates with
# KIRO_API_KEY (drives both the agent and the judge).
ENV_KEYS=(
    KIRO_API_KEY
)

# =============================================================================
# DOCKER CHECKS
# =============================================================================

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "ERROR: Docker not found" >&2
        return 1
    fi
    if ! docker info &> /dev/null 2>&1; then
        echo "ERROR: Docker daemon not running" >&2
        return 1
    fi
    return 0
}

# =============================================================================
# HASH-BASED IMAGE CACHING
# =============================================================================

# Get hash of build inputs for cache key (the Dockerfile). Only the Dockerfile
# affects the image; skills + prompt are added into the workspace at runtime,
# so hashing the whole dir would change the key every run and defeat the cache.
get_dockerfile_hash() {
    local dir="$1"
    local dockerfile="$dir/Dockerfile"

    if [[ ! -f "$dockerfile" ]]; then
        echo ""
        return 1
    fi

    if command -v md5 &> /dev/null; then
        cat "$dockerfile" | md5 -q | cut -c1-8
    else
        cat "$dockerfile" | md5sum | cut -c1-8
    fi
}

# Get image name for a directory (based on Dockerfile hash)
get_image_name() {
    local dir="$1"
    local hash
    hash=$(get_dockerfile_hash "$dir") || return 1
    echo "${IMAGE_PREFIX}:${hash}"
}

# Check if image exists
image_exists() {
    local image_name="$1"
    docker images -q "$image_name" 2>/dev/null | grep -q .
}

# =============================================================================
# DOCKER BUILD
# =============================================================================

# Build Docker image with caching
# Usage: docker_build <directory> [--force]
# Output: image name on stdout
docker_build() {
    local dir="$1"
    local force="${2:-}"

    local dockerfile="$dir/Dockerfile"
    if [[ ! -f "$dockerfile" ]]; then
        echo "ERROR: No Dockerfile in $dir" >&2
        return 1
    fi

    local image_name
    image_name=$(get_image_name "$dir") || return 1

    # Check cache unless forced
    if [[ "$force" != "--force" ]] && image_exists "$image_name"; then
        echo "$image_name"
        return 0
    fi

    if docker build -t "$image_name" -f "$dockerfile" "$dir" >&2; then
        echo "$image_name"
        return 0
    else
        echo "ERROR: Build failed" >&2
        return 1
    fi
}

# =============================================================================
# DOCKER RUN
# =============================================================================

# Build env var arguments for docker run (populates ENV_ARGS array)
build_env_args() {
    ENV_ARGS=()
    for key in "${ENV_KEYS[@]}"; do
        if [[ -n "${!key:-}" ]]; then
            ENV_ARGS+=("-e" "$key=${!key}")
        fi
    done
}

# Run Kiro CLI in Docker
# Usage: docker_run_kiro <directory> <prompt> [--model MODEL] [--timeout SECONDS]
docker_run_kiro() {
    local dir="$1"
    local prompt="$2"
    shift 2

    local model=""
    local timeout="300"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)
                model="$2"
                shift 2
                ;;
            --timeout)
                timeout="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done

    local image_name
    image_name=$(docker_build "$dir") || return 1

    build_env_args

    local cmd=(
        kiro-cli chat "$prompt"
        --no-interactive
        -a
    )

    if [[ -n "$model" ]]; then
        cmd+=(--model "$model")
    fi

    # Auth: KIRO_API_KEY env var (CI), or fall back to the host kiro-cli SSO
    # cache for local dev when the key isn't set.
    local auth_args=()
    if [[ -z "${KIRO_API_KEY:-}" ]]; then
        local auth_db="$HOME/.local/share/kiro-cli/data.sqlite3"
        if [[ -f "$auth_db" ]]; then
            auth_args+=("-v" "$auth_db:/root/.local/share/kiro-cli/data.sqlite3")
        fi
    fi

    if [[ -n "$TIMEOUT_CMD" ]]; then
        $TIMEOUT_CMD "$timeout" docker run --rm \
            -v "$dir:/workspace" \
            -w /workspace \
            ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} \
            ${auth_args[@]+"${auth_args[@]}"} \
            "$image_name" \
            "${cmd[@]}"
    else
        docker run --rm \
            -v "$dir:/workspace" \
            -w /workspace \
            ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} \
            ${auth_args[@]+"${auth_args[@]}"} \
            "$image_name" \
            "${cmd[@]}"
    fi
}

# =============================================================================
# CLI MODE
# =============================================================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    cmd="${1:-help}"
    shift || true

    case "$cmd" in
    check)
        if check_docker; then
            echo "OK"
        else
            exit 1
        fi
        ;;
    build)
        dir="${1:-}"
        force="${2:-}"
        if [[ -z "$dir" ]]; then
            die "Usage: $0 build <directory> [--force]"
        fi
        docker_build "$(realpath "$dir")" "$force"
        ;;
    run-kiro)
        dir="${1:-}"
        prompt="${2:-}"
        if [[ -z "$dir" || -z "$prompt" ]]; then
            die "Usage: $0 run-kiro <directory> <prompt> [--model MODEL] [--timeout SECONDS]"
        fi
        shift 2
        docker_run_kiro "$(realpath "$dir")" "$prompt" "$@"
        ;;
    help|*)
        cat <<EOF
Docker utilities for the NAD skill benchmark

Usage: $0 <command> [args...]

Commands:
  check                              Check if Docker is available
  build <dir> [--force]              Build image (cached by Dockerfile hash)
  run-kiro <dir> <prompt> [opts]     Run kiro-cli in the container

Options for run-kiro:
  --model MODEL      Model to use
  --timeout SECONDS  Timeout (default: 300)

Auth:
  KIRO_API_KEY is passed into the container (agent + judge). If unset, the host
  kiro-cli SSO cache (~/.local/share/kiro-cli/data.sqlite3) is mounted for local dev.
EOF
        ;;
    esac
fi
