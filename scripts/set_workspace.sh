#!/bin/bash

get_project_top() {
    local SCRIPT_DIR
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    echo "$SCRIPT_DIR"
}

normalize_build_type()
{
    case "${1,,}" in
        debug)
            echo "Debug"
            ;;
        release)
            echo "Release"
            ;;
        relwithdebinfo)
            echo "RelWithDebInfo"
            ;;
        minsizerel)
            echo "MinSizeRel"
            ;;
        *)
            return 1
            ;;
    esac
}

normalize_cuda_profiling()
{
    case "${1,,}" in
        on)
            echo "ON"
            ;;
        off)
            echo "OFF"
            ;;
        *)
            return 1
            ;;
    esac
}

normalize_cuda_sanitizer()
{
    case "${1,,}" in
        on)
            echo "ON"
            ;;
        off)
            echo "OFF"
            ;;
        *)
            return 1
            ;;
    esac
}

normalize_taa()
{
    case "${1,,}" in
        on)
            echo "ON"
            ;;
        off)
            echo "OFF"
            ;;
        *)
            return 1
            ;;
    esac
}

normalize_force_affine_barycentrics()
{
    case "${1,,}" in
        on)
            echo "ON"
            ;;
        off)
            echo "OFF"
            ;;
        *)
            return 1
            ;;
    esac
}

function clear_vars()
{
    export PATH="$(getconf PATH):$HOME/bin"

    unset LD_LIBRARY_PATH
    unset CC
    unset CXX
    unset CMAKE

    unset SV_OUT
    unset SV_PROJECT_TOP
}

function set_vars()
{
    local T
    T="$(get_project_top)"
    if [ ! "$T" ]; then
        echo "Unable to find the top of the tree."
        return
    fi

    clear_vars

    export SV_PROJECT_TOP="$T"
    export SV_OUT="$T/out/"
    export PATH="$PATH:$SV_OUT/usr/bin"
    export LD_LIBRARY_PATH="$SV_OUT/usr/lib:$SV_OUT/usr/lib64"

    export CC="gcc"
    export CXX="g++"
    export CMAKE="cmake"
    export SV_BUILD_TYPE="Debug"
    export SV_CUDA_PROFILING="OFF"
    export SV_CUDA_SANITIZER="OFF"
    export SV_TAA="ON"
    export SV_FORCE_AFFINE_BARYCENTRICS="OFF"
}

function print_vars()
{
    printf "Current build configuration\n"
    printf "===========================\n"
    printf "Build type:\t\t$SV_BUILD_TYPE\n"
    printf "CUDA profiling:\t\t$SV_CUDA_PROFILING\n"
    printf "CUDA sanitizer:\t\t$SV_CUDA_SANITIZER\n"
    printf "TAA:\t\t\t$SV_TAA\n"
    printf "Force affine bary:\t$SV_FORCE_AFFINE_BARYCENTRICS\n"
    printf "Building at:\t\t$SV_PROJECT_TOP\n"
    printf "CC:\t\t\t$CC\n"
    printf "CXX:\t\t\t$CXX\n"
    printf "CMAKE:\t\t\t$CMAKE\n"
    printf "PATH:\t\t\t$PATH\n"
    printf "LD_LIBRARY_PATH:\t$LD_LIBRARY_PATH\n"
}

function print_workspace_options()
{
    printf "\nWorkspace toggle commands\n"
    printf "=========================\n"
    printf "set_build_type <Debug|Release|RelWithDebInfo|MinSizeRel> [current: %s]\n" "$SV_BUILD_TYPE"
    printf "set_cuda_profiling <on|off> [current: %s]\n" "${SV_CUDA_PROFILING,,}"
    printf "set_cuda_sanitizer <on|off> [current: %s]\n" "${SV_CUDA_SANITIZER,,}"
    printf "set_taa <on|off> [current: %s]\n" "${SV_TAA,,}"
    printf "set_force_affine_barycentrics <on|off> [current: %s]\n" "${SV_FORCE_AFFINE_BARYCENTRICS,,}"
}

clear_vars
set_vars
print_vars
print_workspace_options

function set_build_type()
{
    if [ $# -ne 1 ]; then
        echo "usage: set_build_type <Debug|Release|RelWithDebInfo|MinSizeRel>"
        return 1
    fi

    local normalized
    normalized="$(normalize_build_type "$1")" || {
        echo "Unsupported build type: $1"
        echo "Supported values: Debug, Release, RelWithDebInfo, MinSizeRel"
        return 1
    }

    export SV_BUILD_TYPE="$normalized"
    echo "Build type set to $SV_BUILD_TYPE"
}

function set_cuda_profiling()
{
    if [ $# -ne 1 ]; then
        echo "usage: set_cuda_profiling <on|off>"
        return 1
    fi

    local normalized
    normalized="$(normalize_cuda_profiling "$1")" || {
        echo "Unsupported CUDA profiling mode: $1"
        echo "Supported values: on, off"
        return 1
    }

    export SV_CUDA_PROFILING="$normalized"
    echo "CUDA profiling set to $SV_CUDA_PROFILING"
}

function set_cuda_sanitizer()
{
    if [ $# -ne 1 ]; then
        echo "usage: set_cuda_sanitizer <on|off>"
        return 1
    fi

    local normalized
    normalized="$(normalize_cuda_sanitizer "$1")" || {
        echo "Unsupported CUDA sanitizer mode: $1"
        echo "Supported values: on, off"
        return 1
    }

    export SV_CUDA_SANITIZER="$normalized"
    echo "CUDA sanitizer set to $SV_CUDA_SANITIZER"
}

function set_taa()
{
    if [ $# -ne 1 ]; then
        echo "usage: set_taa <on|off>"
        return 1
    fi

    local normalized
    normalized="$(normalize_taa "$1")" || {
        echo "Unsupported TAA mode: $1"
        echo "Supported values: on, off"
        return 1
    }

    export SV_TAA="$normalized"
    echo "TAA set to $SV_TAA"
}

function set_force_affine_barycentrics()
{
    if [ $# -ne 1 ]; then
        echo "usage: set_force_affine_barycentrics <on|off>"
        return 1
    fi

    local normalized
    normalized="$(normalize_force_affine_barycentrics "$1")" || {
        echo "Unsupported affine barycentrics mode: $1"
        echo "Supported values: on, off"
        return 1
    }

    export SV_FORCE_AFFINE_BARYCENTRICS="$normalized"
    echo "Force affine barycentrics set to $SV_FORCE_AFFINE_BARYCENTRICS"
}

function c()
{
    if [ -d "$SV_OUT" ]; then
        echo "The output directory: $SV_OUT will be removed"
        read -p "Continue? (y/N): " CONFIRM
        if [ ! -z "$CONFIRM" ] && [ "$CONFIRM" = "y" -o "$CONFIRM" = "Y" ]; then
            if rm -rf "$SV_OUT"; then
                echo "removed"
            fi
        fi
    else
        echo "$SV_OUT has already been deleted"
    fi
}

function b()
{
    if [ ! "$SV_PROJECT_TOP" ]; then
        echo "Failed to locate the top of the tree"
        return 1
    fi

    if [ ! "$SV_OUT" = "" ]; then
        mkdir -p "$SV_OUT"
    fi

    cd "$SV_OUT" || return 1
    "$CMAKE" -G"Unix Makefiles" \
        -DCMAKE_BUILD_TYPE="$SV_BUILD_TYPE" \
        -DWITH_PROFILE_CUDA_TIME="$SV_CUDA_PROFILING" \
        -DWITH_CUDA_COMPUTE_SANITIZER="$SV_CUDA_SANITIZER" \
        -DWITH_TAA="$SV_TAA" \
        -DCUDARF_FORCE_AFFINE_BARYCENTRICS="$SV_FORCE_AFFINE_BARYCENTRICS" \
        "$SV_PROJECT_TOP" || return 1
    cd "$SV_PROJECT_TOP" || return 1

    local build_threads="${MAX_BUILD_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
    make -C "$SV_OUT" -j "$build_threads" "$@"

    "$SV_PROJECT_TOP/scripts/merge_compile_commands.sh"
}
