#!/usr/bin/env bash
# global-run.sh - combine and run clusterOT/yelp.sh, OT/yelp.sh, fedOT/yelp.sh
# Usage: ./global-run.sh [--cluster] [--ot] [--fed] [--all] [--parallel] [--help]

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# print current dir
echo "Running global-yelp.sh from root dir: $ROOT_DIR"
SCRIPTS=(
    "new_scripts/clusterOT/yelp.sh"
    "new_scripts/OT/yelp.sh"
    "new_scripts/fedOT/yelp.sh"
)
LOGDIR="$ROOT_DIR/logs"
PARALLEL=false

# flags default: run all unless specific flags provided
RUN_CLUSTER=false
RUN_OT=false
RUN_FED=false
ANY_FLAG=false

print_help() {
    cat <<EOF
Usage: $0 [--cluster] [--ot] [--fed] [--all] [--parallel] [--help]
    --cluster   run clusterOT/yelp.sh
    --ot        run OT/yelp.sh
    --fed       run fedOT/yelp.sh
    --all       run all (default)
    --parallel  run selected scripts in parallel (background) and wait
    --help      show this help
EOF
}

# parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster) RUN_CLUSTER=true; ANY_FLAG=true; shift ;;
        --ot)      RUN_OT=true;      ANY_FLAG=true; shift ;;
        --fed)     RUN_FED=true;     ANY_FLAG=true; shift ;;
        --all)     RUN_CLUSTER=true; RUN_OT=true; RUN_FED=true; ANY_FLAG=true; shift ;;
        --parallel) PARALLEL=true; shift ;;
        --help) print_help; exit 0 ;;
        *) echo "Unknown arg: $1"; print_help; exit 2 ;;
    esac
done

# If no specific flags, run all
if ! $ANY_FLAG; then
    RUN_CLUSTER=true
    RUN_OT=true
    RUN_FED=true
fi

mkdir -p "$LOGDIR"

run_one() {
    local name="$1" relpath="$2"
    local path="$ROOT_DIR/$relpath"
    local logfile="$LOGDIR/${name}.log"

    if [[ ! -f "$path" ]]; then
        echo "[SKIP] $relpath not found at $path"
        return 0
    fi

    echo "===== START: $name ($relpath) ====="
    # ensure executable or run with bash
    if [[ -x "$path" ]]; then
        if $PARALLEL; then
            "$path" 2>&1 | tee "$logfile" &
            echo "[STARTED in background] $name -> $logfile"
        else
            "$path" 2>&1 | tee "$logfile"
        fi
    else
        if $PARALLEL; then
            bash "$path" 2>&1 | tee "$logfile" &
            echo "[STARTED in background] $name -> $logfile"
        else
            bash "$path" 2>&1 | tee "$logfile"
        fi
    fi
    echo "===== END: $name ====="
}

# Run selected scripts in desired order (cluster -> OT -> fed)
$RUN_CLUSTER && run_one "clusterOT-yelp" "${SCRIPTS[0]}"
$RUN_OT      && run_one "OT-yelp"        "${SCRIPTS[1]}"
$RUN_FED     && run_one "fedOT-yelp"     "${SCRIPTS[2]}"

# wait if parallel
if $PARALLEL; then
    wait
fi

echo "All requested jobs finished. Logs are in: $LOGDIR"