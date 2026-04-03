#!/usr/bin/env bash
#
# ThessLink RL v2 -- Parallel training launcher
#
# Usage:
#   ./train.sh                  # train all algorithms (iql qmix vdn mappo coma)
#   ./train.sh qmix mappo       # train only qmix and mappo
#   ./train.sh --status          # show live training dashboard
#   ./train.sh --kill            # kill all running training processes
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ALL_ALGOS=(iql qmix vdn mappo coma)
RESULTS_DIR="epymarl/results"
LOGS_DIR="$RESULTS_DIR/logs"
EPYMARL_SRC="epymarl/src"
VENV=".venv/bin/activate"

# ── Helpers ──────────────────────────────────────────────────────────────

log()  { echo -e "\033[1;32m[train]\033[0m $*"; }
warn() { echo -e "\033[1;33m[train]\033[0m $*"; }
err()  { echo -e "\033[1;31m[train]\033[0m $*" >&2; }

kill_training() {
    log "Killing running training processes..."
    pkill -f "$EPYMARL_SRC/main.py" 2>/dev/null && log "Killed." || log "No processes found."
}

show_status() {
    export LC_NUMERIC=C
    echo ""
    echo " ALG  |   T_ENV |   RETURN | REACH% | EP_LEN"
    echo "------|---------|----------|--------|-------"
    for alg in "${ALL_ALGOS[@]}"; do
        local logf="$LOGS_DIR/${alg}.log"
        [ -f "$logf" ] || continue
        local tenv ret reach eplen n r_perc
        tenv=$(grep -a 't_env:' "$logf" 2>/dev/null | tail -n1 | awk -F't_env: ' '{print $2}' | awk '{print $1}' | tr -d ',')
        ret=$(grep -a 'test_return_mean:' "$logf" 2>/dev/null | tail -n1 | awk -F'test_return_mean: ' '{print $2}' | awk '{print $1}' | tr -d ',')
        reach=$(grep -a 'test_battle_won_mean:' "$logf" 2>/dev/null | tail -n1 | awk -F'test_battle_won_mean: ' '{print $2}' | awk '{print $1}' | tr -d ',')
        eplen=$(grep -a 'test_ep_length_mean:' "$logf" 2>/dev/null | tail -n1 | awk -F'test_ep_length_mean: ' '{print $2}' | awk '{print $1}' | tr -d ',')
        case $alg in
            iql)   n="IQL  ";;
            qmix)  n="QMIX ";;
            vdn)   n="VDN  ";;
            mappo) n="MAPPO";;
            coma)  n="COMA ";;
            *)     n="$alg";;
        esac
        r_perc=$(echo "${reach:-0} * 100" | bc -l 2>/dev/null || echo "0")
        printf " %5s | %7s | %8.4f | %5.1f%% | %5.1f\n" \
            "$n" "${tenv:-0}" "${ret:-0}" "$r_perc" "${eplen:-0}"
    done
    echo ""
}

# ── Parse arguments ──────────────────────────────────────────────────────

if [[ "${1:-}" == "--status" ]]; then
    watch -n 5 "$0 --status-once"
    exit 0
fi

if [[ "${1:-}" == "--status-once" ]]; then
    show_status
    exit 0
fi

if [[ "${1:-}" == "--kill" ]]; then
    kill_training
    exit 0
fi

# Algorithms to train: args or all
if [[ $# -gt 0 ]]; then
    ALGOS=("$@")
else
    ALGOS=("${ALL_ALGOS[@]}")
fi

# ── Setup ────────────────────────────────────────────────────────────────

log "Pulling latest changes..."
git fetch origin main
git reset --hard origin/main

log "Killing previous training processes..."
kill_training

# ── Virtualenv ───────────────────────────────────────────────────────────

if [ ! -f "$VENV" ]; then
    log "Creating virtualenv..."
    python3 -m venv .venv
fi
log "Activating virtualenv..."
source "$VENV"

# ── EPyMARL ──────────────────────────────────────────────────────────────

if [ ! -d "epymarl" ]; then
    log "Cloning EPyMARL..."
    git clone https://github.com/uoe-agents/epymarl.git

    log "Installing dependencies..."
    pip install -r epymarl/requirements.txt --quiet
    pip install -e . --quiet
fi

log "Copying thesslink env config into EPyMARL..."
cp epymarl_config/thesslink.yaml "$EPYMARL_SRC/config/envs/thesslink.yaml"

log "Applying patches to EPyMARL..."
git -C epymarl checkout -- . 2>/dev/null || true
git -C epymarl apply ../epymarl_config/patches/epymarl.patch && log "Patches applied." || warn "Patches already applied or failed."

# Validate algorithm names
for alg in "${ALGOS[@]}"; do
    if [ ! -f "$EPYMARL_SRC/config/algs/${alg}.yaml" ]; then
        err "Unknown algorithm: $alg"
        err "Available: ${ALL_ALGOS[*]}"
        exit 1
    fi
done

# ── Smoke test ───────────────────────────────────────────────────────────

log "Running smoke test..."
if python smoke_test.py; then
    log "Smoke test passed!"
else
    err "Smoke test FAILED — aborting training."
    exit 1
fi

log "Cleaning previous results..."
rm -rf "$RESULTS_DIR"/*
mkdir -p "$LOGS_DIR"

# ── Launch training ──────────────────────────────────────────────────────

log "Launching ${#ALGOS[@]} algorithm(s): ${ALGOS[*]}"
echo ""

PIDS=()
for alg in "${ALGOS[@]}"; do
    logfile="$LOGS_DIR/${alg}.log"
    log "  Starting $alg -> $logfile"
    nohup python "$EPYMARL_SRC/main.py" \
        --config="$alg" \
        --env-config=thesslink \
        with \
        local_results_path=epymarl/results \
        save_model=True \
        save_model_interval=50000 \
        t_max=2000000 \
        > "$logfile" 2>&1 &
    PIDS+=($!)
done

echo ""
log "All training jobs launched:"
for i in "${!ALGOS[@]}"; do
    echo "  ${ALGOS[$i]}  PID=${PIDS[$i]}  log=$LOGS_DIR/${ALGOS[$i]}.log"
done

echo ""
log "Monitor with:  ./train.sh --status"
log "Kill all with: ./train.sh --kill"
log "Tail a log:    tail -f $LOGS_DIR/<algo>.log"
