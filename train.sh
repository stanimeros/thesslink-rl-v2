#!/usr/bin/env bash
#
# ThessLink RL -- Parallel training launcher
#
# Usage:
#   ./train.sh                  # train all algorithms (iql qmix vdn mappo coma)
#   ./train.sh qmix mappo       # train only qmix and mappo
#   ./train.sh --status          # live dashboard (watch -n 2; Ctrl+C to stop)
#   ./train.sh --kill            # kill all running training processes
#
# Results layout (epymarl/results/): logs/<alg>.log (nohup), sacred/, models/
# The tree is wiped before smoke and again after smoke so full training only
# sees fresh Sacred/model paths for the active ENV_CONFIG YAML.
#
set -euo pipefail

# Resolve repo root: dirname "$0" is wrong when invoked as `bash train.sh` from another
# directory ($0 is just "train.sh", so `cd .` follows the caller's cwd, not this file).
_script="${BASH_SOURCE[0]:-$0}"
if [[ -L "$_script" ]] && command -v readlink >/dev/null; then
    if _r="$(readlink -f "$_script" 2>/dev/null)"; then
        _script="$_r"
    fi
fi
SCRIPT_DIR="$(cd "$(dirname "$_script")" && pwd)"
cd "$SCRIPT_DIR"

ALL_ALGOS=(iql qmix vdn mappo coma)
RESULTS_DIR="epymarl/results"
# Absolute path so --status-once works under watch/cron regardless of caller cwd.
LOGS_DIR="$SCRIPT_DIR/$RESULTS_DIR/logs"
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

# EPyMARL + Sacred write under epymarl/results/{sacred,models}; nohup logs: epymarl/results/logs/<alg>.log
prepare_results_tree() {
    local phase="$1"
    log "Resetting results tree ($phase)..."
    rm -rf "$RESULTS_DIR"
    mkdir -p "$LOGS_DIR" "$RESULTS_DIR/sacred" "$RESULTS_DIR/models"
}

show_status() {
    # grep exits 1 when there is no match; with pipefail + set -e that aborts the script
    # before any row is printed. Stats lines can be missing early in training.
    set +o pipefail
    export LC_NUMERIC=C
    echo ""
    echo " ALG  |   T_ENV |   RETURN |   AGR% |    GM% | REACH% | EP_LEN"
    echo "------|---------|----------|--------|--------|--------|-------"
    for alg in "${ALL_ALGOS[@]}"; do
        local logf="$LOGS_DIR/${alg}.log"
        [ -f "$logf" ] || continue
        local tenv ret neg neg_opt reach eplen n n_perc no_perc r_perc
        tenv=$(grep -a 't_env:' "$logf" 2>/dev/null | tail -n1 | awk -F't_env: ' '{print $2}' | awk '{print $1}' | tr -d ',')
        # Return columns: IQL/MAPPO use per-agent rewards; QMIX/VDN/COMA use aggregated (see algo_extra_args)
        ret=$(grep -a 'test_total_return_mean:' "$logf" 2>/dev/null | tail -n1 | awk -F'test_total_return_mean: ' '{print $2}' | awk '{print $1}' | tr -d ',')
        if [ -z "$ret" ]; then
            ret=$(grep -a 'test_return_mean:' "$logf" 2>/dev/null | tail -n1 | awk -F'test_return_mean: ' '{print $2}' | awk '{print $1}' | tr -d ',')
        fi
        neg=$(grep -a 'test_negotiation_agreed_mean:' "$logf" 2>/dev/null | tail -n1 | awk -F'test_negotiation_agreed_mean: ' '{print $2}' | awk '{print $1}' | tr -d ',')
        neg_opt=$(grep -a 'test_negotiation_optimal_mean:' "$logf" 2>/dev/null | tail -n1 | awk -F'test_negotiation_optimal_mean: ' '{print $2}' | awk '{print $1}' | tr -d ',')
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
        n_perc=$(echo "${neg:-0} * 100" | bc -l 2>/dev/null || echo "0")
        no_perc=$(echo "${neg_opt:-0} * 100" | bc -l 2>/dev/null || echo "0")
        r_perc=$(echo "${reach:-0} * 100" | bc -l 2>/dev/null || echo "0")
        printf " %5s | %7s | %8.4f | %5.1f%% | %5.1f%% | %5.1f%% | %5.1f\n" \
            "$n" "${tenv:-0}" "${ret:-0}" "${n_perc:-0}" "${no_perc:-0}" "${r_perc:-0}" "${eplen:-0}"
    done
    set -o pipefail
    echo ""
}

# ── Parse arguments ──────────────────────────────────────────────────────

if [[ "${1:-}" == "--status" ]]; then
    # Same as: watch -n 2 'echo header && for alg in ...; do ...; done' — logic lives in show_status.
    exec watch -n 2 "$SCRIPT_DIR/train.sh" --status-once
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

log "Installing project requirements..."
pip install -r requirements.txt --quiet

# Same ENV_VERSION / ENV_CONFIG as smoke_test.py and visualize.py (via import config).
eval "$(python3 <<'PY'
import config
print(f"export ENV_VERSION={config.ENV_VERSION}")
print(f"export ENV_CONFIG={config.ENV_CONFIG!r}")
PY
)"

log "Environment version: v${ENV_VERSION} (env-config=${ENV_CONFIG})"

# ── EPyMARL ──────────────────────────────────────────────────────────────

if [ ! -d "epymarl" ]; then
    log "Cloning EPyMARL..."
    git clone https://github.com/uoe-agents/epymarl.git

    log "Installing dependencies..."
    pip install -r epymarl/requirements.txt --quiet
    pip install -e . --quiet
fi

log "Applying patches to EPyMARL..."
PATCH_DIR="$SCRIPT_DIR/epymarl_config/patches"
_patch_base="$PATCH_DIR/epymarl_01_thesslink_base.patch"
_patch_dual="$PATCH_DIR/epymarl_02_dual_policy.patch"
for _p in "$_patch_base" "$_patch_dual"; do
  if [[ ! -f "$_p" ]]; then
    err "Missing EPyMARL patch (commit it to the repo): $_p"
    exit 1
  fi
done
git -C epymarl checkout -- . 2>/dev/null || true
# New files from patch 02 are untracked; remove so re-apply works after checkout -- .
rm -f epymarl/src/controllers/dual_basic_controller.py \
    epymarl/src/modules/agents/dual_rnn_agent.py 2>/dev/null || true
for _patch in "$_patch_base" "$_patch_dual"; do
  if ! git -C epymarl apply "$_patch"; then
    err "git apply failed: $_patch (fresh clone + upstream EPyMARL revision mismatch?)"
    exit 1
  fi
done
log "Patches applied (thesslink base + dual-policy)."

log "Copying ThessLink env YAMLs into EPyMARL (epymarl_config/envs/*.yaml)..."
_copied=0
for _f in epymarl_config/envs/*.yaml; do
    [[ -f "$_f" ]] || continue
    cp "$_f" "$EPYMARL_SRC/config/envs/"
    ((_copied++)) || true
done
if ((_copied == 0)); then
    err "No YAML files in epymarl_config/envs/ — add thesslink*.yaml (or new versions) there."
    exit 1
fi

# Validate algorithm names
for alg in "${ALGOS[@]}"; do
    if [ ! -f "$EPYMARL_SRC/config/algs/${alg}.yaml" ]; then
        err "Unknown algorithm: $alg"
        err "Available: ${ALL_ALGOS[*]}"
        exit 1
    fi
done

# ── Smoke test ───────────────────────────────────────────────────────────

prepare_results_tree "before smoke — only QMIX short run + plots"
log "Smoke will use --env-config=${ENV_CONFIG} (see epymarl/src/config/envs/${ENV_CONFIG}.yaml)"

log "Running smoke test..."
if python smoke_test.py; then
    log "Smoke test passed!"
else
    err "Smoke test FAILED — aborting training."
    exit 1
fi

prepare_results_tree "after smoke — full multi-algo training"

# ── Launch training ──────────────────────────────────────────────────────

algo_extra_args() {
    # EPyMARL: only IQL + MAPPO support common_reward=False (per-agent rewards in the buffer).
    # QMIX / VDN / COMA require common_reward=True (rewards aggregated in GymmaWrapper).
    case "$1" in
        iql|mappo) echo "common_reward=False" ;;
        qmix|vdn|coma) echo "common_reward=True" ;;
        *) echo "common_reward=True" ;;
    esac
}

log "Launching ${#ALGOS[@]} algorithm(s): ${ALGOS[*]}"
echo ""

PIDS=()
for alg in "${ALGOS[@]}"; do
    logfile="$LOGS_DIR/${alg}.log"
    extra=$(algo_extra_args "$alg")
    log "  Starting $alg -> $logfile ${extra:+(${extra})}"
    nohup python "$EPYMARL_SRC/main.py" \
        --config="$alg" \
        --env-config="$ENV_CONFIG" \
        with \
        local_results_path=epymarl/results \
        save_model=True \
        save_model_interval=50000 \
        t_max=2000000 \
        $extra \
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
