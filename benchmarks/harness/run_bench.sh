#!/usr/bin/env bash
# One-command driver for mnem v1.0 benchmarks.
# - Brings up 4 thread-pinned bench lanes via compose.yml
# - Runs 6 benches (LongMemEval, LoCoMo, ConvoMem, MemBench x 2, Hybrid v4)
#   in parallel via a 4-lane token-bucket dispatcher
# - Renders RESULTS.md from per-bench JSONs
# - Tears down

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HARNESS="${REPO_ROOT}/benchmarks/harness"
DATASETS="${REPO_ROOT}/benchmarks/datasets"
cd "${REPO_ROOT}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${REPO_ROOT}/benchmarks/results/${STAMP}"
LOGS="${OUT}/logs"
mkdir -p "${OUT}" "${LOGS}"

COMPOSE="docker compose -f ${HARNESS}/compose.yml"
LANES=(9876 9877 9878 9879)
LANE_SVCS=(mnem-bench-1 mnem-bench-2 mnem-bench-3 mnem-bench-4)

# ---------------------------------------------------------------------------
# 1. Verify datasets present
# ---------------------------------------------------------------------------
for f in \
    "${DATASETS}/longmemeval/longmemeval_s_cleaned.json" \
    "${DATASETS}/locomo/locomo10.json" \
    "${DATASETS}/membench/FirstAgent/simple.json" \
    "${DATASETS}/membench/FirstAgent/highlevel.json"; do
    if [ ! -f "${f}" ]; then
        echo "[fatal] dataset missing: ${f}"
        echo "Run: bash benchmarks/harness/download-datasets.sh"
        exit 1
    fi
done
echo "[ok] datasets present"

# ---------------------------------------------------------------------------
# 2. Build + bring up 4 lanes
# ---------------------------------------------------------------------------
echo "[build] mnem-http:bench-onnx-minilm (onnx-bundled features)"
${COMPOSE} build 2>&1 | tee "${LOGS}/build.log"

echo "[up] starting 4 thread-pinned lanes"
${COMPOSE} up -d 2>&1 | tee "${LOGS}/up.log"

# Host-side health probe (in-container probe disabled - debian-slim has no curl/wget)
for p in "${LANES[@]}"; do
    ok=0
    for i in $(seq 1 30); do
        if curl -sf "http://127.0.0.1:${p}/v1/healthz" >/dev/null; then
            ok=1; break
        fi
        sleep 1
    done
    if [ "${ok}" -ne 1 ]; then
        echo "[fatal] lane :${p} unhealthy after 30s"
        ${COMPOSE} logs
        ${COMPOSE} down
        exit 1
    fi
done
echo "[ok] all lanes healthy"

# ---------------------------------------------------------------------------
# 3. Bench definitions (FILE_BASE | adapter cmd template, {PORT} substituted)
# ---------------------------------------------------------------------------
DS_LME="${DATASETS}/longmemeval/longmemeval_s_cleaned.json"
DS_LOCOMO="${DATASETS}/locomo/locomo10.json"
DS_MEMBENCH="${DATASETS}/membench/FirstAgent"

declare -a BENCHES=(
  "longmemeval-500q|python ${HARNESS}/adapters/longmemeval_session.py --dataset ${DS_LME} --mnem-http http://127.0.0.1:{PORT} --limit 500 --top-k 10 --out ${OUT}/longmemeval-500q.json"
  "longmemeval-500q-hybrid-v4|python ${HARNESS}/adapters/longmemeval_session.py --dataset ${DS_LME} --mnem-http http://127.0.0.1:{PORT} --limit 500 --top-k 10 --hybrid-v4-boost --out ${OUT}/longmemeval-500q-hybrid-v4.json"
  "locomo-session|python ${HARNESS}/adapters/locomo.py --dataset ${DS_LOCOMO} --mnem-http http://127.0.0.1:{PORT} --granularity session --top-k 10 --out ${OUT}/locomo-session.json"
  "convomem-250|python ${HARNESS}/adapters/convomem.py --mnem-http http://127.0.0.1:{PORT} --limit 50 --top-k 10 --out ${OUT}/convomem-250.json"
  "membench-simple-roles|python ${HARNESS}/adapters/membench.py --data-dir ${DS_MEMBENCH} --mnem-http http://127.0.0.1:{PORT} --category simple --topic roles --limit 100 --top-k 5 --out ${OUT}/membench-simple-roles.json"
  "membench-highlevel-movie|python ${HARNESS}/adapters/membench.py --data-dir ${DS_MEMBENCH} --mnem-http http://127.0.0.1:{PORT} --category highlevel --topic movie --limit 100 --top-k 5 --out ${OUT}/membench-highlevel-movie.json"
)

# ---------------------------------------------------------------------------
# 4. Token-bucket dispatcher across 4 lanes
# ---------------------------------------------------------------------------
LANE_PIDS=("" "" "" "")
LANE_BUSY=("" "" "" "")

restart_lane () {
    local idx=$1
    docker restart "${LANE_SVCS[$idx]}" >/dev/null
    until curl -sf "http://127.0.0.1:${LANES[$idx]}/v1/healthz" >/dev/null; do sleep 1; done
}

dispatch () {
    local name=$1 cmd_template=$2 lane=$3
    local port=${LANES[$lane]}
    local cmd="${cmd_template//\{PORT\}/${port}}"
    local log="${LOGS}/${name}.log"
    local tstart=$(date +%s)
    echo "[${name}] -> lane :${port}"
    (
        eval "${cmd}" >"${log}" 2>&1
        rc=$?
        local tend=$(date +%s)
        echo "${name} rc=${rc} dur=$((tend - tstart))s lane=${port}" >>"${OUT}/timing.log"
    ) &
    LANE_PIDS[$lane]=$!
    LANE_BUSY[$lane]="${name}"
}

free_lane () {
    while true; do
        for i in 0 1 2 3; do
            if [ -z "${LANE_BUSY[$i]}" ]; then echo "$i"; return; fi
            local pid=${LANE_PIDS[$i]}
            if [ -n "${pid}" ] && ! kill -0 "${pid}" 2>/dev/null; then
                wait "${pid}" || true
                LANE_PIDS[$i]=""; LANE_BUSY[$i]=""
                restart_lane "$i"
                echo "$i"; return
            fi
        done
        sleep 1
    done
}

T0=$(date +%s)
for bench in "${BENCHES[@]}"; do
    name="${bench%%|*}"
    cmd="${bench#*|}"
    lane=$(free_lane)
    dispatch "${name}" "${cmd}" "${lane}"
done

for i in 0 1 2 3; do
    pid=${LANE_PIDS[$i]:-}
    [ -n "${pid}" ] && wait "${pid}" || true
done
T1=$(date +%s)
echo "[done] total wall=$((T1 - T0))s"

# ---------------------------------------------------------------------------
# 5. Render comparison table
# ---------------------------------------------------------------------------
python "${HARNESS}/comparison_table.py" --results "${OUT}" --out "${OUT}/RESULTS.md"
echo "[done] table -> ${OUT}/RESULTS.md"

# ---------------------------------------------------------------------------
# 6. Tear down
# ---------------------------------------------------------------------------
${COMPOSE} down 2>&1 | tee "${LOGS}/down.log"
echo "[done] artefacts -> ${OUT}"
