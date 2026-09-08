#!/bin/bash

# Compare each faint-only calibrator-removal result with its matching reference.
# Existing pipeline results are read only; no calibration is repeated.

set -u

DIAGNOSTIC_SCRIPT="diagnose_ksigem_faint_step_by_step.py"
BASELINE_MODE="${BASELINE_MODE:-ALL_BASELINES}"
N_BOOTSTRAPS="${N_BOOTSTRAPS:-1}"

FAINT_CALIBRATORS=(
    "HR2342"
    "HR2391"
    "HR2426"
)

for calibrator in "${FAINT_CALIBRATORS[@]}"; do

    echo ""
    echo "======================================================================"
    echo "ksi_Gem faint diagnostic: remove ${calibrator}"
    echo "baseline mode: ${BASELINE_MODE}"
    echo "======================================================================"

    python "${DIAGNOSTIC_SCRIPT}" \
        "${calibrator}" \
        --baseline-mode "${BASELINE_MODE}" \
        --n-bootstraps "${N_BOOTSTRAPS}"

    if [ $? -ne 0 ]; then
        echo "ERROR diagnosing faint removal: ${calibrator}"
        exit 1
    fi

done

echo ""
echo "All three ksi_Gem faint diagnostics finished successfully."
