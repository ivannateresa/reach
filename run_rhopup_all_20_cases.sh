#!/bin/bash

# Run the complete rho Pup comparison matrix.
#
# Baseline states:
#   1. all baselines
#   2. bad baselines removed
#
# Calibrator states:
#   1. no calibrator removed
#   2. every pair formed by removing one bright calibrator and one faint
#      calibrator (3 x 3 = 9 pairs)
#
# Total: 2 x (1 + 9) = 20 pipeline + analysis runs.
#
# rho Pup calibrators:
#
#   BRIGHT:
#       HR3317
#       HD59936
#       HD63734
#
#   FAINT:
#       HD65181
#       HD70942
#       HR3317
#
# Example quick validation:
#
#   N_BOOTSTRAPS=1 ./run_rhopup_all_20_cases.sh
#
# Final example:
#
#   N_BOOTSTRAPS=1000 ./run_rhopup_all_20_cases.sh

set -u

# Both Python programs read this same value.
export N_BOOTSTRAPS="${N_BOOTSTRAPS:-1}"

PIPELINE_SCRIPT="pipeline_rhopup_tests.py"
ANALYSIS_SCRIPT="analyse_rhopup_tests.py"

BRIGHT_CALIBRATORS=(
    "HR3317"
    "HD59936"
    "HD63734"
)

FAINT_CALIBRATORS=(
    "HD65181"
    "HD70942"
    "HR3317"
)

TOTAL_CASES=20
CASE_NUMBER=0

echo ""
echo "======================================================================"
echo "rho Pup COMPLETE COMPARISON MATRIX"
echo "======================================================================"
echo "Bootstrap samples per case: ${N_BOOTSTRAPS}"
echo ""
echo "Bright calibrators:"
printf '  %s\n' "${BRIGHT_CALIBRATORS[@]}"
echo ""
echo "Faint calibrators:"
printf '  %s\n' "${FAINT_CALIBRATORS[@]}"
echo "======================================================================"


run_case() {

    CASE_NUMBER=$((CASE_NUMBER + 1))

    echo ""
    echo "======================================================================"
    echo "CASE ${CASE_NUMBER}/${TOTAL_CASES}: $*"
    echo "======================================================================"

    python "${PIPELINE_SCRIPT}" "$@"

    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR in rho Pup pipeline:"
        echo "  ${PIPELINE_SCRIPT} $*"
        exit 1
    fi

    python "${ANALYSIS_SCRIPT}" "$@"

    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR in rho Pup analysis:"
        echo "  ${ANALYSIS_SCRIPT} $*"
        exit 1
    fi
}


# ======================================================================
# 1-2. REFERENCE CASES
#
# ALL:
#   all baselines + all calibrators
#
# NO_BL:
#   bad baselines removed + all calibrators
# ======================================================================

run_case ALL

run_case NO_BL


# ======================================================================
# 3-20. CALIBRATOR REMOVAL MATRIX
#
# For every bright/faint calibrator pair:
#
#   NO_CAL
#       all baselines
#       remove one bright calibrator
#       remove one faint calibrator
#
#   NO_BL_NO_CAL
#       remove bad baselines
#       remove one bright calibrator
#       remove one faint calibrator
#
# 3 bright x 3 faint = 9 combinations
# 9 x 2 baseline states = 18 cases
#
# Total = 2 reference + 18 = 20 cases
# ======================================================================

for bright_calibrator in "${BRIGHT_CALIBRATORS[@]}"; do

    for faint_calibrator in "${FAINT_CALIBRATORS[@]}"; do

        run_case \
            NO_CAL \
            "${bright_calibrator}" \
            "${faint_calibrator}"

        run_case \
            NO_BL_NO_CAL \
            "${bright_calibrator}" \
            "${faint_calibrator}"

    done

done


echo ""
echo "======================================================================"
echo "ALL ${TOTAL_CASES} rho Pup CASES FINISHED SUCCESSFULLY"
echo "======================================================================"
