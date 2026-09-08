#!/bin/bash

# Run the complete gam Lep comparison matrix.
#
# Baseline states:
#   1. all baselines
#   2. bad baselines removed
#
# Calibrator states:
#   1. no calibrator removed
#   2. every pair formed by removing one P104 calibrator
#      and one P106 calibrator
#
# gam Lep calibrators:
#
#   P104 FAINT:
#       HD38579
#       HD39913
#       HR2090
#
#   P106 FAINT:
#       HD38295
#       HD42747
#
# 3 x 2 = 6 calibrator pairs
# 6 x 2 baseline states = 12 cases
# + 2 reference cases = 14 total cases
#
# Example quick validation:
#
#   N_BOOTSTRAPS=1 ./run_gamlep_all_14_cases.sh
#
# Final example:
#
#   N_BOOTSTRAPS=1000 ./run_gamlep_all_14_cases.sh

set -u

# Both Python programs read this same value.
export N_BOOTSTRAPS="${N_BOOTSTRAPS:-1}"

PIPELINE_SCRIPT="pipeline_gamlep_tests.py"
ANALYSIS_SCRIPT="analyse_gamlep_tests.py"

P104_CALIBRATORS=(
    "HD38579"
    "HD39913"
    "HR2090"
)

P106_CALIBRATORS=(
    "HD38295"
    "HD42747"
    "HR2090"
)

TOTAL_CASES=20
CASE_NUMBER=0

echo ""
echo "======================================================================"
echo "gam Lep COMPLETE COMPARISON MATRIX"
echo "======================================================================"
echo "Bootstrap samples per case: ${N_BOOTSTRAPS}"
echo ""
echo "P104 calibrators:"
printf '  %s\n' "${P104_CALIBRATORS[@]}"
echo ""
echo "P106 calibrators:"
printf '  %s\n' "${P106_CALIBRATORS[@]}"
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
        echo "ERROR in gam Lep pipeline:"
        echo "  ${PIPELINE_SCRIPT} $*"
        exit 1
    fi

    python "${ANALYSIS_SCRIPT}" "$@"

    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR in gam Lep analysis:"
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
# 3-14. CALIBRATOR REMOVAL MATRIX
#
# For every P104/P106 calibrator pair:
#
#   NO_CAL
#       all baselines
#       remove one P104 calibrator
#       remove one P106 calibrator
#
#   NO_BL_NO_CAL
#       remove bad baselines
#       remove one P104 calibrator
#       remove one P106 calibrator
#
# 3 P104 x 2 P106 = 6 combinations
# 6 x 2 baseline states = 12 cases
#
# Total = 2 reference + 12 = 14 cases
# ======================================================================

for p104_calibrator in "${P104_CALIBRATORS[@]}"; do

    for p106_calibrator in "${P106_CALIBRATORS[@]}"; do

        run_case \
            NO_CAL \
            "${p104_calibrator}" \
            "${p106_calibrator}"

        run_case \
            NO_BL_NO_CAL \
            "${p104_calibrator}" \
            "${p106_calibrator}"

    done

done


echo ""
echo "======================================================================"
echo "ALL ${TOTAL_CASES} gam Lep CASES FINISHED SUCCESSFULLY"
echo "======================================================================"
