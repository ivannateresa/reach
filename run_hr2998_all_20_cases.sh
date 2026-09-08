#!/bin/bash

# Run the complete HR2998 comparison matrix.
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

set -u

# Both Python programs read this same value.  For a quick validation use:
# N_BOOTSTRAPS=1 ./run_hr2998_all_20_cases.sh
export N_BOOTSTRAPS="${N_BOOTSTRAPS:-1}"

PIPELINE_SCRIPT="pipeline_hr2998_tests.py"
ANALYSIS_SCRIPT="analyse_hr2998_tests.py"

BRIGHT_CALIBRATORS=(
    "HR_3069"
    "HD_65491"
    "HD_66162"
)

FAINT_CALIBRATORS=(
    "HD_61574"
    "HR_3069"
    "HD_65187"
)

TOTAL_CASES=20
CASE_NUMBER=0

echo "HR2998 bootstrap samples per case: ${N_BOOTSTRAPS}"


run_case() {

    CASE_NUMBER=$((CASE_NUMBER + 1))

    echo ""
    echo "======================================================================"
    echo "CASE ${CASE_NUMBER}/${TOTAL_CASES}: $*"
    echo "======================================================================"

    python "${PIPELINE_SCRIPT}" "$@"

    if [ $? -ne 0 ]; then
        echo "ERROR in pipeline: $*"
        exit 1
    fi

    python "${ANALYSIS_SCRIPT}" "$@"

    if [ $? -ne 0 ]; then
        echo "ERROR in analysis: $*"
        exit 1
    fi
}


# Reference cases: no calibrator removed.
run_case ALL
run_case NO_BL


# Remove one calibrator from bright and one from faint.
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
echo "ALL ${TOTAL_CASES} HR2998 CASES FINISHED SUCCESSFULLY"
echo "======================================================================"
