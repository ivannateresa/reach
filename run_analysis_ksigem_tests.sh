#!/bin/bash

# Run the complete ksi Gem comparison matrix.
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

PIPELINE_SCRIPT="pipeline_ksigem_tests.py"
ANALYSIS_SCRIPT="analyse_ksigem_tests.py"

BRIGHT_CALIBRATORS=(
    "12_Mon"
    "HR2426"
    "HR2610"
)

FAINT_CALIBRATORS=(
    "HR2342"
    "HR2391"
    "HR2426"
)

TOTAL_CASES=20
CASE_NUMBER=0


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
echo "ALL ${TOTAL_CASES} KSI GEM CASES FINISHED SUCCESSFULLY"
echo "======================================================================"
