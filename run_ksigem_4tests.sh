#!/bin/bash

echo "============================================================"
echo "1/4 - ALL"
echo "============================================================"

python pipeline_ksigem_tests.py ALL

if [ $? -ne 0 ]; then
    echo "ERROR in ALL"
    exit 1
fi


echo ""
echo "============================================================"
echo "2/4 - NO_BL"
echo "============================================================"

python pipeline_ksigem_tests.py NO_BL

if [ $? -ne 0 ]; then
    echo "ERROR in NO_BL"
    exit 1
fi


echo ""
echo "============================================================"
echo "3/4 - NO_CAL"
echo "bright: remove HR2426"
echo "faint : remove HD123255"
echo "============================================================"

python pipeline_ksigem_tests.py NO_CAL HR2426 HD123255


if [ $? -ne 0 ]; then
    echo "ERROR in NO_CAL"
    exit 1
fi


echo ""
echo "============================================================"
echo "4/4 - NO_BL_NO_CAL"
echo "bright: remove HR2426"
echo "faint : remove HR2610"
echo "============================================================"

python pipeline_ksigem_tests.py NO_BL_NO_CAL HR2426 HD123255

if [ $? -ne 0 ]; then
    echo "ERROR in NO_BL_NO_CAL"
    exit 1
fi


echo ""
echo "============================================================"
echo "ALL 4 KSI GEM TESTS FINISHED SUCCESSFULLY"
echo "============================================================"
