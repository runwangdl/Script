# #!/bin/bash

# # Initialize linting error log file
# echo "Generating linting error report..." > linting_error_report.log

# # Format Python with yapf
# echo "Running yapf for Python formatting..." >> linting_error_report.log
# yapf -rpd -e "third_party/" -e "install/" -e "toolchain/" -e "DeeployTest/Tests/" . > /dev/null 2>> linting_error_report.log

# echo "Running isort for Python imports formatting..." >> linting_error_report.log
# Run isort --sg "**/third_party/*"  --sg "install/*" --sg "toolchain/*" ./ -c -v
#   isort --sg "**/third_party/*"  --sg "install/*" --sg "toolchain/*" ./ -c -v
#   autoflake -c -r --remove-all-unused-imports --ignore-init-module-imports --exclude "*/third_party/**" ./
#   shell: bash --noprofile --norc -e -o pipefail {0}
# echo "Running clang-format for C/C++ formatting..." >> linting_error_report.log
# # Format C/C++ files using clang-format
# python scripts/run_clang_format.py -e "*/third_party/*" -e "*/install/*" -e "*/toolchain/*" -ir --clang-format-executable=${LLVM_INSTALL_DIR}/bin/clang-format ./ scripts > /dev/null 2>> linting_error_report.log


# # License checks
# echo "Checking Python file licenses..." >> linting_error_report.log
# grep -Lr "SPDX-License-Identifier: Apache-2.0" --exclude-dir="toolchain" --exclude-dir="install" --exclude-dir=".git" . --exclude-dir="third_party" --exclude-dir="TEST_*" --exclude "run_clang_format.py" | grep ".*\.py$" || [[ $? == 1 ]] >> linting_error_report.log 2>&1

# echo "Checking C file licenses..." >> linting_error_report.log
# grep -Lr "SPDX-License-Identifier: Apache-2.0" --exclude-dir="toolchain" --exclude-dir="install" --exclude-dir=".git" . --exclude-dir="third_party" --exclude-dir="TEST_*" --exclude-dir="runtime" | grep ".*\.c$" || [[ $? == 1 ]] >> linting_error_report.log 2>&1

# echo "Checking C header file licenses..." >> linting_error_report.log
# grep -Lr "SPDX-License-Identifier: Apache-2.0" --exclude-dir="toolchain" --exclude-dir="install" --exclude-dir=".git" . --exclude-dir="third_party" --exclude-dir="TEST_*" --exclude-dir="runtime" | grep ".*\.h$" || [[ $? == 1 ]] >> linting_error_report.log 2>&1


# echo "Linting error report generated in linting_error_report.log."


#!/bin/bash
yapf -rpi -e "third_party/" -e "install/" -e "toolchain/" -e "DeeployTest/Tests" .
# Initialize linting error log file
echo "Generating linting error report..." > linting_error_report.log

# Run isort and check for errors
echo "Running isort for Python imports formatting..." >> linting_error_report.log
isort_output=$(isort --sg "**/third_party/*" --sg "install/*" --sg "toolchain/*" ./ -c -v 2>&1)
isort_exit_code=$?
if [ $isort_exit_code -ne 0 ]; then
    echo "$isort_output" >> linting_error_report.log
    echo "isort detected issues. See linting_error_report.log for details."
    exit $isort_exit_code
fi

# Run autoflake and check for errors
echo "Running autoflake to remove unused imports..." >> linting_error_report.log
autoflake_output=$(autoflake -c -r --remove-all-unused-imports --ignore-init-module-imports --exclude "*/third_party/**" ./ 2>&1)
autoflake_exit_code=$?
if [ $autoflake_exit_code -ne 0 ]; then
    echo "$autoflake_output" >> linting_error_report.log
    echo "autoflake detected issues. See linting_error_report.log for details."
    exit $autoflake_exit_code
fi

echo "Linting error report generated in linting_error_report.log."
