#!/bin/bash
#

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 1

echo "Running all examples in the 'examples' folder. This may take a while..."

exit_status=0
for directory in basic advanced animation volumetric pyplot extras; do
    (cd -- "$script_dir/$directory" && ./run_all.sh) || exit_status=1
done

echo "All examples have been processed."
exit "$exit_status"
