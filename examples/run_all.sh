#!/bin/bash
#

echo "Running all examples in the 'examples' folder. This may take a while..."

cd basic;       ./run_all.sh; cd ..

cd advanced;    ./run_all.sh; cd ..

cd animation; ./run_all.sh; cd ..

cd volumetric;  ./run_all.sh; cd ..

cd pyplot;      ./run_all.sh; cd ..

cd other;       ./run_all.sh; cd ..

echo "All examples have been processed."
exit(0)
