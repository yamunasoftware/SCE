#!/bin/bash

echo "Started."
cd ../src/ml
python -B -c 'from logistic_interface import train; train()'
python -B -c 'from svm_interface import train; train()'
echo "Duration: $SECONDS seconds"