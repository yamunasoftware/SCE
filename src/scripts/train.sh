#!/bin/bash

echo "Started."
cd ../ml
python -B -c 'from logistic_interface import train; train()'
python -B -c 'from svm_interface import train; train()'
echo "Duration: $SECONDS seconds"