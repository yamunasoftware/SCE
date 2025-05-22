#!/bin/bash

echo "Started."
cd ../test
python -B test.py
python -B api_test.py
echo "Duration: $SECONDS seconds"