#!/bin/bash

echo "Started."
cd ../src/api
python -B serve.py
echo "Duration: $SECONDS seconds"