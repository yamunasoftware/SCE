#!/bin/bash

echo "Started."
cd ../api
python -B serve.py
echo "Duration: $SECONDS seconds"