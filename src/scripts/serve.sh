#!/bin/bash

echo "Started."
cd ..
cd api
python serve.py
python -c 'import serve; serve.clear_log()'
echo "Duration: $SECONDS seconds"