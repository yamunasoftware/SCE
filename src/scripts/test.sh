#!/bin/bash

echo "Started."
cd ..
cd test
python test.py
cd ..
cd api
python api_test.py
python -c 'import serve; serve.clear_log()'
echo "Duration: $SECONDS seconds"