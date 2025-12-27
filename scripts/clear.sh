#!/bin/bash

echo "Started."
cd ../src/api
python -B -c 'import serve; serve.clear_log()'
echo "Duration: $SECONDS seconds"