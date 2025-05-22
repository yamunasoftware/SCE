#!/bin/bash

echo "Started."
cd ../api
python -B -c 'import serve; serve.clear_log()'
echo "Duration: $SECONDS seconds"