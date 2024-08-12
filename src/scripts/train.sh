#!/bin/bash

echo "Started."
cd ..
cd ml
python -c 'import training; training.train()'
echo "Duration: $SECONDS seconds"