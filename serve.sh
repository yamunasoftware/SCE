#!/bin/bash

cd ../src/api
python -B startup.py
fastapi run serve.py --host 0.0.0.0 --port 8000