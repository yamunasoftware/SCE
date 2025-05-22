#!/bin/bash

cd src/scripts
while true
do
  read -p "SCE >> " cmd

  if [[ "$cmd" == "serve" ]]; then
    bash serve.sh
  elif [[ "$cmd" == "test" ]]; then
    bash test.sh
  elif [[ "$cmd" == "train" ]]; then
    bash train.sh
  elif [[ "$cmd" == "clear" ]]; then
    bash clear.sh
  elif [[ "$cmd" == "exit" ]]; then
    echo "Exiting..."
    exit 0
  elif [[ "$cmd" == "help" ]]; then
    echo "Commands: serve, test, train, exit, help"
  else
    echo "Invalid Command"
  fi
done