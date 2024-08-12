#!/bin/bash

cd src
cd scripts
cmd=""

while [ "$cmd" != "exit" ]
do
  read -p ">> " cmd

  if [[ "$cmd" == "serve" ]]; then
    bash serve.sh
  elif [[ "$cmd" == "test" ]]; then
    bash test.sh
  elif [[ "$cmd" == "train" ]]; then
    bash train.sh
  elif [[ "$cmd" == "exit" ]]; then
    echo "Exiting..."
  elif [[ "$cmd" == "help" ]]; then
    echo "Commands: serve, test, train, exit, help"
  else
    echo "Invalid Command"
  fi
done