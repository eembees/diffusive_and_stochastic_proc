#!/bin/zsh
for i in $(seq 0.1 0.1 0.9)
  do
    python ch11_3_TASEP.py --density $i
  done