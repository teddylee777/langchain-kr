#!/usr/bin/env zsh

set -e

export PYTHONPATH=$PWD
export APP_ENV=prod
# 필요 시에 local에서 실행할 것
#export APP_ENV=local

jupyter lab \
  --config config/jupyter.py

