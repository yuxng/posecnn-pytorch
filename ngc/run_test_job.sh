#!/bin/bash

# Usage: ./train_ngc.sh [run-name] [script-path]

set -e
set -v

ngc batch run \
    --instance "dgx1v.16g.2.norm" \
    --name "test_job" \
    --image "nvcr.io/nvidian/robotics/posecnn-pytorch:latest" \
    --result /result \
    --datasetid "58777:/posecnn/data/models" \
    --datasetid "58774:/posecnn/data/backgrounds" \
    --datasetid "8187:/posecnn/data/coco" \
    --workspace posecnn:/posecnn \
    --commandline "sleep 168h"
