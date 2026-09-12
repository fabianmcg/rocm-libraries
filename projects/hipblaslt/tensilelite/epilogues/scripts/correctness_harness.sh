#!/bin/bash

export TENSILE_DISABLE_HELPER_CACHE=1
cd /home/fmoracor/rocm-libraries/projects/hipblaslt/tensilelite
rm -rf /tmp/fused-harness
./Tensile/bin/Tensile /home/fmoracor/rocm-libraries/projects/hipblaslt/tensilelite/epilogues/YAMLs/fused.yaml /tmp/fused-harness
