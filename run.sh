#!/bin/bash


for l in sparsemax softmax ste sts; do
    for g in True False; do
        if [ $g = "True" ]; then
            device=0
        else
            device=1
        fi
        python test_latent_variable_methods.py device=$device latent_mapping=$l gumbel=$g &
    done
done
