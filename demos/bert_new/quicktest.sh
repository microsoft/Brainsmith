#!/bin/bash
# Quick test script for BERT accelerator using ultra-small configuration

# Use ultra-small mode for fast testing (96D hidden, 1 layer, 3 heads)
python ./end2end_bert.py \
    --output-dir ./quicktest_output \
    --target-fps 3000 \
    --clock-period 5.0 \
    --board V80
