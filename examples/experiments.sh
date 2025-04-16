#!/bin/bash
# No 'set -e' here, so the script won't exit on error

CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir data/DTU/scan24/ --data_factor 1 --result_dir ./results/dtu_scan24 --sdf_loss --disable_viewer 

CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir data/DTU/scan37/ --data_factor 1 --result_dir ./results/dtu_scan37 --sdf_loss --disable_viewer 

CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir data/DTU/scan40/ --data_factor 1 --result_dir ./results/dtu_scan40 --sdf_loss --disable_viewer 

CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir data/DTU/scan55/ --data_factor 1 --result_dir ./results/dtu_scan55 --sdf_loss --disable_viewer 

CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir data/DTU/scan63/ --data_factor 1 --result_dir ./results/dtu_scan63 --sdf_loss --disable_viewer 


echo "All commands finished!"
