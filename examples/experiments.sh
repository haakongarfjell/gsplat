#!/bin/bash
# No 'set -e' here, so the script won't exit on error

# python mesh_extract.py --data_dir data/DTU/scan24_original/ --result_dir results/dtu_test_sdf_refine_02 --ckpt ckpts/ckpt_6999_rank0.pt --gt_mesh_dir results/meshes/monkey.ply --step 6999

CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir data/DTU/scan24/ --data_factor 1 --result_dir ./results/dtu_sdf_scan24 --sdf_loss --disable_viewer 
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir data/DTU/scan37/ --data_factor 1 --result_dir ./results/dtu_sdf_scan37 --sdf_loss --disable_viewer 
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir data/DTU/scan40/ --data_factor 1 --result_dir ./results/dtu_sdf_scan40 --sdf_loss --disable_viewer 
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir data/DTU/scan55/ --data_factor 1 --result_dir ./results/dtu_sdf_scan55 --sdf_loss --disable_viewer 
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir data/DTU/scan63/ --data_factor 1 --result_dir ./results/dtu_sdf_scan63 --sdf_loss --disable_viewer 
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir data/DTU/scan65/ --data_factor 1 --result_dir ./results/dtu_sdf_scan65 --sdf_loss --disable_viewer 
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir data/DTU/scan69/ --data_factor 1 --result_dir ./results/dtu_sdf_scan69 --sdf_loss --disable_viewer 
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir data/DTU/scan83/ --data_factor 1 --result_dir ./results/dtu_sdf_scan83 --sdf_loss --disable_viewer 
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir data/DTU/scan97/ --data_factor 1 --result_dir ./results/dtu_sdf_scan97 --sdf_loss --disable_viewer 
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir data/DTU/scan105/ --data_factor 1 --result_dir ./results/dtu_sdf_scan105 --sdf_loss --disable_viewer 
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir data/DTU/scan106/ --data_factor 1 --result_dir ./results/dtu_sdf_scan106 --sdf_loss --disable_viewer 
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir data/DTU/scan110/ --data_factor 1 --result_dir ./results/dtu_sdf_scan110 --sdf_loss --disable_viewer 
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir data/DTU/scan114/ --data_factor 1 --result_dir ./results/dtu_sdf_scan114 --sdf_loss --disable_viewer 
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir data/DTU/scan118/ --data_factor 1 --result_dir ./results/dtu_sdf_scan118 --sdf_loss --disable_viewer 
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir data/DTU/scan122/ --data_factor 1 --result_dir ./results/dtu_sdf_scan122 --sdf_loss --disable_viewer 

