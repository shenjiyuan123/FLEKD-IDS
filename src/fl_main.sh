#!/bin/bash
#SBATCH -J fl
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --partition=RTXq
#SBATCH -w node31
#SBATCH --output /export/home2/jiyuan/CIC2019/outs/flmain.out


# CUDA_VISIBLE_DEVICES=0 python src/fl_main.py --KD --noniid --noniid_strategy serial
CUDA_VISIBLE_DEVICES=0 python src/fl_main.py --KD --drop_label --drop_strategy serial --distill_epochs 2 --wk_iters 15 --iters 1