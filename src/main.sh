#!/bin/bash
#SBATCH -J fl
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --partition=RTXq
#SBATCH -w node31
#SBATCH --output /export/home2/jiyuan/CIC2019/outs/main.out

CUDA_VISIBLE_DEVICES=0 python src/main.py