#!/usr/bin/env bash
#
#SBATCH -J ProjectSlurm
#SBATCH -c 2
#SBATCH -o project.out
#SBATCH -e project.err
#SBATCH -p wacc 
#SBATCH --gres=gpu:1

# echo "Loading CUDA Module"
module load cuda

nvcc canny.cu canny_main.cu canny_cpu.cu pixel.cu  -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o canny_main

./canny_main
