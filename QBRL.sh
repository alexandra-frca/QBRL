#!/bin/bash
#SBATCH --job-name=myJOB
#SBATCH --time=0:10:0
#SBATCH --partition=fct
#SBATCH --qos=cpca095372023
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --nodes=1
#SBATCH --mem=10000
#SBATCH --ntasks-per-node=1

python3 script.py