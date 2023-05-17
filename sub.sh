#!/bin/bash
module load cuda/11.4
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 15
#SBATCH -M swarm
#SBATCH -p gpu
#SBATCH --gres=gpu:3
#SBATCH --no-requeue
python train.py