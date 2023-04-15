# sh scripts/train.sh
srun -p regular --mpi=pmi2 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=5 --kill-on-bad-exit=1 python ram_train.py
