# sh scripts/collect.sh
num_parts=10
for ((i=0; i<num_parts; i++)); do
  srun -p regular --mpi=pmi2 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=5 --kill-on-bad-exit=1 --quotatype auto python ram_collect.py --id $i &
done
wait
