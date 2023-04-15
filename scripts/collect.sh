# sh scripts/run.sh
num_parts=25
for ((i=0; i<num_parts; i++)); do
  srun -p regular --mpi=pmi2 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=5 --kill-on-bad-exit=1 python ram_collect.py --id $i &
  sleep 180
done
wait
