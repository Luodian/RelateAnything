# sh scripts/collect.sh
# num_parts=10
# for ((i=0; i<num_parts; i++)); do
#   srun -p regular --mpi=pmi2 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=5 --kill-on-bad-exit=1 --quotatype auto python ram_collect.py --id $i &
# done
# wait

num_parts=10
start_id=40000
total_images=10000

for ((i=0; i<num_parts; i++)); do
  srun -p regular --mpi=pmi2 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=5 \
  --kill-on-bad-exit=1 --job-name collect --quotatype auto \
  python ram_collect.py --id $i \
  --start_id $start_id --num_parts $num_parts --total_images $total_images &
done

# num_parts=4
# for ((i=0; i<num_parts; i++)); do
#   srun -p regular --mpi=pmi2 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=5 --kill-on-bad-exit=1 --job-name collect --quotatype auto python ram_collect.py --id $i --start_id 10000&
# done
# wait

# for ((i=0; i<num_parts; i++)); do
#   srun -p regular --mpi=pmi2 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=5 --kill-on-bad-exit=1 --quotatype auto python ram_collect.py --id $i --start_id 20000&
# done
# wait

# for ((i=0; i<num_parts; i++)); do
#   srun -p regular --mpi=pmi2 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=5 --kill-on-bad-exit=1 --quotatype auto python ram_collect.py --id $i --start_id 30000&
# done
# wait

# for ((i=0; i<num_parts; i++)); do
#   srun -p regular --mpi=pmi2 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=5 --kill-on-bad-exit=1 --quotatype auto python ram_collect.py --id $i --start_id 40000&
# done
# wait