work_path=$(pwd)
partition=${1}


job_name='learning_to_rank'

srun --mpi=pmi2 -p ${partition} --gres=gpu:1 --job-name=${job_name} \
    python -u prank_train.py --bias_number ${2} --iterations ${3} --embed_size ${4}
