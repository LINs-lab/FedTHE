declare -a lr=0.1
declare -a seed=7
declare -a local_n_epochs=1
declare -a n_personalized_epochs=1
declare -a regularized_factor=0.1
declare -a participation_ratio=0.5
declare -a n_comm_rounds=5
declare -a n_clients=10
declare -a save_every_n_round=1

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $HOME/conda/envs/pytorch-py3.8/bin/python run.py \
    --arch simple_cnn --complex_arch master=simple_cnn,worker=simple_cnn --experiment debug \
    --data cifar10 \
    --train_data_ratio 1 --val_data_ratio 0 \
    --batch_size 32 --min_batch_size 32 --num_workers 0 \
    --partition_data_conf distribution=non_iid_dirichlet,non_iid_alpha=1,size_conf=1:1 \
    --n_clients ${n_clients} --participation_ratio ${participation_ratio} --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=federated_average \
    --n_comm_rounds ${n_comm_rounds} --local_n_epochs ${local_n_epochs} \
    --optimizer sgd --lr ${lr} --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --weight_decay 1e-4 --use_nesterov False --momentum_factor 0 \
    --display_log True --track_time True --display_tracked_time True --python_path $HOME/conda/envs/pytorch-py3.8/bin/python --hostfile hostfile \
    --manual_seed ${seed} --same_seed_process False
