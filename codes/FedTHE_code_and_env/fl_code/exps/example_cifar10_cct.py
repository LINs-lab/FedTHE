import itertools


class NewConf(object):
    # create the list of hyper-parameters to be replaced.
    to_be_replaced = dict(
        # general for world.
        experiment=["debug"],
        world_conf=["0,0,1,1,100"],
        on_cuda=[True],
        python_path=["$HOME/conda/envs/pytorch-py3.8/bin/python"],
        hostfile=["hostfile"],
        manual_seed=[7],
        same_seed_process=[False],
        # general for the training.
        track_time=[True],
        display_tracked_time=[True],
        # general for fl.
        n_clients=[20],
        data=["cifar10"],
        data_dir=["your_data_directory"],
        batch_size=[32],
        num_workers=[0],
        # fl master
        n_comm_rounds=[100],
        early_stopping_rounds=[0],
        # fl clients
        rep_len=[128],  # determined by the representation width of model
        comm_buffer_size=[200],
        arch=["compact_conv_transformer"],
        complex_arch=[
            "master=compact_conv_transformer,worker=compact_conv_transformer"
        ],
        optimizer=["adam"],
        # likely to be changed
        local_n_epochs=[5],
        n_personalized_epochs=[1],
        lr=[0.001],
        personal_lr=[0.001],
        participation_ratio=[1.0],
        partition_data_conf=[
            "distribution=non_iid_dirichlet,non_iid_alpha=1.0,size_conf=1:1"
        ],
        personalization_scheme=[
            "method=Fine_tune",
            "method=T3A",
            "method=FedRod",
            "method=THE",
            "method=THE_FT",
            "method=Memo_personal",
        ],
    )
