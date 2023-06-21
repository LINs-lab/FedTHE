import itertools


class NewConf(object):
    # create the list of hyper-parameters to be replaced.
    to_be_replaced = dict(
        # general for world.
        experiment=["debug"],
        # use world to control the distribution of clients on cuda devices.
        # for advanced usage, use world_conf instead, see ./pcode/utils/topology.py
        world=["0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1"],
        on_cuda=[True],
        python_path=["/opt/conda/bin/python"],
        hostfile=["env/hostfile"],
        manual_seed=[7],
        same_seed_process=[False],
        # general for the training.
        track_time=[True],
        display_tracked_time=[True],
        # general for fl.
        n_clients=[20],
        data=["cifar10"],
        data_dir=["~/data"],
        batch_size=[32],
        num_workers=[0],
        # fl master
        n_comm_rounds=[100],
        early_stopping_rounds=[0],
        # fl clients
        arch=["simple_cnn"],
        complex_arch=["master=simple_cnn,worker=simple_cnn"],
        optimizer=["sgd"],
        momentum_factor=[0],
        # likely to be changed
        local_n_epochs=[5],
        n_personalized_epochs=[1],
        lr=[0.01],
        personal_lr=[0.01],
        participation_ratio=[1.0],
        partition_data_conf=["distribution=non_iid_dirichlet,non_iid_alpha=0.1,size_conf=1:1",
                             "distribution=non_iid_dirichlet,non_iid_alpha=1.0,size_conf=1:1"],
        personalization_scheme=["method=Normal",
                                "method=Fine_tune",
                                "method=FedRep",
                                "method=Ditto",
                                "method=GMA",
                                "method=FedRod",
                                "method=Memo_global",
                                "method=Memo_personal",
                                "method=THE",
                                "method=THE_FT",
                                "method=T3A",
                                "method=ttt",
                                "method=tsa",
                                "method=knn_per",
                                ],
    )