python train_maniflow_rl.py \
    --config-path=equi_diffpo/config \
    --config-name=train_maniflow_pointcloud_rl \
    task_name=three_piece_assembly_d2 \
    rl_training.num_envs=16\
    policy.checkpoint="data/outputs/2026.02.13/01.12.01_train_maniflow_pointcloud_three_piece_assembly_d2/checkpoints/epoch0043-test_mean_score0.540.ckpt"\
        policy.noise_level=0.3\
    policy.noise_anneal=false
