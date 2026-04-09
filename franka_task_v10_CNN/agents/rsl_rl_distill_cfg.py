# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCNNCfg,
)


@configclass
class FrankaLiftDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 256
    max_iterations = 3000
    save_interval = 50
    resume = False
    load_run = "teacher_model"
    load_checkpoint = "model_9999.pt"
    experiment_name = "Franka_distill_v0"
    obs_groups = {"policy": ["policy_proprio", "policy_vision"], "teacher": ["critic"]}
    policy = RslRlDistillationStudentTeacherCNNCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[256, 128, 64],
        teacher_hidden_dims=[256, 128, 64], 
        activation="elu",
        student_cnn_cfg = {
            "output_channels": [32, 64, 64], 
            "kernel_size": 3,            
            "stride": 2,                 
            "activation": "elu",         
            "flatten": True              
        }
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=10,
        learning_rate=3.0e-4,
        gradient_length=15,
    )
