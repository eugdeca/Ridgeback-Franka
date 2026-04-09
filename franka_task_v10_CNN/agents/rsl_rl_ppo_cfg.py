# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlActorCriticCNNCfg, RslRlPpoAlgorithmCfg

num_steps_per_env_glob = 256 

@configclass
class FrankaPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for the Franka Reach Task with VISION."""
    
    
    num_steps_per_env = num_steps_per_env_glob
    max_iterations = 11000
    save_interval = 50
    experiment_name = "FrankaReach_v10"

    obs_groups = {
        "policy": ["policy_proprio", "policy_vision"], 
        "critic": ["critic"], 
    }

    policy = RslRlActorCriticCNNCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        
        actor_hidden_dims=[256, 128, 64], 
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        
        actor_cnn_cfg=RslRlActorCriticCNNCfg.CNNCfg(
            output_channels=[16, 32],
            kernel_size=3,
            stride=2,
            activation="elu",
            flatten=True
        ),
        critic_cnn_cfg=None,
    )
    
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,     
        learning_rate=5.0e-4,  
        num_learning_epochs=5, 
        num_mini_batches=16, 
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )