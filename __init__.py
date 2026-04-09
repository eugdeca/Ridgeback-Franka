import gymnasium as gym

# Import agent files
from . import agents

##
# Register the Franka Reach environment in Gymnasium
##

gym.register(
    # 1. Unique ID: This is the name used in gym.make() and --task
    id="Isaac-FrankaReach-v10",
    
    # 2. Entry-Point: Exactly as in the tutorial, 
    #    points to the Isaac Lab base class.
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    
    # 3. Disable Gym checker (required for Isaac Lab)
    disable_env_checker=True,
    
    # 4. Arguments (kwargs) passed to ManagerBasedRLEnv
    kwargs={
        # This is the pointer to YOUR configuration class.
        # f"{__name__}" means "in this same package" (franka_task)
        "env_cfg_entry_point": f"{__name__}.franka_rl_env_cfg:FrankaReachEnvCfg",
        
        # (Optional but recommended) Pointer to training config
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Franka-Reach-Distill-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_rl_env_cfg:FrankaReachEnvCfg", 
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distill_cfg:FrankaLiftDistillationRunnerCfg",
    },
)