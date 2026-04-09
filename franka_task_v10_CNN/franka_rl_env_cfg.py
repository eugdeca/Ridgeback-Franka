import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm, SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from .franka_base_env_cfg import FrankaSceneCfg, ActionsCfg, ObservationsCfg, EventCfg, CommandsCfg

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp_lift
from . import reward_custom, curriculum_custom

num_steps_start = 2500+1000
num_steps_1 =  4500+1000
num_steps_2 = 0+1000
num_steps_3 = 2000+1000
num_steps_4 = 7500+1000
num_steps_5 = 9500+1000
num_steps_6 = 5000+1000
num_steps_7 = 7000+1000

# ================================================================================
#                               REWARDS 
# ================================================================================

@configclass
class RewardsCfg:
    """Reward terms for Franka reaching task."""
    # ==========================================================
    # GENERAL REWARDS
    # ==========================================================

    # Reward to penalize control actions that saturate actuators
    joint_vel_limits_penalty = RewTerm(
        func=reward_custom.joint_vel_limits_exp,
        weight=-0.01,
    )

    # Reward on Franka action rate, excluding fingers
    action_rate = RewTerm(func=reward_custom.action_rate_l2_exp, 
                          weight= -1,      # 1.0 in the successful test
                          params={"scale":0.05}, 
    )

    # # Reward on base velocity to keep it stationary
    # limit_specific_joints_vel_base = RewTerm(
    #     func=reward_custom.joint_vel_exp,  
    #     weight=0.6,                          
    #     params={
    #         "asset_cfg": SceneEntityCfg("franka"), 
    #         "joint_names_to_check": [
    #             "dummy_base_prismatic_x_joint",
    #             "dummy_base_prismatic_y_joint",
    #             "dummy_base_revolute_z_joint"],
    #         "scale" : 1.5
    #     }
    # )

    # ==========================================================
    # EE APPROACH TO CUBE REWARDS
    # ==========================================================

    # # Reward on distance between ee and cube on x-y
    # end_effector_position_tracking_x_y = RewTerm(
    #     func=reward_custom.position_command_error_xy,
    #     weight=2,           #3 a= 0.45
    #     params={"asset_cfg": SceneEntityCfg("franka", body_names="endeffector"), "target_cfg": SceneEntityCfg("cuboid"),"scale" : 0.7},
    # )


    # # Reward on distance between ee and cube on x-y in proximity
    # end_effector_position_tracking_x_y_proximity = RewTerm(
    #     func=reward_custom.position_command_error_xy,
    #     weight=2,          #10 a=0.05
    #     params={"asset_cfg": SceneEntityCfg("franka", body_names="endeffector"), "target_cfg": SceneEntityCfg("cuboid"),"scale" : 0.06},
    # )

    # # Reward on distance between ee and cube on z-axis
    # end_effector_position_tracking_z = RewTerm(
    #     func=reward_custom.position_command_error_z,
    #     weight=0.5,          #10 a=0.05
    #     params={"asset_cfg": SceneEntityCfg("franka", body_names="endeffector"), "target_cfg": SceneEntityCfg("cuboid"),"scale" : 0.3},
    # )

    #  # Reward on distance between ee and cube on z-axis in proximity
    # end_effector_position_tracking_z_proximity = RewTerm(
    #     func=reward_custom.position_command_error_z,
    #     weight=0.5,          #10 a=0.05
    #     params={"asset_cfg": SceneEntityCfg("franka", body_names="endeffector"), "target_cfg": SceneEntityCfg("cuboid"),"scale" : 0.07},
    # )

    # Reward on distance between ee and cube
    end_effector_position_tracking = RewTerm(
        func=reward_custom.position_command_error,
        weight=2,           #3 a= 0.45
        params={"asset_cfg": SceneEntityCfg("franka", body_names="endeffector"), "target_cfg": SceneEntityCfg("cuboid"),"scale" : 0.7},
    )


    # Reward on distance between ee and cube when they are close
    end_effector_position_tracking_proximity = RewTerm(
        func=reward_custom.position_command_error,
        weight=2,          #10 a=0.05
        params={"asset_cfg": SceneEntityCfg("franka", body_names="endeffector"), "target_cfg": SceneEntityCfg("cuboid"),"scale" : 0.06},
    )

    # Reward on ee orientation relative to the cube
    end_effector_orientation_tracking = RewTerm(
        func=reward_custom.orientation_command_error_exp,
        weight=1,
        params={"asset_cfg": SceneEntityCfg("franka", body_names="endeffector"), "target_cfg": SceneEntityCfg("cuboid"),"scale" : 0.9},
    )

    # Reward on ee orientation relative to the cube, to minimize error
    end_effector_orientation_tracking_proximity = RewTerm(
        func=reward_custom.orientation_command_error_exp,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("franka", body_names="endeffector"), "target_cfg": SceneEntityCfg("cuboid"),"scale" : 0.06},
    )

    # ==========================================================
    # LIFTING AND REACHING REWARDS
    # ==========================================================
    
    # Reward for cube orientation with command
    cube_target_orientation = RewTerm(
        func=reward_custom.object_orientation_z_align_reward,
        weight=0.5,
        params={
            "command_name": "ee_pose_command","object_cfg":SceneEntityCfg("cuboid"),"robot_cfg":SceneEntityCfg("franka")
        },
    )

    # Negative reward if ee goes out of bounds
    out_of_bound_penalty = RewTerm(
        func=reward_custom.ee_out_of_bounds,
        weight=-5,
        params={
            "asset_cfg": SceneEntityCfg("franka", body_names=["endeffector"]),
            "body_names": ["endeffector"],
            "bounds": ((-1, -1, 0.0), (2, 1, 50)),
        },
    )

    # Penalty reward if cube's z-axis orientation is modified
    orientation_penalty = RewTerm(
        func=reward_custom.check_cube_z_facing_down,
        weight=-10,           
        params={"asset_cfg": SceneEntityCfg("cuboid"),
            "limit_angle_deg": 89.0,
            "minimal_height": 0.025,
            }
    )
     
    # Boolean reward if the cube is lifted from the ground
    cube_position_lift_tracking = RewTerm(
        func=reward_custom.object_is_lifted,
        weight=5,               #0.025
        params={"minimal_height": 0.025, "object_cfg": SceneEntityCfg("cuboid")},
    )

    # Reward to minimize distance between cube and target
    object_goal_tracking = RewTerm(
        func=mdp_lift.object_goal_distance,
        params={"std": 0.6, "minimal_height": 0.025, "command_name": "ee_pose_command","robot_cfg":SceneEntityCfg("franka", body_names="endeffector"),
                "object_cfg":SceneEntityCfg("cuboid")},
        weight=8,
    )

    # Reward to minimize distance between cube and target in proximity
    object_goal_tracking_proximity = RewTerm(
        func=mdp_lift.object_goal_distance,
        params={"std": 0.12, "minimal_height": 0.025, "command_name": "ee_pose_command","robot_cfg":SceneEntityCfg("franka", body_names="endeffector"),
                "object_cfg":SceneEntityCfg("cuboid")},
        weight=4,
    )
    
    # Reward that is 1 if one finger touches the cube, 2 if both touch it
    grasp = RewTerm(
        func=reward_custom.contact_count_reward,
        weight=1.0,
        params={
            "min_contact_threshold": 0.001,
            "sensor_names": ["contact_sensor_left", "contact_sensor_right"]
        },
    )

    # Reward that praises if the cube is grasped tightly so it does not slip
    grasp_force = RewTerm(
        func=reward_custom.grasping_force_reward,
        weight=1.0,
        params={
            "max_force_reward_cap": 3.0,
            "sensor_names": ["contact_sensor_left", "contact_sensor_right"]
        }
    )
    
    # Reward that praises if the cube reaches a specific distance from the target
    target_reached = RewTerm(
        func=reward_custom.target_reached,
        weight=15.0,
        params={"min_distance": 0.01, "command_name": "ee_pose_command","robot_cfg":SceneEntityCfg("franka", body_names="endeffector"),
                "object_cfg":SceneEntityCfg("cuboid")},
    )


    # # Negative reward if it does not lift the cube
    # cube_not_lifted_penality = RewTerm(
    #     func=reward_custom.termination_lift_timeout,
    #     weight= -10,
    #     params={
    #         "object_cfg": SceneEntityCfg("cuboid"), # Make sure the name is correct in your YAML
    #         "height_threshold": 0.025, #0.025----------------------------------------------------------------------------------
    #         "time_limit": 5.0, 
    #     },
    # )

# ================================================================================
#                               TERMINATIONS 
# ================================================================================

@configclass
class TerminationsCfg:  
    """Termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Termination if Franka ee goes out of the box
    ee_out_of_bounds_term = DoneTerm(
        func=reward_custom.ee_out_of_bounds,
        params={
            "asset_cfg": SceneEntityCfg("franka", body_names=["endeffector"]),
            "body_names": ["endeffector"],
            "bounds": ((-1, -1, 0.0), (2, 1, 50)),
        },
    )

    # Termination if cube z-axis is no longer normal to the floor
    wrong_orientation = DoneTerm(
        func=reward_custom.check_cube_z_facing_down,
        params={
            "asset_cfg": SceneEntityCfg("cuboid"),
            "limit_angle_deg": 89.0,  
            "minimal_height": 0.025,                   #0.025-----------------------------------------------
        },
    )
    
    # # Lifting checkpoint: If it is not high after 3s, reset.
    # lift_timeout_check = DoneTerm(
    #     func=reward_custom.termination_lift_timeout, # Use the function defined above
    #     params={
    #         "object_cfg": SceneEntityCfg("cuboid"), # Make sure the name is correct in your YAML
    #         "height_threshold": 0.025, #0.025----------------------------------------------------------------------------------
    #         "time_limit": 5.0, 
    #     },
    # )

# ================================================================================
#                               CURRICULUM 
# ================================================================================ 


@configclass
class CurriculumCfg:

    # ==========================================================
    # CURRICULUM COMMAND X-AXIS 
    # ==========================================================
    expand_command_x1 = CurrTerm(
        func=curriculum_custom.modify_command_range_gradual,
        params={
            "term_name": "ee_pose_command",
            "range_name": "pos_x",
            "interpolation": "linear",
            
            "initial_min": 0.87, 
            "initial_max": 0.87,
            
            "final_min": 0.87,   
            "final_max": 1.37,   #1.87
            
            "num_steps_start": num_steps_start, 
            "num_steps_end": num_steps_1
        }
    )

    expand_command_x2 = CurrTerm(
        func=curriculum_custom.modify_command_range_gradual,
        params={
            "term_name": "ee_pose_command",
            "range_name": "pos_x",
            "interpolation": "linear",
            
            "initial_min": 0.87, 
            "initial_max": 1.37,
            
            "final_min": 0.87,   
            "final_max": 1.87,   
            
            "num_steps_start": num_steps_4, 
            "num_steps_end": num_steps_5
        }
    )


    # ==========================================================
    # CURRICULUM COMMAND Y-AXIS
    # ==========================================================
    expand_command_y1 = CurrTerm(
        func=curriculum_custom.modify_command_range_gradual,
        params={
            "term_name": "ee_pose_command",
            "range_name": "pos_y",
            "interpolation": "linear",
            
            "initial_min": 0.0, 
            "initial_max": 0.0,
            
            "final_min": -0.25,   #-0.5
            "final_max": 0.25,      #0.5
            
            "num_steps_start": num_steps_start, 
            "num_steps_end": num_steps_1
        }
    )

    expand_command_y2 = CurrTerm(
        func=curriculum_custom.modify_command_range_gradual,
        params={
            "term_name": "ee_pose_command",
            "range_name": "pos_y",
            "interpolation": "linear",
            
            "initial_min": -0.25, 
            "initial_max": 0.25,
            
            "final_min": -0.5,   
            "final_max": 0.5,
            
            "num_steps_start": num_steps_4, 
            "num_steps_end": num_steps_5
        }
    )

    # ==========================================================
    # CURRICULUM FRANKA JOINTS 
    # ==========================================================
    # expand_franka_joint_noise = CurrTerm(
    #     func=curriculum_custom.modify_event_param_range,
    #     params={
    #         "term_name": "reset_franka_joints",  
    #         "param_name": "position_range",      
    #         "interpolation": "linear",

    #         "initial_min": 0.0, 
    #         "initial_max": 0.0,

    #         "final_min": -0.1,   
    #         "final_max": 0.1, 
            
    #         "num_steps_start": num_steps_start, 
    #         "num_steps_end": num_steps_3
    #     }
    # )

    # expand_franka_joint_noise_up = CurrTerm(
    #     func=curriculum_custom.modify_event_param_range,
    #     params={
    #         "term_name": "reset_franka_joints_up",  
    #         "param_name": "position_range",      
    #         "interpolation": "linear",

    #         "initial_min": 0.0, 
    #         "initial_max": 0.0,

    #         "final_min": 0.0,   
    #         "final_max": 0.25, 
            
    #         "num_steps_start": num_steps_start, 
    #         "num_steps_end": num_steps_3
    #     }
    # )

    # expand_franka_joint_noise_down = CurrTerm(
    #     func=curriculum_custom.modify_event_param_range,
    #     params={
    #         "term_name": "reset_franka_joints_down",  
    #         "param_name": "position_range",      
    #         "interpolation": "linear",

    #         "initial_min": 0.0, 
    #         "initial_max": 0.0,

    #         "final_min": -0.25,   
    #         "final_max": 0.0, 
            
    #         "num_steps_start": num_steps_start, 
    #         "num_steps_end": num_steps_3
    #     }
    # )

    # ==========================================================
    # CURRICULUM CUBE SPAWN X-AXIS 
    # ========================================================== 
    expand_cuboid_spawn_x1 = CurrTerm(
        func=curriculum_custom.modify_event_pose_range,
        params={
            "term_name": "reset_cuboid_position", 
            "axis_name": "x",                     
            "interpolation": "linear",

            "initial_min": 0.0, 
            "initial_max": 0.0,

            "final_min": 0.0,   
            "final_max": 0.5, #1
            
            "num_steps_start": num_steps_2, 
            "num_steps_end": num_steps_3
        }
    )

    expand_cuboid_spawn_x2 = CurrTerm(
        func=curriculum_custom.modify_event_pose_range,
        params={
            "term_name": "reset_cuboid_position", 
            "axis_name": "x",                     
            "interpolation": "linear",

            "initial_min": 0.0, 
            "initial_max": 0.5,

            "final_min": 0.0,   
            "final_max": 1.0, 
            
            "num_steps_start": num_steps_6, 
            "num_steps_end": num_steps_7
        }
    )
    # ==========================================================
    # CURRICULUM CUBE SPAWN Y-AXIS
    # ==========================================================
    expand_cuboid_spawn_y1 = CurrTerm(
        func=curriculum_custom.modify_event_pose_range,
        params={
            "term_name": "reset_cuboid_position",
            "axis_name": "y",                    
            "interpolation": "linear",

            "initial_min": 0.0, 
            "initial_max": 0.0,

            "final_min": -0.15,   #-0.25
            "final_max": 0.15,  #0.25
            
            "num_steps_start": num_steps_2, 
            "num_steps_end": num_steps_3
        }
    )

    expand_cuboid_spawn_y2 = CurrTerm(
        func=curriculum_custom.modify_event_pose_range,
        params={
            "term_name": "reset_cuboid_position",
            "axis_name": "y",                    
            "interpolation": "linear",

            "initial_min": -0.15, 
            "initial_max": 0.15,

            "final_min": -0.3,   
            "final_max": 0.3, 
            
            "num_steps_start": num_steps_6, 
            "num_steps_end": num_steps_7
        }
    )
    # ==========================================================
    # CURRICULUM REWARDS
    # ==========================================================
    # action_rate = CurrTerm(
    #         func=curriculum_custom.modify_reward_weight_gradual, params={"term_name": "action_rate", "interpolation" : "linear",
    #                                                 "initial_weight": 0.0, "final_weight" : 1.0,
    #                                                 "num_steps_start" : num_steps_end, "num_steps_end" : num_steps_end2}
    #     )
    
    # joint_vel_limits_penalty = CurrTerm(
    #         func=curriculum_custom.modify_reward_weight_gradual, params={"term_name": "joint_vel_limits_penalty", "interpolation" : "linear",
    #                                                 "initial_weight": 0.0, "final_weight" : -0.1,
    #                                                 "num_steps_start" : num_steps_start, "num_steps_end" : num_steps_start+1000}
    #     )



# ================================================================================
#                               MANAGER_BASED RL ENVIRONMENT CONFIGURATION
# ================================================================================ 

@configclass
class FrankaReachEnvCfg(ManagerBasedRLEnvCfg):
    """Final RL environment configuration for Franka."""

    scene: FrankaSceneCfg = FrankaSceneCfg(num_envs=64, env_spacing=3)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    #curriculum: CurriculumCfg = CurriculumCfg()


    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 8.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.viewer.eye = (3.0, 0.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)