"""Configuration for the Franka Panda robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import os

# ================================================================================
#                               FRANKA CONFIGURATION
# ================================================================================
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_FRANKA_USD_PATH = os.path.join(_CURRENT_DIR, "Collected_ridgeback_franka", "franka_flat.usd")

FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_FRANKA_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "dummy_base_prismatic_x_joint": 0.0,
            "dummy_base_prismatic_y_joint": 0.0,
            "dummy_base_revolute_z_joint": 0.0,
            "panda_joint1": 0.0,
            "panda_joint2": 1.309,  
            "panda_joint3": 0.0,
            "panda_joint4": -1.5708,
            "panda_joint5": 0.0,
            "panda_joint6": 2.9,
            "panda_joint7": 0.785398,
            "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_base_z": ImplicitActuatorCfg(
            joint_names_expr=["dummy_base_revolute_z_joint"],
            effort_limit_sim=84.0,
            velocity_limit_sim=1.0,
            stiffness=0.0,
            damping=100.0,
        ),
        "panda_base_y": ImplicitActuatorCfg(
            joint_names_expr=["dummy_base_prismatic_y_joint"],
            effort_limit_sim=84.0,
            velocity_limit_sim=0.5,
            stiffness=0.0,
            damping=300.0, 
        ),
        "panda_base_x": ImplicitActuatorCfg(
            joint_names_expr=["dummy_base_prismatic_x_joint"],
            effort_limit_sim=84.0,
            velocity_limit_sim=0.5,
            stiffness=0.0,
            damping=300.0,
        ),
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=12.0,
            velocity_limit_sim=0.05,
            stiffness=80,
            damping=4,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

