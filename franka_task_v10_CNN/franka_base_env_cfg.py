"""Rest everything follows."""

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.envs.mdp as mdp
from .franka_cfg import FRANKA_PANDA_CFG
from isaaclab.sensors import ContactSensorCfg
from . import reward_custom
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns, MultiMeshRayCasterCameraCfg
from isaaclab.markers.config import VisualizationMarkersCfg
import isaaclab.utils.math as math_utils


def euler_to_quat(roll_deg, pitch_deg, yaw_deg):
    """
    Converts Euler angles (in degrees) to Quaternion (w, x, y, z).
    Rotation order: X (Roll) -> Y (Pitch) -> Z (Yaw).
    """
    # Convert to radians
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    # Trigonometric calculations
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # Formula for the quaternion
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (w, x, y, z)

# ================================================================================
#                               COMMAND
# ================================================================================ 

@configclass
class CommandsCfg:

    ee_pose_command = mdp.UniformPoseCommandCfg(
        asset_name="franka",
        body_name="endeffector",
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.87, 1.87),                 #(0.87, 1.87)
            pos_y=(-0.5, 0.5),                  #(-0.5, 0.5)
            pos_z=(0.2, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),  
            yaw=(math.pi, math.pi),
                    
        ),
    )
    

RAY_CASTER_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)

# ================================================================================
#                               SCENE DEFINITION
# ================================================================================ 

@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
    """Configuration for a Franka scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    franka: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    cuboid: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CuboidRigid",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,0.0,1.0),metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.87, 0.0, 0.02),rot=(1,0,0,0))
    )

    # Left Finger Sensor
    contact_sensor_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger", 
        update_period=0.0, 
        history_length=1, 
        debug_vis=False, 
        filter_prim_paths_expr=["{ENV_REGEX_NS}/CuboidRigid"],        
    )   

    # Right Finger Sensor
    contact_sensor_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger", 
        update_period=0.0, 
        history_length=1, 
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/CuboidRigid"], 
    )
    
    # Frame on franka ee
    # sdr = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/endeffector/Sdr",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
    #         scale=(0.1,0.1,0.1),
    #     ),
        
    # )

    # Frame on cube
    # hand_frame_visualizer = AssetBaseCfg(
        
    #     prim_path="{ENV_REGEX_NS}/CuboidRigid/CubeFrameVisualizer",
        
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
    #         scale=(0.1, 0.1, 0.1),  
    #     ),
    # )
    
    # MultiMeshRayCaster
    # camera = MultiMeshRayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/panda_link0/camera_frame",
    #     update_period=0,
    #     mesh_prim_paths=[
    #         "/World/defaultGroundPlane",
    #         MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/CuboidRigid"),
    #     ],
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.03, size=(0.6, 1.5), direction=(0, 0, -1)),
    #     debug_vis=True,
    #     visualizer_cfg=RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCaster"),
    # )

    camera = MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0/camera_frame",
        update_period=0,
        ray_alignment="base",
        max_distance=10.0,
        data_types=["distance_to_camera"],
        mesh_prim_paths=[
            "/World/defaultGroundPlane",
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/CuboidRigid"),
        ],
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 1.0), 
            # Calculate quaternion from Euler (Roll, Pitch, Yaw)
            # Roll = 0
            # Pitch = 45 degrees (tilted downward/forward)
            # Yaw = 90 degrees (rotation around vertical Z-axis)
            rot=euler_to_quat(160, 0, 90),
            convention="ros", 
        ),
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            height=40, #40
            width=40,  #40
            focal_length=1.0,
            horizontal_aperture=2 * math.tan(math.radians(56.62 / 2)), #46.62 / 2
            vertical_aperture=2 * math.tan(math.radians(50.73 / 2)),    #40.73 / 2
        ),
        debug_vis=True,
        attach_yaw_only=False,
        visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/RayCaster",
            markers={
                "hit": sim_utils.SphereCfg(
                    radius=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        ),
    )

    


import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg



# ================================================================================
#                               ACTIONS DEFINITION
# ================================================================================ 

@configclass
class ActionsCfg:

    # Action definition for the base, velocity control is executed
    base_vel = mdp.JointVelocityActionCfg(
        asset_name="franka",  
        joint_names=[  
            "dummy_base_prismatic_x_joint",
            "dummy_base_prismatic_y_joint",
            "dummy_base_revolute_z_joint",    
        ],
        scale=1.0,

        use_default_offset=True,
    )

    # Action definition for Franka, position control is executed
    joint_positions = mdp.JointPositionActionCfg(
        asset_name="franka",  
        joint_names=[    
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
            "panda_finger_joint.*",
        ],
        scale=1.0,

        use_default_offset=True,
    )

    
# ================================================================================
#                               OBSERVATIONS DEFINITION
# ================================================================================ 

def permuted_image_obs(env, sensor_cfg, data_type, normalize=True):
    """Gets the image and shifts the channels forward for the CNN."""
    # 1. Get the original image (shape: [N, H, W, C])
    img = mdp.image(env, sensor_cfg=sensor_cfg, data_type=data_type, normalize=normalize)
    
    # 2. Permute dimensions for PyTorch (shape: [N, C, H, W])
    # Example: [num_envs, 40, 40, 1] -> [num_envs, 1, 40, 40]
    return img.permute(0, 3, 1, 2)

@configclass
class ObservationsCfg:
    """Observations definition for the Franka environment."""

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for the agent (the policy)."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="franka",
                    joint_names=[
                        "dummy_base_prismatic_x_joint",
                        "dummy_base_prismatic_y_joint",
                        "dummy_base_revolute_z_joint",
                        "panda_joint1",
                        "panda_joint2",
                        "panda_joint3",
                        "panda_joint4",
                        "panda_joint5",
                        "panda_joint6",
                        "panda_joint7",
                        "panda_finger_joint.*",
                    ],
                )
            },
        )

        
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="franka",
                    joint_names=[
                        "dummy_base_prismatic_x_joint",
                        "dummy_base_prismatic_y_joint",
                        "dummy_base_revolute_z_joint",
                        "panda_joint1",
                        "panda_joint2",
                        "panda_joint3",
                        "panda_joint4",
                        "panda_joint5",
                        "panda_joint6",
                        "panda_joint7",
                        "panda_finger_joint.*",
                    ],
                )
            },
        )

        
        ee_pos = ObsTerm(
            func=mdp.body_pose_w,  
            params={
                "asset_cfg": SceneEntityCfg(
                    name="franka",
                    body_names=["endeffector"],  
                )
            },
        )

        pos_cube = ObsTerm(
                    func=mdp.root_pos_w, 
                    params={"asset_cfg": SceneEntityCfg(
                    name="cuboid")})
        orientation_cube = ObsTerm(
                    func=mdp.root_quat_w, 
                    params={"asset_cfg": SceneEntityCfg(
                    name="cuboid")})
        
        # position where to bring the cube
        # Pass the distance vector between command-cube, they are referenced in the body frame (env0)
        object_goal_error = ObsTerm(
            func=reward_custom.command_to_object_error_b,  
            params={
                "command_name": "ee_pose_command",
                "robot_cfg": SceneEntityCfg("franka"), 
                "object_cfg": SceneEntityCfg("cuboid"), 
            }
        )

        def __post_init__(self) -> None:
            """Post-initialization."""
            print(self.pos_cube)
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ProprioceptionCfg(ObsGroup):
        """Observations for the student (the policy)."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="franka",
                    joint_names=[
                        "dummy_base_prismatic_x_joint",
                        "dummy_base_prismatic_y_joint",
                        "dummy_base_revolute_z_joint",
                        "panda_joint1",
                        "panda_joint2",
                        "panda_joint3",
                        "panda_joint4",
                        "panda_joint5",
                        "panda_joint6",
                        "panda_joint7",
                        "panda_finger_joint.*",
                    ],
                )
            },
        )

        
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="franka",
                    joint_names=[
                        "dummy_base_prismatic_x_joint",
                        "dummy_base_prismatic_y_joint",
                        "dummy_base_revolute_z_joint",
                        "panda_joint1",
                        "panda_joint2",
                        "panda_joint3",
                        "panda_joint4",
                        "panda_joint5",
                        "panda_joint6",
                        "panda_joint7",
                        "panda_finger_joint.*",
                    ],
                )
            },
        )

        
        ee_pos = ObsTerm(
            func=reward_custom.ee_pose_b,  
            params={
                "asset_cfg": SceneEntityCfg(
                    name="franka",
                    body_names=["endeffector"],  
                )
            },
        )

        # position where to bring the cube
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose_command"})

        # position where to bring the cube
        # Pass the distance vector between command-cube, they are referenced in the body frame (env0)
        # object_goal_error = ObsTerm(
        #     func=reward_custom.command_to_object_error_b,  
        #     params={
        #         "command_name": "ee_pose_command",
        #         "robot_cfg": SceneEntityCfg("franka"), 
        #         "object_cfg": SceneEntityCfg("cuboid"), 
        #     }
        # )

        def __post_init__(self) -> None:
            """Post-initialization."""
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class VisionCfg(ObsGroup):

        camera_obs = ObsTerm(
            func=permuted_image_obs, 
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "distance_to_camera",
                "normalize": True,
            },
        )
        def __post_init__(self):
            self.concatenate_terms = True 

    # Define the two separate channels
    critic: CriticCfg = CriticCfg()
    policy_proprio: ProprioceptionCfg = ProprioceptionCfg()
    policy_vision: VisionCfg = VisionCfg()



# ================================================================================
#                               EVENTS DEFINITION
# ================================================================================ 

@configclass
class EventCfg:
    """Events configuration to reset Franka's pose."""
    # In the reset, Franka's joints are reset with the value defined in the configuration to which
    # a random value in the range defined in "position_range" is added

    # Reset of joints that can increase their angle, joint 4 can go from -90 to -62
    reset_franka_joints_up = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            
            "asset_cfg": SceneEntityCfg(
                name="franka",  
                joint_names=[
                    
                    "panda_joint4",
                 
                ],
            ),

            "position_range": (0.0, 0.15),      
            "velocity_range": (0.0, 0.0),
        },
    )

    # Reset of joints that can only decrease their angle, from 75 to 48
    reset_franka_joints_down = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            
            "asset_cfg": SceneEntityCfg(
                name="franka",  
                joint_names=[
                    "panda_joint2",
                ],
            ),

            "position_range": (-0.15, 0.0),        
            "velocity_range": (0.0, 0.0),
        },
    )

    # Joints that can both increase and decrease
    reset_franka_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            
            "asset_cfg": SceneEntityCfg(
                name="franka",  
                joint_names=[
                    "dummy_base_prismatic_x_joint",
                    "dummy_base_prismatic_y_joint",
                    "dummy_base_revolute_z_joint",
                    "panda_joint1",
                    "panda_joint3",
                    "panda_joint5",
                    "panda_joint6",
                    "panda_joint7",
                    "panda_finger_joint.*",
                ],
            ),


            "position_range": (-0.1, 0.1),           
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_cuboid_position = EventTerm(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg(name="cuboid"),

        "pose_range": {
            "x": (0, 1),           #(0,1)
            "y": (-0.3, 0.3),       #(-0.3, 0.3)    
            "z": (0, 0),            
            "roll": (0, 0),                 
            "pitch": (math.pi, math.pi),          
            "yaw": (math.pi, math.pi)    
        },

        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0)
        }
    }
)


# ================================================================================
#                               MANAGER_BASED ENVIRONMENT DEFINITION
# ================================================================================

@configclass
class FrankaEnvCfg(ManagerBasedEnvCfg):
    scene = FrankaSceneCfg(num_envs=16, env_spacing=2.5)
    observations = ObservationsCfg() 
    actions = ActionsCfg()    
    events = EventCfg()        
    commands: CommandsCfg = CommandsCfg()
    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz