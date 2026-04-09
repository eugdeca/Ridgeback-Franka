import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.assets import RigidObject
from typing import Callable, Tuple, Dict, Any
from collections.abc import Sequence
import isaaclab.envs.mdp as mdp
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase
from .agents.rsl_rl_ppo_cfg import num_steps_per_env_glob
from isaaclab.utils.math import quat_apply, math
from isaaclab.utils.math import combine_frame_transforms,quat_error_magnitude, quat_mul, quat_rotate, quat_rotate_inverse, subtract_frame_transforms
from isaaclab.sensors import ContactSensor




def joint_vel_limits_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("franka"),
    a: float = 30.0,
) -> torch.Tensor:
    """Smooth penalty for joint positions approaching/exceeding the soft limits.
    
    Uses exponential terms: exp(a*(x - upper)) + exp(a*(lower - x)).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_vel[:, asset_cfg.joint_ids]
    upper = asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids]

    # exponential penalties for being outside the limits
    penalty = 1/(0.001 + torch.exp(a * (q + upper))) + 1/(0.001 + torch.exp(a * (upper - q)))
    return penalty.sum(dim=1)

def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    std: float = 29,
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    enable = torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)
    return enable * 1/(0.5 + torch.exp(-object.data.root_pos_w[:, 2]*std))

def target_reached(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_distance: float = 0.01,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    
    return distance < min_distance


def position_command_error_cube(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg,
    scale: float = 0.4
) -> torch.Tensor:
    """
    Calculates the position error (Euclidean distance) between the Cube and the Target Command.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Current Cube Position
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]

    # Desired Position from Command 
    command = env.command_manager.get_command(command_name)
    des_pos_w = command[:, 0:3]

    # Distance Calculation
    dist = torch.abs(curr_pos_w[:, 2] - des_pos_w[:, 2]) 

    reward = torch.exp(-dist / scale)

    return reward 

def get_local_body_pos(env, asset_cfg: SceneEntityCfg, body_names) -> torch.Tensor:
    """
    Function to obtain the position of a body in the
    LOCAL frame of the environment.
    """

    articulation = env.scene[asset_cfg.name]
    body_idx = [articulation.body_names.index(b) for b in body_names]
    body_pos_world = articulation.data.body_pos_w[:, body_idx]
    env_origins = env.scene.env_origins

    body_pos_local = body_pos_world - env_origins.unsqueeze(1)
    
    return body_pos_local.squeeze(1)

def hand_to_target_exp_reward(env, asset_cfg: SceneEntityCfg, body_names, target, scale: float = 0.4) -> torch.Tensor:
    """
    Reward: calculates an exponential reward (y = e^(-distance/scale))
    based on the distance between the hand and a target in the LOCAL space.
    """
    
    hand_pos_local = get_local_body_pos(env, asset_cfg, body_names)
    target_pos_local = torch.as_tensor(target, device=hand_pos_local.device)
    distance = torch.norm(hand_pos_local - target_pos_local, p=2, dim=-1)
    reward = torch.exp(-distance / scale)
    
    return reward

def ee_out_of_bounds(env, asset_cfg: SceneEntityCfg, body_names, bounds) -> torch.Tensor:
    """
    Termination: checks if the end-effector goes out of bounds in the
    LOCAL space of the environment.
    """
    body_pos_local = get_local_body_pos(env, asset_cfg, body_names)
    lower = torch.as_tensor(bounds[0], device=body_pos_local.device)
    upper = torch.as_tensor(bounds[1], device=body_pos_local.device)
    
    # Check if ee is out of bounds
    is_out = ((body_pos_local < lower) | (body_pos_local > upper)).any(dim=-1)
    
    return is_out

def joint_vel_l1_custom(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation using an L1-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    sum_vel = torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    
    return torch.exp(-sum_vel / 0.1)

def position_command_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,target_cfg: SceneEntityCfg,scale: float = 0.4) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    # obtain the desired and current positions
    des_pos_w = target.data.root_state_w[:, 0:3]
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    dist = torch.norm(curr_pos_w - des_pos_w, dim=1)
    reward = torch.exp(-dist / scale)
    return reward

def action_rate_l2_exp(env: ManagerBasedRLEnv,scale: float = 0.4) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    action_rate = torch.sum(torch.square(env.action_manager.action[:,:10] - env.action_manager.prev_action[:,:10]), dim=1)
    reward = 1/(0.01 + torch.exp(-action_rate * scale)) - 1
    return reward

def orientation_command_error_exp(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,target_cfg: SceneEntityCfg,scale: float = 0.4) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    # obtain the desired and current orientations
    des_quat_w = target.data.root_state_w[:, 3:7]
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    quat_error =  quat_error_magnitude(curr_quat_w, des_quat_w)
    reward = torch.exp(-quat_error / scale)
    return reward

def joint_vel_exp(
    env: ManagerBasedRLEnv,
    joint_names_to_check: list[str],  
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 0.4) -> torch.Tensor:

    """
    Penalize joint velocities ONLY for the specified joints
    if they exceed their MAXIMUM LIMITS (hard limits).
    """
    asset: Articulation = env.scene[asset_cfg.name]

    try:
        joint_indices = asset.find_joints(joint_names_to_check)[0]
    except IndexError:
        print(f"Error: Joint names {joint_names_to_check} not found.")
        return torch.zeros(env.num_envs, device=env.device)
    
    current_vels = asset.data.joint_vel[:, joint_indices]

    current_vels_abs = torch.sum(torch.abs(current_vels), dim=1)
    
    reward = torch.exp(-current_vels_abs / scale)
    return reward

def position_finger_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,scale: float = 0.4) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # obtain the desired and current positions
    curr_pos_w_l = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    curr_pos_w_r = asset.data.body_pos_w[:, asset_cfg.body_ids[1]]  # type: ignore
    dist = torch.norm(curr_pos_w_r - curr_pos_w_l, dim=1)
    reward = torch.exp(-dist / scale)
    return reward

def action_finger_exp(env: ManagerBasedRLEnv,scale: float = 0.4) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    action = torch.sum(torch.abs(env.action_manager.action[:,11:12]), dim=1)
    reward = torch.exp(-action / scale)
    return reward 

def check_cube_z_facing_down(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, 
                             limit_angle_deg: float = 45.0, minimal_height:float=0.04) -> torch.Tensor:
    """
    Returns 1.0 (True) if the cube's Z-axis does NOT point downwards
    within the specified tolerance cone.
    """

    asset = env.scene[asset_cfg.name]
    cube_quat = asset.data.root_quat_w
    not_lifted = asset.data.root_pos_w[:, 2] < minimal_height

    # Create the local Z unit vector (0, 0, 1)
    z_unit_vec = torch.zeros(env.num_envs, 3, device=env.device)
    z_unit_vec[:, 2] = 1.0
    
    cube_z_axis_world = quat_apply(cube_quat, z_unit_vec)
    
    # Calculate alignment with 'Down' (0, 0, -1)
    alignment = -cube_z_axis_world[:, 2]
    
    angle_tensor = torch.tensor(limit_angle_deg, device=env.device)
    
    angle_rad = torch.deg2rad(angle_tensor)
    
    threshold = torch.cos(angle_rad)
    
    # Check for violation
    violation = alignment < threshold
    
    return violation*not_lifted 

def position_finger_grasp(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,scale: float = 0.4) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 1. Get the current steps
    current_steps = env.episode_length_buf  # Tensor [num_envs]

    # 2. Get the time in seconds
    # Multiply the steps by the control delta time (step_dt)
    current_time_seconds = current_steps * env.step_dt

    # 3. Vectorized Logic (Replaces the IF statement)
    # torch.where(condition, value_if_true, value_if_false)
    # If time >= 6, it sets 1.0, otherwise -1.0
    enable = torch.where(current_time_seconds >= 0.0, 1.0, 0.0)
    
    # obtain the desired and current positions
    curr_pos_w_l = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    curr_pos_w_r = asset.data.body_pos_w[:, asset_cfg.body_ids[1]]  # type: ignore
    dist = torch.norm(curr_pos_w_r - curr_pos_w_l, dim=1)
    reward = torch.exp(-dist / scale)
    return reward*enable

def contact_count_reward(env: ManagerBasedRLEnv, min_contact_threshold: float, sensor_names: list[str]) -> torch.Tensor:
    """Count how many fingers are validly touching the object."""
    total_contact_count = 0.0

    # Iterate over each sensor (left and right)
    for name in sensor_names:
        sensor: ContactSensor = env.scene.sensors[name]
        
        # Shape: (NumEnvs, History, NumContacts, 3)
        force_matrix = sensor.data.force_matrix_w_history
        
        # Calculate the force norm for each contact and each history frame
        norm_forces = torch.norm(force_matrix, dim=-1)  # (NumEnvs, History, NumContacts)
        
        # Take the maximum over all contacts and the entire history
        max_force = torch.max(norm_forces.view(norm_forces.shape[0], -1), dim=1)[0]
        
        # Threshold check
        is_touching = max_force > min_contact_threshold
        total_contact_count += is_touching.float()
    return total_contact_count

def grasping_force_reward(env: ManagerBasedRLEnv, max_force_reward_cap: float, sensor_names: list[str]) -> torch.Tensor:
    """Sum the grasping force limited by the cap for all fingers."""
    total_force_reward = 0.0

    for name in sensor_names:
        sensor: ContactSensor = env.scene.sensors[name]
        
        force_matrix = sensor.data.force_matrix_w_history
        norm_forces = torch.norm(force_matrix, dim=-1)  
        
        # Maximum magnitude in the history and across contacts
        max_force = torch.max(norm_forces.view(norm_forces.shape[0], -1), dim=1)[0]
        
        # Apply the cap (maximum reward limit)
        reward_magnitude = torch.clamp(max_force, max=max_force_reward_cap)
        
        total_force_reward += reward_magnitude
    return total_force_reward

def object_orientation_z_align_reward(
    
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    scale: float = 1.8,
    distance_threshold: float = 0.1,  # <--- NEW PARAMETER (10 cm)
) -> torch.Tensor:
    """
    Reward based on Z-axis alignment, ONLY if object is within 'distance_threshold' from target.
    """
    # 1. Retrieve entities
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # 2. Retrieve the command
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_quat_b = command[:, 3:7]

    # 3. Transform the command into the World Frame
    # MODIFICATION: Now we also retrieve 'des_pos_w' to calculate the distance
    des_pos_w, des_quat_w = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, 
        des_pos_b, des_quat_b
    )

    # --- ORIENTATION CALCULATION (Original Code) ---
    vec_z = torch.zeros_like(des_pos_b)
    vec_z[:, 2] = 1.0 

    target_z_dir = quat_rotate(des_quat_w, vec_z)
    object_z_dir = quat_rotate(object.data.root_quat_w, vec_z)
    
    alignment = torch.sum(target_z_dir * object_z_dir, dim=1)
    alignment_error = 1.0 - alignment
    
    # Calculate the base reward for the orientation
    rot_reward = torch.exp(-alignment_error / scale)

    # --- NEW LOGIC: DISTANCE GATING ---
    
    # Calculate Euclidean distance between object and target (in World Frame)
    # dim=-1 computes the norm along x,y,z
    distance = torch.norm(object.data.root_pos_w - des_pos_w, dim=-1)

    # Conditional mask:
    # If distance < threshold -> return rot_reward
    # If distance >= threshold -> return 0.0
    final_reward = torch.where(
        distance < distance_threshold, 
        rot_reward, 
        torch.zeros_like(rot_reward)
    )

    return final_reward

def position_command_error_xy(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    scale: float = 0.4
) -> torch.Tensor:
    """
    Reward that penalizes the position error ONLY on the XY plane.
    Completely ignores the Z axis.
    """
    # Extract entities
    asset: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    
    # Get desired and current positions
    des_pos_w = target.data.root_state_w[:, 0:3]
    # Note: asset_cfg.body_ids[0] is usually the end-effector
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    
    # --- DISTANCE CALCULATION ONLY XY ---
    # Take only columns 0 and 1 (x and y)
    des_pos_xy = des_pos_w[:, 0:2]
    curr_pos_xy = curr_pos_w[:, 0:2]
    
    # Calculate L2 norm (2D Euclidean distance)
    dist_xy = torch.norm(curr_pos_xy - des_pos_xy, dim=1)
    
    # Calculate reward
    reward = torch.exp(-dist_xy / scale)
    return reward

def position_command_error_z(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    scale: float = 0.4,
    xy_threshold: float = 0.01 # 1 centimeter
) -> torch.Tensor:
    """
    Reward for the error along the Z axis.
    Activates the reward ONLY if the distance on the XY plane is less than 'xy_threshold'.
    """
    # Extract entities
    asset: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    
    des_pos_w = target.data.root_state_w[:, 0:3]
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]

    # 1. First, calculate the XY distance to verify the condition
    dist_xy = torch.norm(curr_pos_w[:, 0:2] - des_pos_w[:, 0:2], dim=1)
    
    # 2. Calculate the Z distance (absolute value of the difference)
    # shape becomes (num_envs,)
    dist_z = torch.abs(curr_pos_w[:, 2] - des_pos_w[:, 2])
    
    # 3. Calculate the potential Z reward
    reward_z = torch.exp(-dist_z / scale)
    
    # 4. Apply the filter (Gating)
    # If dist_xy < 0.01 -> use reward_z
    # Otherwise -> 0.0
    final_reward = torch.where(
        dist_xy < xy_threshold,
        reward_z,
        torch.zeros_like(reward_z)
    )
    
    return final_reward

def termination_lift_timeout(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    height_threshold: float = 0.05,
    time_limit: float = 3.5,
) -> torch.Tensor:
    """
    Terminates the episode if the object has not been lifted above 'height_threshold'
    within 'time_limit' seconds.
    """
    # 1. Retrieve the object (the cube)
    object: RigidObject = env.scene[object_cfg.name]
    
    # 2. Calculate current time in seconds
    # env.episode_length_buf counts the steps, step_dt is the step duration (e.g., 0.02s)
    current_time_s = env.episode_length_buf * env.step_dt
    
    # 3. Check if the time limit has been exceeded
    # Returns True if we are past 3 seconds
    is_time_expired = current_time_s > time_limit
    
    # 4. Check object height (World Frame)
    # object.data.root_pos_w[:, 2] is the Z coordinate
    object_z = object.data.root_pos_w[:, 2]
    
    # Returns True if the object is "low" (not lifted)
    is_below_threshold = object_z < height_threshold
    
    # 5. Reset Logic:
    # Reset ONLY IF (Time Expired) AND (Object is Low)
    reset_condition = torch.logical_and(is_time_expired, is_below_threshold)
    
    return reset_condition


def command_to_object_error_b(
    env,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Returns the error vector (Command - Object) in the robot's Base Frame.
    Returns a tensor (num_envs, 3).
    """
    # 1. Retrieve entities
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    
    # 2. Command: Already in the Base Frame (e.g., [0.5, 0, 0])
    # Shape: (num_envs, 3)
    # Note: env.command_manager.get_command often returns (num_envs, 7) or (num_envs, 3)
    # Assume it takes the first 3 for position.
    cmd_pos_b = env.command_manager.get_command(command_name)[:, :3]

    # 3. Object: It is in the World Frame -> Move it to the Base Frame
    # Vector from robot to object in the world
    vec_robot_to_obj_w = obj.data.root_pos_w - robot.data.root_pos_w
    
    # Rotate the vector using the inverse of the robot's orientation
    # This way, if the robot is rotated, the "front" vector remains "front" for it
    obj_pos_b = quat_rotate_inverse(robot.data.root_quat_w, vec_robot_to_obj_w)

    # 4. Vector Error Calculation (Command - Object)
    # If the value is [0.1, 0, 0], it means "The command is 10cm ahead of the object"
    error_vector_b = cmd_pos_b - obj_pos_b
    
    return error_vector_b

def ee_pose_b(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("franka")):
    """
    Returns the End-Effector position and quaternion in the Robot's Base frame.
    Output: concatenation of position (3) and quaternion (4) -> (Num_Envs, 7)
    """
    # 1. Retrieve the robot asset
    robot = env.scene[asset_cfg.name]
    
    # 2. Get the base (Root) pose in World Frame
    # Shape: (N, 3) and (N, 4)
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    
    # 3. Get the End-Effector (Body) pose in World Frame
    # We retrieve the body index from the cfg
    body_idx = robot.find_bodies(asset_cfg.body_names)[0]
    ee_pos_w = robot.data.body_pos_w[:, body_idx[0], :]
    ee_quat_w = robot.data.body_quat_w[:, body_idx[0], :]
    
    # 4. Calculate the relative transformation: T_base_ee = inv(T_world_base) * T_world_ee
    # IsaacLab has a specific utility to subtract frames
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
    )
    
    # Returns pos (3) + quat (4) = 7
    return torch.cat([ee_pos_b, ee_quat_b], dim=-1)