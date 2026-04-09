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
from isaaclab.utils.math import combine_frame_transforms,quat_error_magnitude, quat_mul
from isaaclab.sensors import ContactSensor


def exponential_interpolation(initial_value, final_value, start_time, end_time, current_time):
    interpolated_value = initial_value + (final_value - initial_value) * math.exp(5*(current_time - end_time)/(end_time - start_time))
    return interpolated_value

def linear_interpolation(initial_value, final_value, start_time, end_time, current_time):
    interpolated_value = initial_value + (final_value - initial_value) * (current_time - start_time) / (end_time - start_time)
    return interpolated_value

class modify_command_range_gradual(ManagerTermBase):
    """
    Curriculum that modifies a range (min, max) of a command based on steps.
    Uses get_term() to access the CommandManager.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.command_name = cfg.params["term_name"]
        self.range_name = cfg.params["range_name"]
        self._cmd_term = env.command_manager.get_term(self.command_name)
        self._ranges_cfg = self._cmd_term.cfg.ranges

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        range_name: str,
        interpolation: str,
        initial_min: float, initial_max: float,
        final_min: float, final_max: float,
        num_steps_start: int, num_steps_end: int,
    ):
        if interpolation == "linear":
            interpl_fn = linear_interpolation
        elif interpolation == "exponential":
            interpl_fn = exponential_interpolation
        else:
            raise NotImplementedError(f"Unknown interpolation type: '{interpolation}'.")


        current_step = env.common_step_counter

        start_step_abs = num_steps_start*num_steps_per_env_glob
        end_step_abs = num_steps_end*num_steps_per_env_glob

        new_min = initial_min
        new_max = initial_max
        updated = False
   
        if current_step < start_step_abs:
            new_min = initial_min
            new_max = initial_max
            updated = True 

 
        elif start_step_abs <= current_step < end_step_abs:
            new_min = interpl_fn(initial_min, final_min, start_step_abs, end_step_abs, current_step)
            new_max = interpl_fn(initial_max, final_max, start_step_abs, end_step_abs, current_step)
            updated = True


        elif current_step >= end_step_abs:
            new_min = final_min
            new_max = final_max
            # Check to avoid unnecessary writes
            current_vals = getattr(self._ranges_cfg, range_name)
            if current_vals != (final_min, final_max):
                updated = True

        # 4. Application (Write to memory)
        if updated:
            # Since tuples are immutable, we create a new one
            new_range_tuple = (new_min, new_max)
            # Write directly to the live object's configuration
            setattr(self._ranges_cfg, range_name, new_range_tuple)
            
        return new_max

# Class to modify the initial joint configuration of Franka during reset
class modify_event_param_range(ManagerTermBase):
    """
    Modifies a top-level parameter of an Event (e.g., 'position_range').
    Uses the official get_term_cfg API of the EventManager.
    """
    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.term_name = cfg.params["term_name"]
        self.param_name = cfg.params["param_name"]
        

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        param_name: str,
        interpolation: str,
        initial_min: float, initial_max: float,
        final_min: float, final_max: float,
        num_steps_start: int, num_steps_end: int,
    ):
        # 1. Retrieve the OFFICIAL configuration
        # This method searches across all modes (reset, startup, etc.)
        term_cfg = env.event_manager.get_term_cfg(term_name)

        # 2. Select interpolation
        if interpolation == "linear":
            interpl_fn = linear_interpolation
        elif interpolation == "exponential":
            interpl_fn = exponential_interpolation
        else:
            raise NotImplementedError

        # 3. Step Management
        current_step = env.common_step_counter
        # Assuming the values are already multiplied (e.g., 600*256) in the Config
        start_step = num_steps_start*num_steps_per_env_glob 
        end_step = num_steps_end*num_steps_per_env_glob

        new_min, new_max = initial_min, initial_max
        
        # 4. Calculation
        if current_step < start_step:
            new_min, new_max = initial_min, initial_max
        elif start_step <= current_step < end_step:
            new_min = interpl_fn(initial_min, final_min, start_step, end_step, current_step)
            new_max = interpl_fn(initial_max, final_max, start_step, end_step, current_step)
        else:
            new_min, new_max = final_min, final_max

        # 5. DIRECT modification of the params dictionary
        # Since term_cfg is an object (passed by reference), 
        # by modifying .params[...] we are modifying the original that will be used at the next reset.
        current_val = term_cfg.params[param_name]
        
        if current_val != (new_min, new_max):
            term_cfg.params[param_name] = (new_min, new_max)
            
        return new_max
    

class modify_event_pose_range(ManagerTermBase):

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.term_name = cfg.params["term_name"]
        self.axis_name = cfg.params["axis_name"]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        axis_name: str,
        interpolation: str,
        initial_min: float, initial_max: float,
        final_min: float, final_max: float,
        num_steps_start: int, num_steps_end: int,
    ):
        # 1. Retrieve Configuration
        term_cfg = env.event_manager.get_term_cfg(term_name)

        # 2. Interpolation
        if interpolation == "linear":
            interpl_fn = linear_interpolation
        elif interpolation == "exponential":
            interpl_fn = exponential_interpolation
        else:
            raise NotImplementedError

        # 3. Step Management
        current_step = env.common_step_counter
        start_step = num_steps_start *num_steps_per_env_glob
        end_step = num_steps_end*num_steps_per_env_glob

        new_min, new_max = initial_min, initial_max

        # 4. Calculation
        if current_step < start_step:
            new_min, new_max = initial_min, initial_max
        elif start_step <= current_step < end_step:
            new_min = interpl_fn(initial_min, final_min, start_step, end_step, current_step)
            new_max = interpl_fn(initial_max, final_max, start_step, end_step, current_step)
        else:
            new_min, new_max = final_min, final_max

        # 5. Nested Modification
        # Access the pose_range dictionary inside params
        pose_range_dict = term_cfg.params["pose_range"]
        current_val = pose_range_dict[axis_name]
        
        if current_val != (new_min, new_max):
            # Assign the new tuple
            pose_range_dict[axis_name] = (new_min, new_max)
            
        return new_max
    
class modify_reward_weight_gradual(ManagerTermBase):
    """Curriculum that modifies the reward weight based on a step-wise schedule."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # obtain term configuration
        term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(term_name)

    def __call__(
        self,
        env: ManagerBasedRLEnv, 
        env_ids: Sequence[int], 
        term_name: str,
        interpolation: str, 
        initial_weight: float, final_weight: float,
        num_steps_start: int, num_steps_end: int
    ) -> float:
        if interpolation == "linear":
            interpl_fn = linear_interpolation
        elif interpolation == "exponential":
            interpl_fn = exponential_interpolation
        else:
            raise NotImplementedError(f"Unknown interpolation type: '{interpolation}'.")
        # update term settings
        if num_steps_start*num_steps_per_env_glob <= env.common_step_counter:
            if env.common_step_counter < num_steps_end*num_steps_per_env_glob:
                self._term_cfg.weight = interpl_fn(initial_weight, final_weight, num_steps_start*num_steps_per_env_glob, 
                                                   num_steps_end*num_steps_per_env_glob, env.common_step_counter)
            elif env.common_step_counter >= num_steps_end*num_steps_per_env_glob:
                self._term_cfg.weight = final_weight
            env.reward_manager.set_term_cfg(term_name, self._term_cfg)

        return self._term_cfg.weight