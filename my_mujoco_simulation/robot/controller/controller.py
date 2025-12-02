import mujoco
import numpy as np
from my_mujoco_simulation.robot.gripper.finger_controller import FingerController

class Controller:
    def __init__(self, model, data, robot):
        self.robot = robot
        self.model = model
        self.data = data
        self.dt = self.model.opt.timestep
        self.gripper_control()

    def gripper_control(self):
        ids = self.robot.get_robot_dict()["actuatorsindex"]

        for k in ids.keys():
            if "finger" in k:
                self.grip_controller = FingerController(self.model, self.data, self.robot)
    # Core Control Methods
    def set_joint_position(self, target_positions):
        pass
    
    def set_joint_velocity(self, target_velocities):
        ids = self.robot.get_robot_dict()["actuatorsindex"]
        act_names = [k for k in ids.keys() if "arm" in k]
        arm_ids = [ids[k] for k in act_names]

        if len(target_velocities) != len(arm_ids):
            raise ValueError(
                f"Expected {len(arm_ids)} target velocities for 'arm' actuators, "
                f"but got {len(target_velocities)}."
            )
        for i, aid in enumerate(arm_ids):
            if self._is_velocity_actuator(aid):
                self.data.ctrl[aid] = target_velocities[i]

            elif self._is_position_actuator(aid):
                jnt_id = self.model.actuator_trnid[aid, 0]
                qpos = self.data.qpos[jnt_id]
                qpos_des = np.clip(
                    qpos + target_velocities[i] * self.dt,
                    self.model.jnt_range[jnt_id, 0],
                    self.model.jnt_range[jnt_id, 1],
                )
                self.data.ctrl[aid] = qpos_des

            elif self._is_general_actuator(aid):
                jnt_id = self.model.actuator_trnid[aid, 0]
                qpos_des = np.clip(
                    self.data.ctrl[aid] + target_velocities[i] * self.dt,
                    self.model.jnt_range[jnt_id, 0],
                    self.model.jnt_range[jnt_id, 1],
                )
                self.data.ctrl[aid] = qpos_des
    
    def set_joint_torque(self, torques):
        pass

    def set_gripper_state(self, open_fraction=1.0):
        self.grip_controller.set_gripper_state(open_fraction)
        ids = self.robot.get_robot_dict()["actuatorsindex"]
        act_names = [k for k in ids.keys() if "arm" in k]
        arm_ids = [ids[k] for k in act_names]
        self.set_joint_velocity(np.zeros(len(arm_ids)))

    def open_gripper(self):
        self.grip_controller.open_gripper()

    def close_gripper(self):
        self.grip_controller.close_gripper()

    def _is_gripper_open(self):
        return self.grip_controller._is_gripper_open()

    def _is_gripper_close(self):
        return self.grip_controller._is_gripper_close()
    
    def get_gripper_opening(self):
        return self.grip_controller.get_gripper_opening()

    # Utility
    def zero_all(self):
        """Zero all control inputs."""
        self.data.ctrl[:] = 0.0

    # Actuator & joint access
    def _is_position_actuator(self, act_id):
        return self.robot.get_robot_dict()["actuatorstype"][act_id] == "position"

    def _is_velocity_actuator(self, act_id):
        return self.robot.get_robot_dict()["actuatorstype"][act_id] == "velocity"
    
    def _is_general_actuator(self, act_id):
        return self.robot.get_robot_dict()["actuatorstype"][act_id] == "general"

    def _joint_qpos_from_actuator(self, act_id):
        """Find the joint position corresponding to a given actuator."""
        joint_id = self.model.actuator_trnid[act_id, 0]
        qpos_addr = self.model.jnt_qposadr[joint_id]
        return self.data.qpos[qpos_addr]
