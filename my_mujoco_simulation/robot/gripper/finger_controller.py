import mujoco
import numpy as np

class FingerController:
    def __init__(self, model, data, robot):
        self.robot = robot
        self.model = model
        self.data = data
        self.dt = self.model.opt.timestep
        self.grip_act_name = [name for name in robot.get_robot_dict()["actuators"] if "finger" in name][0]
        self.grip_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.grip_act_name)
        self.actuator_range = self.model.actuator_ctrlrange[self.grip_act_id]
        self.tend_id = self.model.actuator_trnid[self.grip_act_id][0]
        self.tend_adr = self.model.tendon_adr[self.tend_id]
        self.joint_ranges = self.model.jnt_range[self.model.wrap_objid[self.tend_adr]]
        self.joint_coeff = self.model.wrap_prm[self.tend_adr]
        
    def set_gripper_state(self, open_fraction=1.0):
        
        if not self.grip_act_id:
            raise ValueError("No gripper actuators found (expected names containing 'grip' or 'finger').")

        open_fraction = np.clip(open_fraction, self.actuator_range[0], self.actuator_range[1])
        self.data.ctrl[self.grip_act_id] = open_fraction


    def open_gripper(self):
        self.set_gripper_state(open_fraction=self.actuator_range[1])

    def close_gripper(self):
        self.set_gripper_state(open_fraction=self.actuator_range[0])

    def get_type(self):
        return "finger"

    def get_gripper_state(self):
        return self.data.ctrl[self.grip_act_id]

    def get_gripper_opening(self):
        return self.data.qpos[self.model.wrap_objid[self.tend_adr]]

    def _is_gripper_open(self):
        open_range = self.joint_ranges[1] * self.joint_coeff * 2
        joint_pos = self.data.qpos[self.model.wrap_objid[self.tend_adr]] #This is only one of the two joints, next consider both
        
        return abs(open_range - joint_pos) < 0.001
    
    def _is_gripper_close(self):
        open_range = self.joint_ranges[0] * self.joint_coeff * 2
        joint_pos = self.data.qpos[self.model.wrap_objid[self.tend_adr]]
        if abs(open_range - joint_pos) < 0.001:
            return True
        
        joint_vel = self.data.qvel[self.model.wrap_objid[self.tend_adr]]
        if joint_vel < 1e-3 and joint_pos < self.joint_ranges[1] * self.joint_coeff * 2:
            return True
        
        return False