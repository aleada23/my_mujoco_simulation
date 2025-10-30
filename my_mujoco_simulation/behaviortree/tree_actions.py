# tree_actions.py
import mujoco
from py_trees.behaviour import Behaviour
import py_trees
import numpy as np
import time

import MuJoCo_simulation.robots.tiago_actions_utils as tiago_actions_utils
import MuJoCo_simulation.robots.tiago_utils as tiago_utils


class MoveToHome(py_trees.behaviour.Behaviour):
    def __init__(self, name, model, data, robot, move_duration=20.0, kp=0.1, tolerance=1e-6):
        super().__init__(name)
        self.model = model
        self.data = data
        self.robot = robot
        self.move_duration = move_duration
        self.kp = kp
        self.tolerance = tolerance

        self.arm_joints = np.array(robot["joints"][:7])
        self.home_config = np.array([0.20, -1.34, -0.20, 1.94, -1.57, 1.37, 0.0])
        self.start_time = None
        self.start_qpos = np.zeros(7)

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [MoveToHome::setup()]")

    def initialise(self):
        self.logger.debug(f"{self.name} [MoveToHome::initialise()]")
        self.start_time = self.data.time
        self.start_qpos = np.array([self.data.qpos[self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]] for j in self.arm_joints])
        self.actuators_name = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu) if "arm" in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)]

    def update(self):
        current_qpos = np.array([self.data.qpos[self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]] for j in self.arm_joints])
        for i, act_name in enumerate(self.actuators_name):
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            error = self.home_config[i] - current_qpos[i]
            self.data.ctrl[actuator_id] = self.kp * error

        if np.allclose(current_qpos, self.home_config, atol=self.tolerance):
            self.logger.debug(f"{self.name}: Reached home position.")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.debug(f"{self.name} [MoveToHome::terminate()] -> {new_status}")


class ApproachTable(py_trees.behaviour.Behaviour):
    def __init__(self, name, model, data, robot, target_pos, table_normal=np.array([0, 0, 1]), kp=1.0, tolerance=1e-2):
        super().__init__(name)
        self.model = model
        self.data = data
        self.robot = robot
        self.target_pos = np.array(target_pos)
        self.table_normal = np.array(table_normal)
        self.kp = kp
        self.tolerance = tolerance
        self.start_time = None

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [ApproachTable::setup()]")

    def initialise(self):
        self.start_time = self.data.time
        self.logger.debug(f"{self.name} [ApproachTable::initialise()]")

    def update(self):
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot0_ee_pose")
        ee_pos, ee_quat = tiago_utils.tiago_forward_kinematic("base", "robot0_ee_pose", self.model, self.data)
        J_full = tiago_actions_utils.jacobian_in_base_frame(self.model, self.data, ee_id, base_body="robot0_base_link")

        position_error = self.target_pos - ee_pos

        # Simple fixed orientation
        target_quat = np.array([1.0, 0, 0, 0])
        orn_err = tiago_actions_utils.quat_error(ee_quat, target_quat)

        error = np.hstack([position_error, orn_err])
        dq = np.linalg.pinv(J_full[:, 19:26]) @ (self.kp * error)
        self.data.ctrl[:7] = dq

        if np.allclose(error[:], 0, atol=self.tolerance):
            self.logger.debug(f"{self.name}: Reached table pose.")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.debug(f"{self.name} [ApproachTable::terminate()] -> {new_status}")


class MoveDownUntilContact(py_trees.behaviour.Behaviour):
    def __init__(self, name, model, data, robot, down_joint="robot0_torso_lift_joint", step_size=0.0001, torque_sensor_name="robot0_wrist_torque", torque_threshold=2.0):
        super().__init__(name)
        self.model = model
        self.data = data
        self.robot = robot
        self.down_joint = down_joint
        self.step_size = step_size
        self.torque_sensor_name = torque_sensor_name
        self.torque_threshold = torque_threshold
        self.start_time = None
        self.joint_id = None
        self.actuator_id = None

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [MoveDownUntilContact::setup()]")

    def initialise(self):
        self.start_time = self.data.time
        self.joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.down_joint)
        self.actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self.down_joint}_position")
        self.logger.debug(f"{self.name} [MoveDownUntilContact::initialise()]")

    def update(self):
        torque = np.linalg.norm(self.data.sensordata)
        if torque > self.torque_threshold and (self.data.time - self.start_time) > 1.0:
            self.logger.debug(f"{self.name}: Contact detected. Torque={torque:.3f}")
            return py_trees.common.Status.SUCCESS

        current_pos = self.data.qpos[self.model.jnt_qposadr[self.joint_id]]
        self.data.ctrl[self.actuator_id] = max(0, current_pos - self.step_size)
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.debug(f"{self.name} [MoveDownUntilContact::terminate()] -> {new_status}")


class AcquireTableHeight(py_trees.behaviour.Behaviour):
    def __init__(self, name, model, data, robot=None):
        super().__init__(name)
        self.model = model
        self.data = data
        self.robot = robot
        self.l_site_name = "robot0_left_fingertip"
        self.l_site_id = None

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [AcquireTableHeight::setup()]")

    def initialise(self):
        self.l_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.l_site_name)
        self.logger.debug(f"{self.name} [AcquireTableHeight::initialise()]")

    def update(self):
        self.site_pos = self.data.site_xpos[self.l_site_id].copy()
        self.site_quat = self.data.site_xmat[self.l_site_id].reshape(3,3).copy()
        self.logger.debug(f"{self.name}: Table height acquired = {self.site_pos[2]:.3f}")
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        self.logger.debug(f"{self.name} [AcquireTableHeight::terminate()] -> {new_status}")


class ApproachCylinder(py_trees.behaviour.Behaviour):
    def __init__(self, name, model, data, robot, target_pos=None, kp=1.0, tolerance=1e-2):
        super().__init__(name)
        self.model = model
        self.data = data
        self.robot = robot
        self.kp = kp
        self.tolerance = tolerance
        self.target_pos = np.array(target_pos) if target_pos is not None else np.array([0.5, -0.2, 0.5])

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [ApproachCylinder::setup()]")

    def initialise(self):
        self.start_time = self.data.time
        self.logger.debug(f"{self.name} [ApproachCylinder::initialise()]")

    def update(self):
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot0_ee_pose")
        ee_pos, ee_quat = tiago_utils.tiago_forward_kinematic("base", "robot0_ee_pose", self.model, self.data)
        J_full = tiago_actions_utils.jacobian_in_base_frame(self.model, self.data, ee_id, base_body="robot0_base_link")

        position_error = self.target_pos - ee_pos
        target_quat = np.array([0.7071068, 0, 0.7071068, 0])
        orn_err = tiago_actions_utils.quat_error(ee_quat, target_quat)

        error = np.hstack([position_error, orn_err])
        dq = np.linalg.pinv(J_full[:, 19:26]) @ (self.kp * error)
        self.data.ctrl[:7] = dq

        if np.allclose(error, 0, atol=self.tolerance):
            self.logger.debug(f"{self.name}: Reached cylinder pose.")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.debug(f"{self.name} [ApproachCylinder::terminate()] -> {new_status}")
        if new_status == py_trees.common.Status.SUCCESS:
            self.data.ctrl[:7] = 0.0


class OpenGripper(py_trees.behaviour.Behaviour):
    def __init__(self, name, model, data, robot, target_width=0.045):
        super().__init__(name)
        self.model = model
        self.data = data
        self.robot = robot
        self.target_width = target_width

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [OpenGripper::setup()]")

    def initialise(self):
        self.logger.debug(f"{self.name} [OpenGripper::initialise()]")

    def update(self):
        self.data.ctrl[7:9] = self.target_width
        if np.allclose(self.data.ctrl[7:9], self.target_width, atol=1e-5):
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.debug(f"{self.name} [OpenGripper::terminate()] -> {new_status}")


class GraspCylinder(ApproachCylinder):
    def update(self):
        status = super().update()
        if status == py_trees.common.Status.SUCCESS:
            self.data.ctrl[7:9] = 0.0  # Close gripper
        return status


class LiftCylinder(ApproachCylinder):
    def update(self):
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot0_ee_pose")
        ee_pos, ee_quat = tiago_utils.tiago_forward_kinematic("base", "robot0_ee_pose", self.model, self.data)
        self.target_pos[2] += 0.1 
