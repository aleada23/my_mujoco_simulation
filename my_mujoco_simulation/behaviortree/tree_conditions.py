# conditions.py
import mujoco
from py_trees.behaviour import Behaviour
import py_trees
import numpy as np

class IsHomeConf(Behaviour):
    def __init__(self, name: str, model, data, robot):
        super().__init__(name)
        self.home_config = np.array([0.20, -1.34, -0.20, 1.94, -1.57, 1.37, 0.0])
        self.joint_list = np.array(robot["joints"][:7])
        self.model = model
        self.data = data

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [IsHomeConf::setup()]")

    def initialise(self):
        self.logger.debug(f"{self.name} [IsHomeConf::initialise()]")

    def update(self):
        current_qpos = np.array([self.data.qpos[self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]] for j in self.joint_list])
        if np.allclose(current_qpos, self.home_config, rtol=1e-2, atol=1e-3):
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class IsGripperPose(Behaviour):
    def __init__(self, name, model, data, gripper_body_name, target_pose, pos_tol=1e-3, ori_tol=1e-3):
        super().__init__(name)
        self.model = model
        self.data = data
        self.gripper_body_name = gripper_body_name
        self.target_pos = np.array(target_pose[:3])
        self.target_quat = np.array(target_pose[3:])
        self.pos_tol = pos_tol
        self.ori_tol = ori_tol

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [IsGripperPose::setup()]")

    def initialise(self):
        self.logger.debug(f"{self.name} [IsGripperPose::initialise()]")

    def update(self):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.gripper_body_name)
        current_pos = self.data.xpos[body_id].copy()
        current_quat = self.data.xquat[body_id].copy()
        pos_error = np.linalg.norm(current_pos - self.target_pos)
        dot = np.clip(np.abs(np.dot(current_quat, self.target_quat)), -1.0, 1.0)
        ori_error = 2 * np.arccos(dot)
        if pos_error <= self.pos_tol and ori_error <= self.ori_tol:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class IsTableReachable(Behaviour):
    def __init__(self, name, model, data, gripper_body_name, table_pose, pose_upper_bounds=[2,2,1], pose_lower_bounds=[-2,-2,0], touch_distance=0.5):
        super().__init__(name)
        self.model = model
        self.data = data
        self.gripper_body_name = gripper_body_name
        self.table_pos = np.array(table_pose)
        self.touch_distance = touch_distance
        self.pose_upper_bounds = np.array(pose_upper_bounds)
        self.pose_lower_bounds = np.array(pose_lower_bounds)

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [IsTableReachable::setup()]")

    def initialise(self):
        self.logger.debug(f"{self.name} [IsTableReachable::initialise()]")

    def update(self):
        if self.table_pos is None:
            return py_trees.common.Status.FAILURE
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.gripper_body_name)
        gripper_pos = self.data.xpos[body_id].copy()
        distance = np.linalg.norm(gripper_pos - self.table_pos)
        if (self.table_pos < self.pose_lower_bounds).any() or (self.table_pos > self.pose_upper_bounds).any():
            return py_trees.common.Status.FAILURE
        if distance <= self.touch_distance:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class IsObjectDetected(Behaviour):
    def __init__(self, name, object_pose):
        super().__init__(name)
        self.object_pose = object_pose

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [IsObjectDetected::setup()]")

    def initialise(self):
        self.logger.debug(f"{self.name} [IsObjectDetected::initialise()]")

    def update(self):
        if self.object_pose is None:
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class IsObjectReachable(Behaviour):
    def __init__(self, name, model, data, gripper_body_name, object_pose, table_geom_name, reach_distance=0.9):
        super().__init__(name)
        self.model = model
        self.data = data
        self.gripper_body_name = gripper_body_name
        self.object_pose = np.array(object_pose) if object_pose is not None else None
        self.table_geom_name = table_geom_name
        self.reach_distance = reach_distance

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [IsObjectReachable::setup()]")

    def initialise(self):
        self.logger.debug(f"{self.name} [IsObjectReachable::initialise()]")

    def update(self):
        if self.object_pose is None:
            return py_trees.common.Status.FAILURE
        gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.gripper_body_name)
        gripper_pos = self.data.xpos[gripper_id]
        distance = np.linalg.norm(gripper_pos - self.object_pose)
        if distance <= self.reach_distance:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class IsObjectPickedUp(Behaviour):
    def __init__(self, name, model, data, gripper_body_name, object_body_name, table_geom_name, lift_height_threshold=0.05, grasp_distance_threshold=0.05):
        super().__init__(name)
        self.model = model
        self.data = data
        self.gripper_body_name = gripper_body_name
        self.object_body_name = object_body_name
        self.table_geom_name = table_geom_name
        self.lift_height_threshold = lift_height_threshold
        self.grasp_distance_threshold = grasp_distance_threshold

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [IsObjectPickedUp::setup()]")

    def initialise(self):
        self.logger.debug(f"{self.name} [IsObjectPickedUp::initialise()]")

    def update(self):
        gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.gripper_body_name)
        object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.object_body_name)
        table_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.table_geom_name)
        gripper_pos = self.data.xpos[gripper_id]
        object_pos = self.data.xpos[object_id]
        table_pos = self.data.xpos[table_id]
        table_height = table_pos[2] + self.model.geom_size[table_id][2]
        lifted = (object_pos[2] - table_height) > self.lift_height_threshold
        close_enough = np.linalg.norm(gripper_pos - object_pos) <= self.grasp_distance_threshold
        if lifted and close_enough:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class IsTableHeightValid(Behaviour):
    def __init__(self, name, measured_height, lower_bound=0.2, upper_bound=1.2):
        super().__init__(name)
        self.measured_height = measured_height
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [IsTableHeightValid::setup()]")

    def initialise(self):
        self.logger.debug(f"{self.name} [IsTableHeightValid::initialise()]")

    def update(self):
        height = self.measured_height() if callable(self.measured_height) else self.measured_height
        if height is None or np.isnan(height):
            return py_trees.common.Status.FAILURE
        if self.lower_bound <= height <= self.upper_bound:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class IsObjectMassValid(Behaviour):
    def __init__(self, name, measured_mass, lower_bound=0.0, upper_bound=2.0):
        super().__init__(name)
        self.measured_mass = measured_mass
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [IsObjectMassValid::setup()]")

    def initialise(self):
        self.logger.debug(f"{self.name} [IsObjectMassValid::initialise()]")

    def update(self):
        mass = self.measured_mass() if callable(self.measured_mass) else self.measured_mass
        if mass is None or np.isnan(mass):
            return py_trees.common.Status.FAILURE
        if self.lower_bound <= mass <= self.upper_bound:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE
