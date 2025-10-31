import mujoco
import py_trees
import numpy as np
import time
import xml.etree.ElementTree as ET


import my_mujoco_simulation.robot.robot_action_utils as action_utils
from my_mujoco_simulation.robot.controller.controller import Controller

class MovePose(py_trees.behaviour.Behaviour):
    def __init__(self, name, model, data, robot, target_pose, kp=1.0, tolerance=1e-2):
        super().__init__(name)
        self.model = model
        self.data = data
        self.robot = robot
        self.target_pose = np.array(target_pose)
        self.kp = kp
        self.tolerance = tolerance
        self.start_time = None
        self.controller = Controller(self.model, self.data, self.robot)

        robot_path = self.robot.get_robot_dict()["path"]
        tree = ET.parse(robot_path)
        root = tree.getroot()
        sites = [b.attrib.get("name") for b in root.findall(".//site[@name]")]
        self.ee_frame = f"{robot.get_robot_dict()["name"]}_{sites[-1]}" if sites else None

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [MovePose::setup()]")

    def initialise(self):
        self.start_time = self.data.time
        self.logger.debug(f"{self.name} [MovePose::initialise()]")

    def update(self):      
        dq, error = action_utils.inverse_kinematic(self.model, self.data, self.robot, self.ee_frame, self.target_pose)
        self.controller.set_joint_velocity(dq)
        if np.allclose(error[:], 0, atol=self.tolerance):
            self.logger.debug(f"{self.name}: Reached  pose.")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

class OpenGripper(py_trees.behaviour.Behaviour):
    def __init__(self, name, model, data, robot):
        super().__init__(name)
        self.model = model
        self.data = data
        self.robot = robot
        self.controller = Controller(self.model, self.data, self.robot)

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [OpenGripper::setup()]")

    def initialise(self):
        self.logger.debug(f"{self.name} [OpenGripper::initialise()]")

    def update(self):
        self.controller.set_joint_velocity(np.zeros(7))
        self.controller.open_gripper()
        if self.controller._is_gripper_open():
            self.logger.debug(f"{self.name}: Gripper is open.")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.debug(f"{self.name} [OpenGripper::terminate()] -> {new_status}")

class CloseGripper(py_trees.behaviour.Behaviour):
    def __init__(self, name, model, data, robot):
        super().__init__(name)
        self.model = model
        self.data = data
        self.robot = robot
        self.controller = Controller(self.model, self.data, self.robot)

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [OpenGripper::setup()]")

    def initialise(self):
        self.logger.debug(f"{self.name} [OpenGripper::initialise()]")

    def update(self):
        self.controller.set_joint_velocity(np.zeros(7))
        self.controller.close_gripper()
        if self.controller._is_gripper_close():
            self.logger.debug(f"{self.name}: Gripper is close.")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.debug(f"{self.name} [OpenGripper::terminate()] -> {new_status}")

class MoveDownUntillContact(py_trees.behaviour.Behaviour):
    def __init__(self, name, model, data, robot, target_pose, kp=10.0, tolerance=1e-2, torque_threshold = 2):
        super().__init__(name)
        self.model = model
        self.data = data
        self.robot = robot
        self.target_pose = np.array(target_pose)
        self.kp = kp
        self.tolerance = tolerance
        self.start_time = None
        self.controller = Controller(self.model, self.data, self.robot)
        self.torque_threshold = torque_threshold

        robot_path = self.robot.get_robot_dict()["path"]
        tree = ET.parse(robot_path)
        root = tree.getroot()
        sites = [b.attrib.get("name") for b in root.findall(".//site[@name]")]
        self.ee_frame = f"{robot.get_robot_dict()["name"]}_{sites[-1]}" if sites else None

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [ApproachTable::setup()]")

    def initialise(self):
        self.start_time = self.data.time
        self.logger.debug(f"{self.name} [ApproachTable::initialise()]")

    def update(self):
        self.current_time = self.data.time - self.start_time
        self.target_pose[2] = self.target_pose[2] - 0.0001
        dq, _ = action_utils.inverse_kinematic(self.model, self.data, self.robot, self.ee_frame, self.target_pose)
        self.controller.set_joint_velocity(dq)

        torque = np.array([self.robot.get_sensor_data(self.data, 3), self.robot.get_sensor_data(self.data, 4), self.robot.get_sensor_data(self.data, 5)])
        torque_norm = np.linalg.norm(torque)

        if torque_norm > self.torque_threshold and self.current_time>1:
            self.logger.debug(f"{self.name}: Contact detected! Torque={torque_norm:.3f}")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

class MoveJoints(py_trees.behaviour.Behaviour):
    def __init__(self, name, model, data, robot, target_pos, kp=1.0, tolerance=1e-2):
        super().__init__(name)
        self.model = model
        self.data = data
        self.robot = robot
        self.target_pos = np.array(target_pos)
        self.kp = kp
        self.tolerance = tolerance
        self.start_time = None
        self.controller = Controller(self.model, self.data, self.robot)

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [MovePose::setup()]")

    def initialise(self):
        self.start_time = self.data.time
        self.logger.debug(f"{self.name} [MovePose::initialise()]")

    def update(self):      
        dq = self.kp * (self.target_pos - self.robot.get_arm_joint_positions(self.model, self.data))
        self.controller.set_joint_velocity(dq)
        if np.allclose(dq[:], 0, atol=self.tolerance):
            self.logger.debug(f"{self.name}: Reached  pose.")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

class MeasureGripperSites(py_trees.behaviour.Behaviour):
    def __init__(self, name, model, data, robot):
        super().__init__(name)
        self.model = model
        self.data = data
        self.robot = robot
        self.joint_names = np.array(robot.get_robot_dict()["joints"])
        robot_prefix = robot.get_robot_dict()["name"]
        self.l_site_name = f"{robot_prefix}_left_fingertip"
        self.r_site_name = f"{robot_prefix}_right_fingertip"
        self.l_site_id = None
        self.r_site_id = None
        self.l_site_pos = None
        self.l_site_quat = None
        self.r_site_pos = None
        self.r_site_quat = None
        self.joint_qpos_indices = None
        self.joint_positions = None

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [MeasureGripperSites::setup()]")

    def initialise(self):
        self.logger.debug(f"{self.name} [MeasureGripperSites::initialise()]")

        self.l_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.l_site_name)
        self.r_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.r_site_name)
        if self.l_site_id < 0:
            self.logger.error(f"{self.name}: Site '{self.l_site_name}' not found!")
            return py_trees.common.Status.FAILURE
        if self.r_site_id < 0:
            self.logger.error(f"{self.name}: Site '{self.r_site_name}' not found!")
            return py_trees.common.Status.FAILURE

        self.joint_qpos_indices = [self.model.jnt_qposadr[i] for i in range(self.model.njnt)]

    def update(self):
        self.joint_positions = self.data.qpos[self.joint_qpos_indices].copy()

        self.l_site_pos = self.data.site_xpos[self.l_site_id].copy()
        self.l_site_quat = self.data.site_xmat[self.l_site_id].reshape(3, 3).copy()

        self.r_site_pos = self.data.site_xpos[self.r_site_id].copy()
        self.r_site_quat = self.data.site_xmat[self.r_site_id].reshape(3, 3).copy()

        self.mean_pos = (self.l_site_pos + self.r_site_pos) / 2

        print("Table height", self.mean_pos[2])

        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        self.logger.debug(f"{self.name} [MeasureGripperSites::terminate()] -> {new_status}")

class MeasureMassWithTorque(py_trees.behaviour.Behaviour):
    def __init__(self, name, model, data, robot):
        super().__init__(name)
        self.model = model
        self.data = data
        self.robot = robot
        self.sens_id = 4
        self.torque = []        
        self.robot_prefix = self.robot.get_robot_dict()["name"]
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.robot_prefix}_arm_joint6")
        self.body_id = self.model.jnt_bodyid[joint_id]
        self.dist = 0.103 - 0.0584
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(key=name, access=py_trees.common.Access.WRITE)

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [MeasureGripperSites::setup()]")

    def initialise(self):
        self.logger.debug(f"{self.name} [MeasureGripperSites::initialise()]")
        self.start_time = self.data.time
        

    def update(self):
        if abs(self.data.time - self.start_time) < 5:
            if abs(self.data.time - self.start_time) > 4:
                self.torque.append(self.robot.get_sensor_data(self.data, self.sens_id))
                
            return py_trees.common.Status.RUNNING
        self.mean_torque = np.mean(self.torque)
        self.measured_mass = self.mean_torque / (self.dist * -9.81)
        self.blackboard.set(self.name, self.measured_mass)
        print("object masss", self.measured_mass+0.46)
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        self.logger.debug(f"{self.name} [MeasureGripperSites::terminate()] -> {new_status}")

class MeasureGripperOpnening(py_trees.behaviour.Behaviour):
    def __init__(self, name, model, data, robot):
        super().__init__(name)
        self.model = model
        self.data = data
        self.robot = robot
        self.controller = Controller(self.model, self.data, self.robot)

        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(key=name, access=py_trees.common.Access.WRITE)

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [MeasureGripperSites::setup()]")

    def initialise(self):
        self.logger.debug(f"{self.name} [MeasureGripperSites::initialise()]")
        self.start_time = self.data.time
        

    def update(self):
        if abs(self.data.time - self.start_time) < 3:
            if abs(self.data.time - self.start_time) > 2:
                self.measure = self.controller.get_gripper_opening()
                print("object size", self.measure)
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.RUNNING
        
        return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        self.logger.debug(f"{self.name} [MeasureGripperSites::terminate()] -> {new_status}")

