import mujoco
import py_trees
import numpy as np
import time
import xml.etree.ElementTree as ET

class IsJointConf(py_trees.behaviour.Behaviour):
    def __init__(self, name, model, data, robot, target_pos, tolerance=1e-2):
        super().__init__(name)
        self.model = model
        self.data = data
        self.robot = robot
        self.target_pos = np.array(target_pos)

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [IsJointConf::setup()]")

    def initialise(self):
        self.logger.debug(f"{self.name} [IsJointConf::initialise()]")

    def update(self):
        self.logger.debug(f"{self.name} [IsJointConf::update()]")
        
        if np.allclose(self.robot.get_arm_joint_positions(self.model, self.data), self.target_pos, atol=1e-3):
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

class IsInPose(py_trees.behaviour.Behaviour):
    def __init__(self, name, model, data, robot, target_pose, tolerance=1e-2):
        super().__init__(name)
        self.model = model
        self.data = data
        self.robot = robot
        self.target_pose = np.array(target_pose)

        robot_path = self.robot.get_robot_dict()["path"]
        tree = ET.parse(robot_path)
        root = tree.getroot()
        sites = [b.attrib.get("name") for b in root.findall(".//site[@name]")]
        self.ee_frame = f"{self.robot.get_robot_dict()["name"]}_{sites[-1]}" if sites else None
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_name)
    
    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [IsInPose::setup()]")

    def initialise(self):
        self.logger.debug(f"{self.name} [IsInPose::initialise()]")

    def update(self):
        self.logger.debug(f"{self.name} [IsInPose::update()]")
        ee_pos = data.site_xpos[self.ee_id].copy()
        ee_quat = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quat, data.site_xquat[self.ee_id].copy().flatten())
        ee_pose = np.hstack([ee_pos, ee_quat])
        if np.allclose(see_pose, self.target_pose, atol=1e-3):
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

class IsObjectMassValid(py_trees.behaviour.Behaviour):
    def __init__(self, name, measured_mass = 0.5, lower_bound=0.0, upper_bound=2.0):
        super().__init__(name)
        self.measured_mass = measured_mass
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self, **kwargs):
        self.logger.debug(f"{self.name} [IsObjectMassValid::setup()]")

    def initialise(self):
        self.logger.debug(f"{self.name} [IsObjectMassValid::initialise()]")

    def update(self):
        #mass = self.measured_mass() if callable(self.measured_mass) else self.measured_mass
        #self.logger.debug(f"{self.name}: Measured mass = {mass:.4f} kg")
        print(getattr(self.blackboard, "measured_mass_value", None))
        #if mass is None :
            #self.logger.warning(f"{self.name}: Invalid mass reading (None or NaN)")
            #return py_trees.common.Status.FAILURE

        #if not (self.lower_bound <= mass <= self.upper_bound):
            #self.logger.warning(f"{self.name}: Mass {mass:.3f} kg out of range [{self.lower_bound}, {self.upper_bound}]")
            #return py_trees.common.Status.FAILURE
            
        return py_trees.common.Status.SUCCESS



