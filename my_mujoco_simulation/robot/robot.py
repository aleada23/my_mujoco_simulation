import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import copy

class Robot:
    def __init__(self, robot_path, prefix, init_pos=None, init_orn=None, init_config=None):
        self.robot_path = robot_path
        self.prefix = prefix
        self.init_pos = np.array(init_pos or [0, 0, 0])
        self.init_orn = np.array(init_orn or [1, 0, 0, 0])
        self.init_config = init_config
        # Extract from XML
        self._extract_robot_info()
        self.joint_ids = {}
        self.actuator_ids = {}
        self.sensor_ids = {}

    # XML Parsing
    def _extract_robot_info(self):
        tree = ET.parse(self.robot_path)
        root = tree.getroot()
        self.robot_model = root.attrib.get("model")
        self.joint_names = [
            f"{self.prefix}_{joint.attrib['name']}"
            for joint in root.findall(".//joint[@name]")
        ]
        self.actuator_names = [
            f"{self.prefix}_{act.attrib['name']}"
            for act in root.findall(".//actuator/*[@name]")
        ]
        self.actuator_types = [
            act.tag for act in root.findall(".//actuator/*[@name]")
        ]
        self.sensor_names = [
            f"{self.prefix}_{sen.attrib['name']}"
            for sen in root.findall(".//sensor/*[@name]")
        ]
        #end-effector pose is last body, XML was modified to include it in last position
        bodies = [b.attrib.get("name") for b in root.findall(".//body[@name]")]
        sites = [b.attrib.get("name") for b in root.findall(".//site[@name]")]
        self.ee_name = f"{self.prefix}_{sites[-1]}" if sites else None
        self.base_name = f"{self.prefix}_{bodies[0]}" if bodies else None

    # Model/Data Mappings
    def _build_mujoco_ids(self, model, data):
        #Find models indeces for the joints in model. !!in data the value is not the same !! self.model.jnt_qposadr to trasnform
        for jn in self.joint_names:
            try:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                self.joint_ids[jn] = jid
            except Exception:
                pass
        #Find actuators name for joint control during the simulation
        for an in self.actuator_names:
            try:
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, an)
                self.actuator_ids[an] = aid
            except Exception:
                pass
        #Find sesnsors index for simulation
        for sn in self.sensor_names:
            try:
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sn)
                self.sensor_ids[sn] = sid
            except Exception:
                pass
   
    # Accessors
    def get_joint_positions(self, model, data):
        return np.array([data.qpos[model.jnt_qposadr[self.joint_ids[jn]]] for jn in self.joint_names])

    def get_arm_joint_positions(self, model, data):
        ids = self.get_robot_dict()["joints"]
        act_names = [k for k in ids if "arm" in k]
        return np.array([data.qpos[model.jnt_qposadr[self.joint_ids[i]]] for i in act_names])

    def get_arm_joint_velocities(self, model, data):
        ids = self.get_robot_dict()["joints"]
        act_names = [k for k in ids if "arm" in k]
        return np.array([data.qvel[model.jnt_dofadr[self.joint_ids[jn]]] for jn in act_names])

    def set_actuator_ctrl(self, ctrl_values):
        for i, an in enumerate(self.actuator_names):
            aid = self.actuator_ids.get(an)
            if aid is not None and i < len(ctrl_values):
                self.data.ctrl[aid] = ctrl_values[i]

    def get_Jacobian(self, model, data, frame_name=None):
        if frame_name is None:
            if self.ee_name is None:
                raise ValueError("End-effector not defined and no frame_name provided.")
            frame_name = self.ee_name

        # Resolve body ID
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, frame_name)
        except Exception:
            # Try with prefixed name if available
            prefixed_name = f"{self.prefix}_{frame_name}"
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefixed_name)
        nv = model.nv
        Jp_full = np.zeros((3, nv))  # Linear velocity Jacobian
        Jr_full = np.zeros((3, nv))  # Angular velocity Jacobian

        mujoco.mj_jacSite(model, data, Jp_full, Jr_full, body_id)
        arm_joint_ids = [
            self.joint_ids[jn]
            for jn in self.joint_names
            if "arm_joint" in jn and jn in self.joint_ids
        ]

        if not arm_joint_ids:
            raise ValueError("No joints found matching 'arm_joint' in this robot.")

        dof_indices = [model.jnt_dofadr[jid] for jid in arm_joint_ids]

        # Extract only the relevant columns
        Jp = np.zeros((3, len(dof_indices)))
        Jr = np.zeros((3, len(dof_indices)))

        for i, dof_idx in enumerate(dof_indices):
            Jp[:, i] = Jp_full[:, dof_idx]
            Jr[:, i] = Jr_full[:, dof_idx]
        Jac = np.vstack([Jp, Jr])
        
        return Jac

    def get_Jacobian_in_base(self, model, data, frame_name=None):
        J = self.get_Jacobian(model, data, frame_name)
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.base_name)
        base_rot = data.xmat[base_id].reshape(3,3)
        R = np.zeros((6,6))
        R[:3,:3] = base_rot.T
        R[3:, 3:] = base_rot.T
        return R @ J

    def set_robot_init_config(self, model, data):
        if self.init_config is None:
            return
        # --- Set joint qpos
        for name, qpos_value in zip(self.joint_names, self.init_config):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)            
            qpos_addr = model.jnt_qposadr[jid]
            data.qpos[qpos_addr] = qpos_value
            mujoco.mj_forward(model, data)
        data.ctrl[:] = np.nan  # disables control (no actuator signal)
        for i in range(model.nu):
        # Check actuator type — only handle position actuators
        #if model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_NONE:  # or check XML if needed
            jid = model.actuator_trnid[i, 0]
            if jid >= 0:  # valid joint id
                qpos_addr = model.jnt_qposadr[jid]
                data.ctrl[i] = data.qpos[qpos_addr]
    
    def get_arm_joint_torque(self, model, data):
        ids = self.get_robot_dict()["actuatorsindex"]
        arm_ids = []
        act_names = self.get_robot_dict()["actuators"]
        for j in range(len(act_names)):
            if "arm" in act_names[j]:
                arm_ids.append(ids[act_names[j]])
        return np.array([data.qfrc_actuator[i] for i in arm_ids])


    #Sensors
    def get_sensor_data(self, data, sensor_id):
        #if sensor_id is None:
        #    raise ValueError(f"Sensor '{sensor_name}' not found.")
        return data.sensordata[sensor_id]
    
    # Utilities
    def get_end_effector_pose(self, model, data):
        if self.ee_name is None:
            raise ValueError("End-effector not defined.")
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.ee_name)
        pos = data.site_xpos[ee_id].copy()
        mat = data.site_xmat[ee_id].copy() 
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        pose = np.hstack((pos, quat))
        return self.map_pose_into_robot(model, data, pose)

    def get_end_effector_velocity(self, model, data):
        if self.ee_name is None:
            raise ValueError("End-effector not defined.")
    
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.ee_name)

        jac = self.get_Jacobian_in_base(model, data)

        # Compute site velocity: v = J * qdot
        qvel = self.get_arm_joint_velocities(model, data)
        vel = jac @ qvel #[3 linear, 3 angular]
        return vel

    def get_end_effector_site(self, model):
        if self.ee_name is None:
            raise ValueError("End-effector not defined.")
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.ee_name)
    
    def get_end_effector_name(self, model):
        if self.ee_name is None:
            raise ValueError("End-effector not defined.")
        return self.ee_name

    def get_body_pose_with_frame(self, model, data, body_name, frame_name = "world"):
        if self.ee_name is None:
            raise ValueError("End-effector not defined.")
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, body_name)
        body_pos = data.site_xpos[body_id]
        body_mat = data.site_xmat[body_id].reshape(3,3)

        frame_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.base_name)
        frame_pos = data.xpos[frame_id]
        frame_mat = data.xmat[frame_id].reshape(3,3)

        pos = frame_mat.T @ (body_pos-frame_pos)
        rot = frame_mat.T @ body_mat
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.flatten())
        return pos, quat

    def forward_kinematics(self, body_name):
        """Return position and orientation of a body in world frame."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[bid], self.data.xquat[bid]

    def map_pose_into_robot(self, model, data, pose):
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.base_name)
        base_rot = data.xmat[base_id].reshape(3,3)
        base_pos = data.xpos[base_id]
        pose[:3] = base_rot @ (pose[:3] - base_pos)
        rot = np.zeros((9))
        mujoco.mju_quat2Mat(rot, pose[3:])
        rot = base_rot @ rot.reshape(3,3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.reshape(9,1))
        pose[3:] = quat
        return pose
    
    def map_velocity_into_robot(self, model, data, vel):
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.base_name)
        base_rot = data.xmat[base_id].reshape(3,3)
        return base_rot @ vel
    # Export
    def get_robot_dict(self):
        return copy.deepcopy({
            "type": "robot",
            "model": self.robot_model,
            "name": self.prefix,
            "path": self.robot_path,
            "position": self.init_pos,
            "orientation": self.init_orn,
            "configuration": self.init_config,
            "njoints": len(self.joint_names),
            "joints": self.joint_names,
            "actuators": self.actuator_names,
            "actuatorstype": self.actuator_types,
            "actuatorsindex": self.actuator_ids,
            "sensors": self.sensor_names,
            "ee_name": self.ee_name,
            "base_name": self.base_name,
        })
