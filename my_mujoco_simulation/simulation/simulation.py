import os
import time
import tempfile
import xml.etree.ElementTree as ET
import xml.dom.minidom
import mujoco
import mujoco.viewer
import copy

from my_mujoco_simulation.environment.environment import Environment
from my_mujoco_simulation.robot.robot import Robot
from my_mujoco_simulation.object.object import SimObject
from my_mujoco_simulation.object.table import table
from my_mujoco_simulation.object.geomobj import geomobj
from my_mujoco_simulation.object.pedestal import pedestal
from my_mujoco_simulation.camera import camera


from my_mujoco_simulation.simulation.xml_utils import rename_subtree, ensure_section, merge_section


class Simulation:
    def __init__(self, env_path, use_viewer=True):
        self.env_path = env_path
        self.use_viewer = use_viewer
        self.robot_list = []
        self.object_list = []
        self.environment = Environment(env_path)
        self.environment_tree = None
        self.environment_root = None
        self.sim_model = None
        self.sim_data = None
        self.env_model = None

    #Robot set up
    
    def add_robot(self, robot_path, position=None, orientation=None, prefix=None, init_config = None):
        if prefix is None:
            prefix = f"robot{len(self.robot_list)}"
        robot = Robot(robot_path, prefix, position, orientation, init_config)
        self.robot_list.append(robot)

    def add_robots_to_world(self):
        for robot in self.robot_list:
            robot_dict = robot.get_robot_dict()
            prefix = robot_dict["name"]
            robot_tree = ET.parse(robot_dict["path"])
            robot_root = robot_tree.getroot()

            # Merge common sections
            for tag in ["default", "asset", "actuator", "sensor", "tendon", "equality", "contact"]:
                merge_section(self.environment_root, robot_root, tag, prefix)

            # Merge worldbody with position/orientation overrides
            env_worldbody = ensure_section(self.environment_root, "worldbody")
            robot_worldbody = robot_root.find("worldbody")
            if robot_worldbody is not None:
                for body in robot_worldbody:
                    copy_body = rename_subtree(body, prefix)
                    if "base" in copy_body.attrib.get("name", ""):
                        copy_body.set("pos", robot_dict["position"])
                        copy_body.set("quat", robot_dict["orientation"])
                    env_worldbody.append(copy.deepcopy(copy_body))

    def get_robot(self, index):
        return self.robot_list[index]
    
    #Object set up

    def add_object(self, obj_path, prefix=None, pos=None, quat=None, shape=None, size=None, width=None, length=None, height=None, leg_width=None, leg_height=None, color=None, friction=None, mass=None, fov = None, resolution = None):
        if prefix is None:
            prefix = f"object{len(self.object_list)}"
        if "table" in obj_path:
            _object = table.Table(obj_path, prefix, pos, quat, width, length, height, leg_width, leg_height, color, friction)
        elif "pedestal" in obj_path:
            _object = pedestal.Pedestal(obj_path, prefix, pos, quat, shape, size, color)
        elif "geomobj" in obj_path:
            _object = geomobj.GeometricObject(obj_path, prefix, pos, quat, size, color, friction, mass)
        elif "camera" in obj_path:
            _object = camera.Camera(obj_path, prefix, pos, quat, fov, resolution)
        else:
            _object = SimObject(obj_path, prefix, pos, quat)
        self.object_list.append(_object)

    def add_objects_to_world(self):
        for _object in self.object_list:
            _object.add_sub_tree(self.environment_root)
            """object_dict = _object.get_object_dict()
            prefix = object_dict["name"]
            object_tree = ET.parse(object_dict["path"])
            object_root = object_tree.getroot()
            if "geomobj" in object_dict["path"]:
                pass
            # Merge common sections
            for tag in ["default", "asset", "actuator", "sensor", "tendon", "equality", "contact"]:
                merge_section(self.environment_root, object_root, tag, prefix)

            # Merge worldbody with position/orientation overrides
            env_worldbody = ensure_section(self.environment_root, "worldbody")
            object_worldbody = object_root.find("worldbody")
            if object_worldbody is not None:
                for body in object_worldbody:
                    copy_body = rename_subtree(body, prefix)
                    if "base" in copy_body.attrib.get("name", ""):
                        copy_body.set("pos", object_dict["position"])
                        copy_body.set("quat", object_dict["orientation"])
                    env_worldbody.append(copy.deepcopy(copy_body))
            """
    
    # XML and model setup  

    def add_environment(self):
        self.environment_tree = ET.parse(self.environment.get_path())
        self.environment_root = self.environment_tree.getroot()

    def save_model(self, pretty=False):
        if not self.environment_root:
            raise RuntimeError("Environment not initialized. Call add_environment() first.")
        xml_str = ET.tostring(self.environment_root, encoding="unicode")
        if pretty:
            xml_str = xml.dom.minidom.parseString(xml_str).toprettyxml(newl="")
        with open(self.env_model, "w") as f:
            f.write(xml_str)

    def create_mujoco_model(self):
        self.sim_model = mujoco.MjModel.from_xml_path(self.env_model)
        self.sim_data = mujoco.MjData(self.sim_model)
        
        for robot in self.robot_list:
            robot._build_mujoco_ids(self.sim_model, self.sim_data)

    # Simulation launch 
     
    def launch(self, pretty_xml=False):
        self.add_environment()
        self.add_robots_to_world()
        self.add_objects_to_world()


        # Create a temporary XML file for this run
        with tempfile.NamedTemporaryFile(dir = "",suffix=".xml", delete=False) as tmp:
            self.env_model = tmp.name
            self.save_model(pretty=pretty_xml)

        self.create_mujoco_model()
        print(f"Simulation model created: {self.env_model}")
        for robot in self.robot_list:
            robot._build_mujoco_ids(self.sim_model, self.sim_data)
            robot.set_robot_init_config(self.sim_model, self.sim_data)
            
            
        
        os.remove(self.env_model)
        print(f"Cleaned up temporary model file: {self.env_model}")
        return self.sim_model, self.sim_data
        