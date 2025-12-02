from my_mujoco_simulation.object.object import SimObject
import xml.etree.ElementTree as ET
from my_mujoco_simulation.simulation.xml_utils import rename_subtree, ensure_section, merge_section
import copy
import mujoco
import numpy as np


class GeometricObject(SimObject):
    def __init__(self, obj_path, prefix=None, pos=None, quat=None, size=None, color=None, friction=None, mass=None):
        super().__init__(obj_path, prefix, pos, quat)
        self.obj_path = obj_path
        self.obj_dict["subtype"] = "geometric"
        self.size = size
        self.color = color
        self.friction = friction
        self.mass = mass
        geom = self.root.find(".//geom")
        if geom is not None:
            # Set size if provided
            if size is not None:
                geom.set("size", " ".join(map(str, size)))
            # Set color if provided
            if color is not None:
                geom.set("rgba", " ".join(map(str, color)))
            # Set friction if provided
            if friction is not None:
                geom.set("friction", " ".join(map(str, friction)))
            # Set mass if provided
            if mass is not None:
                geom.set("mass", str(mass))

    def add_sub_tree(self, environment_root):
        object_dict = self.get_object_dict()
        prefix = object_dict["name"]
        object_tree = ET.parse(object_dict["path"])
        object_root = object_tree.getroot()
        # Merge common sections
        for tag in ["default", "asset", "actuator", "sensor", "tendon", "equality", "contact"]:
            merge_section(environment_root, object_root, tag, prefix)

        # Merge worldbody with position/orientation overrides
        env_worldbody = ensure_section(environment_root, "worldbody")
        object_worldbody = object_root.find("worldbody")
        if object_worldbody is not None:
            for body in object_worldbody:
                copy_body = rename_subtree(body, prefix)
                if "base" in copy_body.attrib.get("name", ""):
                    copy_body.set("pos", object_dict["position"])
                    copy_body.set("quat", object_dict["orientation"])
                if "geom" in copy_body.attrib.get("name", ""):
                    copy_body.set("friction", self.friction)
                env_worldbody.append(self.set_object_subtree(copy.deepcopy(copy_body)))
                

    def set_object_subtree(self, element):
        if "name" in element.attrib:
            element.attrib["name"] = f"{element.attrib.get("name", "")}"
        if self.size is not None and "size" in element.attrib:
            element.attrib["size"] = self.size
        if self.mass is not None and "mass" in element.attrib:
            element.attrib["mass"] = self.mass
        if self.friction is not None and "friction" in element.attrib:
            element.attrib["friction"] = self.friction
        if self.color is not None and "rgba" in element.attrib:
            element.attrib["rgba"] = self.color
        for child in element:
            self.set_object_subtree(child)

        return element

    def get_pose(self, model, data):
        tree = ET.parse(self.obj_path)
        root = tree.getroot()
        bodies = [b.attrib.get("name") for b in root.findall(".//body[@name]")]
        object_base = f"{self.prefix}_{bodies[0]}" if bodies else None
        object_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_base)
        obj_quat = data.xquat[object_base_id]
        obj_pos = data.xpos[object_base_id]

        return np.hstack((obj_pos, obj_quat))
    
    def get_body_name(self):
        tree = ET.parse(self.obj_path)
        root = tree.getroot()
        bodies = [b.attrib.get("name") for b in root.findall(".//body[@name]")]
        object_base = f"{self.prefix}_{bodies[0]}" if bodies else None
        return object_base
    
    def get_call_geom_name(self):
        tree = ET.parse(self.obj_path)
        root = tree.getroot()
        geom = [b.attrib.get("name") for b in root.findall(".//geom[@name]")]
        object_geom = f"{self.prefix}_{geom[0]}" if geom else None
        return object_geom