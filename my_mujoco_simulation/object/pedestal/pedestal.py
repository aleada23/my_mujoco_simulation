from my_mujoco_simulation.object.object import SimObject
import xml.etree.ElementTree as ET
from my_mujoco_simulation.simulation.xml_utils import rename_subtree, ensure_section, merge_section
import copy

class Pedestal(SimObject):
    def __init__(self, obj_path, prefix=None, pos=None, quat=None, shape=None, size=None, color=None):
        super().__init__(obj_path, prefix, pos, quat)
        self.obj_dict["subtype"] = "pedestal"
        self.size = size
        self.color = color
        self.shape = shape
        geom = self.root.find(".//geom")
        if geom is not None:
            if shape is not None:
                geom.set("type", shape)
            if size is not None:
                geom.set("size", " ".join(map(str, size)))
            if color is not None:
                geom.set("rgba", " ".join(map(str, color)))

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
                env_worldbody.append(self.set_object_subtree(copy.deepcopy(copy_body)))

    def set_object_subtree(self, element):
        if "name" in element.attrib:
            element.attrib["name"] = f"{element.attrib.get("name", "")}"
        if self.size is not None and "size" in element.attrib:
            element.attrib["size"] = self.size
        if self.color is not None and "rgba" in element.attrib:
            element.attrib["rgba"] = self.color
        for child in element:
            self.set_object_subtree(child)

        return element

