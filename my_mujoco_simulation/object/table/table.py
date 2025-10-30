from my_mujoco_simulation.object.object import SimObject
import xml.etree.ElementTree as ET
from my_mujoco_simulation.simulation.xml_utils import rename_subtree, ensure_section, merge_section
import copy

class Table(SimObject):
    def __init__(self, obj_path, prefix=None, pos=None, quat=None, width=None, length=None, height=None, leg_width=None, leg_height=None, thickness = None, color=None, friction=None):
        super().__init__(obj_path, prefix, pos, quat)
        self.obj_dict["subtype"] = "table"
        self.width=width or 1.0
        self.length=length or 0.5
        self.height=height or 0.8
        self.leg_width=leg_width 
        self.leg_height=leg_height
        self.color=color
        self.friction=friction
        self.thickness = thickness or 0.05
        self.size = [self.height, self.width, self.length, self.thickness] 
        # --- Modify table top ---
        top = self.root.find(".//geom[@name='table_top']")
        if top is not None:
            # Get current XML size
            size_xml = top.attrib.get("size", "1 1 1").split()
            size_xml = [float(s) for s in size_xml]

            # Override dimensions individually if provided
            new_width = width / 2 if width is not None else size_xml[0]
            new_length = length / 2 if length is not None else size_xml[1]
            new_height = height / 20 if height is not None else size_xml[2]

            top.set("size", f"{new_width} {new_length} {new_height}")

            if color is not None:
                top.set("rgba", " ".join(map(str, color)))
            if friction is not None:
                top.set("friction", " ".join(map(str, friction)))

        # --- Modify table legs ---
        legs = [g for g in self.root.findall(".//geom") if "leg" in g.attrib.get("name", "")]
        for leg in legs:
            # Read current XML size
            size_xml = leg.attrib.get("size", "0.05 0.05 0.4").split()
            size_xml = [float(s) for s in size_xml]

            new_leg_width = leg_width if leg_width is not None else size_xml[0]
            new_leg_height = leg_height if leg_height is not None else size_xml[2]

            # X and Y = leg_width, Z = leg_height
            leg.set("size", f"{new_leg_width} {new_leg_width} {new_leg_height}")

            if color is not None:
                leg.set("rgba", " ".join(map(str, color)))
            if friction is not None:
                leg.set("friction", " ".join(map(str, friction)))

    def add_sub_tree(self, environment_root):
        object_dict = self.get_object_dict()
        prefix = object_dict.get("name", "")
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
                env_worldbody.append(self.set_table_subtree(copy.deepcopy(copy_body)))

    def set_table_subtree(self, element):
        if "name" in element.attrib:
            element.attrib["name"] = f"{element.attrib.get("name", "")}"
        if self.size is not None and "size" in element.attrib and "top" in element.attrib.get("name", ""):
            element.attrib["size"] = f'{" ".join(map(str, self.size[1:]))}'
            element.attrib["pos"] = f'0 0 {self.size[0]}'

        if self.size is not None and "size" in element.attrib and "leg" in element.attrib.get("name", ""):
            element.attrib["size"] = f'0.05 0.05 {self.size[0]/2}'
            old_pos = element.attrib.get("pos", "0 0 0")
            parts = old_pos.split()
            parts[-1] = str(self.size[0]/2)
            element.attrib["pos"] = " ".join(parts)
        if self.size is not None and "mass" in element.attrib:
            element.attrib["mass"] = self.mass
        if self.friction is not None and "friction" in element.attrib:
            element.attrib["friction"] = self.friction
        if self.color is not None and "rgba" in element.attrib:
            element.attrib["rgba"] = self.color
        for child in element:
            self.set_table_subtree(child)

        return element
