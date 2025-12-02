import xml.etree.ElementTree as ET
import copy
from my_mujoco_simulation.simulation.xml_utils import rename_subtree, ensure_section, merge_section


class SimObject:
    def __init__(self, obj_path, prefix=None, init_pos=None, init_quat=None):
        self.obj_path = obj_path
        self.prefix = prefix or "object"
        self.init_pos = init_pos or "0 0 0"
        self.init_quat = init_quat or "1 0 0 0"

        # Parse XML
        self.tree = ET.parse(self.obj_path)
        self.root = self.tree.getroot()

        self._create_dict()

    def _create_dict(self):
        self.obj_dict = {
            "type": "object",
            "name": self.prefix,
            "path": self.obj_path,
            "position": self.init_pos,
            "orientation": self.init_quat,
            "geoms": [g.attrib.get("name") for g in self.root.findall(".//geom")],
        }
    def get_object_dict(self):
        return copy.deepcopy(self.obj_dict)
