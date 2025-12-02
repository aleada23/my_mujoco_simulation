import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import copy
from my_mujoco_simulation.simulation.xml_utils import rename_subtree, ensure_section, merge_section


class Camera:
    def __init__(self, camera_path=None, prefix="cam", init_pos=None, init_orn=None, fov=None, resolution=(640, 480)):
        self.camera_path = camera_path
        self.prefix = prefix
        self.init_pos = init_pos or "0 0 0"
        self.init_orn = init_orn or "1 0 0 0"  # quaternion
        self.fov = fov or 60.0  # degrees
        self.resolution = resolution or (640, 480)

        self.name = f"{prefix}_camera"
        self.camera_id = None
        self.model = None
        self.data = None

        if camera_path:
            self._extract_camera_info()

    # XML Parsing
    def _extract_camera_info(self):
        tree = ET.parse(self.camera_path)
        root = tree.getroot()
        cam_element = root.find(".//camera")
        if cam_element is not None:
            self.name = cam_element.attrib.get("name", self.name)
            if "pos" in cam_element.attrib:
                self.init_pos = cam_element.attrib["pos"]
            if "quat" in cam_element.attrib:
                self.init_orn = cam_element.attrib["quat"]
            if "fovy" in cam_element.attrib:
                self.fov = float(cam_element.attrib["fovy"])
        else:
            print(f"[Camera] Warning: No <camera> element found in {self.camera_path}")

    # Model / Mujoco Integration
    def _build_mujoco_id(self, model, data):
        self.model = model
        self.data = data
        try:
            self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.name)
        except Exception:
            self.camera_id = None
            print(f"[Camera] Could not find camera '{self.name}' in the model.")

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
                
    # Rendering
    def render_image(self, model, data, depth=False):
        width, height = self.resolution
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth_buffer = np.zeros((height, width), dtype=np.float32) if depth else None

        if self.camera_id is None:
            try:
                self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.name)
            except Exception:
                raise ValueError(f"Camera '{self.name}' not found in model")

        # Create MuJoCo render context
        with mujoco.Renderer(model, width, height) as renderer:
            renderer.update_scene(data, camera=self.name)
            rgb = renderer.render()
            if depth:
                depth_buffer = renderer.render(depth=True)
        return (rgb, depth_buffer) if depth else rgb

    # Utilities
    def get_intrinsics(self):
        width, height = self.resolution
        fov_y = np.deg2rad(self.fov)
        fy = height / (2 * np.tan(fov_y / 2))
        fx = fy  # assuming square pixels
        cx = width / 2
        cy = height / 2
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])
        return K

    def get_extrinsics(self, model, data):
        if self.camera_id is None:
            raise ValueError("Camera ID not initialized; call _build_mujoco_id first.")

        pos = data.cam_xpos[self.camera_id]
        quat = data.cam_xquat[self.camera_id]
        return pos.copy(), quat.copy()

    # Export
    def get_object_dict(self):
        return copy.deepcopy({
            "type": "camera",
            "path": self.camera_path,
            "name": self.name,
            "position": self.init_pos,
            "orientation": self.init_orn,
            "fov": self.fov,
            "resolution": self.resolution,
            "K": self.get_intrinsics().tolist()
        })
    def set_object_subtree(self, element):
        if "name" in element.attrib:
            element.attrib["name"] = f"{element.attrib.get("name", "")}"
        
        for child in element:
            self.set_object_subtree(child)

        return element






