import xml.etree.ElementTree as ET
import copy

# All known reference attributes that point to other named elements in MuJoCo
REF_ATTRS = {
    "joint", "joint1", "joint2",
    "site", "site1", "site2", "site3",
    "body", "body1", "body2",
    "geom", "mesh", "material", "tendon", "actuator",
    "sensor", "from", "to", "target", "texture",
    "class", "parent", "child", "childclass", "ref", "attach", "targetbody"
}

def rename_subtree(elem, prefix):
    """Recursively prefix all name attributes and reference attributes."""
    name = elem.attrib.get("name")
    if name and not name.startswith(prefix + "_"):
        elem.set("name", f"{prefix}_{name}")

    for attr, val in list(elem.attrib.items()):
        if attr in REF_ATTRS and val:
            parts = [p.strip() for p in val.split(",")]
            new_parts = []
            for p in parts:
                if p and not p.startswith(prefix + "_"):
                    new_parts.append(f"{prefix}_{p}")
                else:
                    new_parts.append(p)
            elem.set(attr, ", ".join(new_parts))

    for child in elem:
        rename_subtree(child, prefix)

    return elem


def ensure_section(root, tag):
    """Return existing section (like <asset> or <worldbody>) or create it."""
    section = root.find(tag)
    if section is None:
        section = ET.SubElement(root, tag)
    return section


def merge_section(dest_root, src_root, tag, prefix):
    """Merge a specific section (e.g., assets, actuators) from one XML tree into another."""
    src_section = src_root.find(tag)
    if src_section is None:
        return
    dest_section = ensure_section(dest_root, tag)
    for item in src_section:
        dest_section.append(rename_subtree(copy.deepcopy(item), prefix))
