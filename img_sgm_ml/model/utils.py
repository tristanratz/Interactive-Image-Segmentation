import os
import hashlib
import xml.etree.ElementTree as ET
from mrcnn import utils

def download_weights():
    weights_path = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.realpath(__file__)
            )
        ),
        "rsc/mask_rcnn_coco.h5")
    if not os.path.exists(weights_path):
        utils.download_trained_weights(weights_path)


def generate_color(string: str) -> (int, int, int, int):
    #name_hash = hash(string)
    # Generate hash
    result = hashlib.md5(bytes(string))
    name_hash = int(result.hexdigest(), 16)
    r = (name_hash & 0xFF0000) >> 16
    g = (name_hash & 0x00FF00) >> 8
    b = name_hash & 0x0000FF
    return r, g, b, 128


def generate_config(config, overwrite: bool = False):
    file = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.realpath(__file__)
                )
            )
        ), "img_sgm/config.xml")
    if not overwrite and os.path.isfile(file):
        print("Using existing config.")
        return
    print("Generating config...")
    view = ET.Element('View')
    brushlabels = ET.SubElement(view, 'BrushLabels')
    brushlabels.set("name", "tag")
    brushlabels.set("toName", "img")
    img = ET.SubElement(view, 'Image')
    img.set("name", "img")
    img.set("value", "$image")
    img.set("zoom", "true")
    img.set("zoomControl", "true")

    for key in config.CLASSES:
        lclass = config.CLASSES[key]
        label = ET.SubElement(brushlabels, 'Label')
        label.set("value", lclass)
        (r, g, b, a) = generate_color(lclass)
        label.set("background", f"rgba({r},{g},{b},{round((a / 255), 2)})")

    tree = ET.ElementTree(view)
    tree.write(file)
    print("Config ready")