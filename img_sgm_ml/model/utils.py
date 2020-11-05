import os
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
from mrcnn import utils

from img_sgm_ml.rle.decode import decode


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
    # name_hash = hash(string)
    # Generate hash
    result = hashlib.md5(bytes(string.encode('utf-8')))
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


def decode_completions_to_bitmap(completion):
    """
    From Label-Studio completion to dict with bitmap
    Args:
        completion: a LS completion of an image

    Returns:
    {   result-count: count of labels
        labels: [label_1, label_2...]
        bitmaps: [ [numpy uint8 image (width x height)] ]
    }

    """
    labels = []
    bitmaps = []
    counter = 0
    for result in completion['result']:
        if result['type'] != 'brushlabels':
            continue

        label = result['value']['brushlabels'][0]
        labels.append(label)

        # result count
        counter += 1

        rle = result['value']['rle']
        width = result['original_width']
        height = result['original_height']

        dec = decode(rle)
        image = np.reshape(dec, [height, width, 4])[:, :, 3]
        bitmap = image.vectorize(lambda l: 1 if l > 0.4 else 0)
        bitmaps.append(bitmap)
    return {
        "results_count": counter,
        "labels": labels,
        "bitmaps": bitmaps
    }


def convert_bitmaps_to_mrnn(bitmap_object, config):
    """
    Converts a Bitmap object (function above) into a mrnn compatible format

    Args:
        bitmap_object: Object returned by decode_completions_to_bitmap
        config: Config with classes

    Returns:
    {   labels: [label_1_index, label_2_index...]
        bitmaps: [ [numpy uint8 image (width x height x instance_count)] ]
    }

    """
    bitmaps = bitmap_object["bitmaps"]
    counter = bitmap_object["result_count"]
    labels = bitmap_object["labels"]
    height = bitmaps[0].shape[0]
    width = bitmaps[0].shape[1]

    bms = np.zeros((height, width, counter), np.int32)
    for i, bm in enumerate(bitmaps):
        bms[:, :, i] = bm

    invlabels = dict(zip(config.CLASSES.values(), config.CLASSES.keys()))
    encoded_labels = [invlabels[l] for l in labels]

    return {
        "labels": encoded_labels,
        "bitmaps": bms,
        "width": width,
        "height": height
    }


def completion_to_mrnn(completion, config):
    return convert_bitmaps_to_mrnn(
        decode_completions_to_bitmap(completion),
        config
    )
