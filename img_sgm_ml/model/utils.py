import hashlib
import json
import os
import random
import logging
import xml.etree.ElementTree as ET

import numpy as np
from mrcnn import utils

from img_sgm_ml.rle.decode import decode


def transform_url(url):
    url = url.replace(" ", "%20")
    if os.getenv("DOCKER"):
        url = url.replace("localhost", "labeltool", 1)
    return url


def download_weights():
    """Downloads coco model if not already downloaded"""
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
    """Create RGB Color for a specific name"""
    # Generate hash
    result = hashlib.md5(bytes(string.encode('utf-8')))
    name_hash = int(result.hexdigest(), 16)
    r = (name_hash & 0xFF0000) >> 16
    g = (name_hash & 0x00FF00) >> 8
    b = name_hash & 0x0000FF
    return r, g, b, 128


def generate_config(config, overwrite: bool = False):
    """Generate config.xml for label-studio."""

    # config file
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

    print("Generating XML config...")
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
        if lclass != u"__background__":
            label = ET.SubElement(brushlabels, 'Label')
            label.set("value", lclass)
            (r, g, b, a) = generate_color(lclass)
            label.set("background", f"rgba({r},{g},{b},{round((a / 255), 2)})")

    tree = ET.ElementTree(view)
    tree.write(file)
    print("Config generated.")


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
    logging.debug("Decoding completion")
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

        logging.debug("Decoding RLE")
        dec = decode(rle)
        image = np.reshape(dec, [height, width, 4])[:, :, 3]
        f = np.vectorize(lambda l: 1 if (float(l)/255) > 0.4 else 0)
        bitmap = f(image)
        bitmaps.append(bitmap)
    logging.debug("Decoding finished")
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
        bitmaps: [ [numpy bool image (width x height x instance_count)] ]
    }

    """
    bitmaps = bitmap_object["bitmaps"]
    counter = bitmap_object["results_count"]
    labels = bitmap_object["labels"]
    height = bitmaps[0].shape[0]
    width = bitmaps[0].shape[1]

    bms = np.zeros((height, width, counter), np.int32)
    for i, bm in enumerate(bitmaps):
        bms[:, :, i] = bm

    invlabels = dict(zip(config.CLASSES.values(), config.CLASSES.keys()))
    encoded_labels = [invlabels[l] for l in labels]
    print("Encoded labels:", encoded_labels)

    return {
        "class_ids": np.array(encoded_labels, np.int32),
        "bitmaps": bms.astype(np.bool),
        "width": width,
        "height": height
    }


def completion_to_mrnn(completion, config):
    return convert_bitmaps_to_mrnn(
        decode_completions_to_bitmap(completion),
        config
    )


def devide_completions(completions, train_share=0.85):
    """
    Devide completions into train and validation set.
    Load allocation from previous trainings from state file

    Args:
        completions: The complete set of completions

    Returns: A set of completions of the first

    """
    completions = [c for c in completions]

    f = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.realpath(__file__))),
        "rsc/data_allocation.json"
    )

    allocs = []
    if os.path.isfile(f):
        allocs = json.load(open(f, "r"))["completions"]

    alloc_train_ids = [x["id"] for x in allocs if x["subset"] == "train"]
    alloc_val_ids = [x["id"] for x in allocs if x["subset"] == "val"]

    # Devide into already classified elements, and unclassified
    train_set = list(filter(lambda x: x['completions'][0]["id"] in alloc_train_ids, completions))
    val_set = list(filter(lambda x: x['completions'][0]["id"] in alloc_val_ids, completions))
    unallocated = list(filter(lambda x: not (x['completions'][0]["id"] in alloc_train_ids
                                             or x['completions'][0]["id"] in alloc_val_ids), completions))

    # allocate yet unclassified elements
    random.shuffle(unallocated)
    add_items = int(round(len(completions) * train_share, 0)) - len(train_set)

    if add_items > 0:
        if add_items < len(completions) - 1:
            train_set = train_set + unallocated[:add_items]
            val_set = val_set + unallocated[add_items:]
        else:
            train_set = train_set + unallocated
    else:
        val_set = val_set + unallocated

    # Save the allocation to file
    allocs_train = [{"id": x['completions'][0]["id"], "subset": "train"} for x in train_set]
    allocs_val = [{"id": x['completions'][0]["id"], "subset": "val"} for x in val_set]

    allocs = allocs_train + allocs_val

    json.dump({"completions": allocs}, open(f, "w"))
    print("Allocation was written. Completions:", len(allocs))

    return train_set, val_set
