import os
import skimage
import sys


MRCNN = os.path.abspath("../img_sgm_ml/Mask_RCNN/")
MRCNN2 = os.path.abspath("../Mask_RCNN/")
sys.path.append(MRCNN)
sys.path.append(MRCNN2)

import numpy as np
from img_sgm_ml.model.utils import generate_color
import matplotlib.pyplot as plt
from img_sgm_ml.model.config import LabelConfig
from img_sgm_ml.model.model import MaskRCNNModel

if __name__ == '__main__':
    config = LabelConfig()
    model = MaskRCNNModel(config, os.path.abspath("../rsc/checkpoints/"))

    image = skimage.io.imread("http://localhost:8080/data/upload/27497d761a27eff60ac7e4cddcda0d37-16.jpg")
    plt.imshow(image)
    plt.show()
    predictions = model.interference(image)

    prediction = predictions[0]

    for i in range(prediction['masks'].shape[2]):
        shape = np.array(prediction['masks'][:, :, i]).shape

        # Expand mask with 3 other dimensions to get RGBa Image
        mask3d = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
        expanded_mask = np.array(prediction['masks'][:, :, i:i + 1])
        mask3d[:, :, :] = expanded_mask

        plt.imshow(mask3d)
        # Load color
        (r, g, b, a) = generate_color(config.CLASSES[prediction["class_ids"][i]])

        mask_image = np.empty((shape[0], shape[1], 4), dtype=np.uint8)
        mask_image[:, :] = [r, g, b, a]

        # multiply mask_image with array to get a bgr image with the color of the mask/name
        mask_image = np.array(mask_image * mask3d, dtype=np.uint8)

        plt.imsave(f"../rsc/out/mask_{i}.png", prediction['masks'][:, :, i])
        plt.imsave(f"../rsc/out/mask_{i}_expanded.png", mask3d[:, :, 3])
        plt.imsave(f"../rsc/out/mask_{i}_color.png", mask_image)

    print()