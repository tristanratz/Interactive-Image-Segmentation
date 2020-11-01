import os
import mrcnn.model as modellib
from mrcnn import utils


class MaskRCNNModel():
    def __init__(self, config):
        self.config = config

        # Create Model
        self.model = modellib.MaskRCNN("inference", config, "./checkpoints")

        # Download wights
        self.download_weights()

        # Load weights
        self.model.load_weights(os.path.join(os.path.abspath("./img_sgm_ml/rsc"), "mask_rcnn_coco.h5"), by_name=True)

    def train(self):
        pass

    def download_weights(self):
        weights_path = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.realpath(__file__)
                )
            ),
            "rsc/mask_rcnn_coco.h5")
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)

    def interference(self, img):
        return self.model.detect([img], verbose=1)

    def batchInterfere(self, imgs):
        return self.model.detect(imgs, verbose=1)
